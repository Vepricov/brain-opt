"""OPT-1/2 stage: fine-tune an A2.Pro checkpoint with brain_opt optimizers."""

from __future__ import annotations

import copy
import json
import os
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from lm_storage_common import (
    clean_dir,
    infer_arch_name,
    load_causal_lm,
    make_lm_tokens,
    make_tiny_checkpoint,
    maybe_load_tokenizer,
    nested,
    parse_methods,
    save_causal_lm,
    train_one_optimizer,
    write_metrics,
)
from log_setup import configure_logging

logger = configure_logging()

WORKDIR = Path(os.environ.get("LM_FINETUNE_WORKDIR", "/tmp/a2pro_lm_finetune"))
LOCAL_IN = WORKDIR / "in_model"
LOCAL_OUT = WORKDIR / "out_model"
LOCAL_METRICS = WORKDIR / "out_metrics"
FRAMEWORK_NAME = os.environ.get("LM_FRAMEWORK_NAME", "torch:2.10.0")
DEFAULT_METHODS = ["AdamW", "Lion", "Muon"]


@dataclass(frozen=True)
class Config:
    optimizers: list[str]
    steps: int = 20
    batch_size: int = 4
    seq_len: int = 64
    train_samples: int = 128
    val_samples: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    auto_scale_lr: bool = True
    seed: int = 123
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def is_offline() -> bool:
    return os.environ.get("LM_FINETUNE_OFFLINE", "").lower() in {"1", "true", "yes"}


def _parse_params(params: dict[str, Any]) -> Config:
    return Config(
        optimizers=parse_methods(nested(params, "optimization", "optimizers", default=params.get("optimizers")), DEFAULT_METHODS),
        steps=int(nested(params, "training", "steps", default=params.get("steps", 20))),
        batch_size=int(nested(params, "training", "batchSize", default=params.get("batchSize", 4))),
        seq_len=int(nested(params, "training", "seqLen", default=params.get("seqLen", 64))),
        train_samples=int(nested(params, "data", "trainSamples", default=params.get("trainSamples", 128))),
        val_samples=int(nested(params, "data", "valSamples", default=params.get("valSamples", 32))),
        lr=float(nested(params, "optimization", "lr", default=params.get("lr", 1e-3))),
        weight_decay=float(nested(params, "optimization", "weightDecay", default=params.get("weightDecay", 0.0))),
        auto_scale_lr=bool(nested(params, "optimization", "autoScaleLr", default=params.get("autoScaleLr", True))),
        seed=int(nested(params, "training", "seed", default=params.get("seed", 123))),
        device=str(nested(params, "training", "device", default=params.get("device", "cuda" if torch.cuda.is_available() else "cpu"))),
    )


def _update(client: Any, message: str, progress: float, *, completed: bool = False) -> None:
    logger.info("progress %.0f%% - %s", progress, message)
    if client is None:
        return
    from PlatformAPI.models.states import CompletedState, ProcessingState

    state = (CompletedState if completed else ProcessingState)()
    state.message = message
    state.progress = progress
    try:
        client.update_state(state)
    except Exception as exc:  # noqa: BLE001
        logger.warning("update_state failed: %s", exc)


def run_experiment(input_dir: Path, cfg: Config) -> dict[str, Any]:
    device = torch.device(cfg.device)
    tokenizer = maybe_load_tokenizer(input_dir)
    base_model = load_causal_lm(input_dir, device)
    vocab_size = int(getattr(base_model.config, "vocab_size", 50257))
    train_tokens = make_lm_tokens(
        samples=cfg.train_samples,
        seq_len=cfg.seq_len,
        vocab_size=vocab_size,
        seed=cfg.seed,
    )
    val_tokens = make_lm_tokens(
        samples=cfg.val_samples,
        seq_len=cfg.seq_len,
        vocab_size=vocab_size,
        seed=cfg.seed + 1,
    )

    rows = []
    trained_models: dict[str, Any] = {}
    for idx, method in enumerate(cfg.optimizers):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        logger.info("training optimizer=%s (%d/%d)", method, idx + 1, len(cfg.optimizers))
        model = copy.deepcopy(base_model)
        row, trained = train_one_optimizer(
            model,
            optimizer_name=method,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            steps=cfg.steps,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            auto_scale_lr=cfg.auto_scale_lr,
            seed=cfg.seed + 10 + idx,
            device=device,
        )
        rows.append(row)
        trained_models[method] = trained

    best = min(rows, key=lambda item: item.val_loss)
    save_causal_lm(trained_models[best.method], LOCAL_OUT, tokenizer=tokenizer)
    metadata = {
        "stage": "lm_finetune",
        "source": "A2.Pro checkpoint storage",
        "best_method": best.method,
        "best_val_loss": best.val_loss,
        "config": asdict(cfg),
        "arch_name": infer_arch_name(input_dir),
    }
    write_metrics(LOCAL_METRICS, rows, metadata)
    return metadata


def run_offline() -> None:
    logger.info("MODE: LM_FINETUNE_OFFLINE")
    cfg = _parse_params({})
    _update(None, "Создание локального входного HF-checkpoint", 5.0)
    make_tiny_checkpoint(LOCAL_IN, seq_len=cfg.seq_len)
    _update(None, "Дообучение модели из локального checkpoint", 20.0)
    metadata = run_experiment(LOCAL_IN, cfg)
    _update(None, f"Готово: best={metadata['best_method']}", 100.0, completed=True)


def run_online() -> None:
    from PlatformAPI import Client
    from platform_io import download_checkpoint, upload_directory

    client = Client()
    run_id = str(client.get_run_id())
    run_suffix = run_id.split("-", 1)[0]
    logger.info("run_id=%s stage=%s", run_id, client.get_stage_name())

    params = client.get_parameters() or {}
    if hasattr(params, "to_dict"):
        params = params.to_dict()
    cfg = _parse_params(params if isinstance(params, dict) else {})
    logger.info("parsed params: %s", json.dumps(asdict(cfg), ensure_ascii=False))

    _update(client, "Скачивание входного checkpoint in_model", 5.0)
    checkpoint = client.get_checkpoint(input_name="in_model")
    clean_dir(LOCAL_IN)
    download_checkpoint(checkpoint, LOCAL_IN)

    _update(client, "Дообучение модели оптимизаторами ОПТ-1/2", 20.0)
    metadata = run_experiment(LOCAL_IN, cfg)

    _update(client, "Загрузка output checkpoint", 85.0)
    arch_name = str(metadata["arch_name"])
    collection = client.get_output("out_model").create_checkpoint_collection(
        arch_name=arch_name,
        framework_name=FRAMEWORK_NAME,
    )
    checkpoint_name = f"{arch_name}-lm-finetune-{metadata['best_method'].lower()}-{run_suffix}"
    mutable = collection.create_checkpoint(
        name=checkpoint_name,
        title=f"LM fine-tune {metadata['best_method']} ({run_suffix})",
        description=f"Checkpoint fine-tuned from A2.Pro storage with {metadata['best_method']}",
    )
    uploaded_model_files = upload_directory(mutable, LOCAL_OUT)
    mutable.close()
    logger.info("uploaded %d files to out_model", uploaded_model_files)

    _update(client, "Загрузка metrics artifact", 95.0)
    artifact = client.get_output("out_metrics").create_artifact()
    uploaded_metrics = upload_directory(artifact, LOCAL_METRICS)
    artifact.close()
    logger.info("uploaded %d files to out_metrics", uploaded_metrics)
    _update(client, "Готово", 100.0, completed=True)


def main() -> None:
    logger.info("stage lm_finetune start")
    if is_offline():
        run_offline()
        return
    try:
        run_online()
    except ImportError:
        logger.warning("PlatformAPI unavailable; falling back to offline run")
        run_offline()
    except Exception as exc:  # noqa: BLE001
        logger.error("stage lm_finetune failed:\n%s", traceback.format_exc())
        try:
            from PlatformAPI import Client
            from PlatformAPI.models.states import FailedState

            state = FailedState()
            state.message = f"Ошибка: {type(exc).__name__}: {str(exc)[:300]}"
            Client().update_state(state)
        except Exception as state_exc:  # noqa: BLE001
            logger.warning("failed to publish failure state: %s", state_exc)
        raise


if __name__ == "__main__":
    main()

