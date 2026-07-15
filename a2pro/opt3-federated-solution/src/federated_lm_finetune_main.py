"""OPT-3 stage: federated fine-tuning of an A2.Pro LM checkpoint."""

from __future__ import annotations

import copy
import json
import os
import statistics
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from brain_opt import AsyncConfig, FedAvgConfig, FederatedClient, run_async_local_sgd, run_async_sgd, run_fedavg
from lm_storage_common import (
    OptimizerRun,
    causal_lm_loss,
    clean_dir,
    evaluate_lm,
    infer_arch_name,
    load_causal_lm,
    make_lm_tokens,
    make_tiny_checkpoint,
    maybe_load_tokenizer,
    nested,
    parse_methods,
    save_causal_lm,
    write_metrics,
)
from log_setup import configure_logging

logger = configure_logging()

WORKDIR = Path(os.environ.get("FEDERATED_LM_WORKDIR", "/tmp/a2pro_federated_lm"))
LOCAL_IN = WORKDIR / "in_model"
LOCAL_OUT = WORKDIR / "out_model"
LOCAL_METRICS = WORKDIR / "out_metrics"
FRAMEWORK_NAME = os.environ.get("FEDERATED_LM_FRAMEWORK_NAME", "torch:2.10.0")
DEFAULT_METHODS = ["fedavg", "async_sgd", "async_local_sgd"]


@dataclass(frozen=True)
class Config:
    methods: list[str]
    clients: int = 6
    samples_per_client: int = 8
    val_samples: int = 24
    seq_len: int = 48
    rounds: int = 3
    updates: int = 8
    clients_per_round: int = 3
    local_steps: int = 1
    batch_size: int = 2
    lr: float = 5e-4
    async_lr: float = 2e-4
    server_lr: float = 1.0
    max_pending: int = 4
    min_delay: int = 1
    max_delay: int = 4
    staleness_power: float = 1.0
    seed: int = 123
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FederatedCausalLM(nn.Module):
    """Adapter making HF causal LM compatible with brain_opt.federated."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits


def is_offline() -> bool:
    return os.environ.get("FEDERATED_LM_OFFLINE", "").lower() in {"1", "true", "yes"}


def _parse_params(params: dict[str, Any]) -> Config:
    return Config(
        methods=parse_methods(nested(params, "federated", "method", default=params.get("method")), DEFAULT_METHODS),
        clients=int(nested(params, "federated", "clients", default=params.get("clients", 6))),
        samples_per_client=int(nested(params, "federated", "samplesPerClient", default=params.get("samplesPerClient", 8))),
        val_samples=int(nested(params, "data", "valSamples", default=params.get("valSamples", 24))),
        seq_len=int(nested(params, "data", "seqLen", default=params.get("seqLen", 48))),
        rounds=int(nested(params, "training", "rounds", default=params.get("rounds", 3))),
        updates=int(nested(params, "training", "updates", default=params.get("updates", 8))),
        clients_per_round=int(nested(params, "training", "clientsPerRound", default=params.get("clientsPerRound", 3))),
        local_steps=int(nested(params, "training", "localSteps", default=params.get("localSteps", 1))),
        batch_size=int(nested(params, "training", "batchSize", default=params.get("batchSize", 2))),
        lr=float(nested(params, "training", "lr", default=params.get("lr", 5e-4))),
        async_lr=float(nested(params, "training", "asyncLr", default=params.get("asyncLr", 2e-4))),
        server_lr=float(nested(params, "training", "serverLr", default=params.get("serverLr", 1.0))),
        max_pending=int(nested(params, "async", "maxPending", default=params.get("maxPending", 4))),
        min_delay=int(nested(params, "async", "minDelay", default=params.get("minDelay", 1))),
        max_delay=int(nested(params, "async", "maxDelay", default=params.get("maxDelay", 4))),
        staleness_power=float(nested(params, "async", "stalenessPower", default=params.get("stalenessPower", 1.0))),
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


def _make_clients(tokens: torch.Tensor, cfg: Config) -> list[FederatedClient]:
    clients = []
    offset = 0
    for idx in range(cfg.clients):
        block = tokens[offset:offset + cfg.samples_per_client]
        offset += cfg.samples_per_client
        clients.append(FederatedClient(x=block, y=block.clone(), name=f"client-{idx:03d}"))
    return clients


def _summarize_history(method: str, history: list[dict[str, Any]], val_loss: float, seconds: float) -> OptimizerRun:
    last = history[-1] if history else {}
    staleness_values = [
        int(item["staleness"])
        for item in history
        if item.get("staleness") is not None
    ]
    return OptimizerRun(
        method=method,
        train_loss=float(last.get("train_loss", float("nan"))),
        val_loss=val_loss,
        steps=len(history),
        tokens_seen=0,
        seconds=seconds,
        mean_staleness=statistics.mean(staleness_values) if staleness_values else None,
        max_staleness=max(staleness_values) if staleness_values else None,
    )


def run_experiment(input_dir: Path, cfg: Config) -> dict[str, Any]:
    device = torch.device(cfg.device)
    tokenizer = maybe_load_tokenizer(input_dir)
    base_hf = load_causal_lm(input_dir, device)
    vocab_size = int(getattr(base_hf.config, "vocab_size", 50257))
    train_tokens = make_lm_tokens(
        samples=cfg.clients * cfg.samples_per_client,
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
    clients = _make_clients(train_tokens, cfg)

    rows = []
    trained_models: dict[str, Any] = {}
    for method in cfg.methods:
        logger.info("federated LM method=%s", method)
        wrapper = FederatedCausalLM(copy.deepcopy(base_hf)).to(device)
        start = time.perf_counter()
        canonical = method.lower().replace("-", "_")
        if canonical == "fedavg":
            result = run_fedavg(
                wrapper,
                clients,
                FedAvgConfig(
                    rounds=cfg.rounds,
                    clients_per_round=min(cfg.clients_per_round, cfg.clients),
                    local_steps=cfg.local_steps,
                    lr=cfg.lr,
                    batch_size=cfg.batch_size,
                    seed=cfg.seed,
                ),
                loss_fn=causal_lm_loss,
            )
        elif canonical == "async_sgd":
            result = run_async_sgd(
                wrapper,
                clients,
                AsyncConfig(
                    updates=cfg.updates,
                    local_steps=1,
                    lr=cfg.async_lr,
                    server_lr=cfg.server_lr,
                    max_pending=cfg.max_pending,
                    min_delay=cfg.min_delay,
                    max_delay=cfg.max_delay,
                    staleness_power=cfg.staleness_power,
                    batch_size=cfg.batch_size,
                    seed=cfg.seed,
                ),
                loss_fn=causal_lm_loss,
            )
        elif canonical == "async_local_sgd":
            result = run_async_local_sgd(
                wrapper,
                clients,
                AsyncConfig(
                    updates=cfg.updates,
                    local_steps=cfg.local_steps,
                    lr=cfg.lr,
                    server_lr=cfg.server_lr,
                    max_pending=cfg.max_pending,
                    min_delay=cfg.min_delay,
                    max_delay=cfg.max_delay,
                    staleness_power=cfg.staleness_power,
                    batch_size=cfg.batch_size,
                    seed=cfg.seed,
                ),
                loss_fn=causal_lm_loss,
            )
        else:
            raise ValueError(f"unknown federated method: {method}")

        elapsed = time.perf_counter() - start
        hf_model = result.model.model
        val_loss = evaluate_lm(hf_model, val_tokens, cfg.batch_size, device)
        row = _summarize_history(method, result.history, val_loss, elapsed)
        rows.append(row)
        trained_models[method] = hf_model

    best = min(rows, key=lambda item: item.val_loss)
    save_causal_lm(trained_models[best.method], LOCAL_OUT, tokenizer=tokenizer)
    metadata = {
        "stage": "federated_lm_finetune",
        "source": "A2.Pro checkpoint storage",
        "best_method": best.method,
        "best_val_loss": best.val_loss,
        "config": asdict(cfg),
        "arch_name": infer_arch_name(input_dir),
    }
    write_metrics(LOCAL_METRICS, rows, metadata)
    return metadata


def run_offline() -> None:
    logger.info("MODE: FEDERATED_LM_OFFLINE")
    cfg = _parse_params({})
    _update(None, "Создание локального входного HF-checkpoint", 5.0)
    make_tiny_checkpoint(LOCAL_IN, seq_len=cfg.seq_len)
    _update(None, "Федеративное дообучение LM из checkpoint", 20.0)
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

    _update(client, "Федеративное дообучение LM методами ОПТ-3", 20.0)
    metadata = run_experiment(LOCAL_IN, cfg)

    _update(client, "Загрузка output checkpoint", 85.0)
    arch_name = str(metadata["arch_name"])
    collection = client.get_output("out_model").create_checkpoint_collection(
        arch_name=arch_name,
        framework_name=FRAMEWORK_NAME,
    )
    checkpoint_name = f"{arch_name}-federated-lm-{metadata['best_method'].lower()}-{run_suffix}"
    mutable = collection.create_checkpoint(
        name=checkpoint_name,
        title=f"Federated LM fine-tune {metadata['best_method']} ({run_suffix})",
        description=f"LM checkpoint fine-tuned from A2.Pro storage with {metadata['best_method']}",
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
    logger.info("stage federated_lm_finetune start")
    if is_offline():
        run_offline()
        return
    try:
        run_online()
    except ImportError:
        logger.warning("PlatformAPI unavailable; falling back to offline run")
        run_offline()
    except Exception as exc:  # noqa: BLE001
        logger.error("stage federated_lm_finetune failed:\n%s", traceback.format_exc())
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
