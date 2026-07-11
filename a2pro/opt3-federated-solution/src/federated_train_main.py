"""OPT-3 stage: federated optimization demo through brain_opt from base image.

The stage intentionally keeps data generation local and deterministic. That
makes it suitable for A2.Pro acceptance runs where network access and external
datasets are not guaranteed, while still exercising the production library API:
FedAvg, AsyncSGD, and Async-Local SGD are imported from ``brain_opt``.

Offline mode:
    FEDERATED_OFFLINE=1 PYTHONPATH=src python src/federated_train_main.py
"""

from __future__ import annotations

import csv
import json
import os
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from log_setup import configure_logging

logger = configure_logging()

WORKDIR = Path(os.environ.get("FEDERATED_WORKDIR", "/tmp/a2pro_federated"))
LOCAL_MODEL = WORKDIR / "out_model"
LOCAL_METRICS = WORKDIR / "out_metrics"
ARCH_NAME = "federated-cifar-smallcnn"
FRAMEWORK_NAME = os.environ.get("FEDERATED_FRAMEWORK_NAME", "torch:2.10.0")


@dataclass
class StageConfig:
    method: str = "all"
    clients: int = 20
    samples_per_client: int = 24
    test_samples: int = 256
    split: str = "label-skew"
    rounds: int = 12
    updates: int = 40
    clients_per_round: int = 5
    local_steps: int = 2
    batch_size: int = 16
    model_width: int = 8
    lr: float = 0.05
    async_lr: float = 0.02
    seed: int = 123
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class MethodSummary:
    method: str
    initial_loss: float
    final_loss: float
    final_accuracy: float
    points: int
    final_time: int | None = None
    mean_staleness: float | None = None
    max_staleness: int | None = None


class SmallCnn(nn.Module):
    def __init__(self, width: int = 8, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(width * 8 * 8, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def is_offline() -> bool:
    return os.environ.get("FEDERATED_OFFLINE", "").lower() in {"1", "true", "yes"}


def _nested(params: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = params
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _enum_value(value: Any, default: str) -> str:
    if isinstance(value, dict):
        return str(value.get("name", default))
    if value is None:
        return default
    return str(value)


def _parse_params(params: dict[str, Any]) -> StageConfig:
    cfg = StageConfig(
        method=_enum_value(_nested(params, "simulation", "method", default=params.get("method", "all")), "all"),
        clients=int(_nested(params, "simulation", "clients", default=params.get("clients", 20))),
        samples_per_client=int(
            _nested(params, "simulation", "samplesPerClient", default=params.get("samplesPerClient", 24))
        ),
        test_samples=int(_nested(params, "simulation", "testSamples", default=params.get("testSamples", 256))),
        split=_enum_value(_nested(params, "simulation", "split", default=params.get("split", "label-skew")), "label-skew"),
        rounds=int(_nested(params, "training", "rounds", default=params.get("rounds", 12))),
        updates=int(_nested(params, "training", "updates", default=params.get("updates", 40))),
        clients_per_round=int(
            _nested(params, "training", "clientsPerRound", default=params.get("clientsPerRound", 5))
        ),
        local_steps=int(_nested(params, "training", "localSteps", default=params.get("localSteps", 2))),
        batch_size=int(_nested(params, "training", "batchSize", default=params.get("batchSize", 16))),
        model_width=int(_nested(params, "training", "modelWidth", default=params.get("modelWidth", 8))),
        lr=float(_nested(params, "training", "lr", default=params.get("lr", 0.05))),
        async_lr=float(_nested(params, "training", "asyncLr", default=params.get("asyncLr", 0.02))),
        seed=int(_nested(params, "training", "seed", default=params.get("seed", 123))),
    )
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: StageConfig) -> None:
    allowed = {"all", "fedavg", "async_sgd", "async_local_sgd"}
    if cfg.method not in allowed:
        raise ValueError(f"method must be one of {sorted(allowed)}")
    if cfg.split not in {"label-skew", "iid"}:
        raise ValueError("split must be 'label-skew' or 'iid'")
    if cfg.clients < 2:
        raise ValueError("clients must be >= 2")
    if cfg.samples_per_client <= 0 or cfg.test_samples <= 0:
        raise ValueError("sample counts must be positive")
    if cfg.rounds <= 0 or cfg.updates <= 0:
        raise ValueError("rounds and updates must be positive")
    if cfg.clients_per_round <= 0:
        raise ValueError("clients_per_round must be positive")
    cfg.clients_per_round = min(cfg.clients_per_round, cfg.clients)
    if cfg.local_steps <= 0 or cfg.batch_size <= 0:
        raise ValueError("local_steps and batch_size must be positive")
    if cfg.lr <= 0 or cfg.async_lr <= 0:
        raise ValueError("learning rates must be positive")


def cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target.long())


def make_synthetic_cifar(samples: int, seed: int, num_classes: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    labels = torch.arange(samples) % num_classes
    labels = labels[torch.randperm(samples, generator=gen)]
    images = 0.10 * torch.randn(samples, 3, 32, 32, generator=gen)
    colors = torch.eye(num_classes, 3)[labels % 3]
    for i, label in enumerate(labels):
        row = 2 + int(label % 5) * 6
        col = 2 + int(label // 5) * 12
        images[i, :, row : row + 8, col : col + 8] += colors[i].view(3, 1, 1) + 0.35
    return images.clamp(0.0, 1.0), labels.long()


def make_clients(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    clients: int,
    samples_per_client: int,
    seed: int,
    split: str,
):
    from brain_opt import FederatedClient

    if split == "iid":
        return _make_iid_clients(x, y, clients=clients, samples_per_client=samples_per_client, seed=seed)

    gen = torch.Generator().manual_seed(seed)
    by_label = {label: (y == label).nonzero(as_tuple=False).flatten().tolist() for label in range(10)}
    for label, ids in by_label.items():
        perm = torch.randperm(len(ids), generator=gen).tolist()
        by_label[label] = [ids[i] for i in perm]
    offsets = {label: 0 for label in range(10)}
    out: list[FederatedClient] = []
    for client_id in range(clients):
        client_labels = [client_id % 10, (client_id * 3 + 1) % 10]
        ids: list[int] = []
        while len(ids) < samples_per_client:
            for label in client_labels:
                pool = by_label[label]
                idx = pool[offsets[label] % len(pool)]
                offsets[label] += 1
                ids.append(idx)
                if len(ids) >= samples_per_client:
                    break
        out.append(FederatedClient(x=x[ids], y=y[ids], name=f"client-{client_id:03d}"))
    return out


def _make_iid_clients(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    clients: int,
    samples_per_client: int,
    seed: int,
):
    from brain_opt import FederatedClient

    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(x.size(0), generator=gen)
    out: list[FederatedClient] = []
    for client_id in range(clients):
        start = client_id * samples_per_client
        end = start + samples_per_client
        if end > order.numel():
            order = order[torch.randperm(order.numel(), generator=gen)]
            start = 0
            end = samples_per_client
        ids = order[start:end].tolist()
        out.append(FederatedClient(x=x[ids], y=y[ids], name=f"client-{client_id:03d}"))
    return out


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    count = 0
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            xb = x[start : start + batch_size].to(device)
            yb = y[start : start + batch_size].to(device)
            logits = model(xb)
            losses.append(float(F.cross_entropy(logits, yb).item()))
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            count += int(yb.numel())
    model.train()
    return sum(losses) / max(1, len(losses)), correct / max(1, count)


def add_rows(rows: list[dict[str, Any]], method: str, history: list[dict[str, Any]]) -> None:
    for item in history:
        step = item.get("round", item.get("update"))
        rows.append(
            {
                "method": method,
                "step": step,
                "time": item.get("time", step),
                "train_loss": item["train_loss"],
                "staleness": item.get("staleness", ""),
                "staleness_weight": item.get("staleness_weight", ""),
            }
        )


def summarize(
    method: str,
    initial_loss: float,
    history: list[dict[str, Any]],
    final_accuracy: float,
) -> MethodSummary:
    stale = [int(row["staleness"]) for row in history if "staleness" in row]
    final = history[-1]
    return MethodSummary(
        method=method,
        initial_loss=initial_loss,
        final_loss=float(final["train_loss"]),
        final_accuracy=final_accuracy,
        points=len(history),
        final_time=int(final["time"]) if "time" in final else None,
        mean_staleness=sum(stale) / len(stale) if stale else None,
        max_staleness=max(stale) if stale else None,
    )


def method_specs(cfg: StageConfig) -> list[tuple[str, Callable[..., Any], Any]]:
    from brain_opt import AsyncConfig, FedAvgConfig, run_async_local_sgd, run_async_sgd, run_fedavg

    specs = [
        (
            "FedAvg",
            run_fedavg,
            FedAvgConfig(
                rounds=cfg.rounds,
                clients_per_round=cfg.clients_per_round,
                local_steps=cfg.local_steps,
                lr=cfg.lr,
                batch_size=cfg.batch_size,
                seed=cfg.seed,
            ),
        ),
        (
            "AsyncSGD",
            run_async_sgd,
            AsyncConfig(
                updates=cfg.updates,
                local_steps=1,
                lr=cfg.async_lr,
                max_pending=6,
                min_delay=1,
                max_delay=5,
                batch_size=cfg.batch_size,
                seed=cfg.seed + 1,
            ),
        ),
        (
            "Async-LocalSGD",
            run_async_local_sgd,
            AsyncConfig(
                updates=cfg.updates,
                local_steps=cfg.local_steps,
                lr=cfg.lr,
                max_pending=6,
                min_delay=1,
                max_delay=5,
                batch_size=cfg.batch_size,
                seed=cfg.seed + 2,
            ),
        ),
    ]
    if cfg.method == "all":
        return specs
    normalized = {
        "fedavg": "FedAvg",
        "async_sgd": "AsyncSGD",
        "async_local_sgd": "Async-LocalSGD",
    }[cfg.method]
    return [item for item in specs if item[0] == normalized]


def run_training(cfg: StageConfig) -> tuple[nn.Module, list[dict[str, Any]], list[MethodSummary], dict[str, Any]]:
    import brain_opt

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    train_x, train_y = make_synthetic_cifar(cfg.clients * cfg.samples_per_client * 2, cfg.seed)
    test_x, test_y = make_synthetic_cifar(cfg.test_samples, cfg.seed + 1)
    clients = make_clients(
        train_x,
        train_y,
        clients=cfg.clients,
        samples_per_client=cfg.samples_per_client,
        seed=cfg.seed,
        split=cfg.split,
    )

    initial_model = SmallCnn(width=cfg.model_width).to(device)
    initial_loss, _ = evaluate(initial_model, train_x, train_y, device, cfg.batch_size)
    rows: list[dict[str, Any]] = []
    summaries: list[MethodSummary] = []
    best_model: nn.Module | None = None
    best_accuracy = -1.0

    for name, fn, method_cfg in method_specs(cfg):
        torch.manual_seed(cfg.seed)
        logger.info("running %s", name)
        model = SmallCnn(width=cfg.model_width).to(device)
        result = fn(model, clients, method_cfg, loss_fn=cross_entropy_loss)
        _, accuracy = evaluate(result.model, test_x, test_y, device, cfg.batch_size)
        summary = summarize(name, initial_loss, result.history, accuracy)
        summaries.append(summary)
        add_rows(rows, name, result.history)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = result.model
        logger.info("%s final_loss=%.4f final_accuracy=%.4f", name, summary.final_loss, summary.final_accuracy)

    if best_model is None:
        raise RuntimeError("no methods were executed")

    metadata = {
        "library": "brain_opt",
        "brain_opt_version": getattr(brain_opt, "__version__", "unknown"),
        "architecture": ARCH_NAME,
        "dataset": "synthetic-cifar-shaped",
        "config": asdict(cfg),
        "best_method": max(summaries, key=lambda item: item.final_accuracy).method,
    }
    return best_model, rows, summaries, metadata


def write_outputs(
    model: nn.Module,
    rows: Sequence[dict[str, Any]],
    summaries: Sequence[MethodSummary],
    metadata: dict[str, Any],
) -> None:
    LOCAL_MODEL.mkdir(parents=True, exist_ok=True)
    LOCAL_METRICS.mkdir(parents=True, exist_ok=True)

    model_cpu = model.to("cpu")
    torch.save(
        {
            "model_state_dict": model_cpu.state_dict(),
            "metadata": metadata,
            "summaries": [asdict(item) for item in summaries],
        },
        LOCAL_MODEL / "model.pt",
    )
    (LOCAL_MODEL / "model_config.json").write_text(
        json.dumps(
            {
                "architecture": ARCH_NAME,
                "framework": FRAMEWORK_NAME,
                "model_class": "SmallCnn",
                "metadata": metadata,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    _write_csv(LOCAL_METRICS / "metrics.csv", rows)
    summary_payload = {
        "metadata": metadata,
        "summaries": [asdict(item) for item in summaries],
    }
    (LOCAL_METRICS / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = ["# OPT-3 Federated Training", ""]
    lines.append(f"- `library`: {metadata['library']} {metadata['brain_opt_version']}")
    lines.append(f"- `dataset`: {metadata['dataset']}")
    lines.append(f"- `best_method`: {metadata['best_method']}")
    for item in summaries:
        stale = "" if item.mean_staleness is None else f", mean_staleness={item.mean_staleness:.2f}"
        lines.append(
            f"- `{item.method}`: final_loss={item.final_loss:.4f}, "
            f"final_accuracy={item.final_accuracy:.4f}{stale}"
        )
    (LOCAL_METRICS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _maybe_plot(rows, summaries)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _maybe_plot(rows: Sequence[dict[str, Any]], summaries: Sequence[MethodSummary]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    methods = sorted({row["method"] for row in rows})
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in methods:
        data = [row for row in rows if row["method"] == method]
        ax.plot([row["step"] for row in data], [row["train_loss"] for row in data], label=method)
    ax.set_yscale("log")
    ax.set_xlabel("round/update")
    ax.set_ylabel("global train cross entropy")
    ax.set_title("OPT-3 federated demo: loss by communication step")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(LOCAL_METRICS / "loss_by_step.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([item.method for item in summaries], [item.final_accuracy for item in summaries])
    ax.set_ylabel("final test accuracy")
    ax.set_title("OPT-3 federated demo: final accuracy")
    fig.tight_layout()
    fig.savefig(LOCAL_METRICS / "final_accuracy.png", dpi=160)
    plt.close(fig)

    stale = [int(row["staleness"]) for row in rows if str(row.get("staleness", "")) not in {"", "None"}]
    if stale:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(stale, bins=range(min(stale), max(stale) + 2), align="left", rwidth=0.85)
        ax.set_xlabel("staleness")
        ax.set_ylabel("updates")
        ax.set_title("OPT-3 federated demo: async staleness")
        fig.tight_layout()
        fig.savefig(LOCAL_METRICS / "staleness_hist.png", dpi=160)
        plt.close(fig)


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


def run_offline() -> None:
    cfg = _parse_params({})
    logger.info("MODE: FEDERATED_OFFLINE")
    _update(None, "Генерация федеративного датасета", 10.0)
    model, rows, summaries, metadata = run_training(cfg)
    _update(None, "Сохранение локальных артефактов", 85.0)
    write_outputs(model, rows, summaries, metadata)
    _update(None, f"Готово: {LOCAL_METRICS}", 100.0, completed=True)


def run_online() -> None:
    from PlatformAPI import Client
    from platform_io import upload_directory

    client = Client()
    run_id = str(client.get_run_id())
    run_suffix = run_id.split("-", 1)[0]
    logger.info("run_id=%s stage=%s", run_id, client.get_stage_name())

    params = client.get_parameters() or {}
    if hasattr(params, "to_dict"):
        params = params.to_dict()
    cfg = _parse_params(params if isinstance(params, dict) else {})
    logger.info("parsed params: %s", json.dumps(asdict(cfg), ensure_ascii=False))

    _update(client, "Генерация федеративного датасета", 10.0)
    model, rows, summaries, metadata = run_training(cfg)
    _update(client, "Сохранение локального checkpoint и метрик", 75.0)
    write_outputs(model, rows, summaries, metadata)

    _update(client, "Загрузка checkpoint-collection", 85.0)
    collection = client.get_output("out_model").create_checkpoint_collection(
        arch_name=ARCH_NAME,
        framework_name=FRAMEWORK_NAME,
    )
    checkpoint_name = f"{ARCH_NAME}-{metadata['best_method'].lower()}-{run_suffix}"
    mutable = collection.create_checkpoint(
        name=checkpoint_name,
        title=f"OPT-3 {metadata['best_method']} ({run_suffix})",
        description=f"Federated SmallCnn checkpoint, best method: {metadata['best_method']}",
    )
    uploaded_model_files = upload_directory(mutable, LOCAL_MODEL)
    mutable.close()
    logger.info("uploaded %d model files to out_model", uploaded_model_files)

    _update(client, "Загрузка metrics artifact", 95.0)
    artifact = client.get_output("out_metrics").create_artifact()
    uploaded_metrics = upload_directory(artifact, LOCAL_METRICS)
    artifact.close()
    logger.info("uploaded %d metric files to out_metrics", uploaded_metrics)
    _update(client, "Готово", 100.0, completed=True)


def main() -> None:
    logger.info("stage federated_train start")
    if is_offline():
        run_offline()
        return
    try:
        run_online()
    except ImportError:
        logger.warning("PlatformAPI unavailable; falling back to offline run")
        run_offline()
    except Exception as exc:  # noqa: BLE001
        logger.error("stage federated_train failed:\n%s", traceback.format_exc())
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
