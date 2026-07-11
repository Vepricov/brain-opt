"""OPT-3 federated optimization demo.

Runs a small non-IID federated regression study with:
  * FedAvg at different active-client counts;
  * AsyncSGD with delayed gradients;
  * Async-Local SGD with delayed local deltas.

Outputs are written to ``runs/opt3_federated_demo`` by default:
``metrics.csv``, ``summary.json``, ``summary.md`` and optional PNG plots.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from brain_opt import (
    AsyncConfig,
    FedAvgConfig,
    FederatedClient,
    run_async_local_sgd,
    run_async_sgd,
    run_fedavg,
)


@dataclass
class RunSummary:
    method: str
    initial_loss: float
    final_loss: float
    improvement_x: float
    points: int
    final_time: int | None = None
    mean_staleness: float | None = None
    max_staleness: int | None = None


def make_clients(
    *,
    n_clients: int,
    samples_per_client: int,
    n_features: int,
    seed: int,
) -> list[FederatedClient]:
    """Create a deterministic non-IID linear-regression federation."""
    gen = torch.Generator().manual_seed(seed)
    true_w = torch.linspace(-1.2, 1.8, n_features).view(n_features, 1)
    true_b = torch.tensor([0.25])
    clients = []

    for client_id in range(n_clients):
        cluster = client_id % 4
        shift = (cluster - 1.5) * 0.45
        scale = 0.7 + 0.15 * cluster
        x = scale * torch.randn(samples_per_client, n_features, generator=gen) + shift
        noise = 0.03 * torch.randn(samples_per_client, 1, generator=gen)
        y = x @ true_w + true_b + noise
        clients.append(FederatedClient(x=x, y=y, name=f"client-{client_id:03d}"))

    return clients


def make_model(n_features: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Linear(n_features, 1)


def mse(model: nn.Module, clients: Iterable[FederatedClient]) -> float:
    total = 0.0
    count = 0
    with torch.no_grad():
        for client in clients:
            pred = model(client.x)
            total += float(((pred - client.y) ** 2).sum().item())
            count += client.num_samples
    return total / count


def add_rows(rows: list[dict], method: str, history: list[dict], config: dict) -> None:
    for item in history:
        step = item.get("round", item.get("update"))
        rows.append({
            "method": method,
            "step": step,
            "time": item.get("time", step),
            "train_loss": item["train_loss"],
            "staleness": item.get("staleness", ""),
            "staleness_weight": item.get("staleness_weight", ""),
            **config,
        })


def summarize(method: str, initial_loss: float, history: list[dict]) -> RunSummary:
    final = float(history[-1]["train_loss"])
    stale = [int(row["staleness"]) for row in history if "staleness" in row]
    return RunSummary(
        method=method,
        initial_loss=initial_loss,
        final_loss=final,
        improvement_x=initial_loss / final if final > 0 else float("inf"),
        points=len(history),
        final_time=int(history[-1]["time"]) if "time" in history[-1] else None,
        mean_staleness=(sum(stale) / len(stale)) if stale else None,
        max_staleness=max(stale) if stale else None,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summaries: list[RunSummary]) -> None:
    lines = [
        "# OPT-3 Federated Optimization Demo",
        "",
        "| method | initial loss | final loss | improvement | points | final time | mean staleness | max staleness |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| {method} | {initial:.6f} | {final:.6f} | {improvement:.2f}x | "
            "{points} | {time} | {mean_stale} | {max_stale} |".format(
                method=item.method,
                initial=item.initial_loss,
                final=item.final_loss,
                improvement=item.improvement_x,
                points=item.points,
                time="" if item.final_time is None else item.final_time,
                mean_stale="" if item.mean_staleness is None else f"{item.mean_staleness:.2f}",
                max_stale="" if item.max_staleness is None else item.max_staleness,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_plot(out_dir: Path, rows: list[dict]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    methods = sorted({row["method"] for row in rows})
    paths = []

    fig, ax = plt.subplots(figsize=(9, 5))
    for method in methods:
        data = [row for row in rows if row["method"] == method]
        ax.plot([row["step"] for row in data], [row["train_loss"] for row in data], label=method)
    ax.set_yscale("log")
    ax.set_xlabel("round/update")
    ax.set_ylabel("global train MSE")
    ax.set_title("OPT-3 demo: loss by communication step")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "loss_by_step.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(9, 5))
    for method in methods:
        data = [row for row in rows if row["method"] == method]
        ax.plot([row["time"] for row in data], [row["train_loss"] for row in data], label=method)
    ax.set_yscale("log")
    ax.set_xlabel("simulated time")
    ax.set_ylabel("global train MSE")
    ax.set_title("OPT-3 demo: sync rounds vs async wall-clock")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "loss_by_time.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    stale = [
        int(row["staleness"])
        for row in rows
        if str(row.get("staleness", "")) not in {"", "None"}
    ]
    if stale:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(stale, bins=range(min(stale), max(stale) + 2), align="left", rwidth=0.85)
        ax.set_xlabel("staleness")
        ax.set_ylabel("updates")
        ax.set_title("OPT-3 demo: async staleness distribution")
        fig.tight_layout()
        path = out_dir / "staleness_hist.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/opt3_federated_demo")
    parser.add_argument("--clients", type=int, default=40)
    parser.add_argument("--samples-per-client", type=int, default=32)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clients = make_clients(
        n_clients=args.clients,
        samples_per_client=args.samples_per_client,
        n_features=args.features,
        seed=args.seed,
    )

    rows: list[dict] = []
    summaries: List[RunSummary] = []

    for active in (5, 10, 20):
        method = f"FedAvg-{active}clients"
        model = make_model(args.features, args.seed)
        initial = mse(model, clients)
        cfg = FedAvgConfig(
            rounds=40,
            clients_per_round=active,
            local_steps=2,
            lr=0.04,
            batch_size=16,
            seed=args.seed + active,
        )
        result = run_fedavg(model, clients, cfg)
        add_rows(rows, method, result.history, {"active_clients": active, "local_steps": cfg.local_steps})
        summaries.append(summarize(method, initial, result.history))

    async_specs = [
        (
            "AsyncSGD",
            run_async_sgd,
            AsyncConfig(
                updates=160,
                local_steps=1,
                lr=0.02,
                server_lr=1.0,
                max_pending=8,
                min_delay=1,
                max_delay=6,
                staleness_power=1.0,
                batch_size=16,
                seed=args.seed + 100,
            ),
        ),
        (
            "Async-LocalSGD",
            run_async_local_sgd,
            AsyncConfig(
                updates=100,
                local_steps=3,
                lr=0.04,
                server_lr=1.0,
                max_pending=8,
                min_delay=1,
                max_delay=6,
                staleness_power=1.0,
                batch_size=16,
                seed=args.seed + 200,
            ),
        ),
    ]
    for method, fn, cfg in async_specs:
        model = make_model(args.features, args.seed)
        initial = mse(model, clients)
        result = fn(model, clients, cfg)
        add_rows(rows, method, result.history, {"active_clients": "", "local_steps": cfg.local_steps})
        summaries.append(summarize(method, initial, result.history))

    write_csv(out_dir / "metrics.csv", rows)
    (out_dir / "summary.json").write_text(
        json.dumps([asdict(item) for item in summaries], indent=2),
        encoding="utf-8",
    )
    write_markdown(out_dir / "summary.md", summaries)
    plot_paths = maybe_plot(out_dir, rows)

    print(f"Wrote {out_dir / 'metrics.csv'}")
    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'summary.md'}")
    for path in plot_paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
