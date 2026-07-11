"""OPT-3 CIFAR-shaped federated optimization demo.

Default mode uses a synthetic CIFAR-shaped classification dataset so the demo
runs without downloads. Pass ``--dataset cifar10 --download`` to run on a
CIFAR-10 subset with the same client splitting and reporting pipeline.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from brain_opt import AsyncConfig, FedAvgConfig, FederatedClient, run_async_local_sgd, run_async_sgd, run_fedavg


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


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x))


class SmallResNet(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 16) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.GroupNorm(4, width),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(ResidualBlock(width), ResidualBlock(width))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.stem(x))
        x = self.pool(x).flatten(1)
        return self.head(x)


def cross_entropy_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, target.long())


def make_synthetic_cifar(
    *,
    samples: int,
    seed: int,
    num_classes: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    labels = torch.arange(samples) % num_classes
    labels = labels[torch.randperm(samples, generator=gen)]
    images = 0.10 * torch.randn(samples, 3, 32, 32, generator=gen)
    colors = torch.eye(num_classes, 3)[labels % 3]
    for i, label in enumerate(labels):
        row = 2 + int(label % 5) * 6
        col = 2 + int(label // 5) * 12
        images[i, :, row:row + 8, col:col + 8] += colors[i].view(3, 1, 1) + 0.35
    return images.clamp(0.0, 1.0), labels.long()


def load_cifar10(
    *,
    data_dir: str,
    train_samples: int,
    test_samples: int,
    seed: int,
    download: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        from torchvision import datasets, transforms

        transform = transforms.Compose([transforms.ToTensor()])
        train = datasets.CIFAR10(data_dir, train=True, download=download, transform=transform)
        test = datasets.CIFAR10(data_dir, train=False, download=download, transform=transform)
        return _sample_torchvision_cifar(train, test, train_samples, test_samples, seed)
    except ModuleNotFoundError:
        return _load_hf_cifar10(train_samples=train_samples, test_samples=test_samples, seed=seed)


def _sample_torchvision_cifar(
    train,
    test,
    train_samples: int,
    test_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    train_idx = torch.randperm(len(train), generator=gen)[:train_samples].tolist()
    test_idx = torch.randperm(len(test), generator=gen)[:test_samples].tolist()
    train_x = torch.stack([train[i][0] for i in train_idx])
    train_y = torch.tensor([train[i][1] for i in train_idx], dtype=torch.long)
    test_x = torch.stack([test[i][0] for i in test_idx])
    test_y = torch.tensor([test[i][1] for i in test_idx], dtype=torch.long)
    return train_x, train_y, test_x, test_y


def _load_hf_cifar10(
    *,
    train_samples: int,
    test_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from datasets import load_dataset

    train = load_dataset("cifar10", split="train")
    test = load_dataset("cifar10", split="test")
    gen = torch.Generator().manual_seed(seed)
    train_idx = torch.randperm(len(train), generator=gen)[:train_samples].tolist()
    test_idx = torch.randperm(len(test), generator=gen)[:test_samples].tolist()
    train_x, train_y = _hf_rows_to_tensors(train, train_idx)
    test_x, test_y = _hf_rows_to_tensors(test, test_idx)
    return train_x, train_y, test_x, test_y


def _hf_rows_to_tensors(dataset, ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    images = []
    labels = []
    for idx in ids:
        row = dataset[idx]
        arr = torch.tensor(np.asarray(row["img"]), dtype=torch.float32).permute(2, 0, 1) / 255.0
        images.append(arr)
        labels.append(int(row["label"]))
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def make_clients(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    clients: int,
    samples_per_client: int,
    seed: int,
    split: str = "label-skew",
) -> list[FederatedClient]:
    if split == "iid":
        return make_iid_clients(x, y, clients=clients, samples_per_client=samples_per_client, seed=seed)
    if split != "label-skew":
        raise ValueError("split must be 'label-skew' or 'iid'")
    gen = torch.Generator().manual_seed(seed)
    by_label = {label: (y == label).nonzero(as_tuple=False).flatten().tolist() for label in range(10)}
    for label, ids in by_label.items():
        perm = torch.randperm(len(ids), generator=gen).tolist()
        by_label[label] = [ids[i] for i in perm]
    offsets = {label: 0 for label in range(10)}
    out: list[FederatedClient] = []
    for client_id in range(clients):
        labels = [client_id % 10, (client_id * 3 + 1) % 10]
        ids: list[int] = []
        while len(ids) < samples_per_client:
            for label in labels:
                pool = by_label[label]
                if not pool:
                    continue
                idx = pool[offsets[label] % len(pool)]
                offsets[label] += 1
                ids.append(idx)
                if len(ids) >= samples_per_client:
                    break
        out.append(FederatedClient(x=x[ids], y=y[ids], name=f"client-{client_id:03d}"))
    return out


def make_iid_clients(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    clients: int,
    samples_per_client: int,
    seed: int,
) -> list[FederatedClient]:
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(x.size(0), generator=gen)
    out: list[FederatedClient] = []
    for client_id in range(clients):
        start = client_id * samples_per_client
        end = start + samples_per_client
        if end > order.numel():
            start = 0
            end = samples_per_client
            order = order[torch.randperm(order.numel(), generator=gen)]
        ids = order[start:end].tolist()
        out.append(FederatedClient(x=x[ids], y=y[ids], name=f"client-{client_id:03d}"))
    return out


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device, batch_size: int) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    count = 0
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            xb = x[start:start + batch_size].to(device)
            yb = y[start:start + batch_size].to(device)
            logits = model(xb)
            losses.append(float(F.cross_entropy(logits, yb).item()))
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            count += int(yb.numel())
    model.train()
    return sum(losses) / max(1, len(losses)), correct / max(1, count)


def add_rows(rows: list[dict], method: str, history: list[dict], extra: dict) -> None:
    for item in history:
        step = item.get("round", item.get("update"))
        rows.append({
            "method": method,
            "step": step,
            "time": item.get("time", step),
            "train_loss": item["train_loss"],
            "staleness": item.get("staleness", ""),
            "staleness_weight": item.get("staleness_weight", ""),
            **extra,
        })


def summarize(
    method: str,
    initial_loss: float,
    result_history: list[dict],
    final_accuracy: float,
) -> MethodSummary:
    stale = [int(row["staleness"]) for row in result_history if "staleness" in row]
    return MethodSummary(
        method=method,
        initial_loss=initial_loss,
        final_loss=float(result_history[-1]["train_loss"]),
        final_accuracy=final_accuracy,
        points=len(result_history),
        final_time=int(result_history[-1]["time"]) if "time" in result_history[-1] else None,
        mean_staleness=sum(stale) / len(stale) if stale else None,
        max_staleness=max(stale) if stale else None,
    )


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(out_dir: Path, rows: list[dict], summaries: list[MethodSummary], config: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "metrics.csv", rows)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"config": config, "summaries": [asdict(s) for s in summaries]}, f, indent=2)
    lines = ["# OPT-3 CIFAR Federated Demo", ""]
    for item in summaries:
        stale = "" if item.mean_staleness is None else f", mean_staleness={item.mean_staleness:.2f}"
        lines.append(
            f"- `{item.method}`: final_loss={item.final_loss:.4f}, "
            f"final_accuracy={item.final_accuracy:.4f}{stale}"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    maybe_plot(out_dir, rows, summaries)


def maybe_plot(out_dir: Path, rows: list[dict], summaries: list[MethodSummary]) -> None:
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
    ax.set_title("OPT-3 CIFAR demo: loss by communication step")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_by_step.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([s.method for s in summaries], [s.final_accuracy for s in summaries])
    ax.set_ylabel("final test accuracy")
    ax.set_title("OPT-3 CIFAR demo: final accuracy")
    fig.tight_layout()
    fig.savefig(out_dir / "final_accuracy.png", dpi=160)
    plt.close(fig)

    stale = [int(row["staleness"]) for row in rows if str(row.get("staleness", "")) not in {"", "None"}]
    if stale:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(stale, bins=range(min(stale), max(stale) + 2), align="left", rwidth=0.85)
        ax.set_xlabel("staleness")
        ax.set_ylabel("updates")
        ax.set_title("OPT-3 CIFAR demo: async staleness")
        fig.tight_layout()
        fig.savefig(out_dir / "staleness_hist.png", dpi=160)
        plt.close(fig)


def run_method(
    name: str,
    fn: Callable,
    model: nn.Module,
    clients: list[FederatedClient],
    config,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
    batch_size: int,
    initial_loss: float,
) -> tuple[MethodSummary, list[dict]]:
    model = model.to(device)
    result = fn(model, clients, config, loss_fn=cross_entropy_loss)
    _, acc = evaluate(result.model, test_x, test_y, device, batch_size)
    rows: list[dict] = []
    add_rows(rows, name, result.history, {})
    return summarize(name, initial_loss, result.history, acc), rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/opt3_cifar_federated_demo")
    parser.add_argument("--dataset", choices=["synthetic", "cifar10"], default="synthetic")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--samples-per-client", type=int, default=24)
    parser.add_argument("--test-samples", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--clients-per-round", type=int, default=5)
    parser.add_argument("--split", choices=["label-skew", "iid"], default="label-skew")
    parser.add_argument("--local-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-width", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--async-lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.dataset == "synthetic":
        train_x, train_y = make_synthetic_cifar(
            samples=args.clients * args.samples_per_client * 2,
            seed=args.seed,
        )
        test_x, test_y = make_synthetic_cifar(samples=args.test_samples, seed=args.seed + 1)
    else:
        train_x, train_y, test_x, test_y = load_cifar10(
            data_dir=args.data_dir,
            train_samples=args.clients * args.samples_per_client * 2,
            test_samples=args.test_samples,
            seed=args.seed,
            download=args.download,
        )
    clients = make_clients(
        train_x,
        train_y,
        clients=args.clients,
        samples_per_client=args.samples_per_client,
        seed=args.seed,
        split=args.split,
    )
    initial_model = SmallResNet(width=args.model_width)
    initial_loss, _ = evaluate(initial_model.to(device), train_x, train_y, device, args.batch_size)
    rows: list[dict] = []
    summaries: list[MethodSummary] = []

    specs = [
        (
            "FedAvg",
            run_fedavg,
            FedAvgConfig(
                rounds=args.rounds,
                clients_per_round=args.clients_per_round,
                local_steps=args.local_steps,
                lr=args.lr,
                batch_size=args.batch_size,
                seed=args.seed,
            ),
        ),
        (
            "AsyncSGD",
            run_async_sgd,
            AsyncConfig(
                updates=args.updates,
                local_steps=1,
                lr=args.async_lr,
                max_pending=6,
                min_delay=1,
                max_delay=5,
                batch_size=args.batch_size,
                seed=args.seed + 1,
            ),
        ),
        (
            "Async-LocalSGD",
            run_async_local_sgd,
            AsyncConfig(
                updates=args.updates,
                local_steps=args.local_steps,
                lr=args.lr,
                max_pending=6,
                min_delay=1,
                max_delay=5,
                batch_size=args.batch_size,
                seed=args.seed + 2,
            ),
        ),
    ]
    for name, fn, cfg in specs:
        torch.manual_seed(args.seed)
        model = SmallResNet(width=args.model_width)
        summary, method_rows = run_method(
            name,
            fn,
            model,
            clients,
            cfg,
            test_x,
            test_y,
            device,
            args.batch_size,
            initial_loss,
        )
        summaries.append(summary)
        rows.extend(method_rows)
        print(summary)
    write_outputs(Path(args.out_dir), rows, summaries, vars(args))


if __name__ == "__main__":
    main()
