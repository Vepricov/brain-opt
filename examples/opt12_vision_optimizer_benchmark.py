"""OPT-1/2 vision optimizer benchmark.

This benchmark is intentionally small and download-free. It trains the same
CNN classifier on a deterministic CIFAR-shaped task with AdamW, Lion and Muon,
optionally sweeping a user-facing LR grid per optimizer. The goal is to produce
clean optimizer-comparison artifacts for OPT-1/2 without relying on external
robotics logs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain_opt import get_optimizer, scale_lr


@dataclass
class RunSummary:
    optimizer: str
    lr: float
    effective_lr: float
    train_loss: float
    val_loss: float
    val_accuracy: float
    steps: int
    samples_seen: int
    seconds: float
    peak_memory_mb: float | None


class SmallVisionNet(nn.Module):
    def __init__(self, width: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2 * width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(2 * width * 8 * 8, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_synthetic_cifar(
    samples: int,
    seed: int,
    *,
    noise_std: float,
    patch_strength: float,
    cue_strength: float,
    num_classes: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    labels = torch.arange(samples) % num_classes
    labels = labels[torch.randperm(samples, generator=gen)]
    images = noise_std * torch.randn(samples, 3, 32, 32, generator=gen)
    colors = torch.eye(num_classes, 3)[labels % 3]
    for i, label in enumerate(labels):
        row = 2 + int(label % 5) * 6
        col = 2 + int(label // 5) * 12
        images[i, :, row : row + 8, col : col + 8] += colors[i].view(3, 1, 1) * patch_strength
        images[i, :, 26 - row // 2 : 30 - row // 2, 26 - col // 3 : 30 - col // 3] += cue_strength
    return images.clamp(0.0, 1.0), labels.long()


def iter_batches(x: torch.Tensor, y: torch.Tensor, batch_size: int, generator: torch.Generator):
    order = torch.randperm(x.size(0), generator=generator)
    for start in range(0, x.size(0), batch_size):
        ids = order[start : start + batch_size]
        yield x[ids], y[ids]


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
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


def train_one(args: argparse.Namespace, optimizer_name: str, device: torch.device) -> RunSummary:
    torch.manual_seed(args.seed)
    train_x, train_y = make_synthetic_cifar(
        args.train_samples,
        args.seed,
        noise_std=args.noise_std,
        patch_strength=args.patch_strength,
        cue_strength=args.cue_strength,
    )
    val_x, val_y = make_synthetic_cifar(
        args.val_samples,
        args.seed + 1,
        noise_std=args.noise_std,
        patch_strength=args.patch_strength,
        cue_strength=args.cue_strength,
    )
    model = SmallVisionNet(width=args.model_width).to(device)
    opt = get_optimizer(
        model.parameters(),
        name=optimizer_name,
        lr=args.lr,
        weight_decay=args.weight_decay,
        auto_scale_lr=not args.no_auto_scale_lr,
    )
    gen = torch.Generator().manual_seed(args.seed + 2)
    start_time = time.perf_counter()
    last_loss = math.nan
    step = 0
    samples_seen = 0
    while step < args.steps:
        for xb, yb in iter_batches(train_x, train_y, args.batch_size, gen):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
            samples_seen += int(yb.numel())
            step += 1
            if step >= args.steps:
                break
    elapsed = time.perf_counter() - start_time
    val_loss, val_accuracy = evaluate(model, val_x, val_y, args.batch_size, device)
    return RunSummary(
        optimizer=optimizer_name,
        lr=args.lr,
        effective_lr=effective_lr(optimizer_name, args),
        train_loss=last_loss,
        val_loss=val_loss,
        val_accuracy=val_accuracy,
        steps=step,
        samples_seen=samples_seen,
        seconds=elapsed,
        peak_memory_mb=peak_memory_mb(device),
    )


def effective_lr(optimizer_name: str, args: argparse.Namespace) -> float:
    if args.no_auto_scale_lr:
        return float(args.lr)
    return float(scale_lr(optimizer_name, args.lr))


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    return None


def parse_optimizer_lr_grid(spec: str) -> dict[str, list[float]]:
    if not spec.strip():
        return {}
    grid: dict[str, list[float]] = {}
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"bad --optimizer-lr-grid chunk: {chunk!r}")
        name, values = chunk.split("=", 1)
        lrs = [float(item) for item in values.replace("|", ",").split(",") if item.strip()]
        if not lrs:
            raise ValueError(f"empty LR list for optimizer {name!r}")
        grid[name.strip().lower()] = lrs
    return grid


def lrs_for_optimizer(args: argparse.Namespace, optimizer_name: str) -> list[float]:
    grid = parse_optimizer_lr_grid(args.optimizer_lr_grid)
    return grid.get(optimizer_name.lower(), [args.lr])


def write_outputs(out_dir: Path, summaries: Sequence[RunSummary]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(item) for item in summaries]
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    keys = list(rows[0].keys()) if rows else []
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# OPT-1/2 Vision Optimizer Benchmark", ""]
    for item in summaries:
        lines.append(
            f"- `{item.optimizer}` lr={item.lr:g} effective_lr={item.effective_lr:g}: "
            f"val_accuracy={item.val_accuracy:.4f}, val_loss={item.val_loss:.4f}, "
            f"train_loss={item.train_loss:.4f}, seconds={item.seconds:.2f}"
        )
    best_by_optimizer: dict[str, RunSummary] = {}
    for item in summaries:
        old = best_by_optimizer.get(item.optimizer)
        if old is None or item.val_accuracy > old.val_accuracy:
            best_by_optimizer[item.optimizer] = item
    lines.extend(["", "## Best per optimizer", ""])
    for name, item in sorted(best_by_optimizer.items()):
        lines.append(
            f"- `{name}`: val_accuracy={item.val_accuracy:.4f}, "
            f"val_loss={item.val_loss:.4f}, lr={item.lr:g}, effective_lr={item.effective_lr:g}"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    maybe_plot(out_dir, summaries)


def maybe_plot(out_dir: Path, summaries: Sequence[RunSummary]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = [f"{item.optimizer}\n{item.lr:g}" for item in summaries]
    fig, ax = plt.subplots(figsize=(max(8, len(summaries) * 0.8), 4))
    ax.bar(labels, [item.val_accuracy for item in summaries])
    ax.set_ylabel("validation accuracy")
    ax.set_title("OPT-1/2 vision demo: validation accuracy by optimizer/LR")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "val_accuracy_by_optimizer.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/opt12_vision_optimizer_benchmark")
    parser.add_argument("--optimizers", nargs="+", default=["AdamW", "Lion", "Muon"])
    parser.add_argument("--optimizer-lr-grid", default="")
    parser.add_argument("--no-auto-scale-lr", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=1024)
    parser.add_argument("--model-width", type=int, default=32)
    parser.add_argument("--noise-std", type=float, default=0.10)
    parser.add_argument("--patch-strength", type=float, default=1.0)
    parser.add_argument("--cue-strength", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    summaries: list[RunSummary] = []
    for optimizer_name in args.optimizers:
        for lr in lrs_for_optimizer(args, optimizer_name):
            run_args = copy.copy(args)
            run_args.lr = lr
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            summary = train_one(run_args, optimizer_name, device)
            summaries.append(summary)
            print(summary)
    write_outputs(Path(args.out_dir), summaries)


if __name__ == "__main__":
    main()
