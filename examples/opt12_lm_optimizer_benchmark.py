"""OPT-1/2 language-model optimizer benchmark.

The script has two modes.

Smoke mode is fully local and trains a tiny causal LM on a synthetic token
task. It verifies AdamW, Lion and Muon end to end without downloads.

Real mode fine-tunes a Hugging Face causal LM on a text dataset and can score
the resulting checkpoints on HellaSwag by normalized continuation likelihood.
GSM8K generation is optional and intentionally small by default because it is
slow and not a clean optimizer benchmark by itself.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

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
    tokens_seen: int
    steps: int
    seconds: float
    peak_memory_mb: float | None = None
    hellaswag_acc_norm: float | None = None
    gsm8k_exact_match: float | None = None


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.embed(input_ids)
        logits = self.head(self.net(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
            )
        return type("Output", (), {"loss": loss, "logits": logits})


def make_synthetic_tokens(
    *,
    samples: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    starts = torch.randint(0, vocab_size, (samples, 1), generator=gen)
    increments = torch.randint(1, 5, (samples, 1), generator=gen)
    positions = torch.arange(seq_len).view(1, seq_len)
    noise = torch.randint(0, 2, (samples, seq_len), generator=gen)
    return (starts + increments * positions + noise) % vocab_size


def iter_batches(tokens: torch.Tensor, batch_size: int, generator: torch.Generator):
    n = tokens.size(0)
    order = torch.randperm(n, generator=generator)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        batch = tokens[idx]
        yield batch, batch.clone()


def evaluate_synthetic(model: nn.Module, tokens: torch.Tensor, batch_size: int, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, tokens.size(0), batch_size):
            batch = tokens[start:start + batch_size].to(device)
            out = model(batch, labels=batch)
            losses.append(float(out.loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def train_synthetic(args: argparse.Namespace, optimizer_name: str, device: torch.device) -> RunSummary:
    torch.manual_seed(args.seed)
    train_tokens = make_synthetic_tokens(
        samples=args.smoke_train_samples,
        seq_len=args.seq_len,
        vocab_size=args.smoke_vocab_size,
        seed=args.seed,
    )
    val_tokens = make_synthetic_tokens(
        samples=args.smoke_val_samples,
        seq_len=args.seq_len,
        vocab_size=args.smoke_vocab_size,
        seed=args.seed + 1,
    )
    model = TinyCausalLM(args.smoke_vocab_size, hidden_size=args.smoke_hidden_size).to(device)
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
    tokens_seen = 0
    step = 0
    while step < args.steps:
        for x, y in iter_batches(train_tokens, args.batch_size, gen):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x, labels=y)
            out.loss.backward()
            opt.step()
            last_loss = float(out.loss.item())
            tokens_seen += int(x.numel())
            step += 1
            if step >= args.steps:
                break
    elapsed = time.perf_counter() - start_time
    val_loss = evaluate_synthetic(model, val_tokens, args.batch_size, device)
    return RunSummary(
        optimizer=optimizer_name,
        lr=args.lr,
        effective_lr=effective_lr(optimizer_name, args),
        train_loss=last_loss,
        val_loss=val_loss,
        tokens_seen=tokens_seen,
        steps=step,
        seconds=elapsed,
        peak_memory_mb=peak_memory_mb(device),
    )


def load_text_batches(args: argparse.Namespace, tokenizer):
    from datasets import load_dataset

    if args.dataset_config:
        ds = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
    else:
        ds = load_dataset(args.dataset, split=args.dataset_split)
    if args.max_train_samples:
        ds = ds.select(range(min(args.max_train_samples, len(ds))))

    texts = [str(row.get(args.text_field, "")) for row in ds if str(row.get(args.text_field, "")).strip()]
    joined = "\n\n".join(texts)
    encoded = tokenizer(joined, return_tensors="pt", add_special_tokens=True).input_ids[0]
    if encoded.numel() < args.seq_len + 2:
        raise ValueError("not enough tokens for the selected sequence length")
    blocks = []
    max_blocks = min(args.max_train_samples or 1024, (encoded.numel() - 1) // args.seq_len)
    for i in range(max_blocks):
        start = i * args.seq_len
        blocks.append(encoded[start:start + args.seq_len])
    return torch.stack(blocks)


def train_hf(args: argparse.Namespace, optimizer_name: str, device: torch.device) -> tuple[RunSummary, object, object]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokens = load_text_batches(args, tokenizer)
    split = max(1, int(tokens.size(0) * 0.9))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:] if split < tokens.size(0) else tokens[:1]

    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.train()
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
    tokens_seen = 0
    step = 0
    while step < args.steps:
        for x, y in iter_batches(train_tokens, args.batch_size, gen):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=x, labels=y)
            out.loss.backward()
            opt.step()
            last_loss = float(out.loss.item())
            tokens_seen += int(x.numel())
            step += 1
            if step >= args.steps:
                break
    elapsed = time.perf_counter() - start_time
    val_loss = evaluate_hf(model, val_tokens, args.batch_size, device)
    summary = RunSummary(
        optimizer=optimizer_name,
        lr=args.lr,
        effective_lr=effective_lr(optimizer_name, args),
        train_loss=last_loss,
        val_loss=val_loss,
        tokens_seen=tokens_seen,
        steps=step,
        seconds=elapsed,
        peak_memory_mb=peak_memory_mb(device),
    )
    return summary, model, tokenizer


def evaluate_hf(model, tokens: torch.Tensor, batch_size: int, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, tokens.size(0), batch_size):
            x = tokens[start:start + batch_size].to(device)
            out = model(input_ids=x, labels=x)
            losses.append(float(out.loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def evaluate_hellaswag(model, tokenizer, *, samples: int, device: torch.device) -> float:
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation")
    if samples:
        ds = ds.select(range(min(samples, len(ds))))
    correct = 0
    model.eval()
    for row in ds:
        ctx_b = str(row["ctx_b"])
        prompt = f"{row['ctx_a']} {ctx_b[:1].upper()}{ctx_b[1:]}"
        endings = list(row["endings"])
        scores = [
            normalized_completion_logprob(model, tokenizer, prompt, ending, device)
            for ending in endings
        ]
        pred = int(torch.tensor(scores).argmax().item())
        correct += int(pred == int(row["label"]))
    model.train()
    return correct / max(1, len(ds))


def normalized_completion_logprob(model, tokenizer, prompt: str, ending: str, device: torch.device) -> float:
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = tokenizer(prompt + " " + ending, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    if full_ids.size(1) <= prompt_ids.size(1):
        return float("-inf")
    labels = full_ids.clone()
    labels[:, :prompt_ids.size(1)] = -100
    with torch.no_grad():
        out = model(input_ids=full_ids, labels=labels)
    return -float(out.loss.item())


def evaluate_gsm8k(model, tokenizer, *, samples: int, device: torch.device, max_new_tokens: int) -> float:
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="test")
    ds = ds.select(range(min(samples, len(ds))))
    correct = 0
    model.eval()
    for row in ds:
        prompt = f"Question: {row['question']}\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(out[0, ids.size(1):], skip_special_tokens=True)
        correct += int(extract_number(text) == extract_number(row["answer"]))
    model.train()
    return correct / max(1, len(ds))


def extract_number(text: str) -> str:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return nums[-1] if nums else ""


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return None


def write_outputs(out_dir: Path, summaries: Sequence[RunSummary]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(s) for s in summaries]
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    keys = list(rows[0].keys()) if rows else []
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# OPT-1/2 LM Optimizer Benchmark", ""]
    for item in summaries:
        hs = "" if item.hellaswag_acc_norm is None else f", hellaswag={item.hellaswag_acc_norm:.4f}"
        gsm = "" if item.gsm8k_exact_match is None else f", gsm8k={item.gsm8k_exact_match:.4f}"
        lines.append(
            f"- `{item.optimizer}` lr={item.lr:g} effective_lr={item.effective_lr:g}: "
            f"train_loss={item.train_loss:.4f}, val_loss={item.val_loss:.4f}, "
            f"seconds={item.seconds:.2f}{hs}{gsm}"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    maybe_plot(out_dir, summaries)


def maybe_plot(out_dir: Path, summaries: Sequence[RunSummary]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    names = [s.optimizer for s in summaries]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, [s.val_loss for s in summaries])
    ax.set_ylabel("validation loss")
    ax.set_title("OPT-1/2 demo: validation loss by optimizer")
    fig.tight_layout()
    fig.savefig(out_dir / "val_loss_by_optimizer.png", dpi=160)
    plt.close(fig)
    if any(s.hellaswag_acc_norm is not None for s in summaries):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(names, [s.hellaswag_acc_norm or 0.0 for s in summaries])
        ax.set_ylabel("HellaSwag acc_norm")
        ax.set_title("OPT-1/2 demo: HellaSwag by optimizer")
        fig.tight_layout()
        fig.savefig(out_dir / "hellaswag_by_optimizer.png", dpi=160)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/opt12_lm_optimizer_benchmark")
    parser.add_argument("--optimizers", nargs="+", default=["AdamW", "Lion", "Muon"])
    parser.add_argument("--smoke", action="store_true", help="Run local synthetic benchmark without downloads")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-train-samples", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer-lr-grid",
        default="",
        help=(
            "Per-optimizer user-facing LR grid, e.g. "
            "'AdamW=5e-4,1e-3;Lion=1e-4,3e-4;Muon=1e-3,3e-3,1e-2'. "
            "Missing optimizers use --lr."
        ),
    )
    parser.add_argument(
        "--no-auto-scale-lr",
        action="store_true",
        help="Forward --lr values directly to optimizers instead of using brain_opt LR scaling.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hellaswag-samples", type=int, default=0)
    parser.add_argument("--gsm8k-samples", type=int, default=0)
    parser.add_argument("--gsm8k-max-new-tokens", type=int, default=64)
    parser.add_argument("--smoke-train-samples", type=int, default=256)
    parser.add_argument("--smoke-val-samples", type=int, default=64)
    parser.add_argument("--smoke-vocab-size", type=int, default=64)
    parser.add_argument("--smoke-hidden-size", type=int, default=64)
    return parser.parse_args()


def effective_lr(optimizer_name: str, args: argparse.Namespace) -> float:
    if args.no_auto_scale_lr:
        return float(args.lr)
    return float(scale_lr(optimizer_name, args.lr))


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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    summaries: List[RunSummary] = []
    for optimizer_name in args.optimizers:
        for lr in lrs_for_optimizer(args, optimizer_name):
            run_args = copy.copy(args)
            run_args.lr = lr
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            if run_args.smoke:
                summary = train_synthetic(run_args, optimizer_name, device)
            else:
                summary, model, tokenizer = train_hf(run_args, optimizer_name, device)
                if run_args.hellaswag_samples > 0:
                    summary.hellaswag_acc_norm = evaluate_hellaswag(
                        model,
                        tokenizer,
                        samples=run_args.hellaswag_samples,
                        device=device,
                    )
                if run_args.gsm8k_samples > 0:
                    summary.gsm8k_exact_match = evaluate_gsm8k(
                        model,
                        tokenizer,
                        samples=run_args.gsm8k_samples,
                        device=device,
                        max_new_tokens=run_args.gsm8k_max_new_tokens,
                    )
            summaries.append(summary)
            print(summary)
    write_outputs(Path(args.out_dir), summaries)


if __name__ == "__main__":
    main()
