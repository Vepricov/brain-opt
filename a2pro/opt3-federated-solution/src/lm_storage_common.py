"""Shared helpers for A2.Pro LM fine-tuning stages.

The stages use A2.Pro checkpoint storage as the model source. Offline mode
creates a tiny Hugging Face-compatible causal LM checkpoint locally so the
PlatformAPI path can be smoke-tested without network access.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from brain_opt import get_optimizer, scale_lr


@dataclass
class OptimizerRun:
    method: str
    train_loss: float
    val_loss: float
    steps: int
    tokens_seen: int
    seconds: float
    lr: float | None = None
    effective_lr: float | None = None
    peak_memory_mb: float | None = None
    mean_staleness: float | None = None
    max_staleness: int | None = None


def nested(params: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = params
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def clean_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_methods(value: Any, default: Sequence[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        if value.lower() == "all":
            return list(default)
        return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return list(default)


def make_tiny_checkpoint(path: Path, *, seq_len: int, vocab_size: int = 128, hidden_size: int = 64) -> Path:
    """Create a tiny HF-compatible causal LM checkpoint for offline smoke runs."""
    from transformers import GPT2Config, GPT2LMHeadModel

    clean_dir(path)
    n_positions = max(64, seq_len + 8)
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_ctx=n_positions,
        n_embd=hidden_size,
        n_layer=2,
        n_head=2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(path, safe_serialization=False)
    (path / "README.md").write_text(
        "Tiny HF-compatible causal LM checkpoint generated for A2.Pro offline smoke.\n",
        encoding="utf-8",
    )
    return path


def load_causal_lm(path: Path, device: torch.device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True)
    model.to(device)
    model.train()
    return model


def maybe_load_tokenizer(path: Path):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception:
        return None


def save_causal_lm(model: Any, output_dir: Path, *, tokenizer: Any = None) -> None:
    clean_dir(output_dir)
    model.save_pretrained(output_dir, safe_serialization=False)
    if tokenizer is not None:
        try:
            tokenizer.save_pretrained(output_dir)
        except Exception:
            pass


def infer_arch_name(model_dir: Path, default: str = "hf-causal-lm") -> str:
    config_path = model_dir / "config.json"
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return default
    source = str(raw.get("_name_or_path") or raw.get("model_type") or default).strip().strip("/")
    if not source or source.startswith(("/tmp", "tmp/", "var/")):
        return default
    return source.rsplit("/", 1)[-1].lower().replace("_", "-")


def make_lm_tokens(*, samples: int, seq_len: int, vocab_size: int, seed: int) -> torch.Tensor:
    """Deterministic synthetic token blocks for storage-first LM smoke runs."""
    gen = torch.Generator().manual_seed(seed)
    usable_vocab = max(8, int(vocab_size))
    starts = torch.randint(0, usable_vocab, (samples, 1), generator=gen)
    increments = torch.randint(1, 7, (samples, 1), generator=gen)
    positions = torch.arange(seq_len).view(1, seq_len)
    noise = torch.randint(0, 3, (samples, seq_len), generator=gen)
    return (starts + increments * positions + noise) % usable_vocab


def iter_batches(tokens: torch.Tensor, batch_size: int, generator: torch.Generator):
    order = torch.randperm(tokens.size(0), generator=generator)
    for start in range(0, tokens.size(0), batch_size):
        idx = order[start:start + batch_size]
        batch = tokens[idx]
        yield batch, batch.clone()


def evaluate_lm(model: Any, tokens: torch.Tensor, batch_size: int, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for start in range(0, tokens.size(0), batch_size):
            x = tokens[start:start + batch_size].to(device)
            out = model(input_ids=x, attention_mask=torch.ones_like(x), labels=x)
            losses.append(float(out.loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def train_one_optimizer(
    model: Any,
    *,
    optimizer_name: str,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    steps: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    auto_scale_lr: bool,
    seed: int,
    device: torch.device,
) -> tuple[OptimizerRun, Any]:
    model.train()
    opt = get_optimizer(
        model.parameters(),
        name=optimizer_name,
        lr=lr,
        weight_decay=weight_decay,
        auto_scale_lr=auto_scale_lr,
    )
    gen = torch.Generator().manual_seed(seed)
    start_time = time.perf_counter()
    tokens_seen = 0
    step = 0
    last_loss = math.nan
    while step < steps:
        for x, y in iter_batches(train_tokens, batch_size, gen):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(input_ids=x, attention_mask=torch.ones_like(x), labels=y)
            out.loss.backward()
            opt.step()
            last_loss = float(out.loss.item())
            tokens_seen += int(x.numel())
            step += 1
            if step >= steps:
                break
    elapsed = time.perf_counter() - start_time
    val_loss = evaluate_lm(model, val_tokens, batch_size, device)
    peak = peak_memory_mb(device)
    effective = lr if not auto_scale_lr else float(scale_lr(optimizer_name, lr))
    return (
        OptimizerRun(
            method=optimizer_name,
            lr=lr,
            effective_lr=effective,
            train_loss=last_loss,
            val_loss=val_loss,
            steps=step,
            tokens_seen=tokens_seen,
            seconds=elapsed,
            peak_memory_mb=peak,
        ),
        model,
    )


def causal_lm_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].contiguous().view(-1, logits.size(-1)),
        target[:, 1:].contiguous().view(-1),
    )


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
    return None


def write_metrics(metrics_dir: Path, rows: Sequence[OptimizerRun], metadata: dict[str, Any]) -> None:
    clean_dir(metrics_dir)
    row_dicts = [asdict(row) for row in rows]
    payload = {"metadata": metadata, "results": row_dicts}
    (metrics_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    keys = list(row_dicts[0].keys()) if row_dicts else []
    with (metrics_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(row_dicts)
    lines = ["# A2.Pro LM Fine-tuning Metrics", ""]
    for key, value in metadata.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    for row in rows:
        stale = "" if row.mean_staleness is None else f", mean_staleness={row.mean_staleness:.2f}"
        lines.append(
            f"- `{row.method}`: train_loss={row.train_loss:.4f}, "
            f"val_loss={row.val_loss:.4f}, steps={row.steps}{stale}"
        )
    (metrics_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    maybe_plot(metrics_dir, rows)


def maybe_plot(metrics_dir: Path, rows: Sequence[OptimizerRun]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    names = [row.method for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, [row.val_loss for row in rows])
    ax.set_ylabel("validation loss")
    ax.set_title("A2.Pro LM fine-tuning: validation loss")
    fig.tight_layout()
    fig.savefig(metrics_dir / "val_loss_by_method.png", dpi=160)
    plt.close(fig)
