"""Single-device causal-LM training from local HF checkpoints and UTF-8 text.

Optional dependencies are imported at runtime. No PlatformAPI, network downloads,
remote model code, synthetic-data fallback or changes to third-party packages.
The output contains a standalone checkpoint or a PEFT adapter, its base manifest,
metrics and a reload check. This is not a resumable optimizer-state checkpoint.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch

from .factory import available_optimizers, get_optimizer
from .simplex_lora import SimplexLoRA, SimplexLoRAConfig
from .weight_lora import WeightLoRA, WeightLoRAConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_manifest(path: Path) -> dict:
    """Portable relative names and hashes, without platform run credentials."""
    return {str(file.relative_to(path)): sha256_file(file)
            for file in sorted(path.rglob("*")) if file.is_file()}


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class TrainingConfig:
    mode: str = "lora"
    initialization: str = "pretrained"
    export_merged: bool = False
    optimizer: str = "AdamW"
    steps: int = 100
    batch_size: int = 1
    accumulation: int = 1
    seq_len: int = 128
    train_blocks: int = 128
    val_blocks: int = 32
    lr: float = 5e-5
    weight_decay: float = 0.0
    auto_scale_lr: bool = False
    rank: int = 8
    lora_alpha: float = 16.0
    target_modules: Tuple[str, ...] = ("q_proj", "o_proj")
    device: str = "cpu"
    dtype: str = "float32"
    seed: int = 123
    gradient_checkpointing: bool = False
    simplex: Optional[SimplexLoRAConfig] = None
    weight_lora: Optional[WeightLoRAConfig] = None

    def __post_init__(self) -> None:
        if self.initialization not in {"pretrained", "scratch"}:
            raise ValueError("initialization must be pretrained or scratch")
        if self.initialization == "scratch" and self.mode != "full":
            raise ValueError("training from scratch requires full mode, not a frozen random base")
        for name in ("export_merged", "auto_scale_lr", "gradient_checkpointing"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.mode not in {"full", "lora", "simplexlora", "weightlora"}:
            raise ValueError("mode must be full, lora, simplexlora or weightlora")
        if self.optimizer not in available_optimizers():
            raise ValueError("unknown optimizer")
        for name in ("steps", "batch_size", "accumulation", "seq_len", "train_blocks", "val_blocks", "rank"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.seq_len < 2 or self.train_blocks < self.batch_size:
            raise ValueError("need seq_len >= 2 and a full training microbatch")
        for name in ("lr", "lora_alpha"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and nonnegative")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("supported dtypes: float32, bfloat16 (no FP16 GradScaler)")
        if torch.device(self.device).type not in {"cpu", "cuda"}:
            raise ValueError("supported devices: cpu, cuda")
        if isinstance(self.target_modules, str) or not self.target_modules or not all(
                isinstance(x, str) and x for x in self.target_modules):
            raise ValueError("target_modules must be a nonempty sequence of names")
        if self.mode == "simplexlora":
            if not isinstance(self.simplex, SimplexLoRAConfig):
                raise ValueError("simplexlora requires a SimplexLoRAConfig")
            if (self.simplex.total_steps != self.steps or self.simplex.initial_rank != self.rank
                    or self.simplex.lr != self.lr):
                raise ValueError("SimplexLoRA steps, initial_rank and lr must match training")
            if self.optimizer != "AdamW" or self.auto_scale_lr or self.weight_decay != 0:
                raise ValueError("SimplexLoRA uses its own unscaled AdamW with zero weight decay")
        elif self.simplex is not None:
            raise ValueError("simplex config is only valid in simplexlora mode")
        if self.mode == "weightlora":
            if not isinstance(self.weight_lora, WeightLoRAConfig):
                raise ValueError("weightlora requires a WeightLoRAConfig")
            if self.weight_lora.total_steps != self.steps or self.weight_lora.lr != self.lr:
                raise ValueError("WeightLoRA steps and lr must match training")
            if self.optimizer != "AdamW" or self.auto_scale_lr or self.weight_decay != 0:
                raise ValueError("WeightLoRA uses its own unscaled AdamW with zero weight decay")
        elif self.weight_lora is not None:
            raise ValueError("weight_lora config is only valid in weightlora mode")


def pack_text(path: Path, tokenizer, blocks: int, seq_len: int) -> torch.Tensor:
    """Pack complete fixed-length blocks, with EOS between nonempty lines."""
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")
    tokens = []
    needed = blocks * seq_len
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            tokens.extend(tokenizer(line, add_special_tokens=False)["input_ids"])
            tokens.append(tokenizer.eos_token_id)
            if len(tokens) >= needed:
                break
    if len(tokens) < needed:
        raise ValueError(f"insufficient real text in {path.name}: need {needed} tokens, got {len(tokens)}")
    return torch.tensor(tokens[:needed], dtype=torch.long).reshape(blocks, seq_len)


def load_tokenizer(path: Path):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    metadata = path / "tokenizer_config.json"
    tokenizer_class = json.loads(metadata.read_text(encoding="utf-8")).get("tokenizer_class") if metadata.is_file() else None
    # A generic serialized tokenizer is not necessarily the architecture's
    # default BPE. Preserve its backend rather than infer one from model_type.
    loader = PreTrainedTokenizerFast if tokenizer_class in {"PreTrainedTokenizerFast", "TokenizersBackend"} else AutoTokenizer
    return loader.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)


def evaluate(model, tokens: torch.Tensor, batch_size: int, device: torch.device) -> float:
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for batch in tokens.split(batch_size):
            inputs = batch.to(device)
            loss = float(model(input_ids=inputs, labels=inputs, use_cache=False).loss)
            if not math.isfinite(loss):
                raise ValueError("non-finite evaluation loss")
            total += loss * len(batch)
    return total / len(tokens)


def resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def token_data_manifest(train: torch.Tensor, validation: torch.Tensor) -> dict:
    def packed_bytes(tokens):
        return tokens.numpy().astype("<i8").tobytes()

    train_blocks = {hashlib.sha256(packed_bytes(row)).hexdigest() for row in train}
    overlaps = sum(hashlib.sha256(packed_bytes(row)).hexdigest() in train_blocks for row in validation)
    if overlaps == len(validation):
        raise ValueError("all validation token blocks occur in training data")
    return {"train_tokens_sha256": hashlib.sha256(packed_bytes(train)).hexdigest(),
            "validation_tokens_sha256": hashlib.sha256(packed_bytes(validation)).hexdigest(),
            "validation_blocks_in_train": overlaps, "validation_blocks": len(validation),
            "overlap_fraction": overlaps / len(validation)}


def _load_base(path: Path, config: TrainingConfig, *, initialize: bool = False):
    from transformers import AutoConfig, AutoModelForCausalLM

    architecture = AutoConfig.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if getattr(architecture, "quantization_config", None):
        raise ValueError("this training path requires an unquantized base; QLoRA is not yet integrated")
    if initialize:
        model = AutoModelForCausalLM.from_config(
            architecture, trust_remote_code=False, torch_dtype=getattr(torch, config.dtype))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False,
            torch_dtype=getattr(torch, config.dtype))
    return model.to(resolve_device(config.device))


def restore_parameter_dtypes(model, original: dict) -> None:
    parameters = dict(model.named_parameters())
    if parameters.keys() != original.keys():
        raise ValueError("merged model parameter names differ from the base model")
    for name, parameter in parameters.items():
        if parameter.dtype != original[name]:
            parameter.data = parameter.data.to(dtype=original[name])


def run_training(base_checkpoint: str, train_text: str, validation_text: str,
                 output_dir: str, config: TrainingConfig) -> dict:
    """Train and verify an export. Refuse any pre-existing output directory."""
    base, train_path, val_path, output = map(
        lambda p: Path(p).resolve(), (base_checkpoint, train_text, validation_text, output_dir))
    if not (base / "config.json").is_file() or not train_path.is_file() or not val_path.is_file():
        raise ValueError("provide a local HF checkpoint and separate UTF-8 training/validation files")
    if base == output or base in output.parents or output in base.parents:
        raise ValueError("output and input checkpoint must not contain each other")
    if train_path == val_path or sha256_file(train_path) == sha256_file(val_path):
        raise ValueError("training and validation files must be different")
    device = resolve_device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
        if config.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("selected CUDA device does not support BF16")
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(config.seed)
    tokenizer = load_tokenizer(base)
    train = pack_text(train_path, tokenizer, config.train_blocks, config.seq_len)
    validation = pack_text(val_path, tokenizer, config.val_blocks, config.seq_len)
    token_manifest = token_data_manifest(train, validation)
    model = _load_base(base, config, initialize=config.initialization == "scratch")
    original_parameter_dtypes = {name: p.dtype for name, p in model.named_parameters()}
    if config.mode != "full":
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=config.rank, lora_alpha=config.lora_alpha, lora_dropout=0,
            target_modules=list(config.target_modules), bias="none", task_type="CAUSAL_LM"))
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    initial_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    controller = (SimplexLoRA(model, config.simplex) if config.mode == "simplexlora"
                  else WeightLoRA(model, config.weight_lora) if config.mode == "weightlora" else None)
    # SoftSignum and SoftMuon schedule their temperature over the whole run.
    schedule = {"total_steps": config.steps} if config.optimizer in {"SoftSignum", "SoftMuon"} else {}
    optimizer = None if controller else get_optimizer(
        (p for p in model.parameters() if p.requires_grad), name=config.optimizer,
        lr=config.lr, weight_decay=config.weight_decay, auto_scale_lr=config.auto_scale_lr, **schedule)
    protocol = {
        "config": asdict(config), "base_checkpoint_files": checkpoint_manifest(base),
        "dataset": {"train_sha256": sha256_file(train_path), "validation_sha256": sha256_file(val_path),
                    **token_manifest},
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        "format": "HF checkpoint" if config.mode == "full" or config.export_merged else "PEFT adapter; base checkpoint required",
        "input_weights_used": config.initialization == "pretrained",
        "base_role": "pretrained weights" if config.initialization == "pretrained" else "architecture and tokenizer only",
        "resume_optimizer_state": False,
        "scope": "library execution; not evidence of a platform run or acceptance",
    }
    if config.mode != "full":
        protocol["versions"]["peft"] = importlib.metadata.version("peft")
    write_json(output / "protocol.json", protocol)
    baseline = evaluate(model, validation, config.batch_size, device)
    generator = torch.Generator().manual_seed(config.seed)
    order = torch.randperm(len(train), generator=generator)
    offset = 0
    history = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    for step in range(config.steps):
        if controller:
            controller.zero_grad()
        else:
            optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(config.accumulation):
            if offset + config.batch_size > len(order):
                order = torch.randperm(len(train), generator=generator)
                offset = 0
            batch = train[order[offset:offset + config.batch_size]].to(device)
            offset += config.batch_size
            loss = model(input_ids=batch, labels=batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError(f"non-finite loss at step {step + 1}")
            step_loss += float(loss.detach()) / config.accumulation
            (loss / config.accumulation).backward()
        if controller:
            update = controller.step()
        else:
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise ValueError("non-finite gradient; optimizer step not applied")
            optimizer.step()
            update = {"step": step + 1, "lr": optimizer.param_groups[0]["lr"]}
        history.append({**update, "loss": step_loss})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    final = evaluate(model, validation, config.batch_size, device)
    final_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    checkpoint = output / "checkpoint"
    if controller:
        controller.save_pretrained(str(checkpoint))
    else:
        model.save_pretrained(str(checkpoint), safe_serialization=True)
    tokenizer.save_pretrained(str(checkpoint))
    if config.mode != "full" and config.export_merged:
        # Keep adapters and rank-search evidence, but offer a standalone model
        # to consumers that do not support PEFT or a second base-model input.
        adapter_path = output / "adapter"
        checkpoint.rename(adapter_path)
        model = model.merge_and_unload(safe_merge=True)
        # PEFT can promote merged parameters. Never cast the whole model here:
        # nonpersistent FP32 rotary buffers are reconstructed at reload time.
        restore_parameter_dtypes(model, original_parameter_dtypes)
        merged_loss = evaluate(model, validation, config.batch_size, device)
        # Low-precision merged weights can round differently. Record this
        # separately; reload itself must reproduce the actual exported model.
        write_json(output / "merge_check.json", {
            "adapter_val_loss": final, "merged_val_loss": merged_loss,
            "absolute_loss_difference": abs(merged_loss - final),
            "quality_gate_applied": False,
        })
        final = merged_loss
        model.save_pretrained(str(checkpoint), safe_serialization=True)
        tokenizer.save_pretrained(str(checkpoint))
    # Release training state before loading a second copy onto the same GPU.
    del loss, batch, model, optimizer, controller
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    standalone = config.mode == "full" or config.export_merged
    reloaded = _load_base(checkpoint if standalone else base, config)
    exported_tokenizer = load_tokenizer(checkpoint)
    reloaded_validation = pack_text(val_path, exported_tokenizer, config.val_blocks, config.seq_len)
    if not torch.equal(validation, reloaded_validation):
        raise RuntimeError("exported tokenizer changed the validation tokens")
    if not standalone:
        from peft import PeftModel
        reloaded = PeftModel.from_pretrained(reloaded, str(checkpoint), is_trainable=False)
    reload_loss = evaluate(reloaded, validation, config.batch_size, device)
    if not math.isclose(final, reload_loss, rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError(f"checkpoint reload changed loss: {final} -> {reload_loss}")
    metrics = {
        "status": "COMPLETED", "baseline_loss": baseline, "val_loss": final,
        "val_perplexity": math.exp(final), "reload_loss": reload_loss, "reload_verified": True,
        "tokenizer_reload_verified": True,
        "initial_trainable_parameters": initial_trainable, "final_trainable_parameters": final_trainable,
        "training_seconds": seconds, "training_peak_allocated_bytes": peak,
        "memory_scope": "process CUDA allocation peak during training; excludes loading/evaluation/reload",
        "input_tokens_seen": config.steps * config.batch_size * config.accumulation * config.seq_len,
        "history": history, "output_checkpoint_files": checkpoint_manifest(checkpoint),
    }
    write_json(output / "metrics.json", metrics)
    return metrics
