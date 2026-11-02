"""SimplexLoRA rank search for ordinary PEFT linear LoRA adapters.

Implements Sec. 3 and App. B of Shalygin et al., "SimplexLoRA: Dynamic Rank
Updating via Alternating Minimization with Simplex Projection" (user PDF).
This is a paper-based implementation, not a copy of the author's PEFT fork.
The inspected author commit 8233d7ba uses ceil; the supplied paper uses floor,
which is used here. Search updates omega with SGD and A/B with AdamW.

Usage with an existing, freshly initialized PeftModel::

    search = SimplexLoRA(model, SimplexLoRAConfig(total_steps=1000))
    for batch in batches:
        search.zero_grad()
        model(**batch).loss.backward()
        search.step()
    search.save_pretrained(output_dir)

Requires PEFT only on construction. Does not patch PEFT or Transformers.
The first version supports one unmerged, non-quantized linear adapter per
target on one device. Targets must have equal parameter cost per rank so
the paper's rank-sum bound also bounds the actual A/B parameter count.
For Qwen with grouped-query attention, e.g. target q_proj and o_proj, not
all q/k/v projections together. Search gates add n temporary parameters.

Rank search uses K cycles of T optimizer steps. Changed ranks reset A/B
and their Adam moments. At the last boundary ALL adapters are reinitialized,
omega is removed from the forward pass, and the final LoRA schedule starts.
Search learning rate decays linearly from search_lr_scale * lr. The final
phase restarts linear warm-up/decay; omega's learning rate is constant.
Saved output is a standard PEFT adapter, not a standalone base checkpoint.
Mid-search save/resume, AMP GradScaler, distributed and quantized execution
are not supported by this controller. BF16 without GradScaler is supported.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn


def project_simplex(values: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto {w >= 0, sum(w) = len(values)}."""
    if values.ndim != 1 or values.numel() == 0 or not values.is_floating_point():
        raise ValueError("expected a nonempty floating-point vector")
    if not torch.isfinite(values).all():
        raise ValueError("simplex weights must be finite")
    # Centering avoids catastrophic cancellation for large common offsets.
    x = values.detach().to(torch.float64)
    x = x - x.max()
    sorted_x = x.sort(descending=True).values
    counts = torch.arange(1, x.numel() + 1, device=x.device, dtype=x.dtype)
    thresholds = (sorted_x.cumsum(0) - x.numel()) / counts
    active = sorted_x > thresholds
    theta = thresholds[active][-1]
    return (x - theta).clamp_min(0).to(values.dtype)


@dataclass(frozen=True)
class SimplexLoRAConfig:
    total_steps: int = 1000
    initial_rank: int = 8
    rank_updates: int = 5
    steps_per_update: int = 10
    lr: float = 1e-4
    omega_lr: float = 5.0
    regularization: float = 0.0
    search_lr_scale: float = 0.1
    warmup_steps: int = 50

    def __post_init__(self) -> None:
        for name in ("total_steps", "initial_rank", "rank_updates", "steps_per_update"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        final_steps = self.total_steps - self.rank_updates * self.steps_per_update
        if final_steps <= 0:
            raise ValueError("total_steps must include training AFTER rank search")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < final_steps:
            raise ValueError("warmup_steps must be nonnegative and shorter than final training")
        for name in ("lr", "omega_lr", "search_lr_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("regularization must be finite and nonnegative")


class SimplexLoRA:
    """Owns the optimizer so resizing cannot leave stale optimizer parameters."""

    def __init__(self, model: nn.Module, config: SimplexLoRAConfig):
        from peft import PeftModel
        from peft.tuners.lora.layer import Linear

        if not isinstance(model, PeftModel) or len(model.peft_config) != 1:
            raise ValueError("expected a PeftModel with exactly one LoRA adapter")
        self.adapter = next(iter(model.peft_config))
        peft_config = model.peft_config[self.adapter]
        if (peft_config.bias != "none" or peft_config.modules_to_save
                or getattr(peft_config, "use_dora", False)
                or getattr(peft_config, "use_rslora", False)):
            raise ValueError("only plain LoRA with bias='none' and no modules_to_save is supported")
        self.layers = {
            name: layer for name, layer in model.named_modules()
            if hasattr(layer, "lora_A") and self.adapter in layer.lora_A
        }
        if not self.layers:
            raise ValueError("no linear LoRA adapters found")
        for layer in self.layers.values():
            if (not isinstance(layer, Linear) or type(layer.get_base_layer()) is not nn.Linear
                    or layer.merged or layer.disable_adapters
                    or layer.active_adapters != [self.adapter]):
                raise ValueError("only active, unmerged, non-quantized linear LoRA is supported")
            if layer.r[self.adapter] != config.initial_rank:
                raise ValueError("all initial ranks must equal config.initial_rank")
            if (layer.lora_A[self.adapter].bias is not None
                    or layer.lora_B[self.adapter].bias is not None):
                raise ValueError("LoRA A/B biases are not supported")
        costs = {layer.in_features + layer.out_features for layer in self.layers.values()}
        if len(costs) != 1:
            raise ValueError("select targets with equal parameter cost per rank; rank budget != parameter budget")
        parameters = self._adapter_parameters()
        trainable = [p for p in model.parameters() if p.requires_grad]
        if {id(p) for p in trainable} != {id(p) for p in parameters}:
            raise ValueError("only the selected A/B matrices may be trainable")
        if len({p.device for p in model.parameters()}) != 1:
            raise ValueError("multi-device and offloaded models are not supported")
        self._adapter_dtypes = {name: layer.lora_A[self.adapter].weight.dtype
                                for name, layer in self.layers.items()}
        self.model = model
        self.config = config
        self.steps_done = 0
        self.search_steps = config.rank_updates * config.steps_per_update
        self.initial_parameter_count = sum(p.numel() for p in parameters)
        self.omega = nn.Parameter(torch.ones(len(self.layers), device=parameters[0].device))
        self.optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=0.0)
        self._hooks: Dict[str, Any] = {}
        for i, (name, layer) in enumerate(self.layers.items()):
            self._attach_gate(name, layer, i)
        self.history: List[Dict[str, Any]] = [self.snapshot()]

    @property
    def searching(self) -> bool:
        return self.steps_done < self.search_steps

    def _adapter_parameters(self) -> List[nn.Parameter]:
        return [matrix[self.adapter].weight for layer in self.layers.values()
                for matrix in (layer.lora_A, layer.lora_B) if self.adapter in matrix]

    def _attach_gate(self, name: str, layer: nn.Module, index: int) -> None:
        if self.adapter not in layer.lora_B:
            return

        def gate(module: nn.Module, inputs: Any, output: torch.Tensor) -> torch.Tensor:
            return output * self.omega[index].to(output.dtype)

        self._hooks[name] = layer.lora_B[self.adapter].register_forward_hook(gate)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self.omega.grad = None

    def _learning_rate(self) -> float:
        cfg = self.config
        if self.searching:
            return cfg.lr * cfg.search_lr_scale * (1 - self.steps_done / self.search_steps)
        step = self.steps_done - self.search_steps
        duration = cfg.total_steps - self.search_steps
        if step < cfg.warmup_steps:
            return cfg.lr * step / cfg.warmup_steps
        return cfg.lr * (duration - step) / (duration - cfg.warmup_steps)

    @torch.no_grad()
    def step(self) -> Dict[str, Any]:
        """Call once after backward (or completed gradient accumulation)."""
        if self.steps_done >= self.config.total_steps:
            raise RuntimeError("configured training budget is exhausted")
        parameters = self._adapter_parameters() + ([self.omega] if self.searching else [])
        gradients = [p.grad for p in parameters if p.grad is not None]
        if not gradients:
            raise RuntimeError("backward must precede step")
        if not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError("non-finite gradient; no optimizer step applied")
        lr = self._learning_rate()
        self.optimizer.param_groups[0]["lr"] = lr
        if self.searching:
            gradient = self.config.regularization * (self.omega - 1)
            if self.omega.grad is not None:
                gradient = gradient + self.omega.grad
            proposed = self.omega - self.config.omega_lr * gradient
            if not torch.isfinite(proposed).all():
                raise ValueError("non-finite importance update; no optimizer step applied")
            self.omega.copy_(proposed)
        self.optimizer.step()
        self.steps_done += 1
        rank_update = (self.steps_done <= self.search_steps
                       and self.steps_done % self.config.steps_per_update == 0)
        if rank_update:
            self._update_ranks()
        return {"step": self.steps_done, "adapter_lr": lr, "rank_update": rank_update}

    @torch.no_grad()
    def _update_ranks(self) -> None:
        projected = project_simplex(self.omega.double())
        ranks = torch.floor(self.config.initial_rank * projected).to(torch.int64).tolist()
        if not any(ranks) or sum(ranks) > len(ranks) * self.config.initial_rank:
            raise RuntimeError("invalid projected rank budget")
        self.omega.copy_(projected)
        final = self.steps_done == self.search_steps
        for i, ((name, layer), rank) in enumerate(zip(self.layers.items(), ranks)):
            if not final and rank == layer.r[self.adapter]:
                continue
            if name in self._hooks:
                self._hooks.pop(name).remove()
            for matrix in (layer.lora_A, layer.lora_B):
                if self.adapter in matrix:
                    self.optimizer.state.pop(matrix[self.adapter].weight, None)
            if rank == 0:
                # PEFT does not accept r=0 in update_layer. Omit this adapter.
                for matrix in (layer.lora_A, layer.lora_B):
                    if self.adapter in matrix:
                        del matrix[self.adapter]
                layer.r[self.adapter] = 0
                layer.scaling[self.adapter] = 0.0
            else:
                layer.update_layer(
                    self.adapter, r=rank, lora_alpha=layer.lora_alpha[self.adapter],
                    lora_dropout=getattr(layer.lora_dropout[self.adapter], "p", 0.0),
                    init_lora_weights=False, use_rslora=False, use_dora=False,
                )
                # PEFT may cast new adapters to BF16 base weights. Preserve the
                # original adapter precision, including after a zero-rank cycle.
                for matrix in (layer.lora_A, layer.lora_B):
                    matrix[self.adapter].to(dtype=self._adapter_dtypes[name])
                layer.reset_lora_parameters(self.adapter, init_lora_weights=True)
                if not final:
                    self._attach_gate(name, layer, i)
        self.optimizer.param_groups[0]["params"] = self._adapter_parameters()
        row = self.snapshot()
        row["projected_omega"] = projected.tolist()
        if final:
            self.omega.fill_(1.0)
            self.omega.requires_grad_(False)
            self.omega.grad = None
            self._update_export_config()
            row["omega"] = self.omega.tolist()
        self.history.append(row)

    def _update_export_config(self) -> None:
        # PEFT targets are names in the unwrapped base model, not wrapper paths.
        prefix = "base_model.model."
        pattern = {name[len(prefix):] if name.startswith(prefix) else name: layer.r[self.adapter]
                   for name, layer in self.layers.items() if layer.r[self.adapter] > 0}
        config = self.model.peft_config[self.adapter]
        config.target_modules = set(pattern)
        config.rank_pattern = pattern

    def snapshot(self) -> Dict[str, Any]:
        count = sum(p.numel() for p in self._adapter_parameters())
        return {
            "step": self.steps_done,
            "phase": "rank_search" if self.searching else "fixed_rank_training",
            "ranks": {name: layer.r[self.adapter] for name, layer in self.layers.items()},
            "omega": self.omega.detach().tolist(),
            "adapter_parameters": count,
            "search_gate_parameters": self.omega.numel() if self.searching else 0,
            "initial_adapter_parameters": self.initial_parameter_count,
            "adapter_parameter_budget_respected": count <= self.initial_parameter_count,
        }

    def save_pretrained(self, output_dir: str) -> None:
        """Save a normal PEFT adapter plus rank-search evidence after final training."""
        if self.steps_done != self.config.total_steps:
            raise RuntimeError("finish configured training before exporting; search resume is unsupported")
        output = Path(output_dir)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("refusing to overwrite nonempty adapter output")
        self.model.save_pretrained(str(output))
        evidence = {"method": "SimplexLoRA", "implementation": "paper-floor-v1",
                    "config": asdict(self.config), "history": self.history,
                    "final": self.snapshot(), "format": "PEFT adapter; base checkpoint required"}
        (output / "simplexlora_rank_history.json").write_text(
            json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
