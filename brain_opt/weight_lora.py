"""WeightLoRA: keep only the necessary LoRA adapters.

Method of the BRAIn Lab (Veprikov, Solodkin et al., "WeightLoRA: Keep Only
Necessary Adapters"). This is a paper-based implementation for PEFT linear
LoRA adapters.

Every adapter ``i`` gets a scalar importance weight ``omega_i``:
``W^i = W_0^i + omega_i A^i B^i``. During a short selection phase ``omega``
is updated by a stochastic gradient step followed by the hard threshold
``H_K`` that keeps the ``K`` entries of largest magnitude and zeroes the
rest. At the end of the phase only the ``K`` surviving adapters are kept;
``omega_i`` is folded into ``B^i`` and the survivors are trained further as
ordinary LoRA. Removed adapters are deleted from the model, so the saved
output is a standard PEFT adapter with fewer target modules.

Usage mirrors :class:`brain_opt.simplex_lora.SimplexLoRA`::

    selector = WeightLoRA(model, WeightLoRAConfig(total_steps=300, keep=16))
    for batch in batches:
        selector.zero_grad()
        model(**batch).loss.backward()
        selector.step()
    selector.save_pretrained(output_dir)
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn


def hard_threshold(values: torch.Tensor, keep: int) -> torch.Tensor:
    """``H_K``: keep the ``keep`` entries of largest magnitude, zero the rest."""
    out = torch.zeros_like(values)
    idx = values.abs().topk(keep).indices
    out[idx] = values[idx]
    return out


@dataclass(frozen=True)
class WeightLoRAConfig:
    total_steps: int = 1000
    keep: int = 8
    selection_steps: int = 50
    lr: float = 1e-4
    omega_lr: float = 1.0

    def __post_init__(self) -> None:
        for name in ("total_steps", "keep", "selection_steps"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.selection_steps >= self.total_steps:
            raise ValueError("total_steps must include training after adapter selection")
        for name in ("lr", "omega_lr"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


class WeightLoRA:
    """Owns the optimizer so removed adapters cannot leave stale optimizer state."""

    def __init__(self, model: nn.Module, config: WeightLoRAConfig):
        from peft import PeftModel

        if not isinstance(model, PeftModel) or len(model.peft_config) != 1:
            raise ValueError("expected a PeftModel with exactly one LoRA adapter")
        self.adapter = next(iter(model.peft_config))
        self.layers = {
            name: layer for name, layer in model.named_modules()
            if hasattr(layer, "lora_A") and self.adapter in layer.lora_A
        }
        if not self.layers:
            raise ValueError("no linear LoRA adapters found")
        if config.keep > len(self.layers):
            raise ValueError(f"keep={config.keep} exceeds the number of adapters ({len(self.layers)})")
        parameters = self._adapter_parameters()
        self.model = model
        self.config = config
        self.steps_done = 0
        self.initial_parameter_count = sum(p.numel() for p in parameters)
        self.omega = nn.Parameter(torch.ones(len(self.layers), device=parameters[0].device))
        self.optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=0.0)
        self._hooks: Dict[str, Any] = {}
        for i, (name, layer) in enumerate(self.layers.items()):
            self._hooks[name] = layer.lora_B[self.adapter].register_forward_hook(self._gate(i))
        self.history: List[Dict[str, Any]] = [self.snapshot()]

    @property
    def selecting(self) -> bool:
        return self.steps_done < self.config.selection_steps

    def _gate(self, index: int):
        def gate(module: nn.Module, inputs: Any, output: torch.Tensor) -> torch.Tensor:
            return output * self.omega[index].to(output.dtype)
        return gate

    def _adapter_parameters(self) -> List[nn.Parameter]:
        return [matrix[self.adapter].weight for layer in self.layers.values()
                for matrix in (layer.lora_A, layer.lora_B) if self.adapter in matrix]

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        self.omega.grad = None

    @torch.no_grad()
    def step(self) -> Dict[str, Any]:
        """Call once after backward (or completed gradient accumulation)."""
        if self.steps_done >= self.config.total_steps:
            raise RuntimeError("configured training budget is exhausted")
        gradients = [p.grad for p in self._adapter_parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError("missing or non-finite gradient; no optimizer step applied")
        if self.selecting and self.omega.grad is not None:
            self.omega.copy_(hard_threshold(self.omega - self.config.omega_lr * self.omega.grad,
                                            self.config.keep))
        self.optimizer.step()
        self.steps_done += 1
        selected = self.steps_done == self.config.selection_steps
        if selected:
            self._select()
        return {"step": self.steps_done, "adapter_lr": self.config.lr, "selection_done": selected}

    @torch.no_grad()
    def _select(self) -> None:
        survivors = set(self.omega.abs().topk(self.config.keep).indices.tolist())
        for i, (name, layer) in enumerate(self.layers.items()):
            self._hooks.pop(name).remove()
            for matrix in (layer.lora_A, layer.lora_B):
                self.optimizer.state.pop(matrix[self.adapter].weight, None)
            if i in survivors:
                layer.lora_B[self.adapter].weight.mul_(self.omega[i].to(layer.lora_B[self.adapter].weight.dtype))
            else:
                for matrix in (layer.lora_A, layer.lora_B):
                    matrix[self.adapter].weight.requires_grad_(False)
                    del matrix[self.adapter]
                layer.r[self.adapter] = 0
                layer.scaling[self.adapter] = 0.0
        self.layers = {name: layer for i, (name, layer) in enumerate(self.layers.items()) if i in survivors}
        self.optimizer.param_groups[0]["params"] = self._adapter_parameters()
        row = self.snapshot()
        row["selected_omega"] = self.omega.tolist()
        self.omega.fill_(1.0)
        self.omega.requires_grad_(False)
        self.omega.grad = None
        prefix = "base_model.model."
        config = self.model.peft_config[self.adapter]
        config.target_modules = {name[len(prefix):] if name.startswith(prefix) else name
                                 for name in self.layers}
        self.history.append(row)

    def snapshot(self) -> Dict[str, Any]:
        count = sum(p.numel() for p in self._adapter_parameters())
        return {
            "step": self.steps_done,
            "phase": "adapter_selection" if self.selecting else "lora_training",
            "adapters": sorted(self.layers),
            "omega": self.omega.detach().tolist(),
            "adapter_parameters": count,
            "initial_adapter_parameters": self.initial_parameter_count,
        }

    def save_pretrained(self, output_dir: str) -> None:
        """Save a normal PEFT adapter plus the selection evidence after training."""
        if self.steps_done != self.config.total_steps:
            raise RuntimeError("finish configured training before exporting")
        output = Path(output_dir)
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("refusing to overwrite nonempty adapter output")
        self.model.save_pretrained(str(output))
        evidence = {"method": "WeightLoRA", "implementation": "paper-hard-threshold-v1",
                    "config": asdict(self.config), "history": self.history, "final": self.snapshot(),
                    "format": "PEFT adapter; base checkpoint required"}
        (output / "weightlora_selection.json").write_text(
            json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
