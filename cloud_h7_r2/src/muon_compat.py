"""PyTorch 2.10 Muon algorithm backported for the Cloud PyTorch 2.1 image."""

from __future__ import annotations

import math

import torch


EPS = 1e-7
DEFAULT_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def _zeropower_via_newtonschulz(
    gradient: torch.Tensor,
    coefficients: tuple[float, float, float],
    steps: int,
    eps: float,
) -> torch.Tensor:
    if steps >= 100:
        raise ValueError("Number of steps must be less than 100")
    if gradient.ndim != 2:
        raise ValueError("Muon only supports 2D gradients")
    if len(coefficients) != 3:
        raise ValueError("Newton-Schulz coefficients must have length 3")
    a, b, c = coefficients
    update = gradient.bfloat16()
    transpose = update.shape[0] > update.shape[1]
    if transpose:
        update = update.T
    update.div_(update.norm().clamp(min=eps))
    for _ in range(steps):
        gram = update @ update.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        update = torch.addmm(update, gram_update, update, beta=a)
    return update.T if transpose else update


def _adjust_lr(lr: float, mode: str | None, shape: torch.Size) -> float:
    rows, columns = shape[:2]
    if mode is None or mode == "original":
        ratio = math.sqrt(max(1.0, rows / columns))
    elif mode == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(max(rows, columns))
    else:
        raise ValueError(f"unsupported Muon learning-rate adjustment: {mode}")
    return lr * ratio


class Muon(torch.optim.Optimizer):
    """Drop-in subset of ``torch.optim.Muon`` used by the H7 campaign."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = DEFAULT_COEFFICIENTS,
        eps: float = EPS,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = None,
    ) -> None:
        if lr < 0.0 or momentum < 0.0 or weight_decay < 0.0:
            raise ValueError("lr, momentum, and weight_decay must be non-negative")
        if adjust_lr_fn not in (None, "original", "match_rms_adamw"):
            raise ValueError(f"unsupported Muon learning-rate adjustment: {adjust_lr_fn}")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.ndim != 2:
                    raise ValueError("Muon only supports 2D parameters")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse or torch.is_complex(parameter):
                    raise RuntimeError("Muon requires dense, real gradients")
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(
                        gradient, memory_format=torch.preserve_format
                    )
                buffer = state["momentum_buffer"]
                buffer.lerp_(gradient, 1.0 - group["momentum"])
                update = (
                    gradient.lerp(buffer, group["momentum"])
                    if group["nesterov"]
                    else buffer
                )
                update = _zeropower_via_newtonschulz(
                    update,
                    group["ns_coefficients"],
                    group["ns_steps"],
                    group["eps"],
                )
                adjusted_lr = _adjust_lr(
                    group["lr"], group["adjust_lr_fn"], parameter.shape
                )
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-adjusted_lr)
        return loss
