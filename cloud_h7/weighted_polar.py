"""One-step diagonal weighted-polar geometry for the isolated H7.3 pilot."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

import torch


def canonical_matrix_name(name: str) -> str:
    for prefix in ("model.", "backbone."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


class DiagonalFactorCollector:
    """Collect diagonal input/output second moments without leaving the device."""

    def __init__(self, model: torch.nn.Module, parameter_names: set[str]) -> None:
        self.model = model
        self.parameter_names = set(parameter_names)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self.input_sums: dict[str, torch.Tensor] = {}
        self.output_sums: dict[str, torch.Tensor] = {}
        self.input_counts: dict[str, int] = {}
        self.output_counts: dict[str, int] = {}

    @staticmethod
    def _accumulate(
        sums: dict[str, torch.Tensor],
        counts: dict[str, int],
        name: str,
        value: torch.Tensor,
    ) -> None:
        rows = value.detach().float().reshape(-1, value.shape[-1])
        row_sum = rows.square().sum(dim=0)
        sums[name] = sums.get(name, torch.zeros_like(row_sum)) + row_sum
        counts[name] = counts.get(name, 0) + rows.shape[0]

    def __enter__(self) -> "DiagonalFactorCollector":
        modules = dict(self.model.named_modules())
        for parameter_name in sorted(self.parameter_names):
            if not parameter_name.endswith(".weight") and parameter_name != "weight":
                raise ValueError(f"expected a weight parameter: {parameter_name}")
            module_name = (
                parameter_name[: -len(".weight")] if parameter_name != "weight" else ""
            )
            module = modules[module_name]
            key = canonical_matrix_name(parameter_name)

            def forward_hook(_module, inputs, _output, *, factor_key=key):
                self._accumulate(
                    self.input_sums, self.input_counts, factor_key, inputs[0]
                )

            def backward_hook(_module, _grad_input, grad_output, *, factor_key=key):
                self._accumulate(
                    self.output_sums, self.output_counts, factor_key, grad_output[0]
                )

            self.handles.append(module.register_forward_hook(forward_hook))
            self.handles.append(module.register_full_backward_hook(backward_hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def factors(self) -> dict[str, dict[str, torch.Tensor]]:
        expected = {canonical_matrix_name(name) for name in self.parameter_names}
        complete = set(self.input_sums) & set(self.output_sums)
        if complete != expected:
            missing = sorted(expected - complete)
            raise RuntimeError(
                f"diagonal factors require completed backward; missing: {missing}"
            )
        return {
            name: {
                "right": self.input_sums[name] / self.input_counts[name],
                "left": self.output_sums[name] / self.output_counts[name],
            }
            for name in sorted(expected)
        }


def _inverse_sqrt_factor(moment: torch.Tensor, damping: float) -> torch.Tensor:
    if damping < 0.0:
        raise ValueError("damping must be non-negative")
    value = moment.detach().float()
    normalized = value / value.mean().clamp_min(torch.finfo(value.dtype).tiny)
    return (normalized + damping).rsqrt()


def polar_direction(
    gradients: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return negative Muon polar directions with legacy shape scaling removed."""
    directions: dict[str, torch.Tensor] = {}
    for name, gradient in gradients.items():
        dummy = torch.nn.Parameter(torch.zeros_like(gradient))
        optimizer = torch.optim.Muon(
            [dummy],
            lr=1.0,
            momentum=0.0,
            weight_decay=0.0,
            adjust_lr_fn=None,
        )
        dummy.grad = gradient.detach().to(dummy.device)
        optimizer.step()
        legacy_shape_scale = math.sqrt(max(1.0, gradient.shape[0] / gradient.shape[1]))
        directions[name] = dummy.detach().float().clone() / legacy_shape_scale
    return directions


def raw_muon_direction(
    gradients: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return the production Muon one-step direction with no carried momentum."""
    directions: dict[str, torch.Tensor] = {}
    for name, gradient in gradients.items():
        dummy = torch.nn.Parameter(torch.zeros_like(gradient))
        optimizer = torch.optim.Muon(
            [dummy],
            lr=1.0,
            momentum=0.0,
            weight_decay=0.0,
            adjust_lr_fn="match_rms_adamw",
        )
        dummy.grad = gradient.detach().to(dummy.device)
        optimizer.step()
        directions[name] = dummy.detach().float().clone()
    return directions


def weighted_polar_direction(
    gradients: Mapping[str, torch.Tensor],
    factors: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    damping: float,
) -> dict[str, torch.Tensor]:
    transformed: dict[str, torch.Tensor] = {}
    inverse_factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, gradient in gradients.items():
        factor = factors[canonical_matrix_name(name)]
        left = _inverse_sqrt_factor(factor["left"], damping)
        right = _inverse_sqrt_factor(factor["right"], damping)
        transformed[name] = gradient.detach().float() * left[:, None] * right[None, :]
        inverse_factors[name] = (left, right)
    polar = polar_direction(transformed)
    return {
        name: polar[name] * inverse_factors[name][0][:, None] * inverse_factors[name][1][None, :]
        for name in polar
    }


def match_global_rms(
    updates: Mapping[str, torch.Tensor],
    *,
    target_rms: float,
    total_parameter_count: int,
) -> tuple[dict[str, torch.Tensor], float]:
    if target_rms <= 0.0:
        raise ValueError("target_rms must be positive")
    updated_count = sum(value.numel() for value in updates.values())
    if total_parameter_count < updated_count:
        raise ValueError("total_parameter_count cannot be smaller than updated tensors")
    square_sum = sum(float(value.detach().float().square().sum()) for value in updates.values())
    if not updates or square_sum == 0.0:
        raise ValueError("updates must contain a nonzero tensor")
    scale = target_rms * math.sqrt(total_parameter_count / square_sum)
    return {name: value.detach().float() * scale for name, value in updates.items()}, scale


def calibrate_scale_strict(
    evaluate: Callable[[float], float],
    *,
    budget: float,
    initial_max_scale: float = 4.0,
    max_scale: float = 16.0,
    tolerance: float = 0.02,
    bisection_steps: int = 16,
) -> tuple[float, float, dict[str, Any]]:
    """Bracket and bisect a functional budget, failing closed when unreachable."""
    if budget <= 0.0 or initial_max_scale <= 0.0 or max_scale < initial_max_scale:
        raise ValueError("budget and scale bounds must be positive and ordered")
    lower_scale = 0.0
    lower_value = float(evaluate(lower_scale))
    upper_scale = initial_max_scale
    upper_value = float(evaluate(upper_scale))
    trials = [{"scale": lower_scale, "value": lower_value}, {"scale": upper_scale, "value": upper_value}]
    while upper_value < budget and upper_scale < max_scale:
        lower_scale, lower_value = upper_scale, upper_value
        upper_scale = min(2.0 * upper_scale, max_scale)
        upper_value = float(evaluate(upper_scale))
        trials.append({"scale": upper_scale, "value": upper_value})
    if not all(math.isfinite(row["value"]) for row in trials):
        raise RuntimeError("functional budget evaluation is non-finite")
    if upper_value < budget:
        raise RuntimeError(
            f"functional budget not bracketed by scale {max_scale}: budget={budget} measured={upper_value}"
        )
    best_scale, best_value = min(
        ((row["scale"], row["value"]) for row in trials),
        key=lambda item: abs(item[1] / budget - 1.0),
    )
    for _ in range(bisection_steps):
        middle_scale = 0.5 * (lower_scale + upper_scale)
        middle_value = float(evaluate(middle_scale))
        if not math.isfinite(middle_value):
            raise RuntimeError("functional budget evaluation is non-finite")
        trials.append({"scale": middle_scale, "value": middle_value})
        if abs(middle_value / budget - 1.0) < abs(best_value / budget - 1.0):
            best_scale, best_value = middle_scale, middle_value
        if middle_value < budget:
            lower_scale, lower_value = middle_scale, middle_value
        else:
            upper_scale, upper_value = middle_scale, middle_value
    relative_error = abs(best_value / budget - 1.0)
    if relative_error > tolerance:
        raise RuntimeError(
            f"functional budget not matched within tolerance: budget={budget} measured={best_value}"
        )
    return best_scale, best_value, {
        "trials": trials,
        "max_scale": max_scale,
        "relative_budget_error": relative_error,
        "tolerance": tolerance,
    }


class HiddenUpdateCommitter:
    """One-shot guard for the only in-place hidden-parameter update."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        self.committed = False

    def commit(self, updates: Mapping[str, torch.Tensor], *, scale: float) -> float:
        if self.committed:
            raise RuntimeError("hidden update already committed")
        parameters = dict(self.model.named_parameters())
        missing = sorted(set(updates) - set(parameters))
        if missing:
            raise KeyError(f"unknown hidden parameters: {missing}")
        realized_square_sum = 0.0
        with torch.no_grad():
            for name, update in updates.items():
                before = parameters[name].detach().clone()
                parameters[name].add_(
                    update.to(device=parameters[name].device, dtype=parameters[name].dtype),
                    alpha=scale,
                )
                realized_square_sum += float(
                    (parameters[name].detach().float() - before.float()).square().sum()
                )
        self.committed = True
        return realized_square_sum

