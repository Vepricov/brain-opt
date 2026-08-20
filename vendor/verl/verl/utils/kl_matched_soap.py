# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Causal, per-update Fisher-KL matched SOAP for the FSDP PPO actor.

The optimizer owns both live SOAP state and a shadow AdamW state. ``propose``
computes the two hypothetical updates from one already-clipped gradient without
changing parameters or optimizer state. Only ``commit`` advances either state.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, cast

import torch
from torch import Tensor
from torch.optim import Optimizer


def partition_actor_parameters(
    named_parameters: Iterable[tuple[str, Tensor]],
) -> tuple[list[tuple[str, Tensor]], list[tuple[str, Tensor]]]:
    """Split actor parameters into SOAP matrices and AdamW auxiliaries."""
    soap, auxiliary = [], []
    for name, parameter in named_parameters:
        path = f".{name}."
        is_actor_matrix = parameter.ndim == 2 and (".self_attn." in path or ".mlp." in path)
        (soap if is_actor_matrix else auxiliary).append((name, parameter))
    return soap, auxiliary


def categorical_fisher_quadratic(logits: Tensor, logits_jvp: Tensor, mask: Tensor) -> Tensor:
    """Return ``0.5 d^T F d`` from exact categorical full-vocabulary logits JVPs."""
    if logits.shape != logits_jvp.shape or logits.ndim != 3:
        raise ValueError("logits and logits_jvp must have matching [batch, sequence, vocabulary] shapes")
    if mask.shape != logits.shape[:-1]:
        raise ValueError("Fisher mask must match logits without the vocabulary dimension")
    selected_logits = logits.float()[mask]
    selected_jvp = logits_jvp.float()[mask]
    if selected_logits.numel() == 0:
        raise ValueError("exact Fisher quadratic requires at least one teacher-forced state")
    probabilities = selected_logits.softmax(dim=-1)
    mean = (probabilities * selected_jvp).sum(dim=-1)
    variance = (probabilities * selected_jvp.square()).sum(dim=-1) - mean.square()
    value = 0.5 * variance.clamp_min(0).mean()
    if not torch.isfinite(value):
        raise FloatingPointError("exact Fisher quadratic is non-finite")
    return value


def exact_logits_jvp(
    module: torch.nn.Module,
    parameter_names: Sequence[str],
    primals: Sequence[Tensor],
    tangents: Sequence[Tensor],
    model_kwargs: Mapping[str, Any],
) -> tuple[Tensor, Tensor]:
    """Evaluate a model and its exact forward-mode logits JVP functionally."""
    from torch.func import functional_call, jvp

    if not (len(parameter_names) == len(primals) == len(tangents)) or not primals:
        raise ValueError("parameter names, primals, and tangents must be non-empty and aligned")

    def logits_function(*values):
        replacements = dict(zip(parameter_names, values, strict=True))
        output = functional_call(module, replacements, (), dict(model_kwargs), strict=False)
        return output.logits if hasattr(output, "logits") else output

    return cast(tuple[Tensor, Tensor], jvp(logits_function, tuple(primals), tuple(tangents)))


def matched_alpha(
    q_adamw: float,
    q_soap: float,
    *,
    minimum: float,
    maximum: float,
    clamp: bool,
) -> tuple[float, float]:
    """Calculate and validate ``sqrt(q_adamw / q_soap)``."""
    if not (0 < minimum <= maximum and math.isfinite(minimum) and math.isfinite(maximum)):
        raise ValueError("alpha bounds must be finite, positive, and ordered")
    if not (math.isfinite(q_adamw) and math.isfinite(q_soap) and q_adamw > 0 and q_soap > 0):
        raise FloatingPointError(f"Fisher quadratics must be finite and positive, got q_A={q_adamw}, q_S={q_soap}")
    raw = math.sqrt(q_adamw / q_soap)
    if not math.isfinite(raw) or raw <= 0:
        raise FloatingPointError(f"matched alpha must be finite and positive, got {raw}")
    if clamp:
        return min(max(raw, minimum), maximum), raw
    if not minimum <= raw <= maximum:
        raise FloatingPointError(f"matched alpha {raw} is outside [{minimum}, {maximum}] with clamping disabled")
    return raw, raw


@dataclass(frozen=True)
class UpdateProposal:
    generation: int
    adamw_directions: Mapping[Tensor, Tensor]
    soap_directions: Mapping[Tensor, Tensor]
    auxiliary_directions: Mapping[Tensor, Tensor]
    next_states: Mapping[Tensor, Mapping[str, Any]]


def _clone_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if isinstance(value, Tensor) else copy.deepcopy(value) for key, value in state.items()}


def _eigenvectors(accumulator: Tensor) -> Tensor:
    accumulator = accumulator.float()
    eye = torch.eye(accumulator.shape[0], device=accumulator.device, dtype=accumulator.dtype)
    _, vectors = torch.linalg.eigh(accumulator + 1e-30 * eye)
    return vectors.flip(1)


def _project(matrix: Tensor, left: Tensor | None, right: Tensor | None) -> Tensor:
    result = matrix
    if left is not None:
        result = left.T @ result
    if right is not None:
        result = result @ right
    return result


def _project_back(matrix: Tensor, left: Tensor | None, right: Tensor | None) -> Tensor:
    result = matrix
    if left is not None:
        result = left @ result
    if right is not None:
        result = result @ right.T
    return result


class KLMatchedSOAP(Optimizer):
    """SOAP actor matrices with per-update exact-Fisher AdamW matching."""

    requires_named_parameters = True
    requires_kl_matched_fisher = True
    route_label = "causal_per_update_kl_matched_soap"

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, Tensor]],
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        soap_betas: tuple[float, float] = (0.95, 0.95),
        soap_eps: float = 1e-8,
        soap_shampoo_beta: float = -1.0,
        soap_precondition_frequency: int = 10,
        soap_max_precond_dim: int = 2048,
        auxiliary_eps: float | None = None,
        alpha_min: float = 0.05,
        alpha_max: float = 20.0,
        alpha_clamp: bool = True,
        fisher_dataset_path: str | None = None,
        fisher_prompt_indices: Sequence[int] = tuple(range(16)),
        fisher_micro_batch_size: int = 1,
    ) -> None:
        named_parameters = list(named_parameters)
        soap_named, auxiliary_named = partition_actor_parameters(named_parameters)
        all_names = [name for name, _ in named_parameters]
        routed_names = [name for name, _ in soap_named + auxiliary_named]
        if not soap_named or not auxiliary_named:
            raise ValueError("KLMatchedSOAP requires non-empty SOAP matrix and AdamW auxiliary routes")
        if len(routed_names) != len(set(routed_names)) or set(routed_names) != set(all_names):
            raise ValueError("actor parameter ownership must be disjoint and exhaustive")
        if len(tuple(fisher_prompt_indices)) != 16 or len(set(fisher_prompt_indices)) != 16:
            raise ValueError("exactly 16 distinct Fisher prompt indices are required")
        if soap_precondition_frequency < 1 or soap_max_precond_dim < 1 or fisher_micro_batch_size < 1:
            raise ValueError("SOAP frequency, maximum dimension, and Fisher micro batch size must be positive")
        matched_alpha(1.0, 1.0, minimum=alpha_min, maximum=alpha_max, clamp=alpha_clamp)

        defaults = {"lr": lr, "weight_decay": weight_decay}
        super().__init__(
            [
                {
                    "params": [parameter for _, parameter in soap_named],
                    "route": "soap_matrix",
                    "parameter_names": tuple(name for name, _ in soap_named),
                    "lr": lr,
                    "weight_decay": weight_decay,
                },
                {
                    "params": [parameter for _, parameter in auxiliary_named],
                    "route": "auxiliary_adamw",
                    "parameter_names": tuple(name for name, _ in auxiliary_named),
                    "lr": lr,
                    "weight_decay": weight_decay,
                },
            ],
            defaults,
        )
        self.parameter_routes = {
            "soap_matrix": tuple(name for name, _ in soap_named),
            "auxiliary_adamw": tuple(name for name, _ in auxiliary_named),
        }
        self.adamw_betas = tuple(betas)
        self.adamw_eps = float(eps)
        self.auxiliary_eps = float(auxiliary_eps if auxiliary_eps is not None else eps)
        self.soap_betas = tuple(soap_betas)
        self.soap_eps = float(soap_eps)
        self.shampoo_beta = float(soap_shampoo_beta if soap_shampoo_beta >= 0 else soap_betas[1])
        self.precondition_frequency = int(soap_precondition_frequency)
        self.max_precond_dim = int(soap_max_precond_dim)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.alpha_clamp = bool(alpha_clamp)
        self.fisher_dataset_path = fisher_dataset_path
        self.fisher_prompt_indices = tuple(int(index) for index in fisher_prompt_indices)
        self.fisher_micro_batch_size = int(fisher_micro_batch_size)
        self._fisher_evaluator: Callable[[Mapping[Tensor, Tensor]], float] | None = None
        self._prompt_identity: Mapping[str, Any] | None = None
        self._update_generation = 0
        self.latest_telemetry: dict[str, float] = {}

    def bind_fisher_evaluator(
        self,
        evaluator: Callable[[Mapping[Tensor, Tensor]], float],
        prompt_identity: Mapping[str, Any],
    ) -> None:
        if self._prompt_identity is not None and dict(self._prompt_identity) != dict(prompt_identity):
            raise RuntimeError("pinned Fisher prompt identity changed after optimizer binding")
        self._fisher_evaluator = evaluator
        self._prompt_identity = dict(prompt_identity)

    def _adamw_proposal(self, parameter: Tensor, gradient: Tensor, state: Mapping[str, Any], eps: float):
        next_state = _clone_state(state)
        step = int(next_state.get("adamw_step", 0)) + 1
        first, second = self.adamw_betas
        exp_avg = next_state.get("adamw_exp_avg", torch.zeros_like(gradient)).mul(first).add(gradient, alpha=1 - first)
        exp_avg_sq = (
            next_state.get("adamw_exp_avg_sq", torch.zeros_like(gradient))
            .mul(second)
            .addcmul(gradient, gradient, value=1 - second)
        )
        normalized = exp_avg / (1 - first**step)
        denominator = (exp_avg_sq / (1 - second**step)).sqrt().add(eps)
        direction = -self._group_for(parameter)["lr"] * normalized / denominator
        group = self._group_for(parameter)
        direction = direction - group["lr"] * group["weight_decay"] * parameter
        next_state.update(adamw_step=step, adamw_exp_avg=exp_avg, adamw_exp_avg_sq=exp_avg_sq)
        return direction, next_state

    def _soap_proposal(self, parameter: Tensor, gradient: Tensor, state: Mapping[str, Any]):
        next_state = _clone_state(state)
        work_gradient = gradient.float()
        step = int(next_state.get("soap_step", 0)) + 1
        rows, columns = work_gradient.shape
        gg_left = next_state.get("soap_gg_left")
        gg_right = next_state.get("soap_gg_right")
        if rows <= self.max_precond_dim:
            if gg_left is None:
                gg_left = torch.zeros(rows, rows, device=gradient.device, dtype=torch.float32)
            gg_left = gg_left.mul(self.shampoo_beta).add(work_gradient @ work_gradient.T, alpha=1 - self.shampoo_beta)
        if columns <= self.max_precond_dim:
            if gg_right is None:
                gg_right = torch.zeros(columns, columns, device=gradient.device, dtype=torch.float32)
            gg_right = gg_right.mul(self.shampoo_beta).add(work_gradient.T @ work_gradient, alpha=1 - self.shampoo_beta)

        left = next_state.get("soap_q_left")
        right = next_state.get("soap_q_right")
        if left is None and gg_left is not None:
            left = _eigenvectors(gg_left)
        if right is None and gg_right is not None:
            right = _eigenvectors(gg_right)
        projected = _project(work_gradient, left, right)
        first, second = self.soap_betas
        exp_avg = (
            next_state.get("soap_exp_avg", torch.zeros_like(work_gradient))
            .mul(first)
            .add(work_gradient, alpha=1 - first)
        )
        exp_avg_sq = (
            next_state.get("soap_exp_avg_sq", torch.zeros_like(projected))
            .mul(second)
            .addcmul(projected, projected, value=1 - second)
        )
        normalized = _project(exp_avg, left, right) / (1 - first**step)
        denominator = (exp_avg_sq / (1 - second**step)).sqrt().add(self.soap_eps)
        preconditioned = _project_back(normalized / denominator, left, right)
        group = self._group_for(parameter)
        direction = -group["lr"] * preconditioned - group["lr"] * group["weight_decay"] * parameter.float()

        # The current direction uses the prior basis (or its deterministic first-step
        # initialization). A scheduled refresh is stored for the next update only.
        if step % self.precondition_frequency == 0:
            old_left, old_right = left, right
            if gg_left is not None:
                left = _eigenvectors(gg_left)
            if gg_right is not None:
                right = _eigenvectors(gg_right)
            # Preserve the diagonal second-moment approximation across a basis
            # change instead of reinterpreting old coordinates in the new basis.
            if old_left is not None and left is not None:
                overlap = (old_left.T @ left).square()
                exp_avg_sq = overlap.T @ exp_avg_sq
            if old_right is not None and right is not None:
                overlap = (old_right.T @ right).square()
                exp_avg_sq = exp_avg_sq @ overlap
        next_state.update(
            soap_step=step,
            soap_exp_avg=exp_avg,
            soap_exp_avg_sq=exp_avg_sq,
        )
        if gg_left is not None:
            next_state["soap_gg_left"] = gg_left
            next_state["soap_q_left"] = left
        if gg_right is not None:
            next_state["soap_gg_right"] = gg_right
            next_state["soap_q_right"] = right
        return direction.to(parameter.dtype), next_state

    def _group_for(self, parameter: Tensor) -> dict[str, Any]:
        for group in self.param_groups:
            if any(candidate is parameter for candidate in group["params"]):
                return group
        raise KeyError("parameter is not owned by optimizer")

    @torch.no_grad()
    def propose(self) -> UpdateProposal:
        adamw, soap, auxiliary, next_states = {}, {}, {}, {}
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("KLMatchedSOAP does not support sparse gradients")
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("KLMatchedSOAP proposal received a non-finite gradient")
                if group["route"] == "soap_matrix":
                    live_state = self.state.get(parameter, {})
                    adam_direction, candidate = self._adamw_proposal(parameter, gradient, live_state, self.adamw_eps)
                    soap_direction, candidate = self._soap_proposal(parameter, gradient, candidate)
                    adamw[parameter] = adam_direction.detach()
                    soap[parameter] = soap_direction.detach()
                    next_states[parameter] = candidate
                else:
                    direction, candidate = self._adamw_proposal(
                        parameter, gradient, self.state.get(parameter, {}), self.auxiliary_eps
                    )
                    auxiliary[parameter] = direction.detach()
                    next_states[parameter] = candidate
        if not adamw or set(adamw) != set(soap):
            raise RuntimeError("SOAP-owned actor matrices must have both AdamW and SOAP proposals")
        return UpdateProposal(self._update_generation, adamw, soap, auxiliary, next_states)

    @torch.no_grad()
    def commit(self, proposal: UpdateProposal, alpha: float) -> None:
        if proposal.generation != self._update_generation:
            raise RuntimeError("stale or already-committed KLMatchedSOAP proposal")
        if not math.isfinite(alpha) or alpha <= 0:
            raise FloatingPointError("committed alpha must be finite and positive")
        for parameter, direction in proposal.soap_directions.items():
            parameter.add_(direction, alpha=alpha)
        for parameter, direction in proposal.auxiliary_directions.items():
            parameter.add_(direction)
        for parameter, candidate in proposal.next_states.items():
            self.state[parameter].clear()
            self.state[parameter].update(candidate)
        self._update_generation += 1

    def step(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, closure: Callable[[], float] | None = None
    ) -> float | None:
        if closure is not None:
            raise ValueError("KLMatchedSOAP does not support optimizer closures")
        if self._fisher_evaluator is None or self._prompt_identity is None:
            raise RuntimeError("FSDPEngine did not bind the exact Fisher evaluator")
        proposal = self.propose()
        with torch.enable_grad():
            q_adamw = float(self._fisher_evaluator(proposal.adamw_directions))
            q_soap = float(self._fisher_evaluator(proposal.soap_directions))
        alpha, raw_alpha = matched_alpha(
            q_adamw,
            q_soap,
            minimum=self.alpha_min,
            maximum=self.alpha_max,
            clamp=self.alpha_clamp,
        )
        self.commit(proposal, alpha)
        self.latest_telemetry = {
            "actor/kl_matched/q_adamw": q_adamw,
            "actor/kl_matched/q_soap": q_soap,
            "actor/kl_matched/alpha_raw": raw_alpha,
            "actor/kl_matched/alpha": alpha,
            "actor/kl_matched/alpha_clamped": float(alpha != raw_alpha),
            "actor/kl_matched/update": float(self._update_generation),
        }
        return None

    def state_dict(self):
        result = super().state_dict()
        result["kl_matched_soap"] = {
            "version": 1,
            "update_generation": self._update_generation,
            "prompt_identity": copy.deepcopy(self._prompt_identity),
            "latest_telemetry": dict(self.latest_telemetry),
            "configuration": self._checkpoint_configuration(),
        }
        return result

    def _checkpoint_configuration(self) -> dict[str, Any]:
        return {
            "adamw_betas": self.adamw_betas,
            "adamw_eps": self.adamw_eps,
            "auxiliary_eps": self.auxiliary_eps,
            "soap_betas": self.soap_betas,
            "soap_eps": self.soap_eps,
            "shampoo_beta": self.shampoo_beta,
            "precondition_frequency": self.precondition_frequency,
            "max_precond_dim": self.max_precond_dim,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "alpha_clamp": self.alpha_clamp,
            "fisher_prompt_indices": self.fisher_prompt_indices,
            "fisher_micro_batch_size": self.fisher_micro_batch_size,
        }

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        metadata = state_dict.pop("kl_matched_soap", None)
        if metadata is None:
            raise RuntimeError("KLMatchedSOAP checkpoint is missing causal proposal metadata")
        restored_identity = metadata.get("prompt_identity")
        if self._prompt_identity is None or restored_identity != self._prompt_identity:
            raise RuntimeError("checkpoint pinned Fisher prompt identity does not match this run")
        if metadata.get("configuration") != self._checkpoint_configuration():
            raise RuntimeError("checkpoint KLMatchedSOAP configuration does not match this run")
        super().load_state_dict(state_dict)
        self._update_generation = int(metadata["update_generation"])
        self.latest_telemetry = dict(metadata.get("latest_telemetry", {}))


def build_teacher_forced_gsm8k(
    dataset_path: str | Path,
    tokenizer: Any,
    prompt_indices: Sequence[int],
) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
    """Tokenize and fingerprint the pinned GSM8K teacher-forced prompt set."""
    import pandas as pd  # pyright: ignore[reportMissingImports]

    frame = pd.read_parquet(dataset_path)
    if min(prompt_indices) < 0 or max(prompt_indices) >= len(frame):
        raise RuntimeError("pinned Fisher prompt indices are outside the GSM8K parquet")
    eos = tokenizer.eos_token_id
    if eos is None:
        raise RuntimeError("Fisher tokenizer must define eos_token_id")
    examples = []
    for index in prompt_indices:
        row = frame.iloc[index].to_dict()
        prompt = row.get("prompt")
        reward_model = row.get("reward_model")
        answer = reward_model.get("ground_truth") if isinstance(reward_model, Mapping) else None
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if prompt is None or not isinstance(answer, str) or not answer:
            raise RuntimeError(f"GSM8K row {index} lacks prompt or reward_model.ground_truth")
        prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False) + [eos]
        examples.append((list(prompt_ids) + answer_ids, len(prompt_ids)))
    maximum = max(len(ids) for ids, _ in examples)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    input_ids = torch.full((len(examples), maximum), pad, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    fisher_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, (ids, prompt_length) in enumerate(examples):
        length = len(ids)
        input_ids[row, :length] = torch.tensor(ids)
        attention_mask[row, :length] = 1
        fisher_mask[row, prompt_length - 1 : length - 1] = True
    digest_payload = {
        "indices": list(prompt_indices),
        "input_ids": input_ids.tolist(),
        "attention_mask": attention_mask.tolist(),
        "fisher_mask": fisher_mask.tolist(),
    }
    identity = {
        "policy": "fixed_gsm8k_teacher_forced_v1",
        "indices": list(prompt_indices),
        "count": len(prompt_indices),
        "sha256": hashlib.sha256(json.dumps(digest_payload, sort_keys=True).encode()).hexdigest(),
    }
    return input_ids, attention_mask, fisher_mask, identity


class ExactLogitsJVPFisher:
    """Exact full-vocabulary logits-JVP Fisher evaluator for one-rank FSDP."""

    def __init__(
        self,
        fsdp_module: torch.nn.Module,
        input_ids: Tensor,
        attention_mask: Tensor,
        fisher_mask: Tensor,
        micro_batch_size: int = 1,
    ) -> None:
        self.fsdp_module = fsdp_module
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.fisher_mask = fisher_mask
        self.micro_batch_size = micro_batch_size

    def __call__(self, directions: Mapping[Tensor, Tensor]) -> float:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if not isinstance(self.fsdp_module, FSDP):
            raise RuntimeError("exact Fisher integration requires legacy FSDP")
        if torch.distributed.get_world_size() != 1:
            raise RuntimeError("exact Fisher logits-JVP currently requires one-rank FSDP")
        module = self.fsdp_module._fsdp_wrapped_module
        name_by_id = {id(parameter): name for name, parameter in module.named_parameters()}
        try:
            names = tuple(name_by_id[id(parameter)] for parameter in directions)
        except KeyError as error:
            raise RuntimeError("SOAP direction cannot be mapped to an unwrapped FSDP parameter") from error
        primals = tuple(parameter for parameter in directions)
        tangents = tuple(directions[parameter] for parameter in directions)
        total_weighted = torch.zeros((), device=primals[0].device, dtype=torch.float64)
        total_states = 0
        was_training = module.training
        module.eval()
        try:
            with FSDP.summon_full_params(self.fsdp_module, recurse=True, writeback=False):
                for start in range(0, self.input_ids.shape[0], self.micro_batch_size):
                    stop = start + self.micro_batch_size
                    ids = self.input_ids[start:stop].to(primals[0].device)
                    attention = self.attention_mask[start:stop].to(primals[0].device)
                    mask = self.fisher_mask[start:stop].to(primals[0].device)

                    logits, logits_tangent = exact_logits_jvp(
                        module,
                        names,
                        primals,
                        tangents,
                        {"input_ids": ids, "attention_mask": attention, "use_cache": False},
                    )
                    states = int(mask.sum().item())
                    quadratic = categorical_fisher_quadratic(logits, logits_tangent, mask)
                    total_weighted += quadratic.double() * states
                    total_states += states
        finally:
            module.train(was_training)
        if total_states == 0:
            raise RuntimeError("pinned Fisher set contains no teacher-forced states")
        result = float((total_weighted / total_states).item())
        if not math.isfinite(result):
            raise FloatingPointError("aggregated exact Fisher quadratic is non-finite")
        return result
