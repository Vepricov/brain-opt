#!/usr/bin/env python3
"""Held-out shadow-update tests for role-conditioned Muon geometry."""
from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import torch

from actor_critic_geometry import _config_from_manifest
from llm_ppo import PpoRollout, TransformersBackend, clipped_policy_loss, partition_parameters
from role_geometry_ablation import _shaped_rewards, one_step_td_residuals


ACTOR_CANDIDATES = (
    "raw_average",
    "muon_batch1",
    "muon_average",
    "consensus_alpha0_top16",
)
CRITIC_CANDIDATES = (
    "raw_average",
    "muon_average",
    "power_alpha0_top16",
    "power_alpha05_top16",
    "power_alpha1_top16",
    "energy99_alpha0_top16",
)


def _top_svd(
    matrix: torch.Tensor, rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = matrix.detach().float()
    q = min(rank, min(value.shape))
    if q == min(value.shape):
        left, singular, right_h = torch.linalg.svd(value, full_matrices=False)
        return left, singular, right_h.mT
    devices = [value.device] if value.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        left, singular, right = torch.svd_lowrank(value, q=q, niter=4)
    order = singular.argsort(descending=True)
    return left[:, order], singular[order], right[:, order]


def match_update_rms(
    updates: Mapping[str, torch.Tensor],
    target_rms: float,
    *,
    total_parameter_count: int | None = None,
) -> tuple[dict[str, torch.Tensor], float]:
    if target_rms <= 0.0:
        raise ValueError("target_rms must be positive")
    square_sum = sum(float(value.float().square().sum()) for value in updates.values())
    updated_count = sum(value.numel() for value in updates.values())
    count = updated_count if total_parameter_count is None else total_parameter_count
    if count < updated_count:
        raise ValueError("total_parameter_count cannot be smaller than updated tensors")
    if count == 0 or square_sum == 0.0:
        raise ValueError("updates must contain a nonzero tensor")
    scale = target_rms * math.sqrt(count / square_sum)
    return {name: value.float() * scale for name, value in updates.items()}, scale


@contextmanager
def temporary_parameter_update(
    model: torch.nn.Module,
    updates: Mapping[str, torch.Tensor],
    *,
    originals: Mapping[str, torch.Tensor] | None = None,
):
    parameters = dict(model.named_parameters())
    missing = sorted(set(updates) - set(parameters))
    if missing:
        raise KeyError(f"unknown parameters: {missing}")
    saved = (
        {name: parameters[name].detach().cpu().clone() for name in updates}
        if originals is None
        else originals
    )
    try:
        with torch.no_grad():
            for name, update in updates.items():
                parameters[name].add_(
                    update.to(device=parameters[name].device, dtype=parameters[name].dtype)
                )
        yield
    finally:
        with torch.no_grad():
            for name in updates:
                parameters[name].copy_(
                    saved[name].to(
                        device=parameters[name].device,
                        dtype=parameters[name].dtype,
                    )
                )


def truncated_power_update(
    matrix: torch.Tensor,
    *,
    alpha: float,
    rank: int = 16,
    energy_threshold: float | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if rank <= 0:
        raise ValueError("rank must be positive")
    if energy_threshold is not None and not 0.0 < energy_threshold <= 1.0:
        raise ValueError("energy_threshold must be in (0, 1]")
    value = matrix.detach().float()
    q = min(rank, min(value.shape))
    left, singular, right = _top_svd(value, rank)
    retained = q
    threshold_reached = energy_threshold is None
    if energy_threshold is not None:
        total_energy = value.square().sum().clamp_min(torch.finfo(value.dtype).tiny)
        cumulative = singular.square().cumsum(0) / total_energy
        positions = (cumulative >= energy_threshold).nonzero(as_tuple=False)
        if positions.numel():
            retained = int(positions[0]) + 1
            threshold_reached = True
    update = -(
        left[:, :retained]
        * singular[:retained].clamp_min(torch.finfo(value.dtype).tiny).pow(alpha)
    ) @ right[:, :retained].mT
    return update, {
        "retained_rank": retained,
        "computed_rank": q,
        "energy_threshold": energy_threshold,
        "threshold_reached": threshold_reached,
        "computed_energy_fraction": float(
            singular.square().sum() / value.square().sum().clamp_min(1e-30)
        ),
    }


def consensus_power_update(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    alpha: float,
    rank: int = 16,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if first.shape != second.shape:
        raise ValueError("consensus gradients must have identical shapes")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    average = 0.5 * (first.detach().float() + second.detach().float())
    avg_left, avg_singular, avg_right = _top_svd(average, rank)
    first_left, _, first_right = _top_svd(first, rank)
    second_left, _, second_right = _top_svd(second, rank)
    left_first = (first_left.mT @ avg_left).square().sum(dim=0).clamp(0.0, 1.0)
    left_second = (second_left.mT @ avg_left).square().sum(dim=0).clamp(0.0, 1.0)
    right_first = (first_right.mT @ avg_right).square().sum(dim=0).clamp(0.0, 1.0)
    right_second = (second_right.mT @ avg_right).square().sum(dim=0).clamp(0.0, 1.0)
    gates = (left_first * left_second * right_first * right_second).clamp_min(0.0).pow(0.25)
    update = -(
        avg_left
        * gates
        * avg_singular.clamp_min(torch.finfo(average.dtype).tiny).pow(alpha)
    ) @ avg_right.mT
    return update, {
        "computed_rank": len(avg_singular),
        "mean_consensus_gate": float(gates.mean()),
        "minimum_consensus_gate": float(gates.min()),
        "maximum_consensus_gate": float(gates.max()),
    }


def collect_matrix_gradients(
    model: torch.nn.Module, loss: torch.Tensor,
) -> dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    loss.backward()
    gradients = {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in partition_parameters(model)["muon"]
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    return gradients


def _average_gradients(
    first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if set(first) != set(second):
        raise ValueError("gradient parameter sets do not match")
    return {name: 0.5 * (first[name] + second[name]) for name in first}


def _production_muon_updates(
    gradients: Mapping[str, torch.Tensor], device: torch.device,
) -> dict[str, torch.Tensor]:
    result = {}
    for name, gradient in gradients.items():
        dummy = torch.nn.Parameter(torch.zeros_like(gradient, device=device))
        optimizer = torch.optim.Muon(
            [dummy],
            lr=1.0,
            momentum=0.95,
            weight_decay=0.0,
            adjust_lr_fn="match_rms_adamw",
        )
        dummy.grad = gradient.to(device)
        optimizer.step()
        result[name] = dummy.detach().float().cpu().clone()
        del optimizer, dummy
    return result


def _spectral_updates(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    alpha: float,
    consensus: bool = False,
    energy_threshold: float | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    updates = {}
    metadata_rows = []
    for name in first:
        left = first[name].to(device)
        right = second[name].to(device)
        if consensus:
            update, metadata = consensus_power_update(
                left, right, alpha=alpha, rank=16,
            )
        else:
            update, metadata = truncated_power_update(
                0.5 * (left + right),
                alpha=alpha,
                rank=16,
                energy_threshold=energy_threshold,
            )
        updates[name] = update.cpu()
        metadata_rows.append(metadata)
        del left, right, update
    summary: dict[str, float] = {}
    for key in (
        "retained_rank",
        "computed_energy_fraction",
        "mean_consensus_gate",
        "minimum_consensus_gate",
        "maximum_consensus_gate",
    ):
        values = [float(row[key]) for row in metadata_rows if key in row]
        if values:
            summary[f"mean_{key}"] = sum(values) / len(values)
    if energy_threshold is not None:
        summary["threshold_reached_fraction"] = sum(
            bool(row.get("threshold_reached")) for row in metadata_rows
        ) / len(metadata_rows)
    return updates, summary


def _candidate_updates(
    role: str,
    candidate: str,
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    average = None
    if candidate == "raw_average":
        average = _average_gradients(first, second)
        return {name: -value for name, value in average.items()}, {}
    if candidate == "muon_batch1":
        if role != "actor":
            raise ValueError("muon_batch1 is actor-only")
        return _production_muon_updates(first, device), {}
    if candidate == "muon_average":
        average = _average_gradients(first, second)
        return _production_muon_updates(average, device), {}
    if candidate == "consensus_alpha0_top16":
        if role != "actor":
            raise ValueError("consensus candidate is actor-only")
        return _spectral_updates(
            first, second, device=device, alpha=0.0, consensus=True,
        )
    if candidate.startswith("power_alpha"):
        alpha = {
            "power_alpha0_top16": 0.0,
            "power_alpha05_top16": 0.5,
            "power_alpha1_top16": 1.0,
        }[candidate]
        return _spectral_updates(
            first, second, device=device, alpha=alpha,
        )
    if candidate == "energy99_alpha0_top16":
        return _spectral_updates(
            first,
            second,
            device=device,
            alpha=0.0,
            energy_threshold=0.99,
        )
    raise ValueError(f"unknown {role} candidate: {candidate}")


def _actor_response_log_distributions(
    backend: TransformersBackend, rollout: PpoRollout,
) -> torch.Tensor:
    logits = backend.actor(
        input_ids=rollout.sequences,
        attention_mask=rollout.attention,
    ).logits[:, :-1].float()
    return logits[:, rollout.prompt_width - 1 :].log_softmax(dim=-1)


def _actor_metrics(
    backend: TransformersBackend,
    rollout: PpoRollout,
    baseline_log_distributions: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        new_log_distributions = _actor_response_log_distributions(backend, rollout)
        sampled_logprobs = new_log_distributions.gather(
            -1,
            rollout.sequences[:, rollout.prompt_width :].unsqueeze(-1),
        ).squeeze(-1)
        loss, diagnostics = clipped_policy_loss(
            sampled_logprobs,
            rollout.old_logprobs,
            rollout.advantages,
            rollout.response_mask,
            backend.config.clip_coef,
        )
        token_kl = (
            baseline_log_distributions.exp()
            * (baseline_log_distributions - new_log_distributions)
        ).sum(dim=-1)
        categorical_kl = (
            token_kl * rollout.response_mask
        ).sum() / rollout.response_mask.sum().clamp_min(1.0)
    return {
        "policy_loss": float(loss),
        "categorical_kl": float(categorical_kl),
        "sampled_approx_kl": diagnostics["approx_kl"],
        "clip_fraction": diagnostics["clip_fraction"],
        "sampled_token_nll": diagnostics["sampled_token_nll"],
    }


def _critic_metrics(
    backend: TransformersBackend,
    rollout: PpoRollout,
    shaped_rewards: torch.Tensor,
    frozen_td_targets: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        values = backend.critic(rollout.sequences, rollout.attention)[
            :, rollout.prompt_width - 1 : -1
        ]
        denominator = rollout.response_mask.sum().clamp_min(1.0)
        gae_value_loss = (
            (values - rollout.returns).square() * rollout.response_mask
        ).sum() / denominator
        frozen_td_loss = (
            (values - frozen_td_targets).square() * rollout.response_mask
        ).sum() / denominator
        self_td = one_step_td_residuals(
            shaped_rewards,
            values,
            rollout.response_mask,
            backend.config.gamma,
        )
        self_td_rms = (
            self_td.square() * rollout.response_mask
        ).sum().div(denominator).sqrt()
        prediction_rms = (
            values.square() * rollout.response_mask
        ).sum().div(denominator).sqrt()
    return {
        "gae_value_mse": float(gae_value_loss),
        "frozen_td_mse": float(frozen_td_loss),
        "self_bootstrap_td_rms": float(self_td_rms),
        "value_prediction_rms": float(prediction_rms),
    }


def _parameter_snapshot(
    model: torch.nn.Module, names: set[str],
) -> dict[str, torch.Tensor]:
    parameters = dict(model.named_parameters())
    return {name: parameters[name].detach().cpu().clone() for name in names}


def _total_trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _realized_global_update_rms(
    model: torch.nn.Module,
    originals: Mapping[str, torch.Tensor],
    total_parameter_count: int,
) -> float:
    parameters = dict(model.named_parameters())
    square_sum = 0.0
    for name, original in originals.items():
        parameter = parameters[name]
        difference = parameter.detach().float() - original.to(parameter.device)
        square_sum += float(difference.square().sum())
    return math.sqrt(square_sum / total_parameter_count)


def _load_target_rms(manifest_path: Path) -> dict[str, float]:
    result_path = manifest_path.parent / "result.json"
    result = json.loads(result_path.read_text())
    return {
        "actor": float(result["actor_update_rms_total_mean"]),
        "critic": float(result["critic_update_rms_total_mean"]),
    }


def run_shadow_ablation(
    manifest_path: Path,
    checkpoint_path: Path | None,
    output_path: Path,
) -> None:
    config = _config_from_manifest(manifest_path)
    torch.manual_seed(config.seed)
    backend = TransformersBackend(config)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        backend.actor.load_state_dict(checkpoint["actor"])
        backend.critic.load_state_dict(checkpoint["critic"])
        del checkpoint
    rollouts = []
    for update in (1, 2, 3):
        torch.manual_seed(config.seed * 10_000 + update)
        rollouts.append(backend.collect_rollout(update))
    target_rms = _load_target_rms(manifest_path)
    with torch.no_grad():
        baseline_actor_log_distributions = _actor_response_log_distributions(
            backend, rollouts[2],
        ).detach()
        heldout_shaped_rewards = _shaped_rewards(backend, rollouts[2])
        heldout_baseline_values = backend.critic(
            rollouts[2].sequences, rollouts[2].attention,
        )[:, rollouts[2].prompt_width - 1 : -1]
        heldout_frozen_td_targets = heldout_baseline_values + one_step_td_residuals(
            heldout_shaped_rewards,
            heldout_baseline_values,
            rollouts[2].response_mask,
            backend.config.gamma,
        )
    results: dict[str, Any] = {
        "version": 1,
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path) if checkpoint_path else "base",
        "seed": config.seed,
        "construction_updates": [1, 2],
        "heldout_update": 3,
        "target_rms": target_rms,
        "actor": {
            "baseline": _actor_metrics(
                backend, rollouts[2], baseline_actor_log_distributions,
            ),
            "candidates": {},
        },
        "critic": {
            "baseline": _critic_metrics(
                backend,
                rollouts[2],
                heldout_shaped_rewards,
                heldout_frozen_td_targets,
            ),
            "candidates": {},
        },
    }

    actor_gradients = []
    for rollout in rollouts[:2]:
        actor_loss, critic_loss, _ = backend.losses(rollout)
        actor_gradients.append(collect_matrix_gradients(backend.actor, actor_loss))
        del actor_loss, critic_loss
    actor_names = set(actor_gradients[0])
    actor_originals = _parameter_snapshot(backend.actor, actor_names)
    actor_parameter_count = _total_trainable_parameter_count(backend.actor)
    for candidate in ACTOR_CANDIDATES:
        raw_updates, metadata = _candidate_updates(
            "actor",
            candidate,
            actor_gradients[0],
            actor_gradients[1],
            backend.device,
        )
        updates, scale = match_update_rms(
            raw_updates,
            target_rms["actor"],
            total_parameter_count=actor_parameter_count,
        )
        with temporary_parameter_update(
            backend.actor, updates, originals=actor_originals,
        ):
            realized_rms = _realized_global_update_rms(
                backend.actor, actor_originals, actor_parameter_count,
            )
            metrics = _actor_metrics(
                backend, rollouts[2], baseline_actor_log_distributions,
            )
        results["actor"]["candidates"][candidate] = {
            **metrics,
            "policy_loss_delta": metrics["policy_loss"]
            - results["actor"]["baseline"]["policy_loss"],
            "matched_update_rms": target_rms["actor"],
            "realized_post_cast_global_update_rms": realized_rms,
            "direction_scale": scale,
            "transform": metadata,
        }
        del raw_updates, updates
    del actor_gradients, actor_originals

    critic_gradients = []
    for rollout in rollouts[:2]:
        actor_loss, critic_loss, _ = backend.losses(rollout)
        critic_gradients.append(collect_matrix_gradients(backend.critic, critic_loss))
        del actor_loss, critic_loss
    critic_names = set(critic_gradients[0])
    critic_originals = _parameter_snapshot(backend.critic, critic_names)
    critic_parameter_count = _total_trainable_parameter_count(backend.critic)
    for candidate in CRITIC_CANDIDATES:
        raw_updates, metadata = _candidate_updates(
            "critic",
            candidate,
            critic_gradients[0],
            critic_gradients[1],
            backend.device,
        )
        updates, scale = match_update_rms(
            raw_updates,
            target_rms["critic"],
            total_parameter_count=critic_parameter_count,
        )
        with temporary_parameter_update(
            backend.critic, updates, originals=critic_originals,
        ):
            realized_rms = _realized_global_update_rms(
                backend.critic, critic_originals, critic_parameter_count,
            )
            metrics = _critic_metrics(
                backend,
                rollouts[2],
                heldout_shaped_rewards,
                heldout_frozen_td_targets,
            )
        results["critic"]["candidates"][candidate] = {
            **metrics,
            "frozen_td_mse_delta": metrics["frozen_td_mse"]
            - results["critic"]["baseline"]["frozen_td_mse"],
            "self_bootstrap_td_rms_delta": metrics["self_bootstrap_td_rms"]
            - results["critic"]["baseline"]["self_bootstrap_td_rms"],
            "matched_update_rms": target_rms["critic"],
            "realized_post_cast_global_update_rms": realized_rms,
            "direction_scale": scale,
            "transform": metadata,
        }
        del raw_updates, updates

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_shadow_ablation(args.manifest, args.checkpoint, args.output)

