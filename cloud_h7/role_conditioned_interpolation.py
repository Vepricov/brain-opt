#!/usr/bin/env python3
"""KL-controlled actor and frozen-TD critic interpolation ablations."""
from __future__ import annotations

import argparse
import json
from typing import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from actor_critic_geometry import _config_from_manifest
from llm_ppo import TransformersBackend
from role_geometry_ablation import _shaped_rewards, one_step_td_residuals
from shadow_geometry_ablation import (
    _actor_metrics,
    _actor_response_log_distributions,
    _candidate_updates,
    _critic_metrics,
    _load_target_rms,
    _parameter_snapshot,
    _realized_global_update_rms,
    _total_trainable_parameter_count,
    collect_matrix_gradients,
    match_update_rms,
    temporary_parameter_update,
)


BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def blend_matched_updates(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
    *,
    beta: float,
    target_rms: float,
    total_parameter_count: int,
) -> dict[str, torch.Tensor]:
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if set(first) != set(second):
        raise ValueError("update parameter sets do not match")
    left, _ = match_update_rms(
        first, target_rms, total_parameter_count=total_parameter_count,
    )
    right, _ = match_update_rms(
        second, target_rms, total_parameter_count=total_parameter_count,
    )
    blended = {
        name: (1.0 - beta) * left[name] + beta * right[name]
        for name in left
    }
    matched, _ = match_update_rms(
        blended, target_rms, total_parameter_count=total_parameter_count,
    )
    return matched


def calibrate_scale_to_budget(
    evaluate_kl: Callable[[float], float],
    *,
    budget: float,
    max_scale: float = 1.0,
    steps: int = 8,
    grid_points: int = 5,
) -> tuple[float, float, dict[str, Any]]:
    if budget <= 0.0:
        raise ValueError("budget must be positive")
    if max_scale <= 0.0:
        raise ValueError("max_scale must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if grid_points < 2:
        raise ValueError("grid_points must be at least two")
    grid = [max_scale * index / (grid_points - 1) for index in range(grid_points)]
    grid_kl = [evaluate_kl(scale) for scale in grid]
    tolerance = max(1e-12, budget * 1e-3)
    monotone = all(
        right + tolerance >= left for left, right in zip(grid_kl, grid_kl[1:])
    )
    diagnostics: dict[str, Any] = {
        "monotone_grid": monotone,
        "used_grid_fallback": not monotone,
        "monotonicity_tolerance": tolerance,
        "grid": [
            {"scale": scale, "categorical_kl": kl}
            for scale, kl in zip(grid, grid_kl)
        ],
    }
    safe = [
        (scale, kl) for scale, kl in zip(grid, grid_kl) if kl <= budget
    ]
    if not safe:
        raise RuntimeError("no calibration grid point satisfies the KL budget")
    if not monotone:
        scale, measured = max(safe, key=lambda row: row[0])
        return scale, measured, diagnostics
    lower, lower_kl = safe[-1]
    if lower == max_scale:
        return lower, lower_kl, diagnostics
    lower_index = grid.index(lower)
    upper = grid[lower_index + 1]
    for _ in range(steps):
        middle = 0.5 * (lower + upper)
        middle_kl = evaluate_kl(middle)
        if middle_kl <= budget:
            lower = middle
            lower_kl = middle_kl
        else:
            upper = middle
    return lower, lower_kl, diagnostics


def _scaled_updates(
    updates: Mapping[str, torch.Tensor], scale: float,
) -> dict[str, torch.Tensor]:
    return {name: value * scale for name, value in updates.items()}


def _candidate_name(beta: float) -> str:
    return f"beta_{int(round(beta * 100)):03d}"


def _critic_evaluation_context(
    backend: TransformersBackend, rollout: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        shaped_rewards = _shaped_rewards(backend, rollout)
        baseline_values = backend.critic(
            rollout.sequences, rollout.attention,
        )[:, rollout.prompt_width - 1 : -1]
        frozen_targets = baseline_values + one_step_td_residuals(
            shaped_rewards,
            baseline_values,
            rollout.response_mask,
            backend.config.gamma,
        )
    return shaped_rewards, frozen_targets


def run_interpolation_ablation(
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
    for update in (1, 2, 3, 4):
        torch.manual_seed(config.seed * 10_000 + update)
        rollouts.append(backend.collect_rollout(update))
    target_rms = _load_target_rms(manifest_path)

    with torch.no_grad():
        actor_calibration_reference = _actor_response_log_distributions(
            backend, rollouts[2],
        ).detach()
        actor_heldout_reference = _actor_response_log_distributions(
            backend, rollouts[3],
        ).detach()
    critic_calibration_rewards, critic_calibration_targets = (
        _critic_evaluation_context(backend, rollouts[2])
    )
    critic_heldout_rewards, critic_heldout_targets = _critic_evaluation_context(
        backend, rollouts[3],
    )

    actor_baseline_calibration = _actor_metrics(
        backend, rollouts[2], actor_calibration_reference,
    )
    actor_baseline_heldout = _actor_metrics(
        backend, rollouts[3], actor_heldout_reference,
    )
    critic_baseline_calibration = _critic_metrics(
        backend,
        rollouts[2],
        critic_calibration_rewards,
        critic_calibration_targets,
    )
    critic_baseline_heldout = _critic_metrics(
        backend,
        rollouts[3],
        critic_heldout_rewards,
        critic_heldout_targets,
    )
    results: dict[str, Any] = {
        "version": 1,
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path) if checkpoint_path else "base",
        "seed": config.seed,
        "construction_updates": [1, 2],
        "calibration_update": 3,
        "heldout_update": 4,
        "target_rms": target_rms,
        "betas": list(BETAS),
        "actor": {
            "baseline_calibration": actor_baseline_calibration,
            "baseline_heldout": actor_baseline_heldout,
            "kl_budget_source": "muon_average_full_rms_on_calibration_update",
            "candidates": {},
        },
        "critic": {
            "baseline_calibration": critic_baseline_calibration,
            "baseline_heldout": critic_baseline_heldout,
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
    actor_muon, _ = _candidate_updates(
        "actor", "muon_average", actor_gradients[0], actor_gradients[1], backend.device,
    )
    actor_consensus, consensus_metadata = _candidate_updates(
        "actor",
        "consensus_alpha0_top16",
        actor_gradients[0],
        actor_gradients[1],
        backend.device,
    )
    actor_muon_matched = blend_matched_updates(
        actor_muon,
        actor_consensus,
        beta=0.0,
        target_rms=target_rms["actor"],
        total_parameter_count=actor_parameter_count,
    )
    with temporary_parameter_update(
        backend.actor, actor_muon_matched, originals=actor_originals,
    ):
        muon_calibration_metrics = _actor_metrics(
            backend, rollouts[2], actor_calibration_reference,
        )
    kl_budget = muon_calibration_metrics["categorical_kl"]
    if kl_budget <= 0.0:
        raise RuntimeError("Muon calibration KL must be positive")
    results["actor"]["kl_budget"] = kl_budget
    results["actor"]["consensus_transform"] = consensus_metadata

    for beta in BETAS:
        updates = blend_matched_updates(
            actor_muon,
            actor_consensus,
            beta=beta,
            target_rms=target_rms["actor"],
            total_parameter_count=actor_parameter_count,
        )

        def evaluate_calibration(scale: float) -> dict[str, float]:
            with temporary_parameter_update(
                backend.actor,
                _scaled_updates(updates, scale),
                originals=actor_originals,
            ):
                return _actor_metrics(
                    backend, rollouts[2], actor_calibration_reference,
                )

        full_scale_calibration = evaluate_calibration(1.0)
        scale, _, calibration_diagnostics = calibrate_scale_to_budget(
            lambda value: evaluate_calibration(value)["categorical_kl"],
            budget=kl_budget,
            max_scale=1.0,
            steps=6,
            grid_points=5,
        )
        calibrated_updates = _scaled_updates(updates, scale)
        calibration_metrics = evaluate_calibration(scale)
        with temporary_parameter_update(
            backend.actor, calibrated_updates, originals=actor_originals,
        ):
            realized_rms = _realized_global_update_rms(
                backend.actor, actor_originals, actor_parameter_count,
            )
            heldout_metrics = _actor_metrics(
                backend, rollouts[3], actor_heldout_reference,
            )
        results["actor"]["candidates"][_candidate_name(beta)] = {
            "beta": beta,
            "calibrated_scale": scale,
            "kl_calibration_diagnostics": calibration_diagnostics,
            "full_scale_target_rms": target_rms["actor"],
            "realized_calibrated_global_update_rms": realized_rms,
            "full_scale_calibration": full_scale_calibration,
            "calibration": calibration_metrics,
            "heldout": {
                **heldout_metrics,
                "policy_loss_delta": heldout_metrics["policy_loss"]
                - actor_baseline_heldout["policy_loss"],
            },
        }
        del updates, calibrated_updates
    del actor_gradients, actor_muon, actor_consensus, actor_muon_matched, actor_originals

    critic_gradients = []
    for rollout in rollouts[:2]:
        actor_loss, critic_loss, _ = backend.losses(rollout)
        critic_gradients.append(collect_matrix_gradients(backend.critic, critic_loss))
        del actor_loss, critic_loss
    critic_names = set(critic_gradients[0])
    critic_originals = _parameter_snapshot(backend.critic, critic_names)
    critic_parameter_count = _total_trainable_parameter_count(backend.critic)
    critic_muon, _ = _candidate_updates(
        "critic", "muon_average", critic_gradients[0], critic_gradients[1], backend.device,
    )
    critic_alpha0, alpha0_metadata = _candidate_updates(
        "critic",
        "power_alpha0_top16",
        critic_gradients[0],
        critic_gradients[1],
        backend.device,
    )
    results["critic"]["alpha0_transform"] = alpha0_metadata
    for beta in BETAS:
        updates = blend_matched_updates(
            critic_muon,
            critic_alpha0,
            beta=beta,
            target_rms=target_rms["critic"],
            total_parameter_count=critic_parameter_count,
        )
        with temporary_parameter_update(
            backend.critic, updates, originals=critic_originals,
        ):
            realized_rms = _realized_global_update_rms(
                backend.critic, critic_originals, critic_parameter_count,
            )
            calibration_metrics = _critic_metrics(
                backend,
                rollouts[2],
                critic_calibration_rewards,
                critic_calibration_targets,
            )
            heldout_metrics = _critic_metrics(
                backend,
                rollouts[3],
                critic_heldout_rewards,
                critic_heldout_targets,
            )
        results["critic"]["candidates"][_candidate_name(beta)] = {
            "beta": beta,
            "full_scale_target_rms": target_rms["critic"],
            "realized_global_update_rms": realized_rms,
            "calibration": {
                **calibration_metrics,
                "frozen_td_mse_delta": calibration_metrics["frozen_td_mse"]
                - critic_baseline_calibration["frozen_td_mse"],
            },
            "heldout": {
                **heldout_metrics,
                "frozen_td_mse_delta": heldout_metrics["frozen_td_mse"]
                - critic_baseline_heldout["frozen_td_mse"],
                "self_bootstrap_td_rms_delta": heldout_metrics[
                    "self_bootstrap_td_rms"
                ] - critic_baseline_heldout["self_bootstrap_td_rms"],
            },
        }
        del updates

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
    run_interpolation_ablation(args.manifest, args.checkpoint, args.output)

