#!/usr/bin/env python3
"""Isolated, non-resumable H7.3 online PPO pilot for Qwen critics."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path("/home/shkodnik/rl_muon")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_ppo import (
    LlmPPOConfig,
    TransformersBackend,
    _validate_config,
    clipped_policy_loss,
    normalized_trapezoid_auc,
    partition_parameters,
)
from role_conditioned_interpolation import _critic_evaluation_context
from shadow_geometry_ablation import _critic_metrics

from weighted_polar import (
    DiagonalFactorCollector,
    HiddenUpdateCommitter,
    calibrate_scale_strict,
    match_global_rms,
    polar_direction,
    raw_muon_direction,
    weighted_polar_direction,
)


ROUTES = ("raw_muon", "own_polar_d01", "own_polar_d1")
TARGET_GLOBAL_RMS = 5.8e-7


def build_actor_optimizer(
    parameters: list[torch.nn.Parameter], lr: float
) -> torch.optim.AdamW:
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)


def route_direction(
    route: str,
    gradients: Mapping[str, torch.Tensor],
    factors: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if route == "raw_muon":
        return raw_muon_direction(gradients)
    if route == "own_polar_d01":
        return weighted_polar_direction(gradients, factors, damping=0.1)
    if route == "own_polar_d1":
        return weighted_polar_direction(gradients, factors, damping=1.0)
    raise ValueError(f"unknown critic route: {route}")


def _response_values(
    critic: torch.nn.Module,
    *,
    sequences: torch.Tensor,
    attention: torch.Tensor,
    prompt_width: int,
) -> torch.Tensor:
    return critic(sequences, attention)[:, prompt_width - 1 : -1]


def functional_value_change_rms(
    critic: torch.nn.Module,
    *,
    sequences: torch.Tensor,
    attention: torch.Tensor,
    prompt_width: int,
    response_mask: torch.Tensor,
    baseline_values: torch.Tensor,
    updates: Mapping[str, torch.Tensor],
    scale: float,
) -> float:
    """Evaluate a hidden proposal functionally, without an in-place trial update."""
    parameters = dict(critic.named_parameters())
    replacements = {
        name: parameters[name] + scale * update.to(
            device=parameters[name].device, dtype=parameters[name].dtype
        )
        for name, update in updates.items()
    }
    with torch.no_grad():
        values = torch.func.functional_call(
            critic,
            replacements,
            (sequences, attention),
            strict=False,
        )[:, prompt_width - 1 : -1]
        denominator = response_mask.sum().clamp_min(1.0)
        change = (
            (values - baseline_values).square() * response_mask
        ).sum().div(denominator).sqrt()
    return float(change)


def _value_change_rms(
    current: torch.Tensor, baseline: torch.Tensor, mask: torch.Tensor
) -> float:
    denominator = mask.sum().clamp_min(1.0)
    return float((((current - baseline).square() * mask).sum() / denominator).sqrt())


def _direction_cosine(
    first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]
) -> float:
    numerator = sum(float((first[name].float() * second[name].float()).sum()) for name in first)
    first_norm = math.sqrt(sum(float(value.float().square().sum()) for value in first.values()))
    second_norm = math.sqrt(sum(float(value.float().square().sum()) for value in second.values()))
    return numerator / (first_norm * second_norm)


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _write_json(path: Path, row: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(row, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _config_from_json(path: Path) -> LlmPPOConfig:
    raw = json.loads(path.read_text())
    values = dict(raw.get("config", raw))
    values["train_prompts_path"] = Path(values["train_prompts_path"])
    values["eval_prompts_path"] = Path(values["eval_prompts_path"])
    config = LlmPPOConfig(**values)
    _validate_config(config, production=True)
    if config.total_updates != 20:
        raise ValueError("H7.3 pilot requires exactly 20 updates")
    if config.ppo_epochs != 1:
        raise ValueError("H7.3 pilot requires exactly one PPO epoch")
    if config.actor_optimizer != "adam":
        raise ValueError("H7.3 actor route is fixed to Adam")
    if config.batch_size != 2 or config.eval_every != 5:
        raise ValueError("H7.3 requires batch_size=2 and eval_every=5")
    if config.actor_adam_lr != 3e-6 or config.critic_adam_lr != 3e-6:
        raise ValueError("H7.3 requires actor and critic auxiliary AdamW LR=3e-6")
    qwen_ids = (config.actor_model_id, config.reference_model_id, config.critic_model_id)
    if any(not model_id.startswith("Qwen/") for model_id in qwen_ids):
        raise ValueError("H7.3 is a Qwen-only PPO pilot")
    return config


def run(config: LlmPPOConfig, route: str, run_dir: Path) -> dict[str, Any]:
    if route not in ROUTES:
        raise ValueError(f"unknown critic route: {route}")
    if not torch.cuda.is_available():
        raise RuntimeError("H7.3 is server-only and requires CUDA")
    run_dir.mkdir(parents=True, exist_ok=False)
    progress_path = run_dir / "progress.jsonl"
    eval_path = run_dir / "eval.jsonl"
    progress_path.write_text("")
    eval_path.write_text("")
    manifest = {
        "version": 1,
        "experiment": "h7_3_online_ppo_pilot",
        "route": route,
        "seed": config.seed,
        "target_global_rms": TARGET_GLOBAL_RMS,
        "resumable": False,
        "pairing": "same seed and prompt files; independent route-specific rollouts",
        "config": {
            **asdict(config),
            "train_prompts_path": str(config.train_prompts_path),
            "eval_prompts_path": str(config.eval_prompts_path),
        },
    }
    _write_json(run_dir / "manifest.json", manifest)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    backend = TransformersBackend(config)
    actor_parameters = [parameter for parameter in backend.actor.parameters() if parameter.requires_grad]
    actor_optimizer = build_actor_optimizer(actor_parameters, config.actor_adam_lr)
    critic_parts = partition_parameters(backend.critic)
    hidden_named = critic_parts["muon"]
    aux_named = critic_parts["adamw"]
    if not hidden_named or not aux_named:
        raise RuntimeError("critic must contain both hidden and auxiliary parameters")
    aux_optimizer = torch.optim.AdamW(
        [parameter for _, parameter in aux_named],
        lr=config.critic_adam_lr,
        weight_decay=0.0,
    )
    hidden_names = {name for name, _ in hidden_named}
    all_critic_parameters = [
        parameter for parameter in backend.critic.parameters() if parameter.requires_grad
    ]
    total_critic_parameter_count = sum(parameter.numel() for parameter in all_critic_parameters)
    evaluations: list[dict[str, Any]] = []

    def evaluate(update: int) -> dict[str, Any]:
        metrics = backend.evaluate()
        row = {"update": update, **metrics}
        _append_jsonl(eval_path, row)
        evaluations.append(row)
        return row

    evaluate(0)
    try:
        for update in range(1, config.total_updates + 1):
            rollout = backend.collect_rollout(update)
            with torch.no_grad():
                pre_step_values = _response_values(
                    backend.critic,
                    sequences=rollout.sequences,
                    attention=rollout.attention,
                    prompt_width=rollout.prompt_width,
                ).detach()
            shaped_rewards, frozen_td_targets = _critic_evaluation_context(backend, rollout)

            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss, unused_critic_loss, policy_metrics = backend.losses(rollout)
            del unused_critic_loss
            actor_loss.backward()
            actor_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(actor_parameters, config.max_grad_norm)
            )
            actor_optimizer.step()
            with torch.no_grad():
                post_actor_logprobs = backend._response_logprobs(
                    backend.actor,
                    rollout.sequences,
                    rollout.attention,
                    rollout.prompt_width,
                )
                post_actor_loss, post_policy_metrics = clipped_policy_loss(
                    post_actor_logprobs,
                    rollout.old_logprobs,
                    rollout.advantages,
                    rollout.response_mask,
                    config.clip_coef,
                )

            backend.critic.zero_grad(set_to_none=True)
            with DiagonalFactorCollector(backend.critic, hidden_names) as collector:
                values = _response_values(
                    backend.critic,
                    sequences=rollout.sequences,
                    attention=rollout.attention,
                    prompt_width=rollout.prompt_width,
                )
                critic_loss = config.value_coef * (
                    (values - rollout.returns).square() * rollout.response_mask
                ).sum() / rollout.response_mask.sum().clamp_min(1.0)
                critic_loss.backward()
            critic_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(all_critic_parameters, config.max_grad_norm)
            )
            gradients = {
                name: parameter.grad.detach().float().clone()
                for name, parameter in hidden_named
                if parameter.grad is not None
            }
            if set(gradients) != hidden_names:
                raise RuntimeError("missing clipped hidden critic gradients")
            factors = collector.factors()

            aux_optimizer.step()
            with torch.no_grad():
                after_aux_values = _response_values(
                    backend.critic,
                    sequences=rollout.sequences,
                    attention=rollout.attention,
                    prompt_width=rollout.prompt_width,
                ).detach()

            raw_direction = route_direction("raw_muon", gradients, factors)
            raw_matched, raw_match_scale = match_global_rms(
                raw_direction,
                target_rms=TARGET_GLOBAL_RMS,
                total_parameter_count=total_critic_parameter_count,
            )
            selected_direction = route_direction(route, gradients, factors)
            selected_matched, selected_match_scale = match_global_rms(
                selected_direction,
                target_rms=TARGET_GLOBAL_RMS,
                total_parameter_count=total_critic_parameter_count,
            )

            evaluate_functional = lambda scale: functional_value_change_rms(
                backend.critic,
                sequences=rollout.sequences,
                attention=rollout.attention,
                prompt_width=rollout.prompt_width,
                response_mask=rollout.response_mask,
                baseline_values=after_aux_values,
                updates=selected_matched,
                scale=scale,
            )
            functional_budget = functional_value_change_rms(
                backend.critic,
                sequences=rollout.sequences,
                attention=rollout.attention,
                prompt_width=rollout.prompt_width,
                response_mask=rollout.response_mask,
                baseline_values=after_aux_values,
                updates=raw_matched,
                scale=1.0,
            )
            calibrated_scale, measured_budget, calibration = calibrate_scale_strict(
                evaluate_functional,
                budget=functional_budget,
                initial_max_scale=4.0,
                max_scale=16.0,
                tolerance=0.02,
            )
            committer = HiddenUpdateCommitter(backend.critic)
            realized_hidden_square_sum = committer.commit(
                selected_matched, scale=calibrated_scale
            )
            realized_hidden_global_rms = math.sqrt(
                realized_hidden_square_sum / total_critic_parameter_count
            )

            post_metrics = _critic_metrics(
                backend, rollout, shaped_rewards, frozen_td_targets
            )
            with torch.no_grad():
                post_values = _response_values(
                    backend.critic,
                    sequences=rollout.sequences,
                    attention=rollout.attention,
                    prompt_width=rollout.prompt_width,
                )
                valid = rollout.response_mask.bool()
                return_variance = rollout.returns[valid].float().var(unbiased=False)
                residual_variance = (
                    rollout.returns[valid] - post_values[valid]
                ).float().var(unbiased=False)
                explained_variance = float(
                    1.0 - residual_variance / return_variance.clamp_min(1e-8)
                )
                realized_functional_value_change_rms = _value_change_rms(
                    post_values, after_aux_values, rollout.response_mask
                )
                realized_functional_budget_ratio = (
                    realized_functional_value_change_rms / functional_budget
                )
                if not 0.98 <= realized_functional_budget_ratio <= 1.02:
                    raise RuntimeError(
                        "committed functional budget mismatch: "
                        f"ratio={realized_functional_budget_ratio}"
                    )
            eval_metrics = None
            if update % config.eval_every == 0 or update == config.total_updates:
                eval_metrics = evaluate(update)
            row = {
                "update": update,
                "route": route,
                "ppo_epochs": 1,
                "actor_loss": float(actor_loss.detach()),
                "post_step_actor_loss": float(post_actor_loss.detach()),
                "critic_loss": float(critic_loss.detach()),
                "actor_grad_norm_before_clip": actor_grad_norm,
                "critic_grad_norm_before_clip": critic_grad_norm,
                "pre_step_approx_kl": policy_metrics["approx_kl"],
                "pre_step_clip_fraction": policy_metrics["clip_fraction"],
                "approx_kl": post_policy_metrics["approx_kl"],
                "clip_fraction": post_policy_metrics["clip_fraction"],
                "sampled_token_nll": post_policy_metrics["sampled_token_nll"],
                "reference_kl": rollout.reference_kl,
                "terminal_reward": rollout.terminal_reward,
                "functional_budget": functional_budget,
                "calibration_value_change_rms": measured_budget,
                "calibration_budget_ratio": measured_budget / functional_budget,
                "realized_functional_value_change_rms": realized_functional_value_change_rms,
                "realized_functional_budget_ratio": realized_functional_budget_ratio,
                "calibrated_scale": calibrated_scale,
                "calibration": calibration,
                "raw_muon_match_scale": raw_match_scale,
                "selected_match_scale": selected_match_scale,
                "realized_hidden_global_rms": realized_hidden_global_rms,
                "direction_cosine_to_raw_muon": _direction_cosine(
                    selected_direction, raw_direction
                ),
                "aux_value_change_rms": _value_change_rms(
                    after_aux_values, pre_step_values, rollout.response_mask
                ),
                "value_change_rms": _value_change_rms(
                    post_values, pre_step_values, rollout.response_mask
                ),
                "post_step_frozen_td_mse": post_metrics["frozen_td_mse"],
                "post_step_gae_mse": post_metrics["gae_value_mse"],
                "post_step_self_bootstrap_td_rms": post_metrics["self_bootstrap_td_rms"],
                "post_step_explained_variance": explained_variance,
                "evaluator": eval_metrics,
            }
            _append_jsonl(progress_path, row)
            backend.critic.zero_grad(set_to_none=True)

        reward_points = [
            (int(row["update"]), float(row["reward_mean"])) for row in evaluations
        ]
        evaluator_points = [
            (int(row["update"]), float(row["evaluator_reward_mean"]))
            for row in evaluations
        ]
        result = {
            "status": "complete",
            "route": route,
            "seed": config.seed,
            "total_updates": config.total_updates,
            "target_global_rms": TARGET_GLOBAL_RMS,
            "resumable": False,
            "final_evaluation": evaluations[-1],
            "eval_reward_auc": normalized_trapezoid_auc(reward_points),
            "evaluator_reward_auc": normalized_trapezoid_auc(evaluator_points),
        }
        _write_json(run_dir / "result.json", result)
        return result
    except Exception as error:
        _write_json(
            run_dir / "failure.json",
            {"status": "failed", "route": route, "seed": config.seed, "error": repr(error)},
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--route", choices=ROUTES, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(_config_from_json(args.config), args.route, args.run_dir)


if __name__ == "__main__":
    main()


