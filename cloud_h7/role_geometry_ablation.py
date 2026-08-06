#!/usr/bin/env python3
"""Offline H7 ablations for critic targets and actor advantage weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from actor_critic_geometry import (
    _config_from_manifest,
    capture_geometry,
    compare_batches,
)
from llm_ppo import PpoRollout, TransformersBackend, clipped_policy_loss


ACTOR_VARIANTS = (
    "ppo_normalized",
    "ppo_raw",
    "ppo_token_shuffle",
    "ppo_random_sign",
    "behavior_cloning",
)
CRITIC_VARIANTS = (
    "gae_bootstrap",
    "mc_no_bootstrap",
    "td_token_shuffle",
    "td_random_sign",
)


def shuffle_valid(
    values: torch.Tensor, mask: torch.Tensor, *, seed: int,
) -> torch.Tensor:
    valid = mask.bool()
    flattened = values[valid]
    generator = torch.Generator(device=values.device).manual_seed(seed)
    permutation = torch.randperm(flattened.numel(), device=values.device, generator=generator)
    result = torch.zeros_like(values)
    result[valid] = flattened[permutation]
    return result


def randomize_sign(
    values: torch.Tensor, mask: torch.Tensor, *, seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=values.device).manual_seed(seed)
    signs = torch.randint(
        0, 2, values.shape, device=values.device, generator=generator,
    ).mul_(2).sub_(1)
    return values.abs() * signs * mask


def discounted_returns(
    rewards: torch.Tensor, mask: torch.Tensor, gamma: float,
) -> torch.Tensor:
    result = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    for position in reversed(range(rewards.shape[1])):
        next_mask = (
            mask[:, position + 1]
            if position + 1 < rewards.shape[1]
            else torch.zeros_like(running)
        )
        running = rewards[:, position] + gamma * next_mask * running
        result[:, position] = running * mask[:, position]
    return result


def one_step_td_residuals(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    result = torch.zeros_like(rewards)
    for position in range(rewards.shape[1]):
        next_mask = (
            mask[:, position + 1]
            if position + 1 < rewards.shape[1]
            else torch.zeros_like(mask[:, position])
        )
        next_value = (
            values[:, position + 1]
            if position + 1 < values.shape[1]
            else torch.zeros_like(values[:, position])
        )
        result[:, position] = (
            rewards[:, position]
            + gamma * next_mask * next_value
            - values[:, position]
        ) * mask[:, position]
    return result


def _shaped_rewards(
    backend: TransformersBackend, rollout: PpoRollout,
) -> torch.Tensor:
    with torch.no_grad():
        reference_logprobs = backend._response_logprobs(
            backend.reference,
            rollout.sequences,
            rollout.attention,
            rollout.prompt_width,
        )
        terminal_rewards = backend._terminal_rewards(rollout.sequences)
        shaped_rewards = -backend.config.kl_coef * (
            rollout.old_logprobs - reference_logprobs
        )
        last_indices = rollout.response_mask.sum(dim=1).long().clamp_min(1) - 1
        shaped_rewards[
            torch.arange(rollout.sequences.shape[0], device=backend.device), last_indices
        ] += terminal_rewards
        return shaped_rewards


def _monte_carlo_targets(
    backend: TransformersBackend,
    rollout: PpoRollout,
    shaped_rewards: torch.Tensor,
) -> torch.Tensor:
    return discounted_returns(
        shaped_rewards, rollout.response_mask, backend.config.gamma,
    )


def _signals(
    backend: TransformersBackend, rollout: PpoRollout, *, batch_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    with torch.no_grad():
        old_values = backend.critic(rollout.sequences, rollout.attention)[
            :, rollout.prompt_width - 1 : -1
        ]
        raw_advantages = (rollout.returns - old_values) * rollout.response_mask
        shaped_rewards = _shaped_rewards(backend, rollout)
        td_residual = one_step_td_residuals(
            shaped_rewards,
            old_values,
            rollout.response_mask,
            backend.config.gamma,
        )
        shuffled_td = shuffle_valid(
            td_residual, rollout.response_mask, seed=batch_seed + 101,
        )
        signed_td = randomize_sign(
            td_residual, rollout.response_mask, seed=batch_seed + 102,
        )
        actor_signals = {
            "ppo_normalized": rollout.advantages,
            "ppo_raw": raw_advantages,
            "ppo_token_shuffle": shuffle_valid(
                rollout.advantages, rollout.response_mask, seed=batch_seed + 201,
            ),
            "ppo_random_sign": randomize_sign(
                rollout.advantages, rollout.response_mask, seed=batch_seed + 202,
            ),
        }
        critic_targets = {
            "gae_bootstrap": rollout.returns,
            "mc_no_bootstrap": _monte_carlo_targets(
                backend, rollout, shaped_rewards,
            ),
            "td_token_shuffle": old_values + shuffled_td,
            "td_random_sign": old_values + signed_td,
        }
    return actor_signals, critic_targets


def _signal_summary(values: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    selected = values[mask.bool()].float()
    return {
        "mean": float(selected.mean()),
        "std": float(selected.std(unbiased=False)),
        "rms": float(selected.square().mean().sqrt()),
    }


def _actor_loss(
    backend: TransformersBackend,
    rollout: PpoRollout,
    variant: str,
    actor_signals: dict[str, torch.Tensor],
) -> torch.Tensor:
    new_logprobs = backend._response_logprobs(
        backend.actor, rollout.sequences, rollout.attention, rollout.prompt_width,
    )
    denominator = rollout.response_mask.sum().clamp_min(1.0)
    if variant == "behavior_cloning":
        return -(new_logprobs * rollout.response_mask).sum() / denominator
    loss, _ = clipped_policy_loss(
        new_logprobs,
        rollout.old_logprobs,
        actor_signals[variant],
        rollout.response_mask,
        backend.config.clip_coef,
    )
    return loss


def _critic_loss(
    backend: TransformersBackend,
    rollout: PpoRollout,
    target: torch.Tensor,
) -> torch.Tensor:
    values = backend.critic(rollout.sequences, rollout.attention)[
        :, rollout.prompt_width - 1 : -1
    ]
    denominator = rollout.response_mask.sum().clamp_min(1.0)
    return backend.config.value_coef * (
        (values - target).square() * rollout.response_mask
    ).sum() / denominator


def _serializable(batch: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        variant: {name: value["metrics"] for name, value in matrices.items()}
        for variant, matrices in batch.items()
    }


def run_ablation(
    manifest_path: Path, checkpoint_path: Path | None, output_path: Path,
) -> None:
    config = _config_from_manifest(manifest_path)
    torch.manual_seed(config.seed)
    backend = TransformersBackend(config)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        backend.actor.load_state_dict(checkpoint["actor"])
        backend.critic.load_state_dict(checkpoint["critic"])
        del checkpoint

    batches = []
    for update in (1, 2):
        batch_seed = config.seed * 10_000 + update
        torch.manual_seed(batch_seed)
        rollout = backend.collect_rollout(update)
        actor_signals, critic_targets = _signals(
            backend, rollout, batch_seed=batch_seed,
        )
        actor_geometry = {}
        actor_summaries = {}
        for variant in ACTOR_VARIANTS:
            loss = _actor_loss(backend, rollout, variant, actor_signals)
            actor_geometry[variant] = capture_geometry(backend.actor, loss)
            signal = (
                -backend._response_logprobs(
                    backend.actor,
                    rollout.sequences,
                    rollout.attention,
                    rollout.prompt_width,
                ).detach()
                if variant == "behavior_cloning"
                else actor_signals[variant]
            )
            actor_summaries[variant] = {
                "loss": float(loss.detach()),
                "signal": _signal_summary(signal, rollout.response_mask),
            }

        critic_geometry = {}
        critic_summaries = {}
        with torch.no_grad():
            fixed_values = backend.critic(rollout.sequences, rollout.attention)[
                :, rollout.prompt_width - 1 : -1
            ]
        for variant in CRITIC_VARIANTS:
            target = critic_targets[variant]
            loss = _critic_loss(backend, rollout, target)
            critic_geometry[variant] = capture_geometry(backend.critic, loss)
            critic_summaries[variant] = {
                "loss": float(loss.detach()),
                "residual": _signal_summary(
                    target - fixed_values, rollout.response_mask,
                ),
            }
        batches.append(
            {
                "actor": actor_geometry,
                "critic": critic_geometry,
                "actor_summaries": actor_summaries,
                "critic_summaries": critic_summaries,
            }
        )

    output = {
        "version": 1,
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path) if checkpoint_path else "base",
        "seed": config.seed,
        "actor_variants": list(ACTOR_VARIANTS),
        "critic_variants": list(CRITIC_VARIANTS),
        "batches": [
            {
                "actor": _serializable(item["actor"]),
                "critic": _serializable(item["critic"]),
                "actor_summaries": item["actor_summaries"],
                "critic_summaries": item["critic_summaries"],
            }
            for item in batches
        ],
        "cross_batch": {
            "actor": {
                variant: compare_batches(
                    batches[0]["actor"][variant], batches[1]["actor"][variant],
                )
                for variant in ACTOR_VARIANTS
            },
            "critic": {
                variant: compare_batches(
                    batches[0]["critic"][variant], batches[1]["critic"][variant],
                )
                for variant in CRITIC_VARIANTS
            },
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ablation(args.manifest, args.checkpoint, args.output)

