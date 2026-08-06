#!/usr/bin/env python3
"""Read-only actor/critic gradient-geometry probe for an existing PPO checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from llm_ppo import LlmPPOConfig, TransformersBackend, partition_parameters


def canonical_parameter_name(name: str) -> str:
    for prefix in ("model.", "backbone."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def module_type(name: str) -> str:
    value = canonical_parameter_name(name)
    for label in (
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
        "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ):
        if label in value:
            return label
    return "other"


def deterministic_sketch(matrix: torch.Tensor, points: int = 4096) -> torch.Tensor:
    flat = matrix.detach().float().reshape(-1)
    if flat.numel() <= points:
        result = flat
    else:
        indices = torch.linspace(
            0, flat.numel() - 1, points, device=flat.device,
        ).round().long()
        result = flat[indices]
    norm = result.norm().clamp_min(torch.finfo(result.dtype).tiny)
    return (result / norm).cpu()


def _top_spectrum(
    matrix: torch.Tensor, *, name: str, rank: int = 8, iterations: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float | None]:
    """Return a reproducible randomized top-spectrum approximation and diagnostics."""
    value = matrix.detach().float()
    q = min(rank + 2, min(value.shape))
    if q < 1:
        raise ValueError("matrix must have two non-empty dimensions")
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little") % (2**31)
    devices = [value.device] if value.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        u, singular, v = torch.svd_lowrank(value, q=q, niter=iterations)
    order = singular.argsort(descending=True)
    u, singular, v = u[:, order], singular[order], v[:, order]
    kept = min(rank, len(singular))
    residuals = []
    denominator = value.norm().clamp_min(torch.finfo(value.dtype).tiny)
    for index in range(kept):
        residuals.append(
            float((value @ v[:, index] - singular[index] * u[:, index]).norm() / denominator)
        )
    boundary_gap = None
    if len(singular) > kept:
        boundary_gap = float((singular[kept - 1] - singular[kept]) / singular[0].clamp_min(1e-30))
    return (
        u[:, :kept].cpu(),
        singular[:kept].cpu(),
        v[:, :kept].cpu(),
        max(residuals, default=0.0),
        boundary_gap,
    )


def matrix_geometry(
    name: str, gradient: torch.Tensor, *, spectral_rank: int = 8,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if gradient.ndim != 2:
        raise ValueError("geometry probe only supports matrices")
    value = gradient.detach().float()
    rows, columns = value.shape
    square_sum = float(value.square().sum())
    frobenius = math.sqrt(square_sum)
    canonical_name = canonical_parameter_name(name)
    left, singular, right, residual, boundary_gap = _top_spectrum(
        value, name=canonical_name, rank=spectral_rank,
    )
    sigma_max_estimate = float(singular[0])
    stable_rank_estimate = square_sum / max(sigma_max_estimate * sigma_max_estimate, 1e-30)
    top_energy = float(singular.square().sum()) / max(square_sum, 1e-30)
    tail_energy = max(0.0, 1.0 - top_energy)
    spectral_buckets = singular.square().tolist() + [tail_energy * square_sum]
    total_bucket = max(sum(spectral_buckets), 1e-30)
    probabilities = [value / total_bucket for value in spectral_buckets if value > 0.0]
    entropy = -sum(probability * math.log(probability) for probability in probabilities)
    entropy /= max(math.log(len(spectral_buckets)), 1.0)
    row_norms = value.square().sum(dim=1).sqrt()
    column_norms = value.square().sum(dim=0).sqrt()

    def coefficient_of_variation(values: torch.Tensor) -> float:
        mean = values.mean().abs().clamp_min(1e-30)
        return float(values.std(unbiased=False) / mean)

    minimum_dimension = min(rows, columns)
    metrics = {
        "name": canonical_name,
        "module_type": module_type(name),
        "rows": rows,
        "columns": columns,
        "gradient_rms": frobenius / math.sqrt(rows * columns),
        "frobenius_norm": frobenius,
        "sigma_max_estimate": sigma_max_estimate,
        "stable_rank_estimate": stable_rank_estimate,
        "stable_rank_fraction_estimate": stable_rank_estimate / minimum_dimension,
        "top_spectrum_rank": len(singular),
        "top_spectrum_energy_estimate": top_energy,
        "tail_energy_fraction_estimate": tail_energy,
        "top_spectrum_max_triplet_residual_rel_fro": residual,
        "top_spectrum_boundary_gap_estimate": boundary_gap,
        "spectral_entropy_proxy": entropy,
        "row_norm_cv": coefficient_of_variation(row_norms),
        "column_norm_cv": coefficient_of_variation(column_norms),
    }
    state = {
        "sketch": deterministic_sketch(value),
        "left": left,
        "right": right,
    }
    return metrics, state


def vector_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    size = min(first.numel(), second.numel())
    if size == 0:
        return 0.0
    left = first.reshape(-1)[:size].float()
    right = second.reshape(-1)[:size].float()
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def subspace_overlap(first: torch.Tensor, second: torch.Tensor) -> float:
    rank = min(first.shape[1], second.shape[1])
    if rank == 0 or first.shape[0] != second.shape[0]:
        return 0.0
    cross = first[:, :rank].float().T @ second[:, :rank].float()
    return float(cross.square().sum() / rank)


def capture_geometry(model: torch.nn.Module, loss: torch.Tensor) -> dict[str, dict[str, Any]]:
    model.zero_grad(set_to_none=True)
    loss.backward()
    result: dict[str, dict[str, Any]] = {}
    for name, parameter in partition_parameters(model)["muon"]:
        if parameter.grad is None:
            continue
        metrics, state = matrix_geometry(name, parameter.grad)
        result[metrics["name"]] = {"metrics": metrics, "state": state}
    model.zero_grad(set_to_none=True)
    return result


def compare_batches(
    first: dict[str, dict[str, Any]], second: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    result = {}
    for name in sorted(set(first) & set(second)):
        one, two = first[name]["state"], second[name]["state"]
        result[name] = {
            "gradient_sketch_cosine": vector_cosine(one["sketch"], two["sketch"]),
            "left_top_subspace_overlap": subspace_overlap(one["left"], two["left"]),
            "right_top_subspace_overlap": subspace_overlap(one["right"], two["right"]),
        }
    return result


def aligned_actor_critic(
    actor: dict[str, dict[str, Any]], critic: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    result = {}
    for name in sorted(set(actor) & set(critic)):
        actor_metrics = actor[name]["metrics"]
        critic_metrics = critic[name]["metrics"]
        actor_state, critic_state = actor[name]["state"], critic[name]["state"]
        result[name] = {
            "actor_to_critic_stable_rank_estimate_ratio": actor_metrics["stable_rank_estimate"]
            / max(critic_metrics["stable_rank_estimate"], 1e-30),
            "actor_minus_critic_tail_energy_estimate": actor_metrics[
                "tail_energy_fraction_estimate"
            ] - critic_metrics["tail_energy_fraction_estimate"],
            "gradient_sketch_cosine": vector_cosine(
                actor_state["sketch"], critic_state["sketch"],
            ),
            "left_top_subspace_overlap": subspace_overlap(
                actor_state["left"], critic_state["left"],
            ),
            "right_top_subspace_overlap": subspace_overlap(
                actor_state["right"], critic_state["right"],
            ),
        }
    return result


def _config_from_manifest(path: Path) -> LlmPPOConfig:
    raw = json.loads(path.read_text())["config"]
    allowed = {field.name for field in fields(LlmPPOConfig)}
    config = {key: value for key, value in raw.items() if key in allowed}
    for key in ("train_prompts_path", "eval_prompts_path"):
        config[key] = Path(config[key])
    return LlmPPOConfig(**config)


def _serializable(batch: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {name: value["metrics"] for name, value in batch.items()}


def run_probe(manifest_path: Path, checkpoint_path: Path | None, output_path: Path) -> None:
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
        torch.manual_seed(config.seed * 10_000 + update)
        rollout = backend.collect_rollout(update)
        actor_loss, critic_loss, loss_metrics = backend.losses(rollout)
        actor = capture_geometry(backend.actor, actor_loss)
        critic = capture_geometry(backend.critic, critic_loss)
        batches.append({"actor": actor, "critic": critic, "loss_metrics": loss_metrics})
    output = {
        "version": 1,
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else "base",
        "seed": config.seed,
        "batches": [
            {
                "actor": _serializable(item["actor"]),
                "critic": _serializable(item["critic"]),
                "loss_metrics": item["loss_metrics"],
                "aligned_actor_critic": aligned_actor_critic(
                    item["actor"], item["critic"],
                ),
            }
            for item in batches
        ],
        "cross_batch": {
            "actor": compare_batches(batches[0]["actor"], batches[1]["actor"]),
            "critic": compare_batches(batches[0]["critic"], batches[1]["critic"]),
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
    arguments = parse_args()
    run_probe(arguments.manifest, arguments.checkpoint, arguments.output)


