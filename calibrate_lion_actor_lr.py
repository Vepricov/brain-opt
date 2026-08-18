#!/usr/bin/env python3
"""Calibrate Muon and Lion LRs to AdamW's occupied-state categorical KL."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from cross_dataset_screen import (
    model_identity,
    verl_identity,
    verify_identity_artifact,
)

MATCH_RELATIVE_TOLERANCE = 0.10
CANDIDATE_MULTIPLIERS = (0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)
CALIBRATION_METRIC = "exact_full_categorical_KL_old_to_new_on_occupied_response_states"


def load_optimizers() -> tuple[type, type]:
    if optimizer_source := os.environ.get("RL_MUON_OPTIMIZERS_SOURCE"):
        optimizer_spec = importlib.util.spec_from_file_location(
            "calibration_optimizers", optimizer_source
        )
        if optimizer_spec is None or optimizer_spec.loader is None:
            raise ImportError(f"cannot load optimizer source: {optimizer_source}")
        optimizer_module = importlib.util.module_from_spec(optimizer_spec)
        optimizer_spec.loader.exec_module(optimizer_module)
        return optimizer_module.Lion, optimizer_module.MuonWithAuxAdamW
    from verl.utils.optimizers import Lion, MuonWithAuxAdamW

    return Lion, MuonWithAuxAdamW


class ValueModel(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(int(backbone.config.hidden_size), 1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return self.value_head(hidden).squeeze(-1)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def extract_answer(text: str) -> str | None:
    matches = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    return matches[-1].replace(",", "") if matches else None


def exact_rewards(texts: list[str], answers: list[str]) -> torch.Tensor:
    return torch.tensor(
        [
            float(extract_answer(text) == answer.replace(",", ""))
            for text, answer in zip(texts, answers)
        ],
        dtype=torch.float32,
    )


def load_rows(path: str, batch_size: int) -> tuple[list[str], list[str]]:
    import pyarrow.parquet as pq

    rows = (
        pq.read_table(path, columns=["prompt", "reward_model"])
        .slice(0, batch_size)
        .to_pylist()
    )
    return (
        [row["prompt"][0]["content"] for row in rows],
        [str(row["reward_model"]["ground_truth"]) for row in rows],
    )


def response_logprobs(
    model: nn.Module,
    sequences: torch.Tensor,
    attention: torch.Tensor,
    prompt_width: int,
) -> torch.Tensor:
    logits = model(input_ids=sequences, attention_mask=attention).logits[:, :-1].float()
    selected = (
        logits.log_softmax(dim=-1)
        .gather(-1, sequences[:, 1:].unsqueeze(-1))
        .squeeze(-1)
    )
    return selected[:, prompt_width - 1 :]


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_coef: float,
) -> torch.Tensor:
    ratio = (new_logprobs - old_logprobs).exp()
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * advantages
    return -masked_mean(torch.minimum(unclipped, clipped), mask)


@torch.no_grad()
def occupied_state_categorical_kl(
    baseline: nn.Module,
    candidate: nn.Module,
    sequences: torch.Tensor,
    attention: torch.Tensor,
    prompt_width: int,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    values = []
    baseline.eval()
    candidate.eval()
    for index in range(sequences.shape[0]):
        inputs = sequences[index : index + 1]
        mask = attention[index : index + 1]
        old_logits = (
            baseline(input_ids=inputs, attention_mask=mask)
            .logits[:, prompt_width - 1 : -1]
            .float()
        )
        new_logits = (
            candidate(input_ids=inputs, attention_mask=mask)
            .logits[:, prompt_width - 1 : -1]
            .float()
        )
        old_logp = old_logits.log_softmax(dim=-1)
        new_logp = new_logits.log_softmax(dim=-1)
        per_state = (old_logp.exp() * (old_logp - new_logp)).sum(dim=-1).squeeze(0)
        values.append(per_state[response_mask[index].bool()])
        del old_logits, new_logits, old_logp, new_logp, per_state
    result = torch.cat(values).float()
    if (
        not len(result)
        or not bool(torch.isfinite(result).all())
        or bool((result < -1e-5).any())
    ):
        raise RuntimeError("invalid occupied-state categorical KL")
    return result.clamp_min(0)


def summarize_kl(values: torch.Tensor) -> dict[str, float | int]:
    return {
        "mean": float(values.mean()),
        "q95": float(torch.quantile(values, 0.95)),
        "occupied_states": int(values.numel()),
    }


def select_candidate(
    trials: list[dict[str, Any]],
    adam_kl: dict[str, float | int],
    route_name: str = "Lion",
) -> dict[str, Any]:
    for trial in trials:
        for metric in ("mean", "q95"):
            trial[f"{metric}_relative_error"] = abs(
                float(trial[metric]) - float(adam_kl[metric])
            ) / max(float(adam_kl[metric]), 1e-12)
    eligible = [
        trial
        for trial in trials
        if trial["mean_relative_error"] <= MATCH_RELATIVE_TOLERANCE
        and trial["q95_relative_error"] <= MATCH_RELATIVE_TOLERANCE
    ]
    if not eligible:
        raise RuntimeError(
            f"no {route_name} learning rate jointly matched Adam mean and q95: {trials}"
        )
    return min(
        eligible,
        key=lambda trial: (
            max(trial["mean_relative_error"], trial["q95_relative_error"]),
            trial["mean_relative_error"] + trial["q95_relative_error"],
            trial["learning_rate"],
        ),
    )


def restore_with_gradients(
    model: nn.Module,
    baseline_state: dict[str, torch.Tensor],
    gradients: dict[str, torch.Tensor],
) -> None:
    model.load_state_dict(baseline_state)
    for name, parameter in model.named_parameters():
        parameter.grad = gradients[name].to(parameter.device, dtype=parameter.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True, choices=("svamp", "arc_easy"))
    parser.add_argument("--data-source", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--model-identity-artifact", required=True)
    parser.add_argument("--verl-identity-artifact", required=True)
    parser.add_argument("--verl-root", required=True)
    parser.add_argument("--routing-overlay-root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-response-length", type=int, default=128)
    parser.add_argument("--adam-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    args = parser.parse_args()
    model_identity_sha256 = verify_identity_artifact(
        Path(args.model_identity_artifact), model_identity(Path(args.model_path))
    )
    verl_identity_sha256 = verify_identity_artifact(
        Path(args.verl_identity_artifact),
        verl_identity(Path(args.verl_root), Path(args.routing_overlay_root)),
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Lion LR calibration requires CUDA")
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    Lion, MuonWithAuxAdamW = load_optimizers()
    seed_all(args.seed)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    prompts, answers = load_rows(args.train_file, args.batch_size)
    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=args.max_prompt_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_width = encoded["input_ids"].shape[1]

    actor = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float32
    ).to(device)
    actor.gradient_checkpointing_enable()
    actor.config.use_cache = False
    reference = (
        AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16
        )
        .to(device)
        .eval()
    )
    reference.requires_grad_(False)
    critic = ValueModel(
        AutoModel.from_pretrained(args.model_path, torch_dtype=torch.float32).to(device)
    ).to(device)
    critic.backbone.gradient_checkpointing_enable()

    actor.eval()
    torch.manual_seed(args.seed)
    with torch.no_grad():
        sequences = actor.generate(
            **encoded,
            max_new_tokens=args.max_response_length,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    response_ids = sequences[:, prompt_width:]
    response_mask = torch.ones_like(response_ids, dtype=torch.float32)
    if tokenizer.eos_token_id is not None:
        for row in range(response_ids.shape[0]):
            eos = (response_ids[row] == tokenizer.eos_token_id).nonzero(as_tuple=False)
            if eos.numel():
                response_mask[row, int(eos[0]) + 1 :] = 0
    attention = torch.cat([encoded["attention_mask"], response_mask.long()], dim=1)
    rewards = exact_rewards(
        tokenizer.batch_decode(response_ids, skip_special_tokens=True), answers
    ).to(device)

    actor.train()
    critic.train()
    with torch.no_grad():
        old_logprobs = (
            response_logprobs(actor, sequences, attention, prompt_width)
            .detach()
            .clone()
        )
        ref_logprobs = (
            response_logprobs(reference, sequences, attention, prompt_width)
            .detach()
            .clone()
        )
        old_values = critic(sequences, attention)[:, prompt_width - 1 : -1]
        shaped_rewards = -0.001 * (old_logprobs - ref_logprobs)
        last_indices = response_mask.sum(dim=1).long().clamp_min(1) - 1
        shaped_rewards[
            torch.arange(sequences.shape[0], device=device), last_indices
        ] += rewards
        advantages = torch.zeros_like(shaped_rewards)
        gae = torch.zeros(sequences.shape[0], device=device)
        for position in reversed(range(shaped_rewards.shape[1])):
            next_mask = (
                response_mask[:, position + 1]
                if position + 1 < shaped_rewards.shape[1]
                else torch.zeros_like(gae)
            )
            next_value = (
                old_values[:, position + 1]
                if position + 1 < old_values.shape[1]
                else torch.zeros_like(gae)
            )
            delta = (
                shaped_rewards[:, position]
                + next_value * next_mask
                - old_values[:, position]
            )
            gae = delta + 0.95 * next_mask * gae
            advantages[:, position] = gae * response_mask[:, position]
        valid = response_mask.bool()
        advantages = (advantages - advantages[valid].mean()) / (
            advantages[valid].std(unbiased=False) + 1e-8
        )

    actor.zero_grad(set_to_none=True)
    loss = policy_loss(
        response_logprobs(actor, sequences, attention, prompt_width),
        old_logprobs,
        advantages,
        response_mask,
        args.clip_coef,
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    baseline_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in actor.state_dict().items()
    }
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in actor.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    if len(gradients) != sum(
        parameter.requires_grad for parameter in actor.parameters()
    ):
        raise RuntimeError(
            "not all trainable actor parameters received a calibration gradient"
        )
    old_logprobs_sha256 = tensor_digest([old_logprobs])
    reference_logprobs_sha256 = tensor_digest([ref_logprobs])
    rollout_sha256 = tensor_digest(
        [sequences, attention, old_logprobs, advantages, response_mask]
    )
    del critic, reference, old_values, shaped_rewards, gae
    torch.cuda.empty_cache()

    baseline = (
        AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32)
        .to(device)
        .eval()
    )
    baseline.load_state_dict(baseline_state)
    baseline.requires_grad_(False)

    muon_trials: list[dict[str, Any]] = []
    lion_trials: list[dict[str, Any]] = []
    restore_with_gradients(actor, baseline_state, gradients)
    adam = torch.optim.AdamW(
        actor.parameters(), lr=args.adam_lr, weight_decay=args.weight_decay
    )
    adam.step()
    adam_kl = summarize_kl(
        occupied_state_categorical_kl(
            baseline, actor, sequences, attention, prompt_width, response_mask
        )
    )
    if float(adam_kl["mean"]) <= 0 or float(adam_kl["q95"]) <= 0:
        raise RuntimeError(
            f"AdamW policy-change dose must be positive for mean and q95: {adam_kl}"
        )
    del adam
    for multiplier in CANDIDATE_MULTIPLIERS:
        learning_rate = args.adam_lr * multiplier
        restore_with_gradients(actor, baseline_state, gradients)
        muon = MuonWithAuxAdamW(
            actor.named_parameters(),
            lr=learning_rate,
            weight_decay=args.weight_decay,
            muon_adjust_lr_fn="match_rms_adamw",
        )
        muon.step()
        summary = summarize_kl(
            occupied_state_categorical_kl(
                baseline, actor, sequences, attention, prompt_width, response_mask
            )
        )
        summary["learning_rate"] = learning_rate
        muon_trials.append(summary)
        del muon
        torch.cuda.empty_cache()

        restore_with_gradients(actor, baseline_state, gradients)
        lion = Lion(
            actor.parameters(),
            lr=learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.99),
        )
        lion.step()
        summary = summarize_kl(
            occupied_state_categorical_kl(
                baseline, actor, sequences, attention, prompt_width, response_mask
            )
        )
        summary["learning_rate"] = learning_rate
        lion_trials.append(summary)
        del lion
        torch.cuda.empty_cache()

    chosen_muon = select_candidate(muon_trials, adam_kl, "Muon")
    chosen_lion = select_candidate(lion_trials, adam_kl, "Lion")
    if tensor_digest([old_logprobs]) != old_logprobs_sha256:
        raise RuntimeError("old policy logprobs mutated during calibration")
    if tensor_digest([ref_logprobs]) != reference_logprobs_sha256:
        raise RuntimeError("reference policy logprobs mutated during calibration")
    result = {
        "schema_version": 3,
        "status": "complete",
        "protocol": "per_dataset_same_frozen_batch_one_production_equivalent_ppo_update",
        "dataset": args.dataset,
        "data_source": args.data_source,
        "seed": args.seed,
        "source_commit": args.source_commit,
        "manifest_sha256": args.manifest_sha256,
        "calibration_metric": CALIBRATION_METRIC,
        "adam_learning_rate": args.adam_lr,
        "adam_actual_full_categorical_kl": adam_kl,
        "muon_candidates": muon_trials,
        "chosen_muon_learning_rate": chosen_muon["learning_rate"],
        "chosen_muon_actual_full_categorical_kl": {
            key: chosen_muon[key]
            for key in (
                "mean",
                "q95",
                "occupied_states",
                "mean_relative_error",
                "q95_relative_error",
            )
        },
        "lion_candidates": lion_trials,
        "chosen_lion_learning_rate": chosen_lion["learning_rate"],
        "chosen_lion_actual_full_categorical_kl": {
            key: chosen_lion[key]
            for key in (
                "mean",
                "q95",
                "occupied_states",
                "mean_relative_error",
                "q95_relative_error",
            )
        },
        "gates": {
            "adam_mean_kl_target": adam_kl["mean"],
            "adam_q95_kl_target": adam_kl["q95"],
            "mean_match_relative_tolerance": MATCH_RELATIVE_TOLERANCE,
            "q95_match_relative_tolerance": MATCH_RELATIVE_TOLERANCE,
            "passed": True,
        },
        "identities": {
            "train_file_sha256": sha256_file(args.train_file),
            "model_snapshot_sha256": model_identity_sha256,
            "verl_implementation_sha256": verl_identity_sha256,
            "rollout_sha256": rollout_sha256,
            "old_logprobs_sha256": old_logprobs_sha256,
            "reference_logprobs_sha256": reference_logprobs_sha256,
            "batch_size": args.batch_size,
            "occupied_states": int(response_mask.sum()),
        },
        "quantile_convention": "torch_linear_interpolation_over_occupied_response_states",
        "kl_direction": "old_policy_to_updated_policy",
        "precision": "float32_actor_and_KL_logits",
        "model_compute_device": "cuda",
        "actor_routes": {
            "muon_actor": "Muon on hidden attention/MLP matrices plus AdamW auxiliaries",
            "lion_actor": "Lion on all actor parameters; strict no-Adam actor route",
        },
    }
    encoded_result = json.dumps(result, sort_keys=True, separators=(",", ":"))
    Path(args.output).write_text(encoded_result + "\n")
    print("RL_MUON_LION_CALIBRATION " + encoded_result, flush=True)


if __name__ == "__main__":
    main()
