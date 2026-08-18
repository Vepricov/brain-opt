#!/usr/bin/env python3
"""Calibrate Lion LR to AdamW's exact occupied-state categorical KL."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

if optimizer_source := os.environ.get("RL_MUON_OPTIMIZERS_SOURCE"):
    optimizer_spec = importlib.util.spec_from_file_location("lion_candidate_optimizers", optimizer_source)
    if optimizer_spec is None or optimizer_spec.loader is None:
        raise ImportError(f"cannot load optimizer source: {optimizer_source}")
    optimizer_module = importlib.util.module_from_spec(optimizer_spec)
    optimizer_spec.loader.exec_module(optimizer_module)
    Lion = optimizer_module.Lion
else:
    from verl.utils.optimizers import Lion

MATCH_RELATIVE_TOLERANCE = 0.10
CANDIDATE_MULTIPLIERS = (0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0)


class ValueModel(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(int(backbone.config.hidden_size), 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
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
        [float(extract_answer(text) == answer.replace(",", "")) for text, answer in zip(texts, answers)],
        dtype=torch.float32,
    )


def load_rows(path: str, batch_size: int) -> tuple[list[str], list[str]]:
    rows = pq.read_table(path, columns=["prompt", "reward_model"]).slice(0, batch_size).to_pylist()
    return (
        [row["prompt"][0]["content"] for row in rows],
        [str(row["reward_model"]["ground_truth"]) for row in rows],
    )


def response_logprobs(
    model: nn.Module, sequences: torch.Tensor, attention: torch.Tensor, prompt_width: int
) -> torch.Tensor:
    logits = model(input_ids=sequences, attention_mask=attention).logits[:, :-1].float()
    selected = logits.log_softmax(dim=-1).gather(-1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
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
        old_logits = baseline(input_ids=inputs, attention_mask=mask).logits[:, prompt_width - 1 : -1].float()
        new_logits = candidate(input_ids=inputs, attention_mask=mask).logits[:, prompt_width - 1 : -1].float()
        old_logp = old_logits.log_softmax(dim=-1)
        new_logp = new_logits.log_softmax(dim=-1)
        per_state = (old_logp.exp() * (old_logp - new_logp)).sum(dim=-1).squeeze(0)
        values.append(per_state[response_mask[index].bool()].cpu())
        del old_logits, new_logits, old_logp, new_logp, per_state
    result = torch.cat(values).float()
    if not len(result) or not bool(torch.isfinite(result).all()) or bool((result < -1e-5).any()):
        raise RuntimeError("invalid occupied-state categorical KL")
    return result.clamp_min(0)


def summarize_kl(values: torch.Tensor) -> dict[str, float | int]:
    return {
        "mean": float(values.mean()),
        "q95": float(torch.quantile(values, 0.95)),
        "occupied_states": int(values.numel()),
    }


def select_candidate(trials: list[dict[str, Any]], adam_kl: dict[str, float | int]) -> dict[str, Any]:
    for trial in trials:
        for metric in ("mean", "q95"):
            trial[f"{metric}_relative_error"] = abs(float(trial[metric]) - float(adam_kl[metric])) / max(
                float(adam_kl[metric]), 1e-12
            )
    eligible = [
        trial
        for trial in trials
        if trial["mean_relative_error"] <= MATCH_RELATIVE_TOLERANCE
        and trial["q95_relative_error"] <= MATCH_RELATIVE_TOLERANCE
    ]
    if not eligible:
        raise RuntimeError(f"no Lion learning rate jointly matched Adam mean and q95: {trials}")
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-response-length", type=int, default=128)
    parser.add_argument("--adam-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Lion LR calibration requires CUDA")
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

    actor = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32).to(device)
    actor.gradient_checkpointing_enable()
    actor.config.use_cache = False
    reference = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16).to(device).eval()
    reference.requires_grad_(False)
    critic = ValueModel(AutoModel.from_pretrained(args.model_path, torch_dtype=torch.float32).to(device)).to(device)
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
    rewards = exact_rewards(tokenizer.batch_decode(response_ids, skip_special_tokens=True), answers).to(device)

    actor.train()
    critic.train()
    with torch.no_grad():
        old_logprobs = response_logprobs(actor, sequences, attention, prompt_width)
        ref_logprobs = response_logprobs(reference, sequences, attention, prompt_width)
        old_values = critic(sequences, attention)[:, prompt_width - 1 : -1]
        shaped_rewards = -0.001 * (old_logprobs - ref_logprobs)
        last_indices = response_mask.sum(dim=1).long().clamp_min(1) - 1
        shaped_rewards[torch.arange(sequences.shape[0], device=device), last_indices] += rewards
        advantages = torch.zeros_like(shaped_rewards)
        gae = torch.zeros(sequences.shape[0], device=device)
        for position in reversed(range(shaped_rewards.shape[1])):
            next_mask = response_mask[:, position + 1] if position + 1 < shaped_rewards.shape[1] else torch.zeros_like(gae)
            next_value = old_values[:, position + 1] if position + 1 < old_values.shape[1] else torch.zeros_like(gae)
            delta = shaped_rewards[:, position] + next_value * next_mask - old_values[:, position]
            gae = delta + 0.95 * next_mask * gae
            advantages[:, position] = gae * response_mask[:, position]
        valid = response_mask.bool()
        advantages = (advantages - advantages[valid].mean()) / (advantages[valid].std(unbiased=False) + 1e-8)

    actor.zero_grad(set_to_none=True)
    loss = policy_loss(response_logprobs(actor, sequences, attention, prompt_width), old_logprobs, advantages, response_mask, args.clip_coef)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    baseline_state = {name: tensor.detach().cpu().clone() for name, tensor in actor.state_dict().items()}
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in actor.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }
    if len(gradients) != sum(parameter.requires_grad for parameter in actor.parameters()):
        raise RuntimeError("not all trainable actor parameters received a calibration gradient")
    rollout_sha256 = tensor_digest([sequences, attention, old_logprobs, advantages, response_mask])
    del critic, reference, ref_logprobs, old_values, shaped_rewards, gae
    torch.cuda.empty_cache()

    baseline = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32).to(device).eval()
    baseline.load_state_dict(baseline_state)
    baseline.requires_grad_(False)

    trials: list[dict[str, Any]] = []
    restore_with_gradients(actor, baseline_state, gradients)
    adam = torch.optim.AdamW(actor.parameters(), lr=args.adam_lr, weight_decay=args.weight_decay)
    adam.step()
    adam_kl = summarize_kl(occupied_state_categorical_kl(
        baseline, actor, sequences, attention, prompt_width, response_mask
    ))
    for multiplier in CANDIDATE_MULTIPLIERS:
        learning_rate = args.adam_lr * multiplier
        restore_with_gradients(actor, baseline_state, gradients)
        lion = Lion(actor.parameters(), lr=learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.99))
        lion.step()
        summary = summarize_kl(occupied_state_categorical_kl(
            baseline, actor, sequences, attention, prompt_width, response_mask
        ))
        summary["learning_rate"] = learning_rate
        trials.append(summary)
        del lion
        torch.cuda.empty_cache()

    chosen = select_candidate(trials, adam_kl)
    model_config = Path(args.model_path) / "config.json"
    result = {
        "schema_version": 1,
        "status": "complete",
        "protocol": "single_frozen_gsm8k_rollout_one_production_equivalent_ppo_update",
        "seed": args.seed,
        "adam_learning_rate": args.adam_lr,
        "adam_actual_full_categorical_kl": adam_kl,
        "lion_candidates": trials,
        "chosen_lion_learning_rate": chosen["learning_rate"],
        "chosen_lion_actual_full_categorical_kl": {
            key: chosen[key]
            for key in ("mean", "q95", "occupied_states", "mean_relative_error", "q95_relative_error")
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
            "model_config_sha256": sha256_file(model_config),
            "rollout_sha256": rollout_sha256,
            "batch_size": args.batch_size,
            "occupied_states": int(response_mask.sum()),
        },
        "quantile_convention": "torch_linear_interpolation_over_occupied_response_states",
        "kl_direction": "old_policy_to_updated_policy",
        "precision": "float32_actor_and_KL_logits",
    }
    encoded_result = json.dumps(result, sort_keys=True, separators=(",", ":"))
    Path(args.output).write_text(encoded_result + "\n")
    print("RL_MUON_LION_CALIBRATION " + encoded_result, flush=True)


if __name__ == "__main__":
    main()
