#!/usr/bin/env python3
"""Server-only PPO harness for optimizer routing on language models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from muon_compat import Muon


PROTOCOL_REVISION = 2
OPTIMIZER_KINDS = ("adam", "muon")


@dataclass(frozen=True)
class LlmPPOConfig:
    train_prompts_path: Path
    eval_prompts_path: Path
    actor_model_id: str = "Qwen/Qwen2.5-0.5B"
    actor_model_revision: str = "UNPINNED"
    reference_model_id: str = "Qwen/Qwen2.5-0.5B"
    reference_model_revision: str = "UNPINNED"
    critic_model_id: str = "Qwen/Qwen2.5-0.5B"
    critic_model_revision: str = "UNPINNED"
    reward_model_id: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    reward_model_revision: str = "UNPINNED"
    reward_positive_label: int = 1
    evaluator_model_id: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    evaluator_model_revision: str = "UNPINNED"
    evaluator_positive_label: int = 1
    actor_optimizer: str = "adam"
    critic_optimizer: str = "adam"
    actor_adam_lr: float = 3e-6
    actor_muon_lr: float = 3e-4
    critic_adam_lr: float = 3e-6
    critic_muon_lr: float = 3e-4
    total_updates: int = 100
    ppo_epochs: int = 4
    eval_every: int = 10
    checkpoint_every: int = 25
    seed: int = 0
    batch_size: int = 8
    max_prompt_tokens: int = 32
    max_new_tokens: int = 24
    clip_coef: float = 0.2
    kl_coef: float = 0.02
    gamma: float = 1.0
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    max_grad_norm: float = 1.0


class Backend(Protocol):
    actor: nn.Module
    critic: nn.Module

    def collect_rollout(self, update: int) -> Any: ...

    def losses(self, rollout: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]: ...

    def evaluate(self) -> dict[str, float]: ...

    def save_checkpoint(self, path: Path, update: int) -> None: ...


@dataclass(frozen=True)
class PpoRollout:
    sequences: torch.Tensor
    attention: torch.Tensor
    prompt_width: int
    response_mask: torch.Tensor
    old_logprobs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    terminal_reward: float
    reference_kl: float


def _is_output_linear(name: str) -> bool:
    components = name.lower().split(".")
    return any(
        component in {"lm_head", "value_head", "score", "classifier", "output", "output_layer"}
        for component in components
    )


def partition_parameters(model: nn.Module) -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Partition trainable parameters into hidden Linear weights and AdamW auxiliaries."""
    hidden_weight_ids: set[int] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and not _is_output_linear(module_name):
            hidden_weight_ids.add(id(module.weight))
    result: dict[str, list[tuple[str, nn.Parameter]]] = {"muon": [], "adamw": []}
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        key = "muon" if id(parameter) in hidden_weight_ids else "adamw"
        result[key].append((name, parameter))
    return result


class RoutedOptimizer:
    def __init__(
        self,
        model: nn.Module,
        kind: str,
        adam_lr: float,
        muon_lr: float,
        weight_decay: float,
    ) -> None:
        if kind not in OPTIMIZER_KINDS:
            raise ValueError(f"unknown optimizer: {kind}")
        if min(adam_lr, muon_lr) <= 0:
            raise ValueError("optimizer learning rates must be positive")
        raw_parts = partition_parameters(model)
        if kind == "adam":
            routed = {"muon": [], "adamw": raw_parts["muon"] + raw_parts["adamw"]}
        else:
            routed = raw_parts
        self.parts = routed
        self.routing = {key: [name for name, _ in values] for key, values in routed.items()}
        self.optimizers: list[torch.optim.Optimizer] = []
        muon_parameters = [parameter for _, parameter in routed["muon"]]
        adamw_parameters = [parameter for _, parameter in routed["adamw"]]
        if muon_parameters:
            self.optimizers.append(
                Muon(
                    muon_parameters,
                    lr=muon_lr,
                    momentum=0.95,
                    weight_decay=weight_decay,
                    adjust_lr_fn="match_rms_adamw",
                )
            )
        if adamw_parameters:
            self.optimizers.append(
                torch.optim.AdamW(adamw_parameters, lr=adam_lr, weight_decay=weight_decay)
            )
        if not self.optimizers:
            raise ValueError("model has no trainable parameters")

    def zero_grad(self) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=True)

    def step_with_metrics(self, max_grad_norm: float) -> dict[str, Any]:
        all_parameters = [parameter for values in self.parts.values() for _, parameter in values]
        grad_square_sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for key, values in self.parts.items():
            grad_square_sums[key] = sum(
                float(parameter.grad.detach().float().square().sum())
                for _, parameter in values
                if parameter.grad is not None
            )
            counts[key] = sum(parameter.numel() for _, parameter in values)
        torch.nn.utils.clip_grad_norm_(all_parameters, max_grad_norm)
        before = {id(parameter): parameter.detach().clone() for parameter in all_parameters}
        for optimizer in self.optimizers:
            optimizer.step()
        square_sums: dict[str, float] = {}
        for key, values in self.parts.items():
            square_sums[key] = sum(
                float((parameter.detach() - before[id(parameter)]).float().square().sum())
                for _, parameter in values
            )
        total_square_sum = sum(square_sums.values())
        total_grad_square_sum = sum(grad_square_sums.values())
        total_count = sum(counts.values())
        return {
            "update_rms": {
                "total": math.sqrt(total_square_sum / max(total_count, 1)),
                "muon": math.sqrt(square_sums["muon"] / max(counts["muon"], 1)),
                "adamw": math.sqrt(square_sums["adamw"] / max(counts["adamw"], 1)),
            },
            "grad_rms": {
                "total": math.sqrt(total_grad_square_sum / max(total_count, 1)),
                "muon": math.sqrt(grad_square_sums["muon"] / max(counts["muon"], 1)),
                "adamw": math.sqrt(grad_square_sums["adamw"] / max(counts["adamw"], 1)),
            },
            "update_square_sum": {"total": total_square_sum, **square_sums},
            "grad_square_sum": {"total": total_grad_square_sum, **grad_square_sums},
            "parameter_count": {"total": total_count, **counts},
        }


def build_optimizer(
    model: nn.Module,
    kind: str,
    adam_lr: float,
    muon_lr: float,
    weight_decay: float = 0.0,
) -> RoutedOptimizer:
    return RoutedOptimizer(model, kind, adam_lr, muon_lr, weight_decay)


def rms_from_totals(square_sum: float, count: float) -> float:
    if square_sum < 0.0 or count < 0.0:
        raise ValueError("RMS totals cannot be negative")
    return math.sqrt(square_sum / max(count, 1.0))


def normalized_trapezoid_auc(points: list[tuple[int, float]]) -> float:
    """Average ordinate under a curve whose abscissa is the PPO update index."""
    if not points:
        raise ValueError("AUC requires at least one point")
    if len(points) == 1:
        return float(points[0][1])
    for left, right in zip(points, points[1:]):
        if right[0] <= left[0]:
            raise ValueError("AUC update indices must be strictly increasing")
    area = sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for (left_x, left_y), (right_x, right_y) in zip(points, points[1:])
    )
    return area / (points[-1][0] - points[0][0])


def clipped_policy_loss(
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_coef: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the masked PPO clipped objective and observable policy diagnostics."""
    denominator = mask.sum().clamp_min(1.0)
    logratio = new_logprobs - old_logprobs
    ratio = logratio.exp()
    surrogate = torch.minimum(
        ratio * advantages,
        ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef) * advantages,
    )
    loss = -(surrogate * mask).sum() / denominator
    with torch.no_grad():
        metrics = {
            "approx_kl": float((((ratio - 1.0) - logratio) * mask).sum() / denominator),
            "clip_fraction": float((((ratio - 1.0).abs() > clip_coef).float() * mask).sum() / denominator),
            "sampled_token_nll": float(((-new_logprobs) * mask).sum() / denominator),
        }
    return loss, metrics


def _config_dict(config: LlmPPOConfig) -> dict[str, Any]:
    row = asdict(config)
    row["train_prompts_path"] = str(config.train_prompts_path)
    row["eval_prompts_path"] = str(config.eval_prompts_path)
    return row


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_hashes(config: LlmPPOConfig) -> dict[str, str]:
    return {
        "train": _file_sha256(config.train_prompts_path),
        "eval": _file_sha256(config.eval_prompts_path),
    }


def runtime_metadata() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "transformers_version": importlib.metadata.version("transformers"),
        "datasets_version": importlib.metadata.version("datasets"),
        "accelerate_version": importlib.metadata.version("accelerate"),
        "cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "device": torch.cuda.get_device_name(0) if cuda_available else "cpu",
    }


def canonical_run_id(config: LlmPPOConfig) -> str:
    payload = json.dumps(
        {
            "protocol_revision": PROTOCOL_REVISION,
            "config": _config_dict(config),
            "prompt_sha256": prompt_hashes(config),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
    return f"llmppo-a-{config.actor_optimizer}_c-{config.critic_optimizer}-{digest}"


def run_manifest(config: LlmPPOConfig, run_id: str) -> dict[str, Any]:
    return {
        "protocol_revision": PROTOCOL_REVISION,
        "run_id": run_id,
        "config": _config_dict(config),
        "prompt_sha256": prompt_hashes(config),
        "runtime": runtime_metadata(),
        "model_revisions": {
            "actor": {"model_id": config.actor_model_id, "revision": config.actor_model_revision},
            "reference": {
                "model_id": config.reference_model_id,
                "revision": config.reference_model_revision,
            },
            "critic": {"model_id": config.critic_model_id, "revision": config.critic_model_revision},
            "reward": {"model_id": config.reward_model_id, "revision": config.reward_model_revision},
            "evaluator": {
                "model_id": config.evaluator_model_id,
                "revision": config.evaluator_model_revision,
            },
        },
    }


def _write_json(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as stream:
            stream.write(text)
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _validate_config(config: LlmPPOConfig, *, production: bool) -> None:
    if config.actor_optimizer not in OPTIMIZER_KINDS or config.critic_optimizer not in OPTIMIZER_KINDS:
        raise ValueError("actor_optimizer and critic_optimizer must be adam or muon")
    if min(
        config.total_updates,
        config.ppo_epochs,
        config.eval_every,
        config.batch_size,
        config.max_prompt_tokens,
        config.max_new_tokens,
    ) <= 0:
        raise ValueError("update, evaluation, batch, and generation counts must be positive")
    if config.checkpoint_every < 0:
        raise ValueError("checkpoint_every cannot be negative")
    if min(
        config.actor_adam_lr,
        config.actor_muon_lr,
        config.critic_adam_lr,
        config.critic_muon_lr,
    ) <= 0:
        raise ValueError("learning rates must be positive")
    if min(config.reward_positive_label, config.evaluator_positive_label) < 0:
        raise ValueError("reward and evaluator positive labels cannot be negative")
    if not 0.0 < config.clip_coef < 1.0 or config.kl_coef < 0.0:
        raise ValueError("clip_coef must be in (0, 1) and kl_coef cannot be negative")
    if not 0.0 < config.gamma <= 1.0 or not 0.0 <= config.gae_lambda <= 1.0:
        raise ValueError("gamma must be in (0, 1] and gae_lambda in [0, 1]")
    if min(config.value_coef, config.max_grad_norm) <= 0.0:
        raise ValueError("value_coef and max_grad_norm must be positive")
    if production:
        identities = {
            (config.actor_model_id, config.actor_model_revision),
            (config.reference_model_id, config.reference_model_revision),
            (config.critic_model_id, config.critic_model_revision),
        }
        if len(identities) != 1:
            raise ValueError("actor, reference, and critic must use the same model ID and revision")
        revisions = (
            config.actor_model_revision,
            config.reference_model_revision,
            config.critic_model_revision,
            config.reward_model_revision,
            config.evaluator_model_revision,
        )
        if any(re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None for revision in revisions):
            raise ValueError("production model revisions must be immutable 40-character commit SHAs")
        if (
            config.reward_model_id,
            config.reward_model_revision,
        ) == (
            config.evaluator_model_id,
            config.evaluator_model_revision,
        ):
            raise ValueError("production requires an independent evaluator model")


class ValueModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.value_head(hidden).squeeze(-1)


def _read_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = row if isinstance(row, str) else row.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"invalid prompt at {path}:{line_number}")
        prompts.append(prompt)
    if not prompts:
        raise ValueError(f"no prompts in {path}")
    return prompts


class TransformersBackend:
    def __init__(self, config: LlmPPOConfig) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("LLM PPO training requires CUDA; CPU fallback is disabled")
        from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
        from transformers import AutoTokenizer

        self.config = config
        self.device = torch.device("cuda")
        frozen_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.actor_model_id, revision=config.actor_model_revision
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.reward_tokenizer = AutoTokenizer.from_pretrained(
            config.reward_model_id, revision=config.reward_model_revision
        )
        self.evaluator_tokenizer = AutoTokenizer.from_pretrained(
            config.evaluator_model_id, revision=config.evaluator_model_revision
        )
        self.actor = AutoModelForCausalLM.from_pretrained(
            config.actor_model_id,
            revision=config.actor_model_revision,
            torch_dtype=torch.float32,
        ).to(self.device)
        self.reference = AutoModelForCausalLM.from_pretrained(
            config.reference_model_id,
            revision=config.reference_model_revision,
            torch_dtype=frozen_dtype,
        ).to(self.device)
        critic_backbone = AutoModel.from_pretrained(
            config.critic_model_id,
            revision=config.critic_model_revision,
            torch_dtype=torch.float32,
        ).to(self.device)
        hidden_size = int(critic_backbone.config.hidden_size)
        self.critic = ValueModel(critic_backbone, hidden_size).to(
            device=self.device, dtype=torch.float32
        )
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            config.reward_model_id,
            revision=config.reward_model_revision,
            torch_dtype=frozen_dtype,
        ).to(self.device)
        self.evaluator_model = AutoModelForSequenceClassification.from_pretrained(
            config.evaluator_model_id,
            revision=config.evaluator_model_revision,
            torch_dtype=frozen_dtype,
        ).to(self.device)
        for frozen in (self.reference, self.reward_model, self.evaluator_model):
            frozen.requires_grad_(False)
            frozen.eval()
        self.train_prompts = _read_prompts(config.train_prompts_path)
        self.eval_prompts = _read_prompts(config.eval_prompts_path)

    def _tokenize(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.max_prompt_tokens,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _generate(self, prompts: list[str], *, sample: bool) -> tuple[torch.Tensor, torch.Tensor, int]:
        encoded = self._tokenize(prompts)
        prompt_width = encoded["input_ids"].shape[1]
        kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if sample:
            kwargs.update({"temperature": 1.0, "top_p": 1.0})
        with torch.no_grad():
            sequences = self.actor.generate(**encoded, **kwargs)
        response_ids = sequences[:, prompt_width:]
        response_mask = torch.ones_like(response_ids, dtype=torch.float32)
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            for row in range(response_ids.shape[0]):
                eos_positions = (response_ids[row] == eos_id).nonzero(as_tuple=False)
                if eos_positions.numel():
                    response_mask[row, int(eos_positions[0]) + 1 :] = 0.0
        attention = torch.cat([encoded["attention_mask"], response_mask.long()], dim=1)
        return sequences, attention, prompt_width

    @staticmethod
    def _response_logprobs(
        model: nn.Module, sequences: torch.Tensor, attention: torch.Tensor, prompt_width: int
    ) -> torch.Tensor:
        logits = model(input_ids=sequences, attention_mask=attention).logits[:, :-1].float()
        token_logprobs = logits.log_softmax(dim=-1).gather(
            -1, sequences[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        return token_logprobs[:, prompt_width - 1 :]

    def _sequence_scores(
        self,
        sequences: torch.Tensor,
        *,
        tokenizer: Any,
        model: nn.Module,
        positive_label: int,
        role: str,
    ) -> torch.Tensor:
        texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        encoded = tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits.float()
        if logits.shape[-1] == 1:
            return logits.squeeze(-1)
        if not 0 <= positive_label < logits.shape[-1]:
            raise ValueError(f"{role}_positive_label is outside the model label range")
        return logits.softmax(dim=-1)[:, positive_label]

    def _terminal_rewards(self, sequences: torch.Tensor) -> torch.Tensor:
        return self._sequence_scores(
            sequences,
            tokenizer=self.reward_tokenizer,
            model=self.reward_model,
            positive_label=self.config.reward_positive_label,
            role="reward",
        )

    def _evaluator_rewards(self, sequences: torch.Tensor) -> torch.Tensor:
        return self._sequence_scores(
            sequences,
            tokenizer=self.evaluator_tokenizer,
            model=self.evaluator_model,
            positive_label=self.config.evaluator_positive_label,
            role="evaluator",
        )

    def collect_rollout(self, update: int) -> PpoRollout:
        start = ((update - 1) * self.config.batch_size) % len(self.train_prompts)
        prompts = [
            self.train_prompts[(start + offset) % len(self.train_prompts)]
            for offset in range(self.config.batch_size)
        ]
        sequences, attention, prompt_width = self._generate(prompts, sample=True)
        response_mask = attention[:, prompt_width:].float()
        with torch.no_grad():
            old_logprobs = self._response_logprobs(
                self.actor, sequences, attention, prompt_width
            )
            reference_logprobs = self._response_logprobs(
                self.reference, sequences, attention, prompt_width
            )
            old_values = self.critic(sequences, attention)[:, prompt_width - 1 : -1]
            terminal_rewards = self._terminal_rewards(sequences)
            shaped_rewards = -self.config.kl_coef * (old_logprobs - reference_logprobs)
            last_indices = response_mask.sum(dim=1).long().clamp_min(1) - 1
            shaped_rewards[torch.arange(sequences.shape[0], device=self.device), last_indices] += terminal_rewards
            advantages = torch.zeros_like(shaped_rewards)
            gae = torch.zeros(sequences.shape[0], device=self.device)
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
                delta = shaped_rewards[:, position] + self.config.gamma * next_value * next_mask - old_values[:, position]
                gae = delta + self.config.gamma * self.config.gae_lambda * next_mask * gae
                advantages[:, position] = gae * response_mask[:, position]
            returns = advantages + old_values
            valid_advantages = advantages[response_mask.bool()]
            advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std(unbiased=False) + 1e-8)

        return PpoRollout(
            sequences=sequences,
            attention=attention,
            prompt_width=prompt_width,
            response_mask=response_mask,
            old_logprobs=old_logprobs,
            advantages=advantages,
            returns=returns,
            terminal_reward=float(terminal_rewards.mean()),
            reference_kl=float(
                ((old_logprobs - reference_logprobs) * response_mask).sum()
                / response_mask.sum().clamp_min(1.0)
            ),
        )

    def losses(self, rollout: PpoRollout) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        new_logprobs = self._response_logprobs(
            self.actor, rollout.sequences, rollout.attention, rollout.prompt_width
        )
        actor_loss, policy_metrics = clipped_policy_loss(
            new_logprobs,
            rollout.old_logprobs,
            rollout.advantages,
            rollout.response_mask,
            self.config.clip_coef,
        )
        values = self.critic(rollout.sequences, rollout.attention)[
            :, rollout.prompt_width - 1 : -1
        ]
        critic_loss = self.config.value_coef * (
            (values - rollout.returns).square() * rollout.response_mask
        ).sum() / rollout.response_mask.sum().clamp_min(1.0)
        valid = rollout.response_mask.bool()
        return_variance = rollout.returns[valid].float().var(unbiased=False)
        residual_variance = (
            rollout.returns[valid] - values[valid].detach()
        ).float().var(unbiased=False)
        td_error_rms = (rollout.returns[valid] - values[valid].detach()).float().square().mean().sqrt()
        return actor_loss, critic_loss, {
            **policy_metrics,
            "value_loss": float(critic_loss.detach()),
            "value_mean": float(values[valid].detach().mean()),
            "explained_variance": float(1.0 - residual_variance / return_variance.clamp_min(1e-8)),
            "td_error_rms": float(td_error_rms),
            "terminal_reward": rollout.terminal_reward,
            "reference_kl": rollout.reference_kl,
        }

    def evaluate(self) -> dict[str, float]:
        was_training = self.actor.training
        self.actor.eval()
        rewards: list[torch.Tensor] = []
        evaluator_rewards: list[torch.Tensor] = []
        for start in range(0, len(self.eval_prompts), self.config.batch_size):
            prompts = self.eval_prompts[start : start + self.config.batch_size]
            sequences, _, _ = self._generate(prompts, sample=False)
            rewards.append(self._terminal_rewards(sequences))
            evaluator_rewards.append(self._evaluator_rewards(sequences))
        self.actor.train(was_training)
        values = torch.cat(rewards).float()
        evaluator_values = torch.cat(evaluator_rewards).float()
        return {
            "reward_mean": float(values.mean()),
            "reward_std": float(values.std(unbiased=False)),
            "evaluator_reward_mean": float(evaluator_values.mean()),
            "evaluator_reward_std": float(evaluator_values.std(unbiased=False)),
        }

    def save_checkpoint(self, path: Path, update: int) -> None:
        torch.save(
            {"update": update, "actor": self.actor.state_dict(), "critic": self.critic.state_dict()},
            path,
        )


def _new_rms_accumulator() -> dict[str, dict[str, dict[str, list[float]]]]:
    return {
        role: {
            metric: {part: [0.0, 0.0] for part in ("total", "muon", "adamw")}
            for metric in ("update", "grad")
        }
        for role in ("actor", "critic")
    }


def _accumulate_step(
    accumulator: dict[str, dict[str, dict[str, list[float]]]],
    role: str,
    metrics: dict[str, Any],
) -> None:
    for metric in ("update", "grad"):
        for part in ("total", "muon", "adamw"):
            cell = accumulator[role][metric][part]
            cell[0] += float(metrics[f"{metric}_square_sum"][part])
            cell[1] += float(metrics["parameter_count"][part])


def _pooled_rms(
    accumulator: dict[str, dict[str, dict[str, list[float]]]], role: str, metric: str
) -> dict[str, float]:
    return {
        part: rms_from_totals(square_sum, count)
        for part, (square_sum, count) in accumulator[role][metric].items()
    }


def train(
    config: LlmPPOConfig,
    run_dir: Path,
    *,
    backend: Backend | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Train one optimizer route and maintain its complete run artifact set."""
    _validate_config(config, production=backend is None)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    resolved_run_id = run_id or canonical_run_id(config)
    manifest = run_manifest(config, resolved_run_id)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError(f"manifest mismatch in {run_dir}")
    else:
        _write_json(manifest_path, manifest, exclusive=True)
    progress_path = run_dir / "progress.jsonl"
    eval_path = run_dir / "eval.jsonl"
    progress_path.write_text("")
    eval_path.write_text("")
    (run_dir / "failure.json").unlink(missing_ok=True)
    (run_dir / "result.json").unlink(missing_ok=True)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    try:
        active_backend = backend or TransformersBackend(config)
        actor_optimizer = build_optimizer(
            active_backend.actor,
            config.actor_optimizer,
            config.actor_adam_lr,
            config.actor_muon_lr,
        )
        critic_optimizer = build_optimizer(
            active_backend.critic,
            config.critic_optimizer,
            config.critic_adam_lr,
            config.critic_muon_lr,
        )
        evaluations: list[dict[str, float | int | str]] = []
        rms_history = _new_rms_accumulator()

        def record_eval(update: int) -> None:
            metrics = active_backend.evaluate()
            reward_mean = float(metrics["reward_mean"])
            reward_std = float(metrics["reward_std"])
            evaluator_reward_mean = float(metrics["evaluator_reward_mean"])
            evaluator_reward_std = float(metrics["evaluator_reward_std"])
            if not all(
                math.isfinite(value)
                for value in (
                    reward_mean,
                    reward_std,
                    evaluator_reward_mean,
                    evaluator_reward_std,
                )
            ):
                raise RuntimeError("non-finite fixed-evaluation reward metrics")
            record = {
                "run_id": resolved_run_id,
                "update": update,
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "evaluator_reward_mean": evaluator_reward_mean,
                "evaluator_reward_std": evaluator_reward_std,
            }
            evaluations.append(record)
            _append_jsonl(eval_path, record)

        record_eval(0)
        for update in range(1, config.total_updates + 1):
            rollout = active_backend.collect_rollout(update)
            update_rms = _new_rms_accumulator()
            epoch_metrics: dict[str, list[float]] = {
                key: []
                for key in (
                    "actor_loss",
                    "critic_loss",
                    "approx_kl",
                    "clip_fraction",
                    "sampled_token_nll",
                    "value_loss",
                    "value_mean",
                    "explained_variance",
                    "td_error_rms",
                    "terminal_reward",
                    "reference_kl",
                )
            }
            for _ in range(config.ppo_epochs):
                actor_loss, critic_loss, loss_metrics = active_backend.losses(rollout)
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_step = actor_optimizer.step_with_metrics(config.max_grad_norm)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_step = critic_optimizer.step_with_metrics(config.max_grad_norm)
                for accumulator in (update_rms, rms_history):
                    _accumulate_step(accumulator, "actor", actor_step)
                    _accumulate_step(accumulator, "critic", critic_step)
                epoch_metrics["actor_loss"].append(float(actor_loss.detach()))
                epoch_metrics["critic_loss"].append(float(critic_loss.detach()))
                for key in epoch_metrics.keys() - {"actor_loss", "critic_loss"}:
                    epoch_metrics[key].append(float(loss_metrics.get(key, 0.0)))
            _append_jsonl(
                progress_path,
                {
                    "run_id": resolved_run_id,
                    "update": update,
                    "ppo_epochs": config.ppo_epochs,
                    **{key: sum(values) / len(values) for key, values in epoch_metrics.items()},
                    "actor_update_rms": _pooled_rms(update_rms, "actor", "update"),
                    "actor_grad_rms": _pooled_rms(update_rms, "actor", "grad"),
                    "critic_update_rms": _pooled_rms(update_rms, "critic", "update"),
                    "critic_grad_rms": _pooled_rms(update_rms, "critic", "grad"),
                },
            )
            if update % config.eval_every == 0:
                record_eval(update)
            if config.checkpoint_every and update % config.checkpoint_every == 0:
                active_backend.save_checkpoint(checkpoints_dir / f"update-{update:06d}.pt", update)
        if evaluations[-1]["update"] != config.total_updates:
            record_eval(config.total_updates)
        auc_points = [
            (int(record["update"]), float(record["reward_mean"])) for record in evaluations
        ]
        evaluator_auc_points = [
            (int(record["update"]), float(record["evaluator_reward_mean"]))
            for record in evaluations
        ]
        result: dict[str, Any] = {
            "status": "complete",
            "run_id": resolved_run_id,
            "total_updates": config.total_updates,
            "final_eval_reward": evaluations[-1]["reward_mean"],
            "final_eval_reward_std": evaluations[-1]["reward_std"],
            "eval_reward_auc": normalized_trapezoid_auc(auc_points),
            "final_evaluator_reward": evaluations[-1]["evaluator_reward_mean"],
            "final_evaluator_reward_std": evaluations[-1]["evaluator_reward_std"],
            "evaluator_reward_auc": normalized_trapezoid_auc(evaluator_auc_points),
            "actor_optimizer_routing": actor_optimizer.routing,
            "critic_optimizer_routing": critic_optimizer.routing,
            "runtime": runtime_metadata(),
            "model_revisions": manifest["model_revisions"],
        }
        for role in ("actor", "critic"):
            for metric in ("update", "grad"):
                for part, value in _pooled_rms(rms_history, role, metric).items():
                    result[f"{role}_{metric}_rms_{part}_mean"] = value
        _write_json(run_dir / "result.json", result)
        return result
    except BaseException as error:
        _write_json(
            run_dir / "failure.json",
            {
                "status": "failed",
                "run_id": resolved_run_id,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--train-prompts-path", type=Path, required=True)
    parser.add_argument("--eval-prompts-path", type=Path, required=True)
    parser.add_argument("--actor-model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--actor-model-revision", required=True)
    parser.add_argument("--reference-model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--reference-model-revision", required=True)
    parser.add_argument("--critic-model-id", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--critic-model-revision", required=True)
    parser.add_argument(
        "--reward-model-id",
        default="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )
    parser.add_argument("--reward-model-revision", required=True)
    parser.add_argument("--reward-positive-label", type=int, default=1)
    parser.add_argument(
        "--evaluator-model-id",
        default="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    )
    parser.add_argument("--evaluator-model-revision", required=True)
    parser.add_argument("--evaluator-positive-label", type=int, default=1)
    parser.add_argument("--actor-optimizer", choices=OPTIMIZER_KINDS, required=True)
    parser.add_argument("--critic-optimizer", choices=OPTIMIZER_KINDS, required=True)
    parser.add_argument("--actor-adam-lr", type=float, default=3e-6)
    parser.add_argument("--actor-muon-lr", type=float, default=3e-4)
    parser.add_argument("--critic-adam-lr", type=float, default=3e-6)
    parser.add_argument("--critic-muon-lr", type=float, default=3e-4)
    parser.add_argument("--total-updates", type=int, default=100)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = LlmPPOConfig(
        train_prompts_path=args.train_prompts_path,
        eval_prompts_path=args.eval_prompts_path,
        actor_model_id=args.actor_model_id,
        actor_model_revision=args.actor_model_revision,
        reference_model_id=args.reference_model_id,
        reference_model_revision=args.reference_model_revision,
        critic_model_id=args.critic_model_id,
        critic_model_revision=args.critic_model_revision,
        reward_model_id=args.reward_model_id,
        reward_model_revision=args.reward_model_revision,
        reward_positive_label=args.reward_positive_label,
        evaluator_model_id=args.evaluator_model_id,
        evaluator_model_revision=args.evaluator_model_revision,
        evaluator_positive_label=args.evaluator_positive_label,
        actor_optimizer=args.actor_optimizer,
        critic_optimizer=args.critic_optimizer,
        actor_adam_lr=args.actor_adam_lr,
        actor_muon_lr=args.actor_muon_lr,
        critic_adam_lr=args.critic_adam_lr,
        critic_muon_lr=args.critic_muon_lr,
        total_updates=args.total_updates,
        ppo_epochs=args.ppo_epochs,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        batch_size=args.batch_size,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        clip_coef=args.clip_coef,
        kl_coef=args.kl_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
    )
    print(json.dumps(train(config, args.run_dir, run_id=args.run_id), sort_keys=True))


if __name__ == "__main__":
    main()

