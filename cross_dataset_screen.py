#!/usr/bin/env python3
"""Deterministic data and result contracts for the cross-dataset PPO screen."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SEED = 0
MAX_STEPS = 50
VALIDATION_STEPS = (0, 25, 50)
ROUTES = ("adamw_actor", "muon_actor", "lion_actor")
BASELINE_ROUTE = "adamw_actor"
SOURCES = {
    "svamp": "cross_dataset/svamp",
    "arc_easy": "cross_dataset/arc_easy",
}
REQUIRED_SAFETY_METRICS = (
    "actor/ppo_kl",
    "actor/pg_clipfrac",
    "critic/vf_clipfrac",
)


class ContractError(RuntimeError):
    """A fail-closed screen contract violation."""


class AmbiguousAnswerError(ContractError):
    """More than one final answer or a non-terminal final answer was found."""


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    source: str
    repository: str
    config: str | None
    revision: str
    train_split: str
    validation_split: str
    expected_train_rows: int
    expected_validation_rows: int


DATASETS = {
    "svamp": DatasetSpec(
        key="svamp",
        source=SOURCES["svamp"],
        repository="ChilleD/SVAMP",
        config=None,
        revision="5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef",
        train_split="train",
        validation_split="test",
        expected_train_rows=700,
        expected_validation_rows=300,
    ),
    "arc_easy": DatasetSpec(
        key="arc_easy",
        source=SOURCES["arc_easy"],
        repository="allenai/ai2_arc",
        config="ARC-Easy",
        revision="210d026faf9955653af8916fad021475a3f00453",
        train_split="train",
        validation_split="validation",
        expected_train_rows=2251,
        expected_validation_rows=570,
    ),
}

_FINAL_LINE = re.compile(r"(?m)^Final answer:[ \t]*(.*?)[ \t]*$")
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[+-]?(?:\d+(?:\.\d+)?|\.\d+))?")
_ARC_LABEL = re.compile(r"[A-E]")


def _terminal_answer_text(solution: str) -> str | None:
    if not isinstance(solution, str):
        raise ContractError("solution must be a string")
    matches = list(_FINAL_LINE.finditer(solution))
    if not matches:
        return None
    if len(matches) != 1:
        raise AmbiguousAnswerError("multiple 'Final answer:' lines")
    match = matches[0]
    if solution[match.end():].strip():
        raise AmbiguousAnswerError("'Final answer:' must be the last non-empty line")
    answer = match.group(1).strip()
    if not answer:
        return None
    return answer


def parse_svamp_answer(solution: str) -> Fraction | None:
    """Parse exactly one terminal numeric answer, without float tolerance."""
    answer = _terminal_answer_text(solution)
    if answer is None:
        return None
    if _NUMBER.fullmatch(answer) is None:
        raise AmbiguousAnswerError(f"invalid SVAMP final answer: {answer!r}")
    try:
        value = Fraction(answer)
    except (ValueError, ZeroDivisionError) as error:
        raise AmbiguousAnswerError(f"invalid SVAMP final answer: {answer!r}") from error
    return value


def parse_arc_easy_answer(solution: str) -> str | None:
    """Parse exactly one terminal upper-case ARC choice label."""
    answer = _terminal_answer_text(solution)
    if answer is None:
        return None
    if _ARC_LABEL.fullmatch(answer) is None:
        raise AmbiguousAnswerError(f"invalid ARC-Easy final answer: {answer!r}")
    return answer


def score_answer(data_source: str, solution: str, ground_truth: str) -> float:
    """Exact custom reward entrypoint; contract violations raise and stop PPO."""
    if data_source == SOURCES["svamp"]:
        predicted = parse_svamp_answer(solution)
        if _NUMBER.fullmatch(ground_truth) is None:
            raise ContractError(f"invalid SVAMP ground truth: {ground_truth!r}")
        expected = Fraction(ground_truth)
    elif data_source == SOURCES["arc_easy"]:
        predicted = parse_arc_easy_answer(solution)
        if _ARC_LABEL.fullmatch(ground_truth) is None:
            raise ContractError(f"invalid ARC-Easy ground truth: {ground_truth!r}")
        expected = ground_truth
    else:
        raise ContractError(f"unexpected data source: {data_source!r}")
    return float(predicted is not None and predicted == expected)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Mapping[str, Any] | None = None,
) -> float:
    """VERL-compatible custom reward function."""
    if extra_info is not None:
        recorded_source = extra_info.get("data_source")
        if recorded_source != data_source:
            raise ContractError(
                f"reward source mismatch: {recorded_source!r} != {data_source!r}"
            )
    return score_answer(data_source, solution_str, ground_truth)


def _prompt(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def adapt_svamp(record: Mapping[str, Any], split: str, index: int) -> dict[str, Any]:
    record_id = str(record.get("ID", "")).strip()
    body = str(record.get("Body", "")).strip()
    question = str(record.get("Question", "")).strip()
    answer = str(record.get("Answer", "")).strip()
    if not record_id or not body or not question or _NUMBER.fullmatch(answer) is None:
        raise ContractError(f"invalid SVAMP record at {split}[{index}]")
    # Canonicalize numerically equivalent decimals (for example, 3.0 -> 3).
    canonical = str(Fraction(answer))
    content = (
        f"{body}\n{question}\n\n"
        "Solve the problem. End with exactly one line in the form "
        "'Final answer: <number>'."
    )
    return {
        "data_source": SOURCES["svamp"],
        "prompt": _prompt(content),
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": canonical},
        "extra_info": {
            "data_source": SOURCES["svamp"],
            "dataset": "svamp",
            "split": split,
            "index": index,
            "id": record_id,
        },
    }


def adapt_arc_easy(record: Mapping[str, Any], split: str, index: int) -> dict[str, Any]:
    record_id = str(record.get("id", "")).strip()
    question = str(record.get("question", "")).strip()
    choices = record.get("choices")
    raw_answer = str(record.get("answerKey", "")).strip()
    if not isinstance(choices, Mapping):
        raise ContractError(f"invalid ARC-Easy choices at {split}[{index}]")
    labels = [str(value).strip() for value in choices.get("label", ())]
    texts = [str(value).strip() for value in choices.get("text", ())]
    if (
        not record_id
        or not question
        or not labels
        or len(labels) != len(texts)
        or len(set(labels)) != len(labels)
        or not 2 <= len(labels) <= 5
        or any(not label or not text for label, text in zip(labels, texts))
        or raw_answer not in labels
    ):
        raise ContractError(f"invalid ARC-Easy record at {split}[{index}]")
    canonical_labels = list("ABCDE"[:len(labels)])
    answer = canonical_labels[labels.index(raw_answer)]
    rendered = "\n".join(
        f"{label}. {text}" for label, text in zip(canonical_labels, texts)
    )
    content = (
        f"{question}\n\n{rendered}\n\n"
        "Choose the correct option. End with exactly one line in the form "
        "'Final answer: <letter>'."
    )
    return {
        "data_source": SOURCES["arc_easy"],
        "prompt": _prompt(content),
        "ability": "science",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "data_source": SOURCES["arc_easy"],
            "dataset": "arc_easy",
            "split": split,
            "index": index,
            "id": record_id,
            "choice_labels": canonical_labels,
            "original_choice_labels": labels,
        },
    }


def adapt_records(dataset: str, records: Iterable[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    if dataset not in DATASETS:
        raise ContractError(f"unsupported dataset: {dataset!r}")
    adapter = adapt_svamp if dataset == "svamp" else adapt_arc_easy
    adapted = [adapter(record, split, index) for index, record in enumerate(records)]
    adapted.sort(key=lambda row: row["extra_info"]["id"])
    ids = [row["extra_info"]["id"] for row in adapted]
    if len(ids) != len(set(ids)):
        raise ContractError(f"duplicate {dataset} IDs in {split}")
    # Reindex after sorting so ordering and provenance are deterministic.
    for index, row in enumerate(adapted):
        row["extra_info"]["index"] = index
    return adapted


def validate_held_out(train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]) -> None:
    train_ids = {row["extra_info"]["id"] for row in train}
    validation_ids = {row["extra_info"]["id"] for row in validation}
    overlap = train_ids & validation_ids
    if overlap:
        raise ContractError(f"train/validation ID overlap: {sorted(overlap)[:5]}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_metric_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            step = row["step"]
            data = row["data"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ContractError(f"invalid metrics row {path}:{line_number}") from error
        if type(step) is not int or step < 0 or not isinstance(data, dict):
            raise ContractError(f"invalid metrics row {path}:{line_number}")
        rows.append({"step": step, "data": data})
    if not rows:
        raise ContractError(f"empty metrics file: {path}")
    return rows


def _finite_metric(data: Mapping[str, Any], key: str, context: str) -> float:
    if key not in data:
        raise ContractError(f"missing metric {key!r} for {context}")
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContractError(f"non-finite metric {key!r} for {context}: {value!r}")
    return float(value)


def validation_metric(data: Mapping[str, Any], source: str, context: str) -> tuple[str, float]:
    candidates = []
    permitted = {
        f"val-core/{source}/reward/mean@1",
        f"val-core/{source}/acc/mean@1",
    }
    for key, value in data.items():
        if not isinstance(key, str) or not key.startswith("val-core/"):
            continue
        if key not in permitted:
            raise ContractError(f"validation source mismatch for {context}: {key!r}")
        candidates.append((key, _finite_metric(data, key, context)))
    if len(candidates) != 1:
        raise ContractError(f"ambiguous or absent validation metric for {context}: {candidates!r}")
    return candidates[0]


def validate_gate_metrics(path: Path, source: str) -> dict[str, Any]:
    rows = read_metric_rows(path)
    by_step: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_step.setdefault(row["step"], []).append(row["data"])
    if 0 not in by_step or 1 not in by_step:
        raise ContractError("gate requires real metric rows at steps 0 and 1")
    validation = [validation_metric(data, source, "gate step 0") for data in by_step[0] if any(
        isinstance(key, str) and key.startswith("val-core/") for key in data
    )]
    if len(validation) != 1:
        raise ContractError("gate requires exactly one baseline validation at step 0")
    metric_name, baseline_reward = validation[0]
    if baseline_reward <= 0:
        raise ContractError("AdamW baseline validation reward must be positive")
    safety_rows = [data for data in by_step[1] if all(key in data for key in REQUIRED_SAFETY_METRICS)]
    if len(safety_rows) != 1:
        raise ContractError("gate requires exactly one step-1 safety row")
    safety = {
        key: _finite_metric(safety_rows[0], key, "gate step 1")
        for key in REQUIRED_SAFETY_METRICS
    }
    return {"validation_metric": metric_name, "baseline_reward": baseline_reward, "safety": safety}


def validate_screen_metrics(path: Path, route: str, source: str) -> dict[str, Any]:
    if route not in ROUTES:
        raise ContractError(f"unexpected route: {route!r}")
    rows = read_metric_rows(path)
    if max(row["step"] for row in rows) != MAX_STEPS:
        raise ContractError(f"route {route} did not stop exactly at step {MAX_STEPS}")
    validation_rows = [
        row
        for row in rows
        if any(isinstance(key, str) and key.startswith("val-core/") for key in row["data"])
    ]
    if [row["step"] for row in validation_rows] != list(VALIDATION_STEPS):
        raise ContractError(
            f"route {route} validation schedule must be exactly {VALIDATION_STEPS}"
        )
    points = []
    metric_name = None
    for expected_step in VALIDATION_STEPS:
        candidates = []
        for row in rows:
            if row["step"] != expected_step:
                continue
            if any(isinstance(key, str) and key.startswith("val-core/") for key in row["data"]):
                candidates.append(validation_metric(row["data"], source, f"{route} step {expected_step}"))
        if len(candidates) != 1:
            raise ContractError(f"route {route} requires one validation at step {expected_step}")
        name, value = candidates[0]
        if metric_name is None:
            metric_name = name
        elif name != metric_name:
            raise ContractError(f"validation metric changed for route {route}")
        points.append({"step": expected_step, "value": value})
    for row in rows:
        for key, value in row["data"].items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                raise ContractError(f"non-finite metric for {route} step {row['step']}: {key}")
    step_one = [row["data"] for row in rows if row["step"] == 1 and all(
        key in row["data"] for key in REQUIRED_SAFETY_METRICS
    )]
    if len(step_one) != 1:
        raise ContractError(f"route {route} requires one step-1 safety row")
    safety = {key: _finite_metric(step_one[0], key, f"{route} step 1") for key in REQUIRED_SAFETY_METRICS}
    return {
        "route": route,
        "terminal_step": MAX_STEPS,
        "validation_metric": metric_name,
        "validation_points": points,
        "step_1_safety": safety,
    }
