#!/usr/bin/env python3
"""Fail-closed collector and gate for the preregistered actor screen."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from cross_dataset_screen import (
    BASELINE_ROUTE,
    DATASETS,
    MAX_STEPS,
    ROUTES,
    SEED,
    ContractError,
    file_sha256,
    validate_gate_metrics,
    validate_screen_metrics,
)


def _one(root: Path, filename: str) -> Path:
    paths = list(root.rglob(filename))
    if len(paths) != 1:
        raise ContractError(f"expected one {filename} below {root}, found {len(paths)}")
    return paths[0]


def _check_provenance(
    root: Path,
    *,
    phase: str,
    route: str,
    dataset: str,
    source_commit: str,
    manifest_sha256: str,
) -> dict:
    provenance = json.loads(_one(root, "run-provenance.json").read_text())
    expected = {
        "protocol": "cross-dataset-actor-screen-v1",
        "phase": phase,
        "route": route,
        "dataset": dataset,
        "data_source": DATASETS[dataset].source,
        "seed": SEED,
        "source_commit": source_commit,
        "manifest_sha256": manifest_sha256,
        "critic_optimizer": "AdamW",
        "preserve_old_logprobs": True,
        "preserve_reference_logprobs": True,
        "actor_optimizer": {
            "adamw_actor": "AdamW",
            "muon_actor": "MuonWithAuxAdamW",
            "lion_actor": "Lion",
        }[route],
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ContractError(
                f"run provenance mismatch for {key} below {root}: "
                f"{provenance.get(key)!r} != {value!r}"
            )
    allowed = set(expected) | {"actor_learning_rate", "lion_calibration_sha256"}
    if set(provenance) != allowed:
        raise ContractError(f"unexpected provenance fields below {root}: {set(provenance) - allowed}")
    learning_rate = provenance.get("actor_learning_rate")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise ContractError(f"invalid actor learning rate below {root}: {learning_rate!r}")
    calibration = provenance.get("lion_calibration_sha256")
    if route == "lion_actor":
        if not isinstance(calibration, str) or len(calibration) != 64 or any(
            character not in "0123456789abcdef" for character in calibration
        ):
            raise ContractError("invalid Lion calibration provenance")
    elif calibration is not None:
        raise ContractError(f"unexpected Lion calibration on {route}")
    return provenance


def validate_gate(
    gate_root: Path,
    dataset: str,
    source_commit: str,
    manifest_sha256: str,
) -> dict:
    _check_provenance(
        gate_root,
        phase="gate",
        route=BASELINE_ROUTE,
        dataset=dataset,
        source_commit=source_commit,
        manifest_sha256=manifest_sha256,
    )
    return validate_gate_metrics(_one(gate_root, "metrics.jsonl"), DATASETS[dataset].source)


def build_result(
    run_root: Path,
    dataset: str,
    seed: int,
    source_commit: str,
    manifest_path: Path,
) -> dict:
    if seed != SEED:
        raise ContractError(f"screen is preregistered for seed {SEED}, got {seed}")
    if dataset not in DATASETS:
        raise ContractError(f"unsupported dataset: {dataset!r}")
    manifest = json.loads(manifest_path.read_text())
    spec = DATASETS[dataset]
    expected_manifest = {
        "dataset": dataset,
        "data_source": spec.source,
        "repository": spec.repository,
        "config": spec.config,
        "revision": spec.revision,
        "seed": SEED,
        "train_split": spec.train_split,
        "validation_split": spec.validation_split,
    }
    for key, value in expected_manifest.items():
        if manifest.get(key) != value:
            raise ContractError(f"manifest mismatch for {key}: {manifest.get(key)!r} != {value!r}")
    manifest_sha256 = file_sha256(manifest_path)
    gate = validate_gate(run_root / "gate", dataset, source_commit, manifest_sha256)
    route_results = {}
    for route in ROUTES:
        route_root = run_root / route
        provenance = _check_provenance(
            route_root,
            phase="screen",
            route=route,
            dataset=dataset,
            source_commit=source_commit,
            manifest_sha256=manifest_sha256,
        )
        route_result = validate_screen_metrics(
            _one(route_root, "metrics.jsonl"), route, spec.source
        )
        route_result["actor_optimizer"] = provenance["actor_optimizer"]
        route_result["actor_learning_rate"] = provenance["actor_learning_rate"]
        route_result["lion_calibration_sha256"] = provenance["lion_calibration_sha256"]
        route_results[route] = route_result
    return {
        "protocol": "cross-dataset-actor-screen-v1",
        "dataset": dataset,
        "data_source": spec.source,
        "seed": SEED,
        "source_commit": source_commit,
        "manifest_sha256": manifest_sha256,
        "dataset_revision": spec.revision,
        "max_steps": MAX_STEPS,
        "validation_steps": [0, 25, 50],
        "critic_optimizer": "AdamW",
        "gate": gate,
        "routes": route_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("seed", type=int)
    parser.add_argument("source_commit")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    if args.seed != SEED:
        raise ContractError(f"screen is preregistered for seed {SEED}")
    manifest_sha256 = file_sha256(args.manifest)
    if args.gate_only:
        result = validate_gate(
            args.run_root / "gate", args.dataset, args.source_commit, manifest_sha256
        )
        print("RL_MUON_CROSS_DATASET_GATE " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    result = build_result(
        args.run_root, args.dataset, args.seed, args.source_commit, args.manifest
    )
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
    (args.run_root / "result.json").write_text(encoded + "\n")
    print("RL_MUON_CROSS_DATASET_RESULT " + encoded, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
