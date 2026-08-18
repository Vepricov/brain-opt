#!/usr/bin/env python3
"""Fail-closed collector and gate for the preregistered actor screen."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from cross_dataset_screen import (
    DATASETS,
    MAX_STEPS,
    ROUTES,
    SEED,
    ContractError,
    file_sha256,
    validate_gate_metrics,
    validate_screen_metrics,
)

CALIBRATION_METRIC = "exact_full_categorical_KL_old_to_new_on_occupied_response_states"
ROUTE_DESCRIPTIONS = {
    "adamw_actor": "AdamW on all actor parameters",
    "muon_actor": "Muon on hidden attention/MLP matrices plus AdamW auxiliaries",
    "lion_actor": "Lion on all actor parameters; strict no-Adam actor route",
}


def _one(root: Path, filename: str) -> Path:
    paths = list(root.rglob(filename))
    if len(paths) != 1:
        raise ContractError(f"expected one {filename} below {root}, found {len(paths)}")
    return paths[0]


def validate_manifest(manifest_path: Path, dataset: str) -> tuple[dict, str]:
    """Validate the exact prepared-data contract and current artifact bytes."""
    if dataset not in DATASETS:
        raise ContractError(f"unsupported dataset: {dataset!r}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid dataset manifest: {manifest_path}") from error
    spec = DATASETS[dataset]
    expected = {
        "schema_version": 1,
        "dataset": dataset,
        "data_source": spec.source,
        "repository": spec.repository,
        "config": spec.config,
        "revision": spec.revision,
        "seed": SEED,
        "train_split": spec.train_split,
        "validation_split": spec.validation_split,
    }
    if not isinstance(manifest, dict) or set(manifest) != set(expected) | {"files"}:
        raise ContractError(
            "manifest schema must match the prepared-data contract exactly"
        )
    for key, value in expected.items():
        if manifest[key] != value:
            raise ContractError(
                f"manifest mismatch for {key}: {manifest[key]!r} != {value!r}"
            )
    files = manifest["files"]
    expected_rows = {
        "train.parquet": spec.expected_train_rows,
        "validation.parquet": spec.expected_validation_rows,
    }
    if not isinstance(files, dict) or set(files) != set(expected_rows):
        raise ContractError(
            "manifest file entries must be exactly train.parquet and validation.parquet"
        )
    for filename, rows in expected_rows.items():
        metadata = files[filename]
        if not isinstance(metadata, dict) or set(metadata) != {"rows", "sha256"}:
            raise ContractError(f"manifest schema invalid for {filename}")
        if type(metadata["rows"]) is not int or metadata["rows"] != rows:
            raise ContractError(f"dataset row-count mismatch: {filename}")
        expected_hash = metadata["sha256"]
        if (
            not isinstance(expected_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
        ):
            raise ContractError(f"invalid dataset hash in manifest: {filename}")
        artifact = manifest_path.parent / filename
        if not artifact.is_file():
            raise ContractError(f"missing dataset artifact: {artifact}")
        observed_hash = file_sha256(artifact)
        if observed_hash != expected_hash:
            raise ContractError(f"dataset hash mismatch: {filename}")
    return manifest, file_sha256(manifest_path)


def _check_provenance(
    root: Path,
    *,
    phase: str,
    route: str,
    dataset: str,
    source_commit: str,
    manifest_sha256: str,
    calibration_sha256: str,
    expected_actor_learning_rate: float,
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
        "actor_route_description": ROUTE_DESCRIPTIONS[route],
        "actor_parameter_routing": {
            "adamw_actor": "all_actor_parameters",
            "muon_actor": "hidden_attention_mlp_matrices_muon;all_auxiliaries_adamw",
            "lion_actor": "all_actor_parameters_lion_no_adam",
        }[route],
        "actor_uses_adamw_auxiliaries": route == "muon_actor",
        "calibration_sha256": calibration_sha256,
        "calibration_metric": CALIBRATION_METRIC,
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ContractError(
                f"run provenance mismatch for {key} below {root}: {provenance.get(key)!r} != {value!r}"
            )
    allowed = set(expected) | {"actor_learning_rate"}
    if set(provenance) != allowed:
        raise ContractError(
            f"unexpected provenance fields below {root}: {set(provenance) - allowed}"
        )
    learning_rate = provenance.get("actor_learning_rate")
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise ContractError(
            f"invalid actor learning rate below {root}: {learning_rate!r}"
        )
    if float(learning_rate) != float(expected_actor_learning_rate):
        raise ContractError(
            f"calibrated actor learning rate mismatch below {root}: {learning_rate!r} != {expected_actor_learning_rate!r}"
        )
    return provenance


def validate_calibration(
    path: Path,
    dataset: str,
    source_commit: str,
    manifest_sha256: str,
    train_sha256: str,
) -> tuple[dict, str]:
    calibration_sha256 = file_sha256(path)
    calibration = json.loads(path.read_text())
    expected = {
        "schema_version": 2,
        "status": "complete",
        "protocol": "per_dataset_same_frozen_batch_one_production_equivalent_ppo_update",
        "dataset": dataset,
        "data_source": DATASETS[dataset].source,
        "seed": SEED,
        "source_commit": source_commit,
        "manifest_sha256": manifest_sha256,
        "calibration_metric": CALIBRATION_METRIC,
        "kl_direction": "old_policy_to_updated_policy",
        "model_compute_device": "cuda",
    }
    for key, value in expected.items():
        if calibration.get(key) != value:
            raise ContractError(
                f"calibration mismatch for {key}: {calibration.get(key)!r} != {value!r}"
            )
    identities = calibration.get("identities", {})
    if identities.get("train_file_sha256") != train_sha256:
        raise ContractError("calibration train-file hash mismatch")
    for key in ("rollout_sha256", "old_logprobs_sha256", "reference_logprobs_sha256"):
        value = identities.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ContractError(f"invalid calibration identity: {key}")
    adam = calibration.get("adam_actual_full_categorical_kl", {})
    for route in ("muon", "lion"):
        lr = calibration.get(f"chosen_{route}_learning_rate")
        metrics = calibration.get(f"chosen_{route}_actual_full_categorical_kl", {})
        if (
            isinstance(lr, bool)
            or not isinstance(lr, (int, float))
            or not math.isfinite(lr)
            or lr <= 0
        ):
            raise ContractError(f"invalid calibrated {route} learning rate")
        for metric in ("mean", "q95"):
            target = adam.get(metric)
            observed = metrics.get(metric)
            relative_error = metrics.get(f"{metric}_relative_error")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in (target, observed, relative_error)
            ):
                raise ContractError(f"invalid {route} categorical KL {metric}")
            if target <= 0 or observed < 0:
                raise ContractError(
                    f"non-positive AdamW dose or negative {route} categorical KL {metric}"
                )
            computed_relative_error = abs(observed - target) / target
            if not math.isclose(
                relative_error, computed_relative_error, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ContractError(
                    f"incorrect {route} categorical KL {metric} relative error"
                )
            if relative_error > 0.10:
                raise ContractError(
                    f"{route} categorical KL {metric} does not match AdamW within 10%"
                )
        if metrics.get("occupied_states") != adam.get(
            "occupied_states"
        ) or not metrics.get("occupied_states"):
            raise ContractError(f"{route} occupied-state count mismatch")
    if calibration.get("gates", {}).get("passed") is not True:
        raise ContractError("calibration did not pass jointly")
    return calibration, calibration_sha256


def validate_gate(
    gate_root: Path,
    dataset: str,
    source_commit: str,
    manifest_sha256: str,
    calibration_sha256: str,
    route: str,
    expected_actor_learning_rate: float,
) -> dict:
    _check_provenance(
        gate_root,
        phase="gate",
        route=route,
        dataset=dataset,
        source_commit=source_commit,
        manifest_sha256=manifest_sha256,
        calibration_sha256=calibration_sha256,
        expected_actor_learning_rate=expected_actor_learning_rate,
    )
    return validate_gate_metrics(
        _one(gate_root, "metrics.jsonl"), DATASETS[dataset].source
    )


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
    manifest, manifest_sha256 = validate_manifest(manifest_path, dataset)
    spec = DATASETS[dataset]
    train_path = manifest_path.parent / "train.parquet"
    if not train_path.is_file() or file_sha256(train_path) != manifest.get(
        "files", {}
    ).get("train.parquet", {}).get("sha256"):
        raise ContractError("manifest-bound train parquet is missing or changed")
    calibration, calibration_sha256 = validate_calibration(
        run_root / "calibration.json",
        dataset,
        source_commit,
        manifest_sha256,
        file_sha256(train_path),
    )
    gates = {
        route: validate_gate(
            run_root / "gates" / route,
            dataset,
            source_commit,
            manifest_sha256,
            calibration_sha256,
            route,
            1e-6
            if route == "adamw_actor"
            else calibration[f"chosen_{route.removesuffix('_actor')}_learning_rate"],
        )
        for route in ROUTES
    }
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
            calibration_sha256=calibration_sha256,
            expected_actor_learning_rate=(
                1e-6
                if route == "adamw_actor"
                else calibration[f"chosen_{route.removesuffix('_actor')}_learning_rate"]
            ),
        )
        route_result = validate_screen_metrics(
            _one(route_root, "metrics.jsonl"), route, spec.source
        )
        route_result["actor_optimizer"] = provenance["actor_optimizer"]
        route_result["actor_learning_rate"] = provenance["actor_learning_rate"]
        route_result["calibration_sha256"] = provenance["calibration_sha256"]
        route_result["actor_route_description"] = provenance["actor_route_description"]
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
        "calibration_sha256": calibration_sha256,
        "calibration_metric": calibration["calibration_metric"],
        "gates": gates,
        "routes": route_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("seed", type=int)
    parser.add_argument("source_commit")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--gate-route", choices=ROUTES)
    args = parser.parse_args()
    if args.seed != SEED:
        raise ContractError(f"screen is preregistered for seed {SEED}")
    _, manifest_sha256 = validate_manifest(args.manifest, args.dataset)
    if args.gate_route:
        manifest = json.loads(args.manifest.read_text())
        train_path = args.manifest.parent / "train.parquet"
        if not train_path.is_file() or file_sha256(train_path) != manifest.get(
            "files", {}
        ).get("train.parquet", {}).get("sha256"):
            raise ContractError("manifest-bound train parquet is missing or changed")
        calibration, calibration_sha256 = validate_calibration(
            args.run_root / "calibration.json",
            args.dataset,
            args.source_commit,
            manifest_sha256,
            file_sha256(train_path),
        )
        result = validate_gate(
            args.run_root / "gates" / args.gate_route,
            args.dataset,
            args.source_commit,
            manifest_sha256,
            calibration_sha256,
            args.gate_route,
            (
                1e-6
                if args.gate_route == "adamw_actor"
                else calibration[
                    f"chosen_{args.gate_route.removesuffix('_actor')}_learning_rate"
                ]
            ),
        )
        print(
            "RL_MUON_CROSS_DATASET_GATE " + json.dumps(result, sort_keys=True),
            flush=True,
        )
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
