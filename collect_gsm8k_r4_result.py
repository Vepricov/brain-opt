#!/usr/bin/env python3
"""Build the bound scientific result for one three-route GSM8K run."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROUTES = ("adam_adam", "muon_actor", "muon_critic")
SAFETY_PREFIXES = (
    "actor/", "critic/", "response_length/", "prompt_length/", "perf/",
)


def build_result(
    run_root: Path, phase: str, seed: int, source_commit: str,
    expected_step: int,
) -> dict:
    routes = {}
    for route in ROUTES:
        paths = list((run_root / route).rglob("metrics.jsonl"))
        if len(paths) != 1:
            raise RuntimeError(
                f"expected one metrics file for {route}, found {len(paths)}")
        rows = [json.loads(line) for line in paths[0].read_text().splitlines()
                if line.strip()]
        rows.sort(key=lambda row: int(row["step"]))
        if not rows or int(rows[-1]["step"]) != expected_step:
            raise RuntimeError(f"missing terminal row for {route} at step {expected_step}")
        terminal = rows[-1]
        validation_metric = None
        validation_points = []
        for row in rows:
            candidates = {
                key: value for key, value in row.get("data", {}).items()
                if (isinstance(key, str) and key.startswith("val-core/")
                    and ("/acc/mean@" in key or "/reward/mean@" in key)
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value)))
            }
            if not candidates:
                continue
            if len(candidates) != 1:
                raise RuntimeError(
                    f"ambiguous validation endpoint for {route}: "
                    f"{sorted(candidates)}")
            key, value = next(iter(candidates.items()))
            if validation_metric is None:
                validation_metric = key
            elif key != validation_metric:
                raise RuntimeError(
                    f"validation metric changed for {route}: "
                    f"{validation_metric} -> {key}")
            validation_points.append({"step": int(row["step"]), "value": value})
        if not validation_points or validation_points[-1]["step"] != expected_step:
            raise RuntimeError(
                f"missing terminal validation endpoint for {route} "
                f"at step {expected_step}")
        if len(validation_points) == 1:
            validation_auc = float(validation_points[0]["value"])
        else:
            area = sum(
                (right["step"] - left["step"])
                * (float(left["value"]) + float(right["value"])) / 2
                for left, right in zip(validation_points, validation_points[1:]))
            validation_auc = area / (
                validation_points[-1]["step"] - validation_points[0]["step"])
        terminal_metrics = {
            key: value for key, value in sorted(terminal.get("data", {}).items())
            if (isinstance(key, str) and len(key) <= 128
                and isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)) and key.startswith(SAFETY_PREFIXES))
        }
        if len(terminal_metrics) > 64:
            raise RuntimeError(f"too many terminal safety metrics for {route}")
        routes[route] = {
            "rows": len(rows), "terminal_step": int(terminal["step"]),
            "validation_metric": validation_metric,
            "validation_points": validation_points,
            "final_validation": validation_points[-1]["value"],
            "validation_auc": validation_auc,
            "terminal_metrics": terminal_metrics,
        }
    return {
        "phase": phase, "seed": seed, "source_commit": source_commit,
        "routes": routes,
    }


def main() -> int:
    if len(sys.argv) != 6:
        raise SystemExit("usage: collect_gsm8k_r4_result.py ROOT PHASE SEED COMMIT STEP")
    run_root = Path(sys.argv[1])
    result = build_result(
        run_root, sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]))
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
    (run_root / "result.json").write_text(encoded + "\n")
    print("RL_MUON_RESULT " + encoded, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
