#!/usr/bin/env python3
"""Read one progress snapshot from a shared Cloud campaign run root."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for raw in stream:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    if not rows:
        raise RuntimeError(f"empty metrics file: {path}")
    rows.sort(key=lambda row: int(row["step"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--route", action="append")
    args = parser.parse_args()
    root = Path(args.run_root)
    if not root.is_dir():
        raise SystemExit(f"run root not found: {root}")

    routes: dict[str, dict[str, object]] = {}
    snapshot: dict[str, object] = {
        "run_root": str(root),
        "probe_time": time.time(),
        "status": None,
        "routes": routes,
    }
    status_path = root / "status.json"
    if status_path.is_file():
        snapshot["status"] = json.loads(status_path.read_text())

    route_names = tuple(args.route or ("lion_actor_adam_critic", "lion_actor_muon_critic"))
    for route in route_names:
        route_root = root / route
        metrics_paths = list(route_root.rglob("metrics.jsonl")) if route_root.is_dir() else []
        if not metrics_paths:
            routes[route] = {"state": "not_started"}
            continue
        if len(metrics_paths) != 1:
            raise RuntimeError(f"expected one metrics.jsonl for {route}, found {len(metrics_paths)}")
        metrics = metrics_paths[0]
        rows = read_jsonl(metrics)
        row = rows[-1]
        stat = metrics.stat()
        data = row.get("data", {})
        interesting = {
            key: value
            for key, value in data.items()
            if any(token in key.lower() for token in ("reward", "acc", "kl", "clip", "grad", "loss"))
        }
        validation_points = []
        for candidate_row in rows:
            candidates = {
                key: value
                for key, value in candidate_row.get("data", {}).items()
                if isinstance(key, str)
                and key.startswith("val-core/")
                and ("/acc/mean@" in key or "/reward/mean@" in key)
                and isinstance(value, (int, float))
            }
            if len(candidates) == 1:
                validation_points.append(
                    {"step": int(candidate_row["step"]), "value": float(next(iter(candidates.values())))}
                )
        validation = None
        if validation_points:
            if len(validation_points) == 1:
                auc = validation_points[0]["value"]
            else:
                area = sum(
                    (right["step"] - left["step"])
                    * (left["value"] + right["value"]) / 2
                    for left, right in zip(validation_points, validation_points[1:])
                )
                auc = area / (validation_points[-1]["step"] - validation_points[0]["step"])
            validation = {
                "points": len(validation_points),
                "latest": validation_points[-1],
                "auc": auc,
            }
        routes[route] = {
            "state": "active_or_complete",
            "step": int(row["step"]),
            "metrics_rows": len(rows),
            "age_seconds": max(0.0, time.time() - stat.st_mtime),
            "metrics": interesting,
            "validation": validation,
        }

    print("RL_MUON_PROGRESS " + json.dumps(snapshot, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
