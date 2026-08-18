#!/usr/bin/env python3
"""Read one progress snapshot from a shared Cloud campaign run root."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def read_last_jsonl(path: Path) -> dict:
    last = None
    with path.open("r", encoding="utf-8") as stream:
        for raw in stream:
            raw = raw.strip()
            if raw:
                last = json.loads(raw)
    if last is None:
        raise RuntimeError(f"empty metrics file: {path}")
    return last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
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

    route_names = ("lion_actor_adam_critic", "lion_actor_muon_critic")
    for route in route_names:
        route_root = root / route
        metrics_paths = list(route_root.rglob("metrics.jsonl")) if route_root.is_dir() else []
        if not metrics_paths:
            routes[route] = {"state": "not_started"}
            continue
        if len(metrics_paths) != 1:
            raise RuntimeError(f"expected one metrics.jsonl for {route}, found {len(metrics_paths)}")
        metrics = metrics_paths[0]
        row = read_last_jsonl(metrics)
        stat = metrics.stat()
        data = row.get("data", {})
        interesting = {
            key: value
            for key, value in data.items()
            if any(token in key.lower() for token in ("reward", "acc", "kl", "clip", "grad", "loss"))
        }
        routes[route] = {
            "state": "active_or_complete",
            "step": int(row["step"]),
            "metrics_rows": sum(1 for line in metrics.read_text().splitlines() if line.strip()),
            "age_seconds": max(0.0, time.time() - stat.st_mtime),
            "metrics": interesting,
        }

    print("RL_MUON_PROGRESS " + json.dumps(snapshot, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
