#!/usr/bin/env bash
set -u
python - <<'PY'
import json
from pathlib import Path

root = Path("/home/jovyan/rl_muon/h7_online_cloud_r1")
payload = {"root": str(root), "seeds": {}}
for seed in range(3):
    seed_root = root / f"seed_{seed}"
    seed_payload = {"status": None, "exit": None, "routes": {}}
    if (seed_root / "status.json").is_file():
        seed_payload["status"] = json.loads((seed_root / "status.json").read_text())
    if (seed_root / "exit").is_file():
        seed_payload["exit"] = (seed_root / "exit").read_text().strip()
    for route in ("raw_muon", "own_polar_d01", "own_polar_d1"):
        run_dir = seed_root / route
        route_payload = {}
        for name in ("result.json", "failure.json"):
            path = run_dir / name
            if path.is_file():
                route_payload[name] = json.loads(path.read_text())
        for name in ("progress.jsonl", "eval.jsonl"):
            path = run_dir / name
            if path.is_file():
                route_payload[name] = [
                    json.loads(line) for line in path.read_text().splitlines() if line.strip()
                ]
        seed_payload["routes"][route] = route_payload
    payload["seeds"][str(seed)] = seed_payload
print(json.dumps(payload, sort_keys=True, allow_nan=False), flush=True)
PY
exit 0
