#!/usr/bin/env python3
"""Remove only the known failed full-run attempts from a reused campaign."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


campaign_root = Path(os.environ["RL_MUON_CAMPAIGN_ROOT"]).resolve()
attempt = os.environ.get("RL_MUON_CLEANUP_ATTEMPT", "padding-fallback")
if not attempt or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in attempt):
    raise SystemExit(f"invalid cleanup attempt: {attempt!r}")

for seed in range(3):
    run_root = campaign_root / f"full_seed{seed}_{attempt}"
    if not run_root.exists():
        print(f"absent: {run_root}", flush=True)
        continue
    if run_root.resolve().parent != campaign_root:
        raise RuntimeError(f"refusing path outside campaign: {run_root}")
    status_path = run_root / "status.json"
    status = json.loads(status_path.read_text())
    expected = {"state": "failed", "phase": "full", "seed": seed}
    observed = {key: status.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"refusing non-failed run {run_root}: {observed}")
    shutil.rmtree(run_root)
    print(f"removed: {run_root}", flush=True)