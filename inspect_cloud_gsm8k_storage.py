#!/usr/bin/env python3
"""Report bounded campaign storage state without traversing large run trees."""
from __future__ import annotations

import json
import os
from pathlib import Path


campaign_root = Path(os.environ["RL_MUON_CAMPAIGN_ROOT"]).resolve()
attempt = os.environ.get("RL_MUON_INSPECT_ATTEMPT", "padding-fallback")
if not attempt or any(
    character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    for character in attempt
):
    raise SystemExit(f"invalid inspect attempt: {attempt!r}")

storage = os.statvfs(campaign_root)
print(
    json.dumps(
        {
            "campaign_root": str(campaign_root),
            "free_bytes": storage.f_bavail * storage.f_frsize,
            "total_bytes": storage.f_blocks * storage.f_frsize,
        },
        sort_keys=True,
    ),
    flush=True,
)
for seed in range(3):
    run_root = campaign_root / f"full_seed{seed}_{attempt}"
    status_path = run_root / "status.json"
    status = None
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            status = {"read_error": str(exc)}
    print(
        json.dumps(
            {
                "seed": seed,
                "run_root": str(run_root),
                "exists": run_root.exists(),
                "status_exists": status_path.is_file(),
                "status": status,
            },
            sort_keys=True,
        ),
        flush=True,
    )
