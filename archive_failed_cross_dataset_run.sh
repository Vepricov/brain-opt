#!/usr/bin/env bash
set -euo pipefail

dataset=${1:?dataset is required}
expected_source_commit=${2:?expected source commit is required}
case "$dataset" in
  svamp|arc_easy) ;;
  *) echo "unsupported dataset: $dataset" >&2; exit 64 ;;
esac
[[ "$expected_source_commit" =~ ^[0-9a-f]{40}$ ]] || {
  echo "invalid expected source commit" >&2
  exit 64
}

campaign_root=${RL_MUON_CAMPAIGN_ROOT:?RL_MUON_CAMPAIGN_ROOT is required}
run_root="$campaign_root/cross_dataset_screen_${dataset}_seed0"
archive_root="${run_root}.failed-${expected_source_commit:0:7}-gitlink"
[[ -d "$run_root" && ! -L "$run_root" ]] || {
  echo "missing or unsafe failed run root: $run_root" >&2
  exit 74
}
[[ ! -e "$archive_root" ]] || {
  echo "archive root already exists: $archive_root" >&2
  exit 74
}

python3 - "$run_root/status.json" "$expected_source_commit" <<'PY'
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
expected_commit = sys.argv[2]
payload = json.loads(status_path.read_text())
if payload.get("state") != "failed":
    raise RuntimeError("refusing to archive a run that is not failed")
if payload.get("source_commit") != expected_commit:
    raise RuntimeError("failed run source commit mismatch")
PY

mv -- "$run_root" "$archive_root"
printf 'archived=%s\n' "$archive_root"
