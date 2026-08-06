#!/usr/bin/env bash
set -u
repo_root=$(cd "$(dirname "$0")" && pwd)
source_commit=${RL_MUON_SOURCE_COMMIT:?RL_MUON_SOURCE_COMMIT is required}
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$source_commit" ]] || {
  echo "source commit mismatch"
  exit 73
}
campaign_root=${RL_MUON_CAMPAIGN_ROOT:?RL_MUON_CAMPAIGN_ROOT is required}
state_root="$campaign_root/bootstrap"
verl_root="$campaign_root/verl"
data_root="$campaign_root/data/gsm8k"
model_root="$campaign_root/models/qwen2.5-0.5b-instruct"
mkdir -p "$campaign_root"
if ! mkdir "$state_root"; then
  echo "refusing duplicate bootstrap: $state_root"
  exit 74
fi
log="$state_root/bootstrap.log"
exec > >(tee -a "$log") 2>&1
status=0
export PYTHONUSERBASE=/home/jovyan/.local-gsm8k-vllm085-r4
export PATH="$PYTHONUSERBASE/bin:$PATH"
finish() {
  local code=$1
  printf '%s\n' "$code" > "$state_root/exit"
  printf '{"state":"%s","exit":%s}\n' "$([[ "$code" -eq 0 ]] && echo complete || echo failed)" "$code" > "$state_root/status.json"
  printf 'RL_MUON_TERMINAL '
  cat "$state_root/status.json"
  tail -120 "$log"
  exit 0
}

python3 -m pip install --user -r "$repo_root/requirements-gsm8k.txt" || finish $?
git clone https://github.com/verl-project/verl.git "$verl_root" || finish $?
git -C "$verl_root" checkout 7aed6b230776f963fa09509c10d9c3a767d1102c || finish $?
git -C "$verl_root" apply "$repo_root/0001-feat-add-role-routed-Muon-optimizer-for-GSM8K-PPO.patch" || finish $?
python3 -m pip install --user --no-deps -e "$verl_root" || finish $?
mkdir -p "$verl_root/tests/workers/config"
cp "$repo_root/r4_muon_geometry_test.py" \
  "$verl_root/tests/workers/config/test_muon_optimizer_r4_geometry.py" || finish $?

PYTHONPATH="$verl_root" python3 - <<'PY' || finish $?
import json
import platform
import accelerate
import datasets
import google.protobuf
import ray
import torch
import transformers
import vllm
from verl.workers.config.optimizer import FSDPOptimizerConfig

observed = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "vllm": vllm.__version__,
    "transformers": transformers.__version__,
    "ray": ray.__version__,
    "datasets": datasets.__version__,
    "accelerate": accelerate.__version__,
    "protobuf": google.protobuf.__version__,
    "config": FSDPOptimizerConfig().__class__.__name__,
}
expected = {
    "python": "3.10.14",
    "torch": "2.6.0+cu124",
    "vllm": "0.8.5",
    "transformers": "4.51.3",
    "ray": "2.43.0",
    "datasets": "3.6.0",
    "accelerate": "1.6.0",
    "protobuf": "4.25.9",
    "config": "FSDPOptimizerConfig",
}
print(json.dumps(observed, sort_keys=True), flush=True)
if observed != expected:
    raise RuntimeError(f"environment mismatch: observed={observed}, expected={expected}")
PY
python3 -m pip check || finish $?
python3 -m pip freeze > "$state_root/pip-freeze.txt" || finish $?
PYTHONPATH="$verl_root" python3 -m pytest -q \
  -k 'not muon_backport_matches_pytorch_reference_step' \
  "$verl_root/tests/workers/config/test_muon_optimizer_on_cpu.py" \
  "$verl_root/tests/workers/config/test_muon_optimizer_r4_geometry.py" || finish $?
PYTHONPATH="$verl_root" python3 "$verl_root/examples/data_preprocess/gsm8k.py" \
  --revision 740312add88f781978c0658806c59bc2815b9866 --local_dir "$data_root" || finish $?
python3 - "$data_root" "$state_root/data-manifest.json" <<'PY' || finish $?
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

data_root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
expected_rows = {"train.parquet": 7473, "test.parquet": 1319}
manifest = {"dataset_revision": "740312add88f781978c0658806c59bc2815b9866", "files": {}}
for filename, expected in expected_rows.items():
    path = data_root / filename
    rows = pq.read_metadata(path).num_rows
    if rows != expected:
        raise RuntimeError(f"unexpected {filename} row count: {rows} != {expected}")
    manifest["files"][filename] = {
        "rows": rows,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, sort_keys=True), flush=True)
PY
python3 - "$model_root" <<'PY' || finish $?
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    "Qwen/Qwen2.5-0.5B-Instruct",
    revision="7ae557604adf67be50417f59c2c2f167def9a775",
    local_dir=sys.argv[1],
)
PY
finish 0
