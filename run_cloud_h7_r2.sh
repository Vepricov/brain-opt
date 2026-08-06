#!/usr/bin/env bash
set -u
seed=${1:?seed is required}
case "$seed" in 0|1|2) ;; *) echo "invalid seed: $seed"; exit 64 ;; esac
repo_root=$(cd "$(dirname "$0")" && pwd)
campaign_root=/home/jovyan/rl_muon/h7_online_cloud_r2
seed_root="$campaign_root/seed_${seed}"
mkdir -p "$campaign_root"
if ! mkdir "$seed_root"; then
  echo "refusing duplicate or resumed seed root: $seed_root"
  exit 74
fi
log="$seed_root/cloud_runner.log"
exec > >(tee -a "$log") 2>&1
write_status() {
  local state=$1 detail=$2
  printf '{"time":"%s","state":"%s","seed":%s,"detail":"%s"}\n' "$(date -Is)" "$state" "$seed" "$detail" > "$seed_root/status.json"
}
finish() {
  local code=$1
  printf '%s\n' "$code" > "$seed_root/exit"
  if [[ "$code" -eq 0 ]]; then
    write_status complete "3/3 routes complete"
  else
    write_status failed "runner_exit=$code"
  fi
  tail -100 "$log"
  # Cloud.ru suppresses logs for failed platform jobs. Scientific success is
  # defined only by the durable status/exit pair and validated result artifacts.
  exit 0
}
bootstrap_status="$campaign_root/bootstrap/status.json"
bootstrap_deadline=$((SECONDS + 3600))
write_status waiting_for_bootstrap "waiting for exact persistent Python environment"
while true; do
  if [[ -f "$bootstrap_status" ]] && grep -q '"state":"complete"' "$bootstrap_status"; then
    break
  fi
  if [[ -f "$bootstrap_status" ]] && grep -q '"state":"failed"' "$bootstrap_status"; then
    echo "bootstrap failed: $bootstrap_status"
    finish 75
  fi
  if (( SECONDS >= bootstrap_deadline )); then
    echo "bootstrap did not complete within 3600 seconds"
    finish 76
  fi
  sleep 15
done
cd "$repo_root"
write_status running "environment validation"
python - <<'PY' || finish $?
import json
import platform
from pathlib import Path
import torch, transformers, datasets, accelerate, numpy
profile = json.loads(Path("cloud_h7_r2/profile.json").read_text())
expected = (
    profile["python"],
    profile["torch"],
    profile["transformers"],
    profile["datasets"],
    profile["accelerate"],
    profile["numpy"],
)
observed = (
    platform.python_version(),
    torch.__version__,
    transformers.__version__,
    datasets.__version__,
    accelerate.__version__,
    numpy.__version__,
)
if observed != expected:
    raise RuntimeError(f"environment mismatch: observed={observed}, expected={expected}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
print({"gpu": torch.cuda.get_device_name(0), "versions": observed}, flush=True)
PY
for route in raw_muon own_polar_d01 own_polar_d1; do
  run_dir="$seed_root/$route"
  if [[ -e "$run_dir/result.json" || -e "$run_dir/failure.json" ]]; then
    echo "refusing to overwrite existing endpoint: $run_dir"
    finish 74
  fi
  write_status running "route=$route"
  printf '{"time":"%s","seed":%s,"route":"%s","event":"start"}\n' "$(date -Is)" "$seed" "$route" >> "$seed_root/progress.jsonl"
  python cloud_h7_r2/src/h7_online_ppo.py --config "cloud_h7/configs/config_seed${seed}.json" --route "$route" --run-dir "$run_dir" || finish $?
  python - "$run_dir" "$seed" "$route" <<'PY' || finish $?
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
seed = int(sys.argv[2])
route = sys.argv[3]

def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def require_finite(value, path="root"):
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite value at {path}: {value}")

if (run_dir / "failure.json").exists():
    raise RuntimeError(f"failure artifact exists: {run_dir / 'failure.json'}")
result = json.loads((run_dir / "result.json").read_text())
progress = load_jsonl(run_dir / "progress.jsonl")
evaluations = load_jsonl(run_dir / "eval.jsonl")
require_finite(result)
require_finite(progress)
require_finite(evaluations)
if result.get("status") != "complete" or result.get("seed") != seed or result.get("route") != route:
    raise RuntimeError(f"invalid result identity: {result}")
if result.get("total_updates") != 20 or len(progress) != 20:
    raise RuntimeError(f"invalid progress length: result={result.get('total_updates')}, rows={len(progress)}")
if [row.get("update") for row in progress] != list(range(1, 21)):
    raise RuntimeError("progress update schedule mismatch")
if [row.get("update") for row in evaluations] != [0, 5, 10, 15, 20]:
    raise RuntimeError("evaluation schedule mismatch")
for row in progress:
    for key in ("calibration_budget_ratio", "realized_functional_budget_ratio"):
        ratio = float(row[key])
        if not 0.98 <= ratio <= 1.02:
            raise RuntimeError(f"strict budget mismatch at update {row['update']}: {key}={ratio}")
print(f"VALIDATED seed={seed} route={route}", flush=True)
PY
  printf '{"time":"%s","seed":%s,"route":"%s","event":"complete"}\n' "$(date -Is)" "$seed" "$route" >> "$seed_root/progress.jsonl"
done
finish 0
