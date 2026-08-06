#!/usr/bin/env bash
set -u
seed=${1:?seed is required}
case "$seed" in 0|1|2) ;; *) echo "invalid seed: $seed"; exit 0 ;; esac
repo_root=$(cd "$(dirname "$0")" && pwd)
campaign_root=/home/jovyan/rl_muon/h7_online_cloud_r1
seed_root="$campaign_root/seed_${seed}"
mkdir -p "$seed_root"
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
  exit 0
}
cd "$repo_root"
write_status running "environment validation"
python - <<'PY' || finish $?
import torch, transformers, datasets, accelerate, numpy
expected = ("2.10.0+cu128", "5.14.1", "5.0.1", "1.14.0", "2.5.1")
observed = (torch.__version__, transformers.__version__, datasets.__version__, accelerate.__version__, numpy.__version__)
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
  python cloud_h7/h7_online_ppo.py --config "cloud_h7/configs/config_seed${seed}.json" --route "$route" --run-dir "$run_dir" || finish $?
  printf '{"time":"%s","seed":%s,"route":"%s","event":"complete"}\n' "$(date -Is)" "$seed" "$route" >> "$seed_root/progress.jsonl"
done
finish 0
