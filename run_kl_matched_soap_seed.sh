#!/usr/bin/env bash
set -euo pipefail

SEED=${SEED:?SEED is required}
GPU_INDEX=${GPU_INDEX:?GPU_INDEX is required}
CAMPAIGN_ROOT=${RL_MUON_CAMPAIGN_ROOT:?RL_MUON_CAMPAIGN_ROOT is required}
CALIBRATION=${CALIBRATION:?CALIBRATION is required}
ADAMW_ACTOR_CHECKPOINT=${ADAMW_ACTOR_CHECKPOINT:?ADAMW_ACTOR_CHECKPOINT is required}
BASELINE_METRICS=${BASELINE_METRICS:?BASELINE_METRICS is required}
SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUTPUT_ROOT=${OUTPUT_ROOT:-$CAMPAIGN_ROOT/kl-matched-soap-seed$SEED}
STATUS=${STATUS:-$OUTPUT_ROOT/run.status}
mkdir -p "$OUTPUT_ROOT"

finish() {
    code=$?
    if (( code == 0 )); then state=complete; else state=failed; fi
    printf '%s seed=%s exit=%d %s\n' "$state" "$SEED" "$code" "$(date -u +%FT%TZ)" > "$STATUS"
}
trap finish EXIT

read -r SOAP_SCALE SOAP_LR RELERR < <(python3 - "$CALIBRATION" <<'PY'
import json, sys
value=json.load(open(sys.argv[1]))
print(value['scale'], value['soap_lr'], value['relative_mean_error'])
PY
)
python3 - "$RELERR" <<'PY'
import sys
assert float(sys.argv[1]) <= 0.002, 'exact-KL calibration exceeds 0.2% relative error'
PY

IFS=, read -r used total < <(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ' ')
free_mib=$((total-used))
needed_mib=20480
remaining_mib=$((free_mib-needed_mib))
printf 'gpu=%d free=%dMiB needed=%dMiB remaining=%dMiB soap_scale=%s soap_lr=%s\n' \
    "$GPU_INDEX" "$free_mib" "$needed_mib" "$remaining_mib" "$SOAP_SCALE" "$SOAP_LR"
(( remaining_mib >= 5120 )) || { echo 'GPU headroom gate failed' >&2; exit 75; }

available_kib=$(python3 - <<'PY'
from pathlib import Path
for line in Path('/proc/meminfo').read_text().splitlines():
    if line.startswith('MemAvailable:'):
        print(line.split()[1]); break
PY
)
(( available_kib >= 64*1024*1024 )) || { echo 'Host RAM gate failed' >&2; exit 75; }

printf 'running seed=%s %s\n' "$SEED" "$(date -u +%FT%TZ)" > "$STATUS"
env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
  SEED="$SEED" EXPECTED_STEP=150 SAVE_FREQ=25 TEST_FREQ=10 \
  ACTOR_ROUTE=soap ACTOR_LR=1e-6 SOAP_LR="$SOAP_LR" \
  RL_MUON_CAMPAIGN_ROOT="$CAMPAIGN_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
  ADAMW_ACTOR_CHECKPOINT="$ADAMW_ACTOR_CHECKPOINT" \
  TMPDIR="/dev/shm/rlm-klm-soap-tmp-s$SEED" \
  RAY_TMPDIR="/dev/shm/rlm-klm-soap-ray-s$SEED" \
  GPU_MEMORY_UTILIZATION=0.45 \
  bash "$SCRIPT_ROOT/run_matched_soap_config_adamw.sh"

SOAP_RUN="$OUTPUT_ROOT/qwen2.5-0.5b_gsm8k_ppo_soap_actor_adamw_critic_seed$SEED"
python3 "$SCRIPT_ROOT/evaluate_soap_gate.py" \
  --baseline-metrics "$BASELINE_METRICS" \
  --soap-metrics "$SOAP_RUN/metrics.jsonl" \
  --kl-artifact "$SOAP_RUN/exact_categorical_kl.json" \
  --expected-step 150 \
  --output "$OUTPUT_ROOT/comparison.json"
