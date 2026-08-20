#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAMPAIGN_ROOT=${RL_MUON_CAMPAIGN_ROOT:-/home/shkodnik/rl_muon/jarvis-gsm8k-r4/campaign-4cdf62757063}
GPU_UUID=${GPU_UUID:-GPU-5430d3bb-f055-7a03-62f9-36ce1cf238c8}
OUTPUT_ROOT=${OUTPUT_ROOT:-$CAMPAIGN_ROOT/causal-kfac-soap-smoke-seed0}
STATUS_PATH=$OUTPUT_ROOT/harness.status
GPU_LOG=$OUTPUT_ROOT/gpu-memory.csv
MAX_GPU_USED_MIB=${MAX_GPU_USED_MIB:-35840}
mkdir -p "$OUTPUT_ROOT"
printf 'running\n' >"$STATUS_PATH"
printf 'timestamp,memory_used_mib,memory_total_mib\n' >"$GPU_LOG"

monitor_gpu() {
    while :; do
        timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        sample=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits || true)
        printf '%s,%s\n' "$timestamp" "$sample" >>"$GPU_LOG"
        used=${sample%%,*}
        used=${used//[[:space:]]/}
        if [[ "$used" =~ ^[0-9]+$ ]] && (( used > MAX_GPU_USED_MIB )); then
            printf 'gpu memory cap exceeded: used=%s MiB cap=%s MiB\n' \
                "$used" "$MAX_GPU_USED_MIB" >"$OUTPUT_ROOT/memory-cap-breach.txt"
            kill -TERM -- "-$training_pid" 2>/dev/null || true
            return
        fi
        sleep 2
    done
}
training_pid=
monitor_pid=
cleanup() {
    if [[ -n "$training_pid" ]]; then
        kill -TERM -- "-$training_pid" 2>/dev/null || true
    fi
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    local peak=0 timestamp used total
    while IFS=, read -r timestamp used total; do
        used=${used//[[:space:]]/}
        if [[ "$used" =~ ^[0-9]+$ ]] && (( used > peak )); then
            peak=$used
        fi
    done <"$GPU_LOG"
    printf 'peak_memory_used_mib=%s\n' "$peak" >"$OUTPUT_ROOT/gpu-memory-peak.txt"
}
trap cleanup EXIT

export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export RL_MUON_CAMPAIGN_ROOT="$CAMPAIGN_ROOT"
export RL_MUON_VERL_ROOT="$SCRIPT_ROOT/vendor/verl"
export OUTPUT_ROOT
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.20}
# Ray's Unix sockets must be private to this harness.  A second smoke used to
# rm the shared directory out from under a live raylet, leaving the driver and
# actors alive while every replacement worker failed to connect forever.
export TMPDIR=${TMPDIR:-/tmp/rlm-kfac-smoke-$$}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/rlm-kfac-ray-$$}
rm -rf "$TMPDIR" "$RAY_TMPDIR"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

setsid bash "$SCRIPT_ROOT/smoke_kl_matched_soap_resume.sh" &
training_pid=$!
monitor_gpu &
monitor_pid=$!
if wait "$training_pid"; then
    training_pid=
    printf 'complete\n' >"$STATUS_PATH"
else
    code=$?
    training_pid=
    printf 'failed exit=%s\n' "$code" >"$STATUS_PATH"
    exit "$code"
fi
