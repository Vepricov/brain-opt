#!/usr/bin/env bash
set -euo pipefail
dataset=${1:?dataset is required}
case "$dataset" in svamp|arc_easy) ;; *) echo "unsupported dataset: $dataset" >&2; exit 64 ;; esac
repo_root=$(cd "$(dirname "$0")" && pwd)
campaign_root=${RL_MUON_CAMPAIGN_ROOT:?RL_MUON_CAMPAIGN_ROOT is required}
python_bin="$campaign_root/venv/bin/python3"
output_root="$campaign_root/data/cross-dataset/$dataset"
[[ -x "$python_bin" ]] || { echo "missing campaign interpreter" >&2; exit 78; }
if [[ -e "$output_root" ]]; then
  echo "refusing existing dataset root: $output_root" >&2
  exit 74
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$repo_root:$campaign_root/verl"
exec "$python_bin" "$repo_root/prepare_cross_dataset.py" "$dataset" "$output_root"
