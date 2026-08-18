#!/usr/bin/env bash
# Orchestrate the preregistered cross-dataset actor screen; never runs on CPU.
set -u
mode=${1:?mode is required (screen)}
dataset=${2:?dataset is required (svamp or arc_easy)}
seed=${3:?seed is required (0)}
[[ "$mode" == screen ]] || { echo "invalid mode: $mode"; exit 64; }
[[ "$seed" == 0 ]] || { echo "cross-dataset screen requires seed 0"; exit 64; }
case "$dataset" in svamp) data_source=cross_dataset/svamp ;; arc_easy) data_source=cross_dataset/arc_easy ;; *) echo "invalid dataset: $dataset"; exit 64 ;; esac

repo_root=$(cd "$(dirname "$0")" && pwd)
source_commit=${RL_MUON_SOURCE_COMMIT:?RL_MUON_SOURCE_COMMIT is required}
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$source_commit" ]] || { echo "source commit mismatch"; exit 73; }
campaign_root=${RL_MUON_CAMPAIGN_ROOT:?RL_MUON_CAMPAIGN_ROOT is required}
verl_root="$campaign_root/verl"
venv_root="$campaign_root/venv"
data_root="$campaign_root/data/cross-dataset/$dataset"
model_root="$campaign_root/models/qwen2.5-0.5b-instruct"
manifest="$data_root/manifest.json"
launcher="$repo_root/routed-scale-source/examples/ppo_trainer/run_qwen2_5_0_5b_cross_dataset_screen.sh"
reward_path="$repo_root/cross_dataset_screen.py"
lion_lr=${RL_MUON_LION_LR:?RL_MUON_LION_LR is required}
lion_calibration_sha256=${RL_MUON_LION_CALIBRATION_SHA256:?RL_MUON_LION_CALIBRATION_SHA256 is required}
run_root="$campaign_root/cross_dataset_screen_${dataset}_seed0"
[[ -f "$manifest" ]] || { echo "missing dataset manifest: $manifest"; exit 77; }
[[ -x "$venv_root/bin/python3" ]] || { echo "missing campaign interpreter"; exit 78; }
if ! mkdir "$run_root"; then echo "refusing duplicate run root: $run_root"; exit 74; fi

export PATH="$venv_root/bin:$PATH"
export PYTHONPATH="$repo_root:$verl_root"
export HF_HOME="$campaign_root/hf-cache"
export TORCH_HOME="$campaign_root/torch-cache"
export TOKENIZERS_PARALLELISM=false
venv_python="$venv_root/bin/python3"
[[ "$(command -v python3)" == "$venv_python" ]] || { echo "campaign interpreter is not first on PATH"; exit 79; }

# Reuse the pinned role-routing overlay used by the existing GSM8K wrappers.
"$venv_python" - "$campaign_root" "$repo_root/routed-scale-source" "$verl_root" <<'PY' || exit $?
import fcntl, hashlib, os, sys
from pathlib import Path
campaign_root, source_root, verl_root = map(Path, sys.argv[1:])
relative_paths = (
    Path("verl/utils/optimizers.py"),
    Path("verl/workers/config/optimizer.py"),
)
with (campaign_root / ".cross-dataset-routing-overlay.lock").open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    for relative_path in relative_paths:
        source_path = source_root / relative_path
        target_path = verl_root / relative_path
        if not source_path.is_file() or not target_path.is_file():
            raise RuntimeError(f"missing routing overlay path: {source_path} or {target_path}")
        payload = source_path.read_bytes()
        compile(payload, str(source_path), "exec")
        if target_path.read_bytes() != payload:
            temporary = target_path.with_suffix(target_path.suffix + f".tmp.{os.getpid()}")
            temporary.write_bytes(payload)
            os.replace(temporary, target_path)
        if hashlib.sha256(target_path.read_bytes()).digest() != hashlib.sha256(payload).digest():
            raise RuntimeError(f"routing overlay hash mismatch: {target_path}")
PY

# Verify the pinned manifest, its parquet payloads, source identity, and GPU before model code.
manifest_sha256=$("$venv_python" - "$manifest" "$dataset" "$data_source" <<'PY'
import hashlib, json, sys
from pathlib import Path
import torch
from cross_dataset_screen import DATASETS
manifest_path, dataset, source = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
manifest = json.loads(manifest_path.read_text())
spec = DATASETS[dataset]
expected = {
    "dataset": dataset,
    "data_source": source,
    "repository": spec.repository,
    "config": spec.config,
    "revision": spec.revision,
    "seed": 0,
    "train_split": spec.train_split,
    "validation_split": spec.validation_split,
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise RuntimeError(f"manifest mismatch for {key}: {manifest.get(key)!r} != {value!r}")
for filename, metadata in manifest.get("files", {}).items():
    if filename not in {"train.parquet", "validation.parquet"}:
        raise RuntimeError(f"unexpected manifest file: {filename}")
    observed = hashlib.sha256((manifest_path.parent / filename).read_bytes()).hexdigest()
    if observed != metadata.get("sha256"):
        raise RuntimeError(f"dataset hash mismatch: {filename}")
expected_rows = {
    "train.parquet": spec.expected_train_rows,
    "validation.parquet": spec.expected_validation_rows,
}
for filename, rows in expected_rows.items():
    if manifest["files"].get(filename, {}).get("rows") != rows:
        raise RuntimeError(f"dataset row-count mismatch: {filename}")
if set(manifest.get("files", {})) != {"train.parquet", "validation.parquet"}:
    raise RuntimeError("manifest must contain train and validation parquet files")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; model numerical compute on CPU is forbidden")
print(hashlib.sha256(manifest_path.read_bytes()).hexdigest())
PY
) || exit $?
[[ "$manifest_sha256" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid manifest hash"; exit 77; }
[[ "$lion_calibration_sha256" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid Lion calibration hash"; exit 64; }
"$venv_python" - "$lion_lr" <<'PY' || exit $?
import math, sys
try:
    value = float(sys.argv[1])
except ValueError as error:
    raise RuntimeError("invalid Lion actor learning rate") from error
if not math.isfinite(value) or value <= 0:
    raise RuntimeError("Lion actor learning rate must be finite and positive")
PY

run_route() {
  local phase=$1 route=$2 output_root=$3
  mkdir "$output_root" || return
  PHASE="$phase" ROUTE="$route" DATASET="$dataset" DATA_SOURCE="$data_source" SEED=0 \
    MODEL_PATH="$model_root" DATA_ROOT="$data_root" OUTPUT_ROOT="$output_root" \
    REWARD_PATH="$reward_path" SOURCE_COMMIT="$source_commit" \
    DATA_MANIFEST_SHA256="$manifest_sha256" LION_ACTOR_LR="$lion_lr" \
    LION_CALIBRATION_SHA256="$lion_calibration_sha256" \
    bash "$launcher"
}

# No screen route starts unless the AdamW baseline has non-zero validation reward
# and a real PPO update produces finite step-1 safety metrics.
run_route gate adamw_actor "$run_root/gate" || exit $?
"$venv_python" "$repo_root/collect_cross_dataset_screen.py" \
  "$run_root" "$dataset" 0 "$source_commit" "$manifest" --gate-only || exit $?

for route in adamw_actor muon_actor lion_actor; do
  run_route screen "$route" "$run_root/$route" || exit $?
done
"$venv_python" "$repo_root/collect_cross_dataset_screen.py" \
  "$run_root" "$dataset" 0 "$source_commit" "$manifest"
