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
run_root="$campaign_root/cross_dataset_screen_${dataset}_seed0"
preflight_root="$campaign_root/.cross_dataset_screen_${dataset}_seed0.preflight.$$"
mkdir "$preflight_root" || exit 74
status_file="$preflight_root/status.json"
write_status() {
  local state=$1 detail=$2 timestamp
  timestamp=$(date -Is)
  printf '{"time":"%s","state":"%s","phase":"screen","dataset":"%s","seed":0,"source_commit":"%s","detail":"%s"}\n' \
    "$timestamp" "$state" "$dataset" "$source_commit" "$detail" > "$status_file"
}
finish() {
  local code=$1 state=complete timestamp
  (( code == 0 )) || state=failed
  timestamp=$(date -Is)
  printf '%s\n' "$code" > "$(dirname "$status_file")/exit"
  printf '{"time":"%s","state":"%s","phase":"screen","dataset":"%s","seed":0,"source_commit":"%s","exit":%s,"detail":"runner_exit=%s"}\n' \
    "$timestamp" "$state" "$dataset" "$source_commit" "$code" "$code" > "$status_file"
  printf 'RL_MUON_TERMINAL {"time":"%s","state":"%s","phase":"screen","dataset":"%s","seed":0,"source_commit":"%s","exit":%s,"detail":"runner_exit=%s"}\n' \
    "$timestamp" "$state" "$dataset" "$source_commit" "$code" "$code"
  exit "$code"
}

deadline=$((SECONDS + 3600))
write_status waiting_for_bootstrap "waiting for validated environment"
while true; do
  bootstrap_state=$(/usr/bin/python3 - "$campaign_root/bootstrap/status.json" <<'PY' 2>/dev/null || true
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    payload = json.loads(path.read_text())
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
state = payload.get("state")
if type(state) is not str or state not in {"complete", "failed"}:
    raise SystemExit(1)
print(state)
PY
  )
  [[ "$bootstrap_state" == complete ]] && break
  [[ "$bootstrap_state" == failed ]] && finish 75
  (( SECONDS < deadline )) || finish 76
  sleep 15
done

[[ -f "$manifest" ]] || { echo "missing dataset manifest: $manifest"; finish 77; }
[[ -x "$venv_root/bin/python3" ]] || { echo "missing campaign interpreter"; finish 78; }

export PATH="$venv_root/bin:$PATH"
export PYTHONPATH="$repo_root:$verl_root"
export HF_HOME="$campaign_root/hf-cache"
export TORCH_HOME="$campaign_root/torch-cache"
export TOKENIZERS_PARALLELISM=false
venv_python="$venv_root/bin/python3"
[[ "$(command -v python3)" == "$venv_python" ]] || { echo "campaign interpreter is not first on PATH"; finish 79; }
source "$repo_root/cloud_verl_compat.sh"
ensure_verl_vllm_compat

# Reuse the pinned role-routing overlay used by the existing GSM8K wrappers.
"$venv_python" - "$campaign_root" "$repo_root/routed-scale-source" "$verl_root" <<'PY' || finish $?
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
import sys
from pathlib import Path
import torch
from collect_cross_dataset_screen import validate_manifest
manifest_path, dataset, source = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
manifest, manifest_sha256 = validate_manifest(manifest_path, dataset)
if manifest["data_source"] != source:
    raise RuntimeError("manifest data source mismatch")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; model numerical compute on CPU is forbidden")
print(manifest_sha256)
PY
) || finish $?
[[ "$manifest_sha256" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid manifest hash"; finish 77; }

# Only claim the permanent run root after bootstrap and every read-only/shared
# compatibility, artifact, interpreter, source, and CUDA preflight has passed.
if ! mkdir "$run_root"; then echo "refusing duplicate run root: $run_root"; finish 74; fi
mv "$preflight_root/status.json" "$run_root/status.json"
rm -f "$preflight_root/exit"
rmdir "$preflight_root"
status_file="$run_root/status.json"
write_status running "preflight complete"

# Materialize and immediately verify byte-derived identities before any model load.
identity_values=$("$venv_python" - "$model_root" "$verl_root" \
  "$repo_root/routed-scale-source" "$run_root" <<'PY'
import sys
from pathlib import Path
from cross_dataset_screen import model_identity, verl_identity, write_identity_artifact
model_root, verl_root, overlay_root, run_root = map(Path, sys.argv[1:])
model = model_identity(model_root)
verl = verl_identity(verl_root, overlay_root)
write_identity_artifact(run_root / "model-identity.json", model)
write_identity_artifact(run_root / "verl-identity.json", verl)
print(model["identity_sha256"], verl["identity_sha256"])
PY
) || finish $?
read -r model_identity_sha256 verl_identity_sha256 extra <<<"$identity_values"
[[ -z "${extra:-}" && "$model_identity_sha256" =~ ^[0-9a-f]{64}$ && \
  "$verl_identity_sha256" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid implementation identities"; finish 65; }
chmod 0444 "$run_root/model-identity.json" "$run_root/verl-identity.json" || finish $?

# Calibrate both non-Adam routes on this dataset's same frozen rollout batch.
# The artifact is immutable once hashed and every gate/screen provenance binds it.
calibration="$run_root/calibration.json"
"$venv_python" "$repo_root/calibrate_lion_actor_lr.py" \
  --model-path "$model_root" \
  --train-file "$data_root/train.parquet" \
  --output "$calibration" \
  --dataset "$dataset" \
  --data-source "$data_source" \
  --source-commit "$source_commit" \
  --manifest-sha256 "$manifest_sha256" \
  --model-identity-artifact "$run_root/model-identity.json" \
  --verl-identity-artifact "$run_root/verl-identity.json" \
  --verl-root "$verl_root" \
  --routing-overlay-root "$repo_root/routed-scale-source" \
  --seed 0 \
  --max-response-length 256 || finish $?
calibration_values=$("$venv_python" - "$calibration" "$dataset" "$data_source" \
  "$source_commit" "$manifest_sha256" "$data_root/train.parquet" \
  "$model_identity_sha256" "$verl_identity_sha256" <<'PY'
import hashlib, json, math, sys
from pathlib import Path
path, dataset, source, commit, manifest_hash, train_path, model_hash, verl_hash = sys.argv[1:]
payload = Path(path).read_bytes()
artifact = json.loads(payload)
expected = {
    "schema_version": 3,
    "status": "complete",
    "protocol": "per_dataset_same_frozen_batch_one_production_equivalent_ppo_update",
    "dataset": dataset,
    "data_source": source,
    "seed": 0,
    "source_commit": commit,
    "manifest_sha256": manifest_hash,
    "calibration_metric": "exact_full_categorical_KL_old_to_new_on_occupied_response_states",
    "model_compute_device": "cuda",
}
for key, value in expected.items():
    if artifact.get(key) != value:
        raise RuntimeError(f"calibration mismatch for {key}")
train_hash = hashlib.sha256(Path(train_path).read_bytes()).hexdigest()
if artifact.get("identities", {}).get("train_file_sha256") != train_hash:
    raise RuntimeError("calibration train-file hash mismatch")
if artifact.get("identities", {}).get("model_snapshot_sha256") != model_hash:
    raise RuntimeError("calibration model snapshot identity mismatch")
if artifact.get("identities", {}).get("verl_implementation_sha256") != verl_hash:
    raise RuntimeError("calibration VERL implementation identity mismatch")
for route in ("muon", "lion"):
    lr = artifact.get(f"chosen_{route}_learning_rate")
    metrics = artifact.get(f"chosen_{route}_actual_full_categorical_kl", {})
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise RuntimeError(f"invalid calibrated {route} learning rate")
    if any(metrics.get(f"{metric}_relative_error", math.inf) > 0.10 for metric in ("mean", "q95")):
        raise RuntimeError(f"{route} did not jointly match AdamW mean and q95 KL")
if artifact.get("gates", {}).get("passed") is not True:
    raise RuntimeError("joint calibration gate failed")
print(artifact["chosen_muon_learning_rate"], artifact["chosen_lion_learning_rate"], hashlib.sha256(payload).hexdigest())
PY
) || finish $?
read -r muon_lr lion_lr calibration_sha256 extra <<<"$calibration_values"
[[ -z "${extra:-}" && "$calibration_sha256" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid calibration values"; finish 65; }
chmod 0444 "$calibration" || finish $?

run_route() {
  local phase=$1 route=$2 output_root=$3
  mkdir "$output_root" || return
  PHASE="$phase" ROUTE="$route" DATASET="$dataset" DATA_SOURCE="$data_source" SEED=0 \
    MODEL_PATH="$model_root" DATA_ROOT="$data_root" OUTPUT_ROOT="$output_root" \
    REWARD_PATH="$reward_path" SOURCE_COMMIT="$source_commit" \
    DATA_MANIFEST_SHA256="$manifest_sha256" MUON_ACTOR_LR="$muon_lr" \
    LION_ACTOR_LR="$lion_lr" CALIBRATION_SHA256="$calibration_sha256" \
    MODEL_IDENTITY_SHA256="$model_identity_sha256" \
    VERL_IDENTITY_SHA256="$verl_identity_sha256" \
    MODEL_IDENTITY_ARTIFACT="$run_root/model-identity.json" \
    VERL_IDENTITY_ARTIFACT="$run_root/verl-identity.json" \
    VERL_ROOT="$verl_root" ROUTING_OVERLAY_ROOT="$repo_root/routed-scale-source" \
    bash "$launcher"
}

# Each route must independently produce real baseline validation and a finite
# step-1 PPO safety update. All three gates pass before any 50-step launch.
for route in adamw_actor muon_actor lion_actor; do
  run_route gate "$route" "$run_root/gates/$route" || finish $?
  "$venv_python" "$repo_root/collect_cross_dataset_screen.py" \
    "$run_root" "$dataset" 0 "$source_commit" "$manifest" "$model_root" \
    "$verl_root" "$repo_root/routed-scale-source" --gate-route "$route" || finish $?
done
for route in adamw_actor muon_actor lion_actor; do
  run_route screen "$route" "$run_root/$route" || finish $?
done
"$venv_python" "$repo_root/collect_cross_dataset_screen.py" \
  "$run_root" "$dataset" 0 "$source_commit" "$manifest" "$model_root" \
  "$verl_root" "$repo_root/routed-scale-source" || finish $?
finish 0
