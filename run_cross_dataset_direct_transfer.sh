#!/usr/bin/env bash
# Direct transfer of the already-used GSM8K optimizer settings to a new dataset.
set -euo pipefail
mode=${1:?mode must be direct}
dataset=${2:?dataset must be svamp or arc_easy}
seed=${3:?seed must be 0, 1, or 2}
[[ "$mode" == direct && "$seed" =~ ^[012]$ ]] || exit 64
case "$dataset" in
  svamp) data_source=cross_dataset/svamp ;;
  arc_easy) data_source=cross_dataset/arc_easy ;;
  *) exit 64 ;;
esac
repo_root=$(cd "$(dirname "$0")" && pwd)
source_commit=${RL_MUON_SOURCE_COMMIT:?RL_MUON_SOURCE_COMMIT is required}
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$source_commit" ]] || { echo source_commit_mismatch; exit 73; }
campaign_root=${RL_MUON_CAMPAIGN_ROOT:?RL_MUON_CAMPAIGN_ROOT is required}
verl_root="$campaign_root/verl"
venv_root="$campaign_root/venv"
model_root="$campaign_root/models/qwen2.5-0.5b-instruct"
data_root="$campaign_root/data/cross-dataset/$dataset"
manifest="$data_root/manifest.json"
launcher="$repo_root/routed-scale-source/examples/ppo_trainer/run_qwen2_5_0_5b_cross_dataset_screen.sh"
reward_path="$repo_root/cross_dataset_screen.py"
run_root="$campaign_root/cross_dataset_direct_${dataset}_seed${seed}_${source_commit:0:7}"
mkdir -p "$run_root"
status="$run_root/status.json"
finish() {
  local rc=$1 state=complete
  (( rc == 0 )) || state=failed
  printf '{"state":"%s","dataset":"%s","seed":%s,"source_commit":"%s","exit":%d}\n' "$state" "$dataset" "$seed" "$source_commit" "$rc" >"$status"
  exit "$rc"
}
trap 'rc=$?; finish "$rc"' ERR
printf '{"state":"preflight","dataset":"%s","seed":%s,"source_commit":"%s"}\n' "$dataset" "$seed" "$source_commit" >"$status"
export PATH="$venv_root/bin:$PATH"
export PYTHONPATH="$repo_root:$verl_root"
export HF_HOME="$campaign_root/hf-cache" TORCH_HOME="$campaign_root/torch-cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
venv_python="$venv_root/bin/python3"
[[ -x "$venv_python" && -f "$manifest" ]]
source "$repo_root/cloud_verl_compat.sh"
ensure_verl_vllm_compat
"$venv_python" - "$campaign_root" "$repo_root/routed-scale-source" "$verl_root" <<'PY'
import fcntl, hashlib, os, sys
from pathlib import Path
campaign_root, source_root, verl_root = map(Path, sys.argv[1:])
with (campaign_root / ".cross-dataset-routing-overlay.lock").open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    for rel in (Path("verl/utils/optimizers.py"), Path("verl/workers/config/optimizer.py")):
        payload = (source_root / rel).read_bytes()
        compile(payload, str(source_root / rel), "exec")
        target = verl_root / rel
        if target.read_bytes() != payload:
            tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
            tmp.write_bytes(payload)
            os.replace(tmp, target)
        assert hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(payload).digest()
PY
"$venv_python" - "$manifest" "$dataset" "$data_source" "$model_root" "$verl_root" "$repo_root/routed-scale-source" "$run_root" <<'PY'
import json, sys
from pathlib import Path
import torch
from collect_cross_dataset_screen import validate_manifest
from cross_dataset_screen import model_identity, verl_identity, write_identity_artifact
manifest, dataset, source, model_root, verl_root, overlay_root, run_root = sys.argv[1:]
_, manifest_hash = validate_manifest(Path(manifest), dataset)
if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
run_root = Path(run_root)
write_identity_artifact(run_root / "model-identity.json", model_identity(Path(model_root)))
write_identity_artifact(run_root / "verl-identity.json", verl_identity(Path(verl_root), Path(overlay_root)))
(run_root / "manifest.sha256").write_text(manifest_hash + "\n")
PY
read -r manifest_sha256 <"$run_root/manifest.sha256"
identity_values_file="$run_root/identity-values.txt"
"$venv_python" - "$run_root" "$identity_values_file" <<'PY'
import json, sys
from pathlib import Path
r = Path(sys.argv[1])
values_path = Path(sys.argv[2])
values_path.write_text(
    json.loads((r / 'model-identity.json').read_text())['identity_sha256']
    + " "
    + json.loads((r / 'verl-identity.json').read_text())['identity_sha256']
    + "\n"
)
PY
read -r model_identity_sha256 verl_identity_sha256 extra <"$identity_values_file"
[[ -z "${extra:-}" ]]
rm -f "$identity_values_file"
"$venv_python" - "$run_root/fixed-transfer.json" "$dataset" "$source_commit" "$seed" <<'PY'
import json, sys
from pathlib import Path
path, dataset, commit, seed = sys.argv[1:]
Path(path).write_text(json.dumps({"protocol":"fixed-gsm8k-learning-rate-transfer","dataset":dataset,"seed":int(seed),"source_commit":commit,"adamw_actor_lr":1e-6,"muon_actor_lr":1e-6,"lion_actor_lr":1e-6,"dose_matched":False}, sort_keys=True)+"\n")
PY
calibration_sha256=$(sha256sum "$run_root/fixed-transfer.json" | cut -d' ' -f1)
run_route() {
  local phase=$1
  local route=$2
  local output_root="$run_root/$phase/$route"
  mkdir -p "$output_root"
  PHASE="$phase" ROUTE="$route" DATASET="$dataset" DATA_SOURCE="$data_source" SEED="$seed" \
    CHECKPOINT_FREQ=25 RESUME_MODE=auto \
    MODEL_PATH="$model_root" DATA_ROOT="$data_root" OUTPUT_ROOT="$output_root" \
    REWARD_PATH="$reward_path" SOURCE_COMMIT="$source_commit" DATA_MANIFEST_SHA256="$manifest_sha256" \
    MUON_ACTOR_LR=1e-6 LION_ACTOR_LR=1e-6 CALIBRATION_SHA256="$calibration_sha256" \
    MODEL_IDENTITY_SHA256="$model_identity_sha256" VERL_IDENTITY_SHA256="$verl_identity_sha256" \
    MODEL_IDENTITY_ARTIFACT="$run_root/model-identity.json" VERL_IDENTITY_ARTIFACT="$run_root/verl-identity.json" \
    VERL_ROOT="$verl_root" ROUTING_OVERLAY_ROOT="$repo_root/routed-scale-source" bash "$launcher"
  test -s "$output_root/qwen2.5-0.5b_${dataset}_ppo_${phase}_${route}_seed${seed}/metrics.jsonl"
}
printf '{"state":"gates","dataset":"%s","seed":%s,"source_commit":"%s"}\n' "$dataset" "$seed" "$source_commit" >"$status"
for route in adamw_actor muon_actor lion_actor; do run_route gate "$route"; done
printf '{"state":"screens","dataset":"%s","seed":%s,"source_commit":"%s"}\n' "$dataset" "$seed" "$source_commit" >"$status"
for route in adamw_actor muon_actor lion_actor; do run_route screen "$route"; done
trap - ERR
finish 0
