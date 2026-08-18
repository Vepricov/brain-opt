#!/usr/bin/env bash
# Qwen2.5-0.5B PPO actor screen. Model execution is GPU-only.
set -euo pipefail

PHASE=${PHASE:?Set PHASE to gate or screen}
ROUTE=${ROUTE:?Set ROUTE to the preregistered actor route}
DATASET=${DATASET:?Set DATASET to svamp or arc_easy}
DATA_SOURCE=${DATA_SOURCE:?Set the exact prepared data source}
SEED=${SEED:?Set SEED=0}
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the pinned local model snapshot}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the prepared dataset directory}
OUTPUT_ROOT=${OUTPUT_ROOT:?Set OUTPUT_ROOT to a fresh route directory}
REWARD_PATH=${REWARD_PATH:?Set REWARD_PATH to cross_dataset_screen.py}
SOURCE_COMMIT=${SOURCE_COMMIT:?Set SOURCE_COMMIT to the checked-out commit}
DATA_MANIFEST_SHA256=${DATA_MANIFEST_SHA256:?Set DATA_MANIFEST_SHA256}
LION_ACTOR_LR=${LION_ACTOR_LR:-}
MUON_ACTOR_LR=${MUON_ACTOR_LR:-}
CALIBRATION_SHA256=${CALIBRATION_SHA256:?Set the per-dataset calibration artifact hash}
MODEL_IDENTITY_SHA256=${MODEL_IDENTITY_SHA256:?Set the verified model snapshot hash}
VERL_IDENTITY_SHA256=${VERL_IDENTITY_SHA256:?Set the verified VERL implementation hash}
MODEL_IDENTITY_ARTIFACT=${MODEL_IDENTITY_ARTIFACT:?Set the model identity artifact}
VERL_IDENTITY_ARTIFACT=${VERL_IDENTITY_ARTIFACT:?Set the VERL identity artifact}
VERL_ROOT=${VERL_ROOT:?Set the active VERL checkout}
ROUTING_OVERLAY_ROOT=${ROUTING_OVERLAY_ROOT:?Set the routing overlay root}
CALIBRATION_METRIC=exact_full_categorical_KL_old_to_new_on_occupied_response_states
[[ "$CALIBRATION_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid calibration hash" >&2; exit 64; }
[[ "$MODEL_IDENTITY_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid model identity hash" >&2; exit 64; }
[[ "$VERL_IDENTITY_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "invalid VERL identity hash" >&2; exit 64; }

[[ "$SEED" == 0 ]] || { echo "cross-dataset screen requires seed 0" >&2; exit 64; }
case "$PHASE" in gate|screen) ;; *) echo "unsupported PHASE=$PHASE" >&2; exit 64 ;; esac
case "$DATASET:$DATA_SOURCE" in
  svamp:cross_dataset/svamp|arc_easy:cross_dataset/arc_easy) ;;
  *) echo "dataset/source mismatch: $DATASET:$DATA_SOURCE" >&2; exit 64 ;;
esac

ACTOR_LR=1e-6
case "$ROUTE" in
  adamw_actor)
    ACTOR_OPT=AdamW
    ACTOR_OPT_IMPL=torch.optim
    ACTOR_OPT_OVERRIDE=null
    ;;
  muon_actor)
    ACTOR_OPT=MuonWithAuxAdamW
    ACTOR_OPT_IMPL=verl.utils.optimizers
    ACTOR_OPT_OVERRIDE='{muon_adjust_lr_fn: match_rms_adamw}'
    ACTOR_LR=${MUON_ACTOR_LR:?Set the per-dataset calibrated Muon actor learning rate}
    ACTOR_ROUTE_DESCRIPTION="Muon on hidden attention/MLP matrices plus AdamW auxiliaries"
    ACTOR_PARAMETER_ROUTING="hidden_attention_mlp_matrices_muon;all_auxiliaries_adamw"
    ACTOR_USES_ADAMW_AUXILIARIES=true
    ;;
  lion_actor)
    ACTOR_OPT=Lion
    ACTOR_OPT_IMPL=verl.utils.optimizers
    ACTOR_OPT_OVERRIDE='{betas: [0.9, 0.99]}'
    ACTOR_LR=${LION_ACTOR_LR:?Set the frozen Lion actor learning rate}
    ACTOR_ROUTE_DESCRIPTION="Lion on all actor parameters; strict no-Adam actor route"
    ACTOR_PARAMETER_ROUTING="all_actor_parameters_lion_no_adam"
    ACTOR_USES_ADAMW_AUXILIARIES=false
    ;;
  *) echo "unsupported ROUTE=$ROUTE" >&2; exit 64 ;;
esac
if [[ "$ROUTE" == adamw_actor ]]; then
  ACTOR_ROUTE_DESCRIPTION="AdamW on all actor parameters"
  ACTOR_PARAMETER_ROUTING="all_actor_parameters"
  ACTOR_USES_ADAMW_AUXILIARIES=false
fi

TOTAL_STEPS=$([[ "$PHASE" == gate ]] && echo 1 || echo 50)
RUN_NAME="qwen2.5-0.5b_${DATASET}_ppo_${PHASE}_${ROUTE}_seed0"
RUN_DIR="$OUTPUT_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR"
export VERL_FILE_LOGGER_PATH="$RUN_DIR/metrics.jsonl"
python3 - "$MODEL_PATH" "$VERL_ROOT" "$ROUTING_OVERLAY_ROOT" \
  "$MODEL_IDENTITY_ARTIFACT" "$VERL_IDENTITY_ARTIFACT" \
  "$MODEL_IDENTITY_SHA256" "$VERL_IDENTITY_SHA256" <<'PY'
import sys
from pathlib import Path
from cross_dataset_screen import model_identity, verl_identity, verify_identity_artifact
model_root, verl_root, overlay_root, model_artifact, verl_artifact = map(Path, sys.argv[1:6])
expected_model, expected_verl = sys.argv[6:]
observed_model = verify_identity_artifact(model_artifact, model_identity(model_root))
observed_verl = verify_identity_artifact(
    verl_artifact, verl_identity(verl_root, overlay_root)
)
if observed_model != expected_model or observed_verl != expected_verl:
    raise RuntimeError("implementation identity argument mismatch")
PY
python3 - "$RUN_DIR/run-provenance.json" "$PHASE" "$ROUTE" "$DATASET" \
  "$DATA_SOURCE" "$SOURCE_COMMIT" "$DATA_MANIFEST_SHA256" "$ACTOR_OPT" \
  "$ACTOR_LR" "$ACTOR_ROUTE_DESCRIPTION" "$ACTOR_PARAMETER_ROUTING" \
  "$ACTOR_USES_ADAMW_AUXILIARIES" "$CALIBRATION_SHA256" "$CALIBRATION_METRIC" \
  "$MODEL_IDENTITY_SHA256" "$VERL_IDENTITY_SHA256" <<'PY'
import json, pathlib, sys
(
    path, phase, route, dataset, data_source, source_commit,
    manifest_sha256, actor_optimizer, actor_lr, route_description,
    parameter_routing, uses_adamw_auxiliaries, calibration_hash, calibration_metric,
    model_identity_hash, verl_identity_hash,
) = sys.argv[1:]
pathlib.Path(path).write_text(json.dumps({
    "protocol": "cross-dataset-actor-screen-v1",
    "phase": phase,
    "route": route,
    "dataset": dataset,
    "data_source": data_source,
    "seed": 0,
    "source_commit": source_commit,
    "manifest_sha256": manifest_sha256,
    "critic_optimizer": "AdamW",
    "preserve_old_logprobs": True,
    "preserve_reference_logprobs": True,
    "actor_optimizer": actor_optimizer,
    "actor_learning_rate": float(actor_lr),
    "actor_route_description": route_description,
    "actor_parameter_routing": parameter_routing,
    "actor_uses_adamw_auxiliaries": uses_adamw_auxiliaries == "true",
    "calibration_sha256": calibration_hash,
    "calibration_metric": calibration_metric,
    "model_snapshot_sha256": model_identity_hash,
    "verl_implementation_sha256": verl_identity_hash,
}, sort_keys=True) + "\n")
PY

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gae \
  algorithm.use_kl_in_reward=True \
  algorithm.kl_ctrl.type=fixed \
  algorithm.kl_ctrl.kl_coef=0.001 \
  custom_reward_function.path="$REWARD_PATH" \
  custom_reward_function.name=compute_score \
  data.train_files="$DATA_ROOT/train.parquet" \
  data.val_files="$DATA_ROOT/validation.parquet" \
  data.train_batch_size=256 \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.seed=0 \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.optimizer="$ACTOR_OPT" \
  actor_rollout_ref.actor.optim.optimizer_impl="$ACTOR_OPT_IMPL" \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.optim.override_optimizer_config="$ACTOR_OPT_OVERRIDE" \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.data_loader_seed=0 \
  actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
  actor_rollout_ref.actor.fsdp_config.seed=0 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  critic.model.path="$MODEL_PATH" \
  critic.model.use_remove_padding=False \
  +critic.model.override_config.attn_implementation=sdpa \
  critic.model.enable_gradient_checkpointing=True \
  critic.optim.optimizer=AdamW \
  critic.optim.optimizer_impl=torch.optim \
  critic.optim.lr=1e-5 \
  critic.optim.weight_decay=0.01 \
  critic.optim.override_optimizer_config=null \
  critic.ppo_mini_batch_size=64 \
  critic.ppo_micro_batch_size_per_gpu=4 \
  critic.data_loader_seed=0 \
  critic.fsdp.use_orig_params=True \
  critic.fsdp.seed=0 \
  trainer.logger='[console,file]' \
  trainer.project_name=rl_muon_cross_dataset_screen \
  trainer.experiment_name="$RUN_NAME" \
  trainer.default_local_dir="$RUN_DIR/checkpoints" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.val_before_train=True \
  trainer.test_freq=25 \
  trainer.save_freq=-1 \
  trainer.total_training_steps="$TOTAL_STEPS" \
  "$@" 2>&1 | tee "$RUN_DIR/train.log"
