#!/usr/bin/env bash
# Seed-selectable causal per-update KL-matched SOAP runner.
set -euo pipefail

SEED=${SEED:-0}
ACTOR_LR=${ACTOR_LR:-1e-6}
EXPECTED_STEP=${EXPECTED_STEP:-150}
SAVE_FREQ=${SAVE_FREQ:-$(( EXPECTED_STEP < 25 ? EXPECTED_STEP : 25 ))}
TEST_FREQ=${TEST_FREQ:-$(( EXPECTED_STEP < 10 ? EXPECTED_STEP : 10 ))}
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAMPAIGN_ROOT=${RL_MUON_CAMPAIGN_ROOT:-/home/shkodnik1917/rl_muon/jarvis-gsm8k-r4/campaign-4cdf62757063}
VERL_ROOT=${RL_MUON_VERL_ROOT:-$repo_root/vendor/verl}
MODEL_PATH=${MODEL_PATH:-$CAMPAIGN_ROOT/models/qwen2.5-0.5b-instruct}
DATA_ROOT=${DATA_ROOT:-$CAMPAIGN_ROOT/data/gsm8k}
OUTPUT_ROOT=${OUTPUT_ROOT:-$CAMPAIGN_ROOT/soap-actor-adamw-critic-pilot-seed$SEED}
ACTOR_OPTIMIZER=KLMatchedSOAP
ACTOR_OPTIMIZER_IMPL=verl.utils.kl_matched_soap
ALPHA_MIN=${ALPHA_MIN:-0.05}
ALPHA_MAX=${ALPHA_MAX:-20.0}
ALPHA_CLAMP=${ALPHA_CLAMP:-true}
FISHER_MICRO_BATCH_SIZE=${FISHER_MICRO_BATCH_SIZE:-1}
ACTOR_OPTIMIZER_OVERRIDE="{eps: 1e-5, soap_precondition_frequency: 10, soap_max_precond_dim: 2048, auxiliary_eps: 1e-5, alpha_min: $ALPHA_MIN, alpha_max: $ALPHA_MAX, alpha_clamp: $ALPHA_CLAMP, fisher_dataset_path: '$DATA_ROOT/test.parquet', fisher_prompt_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], fisher_micro_batch_size: $FISHER_MICRO_BATCH_SIZE}"
RUN_NAME=qwen2.5-0.5b_gsm8k_ppo_kl_matched_soap_seed$SEED
RUN_DIR=$OUTPUT_ROOT/$RUN_NAME
TERMINAL_ACTOR_CHECKPOINT=$RUN_DIR/checkpoints/global_step_$EXPECTED_STEP/actor/model_world_size_1_rank_0.pt

[[ -s "$DATA_ROOT/train.parquet" && -s "$DATA_ROOT/test.parquet" ]] || {
    echo "Pinned real GSM8K parquet files are missing under $DATA_ROOT" >&2
    exit 66
}
[[ -d "$MODEL_PATH" ]] || { echo "Pinned model is missing: $MODEL_PATH" >&2; exit 66; }
[[ "$EXPECTED_STEP" =~ ^[1-9][0-9]*$ ]] || { echo "EXPECTED_STEP must be positive" >&2; exit 64; }
mkdir -p "$RUN_DIR"

export PATH="$CAMPAIGN_ROOT/venv/bin:$PATH"
export PYTHONPATH="$repo_root:$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VERL_FILE_LOGGER_PATH="$RUN_DIR/metrics.jsonl"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export TRITON_LIBCUDA_PATH=/lib/x86_64-linux-gnu
export HF_HOME="$CAMPAIGN_ROOT/hf-cache"
export TRANSFORMERS_CACHE="$CAMPAIGN_ROOT/hf-cache/hub"
export TMPDIR=${TMPDIR:-/dev/shm/rlm-soap-tmp-s$SEED}
export RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/rlm-soap-ray-s$SEED}
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

# vLLM 0.8.5 measures device-wide free memory around sleep(). On a shared GPU,
# an unrelated process may allocate concurrently and make the delta negative
# even though vLLM successfully released its own tagged allocations. Keep the
# diagnostic, but do not turn another process's allocation into our run failure.
VLLM_GPU_WORKER=$CAMPAIGN_ROOT/venv/lib/python3.10/site-packages/vllm/v1/worker/gpu_worker.py
(
    flock 9
    python3 - "$VLLM_GPU_WORKER" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
old = '''        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        logger.info(
'''
new = '''        if freed_bytes < 0:
            logger.warning(
                "Device-wide memory usage increased by %.2f GiB while vLLM was "
                "sleeping; this can be caused by another process on a shared GPU.",
                -freed_bytes / GiB_bytes,
            )
            freed_bytes = 0
        logger.info(
'''
text = path.read_text()
if new in text:
    raise SystemExit(0)
if old not in text:
    raise SystemExit(f"unexpected vLLM sleep implementation in {path}")
path.write_text(text.replace(old, new, 1))
PY
) 9>"$CAMPAIGN_ROOT/.vllm-shared-gpu-sleep-patch.lock"

# vLLM 0.8.5 assumes every CUDA_VISIBLE_DEVICES token is an integer.  Use
# stable GPU UUIDs in launchers so CUDA ordinal reordering cannot select a
# different physical card, and teach this older vLLM to resolve that UUID
# through NVML when it needs the host physical index.
VLLM_CUDA_PLATFORM=$CAMPAIGN_ROOT/venv/lib/python3.10/site-packages/vllm/platforms/cuda.py
(
    flock 9
    python3 - "$VLLM_CUDA_PLATFORM" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
old = '''        physical_device_id = device_ids[device_id]
        return int(physical_device_id)
'''
new = '''        physical_device_id = device_ids[device_id]
        if physical_device_id.startswith("GPU-"):
            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByUUID(physical_device_id)
                return int(pynvml.nvmlDeviceGetIndex(handle))
            finally:
                pynvml.nvmlShutdown()
        return int(physical_device_id)
'''
text = path.read_text()
if new in text:
    raise SystemExit(0)
if old not in text:
    raise SystemExit(f"unexpected vLLM CUDA device mapping in {path}")
path.write_text(text.replace(old, new, 1))
PY
) 9>"$CAMPAIGN_ROOT/.vllm-cuda-uuid-patch.lock"

cd "$VERL_ROOT"
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files="$DATA_ROOT/train.parquet" \
    data.val_files="$DATA_ROOT/test.parquet" \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.seed="$SEED" \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.optimizer="$ACTOR_OPTIMIZER" \
    actor_rollout_ref.actor.optim.optimizer_impl="$ACTOR_OPTIMIZER_IMPL" \
    actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.optim.override_optimizer_config="$ACTOR_OPTIMIZER_OVERRIDE" \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.data_loader_seed="$SEED" \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.actor.fsdp_config.seed="$SEED" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.45}" \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.model.path="$MODEL_PATH" \
    +critic.model.override_config.attn_implementation=sdpa \
    critic.model.use_remove_padding=False \
    critic.model.enable_gradient_checkpointing=True \
    critic.optim.optimizer=AdamW \
    critic.optim.optimizer_impl=torch.optim \
    critic.optim.lr=1e-5 \
    critic.optim.weight_decay=0.01 \
    critic.optim.override_optimizer_config='{eps: 1e-5}' \
    critic.ppo_mini_batch_size=64 \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.data_loader_seed="$SEED" \
    critic.fsdp.use_orig_params=True \
    critic.fsdp.seed="$SEED" \
    trainer.logger='[console,file]' \
    trainer.project_name=rl_muon_gsm8k \
    trainer.experiment_name="$RUN_NAME" \
    trainer.default_local_dir="$RUN_DIR/checkpoints" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.total_training_steps="$EXPECTED_STEP" \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TEST_FREQ" \
    +trainer.save_initial_checkpoint=True \
    trainer.val_before_train=True \
    trainer.max_actor_ckpt_to_keep=8 \
    trainer.max_critic_ckpt_to_keep=8 \
    trainer.resume_mode=auto \
    "$@" 2>&1 | tee -a "$RUN_DIR/train.log"

[[ -f "$TERMINAL_ACTOR_CHECKPOINT" ]] || {
    echo "Terminal KL-matched SOAP actor checkpoint is missing: $TERMINAL_ACTOR_CHECKPOINT" >&2
    exit 70
}
