#!/usr/bin/env bash
set -u
mode=${1:?mode is required: smoke, full, routed-smoke, or routed-full}
seed=${2:?seed is required}
case "$mode" in smoke|full|routed-smoke|routed-full) ;; *) echo "invalid mode: $mode"; exit 64 ;; esac
case "$seed" in 0|1|2) ;; *) echo "invalid seed: $seed"; exit 64 ;; esac
repo_root=$(cd "$(dirname "$0")" && pwd)
source_commit=${RL_MUON_SOURCE_COMMIT:?RL_MUON_SOURCE_COMMIT is required}
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$source_commit" ]] || {
  echo "source commit mismatch"
  exit 73
}
campaign_root=${RL_MUON_CAMPAIGN_ROOT:?RL_MUON_CAMPAIGN_ROOT is required}
verl_root="$campaign_root/verl"
venv_root="$campaign_root/venv"
data_root="$campaign_root/data/gsm8k"
model_root="$campaign_root/models/qwen2.5-0.5b-instruct"
attempt=${RL_MUON_ATTEMPT:-}
if [[ -n "$attempt" && ! "$attempt" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]]; then
  echo "invalid attempt tag: $attempt"
  exit 64
fi
run_root="$campaign_root/${mode}_seed${seed}${attempt:+_$attempt}"
if ! mkdir "$run_root"; then
  echo "refusing duplicate run root: $run_root"
  exit 74
fi
log="$run_root/cloud_runner.log"
exec > >(tee -a "$log") 2>&1
heartbeat_pid=
heartbeat() {
  while sleep 60; do
    printf 'RL_MUON_HEARTBEAT time=%s mode=%s seed=%s elapsed=%ss\n' \
      "$(date -Is)" "$mode" "$seed" "$SECONDS"
  done
}
heartbeat &
heartbeat_pid=$!
write_status() {
  printf '{"time":"%s","state":"%s","mode":"%s","seed":%s,"detail":"%s"}\n' \
    "$(date -Is)" "$1" "$mode" "$seed" "$2" > "$run_root/status.json"
}
finish() {
  local code=$1
  local state
  local timestamp
  if [[ -n "$heartbeat_pid" ]]; then
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
    heartbeat_pid=
  fi
  state=$([[ "$code" -eq 0 ]] && echo complete || echo failed)
  timestamp=$(date -Is)
  printf '%s\n' "$code" > "$run_root/exit"
  printf '{"time":"%s","state":"%s","phase":"%s","seed":%s,"source_commit":"%s","exit":%s,"detail":"runner_exit=%s"}\n' \
    "$timestamp" "$state" \
    "$mode" "$seed" "$source_commit" "$code" "$code" > "$run_root/status.json"
  printf 'RL_MUON_TERMINAL {"time":"%s","state":"%s","phase":"%s","seed":%s,"source_commit":"%s","exit":%s,"detail":"runner_exit=%s"}\n' \
    "$timestamp" "$state" "$mode" "$seed" "$source_commit" "$code" "$code"
  tail -120 "$log"
  exit 0
}

deadline=$((SECONDS + 3600))
write_status waiting_for_bootstrap "waiting for validated environment"
while true; do
  if grep -q '"state":"complete"' "$campaign_root/bootstrap/status.json" 2>/dev/null; then break; fi
  if grep -q '"state":"failed"' "$campaign_root/bootstrap/status.json" 2>/dev/null; then finish 75; fi
  if (( SECONDS >= deadline )); then finish 76; fi
  sleep 15
done

export PATH="$venv_root/bin:$PATH"
export PYTHONPATH="$repo_root:$verl_root"
export HF_HOME="$campaign_root/hf-cache"
export TORCH_HOME="$campaign_root/torch-cache"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export TRITON_LIBCUDA_PATH=/lib/x86_64-linux-gnu
# The pinned VERL snapshot now calls DataProto.to_tensordict(), whose
# NonTensorStack conversion is only available with tensordict >= 0.10.  Older
# campaigns were built with 0.8.3, so repair a reused venv once under a lock.
python3 - "$campaign_root" <<'PY' || finish $?
import fcntl
import importlib.metadata
import subprocess
import sys
from pathlib import Path

from packaging.version import Version

campaign_root = Path(sys.argv[1])
lock_path = campaign_root / ".tensordict010-compat.lock"
with lock_path.open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        installed = Version(importlib.metadata.version("tensordict"))
    except importlib.metadata.PackageNotFoundError:
        installed = Version("0")
    if installed < Version("0.10"):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-cache-dir",
                "--no-deps",
                "--upgrade",
                "tensordict==0.10.0",
                "pyvers==0.1.0",
            ],
            check=True,
        )
    import tensordict

    if Version(tensordict.__version__) < Version("0.10"):
        raise RuntimeError(f"tensordict upgrade did not take effect: {tensordict.__version__}")
    from tensordict.tensorclass import NonTensorData, NonTensorStack

    print(
        f"verified tensordict compatibility: {tensordict.__version__} "
        f"({NonTensorData.__name__}, {NonTensorStack.__name__})",
        flush=True,
    )
PY
# A reused campaign may have been bootstrapped by a source commit predating the
# vLLM 0.8 compatibility patch. Patch that installed snapshot under a lock so
# retries and concurrently queued seeds remain safe without rebuilding the venv.
python3 - "$campaign_root" \
  "$verl_root/verl/workers/rollout/vllm_rollout/vllm_async_server.py" \
  "$verl_root/verl/workers/rollout/vllm_rollout/utils.py" \
  "$verl_root/verl/utils/attention_utils.py" <<'PY' || finish $?
import fcntl
import os
import re
import sys
from pathlib import Path

campaign_root = Path(sys.argv[1])
path = Path(sys.argv[2])
weight_utils_path = Path(sys.argv[3])
attention_utils_path = Path(sys.argv[4])
logprobs_needle = '            "logprobs_mode": self.config.logprobs_mode,\n'
reset_pattern = re.compile(r"^        await engine_client\.reset_mm_cache\(\)\n", re.MULTILINE)
reset_replacement = (
    '        if hasattr(engine_client, "reset_mm_cache"):\n'
    "            await engine_client.reset_mm_cache()\n"
)
drain_pattern = re.compile(r"^        await self\.engine\.wait_for_requests_to_drain\(\)\n", re.MULTILINE)
drain_replacement = (
    '        drain = getattr(self.engine, "wait_for_requests_to_drain", None)\n'
    "        if drain is not None:\n"
    "            await drain()\n"
    "        else:\n"
    "            while self.engine.output_processor.request_states:\n"
    "                await asyncio.sleep(0.01)\n"
)
empty_multimodal_prompt = (
    '        prompt_kwargs = {"prompt_token_ids": prompt_ids, "multi_modal_data": multi_modal_data}\n'
)
compatible_multimodal_prompt = (
    '        prompt_kwargs = {"prompt_token_ids": prompt_ids}\n'
    "        if multi_modal_data:\n"
    '            prompt_kwargs["multi_modal_data"] = multi_modal_data\n'
)
flash_attention_import = (
    "    else:\n"
    "        from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input\n"
)
compatible_attention_import = (
    "    else:\n"
    "        try:\n"
    "            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input\n"
    "        except ModuleNotFoundError as exc:\n"
    '            if exc.name != "flash_attn":\n'
    "                raise\n"
    "            from verl.utils.npu_flash_attn_utils import index_first_axis, pad_input, rearrange, unpad_input\n"
)
lock_path = campaign_root / ".vllm085-compat.lock"
with lock_path.open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    source = path.read_text()
    logprobs_count = source.count(logprobs_needle)
    reset_count = len(reset_pattern.findall(source))
    drain_count = len(drain_pattern.findall(source))
    if logprobs_count not in (0, 1):
        raise RuntimeError(f"unexpected logprobs_mode assignment count in {path}: {logprobs_count}")
    if reset_count == 0 and reset_replacement not in source:
        raise RuntimeError(f"missing expected reset_mm_cache call in {path}")
    if reset_count not in (0, 1):
        raise RuntimeError(f"unexpected reset_mm_cache call count in {path}: {reset_count}")
    if drain_count == 0 and drain_replacement not in source:
        raise RuntimeError(f"missing expected wait_for_requests_to_drain call in {path}")
    if drain_count not in (0, 1):
        raise RuntimeError(f"unexpected wait_for_requests_to_drain call count in {path}: {drain_count}")
    multimodal_count = source.count(empty_multimodal_prompt)
    if multimodal_count == 0 and compatible_multimodal_prompt not in source:
        raise RuntimeError(f"missing expected multi_modal_data prompt construction in {path}")
    if multimodal_count not in (0, 1):
        raise RuntimeError(f"unexpected multi_modal_data prompt count in {path}: {multimodal_count}")
    updated = source.replace(logprobs_needle, "")
    updated = reset_pattern.sub(reset_replacement, updated)
    updated = drain_pattern.sub(drain_replacement, updated)
    updated = updated.replace(empty_multimodal_prompt, compatible_multimodal_prompt)
    if updated != source:
        compile(updated, str(path), "exec")
        temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        temporary.write_text(updated)
        os.replace(temporary, path)
    verified = path.read_text()
    if logprobs_needle in verified:
        raise RuntimeError(f"failed to remove unsupported logprobs_mode from {path}")
    if reset_replacement not in verified:
        raise RuntimeError(f"failed to guard optional reset_mm_cache in {path}")
    if drain_replacement not in verified:
        raise RuntimeError(f"failed to guard optional wait_for_requests_to_drain in {path}")
    if compatible_multimodal_prompt not in verified:
        raise RuntimeError(f"failed to omit empty multi_modal_data for vLLM 0.8 in {path}")
    weight_source = weight_utils_path.read_text()
    public_import_pattern = re.compile(
        r"^            from vllm\.model_executor\.model_loader\.utils import "
        r"process_weights_after_loading\n",
        re.MULTILINE,
    )
    compatible_import = (
        "            try:\n"
        "                from vllm.model_executor.model_loader.utils import process_weights_after_loading\n"
        "            except ImportError:\n"
        "                from vllm.model_executor.model_loader.loader import (\n"
        "                    _process_weights_after_loading as process_weights_after_loading,\n"
        "                )\n"
    )
    public_count = len(public_import_pattern.findall(weight_source))
    if public_count == 0 and compatible_import not in weight_source:
        raise RuntimeError(f"missing expected process_weights_after_loading import in {weight_utils_path}")
    if public_count not in (0, 1):
        raise RuntimeError(
            f"unexpected process_weights_after_loading import count in {weight_utils_path}: {public_count}"
        )
    weight_updated = public_import_pattern.sub(compatible_import, weight_source)
    if weight_updated != weight_source:
        compile(weight_updated, str(weight_utils_path), "exec")
        temporary = weight_utils_path.with_suffix(weight_utils_path.suffix + f".tmp.{os.getpid()}")
        temporary.write_text(weight_updated)
        os.replace(temporary, weight_utils_path)
    if compatible_import not in weight_utils_path.read_text():
        raise RuntimeError(f"failed to add vLLM 0.8 weight post-processing fallback in {weight_utils_path}")
    attention_source = attention_utils_path.read_text()
    flash_attention_count = attention_source.count(flash_attention_import)
    if flash_attention_count == 0 and compatible_attention_import not in attention_source:
        raise RuntimeError(f"missing expected FlashAttention import in {attention_utils_path}")
    if flash_attention_count not in (0, 1):
        raise RuntimeError(
            f"unexpected FlashAttention import count in {attention_utils_path}: {flash_attention_count}"
        )
    attention_updated = attention_source.replace(flash_attention_import, compatible_attention_import)
    if attention_updated != attention_source:
        compile(attention_updated, str(attention_utils_path), "exec")
        temporary = attention_utils_path.with_suffix(attention_utils_path.suffix + f".tmp.{os.getpid()}")
        temporary.write_text(attention_updated)
        os.replace(temporary, attention_utils_path)
    if compatible_attention_import not in attention_utils_path.read_text():
        raise RuntimeError(f"failed to add pure-PyTorch padding fallback in {attention_utils_path}")
print(f"verified vLLM 0.8 argv compatibility: {path}", flush=True)
PY
if [[ "$mode" == routed-* ]]; then
  python3 - "$campaign_root" "$repo_root/routed-scale-source" "$verl_root" <<'PY' || finish $?
import fcntl
import hashlib
import os
import sys
from pathlib import Path

campaign_root = Path(sys.argv[1])
source_root = Path(sys.argv[2])
verl_root = Path(sys.argv[3])
relative_paths = (
    Path("verl/utils/optimizers.py"),
    Path("verl/workers/config/optimizer.py"),
    Path("examples/ppo_trainer/run_qwen2_5_0_5b_gsm8k_optimizer_ablation.sh"),
)

lock_path = campaign_root / ".routed-scale-adamw.lock"
with lock_path.open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    for relative_path in relative_paths:
        source_path = source_root / relative_path
        target_path = verl_root / relative_path
        if not source_path.is_file() or not target_path.is_file():
            raise RuntimeError(f"missing routed-scale overlay path: {source_path} or {target_path}")
        source = source_path.read_bytes()
        if source_path.suffix == ".py":
            compile(source, str(source_path), "exec")
        if target_path.read_bytes() != source:
            temporary = target_path.with_suffix(target_path.suffix + f".tmp.{os.getpid()}")
            temporary.write_bytes(source)
            os.replace(temporary, target_path)
        observed = hashlib.sha256(target_path.read_bytes()).hexdigest()
        expected = hashlib.sha256(source).hexdigest()
        if observed != expected:
            raise RuntimeError(f"routed-scale overlay hash mismatch for {target_path}")
        print(f"verified routed-scale overlay: {relative_path} sha256={observed}", flush=True)
PY
fi
python3 - "$data_root" "$campaign_root/bootstrap/data-manifest.json" <<'PY' || finish $?
import hashlib
import json
import sys
from pathlib import Path

import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
data_root = Path(sys.argv[1])
manifest = json.loads(Path(sys.argv[2]).read_text())
for filename, expected in manifest["files"].items():
    path = data_root / filename
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected["sha256"]:
        raise RuntimeError(f"dataset hash mismatch for {filename}: {observed} != {expected['sha256']}")
print(json.dumps({"gpu": torch.cuda.get_device_name(0), "dataset_manifest": manifest}, sort_keys=True), flush=True)
PY
case "$mode" in
  smoke)
    expected_step=1
    routes=(adam_adam muon_actor muon_critic)
    extra_args=(trainer.total_training_steps=1 trainer.test_freq=1 trainer.save_freq=-1 data.train_batch_size=32 data.max_prompt_length=256 data.max_response_length=64 actor_rollout_ref.actor.ppo_mini_batch_size=16 critic.ppo_mini_batch_size=16)
    ;;
  full)
    expected_step=435
    routes=(adam_adam muon_actor muon_critic)
    extra_args=(trainer.save_freq=-1)
    ;;
  routed-smoke)
    expected_step=1
    routes=(routed_scale_adam_actor)
    extra_args=(trainer.total_training_steps=1 trainer.test_freq=1 trainer.save_freq=-1 data.train_batch_size=32 data.max_prompt_length=256 data.max_response_length=64 actor_rollout_ref.actor.ppo_mini_batch_size=16 critic.ppo_mini_batch_size=16)
    ;;
  routed-full)
    expected_step=435
    routes=(routed_scale_adam_actor)
    extra_args=(trainer.save_freq=-1)
    ;;
esac

for route in "${routes[@]}"; do
  route_root="$run_root/$route"
  mkdir "$route_root" || finish $?
  write_status running "route=$route"
  ROUTE="$route" SEED="$seed" MODEL_PATH="$model_root" DATA_ROOT="$data_root" OUTPUT_ROOT="$route_root" \
    bash "$verl_root/examples/ppo_trainer/run_qwen2_5_0_5b_gsm8k_optimizer_ablation.sh" \
      +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
      +critic.model.override_config.attn_implementation=sdpa \
      actor_rollout_ref.model.use_remove_padding=False \
      critic.model.use_remove_padding=False \
      "${extra_args[@]}" || finish $?
  metrics=$(find "$route_root" -name metrics.jsonl -type f -print -quit)
  if [[ -z "$metrics" || ! -s "$metrics" ]]; then
    echo "missing metrics for route=$route"
    finish 77
  fi
  python3 - "$metrics" "$expected_step" "$route" <<'PY' || finish $?
import json
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_step = int(sys.argv[2])
route = sys.argv[3]
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if not rows:
    raise RuntimeError(f"empty metrics for {route}")
steps = [int(row["step"]) for row in rows]
if max(steps) != expected_step:
    raise RuntimeError(f"terminal step mismatch for {route}: {max(steps)} != {expected_step}")
for row_index, row in enumerate(rows):
    for key, value in row.get("data", {}).items():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise RuntimeError(f"non-finite metric for {route} row={row_index} key={key}: {value}")
print(json.dumps({"route": route, "rows": len(rows), "terminal_step": max(steps)}), flush=True)
PY
done
python3 "$repo_root/collect_gsm8k_r4_result.py" \
  "$run_root" "$mode" "$seed" "$source_commit" "$expected_step" "${routes[@]}" || finish $?
finish 0
