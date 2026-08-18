#!/usr/bin/env bash
# Shared idempotent compatibility setup for pinned VERL with vLLM 0.8.5.
# Caller defines campaign_root, verl_root, venv_python, and finish().
ensure_verl_vllm_compat() {
  export VLLM_USE_V1=1
  export TRITON_LIBCUDA_PATH=/lib/x86_64-linux-gnu
  "$venv_python" - <<'PY' || finish $?
  import sys
  print(f"campaign interpreter: executable={sys.executable} prefix={sys.prefix}", flush=True)
  PY
  # The pinned VERL snapshot now calls DataProto.to_tensordict(), whose
  # NonTensorStack conversion is only available with tensordict >= 0.10.  Older
  # campaigns were built with 0.8.3, so repair a reused venv once under a lock.
  "$venv_python" - "$campaign_root" <<'PY' || finish $?
  import fcntl
  import importlib.util
  import importlib.metadata
  import subprocess
  import sys
  from pathlib import Path

  from packaging.version import Version

  campaign_root = Path(sys.argv[1])
  lock_path = campaign_root / ".tensordict010-compat.lock"
  with lock_path.open("w") as lock:
      fcntl.flock(lock, fcntl.LOCK_EX)
      if importlib.util.find_spec("cloudpickle") is None:
          subprocess.run(
              [
                  sys.executable,
                  "-m",
                  "pip",
                  "install",
                  "--no-cache-dir",
                  "--no-deps",
                  "cloudpickle==3.1.1",
              ],
              check=True,
          )
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
  "$venv_python" - "$campaign_root" \
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
}