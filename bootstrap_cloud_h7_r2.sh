#!/usr/bin/env bash
set -u
root=$(cd "$(dirname "$0")" && pwd)
campaign_root=/home/jovyan/rl_muon/h7_online_cloud_r4
state_root="$campaign_root/bootstrap"
mkdir -p "$campaign_root"
if ! mkdir "$state_root"; then
  echo "cannot create bootstrap state root"
  exit 74
fi
log="$state_root/bootstrap.log"
exec > >(tee -a "$log") 2>&1
status=0
cd "$root"
python - <<'PY' || status=$?
import hashlib
import json
import platform
import tempfile
from pathlib import Path

import accelerate
import datasets
import numpy
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config

profile = json.loads(Path("cloud_h7_r2/profile.json").read_text())
observed = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "accelerate": accelerate.__version__,
    "numpy": numpy.__version__,
}
expected = {key: profile[key] for key in observed}
print(json.dumps({"expected": expected, "observed": observed}, sort_keys=True), flush=True)
if observed != expected:
    raise RuntimeError(f"Cloud compatibility profile mismatch: {observed} != {expected}")

expected_data = {
    Path("cloud_h7/data/imdb-train-efd331a311c6.jsonl"): ("efd331a311c64a5640a4ce529cce3278d44c91da1ca0c9bbbdc95c7465148a7c", 512),
    Path("cloud_h7/data/imdb-eval-8e4a88f98cd5.jsonl"): ("8e4a88f98cd5d0ce6815a05ba6b55049f4f75be7ac6aaa4c5546470916b271d1", 64),
}
for path, (expected_sha256, expected_count) in expected_data.items():
    payload = path.read_bytes()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    observed_count = len(payload.splitlines())
    if observed_sha256 != expected_sha256 or observed_count != expected_count:
        raise RuntimeError(f"dataset mismatch for {path}: sha256={observed_sha256}, count={observed_count}")

models = (
    ("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987"),
    ("lvwerra/distilbert-imdb", "0fc02cd68445b599a9cb2da2368050e7fb31d29a"),
    ("distilbert/distilbert-base-uncased-finetuned-sst-2-english", "714eb0fa89d2f80546fda750413ed43d93601a13"),
)
for model_id, revision in models:
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    print(json.dumps({"model_id": model_id, "revision": revision, "model_type": config.model_type}), flush=True)
tiny_config = Qwen2Config(
    vocab_size=32,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
)
tiny_model = AutoModelForCausalLM.from_config(tiny_config, torch_dtype=torch.float32)
if next(tiny_model.parameters()).dtype != torch.float32:
    raise RuntimeError("Transformers 4 torch_dtype loader contract failed")
with tempfile.TemporaryDirectory() as directory:
    tiny_model.save_pretrained(directory)
    loaded_model = AutoModelForCausalLM.from_pretrained(
        directory, torch_dtype=torch.float32
    )
    if next(loaded_model.parameters()).dtype != torch.float32:
        raise RuntimeError("Transformers 4 from_pretrained contract failed")
print("PROFILE_AND_MODEL_CONFIGS_OK", flush=True)
PY
if [[ "$status" -eq 0 ]]; then
  PYTHONPATH="$root/cloud_h7_r2/src" python -m unittest discover -s cloud_h7_r2/tests -p 'test_*.py' || status=$?
fi
printf '%s\n' "$status" > "$state_root/exit"
if [[ "$status" -eq 0 ]]; then
  printf '{"state":"complete","profile":"cloud-py310-r4"}\n' > "$state_root/status.json.tmp"
else
  printf '{"state":"failed","profile":"cloud-py310-r4","exit":%s}\n' "$status" > "$state_root/status.json.tmp"
fi
mv "$state_root/status.json.tmp" "$state_root/status.json"
tail -120 "$log"
# Preserve platform logs. Scientific state is the durable NFS status/exit pair.
exit 0
