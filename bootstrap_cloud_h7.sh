#!/usr/bin/env bash
set -u
root=$(cd "$(dirname "$0")" && pwd)
state_root=/home/jovyan/rl_muon/h7_online_cloud_r1/bootstrap
mkdir -p "$state_root"
log="$state_root/bootstrap.log"
exec > >(tee -a "$log") 2>&1
status=0
python - <<'PY' || status=$?
import json
from pathlib import Path
import torch
import transformers
import datasets
import accelerate
import numpy
expected = {
    "torch": "2.10.0+cu128",
    "transformers": "5.14.1",
    "datasets": "5.0.1",
    "accelerate": "1.14.0",
    "numpy": "2.5.1",
}
observed = {
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "accelerate": accelerate.__version__,
    "numpy": numpy.__version__,
}
print(json.dumps({"expected": expected, "observed": observed}, sort_keys=True), flush=True)
for name, version in expected.items():
    if observed[name] != version:
        raise RuntimeError(f"version mismatch for {name}: {observed[name]} != {version}")
for config_path in sorted(Path("cloud_h7/configs").glob("config_seed*.json")):
    config = json.loads(config_path.read_text())
    for key in ("train_prompts_path", "eval_prompts_path"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(config[key])
print("BOOTSTRAP_OK", flush=True)
PY
printf '%s\n' "$status" > "$state_root/exit"
if [[ "$status" -eq 0 ]]; then
  printf '{"state":"complete"}\n' > "$state_root/status.json"
else
  printf '{"state":"failed","exit":%s}\n' "$status" > "$state_root/status.json"
fi
tail -80 "$log"
exit 0
