#!/usr/bin/env bash
set -u

repo_root=$(cd "$(dirname "$0")" && pwd)
source_commit=${RL_MUON_SOURCE_COMMIT:?RL_MUON_SOURCE_COMMIT is required}
[[ "$(git -C "$repo_root" rev-parse HEAD)" == "$source_commit" ]] || {
  echo "source commit mismatch"
  exit 73
}
campaign_root=${RL_MUON_CAMPAIGN_ROOT:?RL_MUON_CAMPAIGN_ROOT is required}
venv_root="$campaign_root/venv"
diag_root="$campaign_root/ray-diagnostic-$(date -u +%Y%m%dT%H%M%SZ)-$$"
ray_tmp="/tmp/rlm-ray-$$"
mkdir -p "$diag_root" "$ray_tmp" || exit $?
log="$diag_root/diagnostic.log"
exec > >(tee -a "$log") 2>&1

export PATH="$venv_root/bin:$PATH"
export PYTHONPATH="$repo_root:$campaign_root/verl"
export VLLM_USE_V1=1
export TRITON_LIBCUDA_PATH=/lib/x86_64-linux-gnu
export RAY_TMPDIR="$ray_tmp"

printf '%s\n' '-- limits --'
ulimit -a
printf '%s\n' '-- cgroup --'
for path in /sys/fs/cgroup/pids.max /sys/fs/cgroup/pids.current /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory.current; do
  if [[ -r "$path" ]]; then printf '%s=' "$path"; cat "$path"; fi
done
printf '%s\n' '-- filesystems --'
df -h /dev/shm /tmp "$campaign_root" || true
printf '%s\n' '-- memory --'
free -h || true
printf '%s\n' '-- versions --'
python3 - <<'PY'
import os
import platform
import ray
print({"python": platform.python_version(), "ray": ray.__version__, "RAY_TMPDIR": os.environ.get("RAY_TMPDIR")}, flush=True)
PY

set +e
python3 - "$ray_tmp" <<'PY'
import os
import sys
import ray

temp_dir = sys.argv[1]
print("calling ray.init", flush=True)
context = ray.init(
    _temp_dir=temp_dir,
    runtime_env={
        "env_vars": {
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "VLLM_USE_V1": "1",
            "TRITON_LIBCUDA_PATH": "/lib/x86_64-linux-gnu",
        }
    },
)
print(f"ray context={context}", flush=True)

@ray.remote
def probe():
    import os
    return {"pid": os.getpid(), "vllm_v1": os.environ.get("VLLM_USE_V1")}

print(f"remote result={ray.get(probe.remote())}", flush=True)
ray.shutdown()
PY
code=$?
set -e

printf '%s\n' "ray_probe_exit=$code"
printf '%s\n' '-- ray logs --'
find "$ray_tmp" -type f \( -name 'raylet.out' -o -name 'raylet.err' -o -name 'gcs_server.out' -o -name 'gcs_server.err' -o -name 'dashboard*.log' -o -name 'runtime_env*.log' \) -print0 2>/dev/null |
while IFS= read -r -d '' path; do
  printf '\n===== %s =====\n' "$path"
  tail -n 160 "$path"
done
printf 'RL_MUON_RAY_DIAGNOSTIC {"source_commit":"%s","exit":%s,"root":"%s"}\n' "$source_commit" "$code" "$diag_root"
exit 0
