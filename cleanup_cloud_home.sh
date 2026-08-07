#!/usr/bin/env bash
# Report and reclaim space on the Cloud persistent home.
#
# The GSM8K bootstrap stopped being able to start at all:
#   mkdir: cannot create directory '/home/jovyan/.mlspace-logs/<job>':
#          No space left on device
# It never reached pip; it hung at container start, which is why its logs could
# not even be fetched. Space accumulates because every campaign revision creates
# its own root and its own PYTHONUSERBASE.
#
# Keeps the current campaign root, deletes stale ones and regenerable caches.
set -uo pipefail

KEEP_REVISION="${RL_MUON_KEEP_REVISION:-}"
HOME_DIR=/home/jovyan

echo "=== BEFORE ==="
df -h "$HOME_DIR" 2>&1 || true
echo
echo "--- top level ---"
du -xhd1 "$HOME_DIR" 2>/dev/null | sort -rh | head -20
echo
echo "--- campaign roots ---"
du -xhd1 "$HOME_DIR/rl_muon" 2>/dev/null | sort -rh | head -20
echo
echo "--- python user bases ---"
du -xhd0 "$HOME_DIR"/.local-gsm8k-* 2>/dev/null | sort -rh | head -10

echo
echo "=== RECLAIM ==="

# Job logs are pure history and are what actually failed to be written.
if [ -d "$HOME_DIR/.mlspace-logs" ]; then
  echo "removing .mlspace-logs"
  rm -rf "$HOME_DIR/.mlspace-logs"/* 2>/dev/null || true
fi

# pip and HF caches are regenerable.
for cache in "$HOME_DIR/.cache/pip" "$HOME_DIR/.cache/huggingface/hub" \
             "$HOME_DIR/.cache/uv"; do
  [ -d "$cache" ] && { echo "removing $cache"; rm -rf "$cache" 2>/dev/null || true; }
done

# Stale campaign roots: keep only the one this campaign is using.
if [ -d "$HOME_DIR/rl_muon" ]; then
  for root in "$HOME_DIR"/rl_muon/gsm8k_ppo_r4-*; do
    [ -d "$root" ] || continue
    base=$(basename "$root")
    if [ -n "$KEEP_REVISION" ] && [ "$base" = "gsm8k_ppo_r4-$KEEP_REVISION" ]; then
      echo "keeping $base"
      continue
    fi
    echo "removing stale campaign root $base"
    rm -rf "$root" 2>/dev/null || true
  done
fi

# Stale environments from earlier r-numbers; keep the r4 one in use.
for env in "$HOME_DIR"/.local-gsm8k-*; do
  [ -d "$env" ] || continue
  case "$(basename "$env")" in
    .local-gsm8k-vllm085-r4) echo "keeping $(basename "$env")" ;;
    *) echo "removing stale env $(basename "$env")"; rm -rf "$env" 2>/dev/null || true ;;
  esac
done

echo
echo "=== AFTER ==="
df -h "$HOME_DIR" 2>&1 || true
echo
du -xhd1 "$HOME_DIR" 2>/dev/null | sort -rh | head -20

echo
echo "RL_MUON_CLEANUP_DONE"
