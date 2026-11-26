#!/usr/bin/env bash
# Usage: scripts/profile_all.sh [precision...]   (bf16, fp8, nvfp4; default: all three)
# profile.sh for both engines at each precision, with the server settings of the speed runs.
# Results go to results/$PREFIX/<engine>_<precision>/ (PREFIX defaults to profile), about 3 minutes per run;
# profile_summary.py then puts the runs side by side in results/$PREFIX/summary.md.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PREFIX=${PREFIX:-profile}

# profile.sh refuses a busy GPU, and the previous server may still be releasing its memory.
wait_idle() {
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    [ "$used" -le 1000 ] && [ "$apps" -eq 0 ] && return 0
    sleep 2
  done
}

for p in ${*:-bf16 fp8 nvfp4}; do
  case $p in
    bf16)  model=Qwen/Qwen3-8B;          extra=() ;;
    fp8)   model=nvidia/Qwen3-8B-FP8;    extra=(--kv-cache-dtype fp8) ;;
    nvfp4) model=nvidia/Qwen3-8B-NVFP4;  extra=(--kv-cache-dtype fp8) ;;
    *) echo "unknown precision: $p"; exit 1 ;;
  esac
  wait_idle; "$ROOT/scripts/profile.sh" vllm "$PREFIX/vllm_$p" "$model" "${extra[@]}"
  wait_idle; "$ROOT/scripts/profile.sh" trtllm "$PREFIX/trtllm_$p" "$model" "$ROOT/configs/trtllm_$p.yaml"
done
python3 -I "$ROOT/scripts/profile_summary.py" "$ROOT/results/$PREFIX" > /dev/null
echo "$(date +%T) wrote $ROOT/results/$PREFIX/summary.md"
