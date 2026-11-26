#!/usr/bin/env bash
# Run the standard speed, GSM8K and profiling suites, then regenerate the figures.
# About 4 hours on an idle RTX 4090. Existing results may be overwritten.
# Patched-engine builds and quantized GSM8K reruns are separate experiments.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
S=$ROOT/scripts

wait_idle() {
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    [ "$used" -le 1000 ] && [ "$apps" -eq 0 ] && return 0
    sleep 2
  done
}

# Speed. vLLM runs first at each precision: configs/trtllm_*.yaml carry the KV size vLLM reported.
wait_idle; "$S/bench_vllm.sh"   speed/vllm_bf16    Qwen/Qwen3-8B
wait_idle; "$S/bench_trtllm.sh" speed/trtllm_bf16  Qwen/Qwen3-8B         "$ROOT/configs/trtllm_bf16.yaml"
wait_idle; "$S/bench_vllm.sh"   speed/vllm_fp8     nvidia/Qwen3-8B-FP8   --kv-cache-dtype fp8
wait_idle; "$S/bench_trtllm.sh" speed/trtllm_fp8   nvidia/Qwen3-8B-FP8   "$ROOT/configs/trtllm_fp8.yaml"
wait_idle; "$S/bench_vllm.sh"   speed/vllm_nvfp4   nvidia/Qwen3-8B-NVFP4 --kv-cache-dtype fp8
wait_idle; "$S/bench_trtllm.sh" speed/trtllm_nvfp4 nvidia/Qwen3-8B-NVFP4 "$ROOT/configs/trtllm_nvfp4.yaml"

# Quality: BF16, FP8, NVFP4 and the BF16 rerun, on both engines.
wait_idle; "$S/quality_gsm8k.sh"

# Profiles: the three default captures, then the batch / context sweep, then FP8 with XQA forced.
wait_idle; "$S/profile_all.sh"
wait_idle; PREFIX=profile_sweep \
  CAPTURES="c1_in3900:3900:64:2:1 c2_in128:128:64:4:2 c4_in2048:2048:64:8:4 c16_in128:128:64:32:16 c64_in128:128:64:128:64" \
  "$S/profile_all.sh"
wait_idle; SERVER_ENV="TRTLLM_FORCE_XQA=1" \
  CAPTURES="c1_in128:128:64:4:1 c1_in2048:2048:64:2:1 c1_in3900:3900:64:2:1 c2_in128:128:64:4:2 c4_in128:128:64:8:4" \
  "$S/profile.sh" trtllm profile_force_xqa/trtllm_fp8 nvidia/Qwen3-8B-FP8 "$ROOT/configs/trtllm_fp8.yaml"

python3 -I "$S/plot_figures.py"
