#!/usr/bin/env bash
# Usage: scripts/quality_gsm8k.sh [run...]   (bf16, fp8, nvfp4, bf16_rerun; default: all four)
# Quality check: GSM8K through each engine's server, started with the same settings as the speed runs
# (bench_vllm.sh / bench_trtllm.sh with CLIENT=gsm8k.sh). Results go to results/$PREFIX/<engine>_<precision>/,
# then summarize_quality.py compares every run with BF16 on the same engine, question by question.
# BF16 reruns measure variability under identical settings.
# For quantized reruns, use the bench scripts with CLIENT=gsm8k.sh and
# CLIENT_IMG=llm-bench-eval:lm_eval-0.4.13, writing to $PREFIX/<engine>_<precision>_rerun.
# PREFIX defaults to gsm8k; LIMIT=<n> evaluates only the first n questions (smoke test).
#
# One-time setup (needs network): GSM8K into the HF datasets cache, which the client copies from.
#   docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HOME=/hf -v ~/.cache/huggingface:/hf \
#     --entrypoint python3 llm-bench-eval:lm_eval-0.4.13 -c "import datasets; datasets.load_dataset('openai/gsm8k', 'main')"
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PREFIX=${PREFIX:-gsm8k}
export CLIENT=gsm8k.sh CLIENT_IMG=llm-bench-eval:lm_eval-0.4.13

docker image inspect $CLIENT_IMG >/dev/null 2>&1 \
  || docker build -t $CLIENT_IMG -f "$ROOT/docker/eval.Dockerfile" "$ROOT/docker"
[ -d ~/.cache/huggingface/datasets/openai___gsm8k ] || { echo "GSM8K not in the datasets cache; see the setup above"; exit 1; }

# The bench scripts refuse a busy GPU, and the previous server may still be releasing its memory.
wait_idle() {
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    [ "$used" -le 1000 ] && [ "$apps" -eq 0 ] && return 0
    sleep 2
  done
}

for p in ${*:-bf16 fp8 nvfp4 bf16_rerun}; do
  case $p in
    bf16)
      wait_idle; "$ROOT/scripts/bench_vllm.sh" "$PREFIX/vllm_bf16" Qwen/Qwen3-8B
      wait_idle; "$ROOT/scripts/bench_trtllm.sh" "$PREFIX/trtllm_bf16" Qwen/Qwen3-8B "$ROOT/configs/trtllm_bf16.yaml" ;;
    fp8)
      wait_idle; "$ROOT/scripts/bench_vllm.sh" "$PREFIX/vllm_fp8" nvidia/Qwen3-8B-FP8 --kv-cache-dtype fp8
      wait_idle; "$ROOT/scripts/bench_trtllm.sh" "$PREFIX/trtllm_fp8" nvidia/Qwen3-8B-FP8 "$ROOT/configs/trtllm_fp8.yaml" ;;
    nvfp4)
      wait_idle; "$ROOT/scripts/bench_vllm.sh" "$PREFIX/vllm_nvfp4" nvidia/Qwen3-8B-NVFP4 --kv-cache-dtype fp8
      wait_idle; "$ROOT/scripts/bench_trtllm.sh" "$PREFIX/trtllm_nvfp4" nvidia/Qwen3-8B-NVFP4 "$ROOT/configs/trtllm_nvfp4.yaml" ;;
    bf16_rerun)
      wait_idle; "$ROOT/scripts/bench_vllm.sh" "$PREFIX/vllm_bf16_rerun" Qwen/Qwen3-8B
      wait_idle; "$ROOT/scripts/bench_trtllm.sh" "$PREFIX/trtllm_bf16_rerun" Qwen/Qwen3-8B "$ROOT/configs/trtllm_bf16.yaml" ;;
    *) echo "unknown run: $p"; exit 1 ;;
  esac
done
python3 -I "$ROOT/scripts/summarize_quality.py" "$ROOT/results/$PREFIX"
