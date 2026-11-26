#!/usr/bin/env bash
# Runs inside the eval image (docker/eval.Dockerfile) against the server on port 8000 (see quality_gsm8k.sh):
# lm-eval GSM8K, 5-shot, greedy, plain /v1/completions (no chat template, so Qwen3 does not think).
# LIMIT=<n> runs only the first n questions (smoke test).
set -euo pipefail
# datasets writes lock files next to its cache and /hf is read-only, so work on a copy of the prepared GSM8K.
mkdir -p /tmp/datasets && cp -r /hf/datasets/openai___gsm8k /tmp/datasets/
export HF_DATASETS_CACHE=/tmp/datasets HF_DATASETS_OFFLINE=1

# 16 requests in flight: 5-shot prompts are 750-1,400 tokens and answers ~120, so ~18k tokens, about BF16's KV size.
# A request preempted when KV runs out is recomputed later, which costs time; it is not a quality setting.
echo "$(date +%T) lm_eval $(python3 -c 'import importlib.metadata as m; print(m.version("lm_eval"))') gsm8k ${LIMIT:+limit=$LIMIT}"
lm_eval --model local-completions \
  --model_args "model=$MODEL,base_url=http://127.0.0.1:8000/v1/completions,tokenizer=$TOKENIZER,tokenized_requests=False,num_concurrent=16,max_retries=3,max_length=4096" \
  --tasks gsm8k --num_fewshot 5 --output_path /out/lm_eval --log_samples ${LIMIT:+--limit "$LIMIT"} \
  > /out/lm_eval.log 2>&1
# The per-question records are ~12 MB per run (each prompt is stored once per answer filter); xz makes that ~1 MB.
find /out/lm_eval -name 'samples_*.jsonl' -exec xz -9 {} +
echo "$(date +%T) done"
grep '^|' /out/lm_eval.log
