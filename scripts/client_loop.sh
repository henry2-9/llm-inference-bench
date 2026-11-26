#!/usr/bin/env bash
# Runs inside the vLLM container (see bench_vllm.sh / bench_trtllm.sh): one warmup, then the grid.
# Every run reuses the same seed per run index, so engines and precisions see the same prompts.
# WORKLOADS / CONCURRENCIES override the grid (for smoke tests); the defaults are the full grid.
set -euo pipefail
# Sampling is explicit because each server otherwise uses its own defaults: vLLM reads the model's
# generation_config.json (these values, also what speed/vllm_bf16 ran with), TRT-LLM uses temperature 1.0 and no top-k.
common=(--backend vllm --base-url http://127.0.0.1:8000 --model "$MODEL" --tokenizer "$TOKENIZER"
        --dataset-name random --random-range-ratio 0 --ignore-eos
        --temperature 0.6 --top-p 0.95 --top-k 20
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99)

# vLLM's cumulative count of requests preempted (KV ran out, recomputed later); NA when the server has no such metric.
preemptions() {
  { curl -sf 127.0.0.1:8000/metrics || true; } \
    | awk '/^vllm:num_preemptions_total/{s+=$2; n=1} END{if (n) printf "%d\n", s; else print "NA"}'
}

vllm bench serve "${common[@]}" --random-input-len 128 --random-output-len 64 \
  --num-prompts 16 --max-concurrency 16 --seed 999 > /out/warmup.log 2>&1
echo "$(date +%T) warmup done"

for wl in ${WORKLOADS:-128:512 2048:256}; do
  in=${wl%:*}; out=${wl#*:}
  for c in ${CONCURRENCIES:-1 4 16 64}; do
    n=$(( c * 4 > 8 ? c * 4 : 8 ))
    for r in $(seq 1 "$RUNS"); do
      f=in${in}_out${out}_c${c}_r${r}
      echo "$(date +%T) $f (num_prompts=$n)"
      p0=$(preemptions)
      vllm bench serve "${common[@]}" --random-input-len "$in" --random-output-len "$out" \
        --num-prompts "$n" --max-concurrency "$c" --seed $((1000 + r)) \
        --save-result --result-dir /out/raw --result-filename "$f.json" > "/out/raw/$f.log" 2>&1
      echo "$p0 $(preemptions)" > "/out/raw/$f.preemptions"
    done
  done
done
