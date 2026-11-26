# vllm_bf16

2026-10-31T01:45:08+08:00
model=Qwen/Qwen3-8B snapshot=b968826d9c46dd6066d109eabc6255188de91218 extra=
image=vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
repo_commit=a1473b6
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W

Median over runs; spread = (max - min) / median across runs.

| in/out | conc | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | output tok/s | req/s | failed |
|---|---|---|---|---|---|---|---|---|
| 128/512 | 1 | 26.6 | 27.3 | 17.3 | 17.4 | 57.8 | 0.1 | 0 |
| 128/512 | 4 | 70.9 | 76.6 | 18.2 | 18.3 | 218.3 | 0.4 | 0 |
| 128/512 | 16 | 229.8 | 239.1 | 19.2 | 19.5 | 818.0 | 1.6 | 0 |
| 128/512 | 64 | 857.0 | 7688.9 | 27.1 | 43.0 | 1763.5 | 3.4 | 0 |
| 2048/256 | 1 | 217.2 | 219.3 | 17.6 | 17.6 | 54.3 | 0.2 | 0 |
| 2048/256 | 4 | 647.1 | 862.2 | 19.9 | 21.4 | 178.4 | 0.7 | 0 |
| 2048/256 | 16 | 7794.0 | 9072.5 | 26.3 | 29.9 | 284.0 | 1.1 | 0 |
| 2048/256 | 64 | 45187.8 | 50779.5 | 26.3 | 29.8 | 306.1 | 1.2 | 0 |

## Server log: KV cache

- `(EngineCore pid=100) INFO 10-30 17:46:34 [gpu_worker.py:692] Available KV cache memory: 2.83 GiB`
- `(EngineCore pid=100) INFO 10-30 17:46:34 [kv_cache_utils.py:2464] GPU KV cache size: 20,608 tokens, Maximum concurrency for 4,096 tokens per request: 5.03x`

Max processes on the GPU during the run: 1 (vLLM itself shows 1-2; more means something else ran)
