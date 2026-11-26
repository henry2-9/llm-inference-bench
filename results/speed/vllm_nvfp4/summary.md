# vllm_nvfp4

2026-11-24T15:02:46+08:00
model=nvidia/Qwen3-8B-NVFP4 snapshot=ccd10a893cbca613259517c3efe08e151ddf2b8e extra=--kv-cache-dtype fp8
image=vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
client=client_loop.sh client_image=vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
repo_commit=50d38ce
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W

Median over runs; spread = (max - min) / median across runs.

| in/out | conc | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | output tok/s | req/s | preempted | failed |
|---|---|---|---|---|---|---|---|---|---|
| 128/512 | 1 | 21.1 | 22.2 (±5%) | 7.1 | 7.2 | 140.1 | 0.3 | 0 | 0 |
| 128/512 | 4 | 60.4 | 67.2 | 7.2 | 7.3 | 547.9 | 1.1 | 0 | 0 |
| 128/512 | 16 | 205.6 (±3%) | 222.0 | 7.8 | 8.2 | 1948.3 | 3.8 | 0 | 0 |
| 128/512 | 64 | 460.9 | 833.6 | 12.7 | 13.5 | 4693.5 | 9.2 | 0 | 0 |
| 2048/256 | 1 | 216.2 | 220.8 | 7.2 | 7.2 | 125.0 | 0.5 | 0 | 0 |
| 2048/256 | 4 | 633.9 | 854.1 | 8.7 | 10.2 | 359.6 | 1.4 | 0 | 0 |
| 2048/256 | 16 | 830.3 | 3328.3 | 19.4 | 21.7 | 705.1 | 2.8 | 0 | 0 |
| 2048/256 | 64 | 874.9 | 13285.1 | 68.9 | 70.7 | 881.2 | 3.4 | 0 | 0 |

## Server log: KV cache

- `(EngineCore pid=147) INFO 11-24 07:03:59 [gpu_worker.py:692] Available KV cache memory: 11.68 GiB`
- `(EngineCore pid=147) INFO 11-24 07:03:59 [kv_cache_utils.py:2464] GPU KV cache size: 170,016 tokens, Maximum concurrency for 4,096 tokens per request: 41.51x`

Max processes on the GPU during the run: 3 (the server itself shows 1-2; more means something else ran)
