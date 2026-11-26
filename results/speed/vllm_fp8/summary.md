# vllm_fp8

2026-10-31T16:12:24+08:00
model=nvidia/Qwen3-8B-FP8 snapshot=2cebc4c89e25abc17668c81b01dceaf3d8b914d5 extra=--kv-cache-dtype fp8
image=vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
repo_commit=c91dea6-dirty
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W

Median over runs; spread = (max - min) / median across runs.

| in/out | conc | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | output tok/s | req/s | preempted | failed |
|---|---|---|---|---|---|---|---|---|---|
| 128/512 | 1 | 19.6 | 20.0 | 11.9 | 11.9 | 83.9 | 0.2 | 0 | 0 |
| 128/512 | 4 | 48.9 | 55.1 | 12.0 | 12.1 | 330.3 | 0.6 | 0 | 0 |
| 128/512 | 16 | 155.6 | 166.9 | 12.6 | 12.8 | 1242.7 | 2.4 | 0 | 0 |
| 128/512 | 64 | 317.9 (±10%) | 554.4 | 16.0 | 16.5 | 3857.7 | 7.5 | 0 | 0 |
| 2048/256 | 1 | 141.2 | 144.6 | 12.1 | 12.1 | 79.5 | 0.3 | 0 | 0 |
| 2048/256 | 4 | 409.1 | 553.8 | 13.2 | 14.1 | 270.7 | 1.1 | 0 | 0 |
| 2048/256 | 16 | 537.6 | 2151.0 | 20.7 | 22.1 | 698.3 | 2.7 | 0 | 0 |
| 2048/256 | 64 | 1717.5 | 14199.9 | 51.1 | 67.0 | 1033.4 | 4.0 | 20 | 0 |

## Server log: KV cache

- `(EngineCore pid=147) INFO 10-31 08:13:25 [gpu_worker.py:692] Available KV cache memory: 8.67 GiB`
- `(EngineCore pid=147) INFO 10-31 08:13:25 [kv_cache_utils.py:2464] GPU KV cache size: 126,256 tokens, Maximum concurrency for 4,096 tokens per request: 30.82x`

Max processes on the GPU during the run: 1 (the server itself shows 1-2; more means something else ran)
