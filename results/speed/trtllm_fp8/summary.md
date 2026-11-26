# trtllm_fp8

2026-10-31T16:31:43+08:00
model=nvidia/Qwen3-8B-FP8 snapshot=2cebc4c89e25abc17668c81b01dceaf3d8b914d5 config=trtllm_fp8.yaml extra=
image=nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a
client_image=vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
repo_commit=c91dea6-dirty
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W

Median over runs; spread = (max - min) / median across runs.

| in/out | conc | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | output tok/s | req/s | preempted | failed |
|---|---|---|---|---|---|---|---|---|---|
| 128/512 | 1 | 26.5 | 27.5 | 11.7 | 11.7 | 85.3 | 0.2 | — | 0 |
| 128/512 | 4 | 52.3 | 58.1 (±6%) | 10.3 | 10.3 | 383.8 | 0.7 | — | 0 |
| 128/512 | 16 | 130.3 | 140.7 | 11.3 | 11.4 | 1394.1 | 2.7 | — | 0 |
| 128/512 | 64 | 425.9 (±3%) | 471.7 | 13.9 | 14.6 | 4351.0 | 8.5 | — | 0 |
| 2048/256 | 1 | 114.9 | 117.6 | 11.7 | 11.7 | 82.7 | 0.3 | — | 0 |
| 2048/256 | 4 | 445.8 | 457.2 | 11.4 | 12.7 | 305.8 | 1.2 | — | 0 |
| 2048/256 | 16 | 1027.3 | 1832.1 | 16.9 | 20.3 | 767.2 | 3.0 | — | 0 |
| 2048/256 | 64 | 2011.1 | 13393.0 | 44.3 | 54.5 | 1143.7 | 4.5 | — | 0 |

## Server log: KV cache

- `[10/31/2026-08:32:12] [TRT-LLM] [I] [_torch] max_tokens 65536 is provided. Allowed quota from max_tokens is 4.7368421060964465GiB. New quota is 4.7368421060964465GiB`
- `[10/31/2026-08:32:12] [TRT-LLM] [I] [_torch] KV cache manager v2 device quota set to 4.7368421060964465GiB`
- `[10/31/2026-08:33:06] [TRT-LLM] [I] [_torch] Peak memory during memory usage profiling (torch + non-torch): 17.65 GiB, available KV cache memory when calculating max tokens: 9.54 GiB, fraction is set 0.9, kv size per token is 73728 bytes/token. device total memory 23.51 GiB, temporary kv cache memory during profiling 4.74 GiB`
- `[10/31/2026-08:33:06] [TRT-LLM] [I] [_torch] max_tokens 119943 is provided. Allowed quota from max_tokens is 8.66929706837982GiB. New quota is 8.66929706837982GiB`
- `[10/31/2026-08:33:06] [TRT-LLM] [I] [_torch] KV cache manager v2 device quota set to 8.66929706837982GiB`

Max processes on the GPU during the run: 2 (the server itself shows 1-2; more means something else ran)
