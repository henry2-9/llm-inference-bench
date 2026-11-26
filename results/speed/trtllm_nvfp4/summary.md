# trtllm_nvfp4

2026-11-24T16:33:14+08:00
model=nvidia/Qwen3-8B-NVFP4 snapshot=ccd10a893cbca613259517c3efe08e151ddf2b8e config=trtllm_nvfp4.yaml extra=
image=nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a
client=client_loop.sh client_image=vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
repo_commit=92d1184
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W

Median over runs; spread = (max - min) / median across runs.

| in/out | conc | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | output tok/s | req/s | preempted | failed |
|---|---|---|---|---|---|---|---|---|---|
| 128/512 | 1 | 24.2 | 25.1 (±5%) | 7.4 | 7.4 | 135.3 | 0.3 | — | 0 |
| 128/512 | 4 | 64.0 | 70.0 | 7.5 | 7.5 | 525.3 | 1.0 | — | 0 |
| 128/512 | 16 | 210.8 | 223.2 | 8.2 | 8.5 | 1865.3 | 3.6 | — | 0 |
| 128/512 | 64 | 640.7 (±22%) | 837.7 | 12.7 | 13.8 | 4577.2 | 8.9 | — | 0 |
| 2048/256 | 1 | 214.7 | 221.2 | 8.4 | 8.4 | 108.7 | 0.4 | — | 0 |
| 2048/256 | 4 | 824.4 | 839.2 | 8.5 | 10.7 | 340.6 | 1.3 | — | 0 |
| 2048/256 | 16 | 1804.9 | 3321.4 | 16.6 | 22.5 | 677.8 | 2.6 | — | 0 |
| 2048/256 | 64 | 1905.3 | 13075.9 | 66.5 | 72.3 | 867.0 | 3.4 | — | 0 |

## Server log: KV cache

- `[11/24/2026-08:33:35] [TRT-LLM] [I] [_torch] max_tokens 65536 is provided. Allowed quota from max_tokens is 4.7368421060964465GiB. New quota is 4.7368421060964465GiB`
- `[11/24/2026-08:33:35] [TRT-LLM] [I] [_torch] KV cache manager v2 device quota set to 4.7368421060964465GiB`
- `[11/24/2026-08:34:30] [TRT-LLM] [I] [_torch] Peak memory during memory usage profiling (torch + non-torch): 15.88 GiB, available KV cache memory when calculating max tokens: 11.13 GiB, fraction is set 0.9, kv size per token is 73728 bytes/token. device total memory 23.51 GiB, temporary kv cache memory during profiling 4.74 GiB`
- `[11/24/2026-08:34:30] [TRT-LLM] [I] [_torch] max_tokens 161515 is provided. Allowed quota from max_tokens is 11.67405781056732GiB. New quota is 11.133428191766143GiB`
- `[11/24/2026-08:34:30] [TRT-LLM] [I] [_torch] KV cache manager v2 device quota set to 11.133428191766143GiB`

Max processes on the GPU during the run: 2 (the server itself shows 1-2; more means something else ran)
