# trtllm_bf16

2026-10-31T15:03:35+08:00
model=Qwen/Qwen3-8B snapshot=b968826d9c46dd6066d109eabc6255188de91218 config=trtllm_bf16.yaml extra=
image=nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a
client_image=vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
repo_commit=24d23c3-dirty
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W

Median over runs; spread = (max - min) / median across runs.

| in/out | conc | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | output tok/s | req/s | failed |
|---|---|---|---|---|---|---|---|---|
| 128/512 | 1 | 28.3 | 29.2 | 17.1 | 17.1 | 58.3 | 0.1 | 0 |
| 128/512 | 4 | 71.1 | 77.8 | 18.2 | 18.3 | 218.0 | 0.4 | 0 |
| 128/512 | 16 | 214.9 | 224.2 | 19.3 | 19.6 | 812.5 | 1.6 | 0 |
| 128/512 | 64 | 833.3 | 8259.7 | 28.2 | 42.9 | 1706.6 | 3.3 | 0 |
| 2048/256 | 1 | 215.2 | 216.3 | 17.4 | 17.4 | 55.2 | 0.2 | 0 |
| 2048/256 | 4 | 830.6 | 837.5 | 19.3 | 21.7 | 177.8 | 0.7 | 0 |
| 2048/256 | 16 | 6984.7 | 9102.5 | 26.4 | 29.8 | 291.9 | 1.1 | 0 |
| 2048/256 | 64 | 45131.8 | 49043.2 | 26.5 | 28.4 | 307.7 | 1.2 | 0 |

## Server log: KV cache

- `[10/31/2026-07:04:08] [TRT-LLM] [I] [_torch] max_tokens 19578 is provided. Allowed quota from max_tokens is 2.83013594802469GiB. New quota is 2.83013594802469GiB`
- `[10/31/2026-07:04:08] [TRT-LLM] [I] [_torch] KV cache manager v2 device quota set to 2.83013594802469GiB`
- `[10/31/2026-07:05:03] [TRT-LLM] [I] [_torch] Peak memory during memory usage profiling (torch + non-torch): 21.86 GiB, available KV cache memory when calculating max tokens: 4.03 GiB, fraction is set 0.9, kv size per token is 147456 bytes/token. device total memory 23.51 GiB, temporary kv cache memory during profiling 2.83 GiB`
- `[10/31/2026-07:05:03] [TRT-LLM] [I] [_torch] max_tokens 19578 is provided. Allowed quota from max_tokens is 2.83013594802469GiB. New quota is 2.83013594802469GiB`
- `[10/31/2026-07:05:03] [TRT-LLM] [I] [_torch] KV cache manager v2 device quota set to 2.83013594802469GiB`

Max processes on the GPU during the run: 2 (the server itself shows 1-2; more means something else ran)
