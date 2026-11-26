# profile_rc4 / profile_rc4_patched: the same decode profiles on TensorRT-LLM 1.4.0rc4

`profile_rc4` is the unmodified `nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc4` container (`TRT_IMG=...` for
`scripts/profile.sh`). `profile_rc4_patched` is that container with the engine libraries replaced by a build of
tag `v1.4.0rc4`, in which the FP8 KV cache may use multi-block attention on GPUs older than SM90. In this
version the condition is in `cpp/tensorrt_llm/thop/attentionOp.cpp`, in two places, and both get the same change:

```diff
-    if (xqaParams.kv_cache_data_type == DATA_TYPE_INT8
-        || (xqaParams.kv_cache_data_type == DATA_TYPE_E4M3 && mSM < kSM_90))
+    if (xqaParams.kv_cache_data_type == DATA_TYPE_INT8)
```

Build: `nvcr.io/nvidia/tensorrt-llm/devel:1.4.0rc4`, after `git lfs pull`, with the command in
[`../profile_patched/README.md`](../profile_patched/README.md) but `-j 3`. A single CUDA compile job used up to 7 GB of
memory, so more jobs did not fit in 32 GB next to a desktop session; the build took 2 h 14 min.
Replaced files are the same five as for 1.4.0rc2.

Decode step, ms (median; one capture per cell):

| checkpoint | batch, context | unmodified | patched |
|---|---|---|---|
| FP8 | 1, 128 | 11.76 | 10.21 |
| FP8 | 1, 2048 | 11.78 | 10.32 |
| FP8 | 1, 3900 | 11.98 | 10.47 |
| FP8 | 2, 128 | 13.50 | 10.32 |
| FP8 | 2, 2048 | 13.60 | 10.64 |
| FP8 | 4, 128 | 10.28 | 10.53 |
| FP8 | 4, 2048 | 11.13 | 11.11 |
| NVFP4 | 1, 128 | 7.28 | 7.43 |
| NVFP4 | 1, 2048 | 8.14 | 7.59 |
| NVFP4 | 1, 3900 | 8.96 | 7.75 |
| NVFP4 | 2, 128 | 7.37 | 7.54 |
| NVFP4 | 2, 2048 | 8.21 | 7.91 |
| NVFP4 | 4, 128 | 7.41 | 7.66 |
| NVFP4 | 4, 2048 | 8.29 | 8.31 |

Unmodified, the FP8 checkpoint still falls back to `masked_multihead_attention_kernel` at batch 1 and 2
(2.18 and 3.81 ms per step). The numbers match 1.4.0rc2 (`profile`, `profile_sweep`, `profile_patched`) within 0.06 ms.

GSM8K on this version (`../gsm8k_rc4`: unmodified, run twice; `../gsm8k_rc4_patched`), strict-match,
1,319 questions, and the number of questions that flip between right and wrong:

| checkpoint | unmodified, two runs | patched | unmodified run 1 vs run 2 | patched vs unmodified run 1 / run 2 |
|---|---|---|---|---|
| FP8 | 86.7% / 87.1% | 87.0% | 53 | 76 / 71 |
| NVFP4 | 85.8% / 86.6% | 86.2% | 38 | 47 / 43 |

`bench_trtllm.sh` was run with `CLIENT=gsm8k.sh`, `TRT_IMG` set to the 1.4.0rc4 image and, for the patched
runs, the mounts in `SERVER_DOCKER_ARGS` (see each `env.txt`).
