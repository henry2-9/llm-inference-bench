# vllm_fp8: GPU kernel time (Nsight Systems, CUDA graphs traced per node)

```
2026-11-24T20:35:31+08:00
engine=vllm model=nvidia/Qwen3-8B-FP8 snapshot=2cebc4c89e25abc17668c81b01dceaf3d8b914d5 extra=--kv-cache-dtype fp8
image=vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
nsys=NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0 args=--session-new=bench --trace=cuda --cuda-graph-trace=node
repo_commit=404f449
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W
```

## c1_in128: 4 requests, 128 in / 64 out, 1 at a time

Client: TPOT p50 11.72 ms, TTFT p50 22.1 ms, 4 completed. 117140 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 252 of 255 steps): 11.73 ms, of which kernels 11.68 ms (99.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 10.739 | 91.5% |
| attention | 0.412 | 3.5% |
| other | 0.226 | 1.9% |
| norm | 0.168 | 1.4% |
| sampling | 0.082 | 0.7% |
| activation | 0.047 | 0.4% |
| GPU idle | 0.055 | 0.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `cutlass::Kernel2` | linear | 108 | 4.727 |
| `cutlass::Kernel2` | linear | 36 | 4.658 |
| `internal::gemvx::kernel` | linear | 1 | 1.317 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 0.305 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.108 |
| `triton_red_fused_4` | other | 35 | 0.107 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 35 | 0.092 |
| `flashinfer::PersistentVariableLengthMergeStatesKernel` | other | 36 | 0.071 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 36 | 0.071 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_silu_` | activation | 36 | 0.047 |
| `triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_view_` | linear | 36 | 0.038 |

**Prefill** (first kernel of a request to its first sampled token; median of 4 requests): 21.0 ms, of which kernels 14.3 ms (68.0%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 13.063 | 62.2% |
| attention | 0.603 | 2.9% |
| norm | 0.213 | 1.0% |
| other | 0.180 | 0.9% |
| activation | 0.164 | 0.8% |
| sampling | 0.070 | 0.3% |
| GPU idle | 6.711 | 32.0% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `cutlass::Kernel2` | linear | 108 | 5.984 |
| `cutlass::Kernel2` | linear | 36 | 5.697 |
| `internal::gemvx::kernel` | linear | 1 | 1.324 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 0.472 |
| `triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_silu_` | activation | 36 | 0.164 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.130 |
| `triton_red_fused_4` | other | 35 | 0.117 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 35 | 0.113 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 36 | 0.094 |
| `triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_view_` | linear | 36 | 0.059 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | sampling | 1 | 0.020 |

## c1_in2048: 2 requests, 2048 in / 64 out, 1 at a time

Client: TPOT p50 11.88 ms, TTFT p50 143.1 ms, 2 completed. 58570 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 126 of 127 steps): 11.89 ms, of which kernels 11.84 ms (99.6%).

| part | ms per step | % of step |
|---|---|---|
| linear | 10.742 | 90.4% |
| attention | 0.554 | 4.7% |
| other | 0.236 | 2.0% |
| norm | 0.168 | 1.4% |
| sampling | 0.084 | 0.7% |
| activation | 0.047 | 0.4% |
| GPU idle | 0.053 | 0.4% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `cutlass::Kernel2` | linear | 108 | 4.721 |
| `cutlass::Kernel2` | linear | 36 | 4.666 |
| `internal::gemvx::kernel` | linear | 1 | 1.317 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 0.447 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.108 |
| `triton_red_fused_4` | other | 35 | 0.107 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 35 | 0.092 |
| `flashinfer::PersistentVariableLengthMergeStatesKernel` | other | 36 | 0.082 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 36 | 0.071 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_silu_` | activation | 36 | 0.047 |
| `triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_view_` | linear | 36 | 0.038 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 138.6 ms, of which kernels 131.0 ms (94.5%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 107.142 | 77.3% |
| attention | 13.284 | 9.6% |
| activation | 5.339 | 3.9% |
| norm | 3.836 | 2.8% |
| other | 1.337 | 1.0% |
| sampling | 0.071 | 0.1% |
| GPU idle | 7.553 | 5.5% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `cutlass::Kernel2` | linear | 108 | 89.743 |
| `cutlass::Kernel2` | linear | 36 | 15.837 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 12.670 |
| `triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_silu_` | activation | 36 | 5.339 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 35 | 2.534 |
| `internal::gemvx::kernel` | linear | 1 | 1.316 |
| `triton_red_fused_4` | other | 35 | 1.249 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 36 | 1.208 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.613 |
| `triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_view_` | linear | 36 | 0.246 |
| `triton_red_fused_fused_add_rms_norm_3` | norm | 1 | 0.071 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.051 |

## c4_in128: 8 requests, 128 in / 64 out, 4 at a time

Client: TPOT p50 11.81 ms, TTFT p50 49.8 ms, 8 completed. 59578 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 126 of 129 steps): 11.78 ms, of which kernels 11.73 ms (99.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 10.748 | 91.2% |
| attention | 0.433 | 3.7% |
| other | 0.228 | 1.9% |
| norm | 0.163 | 1.4% |
| sampling | 0.103 | 0.9% |
| activation | 0.053 | 0.5% |
| GPU idle | 0.053 | 0.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `cutlass::Kernel2` | linear | 108 | 4.723 |
| `cutlass::Kernel2` | linear | 36 | 4.668 |
| `cutlass::Kernel2` | linear | 1 | 1.320 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 0.328 |
| `triton_red_fused_4` | other | 35 | 0.109 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.105 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 35 | 0.089 |
| `flashinfer::PersistentVariableLengthMergeStatesKernel` | other | 36 | 0.070 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 36 | 0.069 |
| `triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_silu_` | activation | 36 | 0.053 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_view_` | linear | 36 | 0.038 |
