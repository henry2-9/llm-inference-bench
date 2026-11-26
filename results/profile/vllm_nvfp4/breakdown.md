# vllm_nvfp4: GPU kernel time (Nsight Systems, CUDA graphs traced per node)

```
2026-11-24T20:44:27+08:00
engine=vllm model=nvidia/Qwen3-8B-NVFP4 snapshot=ccd10a893cbca613259517c3efe08e151ddf2b8e extra=--kv-cache-dtype fp8
image=vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
nsys=NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0 args=--session-new=bench --trace=cuda --cuda-graph-trace=node
repo_commit=404f449
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W
```

## c1_in128: 4 requests, 128 in / 64 out, 1 at a time

Client: TPOT p50 7.10 ms, TTFT p50 22.2 ms, 4 completed. 107924 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 252 of 255 steps): 7.10 ms, of which kernels 7.04 ms (99.2%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.140 | 86.5% |
| attention | 0.413 | 5.8% |
| other | 0.224 | 3.2% |
| norm | 0.134 | 1.9% |
| sampling | 0.084 | 1.2% |
| activation | 0.040 | 0.6% |
| GPU idle | 0.060 | 0.8% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.162 |
| `marlin::Marlin` | linear | 72 | 1.679 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 0.306 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.108 |
| `triton_red_fused_3` | other | 35 | 0.106 |
| `flashinfer::PersistentVariableLengthMergeStatesKernel` | other | 36 | 0.071 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 0.069 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 0.062 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 0.040 |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | sampling | 1 | 0.020 |

**Prefill** (first kernel of a request to its first sampled token; median of 4 requests): 21.6 ms, of which kernels 15.3 ms (71.0%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 14.169 | 65.6% |
| attention | 0.600 | 2.8% |
| norm | 0.194 | 0.9% |
| other | 0.178 | 0.8% |
| activation | 0.117 | 0.5% |
| sampling | 0.070 | 0.3% |
| GPU idle | 6.270 | 29.0% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 8.060 |
| `marlin::Marlin` | linear | 72 | 4.809 |
| `internal::gemvx::kernel` | linear | 1 | 1.300 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 0.473 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.127 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 0.117 |
| `triton_red_fused_3` | other | 35 | 0.116 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 0.099 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 0.091 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | sampling | 1 | 0.020 |
| `_apply_write_kernel` | other | 4 | 0.012 |

## c1_in2048: 2 requests, 2048 in / 64 out, 1 at a time

Client: TPOT p50 7.26 ms, TTFT p50 219.0 ms, 2 completed. 54106 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 126 of 127 steps): 7.28 ms, of which kernels 7.22 ms (99.2%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.180 | 84.9% |
| attention | 0.539 | 7.4% |
| other | 0.235 | 3.2% |
| norm | 0.134 | 1.8% |
| sampling | 0.088 | 1.2% |
| activation | 0.040 | 0.5% |
| GPU idle | 0.058 | 0.8% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.189 |
| `marlin::Marlin` | linear | 72 | 1.692 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 0.431 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.108 |
| `triton_red_fused_3` | other | 35 | 0.106 |
| `flashinfer::PersistentVariableLengthMergeStatesKernel` | other | 36 | 0.082 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 0.069 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 0.062 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 0.040 |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | sampling | 1 | 0.019 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 213.3 ms, of which kernels 206.2 ms (96.7%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 183.340 | 86.0% |
| attention | 13.251 | 6.2% |
| activation | 4.982 | 2.3% |
| norm | 3.699 | 1.7% |
| other | 0.883 | 0.4% |
| sampling | 0.071 | 0.0% |
| GPU idle | 7.024 | 3.3% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `marlin::Marlin` | linear | 216 | 182.042 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 12.671 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 4.982 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 2.377 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 1.300 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `triton_red_fused_3` | other | 35 | 0.801 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.581 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.051 |
| `triton_red_fused__to_copy_embedding_rms_norm_0` | norm | 1 | 0.023 |
| `triton_red_fused_1` | other | 1 | 0.022 |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | sampling | 1 | 0.020 |

## c4_in128: 8 requests, 128 in / 64 out, 4 at a time

Client: TPOT p50 7.18 ms, TTFT p50 60.7 ms, 8 completed. 55186 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 126 of 129 steps): 7.18 ms, of which kernels 7.12 ms (99.2%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.185 | 86.1% |
| attention | 0.432 | 6.0% |
| other | 0.230 | 3.2% |
| norm | 0.133 | 1.9% |
| sampling | 0.102 | 1.4% |
| activation | 0.041 | 0.6% |
| GPU idle | 0.058 | 0.8% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.194 |
| `marlin::Marlin` | linear | 72 | 1.679 |
| `cutlass::Kernel2` | linear | 1 | 1.311 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 0.327 |
| `triton_red_fused_3` | other | 35 | 0.111 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.105 |
| `flashinfer::PersistentVariableLengthMergeStatesKernel` | other | 36 | 0.071 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 0.069 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 0.062 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 0.041 |
| `flashinfer::sampling::TopPSamplingFromProbKernel` | sampling | 1 | 0.032 |
