# vllm_fp8: GPU kernel time (Nsight Systems, CUDA graphs traced per node)

```
2026-11-24T22:08:35+08:00
engine=vllm model=nvidia/Qwen3-8B-FP8 snapshot=2cebc4c89e25abc17668c81b01dceaf3d8b914d5 extra=--kv-cache-dtype fp8
image=vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
nsys=NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0 args=--session-new=bench --trace=cuda --cuda-graph-trace=node
repo_commit=9992cdb
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W
```

## c1_in3900: 2 requests, 3900 in / 64 out, 1 at a time

Client: TPOT p50 12.02 ms, TTFT p50 289.0 ms, 2 completed. 59418 kernels recorded.

WARNING: `flashinfer::sampling::TopPSamplingFromProbKernel` ran 130 times, expected 128.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 126 of 129 steps): 12.03 ms, of which kernels 11.98 ms (99.6%).

| part | ms per step | % of step |
|---|---|---|
| linear | 10.730 | 89.2% |
| attention | 0.706 | 5.9% |
| other | 0.239 | 2.0% |
| norm | 0.168 | 1.4% |
| sampling | 0.086 | 0.7% |
| activation | 0.048 | 0.4% |
| GPU idle | 0.054 | 0.4% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `cutlass::Kernel2` | linear | 108 | 4.713 |
| `cutlass::Kernel2` | linear | 36 | 4.661 |
| `internal::gemvx::kernel` | linear | 1 | 1.317 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 0.599 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.107 |
| `triton_red_fused_4` | other | 35 | 0.107 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 35 | 0.092 |
| `flashinfer::PersistentVariableLengthMergeStatesKernel` | other | 36 | 0.085 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 36 | 0.071 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_silu_` | activation | 36 | 0.048 |
| `triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_view_` | linear | 36 | 0.038 |

## c2_in128: 4 requests, 128 in / 64 out, 2 at a time

Client: TPOT p50 11.83 ms, TTFT p50 31.9 ms, 4 completed. 59576 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 126 of 129 steps): 11.75 ms, of which kernels 11.70 ms (99.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 10.739 | 91.4% |
| attention | 0.416 | 3.5% |
| other | 0.229 | 1.9% |
| norm | 0.166 | 1.4% |
| sampling | 0.099 | 0.8% |
| activation | 0.047 | 0.4% |
| GPU idle | 0.056 | 0.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `cutlass::Kernel2` | linear | 108 | 4.726 |
| `cutlass::Kernel2` | linear | 36 | 4.658 |
| `cutlass::Kernel2` | linear | 1 | 1.316 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 0.311 |
| `triton_red_fused_4` | other | 35 | 0.110 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.106 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 35 | 0.091 |
| `flashinfer::PersistentVariableLengthMergeStatesKernel` | other | 36 | 0.071 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 36 | 0.070 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_silu_` | activation | 36 | 0.047 |
| `triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_view_` | linear | 36 | 0.039 |

## c4_in2048: 8 requests, 2048 in / 64 out, 4 at a time

Client: TPOT p50 14.41 ms, TTFT p50 406.2 ms, 8 completed. 62340 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 127 of 134 steps): 12.51 ms, of which kernels 12.46 ms (99.6%).

| part | ms per step | % of step |
|---|---|---|
| linear | 10.761 | 86.0% |
| attention | 1.149 | 9.2% |
| other | 0.230 | 1.8% |
| norm | 0.164 | 1.3% |
| sampling | 0.102 | 0.8% |
| activation | 0.053 | 0.4% |
| GPU idle | 0.056 | 0.4% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `cutlass::Kernel2` | linear | 108 | 4.730 |
| `cutlass::Kernel2` | linear | 36 | 4.672 |
| `cutlass::Kernel2` | linear | 1 | 1.320 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 1.044 |
| `triton_red_fused_4` | other | 35 | 0.110 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.104 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 35 | 0.090 |
| `flashinfer::PersistentVariableLengthMergeStatesKernel` | other | 36 | 0.072 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 36 | 0.069 |
| `triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_silu_` | activation | 36 | 0.053 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_view_` | linear | 36 | 0.038 |

## c16_in128: 32 requests, 128 in / 64 out, 16 at a time

Client: TPOT p50 12.72 ms, TTFT p50 114.1 ms, 32 completed. 60582 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 127 of 131 steps): 12.07 ms, of which kernels 12.01 ms (99.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 10.823 | 89.7% |
| attention | 0.596 | 4.9% |
| other | 0.260 | 2.2% |
| norm | 0.164 | 1.4% |
| sampling | 0.107 | 0.9% |
| activation | 0.057 | 0.5% |
| GPU idle | 0.058 | 0.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `cutlass::Kernel2` | linear | 108 | 4.736 |
| `cutlass::Kernel2` | linear | 36 | 4.706 |
| `cutlass::Kernel2` | linear | 1 | 1.340 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 0.492 |
| `triton_red_fused_4` | other | 35 | 0.111 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.102 |
| `flashinfer::PersistentVariableLengthMergeStatesKernel` | other | 36 | 0.090 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 35 | 0.089 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 36 | 0.070 |
| `triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_silu_` | activation | 36 | 0.057 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_view_` | linear | 36 | 0.040 |

## c64_in128: 128 requests, 128 in / 64 out, 64 at a time

Client: TPOT p50 16.89 ms, TTFT p50 320.8 ms, 128 completed. 63388 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 126 of 137 steps): 14.19 ms, of which kernels 14.12 ms (99.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 11.949 | 84.2% |
| attention | 1.379 | 9.7% |
| other | 0.297 | 2.1% |
| sampling | 0.199 | 1.4% |
| norm | 0.180 | 1.3% |
| activation | 0.100 | 0.7% |
| GPU idle | 0.068 | 0.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `cutlass::Kernel2` | linear | 144 | 10.459 |
| `cutlass::Kernel2` | linear | 1 | 1.446 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 1.277 |
| `triton_red_fused_4` | other | 35 | 0.115 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.102 |
| `triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_silu_` | activation | 36 | 0.100 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 35 | 0.097 |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | sampling | 1 | 0.090 |
| `flashinfer::PersistentVariableLengthMergeStatesKernel` | other | 36 | 0.085 |
| `triton_red_fused__to_copy_clamp_cutlass_scaled_mm_fused_add_rms_norm_m` | norm | 36 | 0.078 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.066 |
| `flashinfer::sampling::TopPSamplingFromProbKernel` | sampling | 1 | 0.044 |
