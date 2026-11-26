# vllm_nvfp4: GPU kernel time (Nsight Systems, CUDA graphs traced per node)

```
2026-11-24T22:18:46+08:00
engine=vllm model=nvidia/Qwen3-8B-NVFP4 snapshot=ccd10a893cbca613259517c3efe08e151ddf2b8e extra=--kv-cache-dtype fp8
image=vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
nsys=NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0 args=--session-new=bench --trace=cuda --cuda-graph-trace=node
repo_commit=9992cdb
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W
```

## c1_in3900: 2 requests, 3900 in / 64 out, 1 at a time

Client: TPOT p50 7.34 ms, TTFT p50 427.5 ms, 2 completed. 55314 kernels recorded.

WARNING: `flashinfer::sampling::TopPSamplingFromProbKernel` ran 130 times, expected 128.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 126 of 129 steps): 7.37 ms, of which kernels 7.31 ms (99.2%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.187 | 84.0% |
| attention | 0.677 | 9.2% |
| other | 0.181 | 2.5% |
| norm | 0.135 | 1.8% |
| sampling | 0.085 | 1.2% |
| activation | 0.040 | 0.5% |
| GPU idle | 0.063 | 0.8% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.192 |
| `marlin::Marlin` | linear | 72 | 1.695 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 0.568 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.109 |
| `flashinfer::PersistentVariableLengthMergeStatesKernel` | other | 36 | 0.086 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 0.069 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 0.062 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_red_fused_3` | other | 35 | 0.049 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 0.040 |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | sampling | 1 | 0.019 |

## c2_in128: 4 requests, 128 in / 64 out, 2 at a time

Client: TPOT p50 7.20 ms, TTFT p50 31.1 ms, 4 completed. 55184 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 126 of 129 steps): 7.10 ms, of which kernels 7.04 ms (99.1%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.180 | 87.0% |
| attention | 0.415 | 5.8% |
| other | 0.170 | 2.4% |
| norm | 0.135 | 1.9% |
| sampling | 0.100 | 1.4% |
| activation | 0.040 | 0.6% |
| GPU idle | 0.063 | 0.9% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.188 |
| `marlin::Marlin` | linear | 72 | 1.684 |
| `cutlass::Kernel2` | linear | 1 | 1.308 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 0.306 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.109 |
| `flashinfer::PersistentVariableLengthMergeStatesKernel` | other | 36 | 0.071 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 0.070 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 0.062 |
| `triton_red_fused_3` | other | 35 | 0.053 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 0.040 |
| `flashinfer::sampling::TopPSamplingFromProbKernel` | sampling | 1 | 0.030 |

## c4_in2048: 8 requests, 2048 in / 64 out, 4 at a time

Client: TPOT p50 10.98 ms, TTFT p50 625.6 ms, 8 completed. 57981 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 127 of 134 steps): 7.85 ms, of which kernels 7.78 ms (99.2%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.229 | 79.4% |
| attention | 1.107 | 14.1% |
| other | 0.176 | 2.2% |
| norm | 0.133 | 1.7% |
| sampling | 0.091 | 1.2% |
| activation | 0.041 | 0.5% |
| GPU idle | 0.066 | 0.8% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.214 |
| `marlin::Marlin` | linear | 72 | 1.701 |
| `cutlass::Kernel2` | linear | 1 | 1.313 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 1.000 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.107 |
| `flashinfer::PersistentVariableLengthMergeStatesKernel` | other | 36 | 0.074 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 0.068 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 0.062 |
| `triton_red_fused_3` | other | 35 | 0.055 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 0.041 |
| `flashinfer::sampling::TopPSamplingFromProbKernel` | sampling | 1 | 0.021 |

## c16_in128: 32 requests, 128 in / 64 out, 16 at a time

Client: TPOT p50 7.56 ms, TTFT p50 205.9 ms, 32 completed. 56479 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 126 of 131 steps): 7.55 ms, of which kernels 7.48 ms (99.1%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.402 | 84.8% |
| attention | 0.576 | 7.6% |
| other | 0.206 | 2.7% |
| norm | 0.134 | 1.8% |
| sampling | 0.116 | 1.5% |
| activation | 0.046 | 0.6% |
| GPU idle | 0.070 | 0.9% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.348 |
| `marlin::Marlin` | linear | 72 | 1.719 |
| `cutlass::Kernel2` | linear | 1 | 1.334 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 0.472 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.104 |
| `flashinfer::PersistentVariableLengthMergeStatesKernel` | other | 36 | 0.090 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 0.069 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 0.062 |
| `triton_red_fused_3` | other | 35 | 0.060 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 0.046 |
| `flashinfer::sampling::TopPSamplingFromProbKernel` | sampling | 1 | 0.042 |

## c64_in128: 128 requests, 128 in / 64 out, 64 at a time

Client: TPOT p50 16.11 ms, TTFT p50 515.7 ms, 128 completed. 60986 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 122 of 140 steps): 10.88 ms, of which kernels 10.80 ms (99.3%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.688 | 79.9% |
| attention | 1.421 | 13.1% |
| other | 0.262 | 2.4% |
| sampling | 0.202 | 1.9% |
| norm | 0.157 | 1.4% |
| activation | 0.072 | 0.7% |
| GPU idle | 0.071 | 0.7% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 108 | 3.884 |
| `marlin::Marlin` | linear | 36 | 3.360 |
| `cutlass::Kernel2` | linear | 1 | 1.443 |
| `flashinfer::BatchPrefillWithPagedKVCacheKernel` | attention | 36 | 1.317 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.104 |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | sampling | 1 | 0.090 |
| `flashinfer::PersistentVariableLengthMergeStatesKernel` | other | 36 | 0.087 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 0.080 |
| `triton_red_fused_3` | other | 35 | 0.079 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 0.074 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 0.072 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.064 |
