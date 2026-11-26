# trtllm_fp8: GPU kernel time (Nsight Systems, CUDA graphs traced per node)

```
2026-11-26T01:07:41+08:00
engine=trtllm model=nvidia/Qwen3-8B-FP8 snapshot=2cebc4c89e25abc17668c81b01dceaf3d8b914d5 config=trtllm_fp8.yaml server_env=
image=nvcr.io/nvidia/tensorrt-llm/release@sha256:813f751bc2c7ed867a53915dedde1c153ca94758e931d989e9c25835f9ba1814
nsys=NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0 args=--session-new=bench --trace=cuda --cuda-graph-trace=node
repo_commit=bb4f8dd-dirty
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W
```

## c1_in128: 4 requests, 128 in / 64 out, 1 at a time

Client: TPOT p50 12.07 ms, TTFT p50 40.3 ms, 4 completed. 109888 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 245 of 255 steps): 11.76 ms, of which kernels 11.52 ms (97.9%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.847 | 75.2% |
| attention | 2.254 | 19.2% |
| other | 0.167 | 1.4% |
| norm | 0.136 | 1.2% |
| sampling | 0.068 | 0.6% |
| activation | 0.050 | 0.4% |
| GPU idle | 0.241 | 2.1% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.548 |
| `tensorrt_llm::_v1::kernels::mmha::masked_multihead_attention_kernel` | attention | 36 | 2.177 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.134 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.122 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.076 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.068 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |
| `at::native::index_elementwise_kernel` | other | 1 | 0.003 |
| `at::native::vectorized_elementwise_kernel` | other | 2 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 4 requests): 32.4 ms, of which kernels 11.4 ms (35.2%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 10.391 | 32.1% |
| attention | 0.432 | 1.3% |
| other | 0.260 | 0.8% |
| norm | 0.219 | 0.7% |
| activation | 0.104 | 0.3% |
| GPU idle | 20.987 | 64.8% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize64x64x64_stage4_warps` | linear | 144 | 9.073 |
| `internal::gemvx::kernel` | linear | 1 | 1.318 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.216 |
| `fmha_v2_flash_attention_e4m3_fp32_64_32_S_qkv_128_causal_sm89_kernel_n` | attention | 36 | 0.202 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.119 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.114 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.107 |
| `silu_and_mul_kernel` | activation | 36 | 0.104 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.087 |
| `at::native::index_elementwise_kernel` | other | 4 | 0.006 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 3 | 0.004 |

## c1_in2048: 2 requests, 2048 in / 64 out, 1 at a time

Client: TPOT p50 11.78 ms, TTFT p50 120.0 ms, 2 completed. 54944 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 127 steps): 11.78 ms, of which kernels 11.55 ms (98.0%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.846 | 75.1% |
| attention | 2.289 | 19.4% |
| other | 0.167 | 1.4% |
| norm | 0.136 | 1.2% |
| sampling | 0.067 | 0.6% |
| activation | 0.050 | 0.4% |
| GPU idle | 0.231 | 2.0% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.547 |
| `tensorrt_llm::_v1::kernels::mmha::masked_multihead_attention_kernel` | attention | 36 | 2.213 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.134 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.122 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.076 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |
| `at::native::index_elementwise_kernel` | other | 1 | 0.003 |
| `at::native::vectorized_elementwise_kernel` | other | 2 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 111.0 ms, of which kernels 104.7 ms (94.4%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 87.860 | 79.2% |
| attention | 7.962 | 7.2% |
| activation | 5.590 | 5.0% |
| norm | 2.503 | 2.3% |
| quantize | 0.624 | 0.6% |
| other | 0.186 | 0.2% |
| GPU idle | 6.227 | 5.6% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x128x64_stage3_war` | linear | 144 | 86.553 |
| `fmha_v2_flash_attention_e4m3_fp32_64_32_S_qkv_128_causal_sm89_kernel_n` | attention | 36 | 6.342 |
| `silu_and_mul_kernel` | activation | 36 | 5.590 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 2.491 |
| `internal::gemvx::kernel` | linear | 1 | 1.307 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.911 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.704 |
| `_fp8_static_quant_kernel` | quantize | 72 | 0.624 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.098 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.018 |
| `at::native::elementwise_kernel` | other | 1 | 0.015 |
| `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object` | norm | 1 | 0.012 |

## c1_in3900: 2 requests, 3900 in / 64 out, 1 at a time

Client: TPOT p50 11.96 ms, TTFT p50 236.5 ms, 2 completed. 54946 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 127 steps): 11.98 ms, of which kernels 11.75 ms (98.1%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.847 | 73.9% |
| attention | 2.481 | 20.7% |
| other | 0.167 | 1.4% |
| norm | 0.136 | 1.1% |
| sampling | 0.067 | 0.6% |
| activation | 0.050 | 0.4% |
| GPU idle | 0.232 | 1.9% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.548 |
| `tensorrt_llm::_v1::kernels::mmha::masked_multihead_attention_kernel` | attention | 36 | 2.405 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.134 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.122 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.076 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |
| `at::native::index_elementwise_kernel` | other | 1 | 0.003 |
| `at::native::vectorized_elementwise_kernel` | other | 2 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 223.2 ms, of which kernels 214.8 ms (96.2%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 172.452 | 77.3% |
| attention | 23.832 | 10.7% |
| activation | 10.600 | 4.7% |
| norm | 6.479 | 2.9% |
| quantize | 1.211 | 0.5% |
| other | 0.239 | 0.1% |
| GPU idle | 8.374 | 3.8% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x64x64_stage4_warp` | linear | 36 | 88.382 |
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x128x64_stage3_war` | linear | 36 | 45.273 |
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize64x128x64_stage4_warp` | linear | 72 | 37.477 |
| `fmha_v2_flash_attention_e4m3_fp32_64_32_S_qkv_128_causal_sm89_kernel_n` | attention | 36 | 20.139 |
| `silu_and_mul_kernel` | activation | 36 | 10.600 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 6.458 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 2.094 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 1.595 |
| `internal::gemvx::kernel` | linear | 1 | 1.320 |
| `_fp8_static_quant_kernel` | quantize | 72 | 1.211 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.120 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.036 |

## c2_in128: 4 requests, 128 in / 64 out, 2 at a time

Client: TPOT p50 13.34 ms, TTFT p50 62.8 ms, 4 completed. 56036 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 13.50 ms, of which kernels 13.26 ms (98.3%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.915 | 66.1% |
| attention | 3.888 | 28.8% |
| norm | 0.173 | 1.3% |
| other | 0.165 | 1.2% |
| sampling | 0.068 | 0.5% |
| activation | 0.050 | 0.4% |
| GPU idle | 0.236 | 1.7% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.609 |
| `tensorrt_llm::_v1::kernels::mmha::masked_multihead_attention_kernel` | attention | 36 | 3.813 |
| `cutlass::Kernel2` | linear | 1 | 1.306 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.171 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.119 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.068 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.003 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |
| `at::native::vectorized_elementwise_kernel` | other | 2 | 0.003 |

## c2_in2048: 4 requests, 2048 in / 64 out, 2 at a time

Client: TPOT p50 14.82 ms, TTFT p50 181.4 ms, 4 completed. 68486 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 155 of 158 steps): 13.60 ms, of which kernels 13.37 ms (98.3%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.914 | 65.6% |
| attention | 3.995 | 29.4% |
| norm | 0.172 | 1.3% |
| other | 0.165 | 1.2% |
| sampling | 0.067 | 0.5% |
| activation | 0.050 | 0.4% |
| GPU idle | 0.231 | 1.7% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.608 |
| `tensorrt_llm::_v1::kernels::mmha::masked_multihead_attention_kernel` | attention | 36 | 3.920 |
| `cutlass::Kernel2` | linear | 1 | 1.306 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.170 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.119 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.003 |
| `at::native::vectorized_elementwise_kernel` | other | 2 | 0.003 |

## c4_in128: 8 requests, 128 in / 64 out, 4 at a time

Client: TPOT p50 10.21 ms, TTFT p50 65.9 ms, 8 completed. 60574 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 10.28 ms, of which kernels 10.02 ms (97.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.997 | 87.5% |
| attention | 0.570 | 5.5% |
| norm | 0.171 | 1.7% |
| other | 0.167 | 1.6% |
| sampling | 0.068 | 0.7% |
| activation | 0.051 | 0.5% |
| GPU idle | 0.254 | 2.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.686 |
| `cutlass::Kernel2` | linear | 1 | 1.311 |
| `kernel_mha` | attention | 36 | 0.395 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.169 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.119 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.100 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.068 |
| `silu_and_mul_kernel` | activation | 36 | 0.051 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

## c4_in2048: 8 requests, 2048 in / 64 out, 4 at a time

Client: TPOT p50 11.24 ms, TTFT p50 450.8 ms, 8 completed. 60574 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 11.13 ms, of which kernels 10.87 ms (97.7%).

| part | ms per step | % of step |
|---|---|---|
| linear | 9.005 | 80.9% |
| attention | 1.411 | 12.7% |
| norm | 0.172 | 1.5% |
| other | 0.167 | 1.5% |
| sampling | 0.068 | 0.6% |
| activation | 0.051 | 0.5% |
| GPU idle | 0.252 | 2.3% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.694 |
| `cutlass::Kernel2` | linear | 1 | 1.311 |
| `kernel_mha` | attention | 36 | 1.236 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.170 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.119 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.099 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.068 |
| `silu_and_mul_kernel` | activation | 36 | 0.051 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |
