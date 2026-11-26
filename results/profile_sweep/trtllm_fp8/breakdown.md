# trtllm_fp8: GPU kernel time (Nsight Systems, CUDA graphs traced per node)

```
2026-11-24T22:14:27+08:00
engine=trtllm model=nvidia/Qwen3-8B-FP8 snapshot=2cebc4c89e25abc17668c81b01dceaf3d8b914d5 config=trtllm_fp8.yaml
image=nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a
nsys=NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0 args=--session-new=bench --trace=cuda --cuda-graph-trace=node
repo_commit=9992cdb
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W
```

## c1_in3900: 2 requests, 3900 in / 64 out, 1 at a time

Client: TPOT p50 12.12 ms, TTFT p50 232.4 ms, 2 completed. 54942 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 127 steps): 12.01 ms, of which kernels 11.78 ms (98.0%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.851 | 73.7% |
| attention | 2.508 | 20.9% |
| other | 0.165 | 1.4% |
| norm | 0.136 | 1.1% |
| sampling | 0.066 | 0.6% |
| activation | 0.050 | 0.4% |
| GPU idle | 0.235 | 2.0% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.552 |
| `tensorrt_llm::_v1::kernels::mmha::masked_multihead_attention_kernel` | attention | 36 | 2.429 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.134 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.121 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.066 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.003 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |
| `at::native::index_elementwise_kernel` | other | 1 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 225.3 ms, of which kernels 216.6 ms (96.1%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 174.117 | 77.3% |
| attention | 23.737 | 10.5% |
| activation | 10.758 | 4.8% |
| norm | 6.547 | 2.9% |
| other | 1.452 | 0.6% |
| GPU idle | 8.710 | 3.9% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x64x64_stage4_warp` | linear | 36 | 89.050 |
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x128x64_stage3_war` | linear | 36 | 45.533 |
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize64x128x64_stage4_warp` | linear | 72 | 38.210 |
| `fmha_v2_flash_attention_e4m3_fp32_64_32_S_qkv_128_causal_sm89_kernel_n` | attention | 36 | 20.080 |
| `silu_and_mul_kernel` | activation | 36 | 10.758 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 6.524 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 2.069 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 1.584 |
| `internal::gemvx::kernel` | linear | 1 | 1.325 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 1.197 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.126 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.047 |

## c2_in128: 4 requests, 128 in / 64 out, 2 at a time

Client: TPOT p50 13.50 ms, TTFT p50 58.3 ms, 4 completed. 56032 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 13.49 ms, of which kernels 13.25 ms (98.2%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.916 | 66.1% |
| attention | 3.871 | 28.7% |
| norm | 0.173 | 1.3% |
| other | 0.164 | 1.2% |
| sampling | 0.069 | 0.5% |
| activation | 0.050 | 0.4% |
| GPU idle | 0.250 | 1.8% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.609 |
| `tensorrt_llm::_v1::kernels::mmha::masked_multihead_attention_kernel` | attention | 36 | 3.793 |
| `cutlass::Kernel2` | linear | 1 | 1.306 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.171 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.119 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.069 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.003 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

## c4_in2048: 8 requests, 2048 in / 64 out, 4 at a time

Client: TPOT p50 11.19 ms, TTFT p50 448.0 ms, 8 completed. 60570 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 11.12 ms, of which kernels 10.86 ms (97.7%).

| part | ms per step | % of step |
|---|---|---|
| linear | 9.013 | 81.1% |
| attention | 1.391 | 12.5% |
| norm | 0.173 | 1.6% |
| other | 0.166 | 1.5% |
| sampling | 0.068 | 0.6% |
| activation | 0.051 | 0.5% |
| GPU idle | 0.259 | 2.3% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.698 |
| `cutlass::Kernel2` | linear | 1 | 1.315 |
| `kernel_mha` | attention | 36 | 1.215 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.171 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.119 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.099 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.068 |
| `silu_and_mul_kernel` | activation | 36 | 0.051 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.004 |

## c16_in128: 32 requests, 128 in / 64 out, 16 at a time

Client: TPOT p50 11.16 ms, TTFT p50 136.1 ms, 32 completed. 60570 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 11.04 ms, of which kernels 10.74 ms (97.3%).

| part | ms per step | % of step |
|---|---|---|
| linear | 9.517 | 86.2% |
| attention | 0.743 | 6.7% |
| norm | 0.171 | 1.6% |
| other | 0.152 | 1.4% |
| sampling | 0.099 | 0.9% |
| activation | 0.052 | 0.5% |
| GPU idle | 0.294 | 2.7% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize32x64x64_stage5_warps` | linear | 144 | 8.182 |
| `cutlass::Kernel2` | linear | 1 | 1.335 |
| `kernel_mha` | attention | 36 | 0.562 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.169 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.100 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.099 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.087 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.076 |
| `silu_and_mul_kernel` | activation | 36 | 0.052 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.009 |
| `at::native::indexSelectSmallIndex` | other | 1 | 0.007 |
| `at::native::indexSelectSmallIndex` | other | 1 | 0.006 |

## c64_in128: 128 requests, 128 in / 64 out, 64 at a time

Client: TPOT p50 13.04 ms, TTFT p50 445.8 ms, 128 completed. 63489 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 130 of 135 steps): 12.73 ms, of which kernels 12.43 ms (97.7%).

| part | ms per step | % of step |
|---|---|---|
| linear | 10.091 | 79.3% |
| attention | 1.799 | 14.1% |
| norm | 0.184 | 1.4% |
| other | 0.177 | 1.4% |
| sampling | 0.102 | 0.8% |
| activation | 0.079 | 0.6% |
| GPU idle | 0.295 | 2.3% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize64x64x64_stage4_warps` | linear | 108 | 7.864 |
| `kernel_mha` | attention | 36 | 1.596 |
| `cutlass::Kernel2` | linear | 1 | 1.437 |
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize32x64x64_stage5_warps` | linear | 36 | 0.788 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.182 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.107 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.102 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.094 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.088 |
| `silu_and_mul_kernel` | activation | 36 | 0.079 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.030 |
| `at::native::vectorized_elementwise_kernel` | other | 1 | 0.010 |
