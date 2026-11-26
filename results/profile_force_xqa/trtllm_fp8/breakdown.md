# trtllm_fp8: GPU kernel time (Nsight Systems, CUDA graphs traced per node)

```
2026-11-24T22:30:11+08:00
engine=trtllm model=nvidia/Qwen3-8B-FP8 snapshot=2cebc4c89e25abc17668c81b01dceaf3d8b914d5 config=trtllm_fp8.yaml server_env=TRTLLM_FORCE_XQA=1
image=nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a
nsys=NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0 args=--session-new=bench --trace=cuda --cuda-graph-trace=node
repo_commit=9992cdb-dirty
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W
```

## c1_in128: 4 requests, 128 in / 64 out, 1 at a time

Client: TPOT p50 10.04 ms, TTFT p50 30.6 ms, 4 completed. 118956 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 252 of 255 steps): 10.06 ms, of which kernels 9.81 ms (97.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.848 | 87.9% |
| attention | 0.554 | 5.5% |
| other | 0.164 | 1.6% |
| norm | 0.132 | 1.3% |
| sampling | 0.066 | 0.7% |
| activation | 0.049 | 0.5% |
| GPU idle | 0.248 | 2.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.549 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_mha` | attention | 36 | 0.376 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.130 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.099 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.076 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.066 |
| `silu_and_mul_kernel` | activation | 36 | 0.049 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.003 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 4 requests): 28.4 ms, of which kernels 11.5 ms (40.4%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 10.467 | 36.8% |
| attention | 0.428 | 1.5% |
| other | 0.260 | 0.9% |
| norm | 0.219 | 0.8% |
| activation | 0.101 | 0.4% |
| GPU idle | 16.945 | 59.6% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize64x64x64_stage4_warps` | linear | 144 | 9.142 |
| `internal::gemvx::kernel` | linear | 1 | 1.325 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.217 |
| `fmha_v2_flash_attention_e4m3_fp32_64_32_S_qkv_128_causal_sm89_kernel_n` | attention | 36 | 0.203 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.114 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.114 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.107 |
| `silu_and_mul_kernel` | activation | 36 | 0.101 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.089 |
| `at::native::index_elementwise_kernel` | other | 4 | 0.006 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 3 | 0.004 |

## c1_in2048: 2 requests, 2048 in / 64 out, 1 at a time

Client: TPOT p50 10.88 ms, TTFT p50 117.5 ms, 2 completed. 59478 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 127 steps): 10.90 ms, of which kernels 10.65 ms (97.7%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.847 | 81.1% |
| attention | 1.388 | 12.7% |
| other | 0.165 | 1.5% |
| norm | 0.136 | 1.2% |
| sampling | 0.068 | 0.6% |
| activation | 0.050 | 0.5% |
| GPU idle | 0.251 | 2.3% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.548 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_mha` | attention | 36 | 1.210 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.134 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.099 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.076 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.068 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.003 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 111.4 ms, of which kernels 104.6 ms (93.9%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 87.971 | 79.0% |
| attention | 7.890 | 7.1% |
| activation | 5.565 | 5.0% |
| norm | 2.494 | 2.2% |
| other | 0.712 | 0.6% |
| GPU idle | 6.764 | 6.1% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x128x64_stage3_war` | linear | 144 | 86.665 |
| `fmha_v2_flash_attention_e4m3_fp32_64_32_S_qkv_128_causal_sm89_kernel_n` | attention | 36 | 6.326 |
| `silu_and_mul_kernel` | activation | 36 | 5.565 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 2.481 |
| `internal::gemvx::kernel` | linear | 1 | 1.307 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.908 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.652 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.527 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.097 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.018 |
| `at::native::elementwise_kernel` | other | 1 | 0.015 |
| `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object` | norm | 1 | 0.012 |

## c1_in3900: 2 requests, 3900 in / 64 out, 1 at a time

Client: TPOT p50 11.71 ms, TTFT p50 236.7 ms, 2 completed. 59478 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 127 steps): 11.72 ms, of which kernels 11.47 ms (97.9%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.847 | 75.5% |
| attention | 2.210 | 18.9% |
| other | 0.165 | 1.4% |
| norm | 0.136 | 1.2% |
| sampling | 0.066 | 0.6% |
| activation | 0.050 | 0.4% |
| GPU idle | 0.249 | 2.1% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.548 |
| `kernel_mha` | attention | 36 | 2.032 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.134 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.099 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.076 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.066 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.003 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 224.5 ms, of which kernels 214.5 ms (95.5%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 172.037 | 76.6% |
| attention | 23.797 | 10.6% |
| activation | 10.628 | 4.7% |
| norm | 6.598 | 2.9% |
| other | 1.412 | 0.6% |
| GPU idle | 10.070 | 4.5% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x64x64_stage4_warp` | linear | 36 | 88.184 |
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x128x64_stage3_war` | linear | 36 | 45.149 |
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize64x128x64_stage4_warp` | linear | 72 | 37.384 |
| `fmha_v2_flash_attention_e4m3_fp32_64_32_S_qkv_128_causal_sm89_kernel_n` | attention | 36 | 20.092 |
| `silu_and_mul_kernel` | activation | 36 | 10.628 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 6.576 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 2.096 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 1.605 |
| `internal::gemvx::kernel` | linear | 1 | 1.320 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 1.176 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.118 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.036 |

## c2_in128: 4 requests, 128 in / 64 out, 2 at a time

Client: TPOT p50 10.08 ms, TTFT p50 55.4 ms, 4 completed. 60640 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 10.16 ms, of which kernels 9.91 ms (97.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.921 | 87.8% |
| attention | 0.537 | 5.3% |
| norm | 0.169 | 1.7% |
| other | 0.162 | 1.6% |
| sampling | 0.069 | 0.7% |
| activation | 0.049 | 0.5% |
| GPU idle | 0.252 | 2.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.614 |
| `cutlass::Kernel2` | linear | 1 | 1.306 |
| `kernel_mha` | attention | 36 | 0.361 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.167 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.118 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.098 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.069 |
| `silu_and_mul_kernel` | activation | 36 | 0.049 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.003 |

## c4_in128: 8 requests, 128 in / 64 out, 4 at a time

Client: TPOT p50 10.38 ms, TTFT p50 75.3 ms, 8 completed. 60642 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 10.29 ms, of which kernels 10.03 ms (97.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 9.009 | 87.5% |
| attention | 0.570 | 5.5% |
| norm | 0.169 | 1.6% |
| other | 0.165 | 1.6% |
| sampling | 0.069 | 0.7% |
| activation | 0.050 | 0.5% |
| GPU idle | 0.259 | 2.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.696 |
| `cutlass::Kernel2` | linear | 1 | 1.312 |
| `kernel_mha` | attention | 36 | 0.393 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.167 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.118 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.099 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.069 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.004 |
