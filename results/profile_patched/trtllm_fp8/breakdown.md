# trtllm_fp8: GPU kernel time (Nsight Systems, CUDA graphs traced per node)

```
2026-11-25T13:55:25+08:00
engine=trtllm model=nvidia/Qwen3-8B-FP8 snapshot=2cebc4c89e25abc17668c81b01dceaf3d8b914d5 config=trtllm_fp8.yaml server_env=
server_docker_args=-v ~/Documents/TensorRT-LLM-build/engine_rc2/cpp-build/tensorrt_llm/libtensorrt_llm.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libtensorrt_llm.so:ro -v ~/Documents/TensorRT-LLM-build/engine_rc2/cpp-build/tensorrt_llm/thop/libth_common.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libth_common.so:ro -v ~/Documents/TensorRT-LLM-build/engine_rc2/cpp-build/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_0.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libdecoder_attention_0.so:ro -v ~/Documents/TensorRT-LLM-build/engine_rc2/cpp-build/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_1.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libdecoder_attention_1.so:ro -v ~/Documents/TensorRT-LLM-build/engine_rc2/cpp-build/tensorrt_llm/nanobind/bindings.cpython-312-x86_64-linux-gnu.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/bindings.cpython-312-x86_64-linux-gnu.so:ro
image=nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a
nsys=NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0 args=--session-new=bench --trace=cuda --cuda-graph-trace=node
repo_commit=04b06dc-dirty
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W
```

## c1_in128: 4 requests, 128 in / 64 out, 1 at a time

Client: TPOT p50 10.23 ms, TTFT p50 31.9 ms, 4 completed. 118956 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 252 of 255 steps): 10.24 ms, of which kernels 9.98 ms (97.4%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.895 | 86.9% |
| attention | 0.663 | 6.5% |
| other | 0.164 | 1.6% |
| norm | 0.136 | 1.3% |
| sampling | 0.066 | 0.6% |
| activation | 0.050 | 0.5% |
| GPU idle | 0.265 | 2.6% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.596 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_mha` | attention | 36 | 0.488 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.134 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.098 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.066 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.003 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 4 requests): 29.6 ms, of which kernels 11.5 ms (38.8%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 10.471 | 35.4% |
| attention | 0.430 | 1.5% |
| other | 0.259 | 0.9% |
| norm | 0.221 | 0.7% |
| activation | 0.102 | 0.3% |
| GPU idle | 18.133 | 61.2% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize64x64x64_stage4_warps` | linear | 144 | 9.148 |
| `internal::gemvx::kernel` | linear | 1 | 1.324 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.218 |
| `fmha_v2_flash_attention_e4m3_fp32_64_32_S_qkv_128_causal_sm89_kernel_n` | attention | 36 | 0.203 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.115 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.114 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.108 |
| `silu_and_mul_kernel` | activation | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.088 |
| `at::native::index_elementwise_kernel` | other | 4 | 0.006 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 3 | 0.004 |

## c1_in2048: 2 requests, 2048 in / 64 out, 1 at a time

Client: TPOT p50 10.31 ms, TTFT p50 115.7 ms, 2 completed. 59478 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 127 steps): 10.32 ms, of which kernels 10.07 ms (97.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.897 | 86.2% |
| attention | 0.751 | 7.3% |
| other | 0.166 | 1.6% |
| norm | 0.136 | 1.3% |
| sampling | 0.067 | 0.6% |
| activation | 0.050 | 0.5% |
| GPU idle | 0.255 | 2.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.598 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_mha` | attention | 36 | 0.574 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.134 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.122 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.098 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.003 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 110.9 ms, of which kernels 104.8 ms (94.5%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 88.096 | 79.4% |
| attention | 7.956 | 7.2% |
| activation | 5.565 | 5.0% |
| norm | 2.495 | 2.2% |
| other | 0.714 | 0.6% |
| GPU idle | 6.112 | 5.5% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x128x64_stage3_war` | linear | 144 | 86.790 |
| `fmha_v2_flash_attention_e4m3_fp32_64_32_S_qkv_128_causal_sm89_kernel_n` | attention | 36 | 6.365 |
| `silu_and_mul_kernel` | activation | 36 | 5.565 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 2.483 |
| `internal::gemvx::kernel` | linear | 1 | 1.306 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.912 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.676 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.529 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.097 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.018 |
| `at::native::elementwise_kernel` | other | 1 | 0.015 |
| `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object` | norm | 1 | 0.012 |

## c1_in3900: 2 requests, 3900 in / 64 out, 1 at a time

Client: TPOT p50 10.60 ms, TTFT p50 232.2 ms, 2 completed. 59478 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 127 steps): 10.51 ms, of which kernels 10.25 ms (97.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.905 | 84.7% |
| attention | 0.928 | 8.8% |
| other | 0.166 | 1.6% |
| norm | 0.136 | 1.3% |
| sampling | 0.066 | 0.6% |
| activation | 0.050 | 0.5% |
| GPU idle | 0.262 | 2.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.605 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_mha` | attention | 36 | 0.750 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.134 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.122 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.098 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.066 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 225.7 ms, of which kernels 217.3 ms (96.3%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 174.080 | 77.1% |
| attention | 23.848 | 10.6% |
| activation | 10.594 | 4.7% |
| norm | 7.313 | 3.2% |
| other | 1.485 | 0.7% |
| GPU idle | 8.376 | 3.7% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x64x64_stage4_warp` | linear | 36 | 89.261 |
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x128x64_stage3_war` | linear | 36 | 45.449 |
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize64x128x64_stage4_warp` | linear | 72 | 38.050 |
| `fmha_v2_flash_attention_e4m3_fp32_64_32_S_qkv_128_causal_sm89_kernel_n` | attention | 36 | 20.180 |
| `silu_and_mul_kernel` | activation | 36 | 10.594 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 7.291 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 2.092 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 1.572 |
| `internal::gemvx::kernel` | linear | 1 | 1.320 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 1.240 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.118 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.045 |

## c2_in128: 4 requests, 128 in / 64 out, 2 at a time

Client: TPOT p50 10.40 ms, TTFT p50 58.0 ms, 4 completed. 60568 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 10.35 ms, of which kernels 10.09 ms (97.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.970 | 86.6% |
| attention | 0.670 | 6.5% |
| norm | 0.172 | 1.7% |
| other | 0.163 | 1.6% |
| sampling | 0.067 | 0.6% |
| activation | 0.050 | 0.5% |
| GPU idle | 0.260 | 2.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.663 |
| `cutlass::Kernel2` | linear | 1 | 1.306 |
| `kernel_mha` | attention | 36 | 0.494 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.170 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.118 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.098 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.073 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

## c2_in2048: 4 requests, 2048 in / 64 out, 2 at a time

Client: TPOT p50 11.63 ms, TTFT p50 171.8 ms, 4 completed. 60640 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 10.70 ms, of which kernels 10.44 ms (97.6%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.978 | 83.9% |
| attention | 1.004 | 9.4% |
| norm | 0.172 | 1.6% |
| other | 0.163 | 1.5% |
| sampling | 0.070 | 0.7% |
| activation | 0.050 | 0.5% |
| GPU idle | 0.259 | 2.4% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.671 |
| `cutlass::Kernel2` | linear | 1 | 1.307 |
| `kernel_mha` | attention | 36 | 0.828 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.170 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.119 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.099 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.070 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

## c4_in128: 8 requests, 128 in / 64 out, 4 at a time

Client: TPOT p50 10.60 ms, TTFT p50 80.8 ms, 8 completed. 60570 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 10.54 ms, of which kernels 10.29 ms (97.6%).

| part | ms per step | % of step |
|---|---|---|
| linear | 9.059 | 85.9% |
| attention | 0.772 | 7.3% |
| norm | 0.172 | 1.6% |
| other | 0.165 | 1.6% |
| sampling | 0.069 | 0.7% |
| activation | 0.051 | 0.5% |
| GPU idle | 0.257 | 2.4% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.746 |
| `cutlass::Kernel2` | linear | 1 | 1.313 |
| `kernel_mha` | attention | 36 | 0.597 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.170 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.118 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.099 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.073 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.069 |
| `silu_and_mul_kernel` | activation | 36 | 0.051 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.004 |

## c4_in2048: 8 requests, 2048 in / 64 out, 4 at a time

Client: TPOT p50 11.18 ms, TTFT p50 447.5 ms, 8 completed. 60642 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 11.11 ms, of which kernels 10.86 ms (97.7%).

| part | ms per step | % of step |
|---|---|---|
| linear | 9.077 | 81.7% |
| attention | 1.322 | 11.9% |
| norm | 0.172 | 1.5% |
| other | 0.166 | 1.5% |
| sampling | 0.068 | 0.6% |
| activation | 0.051 | 0.5% |
| GPU idle | 0.255 | 2.3% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.764 |
| `cutlass::Kernel2` | linear | 1 | 1.313 |
| `kernel_mha` | attention | 36 | 1.145 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.170 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.119 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.099 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.068 |
| `silu_and_mul_kernel` | activation | 36 | 0.051 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.004 |
