# trtllm_fp8: GPU kernel time (Nsight Systems, CUDA graphs traced per node)

```
2026-11-26T03:49:22+08:00
engine=trtllm model=nvidia/Qwen3-8B-FP8 snapshot=2cebc4c89e25abc17668c81b01dceaf3d8b914d5 config=trtllm_fp8.yaml server_env=
server_docker_args=-v /var/tmp/trtllm-rc4/build/cpp-build/tensorrt_llm/libtensorrt_llm.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libtensorrt_llm.so:ro -v /var/tmp/trtllm-rc4/build/cpp-build/tensorrt_llm/thop/libth_common.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libth_common.so:ro -v /var/tmp/trtllm-rc4/build/cpp-build/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_0.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libdecoder_attention_0.so:ro -v /var/tmp/trtllm-rc4/build/cpp-build/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_1.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libdecoder_attention_1.so:ro -v /var/tmp/trtllm-rc4/build/cpp-build/tensorrt_llm/nanobind/bindings.cpython-312-x86_64-linux-gnu.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/bindings.cpython-312-x86_64-linux-gnu.so:ro
image=nvcr.io/nvidia/tensorrt-llm/release@sha256:813f751bc2c7ed867a53915dedde1c153ca94758e931d989e9c25835f9ba1814
nsys=NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0 args=--session-new=bench --trace=cuda --cuda-graph-trace=node
repo_commit=bb4f8dd-dirty
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W
```

## c1_in128: 4 requests, 128 in / 64 out, 1 at a time

Client: TPOT p50 10.28 ms, TTFT p50 68.1 ms, 4 completed. 118960 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 252 of 255 steps): 10.21 ms, of which kernels 9.96 ms (97.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.886 | 87.0% |
| attention | 0.649 | 6.4% |
| other | 0.165 | 1.6% |
| norm | 0.135 | 1.3% |
| sampling | 0.067 | 0.7% |
| activation | 0.050 | 0.5% |
| GPU idle | 0.254 | 2.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.587 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_mha` | attention | 36 | 0.477 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.134 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.097 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |
| `at::native::index_elementwise_kernel` | other | 1 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 4 requests): 49.1 ms, of which kernels 11.4 ms (23.3%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 10.422 | 21.2% |
| attention | 0.436 | 0.9% |
| other | 0.261 | 0.5% |
| norm | 0.219 | 0.4% |
| activation | 0.103 | 0.2% |
| GPU idle | 37.664 | 76.7% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize64x64x64_stage4_warps` | linear | 144 | 9.104 |
| `internal::gemvx::kernel` | linear | 1 | 1.319 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.216 |
| `fmha_v2_flash_attention_e4m3_fp32_64_32_S_qkv_128_causal_sm89_kernel_n` | attention | 36 | 0.203 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.122 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.115 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.108 |
| `silu_and_mul_kernel` | activation | 36 | 0.103 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.089 |
| `at::native::index_elementwise_kernel` | other | 4 | 0.006 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 3 | 0.004 |

## c1_in2048: 2 requests, 2048 in / 64 out, 1 at a time

Client: TPOT p50 10.37 ms, TTFT p50 125.7 ms, 2 completed. 59480 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 127 steps): 10.32 ms, of which kernels 10.06 ms (97.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.894 | 86.2% |
| attention | 0.739 | 7.2% |
| other | 0.166 | 1.6% |
| norm | 0.135 | 1.3% |
| sampling | 0.067 | 0.7% |
| activation | 0.050 | 0.5% |
| GPU idle | 0.260 | 2.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.595 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_mha` | attention | 36 | 0.566 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.134 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.098 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |
| `at::native::index_elementwise_kernel` | other | 1 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 115.6 ms, of which kernels 106.5 ms (92.1%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 89.150 | 77.1% |
| attention | 8.038 | 7.0% |
| activation | 5.912 | 5.1% |
| norm | 2.607 | 2.3% |
| quantize | 0.617 | 0.5% |
| other | 0.191 | 0.2% |
| GPU idle | 9.079 | 7.9% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x128x64_stage3_war` | linear | 144 | 87.844 |
| `fmha_v2_flash_attention_e4m3_fp32_64_32_S_qkv_128_causal_sm89_kernel_n` | attention | 36 | 6.411 |
| `silu_and_mul_kernel` | activation | 36 | 5.912 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 2.594 |
| `internal::gemvx::kernel` | linear | 1 | 1.306 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.914 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.709 |
| `_fp8_static_quant_kernel` | quantize | 72 | 0.617 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.099 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.021 |
| `at::native::elementwise_kernel` | other | 1 | 0.015 |
| `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object` | norm | 1 | 0.013 |

## c1_in3900: 2 requests, 3900 in / 64 out, 1 at a time

Client: TPOT p50 10.53 ms, TTFT p50 243.9 ms, 2 completed. 59482 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 127 steps): 10.47 ms, of which kernels 10.21 ms (97.6%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.895 | 85.0% |
| attention | 0.897 | 8.6% |
| other | 0.166 | 1.6% |
| norm | 0.135 | 1.3% |
| sampling | 0.066 | 0.6% |
| activation | 0.049 | 0.5% |
| GPU idle | 0.254 | 2.4% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.596 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_mha` | attention | 36 | 0.723 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.134 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.098 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.066 |
| `silu_and_mul_kernel` | activation | 36 | 0.049 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |
| `at::native::index_elementwise_kernel` | other | 1 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 227.0 ms, of which kernels 216.9 ms (95.5%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 174.490 | 76.9% |
| attention | 23.860 | 10.5% |
| activation | 10.600 | 4.7% |
| norm | 6.514 | 2.9% |
| quantize | 1.204 | 0.5% |
| other | 0.253 | 0.1% |
| GPU idle | 10.114 | 4.5% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x64x64_stage4_warp` | linear | 36 | 89.570 |
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x128x64_stage3_war` | linear | 36 | 45.665 |
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize64x128x64_stage4_warp` | linear | 72 | 37.935 |
| `fmha_v2_flash_attention_e4m3_fp32_64_32_S_qkv_128_causal_sm89_kernel_n` | attention | 36 | 20.163 |
| `silu_and_mul_kernel` | activation | 36 | 10.600 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 6.492 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 2.100 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 1.594 |
| `internal::gemvx::kernel` | linear | 1 | 1.320 |
| `_fp8_static_quant_kernel` | quantize | 72 | 1.204 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.127 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.044 |

## c2_in128: 4 requests, 128 in / 64 out, 2 at a time

Client: TPOT p50 10.32 ms, TTFT p50 62.4 ms, 4 completed. 60572 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 125 of 129 steps): 10.32 ms, of which kernels 10.07 ms (97.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.956 | 86.7% |
| attention | 0.659 | 6.4% |
| norm | 0.173 | 1.7% |
| other | 0.165 | 1.6% |
| sampling | 0.068 | 0.7% |
| activation | 0.050 | 0.5% |
| GPU idle | 0.256 | 2.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.650 |
| `cutlass::Kernel2` | linear | 1 | 1.306 |
| `kernel_mha` | attention | 36 | 0.489 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.171 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.119 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.098 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.073 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.068 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.003 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

## c2_in2048: 4 requests, 2048 in / 64 out, 2 at a time

Client: TPOT p50 11.56 ms, TTFT p50 173.9 ms, 4 completed. 60644 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 10.64 ms, of which kernels 10.39 ms (97.7%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.968 | 84.3% |
| attention | 0.968 | 9.1% |
| norm | 0.173 | 1.6% |
| other | 0.165 | 1.5% |
| sampling | 0.067 | 0.6% |
| activation | 0.050 | 0.5% |
| GPU idle | 0.245 | 2.3% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.662 |
| `cutlass::Kernel2` | linear | 1 | 1.306 |
| `kernel_mha` | attention | 36 | 0.796 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.171 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.119 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.098 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.073 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.003 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

## c4_in128: 8 requests, 128 in / 64 out, 4 at a time

Client: TPOT p50 10.58 ms, TTFT p50 65.1 ms, 8 completed. 60574 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 10.53 ms, of which kernels 10.27 ms (97.6%).

| part | ms per step | % of step |
|---|---|---|
| linear | 9.045 | 85.9% |
| attention | 0.772 | 7.3% |
| norm | 0.170 | 1.6% |
| other | 0.166 | 1.6% |
| sampling | 0.068 | 0.6% |
| activation | 0.051 | 0.5% |
| GPU idle | 0.258 | 2.4% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.732 |
| `cutlass::Kernel2` | linear | 1 | 1.313 |
| `kernel_mha` | attention | 36 | 0.600 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.168 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.118 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.098 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.073 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.068 |
| `silu_and_mul_kernel` | activation | 36 | 0.051 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

## c4_in2048: 8 requests, 2048 in / 64 out, 4 at a time

Client: TPOT p50 11.21 ms, TTFT p50 446.1 ms, 8 completed. 60646 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 11.11 ms, of which kernels 10.86 ms (97.8%).

| part | ms per step | % of step |
|---|---|---|
| linear | 9.065 | 81.6% |
| attention | 1.343 | 12.1% |
| norm | 0.172 | 1.5% |
| other | 0.167 | 1.5% |
| sampling | 0.067 | 0.6% |
| activation | 0.051 | 0.5% |
| GPU idle | 0.246 | 2.2% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.752 |
| `cutlass::Kernel2` | linear | 1 | 1.313 |
| `kernel_mha` | attention | 36 | 1.170 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.170 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.119 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.099 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.051 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |
