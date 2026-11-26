# trtllm_nvfp4: GPU kernel time (Nsight Systems, CUDA graphs traced per node)

```
2026-11-26T03:56:42+08:00
engine=trtllm model=nvidia/Qwen3-8B-NVFP4 snapshot=ccd10a893cbca613259517c3efe08e151ddf2b8e config=trtllm_nvfp4.yaml server_env=
server_docker_args=-v /var/tmp/trtllm-rc4/build/cpp-build/tensorrt_llm/libtensorrt_llm.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libtensorrt_llm.so:ro -v /var/tmp/trtllm-rc4/build/cpp-build/tensorrt_llm/thop/libth_common.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libth_common.so:ro -v /var/tmp/trtllm-rc4/build/cpp-build/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_0.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libdecoder_attention_0.so:ro -v /var/tmp/trtllm-rc4/build/cpp-build/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_1.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libdecoder_attention_1.so:ro -v /var/tmp/trtllm-rc4/build/cpp-build/tensorrt_llm/nanobind/bindings.cpython-312-x86_64-linux-gnu.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/bindings.cpython-312-x86_64-linux-gnu.so:ro
image=nvcr.io/nvidia/tensorrt-llm/release@sha256:813f751bc2c7ed867a53915dedde1c153ca94758e931d989e9c25835f9ba1814
nsys=NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0 args=--session-new=bench --trace=cuda --cuda-graph-trace=node
repo_commit=bb4f8dd-dirty
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W
```

## c1_in128: 4 requests, 128 in / 64 out, 1 at a time

Client: TPOT p50 7.47 ms, TTFT p50 27.0 ms, 4 completed. 137392 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 252 of 255 steps): 7.43 ms, of which kernels 7.29 ms (98.2%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.199 | 83.4% |
| attention | 0.680 | 9.2% |
| other | 0.166 | 2.2% |
| norm | 0.145 | 2.0% |
| sampling | 0.066 | 0.9% |
| activation | 0.038 | 0.5% |
| GPU idle | 0.136 | 1.8% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.173 |
| `marlin::Marlin` | linear | 72 | 1.728 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_mha` | attention | 36 | 0.503 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.143 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.101 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.066 |
| `silu_and_mul_kernel` | activation | 36 | 0.038 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 4 requests): 24.8 ms, of which kernels 15.2 ms (61.5%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 14.039 | 56.6% |
| attention | 0.576 | 2.3% |
| other | 0.283 | 1.1% |
| norm | 0.231 | 0.9% |
| activation | 0.110 | 0.4% |
| GPU idle | 9.546 | 38.5% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 7.993 |
| `marlin::Marlin` | linear | 72 | 4.743 |
| `internal::gemvx::kernel` | linear | 1 | 1.302 |
| `fmha_v2_flash_attention_bf16_64_32_S_qkv_128_causal_sm89_kernel_nl` | attention | 36 | 0.344 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.229 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.134 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.119 |
| `silu_and_mul_kernel` | activation | 36 | 0.110 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.109 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.091 |
| `at::native::index_elementwise_kernel` | other | 4 | 0.006 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |

## c1_in2048: 2 requests, 2048 in / 64 out, 1 at a time

Client: TPOT p50 7.65 ms, TTFT p50 220.6 ms, 2 completed. 68840 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 127 steps): 7.59 ms, of which kernels 7.45 ms (98.2%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.207 | 81.8% |
| attention | 0.828 | 10.9% |
| other | 0.166 | 2.2% |
| norm | 0.145 | 1.9% |
| sampling | 0.067 | 0.9% |
| activation | 0.038 | 0.5% |
| GPU idle | 0.137 | 1.8% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.183 |
| `marlin::Marlin` | linear | 72 | 1.726 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_mha` | attention | 36 | 0.652 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.143 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.101 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.038 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 212.6 ms, of which kernels 206.9 ms (97.3%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 185.166 | 87.1% |
| attention | 13.495 | 6.3% |
| activation | 5.431 | 2.6% |
| norm | 2.440 | 1.1% |
| other | 0.346 | 0.2% |
| GPU idle | 5.717 | 2.7% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `marlin::Marlin` | linear | 216 | 183.864 |
| `fmha_v2_flash_attention_bf16_64_32_S_qkv_128_causal_sm89_kernel_nl` | attention | 36 | 11.818 |
| `silu_and_mul_kernel` | activation | 36 | 5.431 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 2.428 |
| `internal::gemvx::kernel` | linear | 1 | 1.302 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.941 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.732 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.157 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.099 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.021 |
| `at::native::elementwise_kernel` | other | 1 | 0.015 |
| `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object` | norm | 1 | 0.013 |

## c1_in3900: 2 requests, 3900 in / 64 out, 1 at a time

Client: TPOT p50 7.61 ms, TTFT p50 657.1 ms, 2 completed. 69418 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 127 steps): 7.75 ms, of which kernels 7.61 ms (98.3%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.208 | 80.1% |
| attention | 0.990 | 12.8% |
| other | 0.166 | 2.1% |
| norm | 0.145 | 1.9% |
| sampling | 0.066 | 0.9% |
| activation | 0.038 | 0.5% |
| GPU idle | 0.132 | 1.7% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.180 |
| `marlin::Marlin` | linear | 72 | 1.729 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_mha` | attention | 36 | 0.814 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.143 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.101 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.066 |
| `silu_and_mul_kernel` | activation | 36 | 0.038 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 436.7 ms, of which kernels 408.8 ms (93.6%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 350.173 | 80.2% |
| attention | 40.933 | 9.4% |
| activation | 10.912 | 2.5% |
| norm | 6.394 | 1.5% |
| other | 0.433 | 0.1% |
| GPU idle | 27.895 | 6.4% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `marlin::Marlin` | linear | 396 | 344.933 |
| `fmha_v2_flash_attention_bf16_64_32_S_qkv_128_causal_sm89_kernel_nl` | attention | 36 | 37.264 |
| `silu_and_mul_kernel` | activation | 36 | 10.912 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 6.372 |
| `marlin::Marlin` | linear | 108 | 3.920 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 2.124 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 1.541 |
| `internal::gemvx::kernel` | linear | 1 | 1.321 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.205 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.106 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.041 |
| `at::native::elementwise_kernel` | other | 1 | 0.027 |

## c2_in128: 4 requests, 128 in / 64 out, 2 at a time

Client: TPOT p50 7.47 ms, TTFT p50 56.7 ms, 4 completed. 70292 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 7.54 ms, of which kernels 7.41 ms (98.2%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.246 | 82.8% |
| attention | 0.693 | 9.2% |
| norm | 0.189 | 2.5% |
| other | 0.170 | 2.3% |
| sampling | 0.068 | 0.9% |
| activation | 0.039 | 0.5% |
| GPU idle | 0.133 | 1.8% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.194 |
| `marlin::Marlin` | linear | 72 | 1.745 |
| `cutlass::Kernel2` | linear | 1 | 1.307 |
| `kernel_mha` | attention | 36 | 0.513 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.187 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.123 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.104 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.068 |
| `silu_and_mul_kernel` | activation | 36 | 0.039 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

## c2_in2048: 4 requests, 2048 in / 64 out, 2 at a time

Client: TPOT p50 9.54 ms, TTFT p50 321.5 ms, 4 completed. 70580 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 7.91 ms, of which kernels 7.78 ms (98.4%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.241 | 78.9% |
| attention | 1.078 | 13.6% |
| norm | 0.185 | 2.3% |
| other | 0.166 | 2.1% |
| sampling | 0.068 | 0.9% |
| activation | 0.038 | 0.5% |
| GPU idle | 0.129 | 1.6% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.197 |
| `marlin::Marlin` | linear | 72 | 1.738 |
| `cutlass::Kernel2` | linear | 1 | 1.307 |
| `kernel_mha` | attention | 36 | 0.902 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.183 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.068 |
| `silu_and_mul_kernel` | activation | 36 | 0.038 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

## c4_in128: 8 requests, 128 in / 64 out, 4 at a time

Client: TPOT p50 7.68 ms, TTFT p50 71.1 ms, 8 completed. 70294 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 7.66 ms, of which kernels 7.53 ms (98.3%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.276 | 82.0% |
| attention | 0.792 | 10.3% |
| norm | 0.183 | 2.4% |
| other | 0.169 | 2.2% |
| sampling | 0.067 | 0.9% |
| activation | 0.039 | 0.5% |
| GPU idle | 0.132 | 1.7% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.208 |
| `marlin::Marlin` | linear | 72 | 1.755 |
| `cutlass::Kernel2` | linear | 1 | 1.313 |
| `kernel_mha` | attention | 36 | 0.616 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.181 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.039 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.004 |

## c4_in2048: 8 requests, 2048 in / 64 out, 4 at a time

Client: TPOT p50 8.37 ms, TTFT p50 833.7 ms, 8 completed. 71158 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 8.31 ms, of which kernels 8.19 ms (98.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.289 | 75.6% |
| attention | 1.438 | 17.3% |
| norm | 0.183 | 2.2% |
| other | 0.169 | 2.0% |
| sampling | 0.065 | 0.8% |
| activation | 0.039 | 0.5% |
| GPU idle | 0.128 | 1.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.226 |
| `marlin::Marlin` | linear | 72 | 1.750 |
| `cutlass::Kernel2` | linear | 1 | 1.313 |
| `kernel_mha` | attention | 36 | 1.262 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.181 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.065 |
| `silu_and_mul_kernel` | activation | 36 | 0.039 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.004 |
