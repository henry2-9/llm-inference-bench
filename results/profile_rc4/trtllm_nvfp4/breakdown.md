# trtllm_nvfp4: GPU kernel time (Nsight Systems, CUDA graphs traced per node)

```
2026-11-26T01:14:39+08:00
engine=trtllm model=nvidia/Qwen3-8B-NVFP4 snapshot=ccd10a893cbca613259517c3efe08e151ddf2b8e config=trtllm_nvfp4.yaml server_env=
image=nvcr.io/nvidia/tensorrt-llm/release@sha256:813f751bc2c7ed867a53915dedde1c153ca94758e931d989e9c25835f9ba1814
nsys=NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0 args=--session-new=bench --trace=cuda --cuda-graph-trace=node
repo_commit=bb4f8dd-dirty
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W
```

## c1_in128: 4 requests, 128 in / 64 out, 1 at a time

Client: TPOT p50 7.29 ms, TTFT p50 30.6 ms, 4 completed. 137392 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 246 of 255 steps): 7.28 ms, of which kernels 7.15 ms (98.1%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.157 | 84.5% |
| attention | 0.575 | 7.9% |
| other | 0.166 | 2.3% |
| norm | 0.145 | 2.0% |
| sampling | 0.066 | 0.9% |
| activation | 0.038 | 0.5% |
| GPU idle | 0.136 | 1.9% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.163 |
| `marlin::Marlin` | linear | 72 | 1.696 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_mha` | attention | 36 | 0.399 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.144 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.101 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.066 |
| `silu_and_mul_kernel` | activation | 36 | 0.038 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 4 requests): 27.8 ms, of which kernels 15.2 ms (54.8%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 14.020 | 50.5% |
| attention | 0.579 | 2.1% |
| other | 0.283 | 1.0% |
| norm | 0.232 | 0.8% |
| activation | 0.112 | 0.4% |
| GPU idle | 12.545 | 45.2% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 7.997 |
| `marlin::Marlin` | linear | 72 | 4.724 |
| `internal::gemvx::kernel` | linear | 1 | 1.302 |
| `fmha_v2_flash_attention_bf16_64_32_S_qkv_128_causal_sm89_kernel_nl` | attention | 36 | 0.345 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.229 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.134 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.122 |
| `silu_and_mul_kernel` | activation | 36 | 0.112 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.109 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.091 |
| `at::native::index_elementwise_kernel` | other | 4 | 0.006 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |

## c1_in2048: 2 requests, 2048 in / 64 out, 1 at a time

Client: TPOT p50 8.20 ms, TTFT p50 220.7 ms, 2 completed. 68840 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 127 steps): 8.14 ms, of which kernels 8.00 ms (98.4%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.159 | 75.7% |
| attention | 1.429 | 17.6% |
| other | 0.166 | 2.0% |
| norm | 0.145 | 1.8% |
| sampling | 0.067 | 0.8% |
| activation | 0.038 | 0.5% |
| GPU idle | 0.134 | 1.6% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.165 |
| `marlin::Marlin` | linear | 72 | 1.696 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_mha` | attention | 36 | 1.253 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.143 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.101 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.038 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 214.2 ms, of which kernels 208.0 ms (97.1%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 186.103 | 86.9% |
| attention | 13.642 | 6.4% |
| activation | 5.450 | 2.5% |
| norm | 2.439 | 1.1% |
| other | 0.352 | 0.2% |
| GPU idle | 6.216 | 2.9% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `marlin::Marlin` | linear | 216 | 184.800 |
| `fmha_v2_flash_attention_bf16_64_32_S_qkv_128_causal_sm89_kernel_nl` | attention | 36 | 11.960 |
| `silu_and_mul_kernel` | activation | 36 | 5.450 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 2.426 |
| `internal::gemvx::kernel` | linear | 1 | 1.302 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.943 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.736 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.162 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.099 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.021 |
| `at::native::elementwise_kernel` | other | 1 | 0.015 |
| `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object` | norm | 1 | 0.013 |

## c1_in3900: 2 requests, 3900 in / 64 out, 1 at a time

Client: TPOT p50 8.88 ms, TTFT p50 796.4 ms, 2 completed. 69418 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 125 of 127 steps): 8.96 ms, of which kernels 8.83 ms (98.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.159 | 68.7% |
| attention | 2.255 | 25.2% |
| other | 0.166 | 1.9% |
| norm | 0.145 | 1.6% |
| sampling | 0.066 | 0.7% |
| activation | 0.038 | 0.4% |
| GPU idle | 0.134 | 1.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.166 |
| `kernel_mha` | attention | 36 | 2.079 |
| `marlin::Marlin` | linear | 72 | 1.695 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.143 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.101 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.066 |
| `silu_and_mul_kernel` | activation | 36 | 0.038 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 457.9 ms, of which kernels 415.8 ms (90.8%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 355.371 | 77.6% |
| attention | 42.699 | 9.3% |
| activation | 10.915 | 2.4% |
| norm | 6.415 | 1.4% |
| other | 0.432 | 0.1% |
| GPU idle | 42.096 | 9.2% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `marlin::Marlin` | linear | 396 | 350.106 |
| `fmha_v2_flash_attention_bf16_64_32_S_qkv_128_causal_sm89_kernel_nl` | attention | 36 | 39.053 |
| `silu_and_mul_kernel` | activation | 36 | 10.915 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 6.393 |
| `marlin::Marlin` | linear | 108 | 3.944 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 2.113 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 1.529 |
| `internal::gemvx::kernel` | linear | 1 | 1.321 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.203 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.107 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.040 |
| `at::native::elementwise_kernel` | other | 1 | 0.027 |

## c2_in128: 4 requests, 128 in / 64 out, 2 at a time

Client: TPOT p50 7.39 ms, TTFT p50 58.5 ms, 4 completed. 70292 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 7.37 ms, of which kernels 7.23 ms (98.1%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.194 | 84.0% |
| attention | 0.577 | 7.8% |
| norm | 0.187 | 2.5% |
| other | 0.168 | 2.3% |
| sampling | 0.067 | 0.9% |
| activation | 0.039 | 0.5% |
| GPU idle | 0.137 | 1.9% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.179 |
| `marlin::Marlin` | linear | 72 | 1.707 |
| `cutlass::Kernel2` | linear | 1 | 1.307 |
| `kernel_mha` | attention | 36 | 0.400 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.185 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.122 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.039 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |

## c2_in2048: 4 requests, 2048 in / 64 out, 2 at a time

Client: TPOT p50 9.87 ms, TTFT p50 321.8 ms, 4 completed. 70580 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 8.21 ms, of which kernels 8.08 ms (98.4%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.192 | 75.4% |
| attention | 1.429 | 17.4% |
| norm | 0.185 | 2.3% |
| other | 0.167 | 2.0% |
| sampling | 0.066 | 0.8% |
| activation | 0.038 | 0.5% |
| GPU idle | 0.134 | 1.6% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.180 |
| `marlin::Marlin` | linear | 72 | 1.705 |
| `cutlass::Kernel2` | linear | 1 | 1.307 |
| `kernel_mha` | attention | 36 | 1.253 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.183 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.101 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.066 |
| `silu_and_mul_kernel` | activation | 36 | 0.038 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.003 |

## c4_in128: 8 requests, 128 in / 64 out, 4 at a time

Client: TPOT p50 7.46 ms, TTFT p50 72.1 ms, 8 completed. 70294 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 7.41 ms, of which kernels 7.27 ms (98.2%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.230 | 84.1% |
| attention | 0.584 | 7.9% |
| norm | 0.184 | 2.5% |
| other | 0.169 | 2.3% |
| sampling | 0.068 | 0.9% |
| activation | 0.039 | 0.5% |
| GPU idle | 0.136 | 1.8% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.200 |
| `marlin::Marlin` | linear | 72 | 1.717 |
| `cutlass::Kernel2` | linear | 1 | 1.313 |
| `kernel_mha` | attention | 36 | 0.408 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.182 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.068 |
| `silu_and_mul_kernel` | activation | 36 | 0.039 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.004 |

## c4_in2048: 8 requests, 2048 in / 64 out, 4 at a time

Client: TPOT p50 8.38 ms, TTFT p50 834.8 ms, 8 completed. 71158 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 8.29 ms, of which kernels 8.15 ms (98.4%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.231 | 75.2% |
| attention | 1.463 | 17.7% |
| norm | 0.183 | 2.2% |
| other | 0.169 | 2.0% |
| sampling | 0.066 | 0.8% |
| activation | 0.039 | 0.5% |
| GPU idle | 0.136 | 1.6% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.203 |
| `marlin::Marlin` | linear | 72 | 1.716 |
| `cutlass::Kernel2` | linear | 1 | 1.313 |
| `kernel_mha` | attention | 36 | 1.287 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.181 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.066 |
| `silu_and_mul_kernel` | activation | 36 | 0.039 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.004 |
