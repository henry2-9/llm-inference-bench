# trtllm_bf16: GPU kernel time (Nsight Systems, CUDA graphs traced per node)

```
2026-11-24T22:03:34+08:00
engine=trtllm model=Qwen/Qwen3-8B snapshot=b968826d9c46dd6066d109eabc6255188de91218 config=trtllm_bf16.yaml
image=nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a
nsys=NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0 args=--session-new=bench --trace=cuda --cuda-graph-trace=node
repo_commit=9992cdb
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W
```

## c1_in3900: 2 requests, 3900 in / 64 out, 1 at a time

Client: TPOT p50 17.69 ms, TTFT p50 417.1 ms, 2 completed. 50262 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 127 steps): 17.72 ms, of which kernels 17.60 ms (99.3%).

| part | ms per step | % of step |
|---|---|---|
| linear | 16.223 | 91.6% |
| attention | 1.091 | 6.2% |
| norm | 0.135 | 0.8% |
| sampling | 0.066 | 0.4% |
| other | 0.044 | 0.2% |
| activation | 0.039 | 0.2% |
| GPU idle | 0.117 | 0.7% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `internal::gemvx::kernel` | linear | 108 | 13.547 |
| `internal::gemvx::kernel` | linear | 37 | 2.675 |
| `kernel_mha` | attention | 36 | 0.923 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.133 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.090 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.072 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.066 |
| `silu_and_mul_kernel` | activation | 36 | 0.039 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.003 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |
| `at::native::index_elementwise_kernel` | other | 1 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 410.3 ms, of which kernels 401.9 ms (98.0%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 342.002 | 83.4% |
| attention | 42.019 | 10.2% |
| activation | 10.487 | 2.6% |
| norm | 7.171 | 1.7% |
| other | 0.223 | 0.1% |
| GPU idle | 8.402 | 2.0% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_tn` | linear | 72 | 222.210 |
| `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` | linear | 36 | 87.935 |
| `fmha_v2_flash_attention_bf16_64_32_S_qkv_128_causal_sm89_kernel_nl` | attention | 36 | 37.508 |
| `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` | linear | 36 | 30.537 |
| `silu_and_mul_kernel` | activation | 36 | 10.487 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 7.149 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 2.403 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 2.104 |
| `internal::gemvx::kernel` | linear | 1 | 1.320 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.106 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.035 |
| `at::native::elementwise_kernel` | other | 1 | 0.027 |

## c2_in128: 4 requests, 128 in / 64 out, 2 at a time

Client: TPOT p50 17.94 ms, TTFT p50 51.3 ms, 4 completed. 51208 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 128 of 129 steps): 18.05 ms, of which kernels 17.95 ms (99.4%).

| part | ms per step | % of step |
|---|---|---|
| linear | 16.957 | 93.9% |
| attention | 0.654 | 3.6% |
| norm | 0.177 | 1.0% |
| sampling | 0.066 | 0.4% |
| activation | 0.047 | 0.3% |
| other | 0.045 | 0.2% |
| GPU idle | 0.106 | 0.6% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `cutlass::Kernel2` | linear | 145 | 16.957 |
| `kernel_mha` | attention | 36 | 0.483 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.175 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.093 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.073 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.066 |
| `silu_and_mul_kernel` | activation | 36 | 0.047 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.003 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |
| `at::native::indexSelectSmallIndex` | other | 1 | 0.002 |

## c4_in2048: 8 requests, 2048 in / 64 out, 4 at a time

Client: TPOT p50 19.37 ms, TTFT p50 816.8 ms, 8 completed. 51282 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 19.37 ms, of which kernels 19.27 ms (99.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 17.037 | 87.9% |
| attention | 1.885 | 9.7% |
| norm | 0.176 | 0.9% |
| sampling | 0.067 | 0.3% |
| activation | 0.049 | 0.3% |
| other | 0.047 | 0.2% |
| GPU idle | 0.105 | 0.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `cutlass::Kernel2` | linear | 145 | 17.037 |
| `kernel_mha` | attention | 36 | 1.714 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.174 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.093 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.049 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |
| `at::native::indexSelectSmallIndex` | other | 1 | 0.003 |

## c16_in128: 32 requests, 128 in / 64 out, 16 at a time

Client: TPOT p50 18.88 ms, TTFT p50 216.3 ms, 32 completed. 51210 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 18.88 ms, of which kernels 18.78 ms (99.4%).

| part | ms per step | % of step |
|---|---|---|
| linear | 17.290 | 91.6% |
| attention | 1.086 | 5.8% |
| norm | 0.176 | 0.9% |
| sampling | 0.098 | 0.5% |
| other | 0.066 | 0.3% |
| activation | 0.061 | 0.3% |
| GPU idle | 0.106 | 0.6% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `cutlass::Kernel2` | linear | 145 | 17.290 |
| `kernel_mha` | attention | 36 | 0.912 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.174 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.098 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.094 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.076 |
| `silu_and_mul_kernel` | activation | 36 | 0.061 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.009 |
| `at::native::indexSelectSmallIndex` | other | 1 | 0.007 |
| `at::native::indexSelectSmallIndex` | other | 1 | 0.006 |
| `at::native::reduce_kernel` | other | 1 | 0.006 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.004 |

## c64_in128: 128 requests, 128 in / 64 out, 64 at a time

Client: TPOT p50 22.00 ms, TTFT p50 810.8 ms, 128 completed. 62366 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 130 of 134 steps): 21.88 ms, of which kernels 21.77 ms (99.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 18.575 | 84.9% |
| attention | 2.571 | 11.7% |
| other | 0.247 | 1.1% |
| norm | 0.197 | 0.9% |
| sampling | 0.103 | 0.5% |
| activation | 0.078 | 0.4% |
| GPU idle | 0.112 | 0.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `ampere_bf16_s1688gemm_bf16_64x64_sliced1x4_ldg8_f2f_tn` | linear | 36 | 9.042 |
| `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` | linear | 36 | 4.376 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | linear | 72 | 3.716 |
| `kernel_mha` | attention | 36 | 2.377 |
| `cutlass::Kernel2` | linear | 1 | 1.439 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.195 |
| `cublasLt::splitKreduce_kernel` | other | 72 | 0.164 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.103 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.101 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.088 |
| `silu_and_mul_kernel` | activation | 36 | 0.078 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.031 |
