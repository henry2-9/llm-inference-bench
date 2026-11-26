# trtllm_nvfp4: GPU kernel time (Nsight Systems, CUDA graphs traced per node)

```
2026-11-24T22:23:36+08:00
engine=trtllm model=nvidia/Qwen3-8B-NVFP4 snapshot=ccd10a893cbca613259517c3efe08e151ddf2b8e config=trtllm_nvfp4.yaml
image=nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a
nsys=NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0 args=--session-new=bench --trace=cuda --cuda-graph-trace=node
repo_commit=9992cdb
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W
```

## c1_in3900: 2 requests, 3900 in / 64 out, 1 at a time

Client: TPOT p50 9.05 ms, TTFT p50 449.2 ms, 2 completed. 69414 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 125 of 127 steps): 8.98 ms, of which kernels 8.84 ms (98.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.161 | 68.6% |
| attention | 2.265 | 25.2% |
| other | 0.165 | 1.8% |
| norm | 0.145 | 1.6% |
| sampling | 0.066 | 0.7% |
| activation | 0.038 | 0.4% |
| GPU idle | 0.139 | 1.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.169 |
| `kernel_mha` | attention | 36 | 2.085 |
| `marlin::Marlin` | linear | 72 | 1.693 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.143 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.066 |
| `silu_and_mul_kernel` | activation | 36 | 0.038 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 441.6 ms, of which kernels 432.4 ms (97.9%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 367.382 | 83.2% |
| attention | 45.714 | 10.4% |
| activation | 11.295 | 2.6% |
| norm | 7.588 | 1.7% |
| other | 0.436 | 0.1% |
| GPU idle | 9.197 | 2.1% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `marlin::Marlin` | linear | 396 | 361.718 |
| `fmha_v2_flash_attention_bf16_64_32_S_qkv_128_causal_sm89_kernel_nl` | attention | 36 | 41.556 |
| `silu_and_mul_kernel` | activation | 36 | 11.295 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 7.566 |
| `marlin::Marlin` | linear | 108 | 4.347 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 2.342 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 1.811 |
| `internal::gemvx::kernel` | linear | 1 | 1.318 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.201 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.109 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.044 |
| `at::native::elementwise_kernel` | other | 1 | 0.027 |

## c2_in128: 4 requests, 128 in / 64 out, 2 at a time

Client: TPOT p50 7.41 ms, TTFT p50 53.1 ms, 4 completed. 70288 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 7.36 ms, of which kernels 7.22 ms (98.1%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.190 | 84.1% |
| attention | 0.577 | 7.8% |
| norm | 0.185 | 2.5% |
| other | 0.165 | 2.2% |
| sampling | 0.066 | 0.9% |
| activation | 0.039 | 0.5% |
| GPU idle | 0.141 | 1.9% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.179 |
| `marlin::Marlin` | linear | 72 | 1.703 |
| `cutlass::Kernel2` | linear | 1 | 1.307 |
| `kernel_mha` | attention | 36 | 0.398 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.183 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.120 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.066 |
| `silu_and_mul_kernel` | activation | 36 | 0.039 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.004 |

## c4_in2048: 8 requests, 2048 in / 64 out, 4 at a time

Client: TPOT p50 8.39 ms, TTFT p50 830.1 ms, 8 completed. 71154 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 8.30 ms, of which kernels 8.17 ms (98.3%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.237 | 75.1% |
| attention | 1.472 | 17.7% |
| norm | 0.182 | 2.2% |
| other | 0.168 | 2.0% |
| sampling | 0.068 | 0.8% |
| activation | 0.039 | 0.5% |
| GPU idle | 0.137 | 1.7% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.210 |
| `marlin::Marlin` | linear | 72 | 1.715 |
| `cutlass::Kernel2` | linear | 1 | 1.312 |
| `kernel_mha` | attention | 36 | 1.292 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.180 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.073 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.068 |
| `silu_and_mul_kernel` | activation | 36 | 0.039 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.004 |

## c16_in128: 32 requests, 128 in / 64 out, 16 at a time

Client: TPOT p50 8.01 ms, TTFT p50 216.2 ms, 32 completed. 70434 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 7.90 ms, of which kernels 7.76 ms (98.2%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.483 | 82.0% |
| attention | 0.757 | 9.6% |
| other | 0.186 | 2.4% |
| norm | 0.183 | 2.3% |
| sampling | 0.098 | 1.2% |
| activation | 0.044 | 0.6% |
| GPU idle | 0.141 | 1.8% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.380 |
| `marlin::Marlin` | linear | 72 | 1.770 |
| `cutlass::Kernel2` | linear | 1 | 1.333 |
| `kernel_mha` | attention | 36 | 0.573 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.181 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.104 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.098 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `silu_and_mul_kernel` | activation | 36 | 0.044 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.009 |
| `at::native::indexSelectSmallIndex` | other | 1 | 0.007 |

## c64_in128: 128 requests, 128 in / 64 out, 64 at a time

Client: TPOT p50 11.48 ms, TTFT p50 796.8 ms, 128 completed. 76872 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 138 steps): 11.12 ms, of which kernels 10.97 ms (98.7%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.598 | 77.3% |
| attention | 1.799 | 16.2% |
| other | 0.203 | 1.8% |
| norm | 0.195 | 1.8% |
| sampling | 0.103 | 0.9% |
| activation | 0.066 | 0.6% |
| GPU idle | 0.148 | 1.3% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 108 | 3.814 |
| `marlin::Marlin` | linear | 36 | 3.344 |
| `kernel_mha` | attention | 36 | 1.597 |
| `cutlass::Kernel2` | linear | 1 | 1.439 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.193 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.120 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.110 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.103 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.086 |
| `silu_and_mul_kernel` | activation | 36 | 0.066 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.029 |
| `at::native::vectorized_elementwise_kernel` | other | 1 | 0.010 |
