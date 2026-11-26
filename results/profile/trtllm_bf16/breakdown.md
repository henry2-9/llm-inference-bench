# trtllm_bf16: GPU kernel time (Nsight Systems, CUDA graphs traced per node)

```
2026-11-24T20:32:05+08:00
engine=trtllm model=Qwen/Qwen3-8B snapshot=b968826d9c46dd6066d109eabc6255188de91218 config=trtllm_bf16.yaml
image=nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a
nsys=NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0 args=--session-new=bench --trace=cuda --cuda-graph-trace=node
repo_commit=404f449
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W
```

## c1_in128: 4 requests, 128 in / 64 out, 1 at a time

Client: TPOT p50 17.12 ms, TTFT p50 36.4 ms, 4 completed. 100524 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 252 of 255 steps): 17.14 ms, of which kernels 17.02 ms (99.3%).

| part | ms per step | % of step |
|---|---|---|
| linear | 16.214 | 94.6% |
| attention | 0.531 | 3.1% |
| norm | 0.131 | 0.8% |
| sampling | 0.064 | 0.4% |
| other | 0.040 | 0.2% |
| activation | 0.039 | 0.2% |
| GPU idle | 0.117 | 0.7% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `internal::gemvx::kernel` | linear | 108 | 13.543 |
| `internal::gemvx::kernel` | linear | 37 | 2.671 |
| `kernel_mha` | attention | 36 | 0.390 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.129 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.068 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.068 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.064 |
| `silu_and_mul_kernel` | activation | 36 | 0.039 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.003 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.002 |
| `at::native::index_elementwise_kernel` | other | 1 | 0.002 |

**Prefill** (first kernel of a request to its first sampled token; median of 4 requests): 35.8 ms, of which kernels 22.0 ms (61.3%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 20.888 | 58.3% |
| attention | 0.568 | 1.6% |
| norm | 0.261 | 0.7% |
| other | 0.136 | 0.4% |
| activation | 0.134 | 0.4% |
| GPU idle | 13.852 | 38.7% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` | linear | 36 | 9.923 |
| `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` | linear | 36 | 5.324 |
| `ampere_bf16_s1688gemm_bf16_64x64_sliced1x4_ldg8_f2f_tn` | linear | 36 | 2.786 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | linear | 36 | 1.554 |
| `internal::gemvx::kernel` | linear | 1 | 1.303 |
| `fmha_v2_flash_attention_bf16_64_32_S_qkv_128_causal_sm89_kernel_nl` | attention | 36 | 0.341 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.259 |
| `silu_and_mul_kernel` | activation | 36 | 0.134 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.115 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.109 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.079 |
| `at::native::index_elementwise_kernel` | other | 4 | 0.006 |

## c1_in2048: 2 requests, 2048 in / 64 out, 1 at a time

Client: TPOT p50 17.40 ms, TTFT p50 221.5 ms, 2 completed. 50262 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 127 steps): 17.41 ms, of which kernels 17.29 ms (99.3%).

| part | ms per step | % of step |
|---|---|---|
| linear | 16.215 | 93.1% |
| attention | 0.794 | 4.6% |
| norm | 0.135 | 0.8% |
| sampling | 0.067 | 0.4% |
| other | 0.044 | 0.3% |
| activation | 0.039 | 0.2% |
| GPU idle | 0.121 | 0.7% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `internal::gemvx::kernel` | linear | 108 | 13.540 |
| `internal::gemvx::kernel` | linear | 37 | 2.675 |
| `kernel_mha` | attention | 36 | 0.630 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.133 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.090 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.072 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.039 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.003 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |
| `at::native::index_elementwise_kernel` | other | 1 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 216.1 ms, of which kernels 201.9 ms (93.5%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 180.001 | 83.3% |
| attention | 13.980 | 6.5% |
| activation | 5.093 | 2.4% |
| norm | 2.672 | 1.2% |
| other | 0.190 | 0.1% |
| GPU idle | 14.128 | 6.5% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` | linear | 72 | 116.038 |
| `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` | linear | 72 | 62.661 |
| `fmha_v2_flash_attention_bf16_64_32_S_qkv_128_causal_sm89_kernel_nl` | attention | 36 | 12.042 |
| `silu_and_mul_kernel` | activation | 36 | 5.093 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 2.660 |
| `internal::gemvx::kernel` | linear | 1 | 1.302 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 1.106 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.828 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.099 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.021 |
| `at::native::elementwise_kernel` | other | 1 | 0.015 |
| `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object` | norm | 1 | 0.012 |

## c4_in128: 8 requests, 128 in / 64 out, 4 at a time

Client: TPOT p50 18.18 ms, TTFT p50 75.1 ms, 8 completed. 51210 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 18.18 ms, of which kernels 18.08 ms (99.4%).

| part | ms per step | % of step |
|---|---|---|
| linear | 17.020 | 93.6% |
| attention | 0.725 | 4.0% |
| norm | 0.174 | 1.0% |
| sampling | 0.069 | 0.4% |
| activation | 0.049 | 0.3% |
| other | 0.047 | 0.3% |
| GPU idle | 0.100 | 0.6% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `cutlass::Kernel2` | linear | 145 | 17.020 |
| `kernel_mha` | attention | 36 | 0.556 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.172 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.093 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.073 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.069 |
| `silu_and_mul_kernel` | activation | 36 | 0.049 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |
| `at::native::indexSelectSmallIndex` | other | 1 | 0.003 |
