# trtllm_nvfp4: GPU kernel time (Nsight Systems, CUDA graphs traced per node)

```
2026-11-24T20:49:00+08:00
engine=trtllm model=nvidia/Qwen3-8B-NVFP4 snapshot=ccd10a893cbca613259517c3efe08e151ddf2b8e config=trtllm_nvfp4.yaml
image=nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a
nsys=NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0 args=--session-new=bench --trace=cuda --cuda-graph-trace=node
repo_commit=404f449
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W
```

## c1_in128: 4 requests, 128 in / 64 out, 1 at a time

Client: TPOT p50 7.26 ms, TTFT p50 28.6 ms, 4 completed. 137388 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 252 of 255 steps): 7.28 ms, of which kernels 7.15 ms (98.1%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.156 | 84.5% |
| attention | 0.575 | 7.9% |
| other | 0.166 | 2.3% |
| norm | 0.145 | 2.0% |
| sampling | 0.066 | 0.9% |
| activation | 0.038 | 0.5% |
| GPU idle | 0.136 | 1.9% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.164 |
| `marlin::Marlin` | linear | 72 | 1.694 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_mha` | attention | 36 | 0.394 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.144 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.122 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.066 |
| `silu_and_mul_kernel` | activation | 36 | 0.038 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 4 requests): 26.4 ms, of which kernels 15.2 ms (57.7%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 14.024 | 53.2% |
| attention | 0.564 | 2.1% |
| other | 0.284 | 1.1% |
| norm | 0.231 | 0.9% |
| activation | 0.112 | 0.4% |
| GPU idle | 11.154 | 42.3% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 7.984 |
| `marlin::Marlin` | linear | 72 | 4.739 |
| `internal::gemvx::kernel` | linear | 1 | 1.301 |
| `fmha_v2_flash_attention_bf16_64_32_S_qkv_128_causal_sm89_kernel_nl` | attention | 36 | 0.343 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.229 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.134 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.118 |
| `silu_and_mul_kernel` | activation | 36 | 0.112 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.099 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.092 |
| `at::native::index_elementwise_kernel` | other | 4 | 0.006 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |

## c1_in2048: 2 requests, 2048 in / 64 out, 1 at a time

Client: TPOT p50 8.14 ms, TTFT p50 218.6 ms, 2 completed. 68838 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 127 steps): 8.15 ms, of which kernels 8.01 ms (98.3%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.157 | 75.6% |
| attention | 1.435 | 17.6% |
| other | 0.165 | 2.0% |
| norm | 0.145 | 1.8% |
| sampling | 0.068 | 0.8% |
| activation | 0.038 | 0.5% |
| GPU idle | 0.140 | 1.7% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.166 |
| `marlin::Marlin` | linear | 72 | 1.692 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_mha` | attention | 36 | 1.255 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.143 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.068 |
| `silu_and_mul_kernel` | activation | 36 | 0.038 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 212.5 ms, of which kernels 205.0 ms (96.5%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 183.358 | 86.3% |
| attention | 13.482 | 6.3% |
| activation | 5.424 | 2.6% |
| norm | 2.423 | 1.1% |
| other | 0.351 | 0.2% |
| GPU idle | 7.449 | 3.5% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `marlin::Marlin` | linear | 216 | 182.057 |
| `fmha_v2_flash_attention_bf16_64_32_S_qkv_128_causal_sm89_kernel_nl` | attention | 36 | 11.816 |
| `silu_and_mul_kernel` | activation | 36 | 5.424 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 2.410 |
| `internal::gemvx::kernel` | linear | 1 | 1.301 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.931 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.730 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.161 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.101 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.020 |
| `at::native::elementwise_kernel` | other | 1 | 0.015 |
| `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object` | norm | 1 | 0.013 |

## c4_in128: 8 requests, 128 in / 64 out, 4 at a time

Client: TPOT p50 7.42 ms, TTFT p50 76.6 ms, 8 completed. 70290 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 7.41 ms, of which kernels 7.27 ms (98.1%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.229 | 84.1% |
| attention | 0.587 | 7.9% |
| norm | 0.182 | 2.5% |
| other | 0.167 | 2.3% |
| sampling | 0.068 | 0.9% |
| activation | 0.039 | 0.5% |
| GPU idle | 0.139 | 1.9% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.200 |
| `marlin::Marlin` | linear | 72 | 1.716 |
| `cutlass::Kernel2` | linear | 1 | 1.313 |
| `kernel_mha` | attention | 36 | 0.409 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.180 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.073 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.068 |
| `silu_and_mul_kernel` | activation | 36 | 0.039 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.004 |
