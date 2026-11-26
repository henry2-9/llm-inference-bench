# trtllm_fp8: GPU kernel time (Nsight Systems, CUDA graphs traced per node)

```
2026-11-24T20:40:29+08:00
engine=trtllm model=nvidia/Qwen3-8B-FP8 snapshot=2cebc4c89e25abc17668c81b01dceaf3d8b914d5 config=trtllm_fp8.yaml
image=nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a
nsys=NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0 args=--session-new=bench --trace=cuda --cuda-graph-trace=node
repo_commit=404f449
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W
```

## c1_in128: 4 requests, 128 in / 64 out, 1 at a time

Client: TPOT p50 11.74 ms, TTFT p50 32.2 ms, 4 completed. 109884 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 252 of 255 steps): 11.75 ms, of which kernels 11.52 ms (98.0%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.843 | 75.3% |
| attention | 2.256 | 19.2% |
| other | 0.165 | 1.4% |
| norm | 0.136 | 1.2% |
| sampling | 0.067 | 0.6% |
| activation | 0.050 | 0.4% |
| GPU idle | 0.233 | 2.0% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.544 |
| `tensorrt_llm::_v1::kernels::mmha::masked_multihead_attention_kernel` | attention | 36 | 2.177 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.134 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.122 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.003 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |
| `at::native::index_elementwise_kernel` | other | 1 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 4 requests): 30.1 ms, of which kernels 11.5 ms (38.2%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 10.468 | 34.8% |
| attention | 0.429 | 1.4% |
| other | 0.261 | 0.9% |
| norm | 0.220 | 0.7% |
| activation | 0.104 | 0.3% |
| GPU idle | 18.609 | 61.8% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize64x64x64_stage4_warps` | linear | 144 | 9.145 |
| `internal::gemvx::kernel` | linear | 1 | 1.323 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.217 |
| `fmha_v2_flash_attention_e4m3_fp32_64_32_S_qkv_128_causal_sm89_kernel_n` | attention | 36 | 0.203 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.115 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.113 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.109 |
| `silu_and_mul_kernel` | activation | 36 | 0.104 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.089 |
| `at::native::index_elementwise_kernel` | other | 4 | 0.006 |
| `at::native::unrolled_elementwise_kernel` | other | 3 | 0.004 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |

## c1_in2048: 2 requests, 2048 in / 64 out, 1 at a time

Client: TPOT p50 11.77 ms, TTFT p50 118.6 ms, 2 completed. 54942 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 127 steps): 11.78 ms, of which kernels 11.54 ms (98.0%).

| part | ms per step | % of step |
|---|---|---|
| linear | 8.844 | 75.1% |
| attention | 2.280 | 19.4% |
| other | 0.165 | 1.4% |
| norm | 0.136 | 1.2% |
| sampling | 0.067 | 0.6% |
| activation | 0.050 | 0.4% |
| GPU idle | 0.235 | 2.0% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.545 |
| `tensorrt_llm::_v1::kernels::mmha::masked_multihead_attention_kernel` | attention | 36 | 2.201 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.134 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.122 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.050 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.003 |
| `at::native::unrolled_elementwise_kernel` | other | 2 | 0.003 |
| `at::native::index_elementwise_kernel` | other | 1 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 112.6 ms, of which kernels 104.7 ms (93.0%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 87.998 | 78.2% |
| attention | 7.924 | 7.0% |
| activation | 5.547 | 4.9% |
| norm | 2.498 | 2.2% |
| other | 0.714 | 0.6% |
| GPU idle | 7.898 | 7.0% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x128x64_stage3_war` | linear | 144 | 86.693 |
| `fmha_v2_flash_attention_e4m3_fp32_64_32_S_qkv_128_causal_sm89_kernel_n` | attention | 36 | 6.326 |
| `silu_and_mul_kernel` | activation | 36 | 5.547 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 2.486 |
| `internal::gemvx::kernel` | linear | 1 | 1.306 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.912 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.682 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.527 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.098 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.020 |
| `at::native::elementwise_kernel` | other | 1 | 0.015 |
| `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object` | norm | 1 | 0.013 |

## c4_in128: 8 requests, 128 in / 64 out, 4 at a time

Client: TPOT p50 10.24 ms, TTFT p50 90.0 ms, 8 completed. 60570 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 10.28 ms, of which kernels 10.02 ms (97.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 9.002 | 87.6% |
| attention | 0.567 | 5.5% |
| norm | 0.170 | 1.7% |
| other | 0.165 | 1.6% |
| sampling | 0.069 | 0.7% |
| activation | 0.052 | 0.5% |
| GPU idle | 0.259 | 2.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `tensorrt_llm::_v1::kernels::cuda_core_gemm::cudaCoreGemmV2` | linear | 144 | 7.687 |
| `cutlass::Kernel2` | linear | 1 | 1.314 |
| `kernel_mha` | attention | 36 | 0.390 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.168 |
| `tensorrt_llm::_v1::common::scaleMatrixPerTensorVec` | other | 72 | 0.118 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.099 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.069 |
| `silu_and_mul_kernel` | activation | 36 | 0.052 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.004 |
