# trtllm_nvfp4: GPU kernel time (Nsight Systems, CUDA graphs traced per node)

```
2026-11-25T14:01:17+08:00
engine=trtllm model=nvidia/Qwen3-8B-NVFP4 snapshot=ccd10a893cbca613259517c3efe08e151ddf2b8e config=trtllm_nvfp4.yaml server_env=
server_docker_args=-v ~/Documents/TensorRT-LLM-build/engine_rc2/cpp-build/tensorrt_llm/libtensorrt_llm.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libtensorrt_llm.so:ro -v ~/Documents/TensorRT-LLM-build/engine_rc2/cpp-build/tensorrt_llm/thop/libth_common.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libth_common.so:ro -v ~/Documents/TensorRT-LLM-build/engine_rc2/cpp-build/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_0.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libdecoder_attention_0.so:ro -v ~/Documents/TensorRT-LLM-build/engine_rc2/cpp-build/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_1.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs/libdecoder_attention_1.so:ro -v ~/Documents/TensorRT-LLM-build/engine_rc2/cpp-build/tensorrt_llm/nanobind/bindings.cpython-312-x86_64-linux-gnu.so:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/bindings.cpython-312-x86_64-linux-gnu.so:ro
image=nvcr.io/nvidia/tensorrt-llm/release@sha256:93a3eaf1fa574813d608a7147735134bcfbd06d18d1aa98e8ee0d5da09e51e3a
nsys=NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0 args=--session-new=bench --trace=cuda --cuda-graph-trace=node
repo_commit=04b06dc-dirty
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W
```

## c1_in128: 4 requests, 128 in / 64 out, 1 at a time

Client: TPOT p50 7.41 ms, TTFT p50 29.4 ms, 4 completed. 137388 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 252 of 255 steps): 7.44 ms, of which kernels 7.30 ms (98.1%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.204 | 83.4% |
| attention | 0.677 | 9.1% |
| other | 0.165 | 2.2% |
| norm | 0.145 | 1.9% |
| sampling | 0.066 | 0.9% |
| activation | 0.038 | 0.5% |
| GPU idle | 0.139 | 1.9% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.179 |
| `marlin::Marlin` | linear | 72 | 1.726 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_mha` | attention | 36 | 0.497 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.143 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.066 |
| `silu_and_mul_kernel` | activation | 36 | 0.038 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 4 requests): 27.1 ms, of which kernels 15.2 ms (56.1%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 14.017 | 51.7% |
| attention | 0.562 | 2.1% |
| other | 0.284 | 1.0% |
| norm | 0.231 | 0.9% |
| activation | 0.113 | 0.4% |
| GPU idle | 11.904 | 43.9% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 7.986 |
| `marlin::Marlin` | linear | 72 | 4.730 |
| `internal::gemvx::kernel` | linear | 1 | 1.301 |
| `fmha_v2_flash_attention_bf16_64_32_S_qkv_128_causal_sm89_kernel_nl` | attention | 36 | 0.343 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.228 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.135 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.113 |
| `silu_and_mul_kernel` | activation | 36 | 0.113 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.092 |
| `at::native::index_elementwise_kernel` | other | 4 | 0.006 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |

## c1_in2048: 2 requests, 2048 in / 64 out, 1 at a time

Client: TPOT p50 7.56 ms, TTFT p50 216.1 ms, 2 completed. 68838 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 127 steps): 7.59 ms, of which kernels 7.45 ms (98.2%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.213 | 81.9% |
| attention | 0.819 | 10.8% |
| other | 0.165 | 2.2% |
| norm | 0.145 | 1.9% |
| sampling | 0.067 | 0.9% |
| activation | 0.038 | 0.5% |
| GPU idle | 0.137 | 1.8% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.190 |
| `marlin::Marlin` | linear | 72 | 1.724 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_mha` | attention | 36 | 0.639 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.143 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.101 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.038 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 210.7 ms, of which kernels 204.7 ms (97.2%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 183.056 | 86.9% |
| attention | 13.465 | 6.4% |
| activation | 5.406 | 2.6% |
| norm | 2.440 | 1.2% |
| other | 0.347 | 0.2% |
| GPU idle | 5.967 | 2.8% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `marlin::Marlin` | linear | 216 | 181.753 |
| `fmha_v2_flash_attention_bf16_64_32_S_qkv_128_causal_sm89_kernel_nl` | attention | 36 | 11.800 |
| `silu_and_mul_kernel` | activation | 36 | 5.406 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 2.427 |
| `internal::gemvx::kernel` | linear | 1 | 1.303 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.936 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.725 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.160 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.100 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.018 |
| `at::native::elementwise_kernel` | other | 1 | 0.015 |
| `kernel_cutlass_kernel_flashinfernormkernelsrmsnormRMSNormKernel_object` | norm | 1 | 0.012 |

## c1_in3900: 2 requests, 3900 in / 64 out, 1 at a time

Client: TPOT p50 7.73 ms, TTFT p50 420.7 ms, 2 completed. 69414 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 127 steps): 7.75 ms, of which kernels 7.61 ms (98.2%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.216 | 80.2% |
| attention | 0.978 | 12.6% |
| other | 0.165 | 2.1% |
| norm | 0.145 | 1.9% |
| sampling | 0.066 | 0.9% |
| activation | 0.038 | 0.5% |
| GPU idle | 0.140 | 1.8% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.189 |
| `marlin::Marlin` | linear | 72 | 1.727 |
| `internal::gemvx::kernel` | linear | 1 | 1.299 |
| `kernel_mha` | attention | 36 | 0.798 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.143 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.075 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.066 |
| `silu_and_mul_kernel` | activation | 36 | 0.038 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.003 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 413.8 ms, of which kernels 406.0 ms (98.1%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 347.762 | 84.0% |
| attention | 40.614 | 9.8% |
| activation | 10.785 | 2.6% |
| norm | 6.422 | 1.6% |
| other | 0.431 | 0.1% |
| GPU idle | 7.754 | 1.9% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `marlin::Marlin` | linear | 396 | 342.567 |
| `fmha_v2_flash_attention_bf16_64_32_S_qkv_128_causal_sm89_kernel_nl` | attention | 36 | 36.937 |
| `silu_and_mul_kernel` | activation | 36 | 10.785 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 6.399 |
| `marlin::Marlin` | linear | 108 | 3.877 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 2.091 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 1.583 |
| `internal::gemvx::kernel` | linear | 1 | 1.319 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.208 |
| `tensorrt_llm::_v1::kernels::computeSeqAndPaddingOffsets` | other | 36 | 0.106 |
| `at::native::vectorized_gather_kernel` | other | 1 | 0.035 |
| `at::native::elementwise_kernel` | other | 1 | 0.027 |

## c2_in128: 4 requests, 128 in / 64 out, 2 at a time

Client: TPOT p50 7.47 ms, TTFT p50 53.5 ms, 4 completed. 70288 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 7.51 ms, of which kernels 7.37 ms (98.2%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.239 | 83.0% |
| attention | 0.678 | 9.0% |
| norm | 0.186 | 2.5% |
| other | 0.166 | 2.2% |
| sampling | 0.067 | 0.9% |
| activation | 0.038 | 0.5% |
| GPU idle | 0.139 | 1.8% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.193 |
| `marlin::Marlin` | linear | 72 | 1.738 |
| `cutlass::Kernel2` | linear | 1 | 1.306 |
| `kernel_mha` | attention | 36 | 0.499 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.183 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.038 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.004 |

## c2_in2048: 4 requests, 2048 in / 64 out, 2 at a time

Client: TPOT p50 9.45 ms, TTFT p50 316.2 ms, 4 completed. 70576 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 7.90 ms, of which kernels 7.77 ms (98.3%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.245 | 79.1% |
| attention | 1.064 | 13.5% |
| norm | 0.185 | 2.3% |
| other | 0.166 | 2.1% |
| sampling | 0.068 | 0.9% |
| activation | 0.038 | 0.5% |
| GPU idle | 0.131 | 1.7% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.204 |
| `marlin::Marlin` | linear | 72 | 1.735 |
| `cutlass::Kernel2` | linear | 1 | 1.306 |
| `kernel_mha` | attention | 36 | 0.884 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.183 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.068 |
| `silu_and_mul_kernel` | activation | 36 | 0.038 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.004 |

## c4_in128: 8 requests, 128 in / 64 out, 4 at a time

Client: TPOT p50 7.67 ms, TTFT p50 73.1 ms, 8 completed. 70290 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 7.67 ms, of which kernels 7.53 ms (98.2%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.281 | 81.9% |
| attention | 0.795 | 10.4% |
| norm | 0.183 | 2.4% |
| other | 0.168 | 2.2% |
| sampling | 0.067 | 0.9% |
| activation | 0.040 | 0.5% |
| GPU idle | 0.138 | 1.8% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.216 |
| `marlin::Marlin` | linear | 72 | 1.754 |
| `cutlass::Kernel2` | linear | 1 | 1.312 |
| `kernel_mha` | attention | 36 | 0.615 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.181 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.067 |
| `silu_and_mul_kernel` | activation | 36 | 0.040 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `tensorrt_llm::batch_manager::kv_cache_manager_v2::copyBatchBlockOffset` | attention | 1 | 0.004 |

## c4_in2048: 8 requests, 2048 in / 64 out, 4 at a time

Client: TPOT p50 8.29 ms, TTFT p50 823.2 ms, 8 completed. 71154 kernels recorded.

**Decode step** (cut at `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel`; median of 126 of 129 steps): 8.31 ms, of which kernels 8.18 ms (98.4%).

| part | ms per step | % of step |
|---|---|---|
| linear | 6.295 | 75.7% |
| attention | 1.428 | 17.2% |
| norm | 0.183 | 2.2% |
| other | 0.168 | 2.0% |
| sampling | 0.068 | 0.8% |
| activation | 0.040 | 0.5% |
| GPU idle | 0.131 | 1.6% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `marlin::Marlin` | linear | 72 | 3.235 |
| `marlin::Marlin` | linear | 72 | 1.748 |
| `cutlass::Kernel2` | linear | 1 | 1.311 |
| `kernel_mha` | attention | 36 | 1.247 |
| `kernel_cutlass_kernel_flashinfernormkernelsfused_add_rmsnormFusedAddRM` | norm | 72 | 0.180 |
| `at::native::vectorized_elementwise_kernel` | other | 144 | 0.121 |
| `tensorrt_llm::_v1::kernels::applyBiasRopeUpdateKVCacheV2` | attention | 36 | 0.102 |
| `tensorrt_llm::_v1::kernels::fusedQKNormRopeKernel` | attention | 36 | 0.074 |
| `tensorrt_llm::_v1::kernels::fusedSampling::fusedSamplingKernel` | sampling | 1 | 0.068 |
| `silu_and_mul_kernel` | activation | 36 | 0.040 |
| `at::native::reduce_kernel` | other | 1 | 0.004 |
| `at::native::unrolled_elementwise_kernel` | other | 1 | 0.004 |
