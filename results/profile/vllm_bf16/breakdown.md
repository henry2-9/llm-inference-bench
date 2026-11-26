# vllm_bf16: GPU kernel time (Nsight Systems, CUDA graphs traced per node)

```
2026-11-24T20:27:14+08:00
engine=vllm model=Qwen/Qwen3-8B snapshot=b968826d9c46dd6066d109eabc6255188de91218 extra=
image=vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
nsys=NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0 args=--session-new=bench --trace=cuda --cuda-graph-trace=node
repo_commit=404f449
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W
```

## c1_in128: 4 requests, 128 in / 64 out, 1 at a time

Client: TPOT p50 17.28 ms, TTFT p50 27.1 ms, 4 completed. 107668 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 252 of 255 steps): 17.26 ms, of which kernels 17.17 ms (99.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 16.216 | 94.0% |
| attention | 0.551 | 3.2% |
| other | 0.146 | 0.8% |
| norm | 0.128 | 0.7% |
| sampling | 0.082 | 0.5% |
| activation | 0.040 | 0.2% |
| GPU idle | 0.089 | 0.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `internal::gemvx::kernel` | linear | 108 | 13.545 |
| `internal::gemvx::kernel` | linear | 37 | 2.671 |
| `flash::flash_fwd_splitkv_kernel` | attention | 36 | 0.249 |
| `flash::flash_fwd_splitkv_combine_kernel` | attention | 36 | 0.243 |
| `triton_red_fused_3` | other | 35 | 0.104 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 0.066 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 0.059 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.058 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 0.040 |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | sampling | 1 | 0.019 |
| `flashinfer::sampling::TopPSamplingFromProbKernel` | sampling | 1 | 0.012 |

**Prefill** (first kernel of a request to its first sampled token; median of 4 requests): 26.7 ms, of which kernels 21.4 ms (80.1%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 20.410 | 76.5% |
| attention | 0.400 | 1.5% |
| norm | 0.194 | 0.7% |
| other | 0.177 | 0.7% |
| activation | 0.129 | 0.5% |
| sampling | 0.070 | 0.3% |
| GPU idle | 5.301 | 19.9% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` | linear | 36 | 9.565 |
| `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` | linear | 36 | 5.276 |
| `ampere_bf16_s1688gemm_bf16_64x64_sliced1x4_ldg8_f2f_tn` | linear | 36 | 2.799 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | linear | 36 | 1.471 |
| `internal::gemvx::kernel` | linear | 1 | 1.303 |
| `flash::flash_fwd_splitkv_kernel` | attention | 36 | 0.324 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 0.129 |
| `triton_red_fused_3` | other | 35 | 0.118 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 0.101 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 0.089 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.077 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |

## c1_in2048: 2 requests, 2048 in / 64 out, 1 at a time

Client: TPOT p50 17.54 ms, TTFT p50 214.0 ms, 2 completed. 53834 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 126 of 127 steps): 17.56 ms, of which kernels 17.47 ms (99.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 16.231 | 92.4% |
| attention | 0.838 | 4.8% |
| other | 0.149 | 0.8% |
| norm | 0.128 | 0.7% |
| sampling | 0.085 | 0.5% |
| activation | 0.040 | 0.2% |
| GPU idle | 0.089 | 0.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `internal::gemvx::kernel` | linear | 108 | 13.553 |
| `internal::gemvx::kernel` | linear | 37 | 2.678 |
| `flash::flash_fwd_splitkv_kernel` | attention | 36 | 0.536 |
| `flash::flash_fwd_splitkv_combine_kernel` | attention | 36 | 0.243 |
| `triton_red_fused_3` | other | 35 | 0.104 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 0.066 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 0.059 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.058 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 0.040 |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | sampling | 1 | 0.019 |
| `flashinfer::sampling::TopPSamplingFromProbKernel` | sampling | 1 | 0.014 |

**Prefill** (first kernel of a request to its first sampled token; median of 2 requests): 209.2 ms, of which kernels 203.1 ms (97.1%).

| part | ms per prefill | % of prefill |
|---|---|---|
| linear | 180.649 | 86.4% |
| attention | 12.417 | 5.9% |
| activation | 5.254 | 2.5% |
| norm | 3.532 | 1.7% |
| other | 1.188 | 0.6% |
| sampling | 0.073 | 0.0% |
| GPU idle | 6.074 | 2.9% |

| kernel | part | runs per prefill | ms per prefill |
|---|---|---|---|
| `ampere_bf16_s1688gemm_bf16_64x128_sliced1x2_ldg8_f2f_tn` | linear | 72 | 116.424 |
| `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` | linear | 72 | 62.923 |
| `flash::flash_fwd_splitkv_kernel` | attention | 36 | 12.050 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 5.254 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 2.182 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 1.327 |
| `internal::gemvx::kernel` | linear | 1 | 1.303 |
| `triton_red_fused_3` | other | 35 | 1.097 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.367 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.052 |
| `triton_red_fused_1` | other | 1 | 0.032 |
| `triton_red_fused__to_copy_embedding_rms_norm_0` | norm | 1 | 0.022 |

## c4_in128: 8 requests, 128 in / 64 out, 4 at a time

Client: TPOT p50 18.04 ms, TTFT p50 70.3 ms, 8 completed. 63696 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 126 of 129 steps): 18.03 ms, of which kernels 17.99 ms (99.8%).

| part | ms per step | % of step |
|---|---|---|
| linear | 16.939 | 93.9% |
| attention | 0.489 | 2.7% |
| other | 0.277 | 1.5% |
| norm | 0.138 | 0.8% |
| sampling | 0.102 | 0.6% |
| activation | 0.048 | 0.3% |
| GPU idle | 0.039 | 0.2% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `cutlass::Kernel2` | linear | 145 | 16.939 |
| `flash::flash_fwd_splitkv_kernel` | attention | 36 | 0.300 |
| `flash::flash_fwd_splitkv_combine_kernel` | attention | 36 | 0.128 |
| `at::native::elementwise_kernel` | other | 72 | 0.120 |
| `triton_red_fused_3` | other | 35 | 0.111 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 0.073 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 0.062 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.061 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 0.048 |
| `flashinfer::sampling::TopPSamplingFromProbKernel` | sampling | 1 | 0.032 |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | sampling | 1 | 0.020 |
