# vllm_bf16: GPU kernel time (Nsight Systems, CUDA graphs traced per node)

```
2026-11-24T21:57:52+08:00
engine=vllm model=Qwen/Qwen3-8B snapshot=b968826d9c46dd6066d109eabc6255188de91218 extra=
image=vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3
nsys=NVIDIA Nsight Systems version 2026.5.1.161-265138896106v0 args=--session-new=bench --trace=cuda --cuda-graph-trace=node
repo_commit=9992cdb
NVIDIA GeForce RTX 4090, 580.126.09, 24564 MiB, 450.00 W
```

## c1_in3900: 2 requests, 3900 in / 64 out, 1 at a time

Client: TPOT p50 17.95 ms, TTFT p50 430.4 ms, 2 completed. 54608 kernels recorded.

WARNING: `flashinfer::sampling::TopPSamplingFromProbKernel` ran 130 times, expected 128.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 126 of 129 steps): 17.83 ms, of which kernels 17.74 ms (99.5%).

| part | ms per step | % of step |
|---|---|---|
| linear | 16.228 | 91.0% |
| attention | 1.107 | 6.2% |
| other | 0.151 | 0.8% |
| norm | 0.128 | 0.7% |
| sampling | 0.082 | 0.5% |
| activation | 0.040 | 0.2% |
| GPU idle | 0.090 | 0.5% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `internal::gemvx::kernel` | linear | 108 | 13.551 |
| `internal::gemvx::kernel` | linear | 37 | 2.678 |
| `flash::flash_fwd_splitkv_kernel` | attention | 36 | 0.803 |
| `flash::flash_fwd_splitkv_combine_kernel` | attention | 36 | 0.245 |
| `triton_red_fused_3` | other | 35 | 0.104 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 0.067 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 0.059 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.059 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 0.040 |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | sampling | 1 | 0.019 |
| `flashinfer::sampling::TopPSamplingFromProbKernel` | sampling | 1 | 0.012 |

## c2_in128: 4 requests, 128 in / 64 out, 2 at a time

Client: TPOT p50 18.17 ms, TTFT p50 44.2 ms, 4 completed. 63550 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 128 of 129 steps): 18.01 ms, of which kernels 17.96 ms (99.8%).

| part | ms per step | % of step |
|---|---|---|
| linear | 16.908 | 93.9% |
| attention | 0.505 | 2.8% |
| other | 0.275 | 1.5% |
| norm | 0.139 | 0.8% |
| sampling | 0.089 | 0.5% |
| activation | 0.045 | 0.3% |
| GPU idle | 0.044 | 0.2% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `cutlass::Kernel2` | linear | 145 | 16.908 |
| `flash::flash_fwd_splitkv_kernel` | attention | 36 | 0.273 |
| `flash::flash_fwd_splitkv_combine_kernel` | attention | 36 | 0.170 |
| `at::native::elementwise_kernel` | other | 72 | 0.118 |
| `triton_red_fused_3` | other | 35 | 0.110 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 0.074 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 0.062 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.062 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 0.045 |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | sampling | 1 | 0.020 |
| `flashinfer::sampling::TopPSamplingFromProbKernel` | sampling | 1 | 0.019 |

## c4_in2048: 8 requests, 2048 in / 64 out, 4 at a time

Client: TPOT p50 22.41 ms, TTFT p50 641.2 ms, 8 completed. 65373 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 127 of 134 steps): 19.27 ms, of which kernels 19.22 ms (99.8%).

| part | ms per step | % of step |
|---|---|---|
| linear | 16.940 | 87.9% |
| attention | 1.712 | 8.9% |
| other | 0.281 | 1.5% |
| norm | 0.138 | 0.7% |
| sampling | 0.099 | 0.5% |
| activation | 0.050 | 0.3% |
| GPU idle | 0.044 | 0.2% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `cutlass::Kernel2` | linear | 145 | 16.940 |
| `flash::flash_fwd_splitkv_kernel` | attention | 36 | 1.522 |
| `flash::flash_fwd_splitkv_combine_kernel` | attention | 36 | 0.128 |
| `at::native::elementwise_kernel` | other | 72 | 0.123 |
| `triton_red_fused_3` | other | 35 | 0.111 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 0.073 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 0.062 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.061 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 0.050 |
| `flashinfer::sampling::TopPSamplingFromProbKernel` | sampling | 1 | 0.029 |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | sampling | 1 | 0.020 |

## c16_in128: 32 requests, 128 in / 64 out, 16 at a time

Client: TPOT p50 18.66 ms, TTFT p50 225.1 ms, 32 completed. 64482 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 127 of 131 steps): 18.51 ms, of which kernels 18.46 ms (99.7%).

| part | ms per step | % of step |
|---|---|---|
| linear | 17.045 | 92.1% |
| attention | 0.810 | 4.4% |
| other | 0.294 | 1.6% |
| norm | 0.140 | 0.8% |
| sampling | 0.106 | 0.6% |
| activation | 0.052 | 0.3% |
| GPU idle | 0.050 | 0.3% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `cutlass::Kernel2` | linear | 145 | 17.045 |
| `flash::flash_fwd_splitkv_kernel` | attention | 36 | 0.640 |
| `at::native::elementwise_kernel` | other | 72 | 0.126 |
| `triton_red_fused_3` | other | 35 | 0.112 |
| `flash::flash_fwd_splitkv_combine_kernel` | attention | 36 | 0.107 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 0.074 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 0.063 |
| `vllm::reshape_and_cache_flash_kernel` | attention | 36 | 0.063 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 0.052 |
| `at::native::cunn_SoftMaxForward` | sampling | 1 | 0.050 |
| `flashinfer::sampling::TopPSamplingFromProbKernel` | sampling | 1 | 0.032 |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | sampling | 1 | 0.024 |

## c64_in128: 128 requests, 128 in / 64 out, 64 at a time

Client: TPOT p50 24.54 ms, TTFT p50 587.9 ms, 128 completed. 75515 kernels recorded.

**Decode step** (cut at `flashinfer::sampling::TopPSamplingFromProbKernel`; median of 127 of 137 steps): 21.49 ms, of which kernels 21.40 ms (99.6%).

| part | ms per step | % of step |
|---|---|---|
| linear | 18.285 | 85.1% |
| attention | 2.078 | 9.7% |
| other | 0.561 | 2.6% |
| sampling | 0.196 | 0.9% |
| norm | 0.158 | 0.7% |
| activation | 0.084 | 0.4% |
| GPU idle | 0.081 | 0.4% |

| kernel | part | runs per step | ms per step |
|---|---|---|---|
| `ampere_bf16_s1688gemm_bf16_64x64_sliced1x4_ldg8_f2f_tn` | linear | 36 | 8.785 |
| `ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_tn` | linear | 36 | 4.390 |
| `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8_f2f_stages_64x5_tn` | linear | 72 | 3.674 |
| `flash::flash_fwd_splitkv_kernel` | attention | 36 | 2.014 |
| `cutlass::Kernel2` | linear | 1 | 1.439 |
| `cublasLt::splitKreduce_kernel` | other | 72 | 0.163 |
| `at::native::elementwise_kernel` | other | 72 | 0.143 |
| `triton_red_fused_3` | other | 35 | 0.116 |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | sampling | 1 | 0.089 |
| `triton_poi_fused_mul_silu_slice_1` | activation | 36 | 0.084 |
| `triton_red_fused_fused_add_rms_norm_2` | norm | 36 | 0.081 |
| `triton_red_fused_fused_add_rms_norm_0` | norm | 36 | 0.074 |
