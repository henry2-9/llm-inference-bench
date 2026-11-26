# gsm8k: GSM8K (lm-eval, 5-shot, greedy, /v1/completions)

Accuracy ± lm-eval's standard error.

| run | questions | strict-match | flexible-extract |
|---|---|---|---|
| vllm_bf16 | 1319 | 87.0% ± 0.9 | 87.5% ± 0.9 |
| vllm_bf16_rerun | 1319 | 87.3% ± 0.9 | 87.7% ± 0.9 |
| vllm_fp8 | 1319 | 86.4% ± 0.9 | 86.6% ± 0.9 |
| vllm_fp8_rerun | 1319 | 87.0% ± 0.9 | 87.4% ± 0.9 |
| vllm_nvfp4 | 1319 | 85.9% ± 1.0 | 85.7% ± 1.0 |
| vllm_nvfp4_rerun | 1319 | 85.6% ± 1.0 | 85.2% ± 1.0 |
| trtllm_bf16 | 1319 | 87.3% ± 0.9 | 87.7% ± 0.9 |
| trtllm_bf16_rerun | 1319 | 87.7% ± 0.9 | 88.4% ± 0.9 |
| trtllm_fp8 | 1319 | 86.6% ± 0.9 | 87.0% ± 0.9 |
| trtllm_fp8_rerun | 1319 | 87.3% ± 0.9 | 87.6% ± 0.9 |
| trtllm_nvfp4 | 1319 | 86.3% ± 0.9 | 86.7% ± 0.9 |
| trtllm_nvfp4_rerun | 1319 | 86.5% ± 0.9 | 87.1% ± 0.9 |

Question by question (strict-match): "only A" = right in the first run only, p = McNemar exact test (small p: the difference is unlikely to be run-to-run luck), same text = identical generated output. A *_rerun row is the same setup run twice.

| pair (A vs B) | both right | only A | only B | both wrong | p | same text |
|---|---|---|---|---|---|---|
| vllm_bf16 vs vllm_bf16_rerun | 1141 | 7 | 11 | 160 | 0.48 | 71% |
| vllm_fp8 vs vllm_fp8_rerun | 1117 | 23 | 30 | 149 | 0.41 | 41% |
| vllm_nvfp4 vs vllm_nvfp4_rerun | 1105 | 28 | 24 | 162 | 0.68 | 48% |
| trtllm_bf16 vs trtllm_bf16_rerun | 1149 | 3 | 8 | 159 | 0.23 | 74% |
| trtllm_fp8 vs trtllm_fp8_rerun | 1117 | 25 | 35 | 142 | 0.25 | 44% |
| trtllm_nvfp4 vs trtllm_nvfp4_rerun | 1122 | 16 | 19 | 162 | 0.74 | 64% |
| vllm_bf16 vs vllm_fp8 | 1100 | 48 | 40 | 131 | 0.46 | 24% |
| vllm_bf16 vs vllm_nvfp4 | 1078 | 70 | 55 | 116 | 0.21 | 10% |
| trtllm_bf16 vs trtllm_fp8 | 1096 | 56 | 46 | 121 | 0.37 | 22% |
| trtllm_bf16 vs trtllm_nvfp4 | 1083 | 69 | 55 | 112 | 0.24 | 9% |
| vllm_bf16 vs trtllm_bf16 | 1140 | 8 | 12 | 159 | 0.50 | 70% |
| vllm_fp8 vs trtllm_fp8 | 1097 | 43 | 45 | 134 | 0.92 | 26% |
| vllm_nvfp4 vs trtllm_nvfp4 | 1104 | 29 | 34 | 152 | 0.61 | 39% |
