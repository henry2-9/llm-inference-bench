# gsm8k_rc4: GSM8K (lm-eval, 5-shot, greedy, /v1/completions)

Accuracy ± lm-eval's standard error.

| run | questions | strict-match | flexible-extract |
|---|---|---|---|
| trtllm_fp8 | 1319 | 86.7% ± 0.9 | 87.3% ± 0.9 |
| trtllm_fp8_rerun | 1319 | 87.1% ± 0.9 | 87.6% ± 0.9 |
| trtllm_nvfp4 | 1319 | 85.8% ± 1.0 | 86.1% ± 1.0 |
| trtllm_nvfp4_rerun | 1319 | 86.6% ± 0.9 | 86.8% ± 0.9 |

Question by question (strict-match): "only A" = right in the first run only, p = McNemar exact test (small p: the difference is unlikely to be run-to-run luck), same text = identical generated output. A *_rerun row is the same setup run twice.

| pair (A vs B) | both right | only A | only B | both wrong | p | same text |
|---|---|---|---|---|---|---|
| trtllm_fp8 vs trtllm_fp8_rerun | 1120 | 24 | 29 | 146 | 0.58 | 44% |
| trtllm_nvfp4 vs trtllm_nvfp4_rerun | 1118 | 14 | 24 | 163 | 0.14 | 65% |
