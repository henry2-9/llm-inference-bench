# Results

Each folder holds the raw output of one experiment plus a `summary.md` or `README.md` with the tables.
Every run folder has an `env.txt` (image digest, model snapshot, GPU, settings) and the server log.

| folder | what it is | made by |
|---|---|---|
| `speed/<engine>_<precision>` | Throughput and latency grid: 2 workloads x 4 concurrencies x 3 runs | `bench_vllm.sh`, `bench_trtllm.sh` |
| `speed/trtllm_nvfp4_cpuload` | The NVFP4 grid again while another job loaded the CPU (effect on time to first token) | `bench_trtllm.sh` |
| `gsm8k/` | GSM8K accuracy for every engine and precision; `*_rerun` is the same setup a second time | `quality_gsm8k.sh` |
| `profile/` | Per-kernel GPU time of a decode step and a prefill, three captures per run | `profile_all.sh` |
| `profile_sweep/` | The same with more batch sizes and context lengths | `profile_all.sh` with `CAPTURES` |
| `profile_force_xqa/` | TensorRT-LLM FP8 with `TRTLLM_FORCE_XQA=1` | `profile.sh` with `SERVER_ENV` |
| `profile_patched/`, `gsm8k_patched/` | TensorRT-LLM 1.4.0rc2 rebuilt so that the FP8 KV cache may use multi-block attention on SM89 | `profile.sh`, `bench_trtllm.sh` with `SERVER_DOCKER_ARGS` |
| `profile_rc4/`, `gsm8k_rc4/` | Unmodified TensorRT-LLM 1.4.0rc4 | the same scripts with `TRT_IMG` |
| `profile_rc4_patched/`, `gsm8k_rc4_patched/` | 1.4.0rc4 with the same rebuild | the same scripts with `TRT_IMG` and `SERVER_DOCKER_ARGS` |

Engine names: `vllm`, `trtllm`. Precisions: `bf16`, `fp8`, `nvfp4`.
Nsight Systems traces (`.nsys-rep`, `.sqlite`) are not in the repository; `profile.sh` regenerates them.
