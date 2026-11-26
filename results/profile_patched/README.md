# profile_patched / gsm8k_patched: TRT-LLM 1.4.0rc2 with multi-block attention allowed for FP8 KV on SM89

Same server settings as `profile`, `profile_sweep` and `gsm8k`, but the engine libraries in the release container are
replaced (bind mounts, listed as `server_docker_args` in each `env.txt`) by ones built from source at tag
`v1.4.0rc2` with one line changed in `cpp/tensorrt_llm/common/attentionOp.cpp`:

```diff
-        || (xqaParams.kv_cache_data_type == DATA_TYPE_E4M3 && (mSM < kSM_90 || mSM > kSM_120)))
+        || (xqaParams.kv_cache_data_type == DATA_TYPE_E4M3 && mSM > kSM_120))
```

Build (in `nvcr.io/nvidia/tensorrt-llm/devel:1.4.0rc2`, after `git lfs pull`; 1 h 29 min with 10 threads):

```
python3 scripts/build_wheel.py --build_root /build --use_ccache -a "89" --skip_building_wheel --linking_install_binary -j 10
```

Replaced files: `libs/libtensorrt_llm.so`, `libs/libth_common.so`, `libs/libdecoder_attention_{0,1}.so` and
`bindings.cpython-312-x86_64-linux-gnu.so`. The build targets SM89 only; everything else in the container is unchanged.

Decode step, ms (median; one capture per cell). "forced" is the unmodified engine with `TRTLLM_FORCE_XQA=1` (`profile_force_xqa`).

| checkpoint | batch, context | unmodified | forced | patched |
|---|---|---|---|---|
| FP8 | 1, 128 | 11.75 | 10.06 | 10.24 |
| FP8 | 1, 2048 | 11.78 | 10.90 | 10.32 |
| FP8 | 1, 3900 | 12.01 | 11.72 | 10.51 |
| FP8 | 2, 128 | 13.49 | 10.16 | 10.35 |
| FP8 | 4, 128 | 10.28 | 10.29 | 10.54 |
| FP8 | 4, 2048 | 11.12 | - | 11.11 |
| NVFP4 | 1, 128 | 7.28 | - | 7.44 |
| NVFP4 | 1, 2048 | 8.15 | - | 7.59 |
| NVFP4 | 1, 3900 | 8.98 | - | 7.75 |
| NVFP4 | 2, 128 | 7.36 | - | 7.51 |
| NVFP4 | 4, 128 | 7.41 | - | 7.67 |
| NVFP4 | 4, 2048 | 8.30 | - | 8.31 |

With the FP8 checkpoint, batch 1 and 2 no longer fall back to `masked_multihead_attention_kernel`. Long contexts get
faster; at context 128 `kernel_mha` takes 0.50-0.62 ms per step instead of 0.39.

Why short contexts get slower: the launch grid of `kernel_mha` in the traces (`gridX` = blocks per sequence,
`gridY` = 8 KV heads, `gridZ` = batch). Inside the decode CUDA graph it is always 16 blocks, which is what
`computeMultiBlockCount` gives for a history of 4,096 tokens (the server's `max_seq_len`); the few steps that
run outside the graph use the real length (1 block at context 128, 11 at 2048).

| batch, context | unmodified: grid, µs per call | patched: grid, µs per call |
|---|---|---|
| 1, 128 | 1x8x1, 10.4 (forced XQA) / 11.0 (NVFP4) | 16x8x1, 13.6 / 13.9 |
| 1, 2048 | 1x8x1, 33.7 / 35.1 | 16x8x1, 15.9 / 17.8 |
| 1, 3900 | 1x8x1, 56.5 / 57.9 | 16x8x1, 21.0 / 22.2 |
| 4, 128 | 1x8x4, 11.0 / 11.4 | 16x8x4, 16.7 / 17.2 |
| 4, 2048 | 1x8x4, - / 36.3 | 16x8x4, 31.9 / 34.6 |

(FP8 checkpoint / NVFP4 checkpoint; mean over the recorded calls.)

GSM8K (`gsm8k_patched`, strict-match, 1,319 questions): FP8 86.6% -> 87.1% (29 questions right only before,
36 only after), NVFP4 86.3% -> 86.7% (15 / 21). The counts are similar to those from two runs of the unmodified
engine (`gsm8k/trtllm_{fp8,nvfp4}_rerun`: FP8 25 / 35, NVFP4 16 / 19).
