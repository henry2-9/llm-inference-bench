#!/usr/bin/env bash
# Usage: scripts/profile.sh vllm   <tag> <hf_model_id> [extra vllm serve args...]
#        scripts/profile.sh trtllm <tag> <hf_model_id> <config.yaml>
# Starts the server with the same settings as bench_vllm.sh / bench_trtllm.sh, but under Nsight Systems
# (`nsys launch`, CUDA trace, CUDA graphs traced node by node), and records only while the client sends
# requests (`nsys start` / `nsys stop`). Each capture is first run once unrecorded.
# Saves <capture>.nsys-rep / .sqlite / .json under results/<tag>/; profile_breakdown.py writes breakdown.md.
# CAPTURES="name:input:output:prompts:concurrency ..." overrides the default captures.
# TRT_IMG=<image> selects another TensorRT-LLM release image (default 1.4.0rc2).
# SERVER_ENV="NAME=value ..." sets environment variables in the server container (e.g. TRTLLM_FORCE_XQA=1).
# SERVER_DOCKER_ARGS="..." adds `docker run` arguments for the server (e.g. -v mounts that replace the engine's
# libraries with ones built from source); they are recorded in env.txt.
#
# Both engines are traced with the nsys that ships in the TRT-LLM image. The vLLM image has none, so that
# copy (about 600 MB) is extracted once to $NSYS_DIR and mounted into the vLLM container.
set -euo pipefail
ENGINE=$1; TAG=$2; MODEL=$3; shift 3
TRT_IMG=${TRT_IMG:-nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc2}
VLLM_IMG=vllm/vllm-openai:v0.31.0
NSYS_IN_TRT=/usr/local/cuda-13.4/NsightSystems-cli-2026.5.1
NSYS_DIR=${NSYS_DIR:-$HOME/.cache/llm-inference-bench/$(basename $NSYS_IN_TRT)}
# Batch-1 decode at a short and a long context (the long one also gives the 2048-token prefill), and batch-4 decode.
CAPTURES=${CAPTURES:-c1_in128:128:64:4:1 c1_in2048:2048:64:2:1 c4_in128:128:64:8:4}
NSYS_ARGS=(--session-new=bench --trace=cuda --cuda-graph-trace=node)
env_args=(); for kv in ${SERVER_ENV:-}; do env_args+=(-e "$kv"); done
read -r -a docker_args <<< "${SERVER_DOCKER_ARGS:-}"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/results/$TAG
HF=~/.cache/huggingface
SNAP=$(ls -d "$HF/hub/models--${MODEL//\//--}"/snapshots/* | head -1)
SNAP_DIR=hub/models--${MODEL//\//--}/snapshots/$(basename "$SNAP")

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
if [ "$used" -gt 1000 ] || [ "$apps" -gt 0 ]; then
  echo "GPU busy (${used} MiB used, ${apps} processes); aborting"; exit 1
fi

mkdir -p "$OUT"
case $ENGINE in
  vllm)
    IMG=$VLLM_IMG; NAME=bench_vllm_server; NSYS=/opt/nsys/target-linux-x64/nsys; EXTRA=("$@")
    if [ ! -x "$NSYS_DIR/target-linux-x64/nsys" ]; then
      mkdir -p "$(dirname "$NSYS_DIR")"
      docker rm -f nsys_src >/dev/null 2>&1 || true
      docker create --name nsys_src $TRT_IMG >/dev/null
      docker cp nsys_src:$NSYS_IN_TRT "$NSYS_DIR"
      docker rm nsys_src >/dev/null
    fi
    info="extra=${EXTRA[*]:-}"
    server=(-v "$HF":/root/.cache/huggingface:ro -v "$NSYS_DIR":/opt/nsys:ro --entrypoint $NSYS $IMG
            launch "${NSYS_ARGS[@]}" vllm serve --model "$MODEL" --max-model-len 4096
            --gpu-memory-utilization 0.85 --no-enable-prefix-caching --seed 0 "${EXTRA[@]}") ;;
  trtllm)
    IMG=$TRT_IMG; NAME=bench_trtllm_server; NSYS=nsys; CFG=$(realpath "$1")
    cp "$CFG" "$OUT/server_config.yaml"
    info="config=$(basename "$CFG")"
    server=(--ulimit memlock=-1 --ulimit stack=67108864 -v "$HF/hub":/models/hub:ro -v "$CFG":/cfg.yaml:ro
            --entrypoint $NSYS $IMG launch "${NSYS_ARGS[@]}" trtllm-serve serve "/models/$SNAP_DIR"
            --host 0.0.0.0 --port 8000 --max_seq_len 4096 --config /cfg.yaml) ;;
  *) echo "unknown engine: $ENGINE"; exit 1 ;;
esac

cleanup() {
  # The server container runs as root, so the files nsys wrote to /out are root-owned until this.
  docker exec $NAME chown -R "$(id -u):$(id -g)" /out >/dev/null 2>&1 || true
  docker logs $NAME > "$OUT/server.log" 2>&1 || true
  docker rm -f $NAME >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --name $NAME --gpus all --ipc=host -p 127.0.0.1:8000:8000 -e HF_HUB_OFFLINE=1 "${env_args[@]}" \
  "${docker_args[@]}" -v "$OUT":/out "${server[@]}" >/dev/null

for _ in $(seq 1 180); do
  sleep 5
  [ "$(docker inspect -f '{{.State.Running}}' $NAME)" = true ] || { echo "server exited"; exit 1; }
  curl -sf 127.0.0.1:8000/health >/dev/null && break
done
curl -sf 127.0.0.1:8000/health >/dev/null || { echo "server not ready after 900s"; exit 1; }
echo "$(date +%T) server ready"

{
  date -Iseconds
  echo "engine=$ENGINE model=$MODEL snapshot=$(basename "$SNAP") $info server_env=${SERVER_ENV:-}"
  [ -z "${SERVER_DOCKER_ARGS:-}" ] || echo "server_docker_args=$SERVER_DOCKER_ARGS"
  echo "image=$(docker image inspect -f '{{index .RepoDigests 0}}' $IMG)"
  echo "nsys=$(docker exec $NAME $NSYS --version) args=${NSYS_ARGS[*]}"
  echo "repo_commit=$(git -C "$ROOT" describe --always --dirty)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader
} > "$OUT/env.txt"

# Same client and sampling settings as client_loop.sh.
client() {  # client <file name> <input len> <output len> <prompts> <concurrency>
  docker run --rm --network host --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_OFFLINE=1 \
    -v "$HF":/hf:ro -v "$OUT":/out --entrypoint vllm $VLLM_IMG bench serve \
    --backend vllm --base-url http://127.0.0.1:8000 --model "$MODEL" \
    --tokenizer "/hf/$SNAP_DIR" --dataset-name random --random-range-ratio 0 --ignore-eos \
    --temperature 0.6 --top-p 0.95 --top-k 20 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --random-input-len "$2" --random-output-len "$3" --num-prompts "$4" --max-concurrency "$5" --seed 1001 \
    --save-result --result-dir /out --result-filename "$1.json" > "$OUT/$1.client.log" 2>&1
}

: > "$OUT/captures.tsv"
for cap in $CAPTURES; do
  IFS=: read -r name in out prompts conc <<< "$cap"
  client "$name.unrecorded" "$in" "$out" "$conc" "$conc"
  docker exec $NAME $NSYS start --session=bench -o "/out/$name" --force-overwrite=true --sample=none --cpuctxsw=none >/dev/null
  client "$name" "$in" "$out" "$prompts" "$conc"
  docker exec $NAME $NSYS stop --session=bench > "$OUT/$name.nsys.log" 2>&1
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$in" "$out" "$prompts" "$conc" >> "$OUT/captures.tsv"
  echo "$(date +%T) captured $name"
done
for cap in $CAPTURES; do
  name=${cap%%:*}
  # --force-export: a .sqlite left by an earlier run into the same folder would otherwise be reused (and nsys fails).
  docker exec $NAME $NSYS stats --force-export=true --force-overwrite=true --report cuda_gpu_kern_sum --format csv \
    -o "/out/$name" "/out/$name.nsys-rep" >> "$OUT/$name.nsys.log" 2>&1 \
    || { echo "nsys stats failed for $name:"; tail -5 "$OUT/$name.nsys.log"; exit 1; }
done
rm -f "$OUT"/*.unrecorded.json "$OUT"/*.unrecorded.client.log

cleanup; trap - EXIT
python3 -I "$ROOT/scripts/profile_breakdown.py" "$OUT" > /dev/null
echo "$(date +%T) wrote $OUT/breakdown.md"
