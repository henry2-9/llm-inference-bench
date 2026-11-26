#!/usr/bin/env bash
# Usage: scripts/bench_trtllm.sh <tag> <hf_model_id> <config.yaml> [extra trtllm-serve args...]
# Same flow as bench_vllm.sh, but the server is trtllm-serve. The client is still `vllm bench serve`
# from the vLLM container (client_loop.sh) on the same port, so both engines get identical requests.
# CLIENT=gsm8k.sh (with CLIENT_IMG) runs the quality check on the same server instead; see quality_gsm8k.sh.
# TRT_IMG=<image> selects another TensorRT-LLM release image (default 1.4.0rc2).
# SERVER_DOCKER_ARGS="..." adds `docker run` arguments for the server (e.g. -v mounts that replace the engine's
# libraries with ones built from source); they are recorded in env.txt.
set -euo pipefail
TAG=$1; MODEL=$2; CFG=$(realpath "$3"); shift 3; EXTRA=("$@")
IMG=${TRT_IMG:-nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc2}
CLIENT=${CLIENT:-client_loop.sh}
CLIENT_IMG=${CLIENT_IMG:-vllm/vllm-openai:v0.31.0}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/results/$TAG
NAME=bench_trtllm_server
RUNS=${RUNS:-3}
HF=~/.cache/huggingface
SNAP=$(ls -d "$HF/hub/models--${MODEL//\//--}"/snapshots/* | head -1)
SNAP_DIR=hub/models--${MODEL//\//--}/snapshots/$(basename "$SNAP")
read -r -a docker_args <<< "${SERVER_DOCKER_ARGS:-}"

# Refuse to start unless the GPU is idle: numbers from a shared GPU are not usable.
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
if [ "$used" -gt 1000 ] || [ "$apps" -gt 0 ]; then
  echo "GPU busy (${used} MiB used, ${apps} processes); aborting"; exit 1
fi

mkdir -p "$OUT/raw"
: > "$OUT/cpu_load.txt"
cp "$CFG" "$OUT/server_config.yaml"
{
  date -Iseconds
  echo "model=$MODEL snapshot=$(basename "$SNAP") config=$(basename "$CFG") extra=${EXTRA[*]:-}"
  echo "image=$(docker image inspect -f '{{index .RepoDigests 0}}' $IMG)"
  echo "client=$CLIENT client_image=$(docker image inspect -f '{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}' $CLIENT_IMG)"
  echo "repo_commit=$(git -C "$ROOT" describe --always --dirty)"
  [ -z "${SERVER_DOCKER_ARGS:-}" ] || echo "server_docker_args=$SERVER_DOCKER_ARGS"
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader
} > "$OUT/env.txt"

cleanup() {
  kill "${MON:-}" "${APPS:-}" 2>/dev/null || true
  docker logs $NAME > "$OUT/server.log" 2>&1 || true
  docker rm -f $NAME >/dev/null 2>&1 || true
}
trap cleanup EXIT

# The HF cache can't be mounted read-only as a whole (TRT-LLM writes ~/.cache/huggingface/modules),
# so only hub/ is mounted, at /models.
docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --name $NAME --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -p 127.0.0.1:8000:8000 -e HF_HUB_OFFLINE=1 "${docker_args[@]}" \
  -v "$HF/hub":/models/hub:ro -v "$CFG":/cfg.yaml:ro --entrypoint trtllm-serve \
  $IMG serve "/models/$SNAP_DIR" --host 0.0.0.0 --port 8000 --max_seq_len 4096 \
  --config /cfg.yaml "${EXTRA[@]}" >/dev/null

for _ in $(seq 1 180); do
  sleep 5
  [ "$(docker inspect -f '{{.State.Running}}' $NAME)" = true ] || { echo "server exited"; exit 1; }
  curl -sf 127.0.0.1:8000/health >/dev/null && break
done
curl -sf 127.0.0.1:8000/health >/dev/null || { echo "server not ready after 900s"; exit 1; }
echo "$(date +%T) server ready"

# GPU clocks/power/memory every second, and how many processes use the GPU every 5 s
# (more than one means something else ran during the benchmark).
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu \
  --format=csv -l 1 > "$OUT/gpu_monitor.csv" &
MON=$!
( while sleep 5; do
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true
    # 1-minute CPU load: other CPU jobs raise time to first token for short prompts (not throughput or TPOT).
    cut -d' ' -f1 /proc/loadavg >> "$OUT/cpu_load.txt"
  done ) > "$OUT/compute_apps.txt" &
APPS=$!

docker run --rm --network host --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_HUB_OFFLINE=1 \
  -v "$HF":/hf:ro -v "$OUT":/out -v "$ROOT/scripts":/scripts:ro \
  -e MODEL="$MODEL" -e TOKENIZER="/hf/$SNAP_DIR" -e RUNS="$RUNS" \
  -e WORKLOADS="${WORKLOADS:-}" -e CONCURRENCIES="${CONCURRENCIES:-}" -e LIMIT="${LIMIT:-}" \
  --entrypoint bash $CLIENT_IMG "/scripts/$CLIENT"

# Save the server log and stop the server before summarizing (summary reads the KV cache lines from it).
cleanup; trap - EXIT
if [ "$CLIENT" = client_loop.sh ]; then python3 -I "$ROOT/scripts/summarize.py" "$OUT"; fi
