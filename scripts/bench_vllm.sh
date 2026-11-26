#!/usr/bin/env bash
# Usage: scripts/bench_vllm.sh <tag> <hf_model_id> [extra vllm serve args...]
# Starts vLLM from the official container, runs the speed grid with `vllm bench serve`
# (2 workloads x 4 concurrencies x RUNS runs), and saves everything under results/<tag>/.
# CLIENT=gsm8k.sh (with CLIENT_IMG) runs the quality check on the same server instead; see quality_gsm8k.sh.
set -euo pipefail
TAG=$1; MODEL=$2; shift 2; EXTRA=("$@")
IMG=vllm/vllm-openai:v0.31.0
CLIENT=${CLIENT:-client_loop.sh}
CLIENT_IMG=${CLIENT_IMG:-$IMG}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/results/$TAG
NAME=bench_vllm_server
RUNS=${RUNS:-3}
HF=~/.cache/huggingface
SNAP=$(ls -d "$HF/hub/models--${MODEL//\//--}"/snapshots/* | head -1)

# Refuse to start unless the GPU is idle: numbers from a shared GPU are not usable.
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
if [ "$used" -gt 1000 ] || [ "$apps" -gt 0 ]; then
  echo "GPU busy (${used} MiB used, ${apps} processes); aborting"; exit 1
fi

mkdir -p "$OUT/raw"
: > "$OUT/cpu_load.txt"
{
  date -Iseconds
  echo "model=$MODEL snapshot=$(basename "$SNAP") extra=${EXTRA[*]:-}"
  echo "image=$(docker image inspect -f '{{index .RepoDigests 0}}' $IMG)"
  echo "client=$CLIENT client_image=$(docker image inspect -f '{{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}{{.Id}}{{end}}' $CLIENT_IMG)"
  echo "repo_commit=$(git -C "$ROOT" describe --always --dirty)"
  nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv,noheader
} > "$OUT/env.txt"

cleanup() {
  kill "${MON:-}" "${APPS:-}" 2>/dev/null || true
  docker logs $NAME > "$OUT/server.log" 2>&1 || true
  docker rm -f $NAME >/dev/null 2>&1 || true
}
trap cleanup EXIT

docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d --name $NAME --gpus all --ipc=host -p 127.0.0.1:8000:8000 -e HF_HUB_OFFLINE=1 \
  -v "$HF":/root/.cache/huggingface:ro \
  $IMG --model "$MODEL" --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --no-enable-prefix-caching --seed 0 "${EXTRA[@]}" >/dev/null

for _ in $(seq 1 120); do
  sleep 5
  [ "$(docker inspect -f '{{.State.Running}}' $NAME)" = true ] || { echo "server exited"; exit 1; }
  curl -sf 127.0.0.1:8000/health >/dev/null && break
done
curl -sf 127.0.0.1:8000/health >/dev/null || { echo "server not ready after 600s"; exit 1; }
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
  -e MODEL="$MODEL" -e TOKENIZER="/hf/hub/models--${MODEL//\//--}/snapshots/$(basename "$SNAP")" -e RUNS="$RUNS" \
  -e WORKLOADS="${WORKLOADS:-}" -e CONCURRENCIES="${CONCURRENCIES:-}" -e LIMIT="${LIMIT:-}" \
  --entrypoint bash $CLIENT_IMG "/scripts/$CLIENT"

# Save the server log and stop the server before summarizing (summary reads the KV cache lines from it).
cleanup; trap - EXIT
if [ "$CLIENT" = client_loop.sh ]; then python3 -I "$ROOT/scripts/summarize.py" "$OUT"; fi
