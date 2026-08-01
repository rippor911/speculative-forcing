#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <physical_gpu_id> <worker_index_0_to_3>" >&2
  exit 2
fi

GPU_ID="$1"
WORKER_INDEX="$2"

ROOT="$(pwd)"
PY="${PY:-/home/dataset-assist-0/luojy/efficiency/rippor/envs/speculative-forcing/bin/python}"
FORMAL="${FORMAL:-experiments/E0208C_teacher_rollout_formal}"
GEN="$FORMAL/generate_teacher_shard.py"
WORKER_LOG_DIR="$FORMAL/worker_logs"

mkdir -p "$WORKER_LOG_DIR" "$FORMAL/shards"

if [[ ! -f "$GEN" ]]; then
  echo "missing generator: $GEN" >&2
  exit 3
fi

for SHARD_NUM in $(seq "$WORKER_INDEX" 4 35); do
  SHARD=$(printf '%03d' "$SHARD_NUM")
  PLAN="$FORMAL/plans/shard_${SHARD}.jsonl"
  OUTPUT="$FORMAL/shards/shard_${SHARD}"
  LOG="$WORKER_LOG_DIR/shard_${SHARD}.log"
  STATUS_FILE="$WORKER_LOG_DIR/shard_${SHARD}.status"

  if [[ ! -f "$PLAN" ]]; then
    echo "missing plan: $PLAN" >&2
    exit 4
  fi

  echo "[$(date -Is)] worker=$WORKER_INDEX gpu=$GPU_ID shard=$SHARD START"

  set +e
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
    "$PY" "$GEN" \
      --plan "$PLAN" \
      --output-dir "$OUTPUT" \
      --device cuda:0 \
      --resume \
      > >(tee -a "$LOG") 2>&1
  CODE=$?
  set -e

  printf '%s\n' "$CODE" > "$STATUS_FILE"

  if [[ "$CODE" -ne 0 ]]; then
    echo "[$(date -Is)] worker=$WORKER_INDEX gpu=$GPU_ID shard=$SHARD FAIL code=$CODE" >&2
    exit "$CODE"
  fi

  echo "[$(date -Is)] worker=$WORKER_INDEX gpu=$GPU_ID shard=$SHARD PASS"
done

echo "[$(date -Is)] worker=$WORKER_INDEX gpu=$GPU_ID ALL_PASS"
