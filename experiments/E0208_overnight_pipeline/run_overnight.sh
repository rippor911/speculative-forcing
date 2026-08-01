#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(pwd)"
export PYTHONPATH="$ROOT:$ROOT/experiments/E0202_anchor_replay_audit:$ROOT/experiments/E0203_mcp_short_training:$ROOT/experiments/E0205B_mcp_quality_gate:$ROOT/experiments/E0206B_depth1_single_state_overfit${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-/home/dataset-assist-0/luojy/efficiency/rippor/envs/speculative-forcing/bin/python}"

PIPELINE_DIR="${PIPELINE_DIR:-experiments/E0208_overnight_pipeline}"
FORMAL="${FORMAL:-experiments/E0208C_teacher_rollout_formal}"
TRAIN_DIR="${TRAIN_DIR:-experiments/E0209_depth1_formal_training}"
RUN_LOG_DIR="$PIPELINE_DIR/logs"
STATUS_JSON="$PIPELINE_DIR/overnight_status.json"
LOCK_FILE="$PIPELINE_DIR/overnight.lock"

mkdir -p "$RUN_LOG_DIR"
exec > >(tee -a "$RUN_LOG_DIR/overnight.log") 2>&1

exec 9>"$LOCK_FILE"
if command -v flock >/dev/null 2>&1; then
  flock -n 9 || {
    echo "another overnight pipeline is already running" >&2
    exit 90
  }
fi

write_status() {
  local status="$1"
  local stage="$2"
  local message="$3"
  "$PY" - "$STATUS_JSON" "$status" "$stage" "$message" <<'PY'
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "stage": sys.argv[3],
    "message": sys.argv[4],
    "updated_unix": time.time(),
}
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
tmp.replace(path)
PY
}

on_error() {
  local code=$?
  write_status "FAIL" "${CURRENT_STAGE:-unknown}" "pipeline exited with code $code"
  echo "[$(date -Is)] OVERNIGHT_PIPELINE=FAIL stage=${CURRENT_STAGE:-unknown} code=$code" >&2
  exit "$code"
}
trap on_error ERR

CURRENT_STAGE="preflight"
write_status "RUNNING" "$CURRENT_STAGE" "checking generator plans and training wrapper"

echo "[$(date -Is)] PRECHECK START"

if [[ "$ROOT" != "/home/dataset-assist-0/luojy/efficiency/rippor/speculative-forcing" ]]; then
  echo "run from the repository root; current directory is $ROOT" >&2
  exit 2
fi

for required in \
  "$FORMAL/generate_teacher_shard.py" \
  "$FORMAL/plans/dataset_plan.json" \
  "$PIPELINE_DIR/run_generation_worker.sh" \
  "$PIPELINE_DIR/merge_teacher_dataset.py" \
  "$PIPELINE_DIR/run_depth1_formal.py"
do
  [[ -f "$required" ]] || {
    echo "missing required file: $required" >&2
    exit 3
  }
done

PLAN_COUNT=$(find "$FORMAL/plans" -maxdepth 1 -type f -name 'shard_*.jsonl' | wc -l)
PLAN_LINES=$(cat "$FORMAL"/plans/shard_*.jsonl | wc -l)
[[ "$PLAN_COUNT" -eq 36 ]] || {
  echo "expected 36 shard plans, got $PLAN_COUNT" >&2
  exit 4
}
[[ "$PLAN_LINES" -eq 2304 ]] || {
  echo "expected 2304 plan lines, got $PLAN_LINES" >&2
  exit 5
}

"$PY" -m py_compile \
  "$FORMAL/generate_teacher_shard.py" \
  "$PIPELINE_DIR/merge_teacher_dataset.py" \
  "$PIPELINE_DIR/run_depth1_formal.py"

"$PY" "$PIPELINE_DIR/run_depth1_formal.py" \
  --epochs 3 \
  --schedule-seed 20260731 \
  --preflight-only \
  | tee "$RUN_LOG_DIR/training_preflight.log"

AVAILABLE_GIB=$("$PY" - <<'PY'
import shutil
print(shutil.disk_usage('.').free // 1024**3)
PY
)
echo "available_disk_gib=$AVAILABLE_GIB"
[[ "$AVAILABLE_GIB" -ge 80 ]] || {
  echo "need at least 80 GiB free before generation" >&2
  exit 6
}

echo "[$(date -Is)] PRECHECK PASS"

CURRENT_STAGE="teacher_generation"
write_status "RUNNING" "$CURRENT_STAGE" "generating 2304 teacher samples on four GPUs"

echo "[$(date -Is)] GENERATION START"

PIDS=()
for WORKER in 0 1 2 3; do
  (
    bash "$PIPELINE_DIR/run_generation_worker.sh" "$WORKER" "$WORKER"
  ) > >(tee -a "$RUN_LOG_DIR/worker_${WORKER}.log") 2>&1 &
  PIDS+=("$!")
done

GENERATION_STATUS=0
for PID in "${PIDS[@]}"; do
  wait "$PID" || GENERATION_STATUS=1
done

[[ "$GENERATION_STATUS" -eq 0 ]] || {
  echo "one or more generation workers failed" >&2
  exit 10
}

echo "[$(date -Is)] GENERATION WORKERS PASS"

CURRENT_STAGE="teacher_merge"
write_status "RUNNING" "$CURRENT_STAGE" "validating and merging 36 shard manifests"

"$PY" "$PIPELINE_DIR/merge_teacher_dataset.py" \
  --formal-dir "$FORMAL" \
  --output "$FORMAL/manifest.json" \
  | tee "$RUN_LOG_DIR/merge.log"

CURRENT_STAGE="depth1_training"
write_status "RUNNING" "$CURRENT_STAGE" "training MCP depth-1 for three epochs"

echo "[$(date -Is)] TRAINING START"

if [[ -d "$TRAIN_DIR" ]] && find "$TRAIN_DIR" -mindepth 1 -print -quit | grep -q .; then
  STAMP=$(date +%Y%m%d_%H%M%S)
  mv "$TRAIN_DIR" "${TRAIN_DIR}.old_${STAMP}"
fi
mkdir -p "$TRAIN_DIR"

set +e
CUDA_VISIBLE_DEVICES=0 \
  "$PY" "$PIPELINE_DIR/run_depth1_formal.py" \
    --epochs 3 \
    --schedule-seed 20260731 \
    2>&1 | tee "$RUN_LOG_DIR/depth1_training.log"
TRAIN_CODE=${PIPESTATUS[0]}
set -e

[[ "$TRAIN_CODE" -eq 0 ]] || {
  echo "depth-1 training failed with code $TRAIN_CODE" >&2
  exit "$TRAIN_CODE"
}

CURRENT_STAGE="complete"
write_status "PASS" "$CURRENT_STAGE" "teacher generation, merge, and depth-1 training completed"

echo "[$(date -Is)] OVERNIGHT_PIPELINE=PASS"
echo "teacher_manifest=$FORMAL/manifest.json"
echo "training_dir=$TRAIN_DIR"
echo "training_status=$TRAIN_DIR/wrapper_status.json"
