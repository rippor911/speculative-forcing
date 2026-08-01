#!/usr/bin/env bash
set -Eeuo pipefail

PIPELINE_DIR="${PIPELINE_DIR:-experiments/E0208_overnight_pipeline}"
FORMAL="${FORMAL:-experiments/E0208C_teacher_rollout_formal}"
TRAIN_DIR="${TRAIN_DIR:-experiments/E0209_depth1_formal_training}"

echo "===== OVERNIGHT STATUS ====="
cat "$PIPELINE_DIR/overnight_status.json" 2>/dev/null || true

echo
echo "===== TEACHER MERGE ====="
cat "$FORMAL/merge_audit.json" 2>/dev/null || true

echo
echo "===== TRAINING WRAPPER ====="
cat "$TRAIN_DIR/wrapper_status.json" 2>/dev/null || true

echo
echo "===== CHECKPOINTS ====="
find "$TRAIN_DIR" -type f -name '*.pt' -printf '%TY-%Tm-%Td %TH:%TM:%TS %s %p\n' 2>/dev/null | sort || true

echo
echo "===== LOG TAIL ====="
tail -n 80 "$PIPELINE_DIR/logs/overnight.log" 2>/dev/null || true
