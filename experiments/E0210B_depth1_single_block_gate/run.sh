#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/home/dataset-assist-0/luojy/efficiency/rippor/envs/speculative-forcing/bin/python}"
GPU="${GPU:-0}"
EXP="$ROOT/experiments/E0210B_depth1_single_block_gate"
LOG="$EXP/run.log"

cd "$ROOT"
mkdir -p "$EXP"

export PYTHONPATH="$ROOT:$ROOT/experiments/E0210_depth1_visual_gate:$ROOT/experiments/E0202_anchor_replay_audit:$ROOT/experiments/E0203_mcp_short_training:$ROOT/experiments/E0205B_mcp_quality_gate:$ROOT/experiments/E0206B_depth1_single_state_overfit${PYTHONPATH:+:$PYTHONPATH}"

set -o pipefail
CUDA_VISIBLE_DEVICES="$GPU" \
"$PY" "$EXP/run_single_block_gate.py" \
  --device cuda:0 \
  --overwrite \
  2>&1 | tee "$LOG"

status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$EXP/run.status"
echo "E0210B_EXIT_CODE=$status"
exit "$status"
