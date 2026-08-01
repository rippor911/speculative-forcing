#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dataset-assist-0/luojy/efficiency/rippor/speculative-forcing"
PY="${PY:-/home/dataset-assist-0/luojy/efficiency/rippor/envs/speculative-forcing/bin/python}"
GPU="${GPU:-0}"
EXP="$ROOT/experiments/E0210_depth1_visual_gate/regression"
SCRIPT="$ROOT/experiments/E0210_depth1_visual_gate/run_visual_gate.py"
LOG="$ROOT/experiments/E0210_depth1_visual_gate/regression.log"

cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/experiments/E0202_anchor_replay_audit:$ROOT/experiments/E0203_mcp_short_training:$ROOT/experiments/E0205B_mcp_quality_gate:$ROOT/experiments/E0206B_depth1_single_state_overfit${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$(dirname "$LOG")"
set -o pipefail
CUDA_VISIBLE_DEVICES="$GPU" \
  "$PY" "$SCRIPT" \
    --suite regression \
    --device cuda:0 \
    --overwrite \
    2>&1 | tee "$LOG"
