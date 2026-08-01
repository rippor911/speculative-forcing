#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/dataset-assist-0/luojy/efficiency/rippor/speculative-forcing
PY=/home/dataset-assist-0/luojy/efficiency/rippor/envs/speculative-forcing/bin/python
CODE=experiments/E0211_target_refinement_curve_v2
EXP=experiments/E0211_target_refinement_curve_v2_outputs

cd "$ROOT"

export PYTHONPATH="$PWD:$PWD/experiments/E0202_anchor_replay_audit:$PWD/experiments/E0203_mcp_short_training:$PWD/experiments/E0205B_mcp_quality_gate:$PWD/experiments/E0206B_depth1_single_state_overfit"

echo "===== READ-ONLY PREFLIGHT ====="
git branch --show-current
git rev-parse HEAD
git status --short
df -h .
nvidia-smi

mkdir -p "$EXP"

"$PY" -m py_compile "$CODE/run_refinement_curve.py"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
"$PY" "$CODE/run_refinement_curve.py" \
  --sample-file experiments/E0201_teacher_rollout_smoke/teacher_sample_004.pt \
  --sample-file experiments/E0201_teacher_rollout_smoke/teacher_sample_005.pt \
  --anchor 0 \
  --anchor 1 \
  --anchor 2 \
  --anchor 3 \
  --timing-repeats 3 \
  --output-dir "$EXP" \
  2>&1 | tee "$EXP/run.log"

echo
echo "===== OUTPUT ====="
find "$EXP" -maxdepth 2 -type f -printf '%p\t%k KiB\n' | sort
