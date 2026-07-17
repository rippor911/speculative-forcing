#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/self_forcing_dmd_memory_gap.yaml}"
LOGDIR="${LOGDIR:-logs/self_forcing_dmd_memory_gap}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RDZV_ID="${RDZV_ID:-5235}"
RDZV_BACKEND="${RDZV_BACKEND:-c10d}"
RDZV_ENDPOINT="${RDZV_ENDPOINT:-${MASTER_ADDR:-127.0.0.1:29500}}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --rdzv_id="${RDZV_ID}" \
  --rdzv_backend="${RDZV_BACKEND}" \
  --rdzv_endpoint="${RDZV_ENDPOINT}" \
  train.py \
  --config_path "${CONFIG_PATH}" \
  --logdir "${LOGDIR}" \
  "$@"
