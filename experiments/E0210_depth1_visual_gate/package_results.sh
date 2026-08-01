#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dataset-assist-0/luojy/efficiency/rippor/speculative-forcing"
SUITE="${1:-regression}"
SRC="$ROOT/experiments/E0210_depth1_visual_gate/$SUITE"
OUT="$ROOT/E0210_${SUITE}_visual_results.tar.gz"

if [ ! -f "$SRC/report.json" ]; then
    echo "Missing $SRC/report.json" >&2
    exit 1
fi

tar -czf "$OUT" \
  -C "$ROOT/experiments/E0210_depth1_visual_gate" \
  "$SUITE"

ls -lh "$OUT"
echo "$OUT"
