#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RESULT="experiments/E0210B_depth1_single_block_gate/results"
OUT="E0210B_single_block_results.tar.gz"

test -f "$RESULT/report.json"
tar -czf "$OUT" -C "$(dirname "$RESULT")" "$(basename "$RESULT")"
ls -lh "$OUT"
echo "$ROOT/$OUT"
