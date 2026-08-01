#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dataset-assist-0/luojy/efficiency/rippor/speculative-forcing"
SUITE="${1:-regression}"
OUT="$ROOT/experiments/E0210_depth1_visual_gate/$SUITE"
LOG="$ROOT/experiments/E0210_depth1_visual_gate/$SUITE.log"
PY="${PY:-/home/dataset-assist-0/luojy/efficiency/rippor/envs/speculative-forcing/bin/python}"

cd "$ROOT"
echo "===== LOG TAIL ====="
tail -n 80 "$LOG" 2>/dev/null || true

echo
echo "===== ARTIFACTS ====="
find "$OUT" -maxdepth 2 -type f -printf '%p %s bytes\n' 2>/dev/null | sort

echo
echo "===== SUMMARY ====="
"$PY" - "$OUT/report.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("report=MISSING")
    raise SystemExit(1)
report = json.loads(path.read_text(encoding="utf-8"))
print("status=", report.get("status"))
print("suite=", report.get("suite"))
print("samples=", len(report.get("samples", [])))
print("videos=", report.get("artifacts", {}).get("video_count"))
print("scope_audit=", report.get("formal_checkpoint_scope_audit"))
for name, metrics in report.get("metrics", {}).get("aggregate", {}).items():
    print(
        f"{name}: mse={metrics['draft_target_mse']:.8f} "
        f"progress={metrics['progress_to_target']:.6f} "
        f"cos={metrics['flow_cosine_with_oracle']:.6f}"
    )
print("review=", report.get("artifacts", {}).get("review_html"))
print("manual_gate=", report.get("manual_gate", {}).get("status"))
PY
