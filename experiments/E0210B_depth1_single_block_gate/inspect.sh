#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/home/dataset-assist-0/luojy/efficiency/rippor/envs/speculative-forcing/bin/python}"
RESULT="$ROOT/experiments/E0210B_depth1_single_block_gate/results"

cd "$ROOT"
"$PY" - <<'PY'
import json
from pathlib import Path
p = Path("experiments/E0210B_depth1_single_block_gate/results/report.json")
if not p.is_file():
    raise FileNotFoundError(p)
d = json.loads(p.read_text(encoding="utf-8"))
print("status=", d.get("status"))
print("sample_count=", len(d.get("samples", [])))
print("state_count=", d.get("metrics", {}).get("aggregate", {}).get("state_count"))
print("video_count=", d.get("artifacts", {}).get("video_count"))
print("mse=", d.get("metrics", {}).get("aggregate", {}).get("draft_target_mse"))
print("progress=", d.get("metrics", {}).get("aggregate", {}).get("progress_to_target"))
print("review=", d.get("artifacts", {}).get("review_html"))
PY
find "$RESULT/videos" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
