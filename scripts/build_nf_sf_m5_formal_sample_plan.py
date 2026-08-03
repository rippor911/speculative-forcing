from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.nf_sf_m4 import load_m4_sample_plan
from utils.nf_sf_m5_formal_plan import (
    build_m5_formal_sample_plan,
    validate_m5_formal_sample_plan,
    write_m5_formal_sample_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the locked M5 formal 2048/256 M4-compatible sample plan."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)

    plan = build_m5_formal_sample_plan(
        manifest_path=manifest_path,
        dataset_root=dataset_root,
    )
    validate_m5_formal_sample_plan(
        plan,
        manifest_path=manifest_path,
        dataset_root=dataset_root,
    )
    write_m5_formal_sample_plan(plan, output_path)
    loaded = load_m4_sample_plan(output_path)
    audit = validate_m5_formal_sample_plan(
        loaded,
        manifest_path=manifest_path,
        dataset_root=dataset_root,
        expected_sha256=str(plan["sample_plan_sha256"]),
    )
    print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))
    print("M5_FORMAL_SAMPLE_PLAN_BUILD_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
