from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.nf_sf_m3 import atomic_json_write, load_m3_teacher_sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit one real NF-SF M3 teacher latent sample."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--split_index", type=int, default=None)
    parser.add_argument("--reference_checkpoint", type=Path, default=None)
    parser.add_argument("--output_json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = load_m3_teacher_sample(
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        sample_index=args.sample_index,
        sample_id=args.sample_id,
        split=args.split,
        split_index=args.split_index,
        reference_checkpoint_path=args.reference_checkpoint,
    )
    report = {
        "status": "PASS",
        "sample_metadata": sample.metadata,
    }
    if args.output_json is not None:
        atomic_json_write(report, args.output_json)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
