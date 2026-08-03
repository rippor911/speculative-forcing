from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.nf_sf_m4 import load_m4_sample_plan
from utils.nf_sf_m5_conditionals import build_m5_conditional_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the M5 formal per-identity conditional artifact."
    )
    parser.add_argument("--sample_plan", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    return parser.parse_args()


def _torch_dtype(value: str) -> torch.dtype:
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {value}")


def main() -> int:
    args = parse_args()
    sample_plan_path = Path(args.sample_plan)
    if sample_plan_path.name.lower().endswith(".tmp"):
        raise ValueError("--sample_plan must not end with .tmp")
    sample_plan = load_m4_sample_plan(sample_plan_path, manifest_path=args.manifest)

    from utils.nf_sf_m3 import file_sha256
    from utils.wan_wrapper import WAN_MODELS_ROOT, WanTextEncoder

    dtype = _torch_dtype(args.dtype)
    model_checkpoint_path = (
        WAN_MODELS_ROOT / "Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
    )
    tokenizer_path = WAN_MODELS_ROOT / "Wan2.1-T2V-1.3B/google/umt5-xxl"
    text_encoder = WanTextEncoder().to(device=args.device, dtype=dtype).eval()
    text_encoder.requires_grad_(False)
    try:
        report = build_m5_conditional_artifact(
            sample_plan=sample_plan,
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            output_dir=args.output,
            text_encoder=text_encoder,
            encoder_provenance={
                "encoder_class": "utils.wan_wrapper.WanTextEncoder",
                "model_checkpoint_path": str(model_checkpoint_path.resolve()),
                "model_checkpoint_sha256": file_sha256(model_checkpoint_path),
                "tokenizer_path": str(tokenizer_path.resolve()),
                "device": args.device,
                "dtype": str(dtype),
            },
        )
    finally:
        text_encoder.to("cpu")
        del text_encoder
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print("M5_CONDITIONAL_ARTIFACT_BUILD_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
