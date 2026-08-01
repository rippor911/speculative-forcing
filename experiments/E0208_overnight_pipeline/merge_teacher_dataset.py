from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_CHECKPOINT_SHA256 = (
    "a0413986d9734e02c09504e1520f5697"
    "ba6df731bb2f0f35577485e9cc8f56a3"
)
EXPECTED_TRAIN = 2048
EXPECTED_VALIDATION = 256
EXPECTED_TOTAL = EXPECTED_TRAIN + EXPECTED_VALIDATION
EXPECTED_SHARDS = 36


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-dir",
        type=Path,
        default=Path("experiments/E0208C_teacher_rollout_formal"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_write(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    formal_dir = args.formal_dir.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else (formal_dir / "manifest.json")
    )

    dataset_plan_path = formal_dir / "plans" / "dataset_plan.json"
    if not dataset_plan_path.is_file():
        raise FileNotFoundError(dataset_plan_path)

    dataset_plan = json.loads(dataset_plan_path.read_text(encoding="utf-8"))
    if dataset_plan.get("status") != "PASS":
        raise RuntimeError("dataset plan is not PASS")
    if int(dataset_plan.get("total_count", -1)) != EXPECTED_TOTAL:
        raise RuntimeError("dataset plan total_count differs")
    if int(dataset_plan.get("num_shards", -1)) != EXPECTED_SHARDS:
        raise RuntimeError("dataset plan num_shards differs")

    merged_samples: list[dict[str, Any]] = []
    reference_generation: dict[str, Any] | None = None
    reference_checkpoint: dict[str, Any] | None = None
    shard_records: list[dict[str, Any]] = []

    for shard_index in range(EXPECTED_SHARDS):
        shard_name = f"shard_{shard_index:03d}"
        manifest_path = formal_dir / "shards" / shard_name / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS":
            raise RuntimeError(f"{shard_name} is not PASS")
        if int(manifest["generation"]["num_completed"]) != 64:
            raise RuntimeError(f"{shard_name} is incomplete")
        if manifest["checkpoint"]["sha256"] != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError(f"{shard_name} checkpoint differs")

        if reference_generation is None:
            reference_generation = dict(manifest["generation"])
            reference_checkpoint = dict(manifest["checkpoint"])
        else:
            for key in (
                "num_frames",
                "num_frame_per_block",
                "num_blocks",
                "mcp_depth",
                "valid_anchor_blocks",
                "raw_denoising_steps",
                "warped_denoising_steps",
                "mcp_num_modules",
                "mcp_accel_depths",
                "memory_gap_blocks",
                "last_step_only",
            ):
                if manifest["generation"].get(key) != reference_generation.get(key):
                    raise RuntimeError(f"{shard_name} generation.{key} differs")

        for sample in manifest["samples"]:
            file_path = Path(sample["file"])
            if not file_path.is_file():
                raise FileNotFoundError(file_path)
            actual_file_hash = file_sha256(file_path)
            if actual_file_hash != sample["file_sha256"]:
                raise RuntimeError(f"file hash differs: {file_path}")
            merged_samples.append(dict(sample))

        shard_records.append(
            {
                "shard_id": shard_index,
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": file_sha256(manifest_path),
                "num_samples": len(manifest["samples"]),
            }
        )

    identities = [
        (str(sample["split"]), int(sample["split_index"]))
        for sample in merged_samples
    ]
    sample_indices = [int(sample["sample_index"]) for sample in merged_samples]

    if len(merged_samples) != EXPECTED_TOTAL:
        raise RuntimeError(
            f"expected {EXPECTED_TOTAL} samples, got {len(merged_samples)}"
        )
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate split/split_index identity")
    if len(sample_indices) != len(set(sample_indices)):
        raise RuntimeError("duplicate sample_index")

    split_counts = {
        split: sum(sample["split"] == split for sample in merged_samples)
        for split in ("train", "validation")
    }
    if split_counts != {"train": EXPECTED_TRAIN, "validation": EXPECTED_VALIDATION}:
        raise RuntimeError(f"split counts differ: {split_counts}")

    assert reference_generation is not None
    assert reference_checkpoint is not None

    merged_samples.sort(
        key=lambda sample: (
            0 if sample["split"] == "train" else 1,
            int(sample["split_index"]),
        )
    )

    generation = dict(reference_generation)
    generation.update(
        {
            "num_samples": EXPECTED_TOTAL,
            "num_completed": EXPECTED_TOTAL,
            "num_train": EXPECTED_TRAIN,
            "num_validation": EXPECTED_VALIDATION,
            "num_reserve": 0,
        }
    )

    merged_manifest = {
        "status": "PASS",
        "experiment": "E0208C_teacher_rollout_formal",
        "format": "self_forcing_teacher_manifest_v2_merged",
        "checkpoint": reference_checkpoint,
        "generation": generation,
        "dataset_plan": {
            "path": str(dataset_plan_path.resolve()),
            "sha256": file_sha256(dataset_plan_path),
        },
        "shards": shard_records,
        "samples": merged_samples,
    }

    atomic_json_write(merged_manifest, output)

    audit = {
        "status": "PASS",
        "manifest": str(output.resolve()),
        "manifest_sha256": file_sha256(output),
        "num_samples": len(merged_samples),
        "split_counts": split_counts,
        "num_shards": len(shard_records),
        "all_file_hashes_valid": True,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
    }
    audit_path = formal_dir / "merge_audit.json"
    atomic_json_write(audit, audit_path)

    print(json.dumps(audit, indent=2))
    print("E0208C_FORMAL_MERGE=PASS")


if __name__ == "__main__":
    main()
