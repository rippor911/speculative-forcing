from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


REPRODUCTION_DIR = Path(
    "experiments/"
    "E0208C0_teacher_writer_reproduction"
).resolve()

sys.path.insert(
    0,
    str(REPRODUCTION_DIR),
)

import reproduce_teacher as common  # noqa: E402


SAMPLE_INDEX_OFFSETS = {
    "train": 0,
    "validation": 1_000_000,
    "reserve": 2_000_000,
}

SEED_BASES = {
    "train": 1_000_000,
    "validation": 2_000_000,
    "reserve": 3_000_000,
}

NUM_FRAMES = 21
NUM_FRAME_PER_BLOCK = 3
MCP_DEPTH = 3
VALID_ANCHOR_BLOCKS = [0, 1, 2, 3]

WRITER_FORMAT = "e0208_teacher_writer_v1"
PAYLOAD_FORMAT = "self_forcing_teacher_v1"
MANIFEST_FORMAT = "self_forcing_teacher_manifest_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plan",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    return parser.parse_args()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def load_plan(
    path: Path,
) -> list[dict[str, Any]]:
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            required = {
                "split",
                "split_index",
                "source_line_index",
                "prompt",
                "normalized_sha256",
                "token_count",
                "shard_id",
                "plan_index",
            }

            missing = required - record.keys()

            if missing:
                raise KeyError(
                    f"{path}:{line_number} "
                    f"missing fields: {sorted(missing)}"
                )

            split = str(record["split"])

            if split not in SAMPLE_INDEX_OFFSETS:
                raise ValueError(
                    f"Unsupported split: {split}"
                )

            records.append(record)

    if not records:
        raise RuntimeError("Plan is empty.")

    plan_indices = [
        int(record["plan_index"])
        for record in records
    ]

    if len(plan_indices) != len(set(plan_indices)):
        raise RuntimeError(
            "Duplicate plan_index values."
        )

    identities = [
        (
            str(record["split"]),
            int(record["split_index"]),
        )
        for record in records
    ]

    if len(identities) != len(set(identities)):
        raise RuntimeError(
            "Duplicate split/split_index records."
        )

    return sorted(
        records,
        key=lambda record: int(
            record["plan_index"]
        ),
    )


def sample_index(
    record: dict[str, Any],
) -> int:
    split = str(record["split"])

    return (
        SAMPLE_INDEX_OFFSETS[split]
        + int(record["split_index"])
    )


def sample_seed(
    record: dict[str, Any],
) -> int:
    split = str(record["split"])

    return (
        SEED_BASES[split]
        + int(record["split_index"])
    )


def sample_path(
    output_dir: Path,
    record: dict[str, Any],
) -> Path:
    return output_dir / (
        f"teacher_{record['split']}_"
        f"{int(record['split_index']):06d}.pt"
    )


def atomic_json_write(
    payload: dict[str, Any],
    path: Path,
) -> None:
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def create_source_noise(
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(seed)

    return torch.randn(
        (
            1,
            NUM_FRAMES,
            16,
            60,
            104,
        ),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )


def precompute_embeddings(
    *,
    records: list[dict[str, Any]],
    device: torch.device,
) -> dict[int, dict[str, Any]]:
    encoder = common.WanTextEncoder()

    encoder.eval()
    encoder.requires_grad_(False)

    encoder.to(
        device=device,
        dtype=torch.bfloat16,
    )

    embeddings: dict[
        int,
        dict[str, Any],
    ] = {}

    for record in records:
        index = sample_index(record)

        with torch.no_grad():
            conditional_dict = encoder(
                text_prompts=[
                    str(record["prompt"])
                ]
            )

        embeddings[index] = (
            common.move_to_cpu(
                conditional_dict
            )
        )

        del conditional_dict

    encoder.to("cpu")

    del encoder

    gc.collect()
    torch.cuda.empty_cache()

    return embeddings


def validate_existing_payload(
    *,
    payload: dict[str, Any],
    record: dict[str, Any],
    checkpoint_hash: str,
) -> None:
    expected = {
        "format": PAYLOAD_FORMAT,
        "sample_index": sample_index(record),
        "split": str(record["split"]),
        "split_index": int(
            record["split_index"]
        ),
        "source_line_index": int(
            record["source_line_index"]
        ),
        "prompt": str(record["prompt"]),
        "prompt_sha256": str(
            record["normalized_sha256"]
        ),
        "seed": sample_seed(record),
        "backbone_sha256": checkpoint_hash,
        "num_frames": NUM_FRAMES,
        "num_frame_per_block": (
            NUM_FRAME_PER_BLOCK
        ),
        "mcp_depth": MCP_DEPTH,
    }

    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise RuntimeError(
                f"Existing payload field "
                f"{key!r} differs."
            )

    for tensor_name in (
        "source_noise",
        "target_latent",
    ):
        tensor = payload.get(tensor_name)

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{tensor_name} is not a tensor."
            )

        expected_shape = (
            1,
            NUM_FRAMES,
            16,
            60,
            104,
        )

        if tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"{tensor_name} shape differs: "
                f"{tuple(tensor.shape)}"
            )

        if tensor.dtype != torch.bfloat16:
            raise RuntimeError(
                f"{tensor_name} dtype differs: "
                f"{tensor.dtype}"
            )

        if not bool(
            torch.isfinite(
                tensor.float()
            ).all().item()
        ):
            raise RuntimeError(
                f"{tensor_name} is non-finite."
            )


def manifest_sample_record(
    *,
    path: Path,
    payload: dict[str, Any],
    status: str,
) -> dict[str, Any]:
    return {
        "sample_index": int(
            payload["sample_index"]
        ),
        "split": str(payload["split"]),
        "split_index": int(
            payload["split_index"]
        ),
        "source_line_index": int(
            payload["source_line_index"]
        ),
        "shard_id": int(
            payload["shard_id"]
        ),
        "plan_index": int(
            payload["plan_index"]
        ),
        "file": str(path.resolve()),
        "file_sha256": common.file_sha256(
            path
        ),
        "prompt": str(payload["prompt"]),
        "prompt_sha256": str(
            payload["prompt_sha256"]
        ),
        "seed": int(payload["seed"]),
        "noise_seed": int(
            payload["noise_seed"]
        ),
        "rollout_seed": int(
            payload["rollout_seed"]
        ),
        "generation_seconds": float(
            payload["generation_seconds"]
        ),
        "peak_allocated_gib": float(
            payload["peak_allocated_gib"]
        ),
        "source_noise": (
            common.tensor_summary(
                payload["source_noise"]
            )
        ),
        "target_latent": (
            common.tensor_summary(
                payload["target_latent"]
            )
        ),
        "valid_anchor_blocks": list(
            payload["valid_anchor_blocks"]
        ),
        "status": status,
    }


def build_manifest(
    *,
    status: str,
    plan_path: Path,
    plan: list[dict[str, Any]],
    output_dir: Path,
    checkpoint_hash: str,
    raw_steps: list[int] | None,
    warped_steps: list[float] | None,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    split_counts = Counter(
        str(record["split"])
        for record in plan
    )

    return {
        "status": status,
        "experiment": (
            "E0208C_teacher_rollout_formal"
        ),
        "format": MANIFEST_FORMAT,
        "writer_format": WRITER_FORMAT,
        "writer_git_head": git_head(),
        "plan": {
            "path": str(plan_path.resolve()),
            "sha256": common.file_sha256(
                plan_path
            ),
            "num_records": len(plan),
        },
        "checkpoint": {
            "path": str(
                common.CHECKPOINT_PATH.resolve()
            ),
            "sha256": checkpoint_hash,
        },
        "device": {
            "runtime": str(device),
            "name": torch.cuda.get_device_name(
                device
            ),
        },
        "generation": {
            "num_samples": len(plan),
            "num_completed": len(samples),
            "num_train": split_counts.get(
                "train",
                0,
            ),
            "num_validation": (
                split_counts.get(
                    "validation",
                    0,
                )
            ),
            "num_reserve": split_counts.get(
                "reserve",
                0,
            ),
            "num_frames": NUM_FRAMES,
            "num_frame_per_block": (
                NUM_FRAME_PER_BLOCK
            ),
            "num_blocks": (
                NUM_FRAMES
                // NUM_FRAME_PER_BLOCK
            ),
            "mcp_depth": MCP_DEPTH,
            "valid_anchor_blocks": (
                VALID_ANCHOR_BLOCKS
            ),
            "raw_denoising_steps": raw_steps,
            "warped_denoising_steps": (
                warped_steps
            ),
            "mcp_num_modules": 0,
            "mcp_accel_depths": 0,
            "memory_gap_blocks": 0,
            "last_step_only": True,
            "noise_recipe": {
                "generator": (
                    "torch.Generator(device='cuda')"
                ),
                "dtype": "torch.bfloat16",
                "shape": [
                    1,
                    NUM_FRAMES,
                    16,
                    60,
                    104,
                ],
                "seed_rule": (
                    "split_seed_base + "
                    "split_index"
                ),
                "split_seed_bases": (
                    SEED_BASES
                ),
            },
        },
        "output_dir": str(
            output_dir.resolve()
        ),
        "samples": sorted(
            samples,
            key=lambda record: int(
                record["plan_index"]
            ),
        ),
    }


def main() -> None:
    args = parse_args()

    torch.set_grad_enabled(False)

    device = torch.device(args.device)

    if (
        device.type != "cuda"
        or device.index not in (None, 0)
    ):
        raise ValueError(
            "Use one visible CUDA device "
            "and --device cuda:0."
        )

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Exactly one CUDA device must "
            "be visible to each shard process."
        )

    torch.cuda.set_device(device)

    plan_path = args.plan.resolve()
    output_dir = args.output_dir.resolve()

    plan = load_plan(plan_path)

    if output_dir.exists():
        if not args.resume and any(
            output_dir.iterdir()
        ):
            raise FileExistsError(
                f"Output directory is not empty: "
                f"{output_dir}"
            )
    else:
        output_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

    manifest_path = output_dir / "manifest.json"

    checkpoint_hash = common.file_sha256(
        common.CHECKPOINT_PATH
    )

    if (
        checkpoint_hash
        != common.EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            "Official checkpoint SHA256 mismatch."
        )

    completed: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []

    for record in plan:
        path = sample_path(
            output_dir,
            record,
        )

        if path.exists():
            if not args.resume:
                raise FileExistsError(path)

            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            validate_existing_payload(
                payload=payload,
                record=record,
                checkpoint_hash=(
                    checkpoint_hash
                ),
            )

            completed.append(
                manifest_sample_record(
                    path=path,
                    payload=payload,
                    status="RESUMED_VALID",
                )
            )
        else:
            pending.append(record)

    config = OmegaConf.merge(
        OmegaConf.load(
            common.DEFAULT_CONFIG_PATH
        ),
        OmegaConf.load(
            common.CONFIG_PATH
        ),
    )

    raw_steps: list[int] | None = None
    warped_steps: list[float] | None = None

    if pending:
        print(
            f"precompute_embeddings={len(pending)}",
            flush=True,
        )

        embeddings = precompute_embeddings(
            records=pending,
            device=device,
        )

        print(
            "load_official_target=START",
            flush=True,
        )

        (
            generator,
            raw_steps,
            warped_steps,
            restore_metadata,
        ) = common.load_generator(
            config=config,
            device=device,
        )

        if (
            restore_metadata[
                "mcp_tensor_count"
            ]
            != 0
        ):
            raise RuntimeError(
                "Official Target checkpoint "
                "contains MCP tensors."
            )

        expected_manifest = (
            common.load_manifest()
        )

        expected_generation = (
            expected_manifest[
                "generation"
            ]
        )

        if (
            raw_steps
            != expected_generation[
                "raw_denoising_steps"
            ]
        ):
            raise RuntimeError(
                "Raw denoising schedule differs."
            )

        if (
            warped_steps
            != expected_generation[
                "warped_denoising_steps"
            ]
        ):
            raise RuntimeError(
                "Warped denoising schedule differs."
            )

        for record in pending:
            index = sample_index(record)
            seed = sample_seed(record)

            path = sample_path(
                output_dir,
                record,
            )

            source_noise = (
                create_source_noise(
                    seed=seed,
                    device=device,
                )
            )

            conditional_dict = (
                common.move_to_device(
                    embeddings[index],
                    device,
                )
            )

            pipeline = common.make_pipeline(
                generator,
                warped_steps,
            )

            common.reset_seed(seed)

            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(
                device
            )

            started = time.perf_counter()

            with torch.no_grad():
                output = (
                    pipeline
                    .inference_with_trajectory(
                        noise=source_noise,
                        initial_latent=None,
                        **conditional_dict,
                    )
                )

            target_latent = output[0]

            torch.cuda.synchronize(device)

            elapsed = (
                time.perf_counter()
                - started
            )

            peak_gib = (
                torch.cuda.max_memory_allocated(
                    device
                )
                / 1024**3
            )

            expected_shape = (
                1,
                NUM_FRAMES,
                16,
                60,
                104,
            )

            if (
                tuple(target_latent.shape)
                != expected_shape
            ):
                raise RuntimeError(
                    "Target latent shape differs: "
                    f"{tuple(target_latent.shape)}"
                )

            if (
                target_latent.dtype
                != torch.bfloat16
            ):
                raise RuntimeError(
                    "Target latent dtype differs: "
                    f"{target_latent.dtype}"
                )

            if not bool(
                torch.isfinite(
                    target_latent.float()
                ).all().item()
            ):
                raise RuntimeError(
                    "Target latent is non-finite."
                )

            source_cpu = (
                source_noise
                .detach()
                .cpu()
            )

            target_cpu = (
                target_latent
                .detach()
                .cpu()
            )

            payload = {
                "format": PAYLOAD_FORMAT,
                "writer_format": (
                    WRITER_FORMAT
                ),
                "sample_index": index,
                "split": str(
                    record["split"]
                ),
                "split_index": int(
                    record["split_index"]
                ),
                "source_line_index": int(
                    record[
                        "source_line_index"
                    ]
                ),
                "shard_id": int(
                    record["shard_id"]
                ),
                "plan_index": int(
                    record["plan_index"]
                ),
                "prompt": str(
                    record["prompt"]
                ),
                "prompt_sha256": str(
                    record[
                        "normalized_sha256"
                    ]
                ),
                "token_count": int(
                    record["token_count"]
                ),
                "seed": seed,
                "noise_seed": seed,
                "rollout_seed": seed,
                "source_noise": source_cpu,
                "target_latent": target_cpu,
                "valid_anchor_blocks": (
                    VALID_ANCHOR_BLOCKS
                ),
                "backbone_checkpoint": (
                    str(
                        common
                        .CHECKPOINT_PATH
                        .resolve()
                    )
                ),
                "backbone_sha256": (
                    checkpoint_hash
                ),
                "raw_denoising_steps": (
                    list(raw_steps)
                ),
                "warped_denoising_steps": (
                    list(warped_steps)
                ),
                "num_frames": NUM_FRAMES,
                "num_frame_per_block": (
                    NUM_FRAME_PER_BLOCK
                ),
                "mcp_depth": MCP_DEPTH,
                "generation_seconds": (
                    elapsed
                ),
                "peak_allocated_gib": (
                    peak_gib
                ),
                "writer_git_head": (
                    git_head()
                ),
            }

            common.atomic_torch_save(
                payload,
                path,
            )

            completed.append(
                manifest_sample_record(
                    path=path,
                    payload=payload,
                    status="GENERATED",
                )
            )

            running_manifest = (
                build_manifest(
                    status="RUNNING",
                    plan_path=plan_path,
                    plan=plan,
                    output_dir=output_dir,
                    checkpoint_hash=(
                        checkpoint_hash
                    ),
                    raw_steps=raw_steps,
                    warped_steps=(
                        warped_steps
                    ),
                    samples=completed,
                    device=device,
                )
            )

            atomic_json_write(
                running_manifest,
                manifest_path,
            )

            print(
                f"sample_index={index} "
                f"split={record['split']} "
                f"split_index="
                f"{record['split_index']} "
                f"seconds={elapsed:.3f} "
                f"peak_gib={peak_gib:.3f} "
                f"status=PASS",
                flush=True,
            )

            del source_noise
            del conditional_dict
            del pipeline
            del output
            del target_latent
            del source_cpu
            del target_cpu
            del payload

            gc.collect()
            torch.cuda.empty_cache()

    else:
        old_manifest = json.loads(
            manifest_path.read_text(
                encoding="utf-8"
            )
        )

        raw_steps = old_manifest[
            "generation"
        ]["raw_denoising_steps"]

        warped_steps = old_manifest[
            "generation"
        ]["warped_denoising_steps"]

    if len(completed) != len(plan):
        raise RuntimeError(
            f"Completed {len(completed)} "
            f"of {len(plan)} samples."
        )

    final_manifest = build_manifest(
        status="PASS",
        plan_path=plan_path,
        plan=plan,
        output_dir=output_dir,
        checkpoint_hash=checkpoint_hash,
        raw_steps=raw_steps,
        warped_steps=warped_steps,
        samples=completed,
        device=device,
    )

    atomic_json_write(
        final_manifest,
        manifest_path,
    )

    print(
        "manifest=",
        manifest_path.resolve(),
    )

    print(
        "E0208C1_SHARD=PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
