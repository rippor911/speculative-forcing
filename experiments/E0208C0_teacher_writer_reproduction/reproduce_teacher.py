from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from utils.checkpoint import extract_generator_state_dict
from utils.wan_wrapper import (
    WanDiffusionWrapper,
    WanTextEncoder,
)


ROOT = Path.cwd()

E0201_DIR = Path(
    "experiments/E0201_teacher_rollout_smoke"
)

MANIFEST_PATH = E0201_DIR / "manifest.json"

CONFIG_PATH = Path(
    "configs/self_forcing_dmd_mcp.yaml"
)

DEFAULT_CONFIG_PATH = Path(
    "configs/default_config.yaml"
)

CHECKPOINT_PATH = Path(
    "checkpoints/self_forcing_dmd.pt"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "a0413986d9734e02c09504e1520f5697"
    "ba6df731bb2f0f35577485e9cc8f56a3"
)

OUTPUT_ROOT = Path(
    "experiments/"
    "E0208C0_teacher_writer_reproduction"
)

AUDIT_MODULE_DIR = (
    ROOT
    / "experiments"
    / "E0202_anchor_replay_audit"
)

sys.path.insert(
    0,
    str(AUDIT_MODULE_DIR),
)

from audit_replay import (  # noqa: E402
    build_steps,
    make_pipeline,
    move_to_cpu,
    move_to_device,
    reset_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-indices",
        default="0",
        help="Comma-separated E0201 sample indices.",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--run-name",
        required=True,
    )

    parser.add_argument(
        "--noise-only",
        action="store_true",
    )

    parser.add_argument(
        "--save-samples",
        action="store_true",
    )

    return parser.parse_args()


def parse_indices(value: str) -> list[int]:
    indices = [
        int(part.strip())
        for part in value.split(",")
        if part.strip()
    ]

    if not indices:
        raise ValueError(
            "No sample indices supplied."
        )

    if len(indices) != len(set(indices)):
        raise ValueError(
            "Duplicate sample indices."
        )

    return indices


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(
                8 * 1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def tensor_sha256(
    tensor: torch.Tensor,
) -> str:
    contiguous = (
        tensor.detach()
        .cpu()
        .contiguous()
    )

    raw = (
        contiguous
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )

    return hashlib.sha256(raw).hexdigest()


def tensor_summary(
    tensor: torch.Tensor,
) -> dict[str, Any]:
    value = tensor.detach().float()

    return {
        "shape": [
            int(dim)
            for dim in tensor.shape
        ],
        "dtype": str(tensor.dtype),
        "finite": bool(
            torch.isfinite(value)
            .all()
            .item()
        ),
        "mean": float(
            value.mean().item()
        ),
        "std": float(
            value.std().item()
        ),
        "rms": float(
            value.square()
            .mean()
            .sqrt()
            .item()
        ),
        "sha256": tensor_sha256(
            tensor
        ),
    }


def comparison(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, Any]:
    shape_equal = (
        actual.shape == expected.shape
    )

    dtype_equal = (
        actual.dtype == expected.dtype
    )

    exact_equal = (
        shape_equal
        and dtype_equal
        and bool(
            torch.equal(
                actual,
                expected,
            )
        )
    )

    if shape_equal:
        difference = (
            actual.float()
            - expected.float()
        )

        max_abs_difference = float(
            difference.abs()
            .max()
            .item()
        )

        mse = float(
            difference.square()
            .mean()
            .item()
        )
    else:
        max_abs_difference = None
        mse = None

    return {
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "exact_equal": exact_equal,
        "max_abs_difference": (
            max_abs_difference
        ),
        "mse": mse,
        "actual_sha256": (
            tensor_sha256(actual)
        ),
        "expected_sha256": (
            tensor_sha256(expected)
        ),
    }


def load_manifest() -> dict[str, Any]:
    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    if manifest["status"] != "PASS":
        raise RuntimeError(
            "E0201 manifest is not PASS."
        )

    return manifest


def load_records(
    manifest: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    return {
        int(record["sample_index"]):
        record
        for record in manifest["samples"]
    }


def load_payload(
    record: dict[str, Any],
) -> dict[str, Any]:
    return torch.load(
        record["file"],
        map_location="cpu",
        weights_only=False,
    )


def create_source_noise(
    *,
    seed: int,
    expected: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    # E0201 uses the sample seed once to
    # construct source noise.
    reset_seed(seed)

    return torch.randn(
        tuple(expected.shape),
        device=device,
        dtype=expected.dtype,
    )


def precompute_embeddings(
    *,
    indices: list[int],
    records: dict[int, dict[str, Any]],
    device: torch.device,
) -> dict[int, dict[str, Any]]:
    encoder = WanTextEncoder()

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

    for sample_index in indices:
        payload = load_payload(
            records[sample_index]
        )

        with torch.no_grad():
            conditional_dict = encoder(
                text_prompts=[
                    payload["prompt"]
                ]
            )

        embeddings[sample_index] = (
            move_to_cpu(
                conditional_dict
            )
        )

        del conditional_dict
        del payload

    encoder.to("cpu")

    del encoder

    gc.collect()
    torch.cuda.empty_cache()

    return embeddings


def load_generator(
    *,
    config: Any,
    device: torch.device,
) -> tuple[
    WanDiffusionWrapper,
    list[int],
    list[float],
    dict[str, Any],
]:
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = (
        extract_generator_state_dict(
            checkpoint
        )
    )

    mcp_keys = [
        key
        for key in state_dict
        if (
            key.startswith("mcp.")
            or ".mcp." in key
        )
    ]

    generator = WanDiffusionWrapper(
        **getattr(
            config,
            "model_kwargs",
            {},
        ),
        is_causal=True,
    )

    load_result = (
        generator.load_state_dict(
            state_dict,
            strict=True,
        )
    )

    if (
        load_result.missing_keys
        or load_result.unexpected_keys
    ):
        raise RuntimeError(
            "Strict official backbone "
            f"restore failed: {load_result}"
        )

    del checkpoint
    del state_dict

    gc.collect()

    generator.eval()
    generator.requires_grad_(False)

    generator.to(
        device=device,
        dtype=torch.bfloat16,
    )

    raw_steps, warped_steps = (
        build_steps(
            config,
            generator.get_scheduler(),
        )
    )

    metadata = {
        "mcp_tensor_count": len(
            mcp_keys
        ),
        "missing_keys": list(
            load_result.missing_keys
        ),
        "unexpected_keys": list(
            load_result.unexpected_keys
        ),
    }

    return (
        generator,
        raw_steps,
        warped_steps,
        metadata,
    )


def atomic_torch_save(
    payload: dict[str, Any],
    path: Path,
) -> None:
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    torch.save(
        payload,
        temporary,
    )

    temporary.replace(path)


def main() -> None:
    args = parse_args()

    torch.set_grad_enabled(False)

    indices = parse_indices(
        args.sample_indices
    )

    device = torch.device(args.device)

    if (
        device.type != "cuda"
        or device.index not in (None, 0)
    ):
        raise ValueError(
            "Use one visible GPU and "
            "--device cuda:0."
        )

    torch.cuda.set_device(device)

    run_dir = (
        OUTPUT_ROOT
        / args.run_name
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    checkpoint_hash = file_sha256(
        CHECKPOINT_PATH
    )

    if (
        checkpoint_hash
        != EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError(
            "Official checkpoint "
            "SHA256 mismatch."
        )

    manifest = load_manifest()
    records = load_records(manifest)

    missing_indices = [
        index
        for index in indices
        if index not in records
    ]

    if missing_indices:
        raise KeyError(
            "Unknown E0201 indices: "
            f"{missing_indices}"
        )

    config = OmegaConf.merge(
        OmegaConf.load(
            DEFAULT_CONFIG_PATH
        ),
        OmegaConf.load(
            CONFIG_PATH
        ),
    )

    results: list[dict[str, Any]] = []

    # First reproduce source noise without
    # loading either text encoder or generator.
    old_payloads: dict[
        int,
        dict[str, Any],
    ] = {}

    for sample_index in indices:
        record = records[sample_index]
        payload = load_payload(record)

        old_payloads[sample_index] = payload

        expected_noise = payload[
            "source_noise"
        ]

        actual_noise = create_source_noise(
            seed=int(payload["seed"]),
            expected=expected_noise,
            device=device,
        )

        noise_comparison = comparison(
            actual_noise.detach().cpu(),
            expected_noise,
        )

        results.append(
            {
                "sample_index": (
                    sample_index
                ),
                "split": payload["split"],
                "seed": int(
                    payload["seed"]
                ),
                "source_noise": (
                    noise_comparison
                ),
                "target_latent": None,
                "status": (
                    "NOISE_PASS"
                    if noise_comparison[
                        "exact_equal"
                    ]
                    else "NOISE_FAIL"
                ),
            }
        )

        del actual_noise

    torch.cuda.empty_cache()

    noise_pass = all(
        record["source_noise"][
            "exact_equal"
        ]
        for record in results
    )

    if args.noise_only or not noise_pass:
        status = (
            "PASS"
            if noise_pass
            else "FAIL"
        )

        report = {
            "status": status,
            "mode": "noise_only",
            "indices": indices,
            "checkpoint": {
                "path": str(
                    CHECKPOINT_PATH.resolve()
                ),
                "sha256": (
                    checkpoint_hash
                ),
            },
            "samples": results,
        }

        report_path = (
            run_dir / "report.json"
        )

        report_path.write_text(
            json.dumps(
                report,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            json.dumps(
                report,
                indent=2,
            )
        )

        if status != "PASS":
            raise SystemExit(2)

        return

    embeddings = precompute_embeddings(
        indices=indices,
        records=records,
        device=device,
    )

    (
        generator,
        raw_steps,
        warped_steps,
        restore_metadata,
    ) = load_generator(
        config=config,
        device=device,
    )

    expected_generation = manifest[
        "generation"
    ]

    if (
        raw_steps
        != expected_generation[
            "raw_denoising_steps"
        ]
    ):
        raise RuntimeError(
            "Raw denoising steps differ."
        )

    if (
        warped_steps
        != expected_generation[
            "warped_denoising_steps"
        ]
    ):
        raise RuntimeError(
            "Warped denoising steps differ."
        )

    result_by_index = {
        int(record["sample_index"]):
        record
        for record in results
    }

    for sample_index in indices:
        payload = old_payloads[
            sample_index
        ]

        expected_noise = payload[
            "source_noise"
        ]

        source_noise = (
            create_source_noise(
                seed=int(
                    payload["seed"]
                ),
                expected=expected_noise,
                device=device,
            )
        )

        conditional_dict = (
            move_to_device(
                embeddings[sample_index],
                device,
            )
        )

        pipeline = make_pipeline(
            generator,
            warped_steps,
        )

        # E0201 resets the same sample seed
        # immediately before rollout. This
        # controls all scheduler randn_like
        # calls independently of source-noise
        # construction.
        reset_seed(
            int(payload["seed"])
        )

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
            torch.cuda
            .max_memory_allocated(device)
            / 1024**3
        )

        target_cpu = (
            target_latent
            .detach()
            .cpu()
        )

        target_comparison = comparison(
            target_cpu,
            payload["target_latent"],
        )

        result_record = (
            result_by_index[
                sample_index
            ]
        )

        result_record[
            "target_latent"
        ] = target_comparison

        result_record[
            "generation_seconds"
        ] = elapsed

        result_record[
            "peak_allocated_gib"
        ] = peak_gib

        sample_pass = (
            result_record[
                "source_noise"
            ]["exact_equal"]
            and target_comparison[
                "exact_equal"
            ]
        )

        result_record["status"] = (
            "PASS"
            if sample_pass
            else "FAIL"
        )

        if args.save_samples:
            reproduced_payload = {
                "format": (
                    "self_forcing_teacher_v1"
                ),
                "sample_index": (
                    sample_index
                ),
                "split": payload["split"],
                "prompt": payload["prompt"],
                "seed": int(
                    payload["seed"]
                ),
                "source_noise": (
                    source_noise
                    .detach()
                    .cpu()
                ),
                "target_latent": (
                    target_cpu
                ),
                "valid_anchor_blocks": (
                    list(
                        payload[
                            "valid_anchor_blocks"
                        ]
                    )
                ),
                "backbone_checkpoint": (
                    str(
                        CHECKPOINT_PATH
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
                "num_frames": int(
                    payload["num_frames"]
                ),
                "num_frame_per_block": (
                    int(
                        payload[
                            "num_frame_per_block"
                        ]
                    )
                ),
                "mcp_depth": int(
                    payload["mcp_depth"]
                ),
            }

            sample_path = (
                run_dir
                / (
                    "reproduced_"
                    f"sample_{sample_index:03d}"
                    ".pt"
                )
            )

            atomic_torch_save(
                reproduced_payload,
                sample_path,
            )

            result_record[
                "reproduced_file"
            ] = str(
                sample_path.resolve()
            )

            result_record[
                "reproduced_file_sha256"
            ] = file_sha256(
                sample_path
            )

        print(
            f"sample={sample_index} "
            f"status="
            f"{result_record['status']} "
            f"noise_max_diff="
            f"{result_record['source_noise']['max_abs_difference']} "
            f"target_max_diff="
            f"{target_comparison['max_abs_difference']} "
            f"seconds={elapsed:.3f}",
            flush=True,
        )

        del source_noise
        del conditional_dict
        del pipeline
        del output
        del target_latent
        del target_cpu

        torch.cuda.empty_cache()

    status = (
        "PASS"
        if all(
            record["status"] == "PASS"
            for record in results
        )
        else "FAIL"
    )

    report = {
        "status": status,
        "mode": "full_reproduction",
        "indices": indices,
        "checkpoint": {
            "path": str(
                CHECKPOINT_PATH.resolve()
            ),
            "sha256": checkpoint_hash,
        },
        "config": {
            "path": str(CONFIG_PATH),
            "raw_denoising_steps": (
                raw_steps
            ),
            "warped_denoising_steps": (
                warped_steps
            ),
            "pipeline_overrides": {
                "mcp_num_modules": 0,
                "mcp_accel_depths": 0,
                "memory_gap_blocks": 0,
                "last_step_only": True,
                "independent_first_frame": (
                    False
                ),
                "same_step_across_blocks": (
                    False
                ),
            },
        },
        "restore": restore_metadata,
        "samples": results,
    }

    report_path = (
        run_dir / "report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "report=",
        report_path.resolve(),
    )

    print(
        "E0208C0_REPRODUCTION=",
        status,
    )

    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
