from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import reproduce_teacher as common  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-indices",
        required=True,
        help="Comma-separated E0201 sample indices.",
    )

    parser.add_argument(
        "--run-name",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    return parser.parse_args()


def parse_indices(value: str) -> list[int]:
    indices = [
        int(part.strip())
        for part in value.split(",")
        if part.strip()
    ]

    if not indices:
        raise ValueError("No sample indices supplied.")

    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate sample indices.")

    return indices


def main() -> None:
    args = parse_args()
    indices = parse_indices(args.sample_indices)

    torch.set_grad_enabled(False)

    device = torch.device(args.device)

    if (
        device.type != "cuda"
        or device.index not in (None, 0)
    ):
        raise ValueError(
            "Use one visible GPU and --device cuda:0."
        )

    torch.cuda.set_device(device)

    run_dir = common.OUTPUT_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)

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

    manifest = common.load_manifest()
    records = common.load_records(manifest)

    missing = [
        index
        for index in indices
        if index not in records
    ]

    if missing:
        raise KeyError(
            f"Unknown E0201 sample indices: {missing}"
        )

    config = OmegaConf.merge(
        OmegaConf.load(
            common.DEFAULT_CONFIG_PATH
        ),
        OmegaConf.load(
            common.CONFIG_PATH
        ),
    )

    print(
        "===== PRECOMPUTE EMBEDDINGS =====",
        flush=True,
    )

    embeddings = common.precompute_embeddings(
        indices=indices,
        records=records,
        device=device,
    )

    print(
        "===== LOAD OFFICIAL TARGET =====",
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

    expected_generation = manifest["generation"]

    if (
        raw_steps
        != expected_generation["raw_denoising_steps"]
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

    results: list[dict[str, Any]] = []

    for sample_index in indices:
        payload = common.load_payload(
            records[sample_index]
        )

        seed = int(payload["seed"])

        # Use the actual noise preserved by E0201.
        source_noise = payload[
            "source_noise"
        ].to(
            device=device,
            dtype=torch.bfloat16,
        )

        stored_noise_roundtrip = (
            common.comparison(
                source_noise.detach().cpu(),
                payload["source_noise"],
            )
        )

        conditional_dict = (
            common.move_to_device(
                embeddings[sample_index],
                device,
            )
        )

        pipeline = common.make_pipeline(
            generator,
            warped_steps,
        )

        # This controls scheduler randn_like calls
        # during the Target rollout.
        common.reset_seed(seed)

        torch.cuda.reset_peak_memory_stats(
            device
        )

        started = time.perf_counter()

        with torch.no_grad():
            output = (
                pipeline.inference_with_trajectory(
                    noise=source_noise,
                    initial_latent=None,
                    **conditional_dict,
                )
            )

        target_latent = output[0]

        torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - started

        target_cpu = (
            target_latent
            .detach()
            .cpu()
        )

        target_comparison = (
            common.comparison(
                target_cpu,
                payload["target_latent"],
            )
        )

        peak_gib = (
            torch.cuda.max_memory_allocated(
                device
            )
            / 1024**3
        )

        sample_pass = (
            stored_noise_roundtrip["exact_equal"]
            and target_comparison["exact_equal"]
        )

        record = {
            "sample_index": sample_index,
            "split": payload["split"],
            "seed": seed,
            "stored_source_noise": (
                stored_noise_roundtrip
            ),
            "target_latent": target_comparison,
            "generation_seconds": elapsed,
            "peak_allocated_gib": peak_gib,
            "status": (
                "PASS"
                if sample_pass
                else "FAIL"
            ),
        }

        results.append(record)

        print(
            f"sample={sample_index} "
            f"status={record['status']} "
            f"target_max_diff="
            f"{target_comparison['max_abs_difference']} "
            f"target_mse="
            f"{target_comparison['mse']} "
            f"seconds={elapsed:.3f}",
            flush=True,
        )

        del payload
        del source_noise
        del conditional_dict
        del pipeline
        del output
        del target_latent
        del target_cpu

        gc.collect()
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
        "experiment": (
            "E0208C0_target_reproduction_"
            "from_stored_noise"
        ),
        "indices": indices,
        "checkpoint": {
            "path": str(
                common.CHECKPOINT_PATH.resolve()
            ),
            "sha256": checkpoint_hash,
        },
        "generation": {
            "raw_denoising_steps": raw_steps,
            "warped_denoising_steps": (
                warped_steps
            ),
            "num_frames": 21,
            "num_frame_per_block": 3,
            "mcp_num_modules": 0,
            "mcp_accel_depths": 0,
            "memory_gap_blocks": 0,
            "last_step_only": True,
        },
        "restore": restore_metadata,
        "samples": results,
    }

    report_path = run_dir / "report.json"

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("report=", report_path.resolve())
    print(
        "E0208C0_TARGET_REPRODUCTION=",
        status,
    )

    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
