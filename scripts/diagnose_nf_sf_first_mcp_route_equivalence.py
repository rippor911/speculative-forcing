from __future__ import annotations

import argparse
import gc
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from scripts.eval_nf_sf_full_sequence_deployment import (
    assert_common_payload_fingerprint,
    build_conditioning,
    build_generator,
    dtype_from_arg,
    initialize_runtime,
    merge_config,
    runtime_device,
    select_eval_identity,
    validate_cli_contract,
    validate_config,
)
from utils.nf_sf_first_mcp_route_equivalence import (
    FIRST_MCP_ROUTE_EQUIVALENCE_SCHEMA,
    build_flow_match_scheduler,
    run_first_mcp_route_equivalence_audit,
)
from utils.nf_sf_full_sequence_eval import (
    FULL_SEQUENCE_FRAME_COUNT,
    FULL_SEQUENCE_FRAME_SEQ_LENGTH,
    MODE_TRAINED_MCP1,
    OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    TRAINING_CHECKPOINT_GIT_SHA,
    build_common_inputs_record,
    current_git_head,
    file_sha256,
    load_full_sequence_checkpoint_record,
    tensor_sha256,
    validate_eval_artifact_identity,
)
from utils.nf_sf_m3 import atomic_json_write, atomic_torch_save, move_tensors_to_device
from utils.nf_sf_m4 import load_m4_sample_plan
from utils.nf_sf_m5_samples import M5TeacherSampleStore
from utils.nf_sf_tensors import DEFAULT_S_MAIN, DEFAULT_S_MCP


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NF-SF First-MCP training-route vs deployment-route audit."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/self_forcing_dmd_mcp.yaml"),
    )
    parser.add_argument("--full_sequence_checkpoint", required=True, type=Path)
    parser.add_argument("--sample_plan", required=True, type=Path)
    parser.add_argument("--teacher_manifest", required=True, type=Path)
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--sample_identity", action="append", default=None)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--expected_runtime_git_sha", required=True)
    parser.add_argument(
        "--expected_training_git_sha",
        default=TRAINING_CHECKPOINT_GIT_SHA,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument("--fps", type=int, default=16)
    return parser.parse_args(argv)


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_grad_enabled(False)
    git_sha = current_git_head()
    repo_preflight = validate_cli_contract(args, git_sha=git_sha)
    device, runtime_contract = runtime_device(args.device)
    dtype = dtype_from_arg(args.dtype)
    config = merge_config(args.config)
    validate_config(config)

    sample_plan = load_m4_sample_plan(
        args.sample_plan,
        manifest_path=args.teacher_manifest,
    )
    sample_identity = select_eval_identity(
        sample_plan,
        sample_identity=args.sample_identity,
        num_samples=int(args.num_samples),
    )
    teacher_manifest_sha256 = file_sha256(args.teacher_manifest)
    full_checkpoint = load_full_sequence_checkpoint_record(
        args.full_sequence_checkpoint,
        expected_training_git_sha=str(args.expected_training_git_sha),
    )
    if full_checkpoint.payload is None:
        raise RuntimeError("full-sequence checkpoint payload missing after validation")
    artifact_identity = validate_eval_artifact_identity(
        sample_plan=sample_plan,
        teacher_manifest_sha256=teacher_manifest_sha256,
        checkpoint_payload=full_checkpoint.payload,
        selected_identity=sample_identity,
    )
    teacher_store = M5TeacherSampleStore(
        sample_plan=sample_plan,
        manifest_path=args.teacher_manifest,
        dataset_root=args.dataset_root,
        reference_checkpoint_path=None,
        expected_reference_sha256=OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    )
    with teacher_store.acquire(sample_identity) as teacher_sample:
        teacher_payload = dict(teacher_sample.payload)
        teacher_metadata = dict(teacher_sample.metadata)
        source_noise_cpu = teacher_sample.source_noise.detach().cpu()
        teacher_target_cpu = teacher_sample.target_latent.detach().cpu()

    if int(source_noise_cpu.shape[1]) != FULL_SEQUENCE_FRAME_COUNT:
        raise RuntimeError("selected teacher sample must contain 21 latent frames")
    if tuple(source_noise_cpu.shape) != tuple(teacher_target_cpu.shape):
        raise RuntimeError("teacher target_latent must match source_noise shape")
    if source_noise_cpu.dtype != dtype:
        raise RuntimeError(
            "requested dtype must match stored teacher source_noise dtype; "
            f"requested={dtype}, stored={source_noise_cpu.dtype}"
        )

    conditioning_cpu = build_conditioning(
        prompt=str(teacher_payload["prompt"]),
        device=device,
        dtype=dtype,
    )
    source_noise = source_noise_cpu.to(device=device)
    teacher_target = teacher_target_cpu.to(device=device, dtype=dtype)
    teacher_payload["source_noise"] = source_noise
    conditional_dict = move_tensors_to_device(
        conditioning_cpu,
        device=device,
        floating_dtype=dtype,
    )
    teacher_target_sha256 = tensor_sha256(teacher_target_cpu)
    common_inputs, _ = build_common_inputs_record(
        sample_identity=sample_identity,
        teacher_metadata=teacher_metadata,
        teacher_payload=teacher_payload,
        source_noise=source_noise,
        conditioning=conditional_dict,
        runtime_git_sha=git_sha,
        training_checkpoint_git_sha=str(full_checkpoint.training_git_sha),
        fps=int(args.fps),
        sample_plan_sha256=str(artifact_identity["sample_plan_sha256"]),
        teacher_manifest_sha256=str(artifact_identity["teacher_manifest_sha256"]),
        selected_validation_position=int(
            artifact_identity["selected_validation_position"]
        ),
    )
    common_inputs.update(
        {
            "audit_schema": FIRST_MCP_ROUTE_EQUIVALENCE_SCHEMA,
            "diagnostic_only": True,
            "non_deployable": True,
            "runtime_contract": runtime_contract,
            "repo_preflight": repo_preflight,
            "artifact_identity": artifact_identity,
            "config_path": str(args.config.resolve()),
            "sample_plan_path": str(args.sample_plan.resolve()),
            "teacher_manifest_path": str(args.teacher_manifest.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "frame_seq_length": FULL_SEQUENCE_FRAME_SEQ_LENGTH,
            "teacher_target_sha256": teacher_target_sha256,
            "first_mcp_route_equivalence_contract": {
                "teacher_history_chunks": [0],
                "current_chunk": 1,
                "future_chunk": 2,
                "raw_points": [1000, 999],
                "main_shift": DEFAULT_S_MAIN,
                "mcp_shift": DEFAULT_S_MCP,
                "no_solver_loop": True,
                "no_video_decode": True,
            },
        }
    )
    common_fingerprint = assert_common_payload_fingerprint(common_inputs)
    atomic_json_write(common_inputs, args.output_dir / "common_inputs.json")

    generator = build_generator(
        config=config,
        checkpoint=full_checkpoint,
        mode=MODE_TRAINED_MCP1,
        device=device,
        dtype=dtype,
    )
    try:
        result = run_first_mcp_route_equivalence_audit(
            runtime_factory=lambda: initialize_runtime(
                generator=generator,
                config=config,
                source_noise=source_noise,
            ),
            generator=generator,
            main_scheduler=build_flow_match_scheduler(
                shift=DEFAULT_S_MAIN,
                device=device,
            ),
            mcp_scheduler=build_flow_match_scheduler(
                shift=DEFAULT_S_MCP,
                device=device,
            ),
            source_noise=source_noise,
            teacher_target=teacher_target,
            teacher_payload=teacher_payload,
            conditional_dict=conditional_dict,
            checkpoint_summary=full_checkpoint.to_json(),
            common_inputs=common_inputs,
            common_inputs_fingerprint_sha256=common_fingerprint,
            runtime_git_sha=git_sha,
            training_checkpoint_git_sha=str(full_checkpoint.training_git_sha),
        )
    finally:
        generator.to("cpu")
        del generator
        gc.collect()
        torch.cuda.empty_cache()

    tensors_dir = args.output_dir / "tensors"
    tensors_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = tensors_dir / "route_equivalence_tensors.pt"
    atomic_torch_save(result.tensors, tensor_path)

    manifest = dict(result.manifest)
    manifest["output_dir"] = str(args.output_dir.resolve())
    manifest["repo_preflight"] = repo_preflight
    manifest["artifact_identity"] = artifact_identity
    manifest["tensor_archive"] = {
        "path": str(tensor_path.resolve()),
        "sha256": file_sha256(tensor_path),
    }
    atomic_json_write(manifest, args.output_dir / "first_mcp_route_equivalence.json")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_audit(args)
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "status": manifest["status"],
                "output_dir": manifest["output_dir"],
                "points": {
                    key: {
                        "raw_timestep": value["raw_timestep"],
                        "deployment_mcp_flow_mse_to_exact": value["deployment"][
                            "mcp_flow_mse_to_exact"
                        ],
                        "training_mcp_flow_mse_to_exact": value["training_route"][
                            "mcp_flow_mse_to_exact"
                        ],
                        "mcp_flow_route_mse": value["route_comparison"][
                            "mcp_flow_route_mse"
                        ],
                    }
                    for key, value in manifest["points"].items()
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
