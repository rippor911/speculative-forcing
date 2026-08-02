from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.nf_sf_m4 import (
    M4_DEFAULT_GRID_AUX_WEIGHT,
    M4_DEFAULT_OPTIMIZER_STEPS,
    M4_DEFAULT_PROBE_SEED,
    M4_DEFAULT_TIMING_WARMUP_STEPS,
    M4_DEFAULT_TRAIN_SEED,
    M4_DEFAULT_TRAIN_SUBSET_SIZE,
    M4_DEFAULT_VALIDATION_SEED,
    M4_DEFAULT_VALIDATION_SUBSET_SIZE,
    M4_PAIR_COMMANDS_SCHEMA,
    M4_PAIR_PLAN_SCHEMA,
    M4_PAIR_STATUS_SCHEMA,
    build_m4_sample_plan,
    default_m4_checkpoint_steps,
    default_m4_validation_steps,
    load_m4_sample_plan,
    m4_sample_plan_sha256,
    parse_m4_step_list,
    validate_m4_pair_contract,
    write_m4_json,
    write_m4_sample_plan,
)
from utils.nf_sf_m3 import file_sha256, validate_git_sha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NF-SF v1 M4 Frozen/Joint paired short-run orchestration."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--sample_plan", type=Path, default=None)
    parser.add_argument("--train_subset_size", type=int, default=M4_DEFAULT_TRAIN_SUBSET_SIZE)
    parser.add_argument(
        "--validation_subset_size",
        type=int,
        default=M4_DEFAULT_VALIDATION_SUBSET_SIZE,
    )
    parser.add_argument("--optimizer_steps", type=int, default=M4_DEFAULT_OPTIMIZER_STEPS)
    parser.add_argument("--train_seed", type=int, default=M4_DEFAULT_TRAIN_SEED)
    parser.add_argument("--probe_seed", type=int, default=M4_DEFAULT_PROBE_SEED)
    parser.add_argument("--validation_seed", type=int, default=M4_DEFAULT_VALIDATION_SEED)
    parser.add_argument("--timing_warmup_steps", type=int, default=M4_DEFAULT_TIMING_WARMUP_STEPS)
    parser.add_argument("--validation_steps", default=None)
    parser.add_argument("--checkpoint_steps", default=None)
    parser.add_argument("--backbone_lr", type=float, required=True)
    parser.add_argument("--patch_embedding_lr", type=float, required=True)
    parser.add_argument("--mcp_lr", type=float, required=True)
    parser.add_argument("--weight_decay", type=float, required=True)
    parser.add_argument("--mcp1_grid_aux_weight", type=float, default=M4_DEFAULT_GRID_AUX_WEIGHT)
    parser.add_argument("--dtype", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--plan_only", action="store_true")
    return parser.parse_args()


def git_head() -> str:
    return validate_git_sha(
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip(),
        name="current_git_sha",
    )


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_wrapper_args(args: argparse.Namespace) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    if args.optimizer_steps <= 0:
        raise ValueError("--optimizer_steps must be positive")
    if args.train_subset_size <= 0 or args.validation_subset_size <= 0:
        raise ValueError("M4 subset sizes must be positive")
    if args.timing_warmup_steps < 0:
        raise ValueError("--timing_warmup_steps must be non-negative")
    if args.timing_warmup_steps > args.optimizer_steps:
        raise ValueError("--timing_warmup_steps must be <= --optimizer_steps")
    if args.validation_seed < 0:
        raise ValueError("--validation_seed must be non-negative")
    validation_steps = parse_m4_step_list(
        args.validation_steps
        if args.validation_steps is not None
        else default_m4_validation_steps(args.optimizer_steps),
        optimizer_steps=args.optimizer_steps,
        name="--validation_steps",
        require_zero=True,
    )
    checkpoint_steps = parse_m4_step_list(
        args.checkpoint_steps
        if args.checkpoint_steps is not None
        else default_m4_checkpoint_steps(args.optimizer_steps),
        optimizer_steps=args.optimizer_steps,
        name="--checkpoint_steps",
        require_zero=True,
        require_final=True,
    )
    assert validation_steps is not None
    assert checkpoint_steps is not None
    return validation_steps, checkpoint_steps


def _csv(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def _resolve_python_executable(value: str | Path) -> str:
    value = str(value)
    resolved = shutil.which(value)
    if resolved is not None:
        return str(Path(resolved).resolve())
    return str(Path(value).resolve())


def _train_argv(
    *,
    args: argparse.Namespace,
    mode: str,
    output_dir: Path,
    sample_plan_path: Path,
    validation_steps: tuple[int, ...],
    checkpoint_steps: tuple[int, ...],
) -> list[str]:
    python_executable = _resolve_python_executable(args.python)
    train_script = (ROOT / "scripts" / "train_nf_sf_m3_overfit.py").resolve()
    argv = [
        python_executable,
        "-B",
        str(train_script),
        "--config",
        str(args.config.resolve()),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--output_dir",
        str(output_dir.resolve()),
        "--mode",
        mode,
        "--train_seed",
        str(int(args.train_seed)),
        "--probe_seed",
        str(int(args.probe_seed)),
        "--validation_seed",
        str(int(args.validation_seed)),
        "--optimizer_steps",
        str(int(args.optimizer_steps)),
        "--timing_warmup_steps",
        str(int(args.timing_warmup_steps)),
        "--validation_steps",
        _csv(validation_steps),
        "--checkpoint_steps",
        _csv(checkpoint_steps),
        "--log_interval",
        str(int(args.log_interval)),
        "--backbone_lr",
        str(float(args.backbone_lr)),
        "--patch_embedding_lr",
        str(float(args.patch_embedding_lr)),
        "--mcp_lr",
        str(float(args.mcp_lr)),
        "--weight_decay",
        str(float(args.weight_decay)),
        "--mcp1_grid_aux_weight",
        str(float(args.mcp1_grid_aux_weight)),
        "--dtype",
        str(args.dtype),
        "--device",
        str(args.device),
        "--m4_sample_plan",
        str(sample_plan_path.resolve()),
    ]
    if args.dataset_root is not None:
        argv.extend(["--dataset_root", str(args.dataset_root.resolve())])
    return argv


def _decode_argv(
    *,
    args: argparse.Namespace,
    mode_output_dir: Path,
    sample_plan_path: Path,
    decode_identity: str,
) -> list[str]:
    python_executable = _resolve_python_executable(args.python)
    eval_script = (ROOT / "scripts" / "eval_nf_sf_m3_overfit.py").resolve()
    argv = [
        python_executable,
        "-B",
        str(eval_script),
        "--config",
        str(args.config.resolve()),
        "--m3_checkpoint",
        str((mode_output_dir / f"checkpoint_step{int(args.optimizer_steps):06d}.pt").resolve()),
        "--initial_m3_checkpoint",
        str((mode_output_dir / "checkpoint_step000000.pt").resolve()),
        "--output_dir",
        str((mode_output_dir / "eval_fixed_decode").resolve()),
        "--device",
        str(args.device),
        "--dtype",
        str(args.dtype),
        "--manifest",
        str(args.manifest.resolve()),
        "--m4_sample_plan",
        str(sample_plan_path.resolve()),
        "--m4_decode_sample_identity",
        str(decode_identity),
        "--decode",
    ]
    if args.dataset_root is not None:
        argv.extend(["--dataset_root", str(args.dataset_root.resolve())])
    return argv


def build_pair_plan(
    *,
    args: argparse.Namespace,
    sample_plan: Mapping[str, Any],
    sample_plan_path: Path,
    validation_steps: tuple[int, ...],
    checkpoint_steps: tuple[int, ...],
    current_git_sha: str,
) -> dict[str, Any]:
    reference_checkpoint_sha = file_sha256(args.checkpoint)
    manifest_sha = file_sha256(args.manifest)
    sample_plan_sha = m4_sample_plan_sha256(sample_plan)
    frozen_output = args.output_dir / "frozen"
    joint_output = args.output_dir / "joint"
    python_executable = _resolve_python_executable(args.python)
    train_script_path = str((ROOT / "scripts" / "train_nf_sf_m3_overfit.py").resolve())
    repository_root = str(ROOT.resolve())
    shared_arguments = {
        "python_executable": python_executable,
        "train_script_path": train_script_path,
        "repository_root": repository_root,
        "subprocess_cwd": repository_root,
        "config": str(args.config.resolve()),
        "reference_checkpoint_path": str(args.checkpoint.resolve()),
        "reference_checkpoint_sha256": reference_checkpoint_sha,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": manifest_sha,
        "dataset_root": None
        if args.dataset_root is None
        else str(args.dataset_root.resolve()),
        "sample_plan_path": str(sample_plan_path.resolve()),
        "sample_plan_sha256": sample_plan_sha,
        "train_sample_identities": list(sample_plan["train_sample_identities"]),
        "validation_sample_identities": list(sample_plan["validation_sample_identities"]),
        "fixed_decode_validation_identity": str(
            sample_plan["fixed_decode_validation_identity"]
        ),
        "optimizer_steps": int(args.optimizer_steps),
        "train_seed": int(args.train_seed),
        "probe_seed": int(args.probe_seed),
        "validation_seed": int(args.validation_seed),
        "timestep_noise_contract": "M3 random joint loss local torch.Generator by train_seed",
        "depth_weights": [0.5, 0.2, 0.1],
        "mcp1_grid_aux_weight": float(args.mcp1_grid_aux_weight),
        "mcp1_grid_timesteps": [1000.0, 937.5, 833.3333129882812, 625.0],
        "optimizer": "AdamW",
        "optimizer_type": "AdamW",
        "backbone_lr": float(args.backbone_lr),
        "patch_embedding_lr": float(args.patch_embedding_lr),
        "mcp_lr": float(args.mcp_lr),
        "weight_decay": float(args.weight_decay),
        "dtype": str(args.dtype),
        "checkpoint_steps": list(checkpoint_steps),
        "validation_steps": list(validation_steps),
        "timing_warmup_steps": int(args.timing_warmup_steps),
        "output_schema": "M3 checkpoint v1 plus resolved_config.m4",
    }
    runs = {
        "frozen": {
            **shared_arguments,
            "mode": "frozen",
            "run_label": "frozen",
            "output_dir": str(frozen_output.resolve()),
            "argv": _train_argv(
                args=args,
                mode="frozen",
                output_dir=frozen_output,
                sample_plan_path=sample_plan_path,
                validation_steps=validation_steps,
                checkpoint_steps=checkpoint_steps,
            ),
        },
        "joint": {
            **shared_arguments,
            "mode": "joint",
            "run_label": "joint",
            "output_dir": str(joint_output.resolve()),
            "argv": _train_argv(
                args=args,
                mode="joint",
                output_dir=joint_output,
                sample_plan_path=sample_plan_path,
                validation_steps=validation_steps,
                checkpoint_steps=checkpoint_steps,
            ),
        },
    }
    return {
        "schema": M4_PAIR_PLAN_SCHEMA,
        "created_at_utc": utc_timestamp(),
        "git_sha": current_git_sha,
        "repository_root": repository_root,
        "subprocess_cwd": repository_root,
        "python_executable": python_executable,
        "reference_checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": reference_checkpoint_sha,
        },
        "manifest": {
            "path": str(args.manifest.resolve()),
            "sha256": manifest_sha,
        },
        "sample_plan_sha256": sample_plan_sha,
        "sample_plan_path": str(sample_plan_path.resolve()),
        "shared_arguments": shared_arguments,
        "runs": runs,
        "decode_commands": {
            "frozen": _decode_argv(
                args=args,
                mode_output_dir=frozen_output,
                sample_plan_path=sample_plan_path,
                decode_identity=str(sample_plan["fixed_decode_validation_identity"]),
            ),
            "joint": _decode_argv(
                args=args,
                mode_output_dir=joint_output,
                sample_plan_path=sample_plan_path,
                decode_identity=str(sample_plan["fixed_decode_validation_identity"]),
            ),
        },
    }


PLAN_ONLY_OUTPUT_FILES = {
    "m4_sample_plan.json",
    "m4_pair_plan.json",
    "m4_pair_commands.json",
    "m4_pair_status.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _pair_commands(pair_plan: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": M4_PAIR_COMMANDS_SCHEMA,
        "contract": contract,
        "commands": {
            "frozen": pair_plan["runs"]["frozen"]["argv"],
            "joint": pair_plan["runs"]["joint"]["argv"],
        },
        "decode_commands": pair_plan["decode_commands"],
    }


def _load_existing_plan_only_output(output_dir: Path) -> dict[str, dict[str, Any]] | None:
    if not output_dir.exists():
        return None
    if not output_dir.is_dir():
        raise FileExistsError(f"--output_dir exists and is not a directory: {output_dir}")
    entries = {entry.name for entry in output_dir.iterdir()}
    unknown = sorted(entries - PLAN_ONLY_OUTPUT_FILES)
    if unknown:
        raise FileExistsError(
            f"--output_dir contains files outside a reusable PLAN_ONLY plan: {unknown}"
        )
    missing = sorted(PLAN_ONLY_OUTPUT_FILES - entries)
    if missing:
        raise FileExistsError(
            f"--output_dir is not a complete reusable PLAN_ONLY plan; missing {missing}"
        )
    status = _read_json(output_dir / "m4_pair_status.json")
    if status.get("status") != "PLAN_ONLY":
        raise FileExistsError(
            f"--output_dir already exists but status is not PLAN_ONLY: {output_dir}"
        )
    if (output_dir / "frozen").exists() or (output_dir / "joint").exists():
        raise FileExistsError("--output_dir contains Frozen/Joint training artifacts")
    return {
        "sample_plan": _read_json(output_dir / "m4_sample_plan.json"),
        "pair_plan": _read_json(output_dir / "m4_pair_plan.json"),
        "commands": _read_json(output_dir / "m4_pair_commands.json"),
        "status": status,
    }


def _assert_reusable_plan_only_matches(
    *,
    existing: dict[str, dict[str, Any]],
    sample_plan: dict[str, Any],
    pair_plan: dict[str, Any],
    commands: dict[str, Any],
) -> None:
    if existing["sample_plan"] != sample_plan:
        raise RuntimeError("existing PLAN_ONLY sample plan differs from requested plan")
    comparable_pair_plan = dict(pair_plan)
    comparable_pair_plan["created_at_utc"] = existing["pair_plan"].get("created_at_utc")
    if existing["pair_plan"] != comparable_pair_plan:
        raise RuntimeError("existing PLAN_ONLY pair plan differs from requested plan")
    if existing["commands"] != commands:
        raise RuntimeError("existing PLAN_ONLY commands differ from requested plan")


def _base_status(*, status: str, plan_only: bool) -> dict[str, Any]:
    return {
        "schema": M4_PAIR_STATUS_SCHEMA,
        "status": status,
        "started_at_utc": utc_timestamp(),
        "ended_at_utc": None,
        "plan_only": bool(plan_only),
        "subprocess_cwd": str(ROOT.resolve()),
        "current_stage": None,
        "runs": {
            "frozen": {"exit_code": None, "started_at_utc": None, "ended_at_utc": None},
            "joint": {"exit_code": None, "started_at_utc": None, "ended_at_utc": None},
        },
    }


def _write_status(status: dict[str, Any], output_dir: Path) -> None:
    write_m4_json(status, output_dir / "m4_pair_status.json")


def run_pair(args: argparse.Namespace) -> int:
    validation_steps, checkpoint_steps = validate_wrapper_args(args)
    current_git_sha = git_head()
    output_sample_plan_path = args.output_dir / "m4_sample_plan.json"
    if args.sample_plan is None:
        sample_plan = build_m4_sample_plan(
            manifest_path=args.manifest,
            train_subset_size=args.train_subset_size,
            validation_subset_size=args.validation_subset_size,
            dataset_root=args.dataset_root,
        )
    else:
        sample_plan = load_m4_sample_plan(args.sample_plan, manifest_path=args.manifest)
    pair_plan = build_pair_plan(
        args=args,
        sample_plan=sample_plan,
        sample_plan_path=output_sample_plan_path,
        validation_steps=validation_steps,
        checkpoint_steps=checkpoint_steps,
        current_git_sha=current_git_sha,
    )
    contract = validate_m4_pair_contract(pair_plan)
    commands = _pair_commands(pair_plan, contract)
    existing = _load_existing_plan_only_output(args.output_dir)
    if existing is not None:
        _assert_reusable_plan_only_matches(
            existing=existing,
            sample_plan=sample_plan,
            pair_plan=pair_plan,
            commands=commands,
        )
        pair_plan = existing["pair_plan"]
        commands = existing["commands"]
        if args.plan_only:
            status = existing["status"]
            print(json.dumps(status, indent=2), flush=True)
            return 0
    else:
        args.output_dir.mkdir(parents=False, exist_ok=False)
        write_m4_sample_plan(sample_plan, output_sample_plan_path)
        write_m4_json(pair_plan, args.output_dir / "m4_pair_plan.json")
        write_m4_json(commands, args.output_dir / "m4_pair_commands.json")

    if args.plan_only:
        status = _base_status(status="PLAN_ONLY", plan_only=True)
        status["ended_at_utc"] = utc_timestamp()
        _write_status(status, args.output_dir)
        print(json.dumps(status, indent=2), flush=True)
        return 0

    status = _base_status(status="RUNNING", plan_only=False)
    _write_status(status, args.output_dir)
    try:
        for mode in ("frozen", "joint"):
            status["current_stage"] = mode
            status["runs"][mode]["started_at_utc"] = utc_timestamp()
            _write_status(status, args.output_dir)
            result = subprocess.run(
                pair_plan["runs"][mode]["argv"],
                shell=False,
                check=False,
                cwd=str(ROOT.resolve()),
            )
            status["runs"][mode]["exit_code"] = int(result.returncode)
            status["runs"][mode]["ended_at_utc"] = utc_timestamp()
            if result.returncode != 0:
                status["status"] = "FROZEN_FAILED" if mode == "frozen" else "JOINT_FAILED"
                status["ended_at_utc"] = utc_timestamp()
                _write_status(status, args.output_dir)
                print(json.dumps(status, indent=2), flush=True)
                return int(result.returncode)
            _write_status(status, args.output_dir)
        status["status"] = "COMPLETED"
        status["current_stage"] = None
        status["ended_at_utc"] = utc_timestamp()
        _write_status(status, args.output_dir)
        print(json.dumps(status, indent=2), flush=True)
        return 0
    except KeyboardInterrupt:
        status["status"] = "INTERRUPTED"
        status["ended_at_utc"] = utc_timestamp()
        _write_status(status, args.output_dir)
        print(json.dumps(status, indent=2), flush=True)
        return 130
    except Exception as exc:
        status["status"] = "FAILED"
        status["ended_at_utc"] = utc_timestamp()
        status["exception_type"] = type(exc).__name__
        status["exception"] = str(exc)
        _write_status(status, args.output_dir)
        raise


def main() -> None:
    raise SystemExit(run_pair(parse_args()))


if __name__ == "__main__":
    main()
