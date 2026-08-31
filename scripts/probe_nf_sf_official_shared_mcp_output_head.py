from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.nf_sf_m3 import file_sha256, move_tensors_to_device
from utils.nf_sf_m4 import load_m4_sample_plan
from utils.nf_sf_tensors import (
    DEFAULT_S_MAIN,
    DEFAULT_S_MCP,
    FULL_SEQUENCE_DEPTHS,
    FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION,
    make_generator,
)
from utils.nf_sf_training import (
    FULL_SEQUENCE_DEPTH_WEIGHTS,
    OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    audit_nf_sf_full_sequence_gradients,
    configure_nf_sf_full_sequence_optimizer_plan,
    nf_sf_full_sequence_train_cursor,
    prepare_nf_sf_full_sequence_noisy_batch,
    run_nf_sf_full_sequence_forward_loss,
)


PROBE_SCHEMA = "nf_sf_official_shared_mcp_output_head_gpu_probe_v1"
PROBE_ARTIFACT_FILENAME = "official_shared_mcp_output_head_gpu_probe.json"
PROBE_PASS_LABEL = "OFFICIAL_SHARED_MCP_OUTPUT_HEAD_GPU_PROBE_PASS"
PROBE_FAIL_FRESH_INIT = "FAIL_FRESH_NO_MCP_INIT"
PROBE_FAIL_NONFINITE = "FAIL_NONFINITE"
PROBE_FAIL_GRADIENT = "FAIL_GRADIENT_CONTRACT"
PROBE_FAIL_PARAMETER_MUTATION = "FAIL_PARAMETER_FINGERPRINT_CHANGED"
PROBE_FAIL_ROUTE = "FAIL_SHARED_HEAD_ROUTE"
PROBE_DEFAULT_STEP = 1
PROBE_DEFAULT_SEED = 20260831

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProbeContractError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class BackwardPassReport:
    pass_kind: str
    loss_report: dict[str, Any]
    gradient_report: dict[str, dict[str, Any]]
    gradient_gate: dict[str, Any]
    nf_sf_gradient_audit: dict[str, dict[str, Any]]


def _trainer_helpers() -> Mapping[str, Any]:
    from scripts import train_nf_sf_full_sequence_next_forcing as trainer

    return {
        "assert_finite_loss": trainer.assert_finite_loss,
        "build_fresh_generator": trainer.build_fresh_generator,
        "dtype_from_arg": trainer.dtype_from_arg,
        "make_flow_scheduler": trainer.make_flow_scheduler,
        "memory_snapshot": trainer.memory_snapshot,
        "merge_config": trainer.merge_config,
        "prepare_output_dir": trainer.prepare_output_dir,
        "target_latent_from_sample": trainer.target_latent_from_sample,
        "validate_sample_plan_contract": trainer.validate_sample_plan_contract,
        "validate_store_identity_order": trainer.validate_store_identity_order,
        "write_m4_json": trainer.write_m4_json,
    }


def validate_git_sha(value: str, *, name: str) -> str:
    text = str(value).strip().lower()
    if not _GIT_SHA_RE.fullmatch(text):
        raise ValueError(f"{name} must be a lowercase git SHA")
    return text


def validate_sha256(value: str, *, name: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{name} must be a lowercase SHA256 hex string")
    return text


def current_git_head() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
    return validate_git_sha(value, name="runtime_git_sha")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NF-SF official shared MCP output-head no-step GPU probe."
    )
    parser.add_argument("--execute_real_probe", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/self_forcing_dmd_mcp.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/self_forcing_dmd.pt"),
    )
    parser.add_argument(
        "--expected_checkpoint_sha256",
        default=OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    )
    parser.add_argument("--sample_plan", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset_root", type=Path)
    parser.add_argument("--conditionals_artifact", type=Path)
    parser.add_argument("--sample_identity")
    parser.add_argument("--probe_global_step", type=int, default=PROBE_DEFAULT_STEP)
    parser.add_argument("--probe_seed", type=int, default=PROBE_DEFAULT_SEED)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_runtime_git_sha")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    return parser.parse_args(argv)


def _require_real_probe_args(args: argparse.Namespace) -> None:
    missing = [
        name
        for name in ("sample_plan", "manifest", "dataset_root", "conditionals_artifact")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(
            "--execute_real_probe requires: "
            + ", ".join(f"--{name}" for name in missing)
        )
    if args.expected_runtime_git_sha is None:
        raise ValueError("--execute_real_probe requires --expected_runtime_git_sha")


def _validate_real_static_contract(args: argparse.Namespace, config: Any) -> None:
    if Path(args.config).resolve() != (ROOT / "configs" / "self_forcing_dmd_mcp.yaml").resolve():
        raise RuntimeError("config path must be configs/self_forcing_dmd_mcp.yaml")
    if str(args.device) != "cuda:0":
        raise RuntimeError("real probe requires --device cuda:0")
    if str(args.dtype) != "bf16":
        raise RuntimeError("real probe requires --dtype bf16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required only when --execute_real_probe is set")
    if int(args.probe_global_step) <= 0:
        raise RuntimeError("probe_global_step must be positive")
    if int(getattr(config, "num_frame_per_block", 0)) != 3:
        raise RuntimeError("config.num_frame_per_block must be 3")
    if not bool(getattr(config, "gradient_checkpointing", False)):
        raise RuntimeError("config.gradient_checkpointing must be true")
    expected = validate_sha256(
        str(args.expected_checkpoint_sha256),
        name="--expected_checkpoint_sha256",
    )
    actual = file_sha256(args.checkpoint)
    if actual != expected:
        raise RuntimeError("fresh no-MCP checkpoint SHA256 mismatch")


def _tensor_shape(tensor: torch.Tensor) -> list[int]:
    return [int(dim) for dim in tensor.shape]


def _scalar(tensor: torch.Tensor) -> float:
    return float(tensor.detach().float().item())


def _is_finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach().float()).all().item())


def weighted_mcp_only_loss(
    losses: Any,
    *,
    depth_weights: Sequence[float] = FULL_SEQUENCE_DEPTH_WEIGHTS,
) -> torch.Tensor:
    mcp_losses = tuple(losses.mcp_depth_losses)
    weights = tuple(float(weight) for weight in depth_weights)
    if len(mcp_losses) != len(FULL_SEQUENCE_DEPTHS):
        raise ValueError("MCP-only probe requires MCP1/2/3 losses")
    if len(weights) != len(mcp_losses):
        raise ValueError("depth_weights must align with MCP depth losses")
    total = mcp_losses[0].new_zeros(())
    for weight, loss in zip(weights, mcp_losses):
        total = total + float(weight) * loss
    return total


def loss_report_for_pass(
    losses: Any,
    *,
    backward_loss: torch.Tensor,
    pass_kind: str,
) -> dict[str, Any]:
    mcp_losses = tuple(losses.mcp_depth_losses)
    return {
        "pass_kind": str(pass_kind),
        "main_loss": _scalar(losses.main_loss),
        "mcp_depth_losses": [_scalar(loss) for loss in mcp_losses],
        "full_joint_total_loss": _scalar(losses.total_loss),
        "backward_loss": _scalar(backward_loss),
        "depth_weights": [float(weight) for weight in FULL_SEQUENCE_DEPTH_WEIGHTS],
        "main_loss_in_backward": str(pass_kind) == "full_joint",
        "mcp_losses_in_backward": True,
        "finite": bool(
            _is_finite_tensor(losses.main_loss)
            and _is_finite_tensor(losses.total_loss)
            and _is_finite_tensor(backward_loss)
            and all(_is_finite_tensor(loss) for loss in mcp_losses)
        ),
    }


def _named_main_final_head_parameters(
    generator: Any,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    head = getattr(getattr(generator, "model", None), "head", None)
    if head is None or not hasattr(head, "named_parameters"):
        return ()
    return tuple((f"model.head.{name}", param) for name, param in head.named_parameters())


def _named_module_parameters(
    module: Any,
    *,
    prefix: str,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    if module is None or not hasattr(module, "named_parameters"):
        return ()
    return tuple((f"{prefix}.{name}", param) for name, param in module.named_parameters())


def _mcp_depth_modules(generator: Any) -> tuple[Any, ...]:
    modules = getattr(getattr(generator, "mcp", None), "mcp_modules", None)
    if modules is None:
        return ()
    return tuple(modules)


def _grad_group_summary(
    group_name: str,
    named_params: Sequence[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    trainable = [(name, param) for name, param in named_params if param.requires_grad]
    grad_tensors = 0
    active_grad_tensors = 0
    missing = 0
    finite = True
    sq_norm = 0.0
    max_abs = 0.0
    parameter_records = []
    for name, param in named_params:
        record = {
            "name": str(name),
            "shape": _tensor_shape(param),
            "requires_grad": bool(param.requires_grad),
            "grad_present": False,
            "grad_nonzero": False,
            "grad_norm": None,
            "grad_finite": None,
        }
        if param.requires_grad:
            if param.grad is None:
                missing += 1
            else:
                grad = param.grad.detach().float()
                grad_tensors += 1
                grad_finite = bool(torch.isfinite(grad).all().item())
                finite = finite and grad_finite
                grad_sq = float(grad.square().sum().item())
                grad_norm = grad_sq ** 0.5
                grad_max_abs = float(grad.abs().max().item())
                sq_norm += grad_sq
                max_abs = max(max_abs, grad_max_abs)
                grad_nonzero = grad_norm > 0.0
                if grad_nonzero:
                    active_grad_tensors += 1
                record.update(
                    {
                        "grad_present": True,
                        "grad_nonzero": bool(grad_nonzero),
                        "grad_norm": float(grad_norm),
                        "grad_finite": grad_finite,
                    }
                )
        parameter_records.append(record)
    return {
        "group": str(group_name),
        "parameter_tensors": int(len(named_params)),
        "trainable_tensors": int(len(trainable)),
        "grad_tensors": int(grad_tensors),
        "active_grad_tensors": int(active_grad_tensors),
        "missing_grad_tensors": int(missing),
        "all_finite": bool(finite),
        "aggregate_grad_norm": float(sq_norm ** 0.5),
        "max_abs_grad": float(max_abs),
        "parameters": parameter_records,
    }


def gradient_report_for_probe(generator: Any) -> dict[str, dict[str, Any]]:
    report = {
        "main_final_head": _grad_group_summary(
            "main_final_head",
            _named_main_final_head_parameters(generator),
        ),
        "mcp_fusion": _grad_group_summary(
            "mcp_fusion",
            tuple(
                (f"mcp.fusion.{name}", param)
                for name, param in generator.mcp.fusion.named_parameters()
            ),
        ),
    }
    for depth, module in enumerate(_mcp_depth_modules(generator)[:3], start=1):
        report[f"mcp_depth{depth}_projection"] = _grad_group_summary(
            f"mcp_depth{depth}_projection",
            _named_module_parameters(module.proj, prefix=f"mcp.mcp_modules.{depth - 1}.proj"),
        )
        report[f"mcp_depth{depth}_blocks"] = _grad_group_summary(
            f"mcp_depth{depth}_blocks",
            _named_module_parameters(
                module.blocks,
                prefix=f"mcp.mcp_modules.{depth - 1}.blocks",
            ),
        )
        report[f"dormant_mcp_depth{depth}_independent_head"] = _grad_group_summary(
            f"dormant_mcp_depth{depth}_independent_head",
            _named_module_parameters(
                module.head,
                prefix=f"mcp.mcp_modules.{depth - 1}.head",
            ),
        )
    return report


def _positive_grad(item: Mapping[str, Any]) -> bool:
    return (
        int(item.get("active_grad_tensors", 0)) > 0
        and bool(item.get("all_finite", False))
        and float(item.get("aggregate_grad_norm", 0.0)) > 0.0
    )


def _no_active_grad(item: Mapping[str, Any]) -> bool:
    return (
        int(item.get("active_grad_tensors", 0)) == 0
        and float(item.get("aggregate_grad_norm", 0.0)) == 0.0
        and bool(item.get("all_finite", True))
    )


def validate_gradient_contract(
    report: Mapping[str, Mapping[str, Any]],
    *,
    pass_kind: str,
) -> dict[str, Any]:
    positive = ["main_final_head", "mcp_fusion"]
    for depth in FULL_SEQUENCE_DEPTHS:
        positive.append(f"mcp_depth{int(depth)}_projection")
        positive.append(f"mcp_depth{int(depth)}_blocks")
    no_grad = [
        f"dormant_mcp_depth{int(depth)}_independent_head"
        for depth in FULL_SEQUENCE_DEPTHS
    ]

    failures = []
    for name in positive:
        item = report.get(name)
        if item is None or not _positive_grad(item):
            failures.append(f"{name}:expected_positive_finite_grad")
    for name in no_grad:
        item = report.get(name)
        if item is None or not _no_active_grad(item):
            failures.append(f"{name}:expected_no_active_grad")
    if failures:
        raise ProbeContractError(
            PROBE_FAIL_GRADIENT,
            f"{pass_kind}: " + ", ".join(failures),
        )
    return {
        "status": "PASS",
        "pass_kind": str(pass_kind),
        "positive_gradient_groups": positive,
        "no_active_gradient_groups": no_grad,
    }


def _sha256_tensor(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(str(list(cpu.shape)).encode("utf-8"))
    h.update(cpu.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def parameter_fingerprint_report(module: torch.nn.Module) -> dict[str, Any]:
    entries = []
    for name, tensor in module.state_dict().items():
        if torch.is_tensor(tensor):
            entries.append(
                {
                    "name": str(name),
                    "sha256": _sha256_tensor(tensor),
                    "shape": _tensor_shape(tensor),
                    "dtype": str(tensor.dtype),
                }
            )
    digest = hashlib.sha256(
        "\n".join(f"{item['name']}:{item['sha256']}" for item in entries).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "schema": "nf_sf_parameter_value_fingerprint_v1",
        "tensor_count": int(len(entries)),
        "aggregate_sha256": digest,
        "entries": entries,
    }


def _fingerprint_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": str(report["schema"]),
        "tensor_count": int(report["tensor_count"]),
        "aggregate_sha256": str(report["aggregate_sha256"]),
    }


def compare_parameter_fingerprints(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    unchanged = str(before.get("aggregate_sha256")) == str(after.get("aggregate_sha256"))
    return {
        "unchanged": bool(unchanged),
        "before": _fingerprint_summary(before),
        "after": _fingerprint_summary(after),
    }


def build_parameter_safety_report(
    *,
    before: Mapping[str, Any],
    after_mcp_only_backward: Mapping[str, Any],
    after_full_joint_backward: Mapping[str, Any],
) -> dict[str, Any]:
    mcp_only = compare_parameter_fingerprints(before, after_mcp_only_backward)
    full_joint = compare_parameter_fingerprints(before, after_full_joint_backward)
    return {
        "optimizer_constructed": False,
        "optimizer_step_executed": False,
        "checkpoint_written": False,
        "mcp_only_backward_parameter_fingerprint_unchanged": bool(
            mcp_only["unchanged"]
        ),
        "full_joint_backward_parameter_fingerprint_unchanged": bool(
            full_joint["unchanged"]
        ),
        "parameter_fingerprint_unchanged": bool(
            mcp_only["unchanged"] and full_joint["unchanged"]
        ),
        "mcp_only_backward": mcp_only,
        "full_joint_backward": full_joint,
    }


class SharedHeadRouteRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.phase = "unscoped"
        self._module: Any | None = None
        self._original_forward: Any | None = None
        self._generator: Any | None = None

    def install(self, generator: Any) -> None:
        if getattr(generator, "mcp", None) is None:
            raise RuntimeError("shared-head route recorder requires generator.mcp")
        self._generator = generator
        self._module = generator.mcp
        self._original_forward = generator.mcp.forward

        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            main_output_head = kwargs.get("main_output_head")
            official_shared = bool(kwargs.get("official_shared_mcp_output_head", False))
            self.calls.append(
                {
                    "phase": str(self.phase),
                    "call_index": int(len(self.calls)),
                    "official_shared_mcp_output_head": official_shared,
                    "main_output_head_is_model_head": main_output_head
                    is generator.model.head,
                    "main_output_head_id": None
                    if main_output_head is None
                    else id(main_output_head),
                    "model_head_id": id(generator.model.head),
                    "depth_count": len(tuple(kwargs.get("future_embeds") or ())),
                }
            )
            return self._original_forward(*args, **kwargs)

        generator.mcp.forward = wrapped_forward

    def remove(self) -> None:
        if self._module is not None and self._original_forward is not None:
            self._module.forward = self._original_forward
        self._module = None
        self._original_forward = None
        self._generator = None


def validate_route_report(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not calls:
        raise ProbeContractError(PROBE_FAIL_ROUTE, "no MCP route calls were recorded")
    failures = []
    for call in calls:
        if not bool(call.get("official_shared_mcp_output_head", False)):
            failures.append(f"call{call.get('call_index')}:flag_false")
        if not bool(call.get("main_output_head_is_model_head", False)):
            failures.append(f"call{call.get('call_index')}:head_identity_mismatch")
    if failures:
        raise ProbeContractError(PROBE_FAIL_ROUTE, ", ".join(failures))
    return {
        "status": "PASS",
        "call_count": int(len(calls)),
        "phases": sorted({str(call.get("phase")) for call in calls}),
    }


def _trainable_parameter_count(module: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


def _optimizer_plan_summary(plan: Any) -> dict[str, Any]:
    return {
        "optimizer_constructed": False,
        "mode": plan.mode,
        "groups": [
            {
                "name": audit.name,
                "tensor_count": int(audit.tensor_count),
                "trainable_parameter_count": int(audit.trainable_parameter_count),
                "requires_grad": bool(audit.requires_grad),
                "in_optimizer": bool(audit.in_optimizer),
            }
            for audit in plan.audits
        ],
    }


def _validate_nf_sf_audit(report: Mapping[str, Mapping[str, Any]], *, pass_kind: str) -> None:
    failures = [name for name, item in report.items() if not bool(item.get("pass"))]
    if failures:
        raise ProbeContractError(
            PROBE_FAIL_GRADIENT,
            f"{pass_kind}: NF-SF gradient audit failed: {failures}",
        )


def run_backward_pass(
    generator: torch.nn.Module,
    *,
    conditional_dict: Mapping[str, Any],
    noisy_batch: Any,
    recorder: SharedHeadRouteRecorder,
    pass_kind: str,
) -> BackwardPassReport:
    generator.zero_grad(set_to_none=True)
    recorder.phase = str(pass_kind)
    result = run_nf_sf_full_sequence_forward_loss(
        generator,
        conditional_dict=dict(conditional_dict),
        noisy_batch=noisy_batch,
        objective_mode="next_forcing_full",
    )
    if pass_kind == "mcp_only":
        backward_loss = weighted_mcp_only_loss(result.losses)
    elif pass_kind == "full_joint":
        backward_loss = result.losses.total_loss
    else:
        raise ValueError(f"unknown pass_kind: {pass_kind!r}")
    loss_report = loss_report_for_pass(
        result.losses,
        backward_loss=backward_loss,
        pass_kind=pass_kind,
    )
    if not bool(loss_report["finite"]):
        raise ProbeContractError(
            PROBE_FAIL_NONFINITE,
            f"{pass_kind}: non-finite forward loss",
        )
    backward_loss.backward()
    gradient_report = gradient_report_for_probe(generator)
    gradient_gate = validate_gradient_contract(gradient_report, pass_kind=pass_kind)
    nf_sf_audit = audit_nf_sf_full_sequence_gradients(
        generator,
        objective_mode="next_forcing_full",
    )
    if pass_kind == "full_joint":
        _validate_nf_sf_audit(nf_sf_audit, pass_kind=pass_kind)
    return BackwardPassReport(
        pass_kind=str(pass_kind),
        loss_report=loss_report,
        gradient_report=gradient_report,
        gradient_gate=gradient_gate,
        nf_sf_gradient_audit=nf_sf_audit,
    )


def build_probe_artifact(
    *,
    status: str,
    runtime_git_sha: str,
    checkpoint_report: Mapping[str, Any],
    route_calls: Sequence[Mapping[str, Any]],
    route_gate: Mapping[str, Any] | None,
    mcp_only: BackwardPassReport | None,
    full_joint: BackwardPassReport | None,
    parameter_safety_report: Mapping[str, Any],
    optimizer_plan: Mapping[str, Any] | None,
    sample_identity: str | None,
    sample_cursor: Mapping[str, Any] | None,
    trainable_parameter_count: int,
    memory_report: Mapping[str, Any] | None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": PROBE_SCHEMA,
        "status": PROBE_PASS_LABEL if status == "PASS" else str(status),
        "diagnostic_only": True,
        "official_shared_mcp_output_head": True,
        "fresh_parent": "official_self_forcing_no_mcp_checkpoint",
        "objective_mode": "next_forcing_full",
        "mcp_only_backward_loss_formula": "0.5*MCP1 + 0.2*MCP2 + 0.1*MCP3",
        "full_joint_backward_loss_formula": "Main + 0.5*MCP1 + 0.2*MCP2 + 0.1*MCP3",
        "training_weight_implemented": False,
        "optimizer_constructed": False,
        "optimizer_step_executed": False,
        "checkpoint_written": False,
        "runtime_git_sha": validate_git_sha(runtime_git_sha, name="runtime_git_sha"),
        "checkpoint_report": dict(checkpoint_report),
        "main_shift": float(DEFAULT_S_MAIN),
        "mcp_shift": float(DEFAULT_S_MCP),
        "depth_weights": [float(weight) for weight in FULL_SEQUENCE_DEPTH_WEIGHTS],
        "rng_draw_order_version": FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION,
        "sample_identity": sample_identity,
        "sample_cursor": dict(sample_cursor) if sample_cursor is not None else None,
        "trainable_parameter_count": int(trainable_parameter_count),
        "optimizer_plan": dict(optimizer_plan) if optimizer_plan is not None else None,
        "shared_head_route_calls": [dict(call) for call in route_calls],
        "shared_head_route_gate": dict(route_gate) if route_gate is not None else None,
        "mcp_only_backward": None
        if mcp_only is None
        else {
            "loss_report": mcp_only.loss_report,
            "gradient_report": mcp_only.gradient_report,
            "gradient_gate": mcp_only.gradient_gate,
            "nf_sf_gradient_audit": mcp_only.nf_sf_gradient_audit,
        },
        "full_joint_backward": None
        if full_joint is None
        else {
            "loss_report": full_joint.loss_report,
            "gradient_report": full_joint.gradient_report,
            "gradient_gate": full_joint.gradient_gate,
            "nf_sf_gradient_audit": full_joint.nf_sf_gradient_audit,
        },
        "parameter_safety_report": dict(parameter_safety_report),
        "memory_report": dict(memory_report) if memory_report is not None else {},
        "error": dict(error) if error is not None else None,
    }


def validate_probe_artifact_schema(report: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "status",
        "diagnostic_only",
        "official_shared_mcp_output_head",
        "fresh_parent",
        "mcp_only_backward",
        "full_joint_backward",
        "parameter_safety_report",
        "shared_head_route_gate",
        "optimizer_step_executed",
        "checkpoint_written",
    }
    missing = required - set(report.keys())
    if missing:
        raise ValueError(f"probe artifact missing fields: {sorted(missing)}")
    if report["schema"] != PROBE_SCHEMA:
        raise ValueError("probe artifact schema mismatch")
    if report["official_shared_mcp_output_head"] is not True:
        raise ValueError("probe artifact must set official_shared_mcp_output_head=true")
    if report["fresh_parent"] != "official_self_forcing_no_mcp_checkpoint":
        raise ValueError("probe artifact must use the no-MCP Self-Forcing parent")
    if report["optimizer_step_executed"] is not False:
        raise ValueError("probe artifact must prove optimizer_step_executed=false")
    if report["checkpoint_written"] is not False:
        raise ValueError("probe artifact must prove checkpoint_written=false")
    if report["status"] == PROBE_PASS_LABEL:
        if report["mcp_only_backward"] is None:
            raise ValueError("PASS artifact must include MCP-only backward")
        if report["full_joint_backward"] is None:
            raise ValueError("PASS artifact must include full-joint backward")
        if report["shared_head_route_gate"].get("status") != "PASS":
            raise ValueError("PASS artifact must prove shared-head route identity")
        if not bool(
            report["parameter_safety_report"].get(
                "parameter_fingerprint_unchanged",
                False,
            )
        ):
            raise ValueError("PASS artifact must prove parameter fingerprint unchanged")
    return {"status": "PASS"}


def dry_run_artifact(
    *,
    output_dir: Path,
    runtime_git_sha: str = "0" * 40,
) -> dict[str, Any]:
    report = build_probe_artifact(
        status="DRY_RUN",
        runtime_git_sha=runtime_git_sha,
        checkpoint_report={
            "checkpoint_path": "checkpoints/self_forcing_dmd.pt",
            "expected_checkpoint_sha256": OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
            "checkpoint_contains_mcp_keys": False,
            "fresh_mcp_initialization_required": True,
        },
        route_calls=[],
        route_gate=None,
        mcp_only=None,
        full_joint=None,
        parameter_safety_report={
            "optimizer_constructed": False,
            "optimizer_step_executed": False,
            "checkpoint_written": False,
            "parameter_fingerprint_unchanged": True,
        },
        optimizer_plan=None,
        sample_identity=None,
        sample_cursor=None,
        trainable_parameter_count=0,
        memory_report=None,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / PROBE_ARTIFACT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def run_real_probe(args: argparse.Namespace) -> dict[str, Any]:
    _require_real_probe_args(args)
    helpers = _trainer_helpers()
    config = helpers["merge_config"](args.config)
    _validate_real_static_contract(args, config)
    runtime_git = current_git_head()
    expected_runtime_git = validate_git_sha(
        str(args.expected_runtime_git_sha),
        name="--expected_runtime_git_sha",
    )
    if runtime_git != expected_runtime_git:
        raise RuntimeError("runtime git SHA mismatch")
    helpers["prepare_output_dir"](args.output_dir, resume=False)

    from utils.nf_sf_m5_conditionals import M5ConditionalArtifactStore
    from utils.nf_sf_m5_samples import M5TeacherSampleStore

    device = torch.device(args.device)
    dtype = helpers["dtype_from_arg"](args.dtype)
    sample_plan = load_m4_sample_plan(args.sample_plan, manifest_path=args.manifest)
    helpers["validate_sample_plan_contract"](sample_plan)
    conditional_store = M5ConditionalArtifactStore(
        artifact_dir=args.conditionals_artifact,
        sample_plan=sample_plan,
    )
    teacher_store = M5TeacherSampleStore(
        sample_plan=sample_plan,
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        reference_checkpoint_path=None,
        expected_reference_sha256=OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    )
    helpers["validate_store_identity_order"](
        sample_plan=sample_plan,
        teacher_store=teacher_store,
        conditional_store=conditional_store,
    )

    memory = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    memory["before_model"] = helpers["memory_snapshot"]("before_model", device)
    generator = helpers["build_fresh_generator"](
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
        dtype=dtype,
        official_shared_mcp_output_head=True,
    )
    if not bool(getattr(generator, "official_shared_mcp_output_head", False)):
        raise ProbeContractError(
            PROBE_FAIL_FRESH_INIT,
            "generator did not enable official shared MCP output-head routing",
        )
    plan = configure_nf_sf_full_sequence_optimizer_plan(
        generator,
        objective_mode="next_forcing_full",
        group_lrs={
            "backbone": 1.0,
            "patch_embedding": 1.0,
            "mcp": 1.0,
            "mcp_fusion": 1.0,
        },
    )
    optimizer_plan = _optimizer_plan_summary(plan)
    trainable_count = _trainable_parameter_count(generator)
    memory["after_model"] = helpers["memory_snapshot"]("after_model", device)

    scheduler_main = helpers["make_flow_scheduler"](DEFAULT_S_MAIN)
    scheduler_mcp = helpers["make_flow_scheduler"](DEFAULT_S_MCP)
    train_rng = make_generator(int(args.probe_seed), device)
    cursor = nf_sf_full_sequence_train_cursor(int(args.probe_global_step))
    identity = (
        str(args.sample_identity)
        if args.sample_identity is not None
        else teacher_store.train_identity_for_step(int(args.probe_global_step))
    )

    recorder = SharedHeadRouteRecorder()
    status = "PASS"
    error = None
    route_gate = None
    mcp_only_report = None
    full_joint_report = None
    parameter_before = None
    parameter_after_mcp = None
    parameter_after_full = None
    try:
        with teacher_store.acquire(identity) as sample:
            with conditional_store.acquire(identity) as conditional_cpu:
                clean_target = helpers["target_latent_from_sample"](sample).to(
                    device=device,
                    dtype=dtype,
                )
                conditional = move_tensors_to_device(
                    conditional_cpu,
                    device=device,
                    floating_dtype=dtype,
                )
                noisy_batch = prepare_nf_sf_full_sequence_noisy_batch(
                    clean_target,
                    scheduler_main=scheduler_main,
                    scheduler_mcp=scheduler_mcp,
                    rng=train_rng,
                )
                parameter_before = parameter_fingerprint_report(generator)
                recorder.install(generator)
                mcp_only_report = run_backward_pass(
                    generator,
                    conditional_dict=conditional,
                    noisy_batch=noisy_batch,
                    recorder=recorder,
                    pass_kind="mcp_only",
                )
                parameter_after_mcp = parameter_fingerprint_report(generator)
                generator.zero_grad(set_to_none=True)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                full_joint_report = run_backward_pass(
                    generator,
                    conditional_dict=conditional,
                    noisy_batch=noisy_batch,
                    recorder=recorder,
                    pass_kind="full_joint",
                )
                parameter_after_full = parameter_fingerprint_report(generator)
                route_gate = validate_route_report(recorder.calls)
    except ProbeContractError as exc:
        status = exc.code
        error = {"code": exc.code, "message": str(exc)}
    except RuntimeError as exc:
        status = PROBE_FAIL_NONFINITE if "non-finite" in str(exc) else "FAIL_RUNTIME"
        error = {"code": status, "message": str(exc)}
    finally:
        recorder.remove()

    if parameter_before is None:
        parameter_before = parameter_fingerprint_report(generator)
    if parameter_after_mcp is None:
        parameter_after_mcp = parameter_fingerprint_report(generator)
    if parameter_after_full is None:
        parameter_after_full = parameter_fingerprint_report(generator)
    parameter_safety = build_parameter_safety_report(
        before=parameter_before,
        after_mcp_only_backward=parameter_after_mcp,
        after_full_joint_backward=parameter_after_full,
    )
    if (
        status == "PASS"
        and not bool(parameter_safety["parameter_fingerprint_unchanged"])
    ):
        status = PROBE_FAIL_PARAMETER_MUTATION
        error = {
            "code": status,
            "message": "parameter fingerprint changed across no-step backward",
        }
    generator.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    memory["after_cleanup"] = helpers["memory_snapshot"]("after_cleanup", device)

    checkpoint_report = {
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "expected_checkpoint_sha256": validate_sha256(
            str(args.expected_checkpoint_sha256),
            name="--expected_checkpoint_sha256",
        ),
        "checkpoint_contains_mcp_keys": False,
        "fresh_mcp_initialized_from_backbone": bool(
            getattr(generator, "mcp_initialized_from_backbone", False)
        ),
        "sample_plan_sha256": str(sample_plan["sample_plan_sha256"]),
        "manifest_sha256": file_sha256(args.manifest),
        "conditionals_artifact_sha256": str(conditional_store.artifact_sha256),
    }
    report = build_probe_artifact(
        status=status,
        runtime_git_sha=runtime_git,
        checkpoint_report=checkpoint_report,
        route_calls=recorder.calls,
        route_gate=route_gate,
        mcp_only=mcp_only_report,
        full_joint=full_joint_report,
        parameter_safety_report=parameter_safety,
        optimizer_plan=optimizer_plan,
        sample_identity=str(identity),
        sample_cursor=cursor,
        trainable_parameter_count=trainable_count,
        memory_report=memory,
        error=error,
    )
    helpers["write_m4_json"](report, args.output_dir / PROBE_ARTIFACT_FILENAME)
    if status != "PASS":
        raise RuntimeError(f"official shared MCP output-head probe failed: {status}")
    validate_probe_artifact_schema(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute_real_probe:
        report = dry_run_artifact(output_dir=args.output_dir)
        print(report["status"])
        return 0
    report = run_real_probe(args)
    print(report["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
