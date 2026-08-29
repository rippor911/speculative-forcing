from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.nf_sf_full_sequence_continuation import (
    CONTINUATION_OBJECTIVE_MODE,
    load_continuation_parent_checkpoint,
    restore_continuation_state,
    rng_fingerprint,
    validate_git_sha,
    validate_optimizer_contract_for_continuation,
)
from utils.nf_sf_m3 import file_sha256, move_tensors_to_device
from utils.nf_sf_m4 import load_m4_sample_plan
from utils.nf_sf_tensors import (
    DEFAULT_S_MAIN,
    DEFAULT_S_MCP,
    FULL_SEQUENCE_CHUNK_FRAMES,
    FULL_SEQUENCE_NUM_CHUNKS,
    FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION,
)
from utils.nf_sf_training import (
    FULL_SEQUENCE_CHUNK_TOKENS,
    FULL_SEQUENCE_DEPTH_WEIGHTS,
    build_full_sequence_mcp_anchor_inputs,
    collect_nf_sf_parameter_groups,
    nf_sf_full_sequence_train_cursor,
    prepare_nf_sf_full_sequence_noisy_batch,
    validate_nf_sf_full_sequence_objective_mode,
)


PROBE_SCHEMA = "nf_sf_paper_fidelity_mcp1_gpu_probe_v1"
PROBE_DIAGNOSTIC_LABEL = "paper_fidelity_mcp1_real_gpu_no_step_probe"
PROBE_ARTIFACT_FILENAME = "paper_fidelity_mcp1_gpu_probe.json"
PROBE_PASS_LABEL = "PAPER_FIDELITY_MCP1_GPU_PROBE_PASS"
PROBE_FAIL_NONFINITE = "FAIL_NONFINITE"
PROBE_FAIL_GRADIENT = "FAIL_GRADIENT_EXPECTATION"
PROBE_FAIL_CLEAN_PATH = "FAIL_CLEAN_PATH_NOT_TRAINABLE"
PROBE_FAIL_PARAMETER_MUTATION = "FAIL_PARAMETER_SHA_CHANGED"

STEP6500_PARENT_CHECKPOINT = Path(
    "/home/dataset-assist-0/luojy/efficiency/rippor/experiment_outputs/"
    "nf_sf_full_sequence_continuation/c3f8988/"
    "continuation_5000_6500_20260817_165916/"
    "checkpoint_step006500.pt"
)
STEP6500_PARENT_STEP = 6500
STEP6500_PARENT_CHECKPOINT_SHA256 = (
    "9ef57cb2d3e5f20b244129317af4a0e1d2b1c810ba65ec970892e60ccbd34f4f"
)
STEP6500_PARENT_GIT_SHA = "c3f89888bf6da31b48650f0a680dd6534943f56f"
CANONICAL_CONFIG_PATH = (ROOT / "configs" / "self_forcing_dmd_mcp.yaml").resolve()
TAP_LAYERS = (3, 11, 19, 29)
MCP_BLOCKS_PER_DEPTH = 3
POSITIVE_GRADIENT_GROUPS = ("backbone", "patch_embedding", "mcp_fusion", "mcp_depth1")
NO_GRADIENT_GROUPS = ("mcp_depth2", "mcp_depth3", "main_final_head")


class ProbeContractError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class ProbeForwardResult:
    outputs: Any
    mcp_anchor_inputs: tuple[dict[str, Any], ...]
    mcp1_anchor_losses: tuple[torch.Tensor, ...]
    probe_loss: torch.Tensor


def _trainer_helpers() -> Mapping[str, Any]:
    from scripts import train_nf_sf_full_sequence_next_forcing as trainer

    return {
        "assert_finite_loss": trainer.assert_finite_loss,
        "assert_global_rng_equal": trainer.assert_global_rng_equal,
        "build_fresh_generator": trainer.build_fresh_generator,
        "build_optimizer": trainer.build_optimizer,
        "capture_global_rng_state": trainer.capture_global_rng_state,
        "dtype_from_arg": trainer.dtype_from_arg,
        "make_flow_scheduler": trainer.make_flow_scheduler,
        "memory_snapshot": trainer.memory_snapshot,
        "merge_config": trainer.merge_config,
        "optimizer_contract": trainer.optimizer_contract,
        "prepare_output_dir": trainer.prepare_output_dir,
        "target_latent_from_sample": trainer.target_latent_from_sample,
        "validate_sample_plan_contract": trainer.validate_sample_plan_contract,
        "validate_store_identity_order": trainer.validate_store_identity_order,
        "write_m4_json": trainer.write_m4_json,
    }


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
        description="NF-SF paper-fidelity MCP1 real GPU no-step probe."
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
        "--parent_checkpoint",
        type=Path,
        default=STEP6500_PARENT_CHECKPOINT,
    )
    parser.add_argument(
        "--expected_parent_checkpoint_sha256",
        default=STEP6500_PARENT_CHECKPOINT_SHA256,
    )
    parser.add_argument(
        "--expected_parent_global_step",
        type=int,
        choices=(STEP6500_PARENT_STEP,),
        default=STEP6500_PARENT_STEP,
    )
    parser.add_argument(
        "--expected_parent_checkpoint_git_sha",
        default=STEP6500_PARENT_GIT_SHA,
    )
    parser.add_argument("--expected_runtime_git_sha")
    parser.add_argument("--sample_plan", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset_root", type=Path)
    parser.add_argument("--conditionals_artifact", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16",), default="bf16")
    parser.add_argument(
        "--probe_global_step",
        type=int,
        default=STEP6500_PARENT_STEP + 1,
    )
    return parser.parse_args(argv)


def _require_real_probe_paths(args: argparse.Namespace) -> None:
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


def build_six_anchor_plan(
    *,
    chunk_tokens: int = FULL_SEQUENCE_CHUNK_TOKENS,
    num_chunks: int = FULL_SEQUENCE_NUM_CHUNKS,
    chunk_frames: int = FULL_SEQUENCE_CHUNK_FRAMES,
) -> tuple[dict[str, int], ...]:
    if int(num_chunks) != 7:
        raise ValueError("MCP1 paper-fidelity probe requires seven chunks")
    if int(chunk_tokens) <= 0 or int(chunk_frames) <= 0:
        raise ValueError("chunk_tokens and chunk_frames must be positive")
    target_tokens = int(chunk_tokens)
    return tuple(
        {
            "anchor_index": anchor_index,
            "source_chunk_index": anchor_index,
            "target_chunk_index": anchor_index + 1,
            "clean_token_count": anchor_index * target_tokens,
            "target_token_count": target_tokens,
            "total_token_count": (anchor_index + 1) * target_tokens,
            "future_start_frame": (anchor_index + 1) * int(chunk_frames),
        }
        for anchor_index in range(int(num_chunks) - 1)
    )


def _output_field(outputs: Any, key: str) -> Any:
    if isinstance(outputs, Mapping):
        return outputs[key]
    return getattr(outputs, key)


def _tensor_shape(tensor: torch.Tensor) -> list[int]:
    return [int(dim) for dim in tensor.shape]


def _is_finite_tensor(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor.detach().float()).all().item())


def compute_mcp1_exact_anchor_losses(
    mcp1_flow_pred: torch.Tensor,
    target_flow_mcp1: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if tuple(mcp1_flow_pred.shape) != tuple(target_flow_mcp1.shape):
        raise ValueError(
            "MCP1 prediction shape mismatch: "
            f"{tuple(mcp1_flow_pred.shape)} != {tuple(target_flow_mcp1.shape)}"
        )
    if mcp1_flow_pred.ndim < 2 or int(mcp1_flow_pred.shape[1]) != 6:
        raise ValueError("MCP1 probe expects exactly six anchors")
    return tuple(
        F.mse_loss(
            mcp1_flow_pred[:, anchor_index].float(),
            target_flow_mcp1[:, anchor_index].float(),
            reduction="mean",
        )
        for anchor_index in range(int(mcp1_flow_pred.shape[1]))
    )


def run_paper_fidelity_mcp1_forward_loss(
    generator: Any,
    *,
    conditional_dict: Mapping[str, Any],
    noisy_batch: Any,
) -> ProbeForwardResult:
    mcp_anchor_inputs = tuple(build_full_sequence_mcp_anchor_inputs(noisy_batch))
    outputs = generator.forward_full_sequence_next_forcing(
        noisy_image_or_video=noisy_batch.noisy_main,
        clean_x=noisy_batch.clean_target,
        conditional_dict=dict(conditional_dict),
        timestep_main=noisy_batch.timestep_main,
        mcp_anchor_inputs=mcp_anchor_inputs,
        paper_fidelity_mcp1_mask=True,
    )
    mcp_flow_preds_by_depth = tuple(_output_field(outputs, "mcp_flow_preds_by_depth"))
    if len(mcp_flow_preds_by_depth) != 3:
        raise RuntimeError("probe forward must preserve canonical MCP1/2/3 outputs")
    mcp1_losses = compute_mcp1_exact_anchor_losses(
        mcp_flow_preds_by_depth[0],
        noisy_batch.target_flow_mcp_depths[0],
    )
    probe_loss = torch.stack(mcp1_losses).mean()
    return ProbeForwardResult(
        outputs=outputs,
        mcp_anchor_inputs=mcp_anchor_inputs,
        mcp1_anchor_losses=mcp1_losses,
        probe_loss=probe_loss,
    )


def build_anchor_reports(
    *,
    forward: ProbeForwardResult,
    anchor_plan: Sequence[Mapping[str, int]] | None = None,
) -> list[dict[str, Any]]:
    plan = tuple(anchor_plan or build_six_anchor_plan())
    preds = _output_field(forward.outputs, "mcp_flow_preds_by_depth")[0]
    if len(plan) != len(forward.mcp1_anchor_losses):
        raise RuntimeError("anchor plan/loss count mismatch")
    reports = []
    for anchor_index, (contract, loss) in enumerate(zip(plan, forward.mcp1_anchor_losses)):
        pred = preds[:, anchor_index]
        reports.append(
            {
                "anchor_index": int(anchor_index),
                "source_chunk_index": int(contract["source_chunk_index"]),
                "target_chunk_index": int(contract["target_chunk_index"]),
                "clean_token_count": int(contract["clean_token_count"]),
                "target_token_count": int(contract["target_token_count"]),
                "total_token_count": int(contract["total_token_count"]),
                "future_start_frame": int(contract["future_start_frame"]),
                "flow_shape": _tensor_shape(pred),
                "finite": bool(_is_finite_tensor(pred) and _is_finite_tensor(loss)),
                "exact_fm_mse": float(loss.detach().float().item()),
            }
        )
    return reports


def validate_anchor_reports(reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = build_six_anchor_plan()
    if len(reports) != len(expected):
        raise ProbeContractError(PROBE_FAIL_NONFINITE, "probe must report six MCP1 anchors")
    for expected_item, actual in zip(expected, reports):
        for key in (
            "anchor_index",
            "clean_token_count",
            "target_token_count",
            "total_token_count",
            "future_start_frame",
        ):
            if int(actual[key]) != int(expected_item[key]):
                raise ProbeContractError(
                    PROBE_FAIL_NONFINITE,
                    f"anchor {actual.get('anchor_index')} {key} mismatch",
                )
        if not bool(actual.get("finite", False)):
            raise ProbeContractError(
                PROBE_FAIL_NONFINITE,
                f"anchor {actual.get('anchor_index')} produced non-finite output/loss",
            )
        value = float(actual["exact_fm_mse"])
        if not math.isfinite(value):
            raise ProbeContractError(
                PROBE_FAIL_NONFINITE,
                f"anchor {actual.get('anchor_index')} exact FM MSE is non-finite",
            )
    return {"status": "PASS", "anchor_count": len(reports)}


class ProbeInstrumentation:
    def __init__(self, *, memory_snapshot: Any, device: torch.device) -> None:
        self.memory_snapshot = memory_snapshot
        self.device = device
        self.anchor_records: list[dict[str, Any]] = []
        self.fusion_records: list[dict[str, Any]] = []
        self._module: Any | None = None
        self._original_forward: Any | None = None
        self._fusion_handle: Any | None = None
        self._active_anchor_index: int | None = None
        self._active_fusion_call: int = 0

    def install(self, generator: Any) -> None:
        if getattr(generator, "mcp", None) is None:
            raise RuntimeError("probe instrumentation requires generator.mcp")
        self._module = generator.mcp
        self._original_forward = generator.mcp.forward

        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            anchor_index = len(self.anchor_records)
            self._active_anchor_index = anchor_index
            self._active_fusion_call = 0
            target_features = tuple(kwargs.get("features") or ())
            clean_features = tuple(
                kwargs.get("paper_fidelity_clean_prefix_features") or ()
            )
            future_start_frames = list(kwargs.get("future_start_frames") or ())
            timesteps = list(kwargs.get("timesteps") or ())
            before = self.memory_snapshot(
                f"before_anchor{anchor_index}",
                self.device,
            )
            try:
                output = self._original_forward(*args, **kwargs)
            finally:
                self._active_anchor_index = None
            after = self.memory_snapshot(f"after_anchor{anchor_index}", self.device)
            self.anchor_records.append(
                {
                    "anchor_index": int(anchor_index),
                    "paper_fidelity_mcp1_mask": bool(
                        kwargs.get("paper_fidelity_mcp1_mask", False)
                    ),
                    "direct_clean_context_kv": bool(
                        kwargs.get("direct_clean_context_kv", False)
                    ),
                    "target_feature_token_count": (
                        int(target_features[0].shape[1]) if target_features else None
                    ),
                    "clean_prefix_token_count": (
                        int(clean_features[0].shape[1]) if clean_features else 0
                    ),
                    "future_start_frames": [int(value) for value in future_start_frames],
                    "timestep_shapes": [
                        _tensor_shape(value) if torch.is_tensor(value) else None
                        for value in timesteps
                    ],
                    "output_count": int(len(output)),
                    "memory_before": before,
                    "memory_after": after,
                }
            )
            return output

        def fusion_hook(module: Any, inputs: tuple[Any, ...], output: Any) -> None:
            if not torch.is_tensor(output):
                return
            if output.requires_grad:
                output.retain_grad()
            anchor_index = self._active_anchor_index
            call_index = int(self._active_fusion_call)
            role = "noisy_current_h_fuse" if call_index == 0 else "clean_h_fuse"
            self.fusion_records.append(
                {
                    "anchor_index": anchor_index,
                    "call_index": call_index,
                    "role": role,
                    "token_count": int(output.shape[1]),
                    "shape": _tensor_shape(output),
                    "requires_grad": bool(output.requires_grad),
                    "output": output,
                }
            )
            self._active_fusion_call += 1

        generator.mcp.forward = wrapped_forward
        self._fusion_handle = generator.mcp.fusion.register_forward_hook(fusion_hook)

    def remove(self) -> None:
        if self._fusion_handle is not None:
            self._fusion_handle.remove()
            self._fusion_handle = None
        if self._module is not None and self._original_forward is not None:
            self._module.forward = self._original_forward
        self._module = None
        self._original_forward = None


def _grad_summary_from_tensor(tensor: torch.Tensor | None) -> dict[str, Any]:
    if tensor is None or tensor.grad is None:
        return {
            "present": False,
            "finite": None,
            "norm": 0.0,
            "max_abs": 0.0,
        }
    grad = tensor.grad.detach().float()
    finite = bool(torch.isfinite(grad).all().item())
    return {
        "present": True,
        "finite": finite,
        "norm": float(grad.norm().item()),
        "max_abs": float(grad.abs().max().item()),
    }


def build_clean_path_report(
    fusion_records: Sequence[Mapping[str, Any]],
    *,
    anchor_plan: Sequence[Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    plan = tuple(anchor_plan or build_six_anchor_plan())
    by_anchor: list[dict[str, Any]] = []
    clean_failures = []
    noisy_failures = []
    for contract in plan:
        anchor_index = int(contract["anchor_index"])
        records = [
            record
            for record in fusion_records
            if record.get("anchor_index") == anchor_index
        ]
        noisy = next(
            (record for record in records if record.get("role") == "noisy_current_h_fuse"),
            None,
        )
        clean = next(
            (record for record in records if record.get("role") == "clean_h_fuse"),
            None,
        )
        noisy_grad = _grad_summary_from_tensor(
            noisy.get("output") if noisy is not None else None
        )
        clean_grad = _grad_summary_from_tensor(
            clean.get("output") if clean is not None else None
        )
        expected_clean_tokens = int(contract["clean_token_count"])
        item = {
            "anchor_index": anchor_index,
            "expected_clean_token_count": expected_clean_tokens,
            "expected_noisy_token_count": int(contract["target_token_count"]),
            "noisy_h_fuse": {
                "present": noisy is not None,
                "token_count": int(noisy["token_count"]) if noisy is not None else None,
                "requires_grad": bool(noisy["requires_grad"]) if noisy is not None else None,
                "grad": noisy_grad,
            },
            "clean_h_fuse": {
                "present": clean is not None,
                "token_count": int(clean["token_count"]) if clean is not None else None,
                "requires_grad": bool(clean["requires_grad"]) if clean is not None else None,
                "grad": clean_grad,
            },
        }
        if (
            noisy is None
            or int(noisy["token_count"]) != int(contract["target_token_count"])
            or not bool(noisy_grad["present"])
            or not bool(noisy_grad["finite"])
            or float(noisy_grad["norm"]) <= 0.0
        ):
            noisy_failures.append(anchor_index)
        if expected_clean_tokens > 0:
            if (
                clean is None
                or int(clean["token_count"]) != expected_clean_tokens
                or not bool(clean_grad["present"])
                or not bool(clean_grad["finite"])
                or float(clean_grad["norm"]) <= 0.0
            ):
                clean_failures.append(anchor_index)
        elif clean is not None:
            clean_failures.append(anchor_index)
        by_anchor.append(item)
    status = "PASS" if not clean_failures and not noisy_failures else PROBE_FAIL_CLEAN_PATH
    return {
        "status": status,
        "anchor_reports": by_anchor,
        "clean_path_required_anchor_indices": [1, 2, 3, 4, 5],
        "clean_path_failure_anchor_indices": clean_failures,
        "noisy_path_failure_anchor_indices": noisy_failures,
    }


def validate_clean_path_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    if report.get("status") != "PASS":
        raise ProbeContractError(
            PROBE_FAIL_CLEAN_PATH,
            "anchor1..5 clean_h_fuse or noisy_h_fuse gradient gate failed",
        )
    return report


def _named_main_final_head_parameters(generator: Any) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    head = getattr(getattr(generator, "model", None), "head", None)
    if head is None or not hasattr(head, "named_parameters"):
        return ()
    return tuple((f"model.head.{name}", param) for name, param in head.named_parameters())


def _grad_group_summary(
    group_name: str,
    named_params: Sequence[tuple[str, torch.nn.Parameter]],
) -> dict[str, Any]:
    trainable = [(name, param) for name, param in named_params if param.requires_grad]
    grad_tensors = 0
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
                sq_norm += grad_sq
                grad_norm = grad_sq ** 0.5
                grad_max_abs = float(grad.abs().max().item())
                max_abs = max(max_abs, grad_max_abs)
                record.update(
                    {
                        "grad_present": True,
                        "grad_norm": grad_norm,
                        "grad_finite": grad_finite,
                    }
                )
        parameter_records.append(record)
    norm = sq_norm ** 0.5
    return {
        "group": str(group_name),
        "parameter_tensors": int(len(named_params)),
        "trainable_tensors": int(len(trainable)),
        "grad_tensors": int(grad_tensors),
        "missing_grad_tensors": int(missing),
        "all_finite": bool(finite),
        "aggregate_grad_norm": float(norm),
        "max_abs_grad": float(max_abs),
        "parameters": parameter_records,
    }


def gradient_report_for_probe(generator: Any) -> dict[str, dict[str, Any]]:
    groups = {
        name: tuple(named_params)
        for name, named_params in collect_nf_sf_parameter_groups(generator).items()
    }
    groups["main_final_head"] = _named_main_final_head_parameters(generator)
    return {
        name: _grad_group_summary(name, named_params)
        for name, named_params in groups.items()
    }


def validate_gradient_report_for_probe(
    report: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    for name, item in report.items():
        if int(item.get("grad_tensors", 0)) > 0 and not bool(item.get("all_finite", False)):
            raise ProbeContractError(
                PROBE_FAIL_NONFINITE,
                f"non-finite gradient in group {name}",
            )
    failures = []
    for name in POSITIVE_GRADIENT_GROUPS:
        item = report.get(name)
        if item is None:
            failures.append(f"{name}:missing_group")
            continue
        if (
            int(item.get("grad_tensors", 0)) <= 0
            or not bool(item.get("all_finite", False))
            or float(item.get("aggregate_grad_norm", 0.0)) <= 0.0
        ):
            failures.append(f"{name}:expected_positive_finite_grad")
    for name in NO_GRADIENT_GROUPS:
        item = report.get(name)
        if item is None:
            failures.append(f"{name}:missing_group")
            continue
        if int(item.get("grad_tensors", 0)) != 0:
            failures.append(f"{name}:expected_no_grad")
    if failures:
        raise ProbeContractError(PROBE_FAIL_GRADIENT, ", ".join(failures))
    return {
        "status": "PASS",
        "positive_gradient_groups": list(POSITIVE_GRADIENT_GROUPS),
        "no_gradient_groups": list(NO_GRADIENT_GROUPS),
    }


def _sha256_tensor(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(str(list(cpu.shape)).encode("utf-8"))
    h.update(cpu.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def parameter_sha256_report(module: torch.nn.Module) -> dict[str, Any]:
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
        "\n".join(f"{item['name']}:{item['sha256']}" for item in entries).encode("utf-8")
    ).hexdigest()
    return {
        "tensor_count": len(entries),
        "aggregate_sha256": digest,
        "entries": entries,
    }


def parameter_sha_unchanged_report(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    unchanged = str(before.get("aggregate_sha256")) == str(after.get("aggregate_sha256"))
    return {
        "status": "PASS" if unchanged else PROBE_FAIL_PARAMETER_MUTATION,
        "unchanged": bool(unchanged),
        "before_aggregate_sha256": str(before.get("aggregate_sha256")),
        "after_aggregate_sha256": str(after.get("aggregate_sha256")),
        "before_tensor_count": int(before.get("tensor_count", 0)),
        "after_tensor_count": int(after.get("tensor_count", 0)),
    }


def trainable_parameter_count(module: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


def no_step_no_checkpoint_safety_report(
    *,
    parameter_before: Mapping[str, Any],
    parameter_after: Mapping[str, Any],
    trainable_parameter_count_before: int,
    trainable_parameter_count_after: int,
) -> dict[str, Any]:
    sha_report = parameter_sha_unchanged_report(parameter_before, parameter_after)
    return {
        "optimizer_step_executed": False,
        "checkpoint_written": False,
        "parameter_sha_unchanged": bool(sha_report["unchanged"]),
        "parameter_sha256": sha_report,
        "trainable_parameter_count_before": int(trainable_parameter_count_before),
        "trainable_parameter_count_after": int(trainable_parameter_count_after),
        "trainable_parameter_count_unchanged": (
            int(trainable_parameter_count_before)
            == int(trainable_parameter_count_after)
        ),
    }


def _overall_cuda_peak(
    snapshots: Mapping[str, Mapping[str, Any]],
    anchor_records: Sequence[Mapping[str, Any]],
) -> dict[str, int | bool]:
    peak_allocated = 0
    peak_reserved = 0
    total = 0
    saw_cuda = False
    candidates: list[Mapping[str, Any]] = list(snapshots.values())
    for record in anchor_records:
        for key in ("memory_before", "memory_after"):
            memory = record.get(key)
            if isinstance(memory, Mapping):
                candidates.append(memory)
    for memory in candidates:
        if not bool(memory.get("cuda", False)):
            continue
        saw_cuda = True
        peak_allocated = max(peak_allocated, int(memory.get("max_allocated", 0)))
        peak_reserved = max(peak_reserved, int(memory.get("max_reserved", 0)))
        total = max(total, int(memory.get("total", 0)))
    return {
        "cuda": bool(saw_cuda),
        "max_allocated": int(peak_allocated),
        "max_reserved": int(peak_reserved),
        "total": int(total),
    }


def build_memory_report(
    *,
    snapshots: Mapping[str, Mapping[str, Any]],
    anchor_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    anchor5 = next(
        (record for record in anchor_records if int(record.get("anchor_index", -1)) == 5),
        None,
    )
    return {
        "snapshots": dict(snapshots),
        "anchors": [dict(record) for record in anchor_records],
        "anchor5": dict(anchor5) if anchor5 is not None else None,
        "overall_peak": _overall_cuda_peak(snapshots, anchor_records),
    }


def build_probe_artifact(
    *,
    status: str,
    runtime_git_sha: str,
    checkpoint_provenance: Mapping[str, Any],
    anchors: Sequence[Mapping[str, Any]],
    gradient_report: Mapping[str, Any],
    gradient_gate: Mapping[str, Any] | None,
    clean_path_report: Mapping[str, Any],
    memory_report: Mapping[str, Any],
    safety_report: Mapping[str, Any],
    sample_identity: str | None,
    sample_cursor: Mapping[str, Any] | None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    pass_status = status == "PASS"
    return {
        "schema": PROBE_SCHEMA,
        "status": PROBE_PASS_LABEL if pass_status else str(status),
        "diagnostic_label": PROBE_DIAGNOSTIC_LABEL,
        "runtime_git_sha": validate_git_sha(runtime_git_sha, name="runtime_git_sha"),
        "checkpoint_provenance": dict(checkpoint_provenance),
        "paper_exact_reproduction": False,
        "paper_fidelity_mcp1_mask": True,
        "canonical_path_modified_by_probe": False,
        "objective_mode": CONTINUATION_OBJECTIVE_MODE,
        "main_shift": float(DEFAULT_S_MAIN),
        "mcp_shift": float(DEFAULT_S_MCP),
        "taps": list(TAP_LAYERS),
        "depth_weights": list(FULL_SEQUENCE_DEPTH_WEIGHTS),
        "mcp_blocks_per_depth": MCP_BLOCKS_PER_DEPTH,
        "rng_draw_order_version": FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION,
        "loss_contract": {
            "probe_loss": "mean(six MCP1 exact FM anchor losses)",
            "main_loss_in_backward": False,
            "mcp2_loss_in_backward": False,
            "mcp3_loss_in_backward": False,
            "optimizer_step_executed": False,
        },
        "sample_identity": sample_identity,
        "sample_cursor": dict(sample_cursor) if sample_cursor is not None else None,
        "anchors": [dict(item) for item in anchors],
        "gradient_report": dict(gradient_report),
        "gradient_gate": dict(gradient_gate) if gradient_gate is not None else None,
        "clean_path_report": dict(clean_path_report),
        "memory_report": dict(memory_report),
        "parameter_sha_unchanged": bool(safety_report.get("parameter_sha_unchanged", False)),
        "optimizer_step_executed": False,
        "checkpoint_written": False,
        "safety_report": dict(safety_report),
        "error": dict(error) if error is not None else None,
    }


def validate_probe_artifact_schema(report: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "status",
        "diagnostic_label",
        "runtime_git_sha",
        "checkpoint_provenance",
        "paper_fidelity_mcp1_mask",
        "anchors",
        "gradient_report",
        "clean_path_report",
        "memory_report",
        "parameter_sha_unchanged",
        "optimizer_step_executed",
        "checkpoint_written",
    }
    missing = required - set(report.keys())
    if missing:
        raise ValueError(f"probe artifact missing fields: {sorted(missing)}")
    if report["schema"] != PROBE_SCHEMA:
        raise ValueError("probe artifact schema mismatch")
    if report["diagnostic_label"] != PROBE_DIAGNOSTIC_LABEL:
        raise ValueError("probe artifact diagnostic label mismatch")
    if report["paper_fidelity_mcp1_mask"] is not True:
        raise ValueError("probe artifact must set paper_fidelity_mcp1_mask=true")
    if report["optimizer_step_executed"] is not False:
        raise ValueError("probe artifact must prove optimizer_step_executed=false")
    if report["checkpoint_written"] is not False:
        raise ValueError("probe artifact must prove checkpoint_written=false")
    if len(report["anchors"]) != 6:
        raise ValueError("probe artifact must report six MCP1 anchors")
    validate_anchor_reports(report["anchors"])
    if report["status"] == PROBE_PASS_LABEL:
        if report["parameter_sha_unchanged"] is not True:
            raise ValueError("PASS artifact must prove parameter SHA unchanged")
        if report["clean_path_report"].get("status") != "PASS":
            raise ValueError("PASS artifact must include PASS clean-path report")
    return {"status": "PASS"}


def dry_run_artifact(*, output_dir: Path, runtime_git_sha: str = "0" * 40) -> dict[str, Any]:
    anchor_plan = build_six_anchor_plan()
    anchors = [
        {
            **dict(anchor),
            "flow_shape": None,
            "finite": True,
            "exact_fm_mse": 0.0,
        }
        for anchor in anchor_plan
    ]
    report = build_probe_artifact(
        status="DRY_RUN",
        runtime_git_sha=runtime_git_sha,
        checkpoint_provenance={
            "parent_checkpoint_path": str(STEP6500_PARENT_CHECKPOINT),
            "expected_parent_checkpoint_sha256": STEP6500_PARENT_CHECKPOINT_SHA256,
            "expected_parent_global_step": STEP6500_PARENT_STEP,
            "expected_parent_checkpoint_git_sha": STEP6500_PARENT_GIT_SHA,
        },
        anchors=anchors,
        gradient_report={},
        gradient_gate=None,
        clean_path_report={
            "status": "DRY_RUN",
            "clean_path_required_anchor_indices": [1, 2, 3, 4, 5],
            "anchor_reports": [],
        },
        memory_report={"snapshots": {}, "anchors": [], "anchor5": None},
        safety_report={
            "optimizer_step_executed": False,
            "checkpoint_written": False,
            "parameter_sha_unchanged": True,
            "trainable_parameter_count_before": 0,
            "trainable_parameter_count_after": 0,
            "trainable_parameter_count_unchanged": True,
        },
        sample_identity=None,
        sample_cursor=None,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / PROBE_ARTIFACT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _validate_real_static_contract(args: argparse.Namespace, config: Any) -> None:
    if Path(args.config).resolve() != CANONICAL_CONFIG_PATH:
        raise RuntimeError("config path must be configs/self_forcing_dmd_mcp.yaml")
    if str(args.device) != "cuda:0":
        raise RuntimeError("real probe requires --device cuda:0")
    if str(args.dtype) != "bf16":
        raise RuntimeError("real probe requires --dtype bf16")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required only when --execute_real_probe is set")
    if int(getattr(config, "num_frame_per_block", 0)) != FULL_SEQUENCE_CHUNK_FRAMES:
        raise RuntimeError("config.num_frame_per_block must be 3")
    if not bool(getattr(config, "gradient_checkpointing", False)):
        raise RuntimeError("config.gradient_checkpointing must be true")
    if int(args.probe_global_step) != STEP6500_PARENT_STEP + 1:
        raise RuntimeError("probe_global_step must be 6501 for the step6500 parent")
    validate_nf_sf_full_sequence_objective_mode(CONTINUATION_OBJECTIVE_MODE)


def run_real_probe(args: argparse.Namespace) -> dict[str, Any]:
    _require_real_probe_paths(args)
    helpers = _trainer_helpers()
    config = helpers["merge_config"](args.config)
    _validate_real_static_contract(args, config)
    device = torch.device(args.device)
    dtype = helpers["dtype_from_arg"](args.dtype)
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

    sample_plan = load_m4_sample_plan(args.sample_plan, manifest_path=args.manifest)
    helpers["validate_sample_plan_contract"](sample_plan)
    conditional_store = M5ConditionalArtifactStore(
        artifact_dir=args.conditionals_artifact,
        sample_plan=sample_plan,
    )
    parent = load_continuation_parent_checkpoint(
        args.parent_checkpoint,
        expected_parent_checkpoint_sha256=args.expected_parent_checkpoint_sha256,
        expected_parent_global_step=int(args.expected_parent_global_step),
        expected_parent_checkpoint_git_sha=args.expected_parent_checkpoint_git_sha,
        sample_plan_sha256=str(sample_plan["sample_plan_sha256"]),
        manifest_sha256=file_sha256(args.manifest),
        conditionals_artifact_sha256=conditional_store.artifact_sha256,
    )
    teacher_store = M5TeacherSampleStore(
        sample_plan=sample_plan,
        manifest_path=args.manifest,
        dataset_root=args.dataset_root,
        reference_checkpoint_path=None,
        expected_reference_sha256=str(parent.payload["reference_checkpoint"]["sha256"]),
    )
    helpers["validate_store_identity_order"](
        sample_plan=sample_plan,
        teacher_store=teacher_store,
        conditional_store=conditional_store,
    )

    snapshots: dict[str, Mapping[str, Any]] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    snapshots["before_model"] = helpers["memory_snapshot"]("before_model", device)
    generator = helpers["build_fresh_generator"](
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
        dtype=dtype,
    )
    optimizer, optimizer_summary = helpers["build_optimizer"](
        generator,
        objective_mode=CONTINUATION_OBJECTIVE_MODE,
        backbone_lr=float(parent.payload["resolved_config"]["backbone_lr"]),
        patch_embedding_lr=float(parent.payload["resolved_config"]["patch_embedding_lr"]),
        mcp_lr=float(parent.payload["resolved_config"]["mcp_lr"]),
        weight_decay=float(parent.payload["resolved_config"]["weight_decay"]),
    )
    validate_optimizer_contract_for_continuation(
        parent.payload,
        active_optimizer_contract=helpers["optimizer_contract"](optimizer),
    )
    train_rng = torch.Generator(device=device)
    validation_base_rng = torch.Generator(device=device)
    restore_report = restore_continuation_state(
        generator=generator,
        optimizer=optimizer,
        train_rng=train_rng,
        validation_base_rng=validation_base_rng,
        payload=parent.payload,
        device=device,
    )
    snapshots["after_model_load"] = helpers["memory_snapshot"]("after_model_load", device)

    trainable_before = trainable_parameter_count(generator)
    parameter_before = parameter_sha256_report(generator)
    optimizer.zero_grad(set_to_none=True)
    scheduler_main = helpers["make_flow_scheduler"](DEFAULT_S_MAIN)
    scheduler_mcp = helpers["make_flow_scheduler"](DEFAULT_S_MCP)
    cursor = nf_sf_full_sequence_train_cursor(int(args.probe_global_step))
    identity = teacher_store.train_identity_for_step(int(args.probe_global_step))
    anchor_reports: list[dict[str, Any]] = []
    gradient_report: dict[str, Any] = {}
    gradient_gate: dict[str, Any] | None = None
    clean_path_report: dict[str, Any] = {}
    instrumentation = ProbeInstrumentation(
        memory_snapshot=helpers["memory_snapshot"],
        device=device,
    )
    status = "PASS"
    error: dict[str, Any] | None = None
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
                before_global_rng = helpers["capture_global_rng_state"](device)
                instrumentation.install(generator)
                try:
                    forward = run_paper_fidelity_mcp1_forward_loss(
                        generator,
                        conditional_dict=conditional,
                        noisy_batch=noisy_batch,
                    )
                finally:
                    instrumentation.remove()
                helpers["assert_global_rng_equal"](
                    before_global_rng,
                    helpers["capture_global_rng_state"](device),
                )
                helpers["assert_finite_loss"](forward.probe_loss, name="probe_loss")
                anchor_reports = build_anchor_reports(forward=forward)
                validate_anchor_reports(anchor_reports)
                snapshots["after_forward"] = helpers["memory_snapshot"](
                    "after_forward",
                    device,
                )
                forward.probe_loss.backward()
                helpers["assert_global_rng_equal"](
                    before_global_rng,
                    helpers["capture_global_rng_state"](device),
                )
                snapshots["after_backward"] = helpers["memory_snapshot"](
                    "after_backward",
                    device,
                )
                clean_path_report = build_clean_path_report(
                    instrumentation.fusion_records,
                )
                gradient_report = gradient_report_for_probe(generator)
                gradient_gate = validate_gradient_report_for_probe(gradient_report)
                validate_clean_path_report(clean_path_report)
                del forward
                del noisy_batch
                del conditional
                del clean_target
    except ProbeContractError as exc:
        status = exc.code
        error = {"code": exc.code, "message": str(exc)}
    except RuntimeError as exc:
        status = PROBE_FAIL_NONFINITE if "non-finite" in str(exc) else "FAIL_RUNTIME"
        error = {"code": status, "message": str(exc)}
    finally:
        instrumentation.remove()

    parameter_after = parameter_sha256_report(generator)
    trainable_after = trainable_parameter_count(generator)
    safety_report = no_step_no_checkpoint_safety_report(
        parameter_before=parameter_before,
        parameter_after=parameter_after,
        trainable_parameter_count_before=trainable_before,
        trainable_parameter_count_after=trainable_after,
    )
    if not safety_report["parameter_sha_unchanged"] and status == "PASS":
        status = PROBE_FAIL_PARAMETER_MUTATION
        error = {"code": status, "message": "parameter SHA changed without optimizer.step"}
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    snapshots["after_cleanup"] = helpers["memory_snapshot"]("after_cleanup", device)

    memory_report = build_memory_report(
        snapshots=snapshots,
        anchor_records=instrumentation.anchor_records,
    )
    checkpoint_provenance = {
        "parent_checkpoint_path": str(parent.path),
        "parent_checkpoint_sha256": str(parent.sha256),
        "expected_parent_checkpoint_sha256": str(args.expected_parent_checkpoint_sha256),
        "parent_global_step": int(parent.parent_global_step),
        "parent_checkpoint_git_sha": str(parent.parent_git_sha),
        "expected_parent_checkpoint_git_sha": str(args.expected_parent_checkpoint_git_sha),
        "official_checkpoint_path": str(args.checkpoint),
        "official_checkpoint_sha256": file_sha256(args.checkpoint),
        "sample_plan_sha256": str(sample_plan["sample_plan_sha256"]),
        "manifest_sha256": file_sha256(args.manifest),
        "conditionals_artifact_sha256": str(conditional_store.artifact_sha256),
        "restore_report": restore_report,
        "optimizer_summary": optimizer_summary,
        "rng_fingerprint_after_probe": rng_fingerprint(
            train_rng=train_rng,
            validation_base_rng=validation_base_rng,
            device=device,
        ),
    }
    report = build_probe_artifact(
        status=status,
        runtime_git_sha=runtime_git,
        checkpoint_provenance=checkpoint_provenance,
        anchors=anchor_reports,
        gradient_report=gradient_report,
        gradient_gate=gradient_gate,
        clean_path_report=clean_path_report,
        memory_report=memory_report,
        safety_report=safety_report,
        sample_identity=str(identity),
        sample_cursor=cursor,
        error=error,
    )
    helpers["write_m4_json"](report, args.output_dir / PROBE_ARTIFACT_FILENAME)
    if status != "PASS":
        raise RuntimeError(f"paper-fidelity MCP1 probe failed: {status}")
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
