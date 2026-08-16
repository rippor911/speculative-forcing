from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

import utils.nf_sf_full_sequence_eval as deployment
from utils.nf_sf_m3 import tensor_sha256, tensor_summary
from utils.nf_sf_tensors import (
    DEFAULT_NUM_TRAIN_TIMESTEPS,
    DEFAULT_S_MAIN,
    DEFAULT_S_MCP,
    FULL_SEQUENCE_CHUNK_FRAMES,
    FULL_SEQUENCE_DEPTHS,
    FULL_SEQUENCE_FRAME_COUNT,
    FULL_SEQUENCE_NUM_CHUNKS,
    flow_match_shift_timesteps,
)
from utils.nf_sf_training import (
    NFSFFullSequenceNoisyBatch,
    build_full_sequence_mcp_anchor_inputs,
    build_full_sequence_mcp_anchor_specs,
)
from utils.scheduler import FlowMatchScheduler


FIRST_MCP_ROUTE_EQUIVALENCE_SCHEMA = (
    "nf_sf_full_sequence_first_mcp_route_equivalence_v1"
)
FIRST_MCP_ROUTE_EQUIVALENCE_TENSOR_SCHEMA = (
    "nf_sf_full_sequence_first_mcp_route_equivalence_tensors_v1"
)
POINT_DEPLOYMENT_ENDPOINT = "deployment_endpoint_raw1000"
POINT_TRAINING_EDGE = "training_edge_raw999"
RAW_DEPLOYMENT_ENDPOINT = 1000
RAW_TRAINING_EDGE = 999
HISTORY_CHUNK_INDEX = 0
CURRENT_CHUNK_INDEX = 1
FUTURE_CHUNK_INDEX = 2
FUTURE_START_FRAME = FUTURE_CHUNK_INDEX * FULL_SEQUENCE_CHUNK_FRAMES
CURRENT_START_FRAME = CURRENT_CHUNK_INDEX * FULL_SEQUENCE_CHUNK_FRAMES


@dataclass(frozen=True)
class FirstMCPRouteEquivalenceResult:
    manifest: dict[str, Any]
    tensors: dict[str, Any]


def build_flow_match_scheduler(*, shift: float, device: torch.device | str) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(
        shift=float(shift),
        sigma_min=0.0,
        extra_one_step=True,
    )
    scheduler.set_timesteps(DEFAULT_NUM_TRAIN_TIMESTEPS, training=True)
    scheduler.sigmas = scheduler.sigmas.to(device)
    scheduler.timesteps = scheduler.timesteps.to(device)
    return scheduler


def raw_timestep_in_training_support(raw_timestep: int) -> bool:
    value = int(raw_timestep)
    return 0 <= value < DEFAULT_NUM_TRAIN_TIMESTEPS


def build_route_equivalence_point(raw_timestep: int) -> dict[str, Any]:
    raw_tensor = torch.tensor([float(raw_timestep)], dtype=torch.float32)
    main = flow_match_shift_timesteps(
        raw_tensor,
        shift=DEFAULT_S_MAIN,
        num_train_timesteps=DEFAULT_NUM_TRAIN_TIMESTEPS,
    )
    mcp = flow_match_shift_timesteps(
        raw_tensor,
        shift=DEFAULT_S_MCP,
        num_train_timesteps=DEFAULT_NUM_TRAIN_TIMESTEPS,
    )
    return {
        "raw_timestep": int(raw_timestep),
        "main_warped_timestep": float(main.item()),
        "mcp_warped_timestep": float(mcp.item()),
        "in_training_raw_support": raw_timestep_in_training_support(raw_timestep),
    }


def run_first_mcp_route_equivalence_audit(
    *,
    runtime_factory: Callable[[], deployment.DeploymentRuntime],
    generator: Any,
    main_scheduler: Any,
    mcp_scheduler: Any,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    teacher_payload: Mapping[str, Any],
    conditional_dict: Mapping[str, Any],
    checkpoint_summary: Mapping[str, Any],
    common_inputs: Mapping[str, Any],
    common_inputs_fingerprint_sha256: str,
    runtime_git_sha: str,
    training_checkpoint_git_sha: str,
) -> FirstMCPRouteEquivalenceResult:
    _validate_source_and_teacher(source_noise, teacher_target)
    rng_plan = deployment.build_absolute_chunk_rng_plan(
        source_noise=source_noise,
        rollout_seed=int(teacher_payload["rollout_seed"]),
        num_denoising_steps=len(deployment.RAW_DEPLOYMENT_SCHEDULE),
        chunk_frames=FULL_SEQUENCE_CHUNK_FRAMES,
    )
    points: dict[str, dict[str, Any]] = {}
    tensors: dict[str, Any] = {
        "schema": FIRST_MCP_ROUTE_EQUIVALENCE_TENSOR_SCHEMA,
        "teacher_chunk0": _chunk(teacher_target, HISTORY_CHUNK_INDEX).detach().cpu(),
        "teacher_chunk1": _chunk(teacher_target, CURRENT_CHUNK_INDEX).detach().cpu(),
        "teacher_chunk2": _chunk(teacher_target, FUTURE_CHUNK_INDEX).detach().cpu(),
        "source_noise_chunk1": _chunk(source_noise, CURRENT_CHUNK_INDEX).detach().cpu(),
        "source_noise_chunk2": _chunk(source_noise, FUTURE_CHUNK_INDEX).detach().cpu(),
        "points": {},
    }
    for point_name, raw in (
        (POINT_DEPLOYMENT_ENDPOINT, RAW_DEPLOYMENT_ENDPOINT),
        (POINT_TRAINING_EDGE, RAW_TRAINING_EDGE),
    ):
        point = build_route_equivalence_point(raw)
        deployment_result = _run_deployment_route_point(
            runtime=runtime_factory(),
            main_scheduler=main_scheduler,
            mcp_scheduler=mcp_scheduler,
            source_noise=source_noise,
            teacher_target=teacher_target,
            conditional_dict=conditional_dict,
            rng_plan=rng_plan,
            point=point,
        )
        training_result = _run_training_route_point(
            generator=generator,
            main_scheduler=main_scheduler,
            mcp_scheduler=mcp_scheduler,
            source_noise=source_noise,
            teacher_target=teacher_target,
            conditional_dict=conditional_dict,
            point=point,
        )
        comparison = _compare_route_results(deployment_result, training_result)
        points[point_name] = {
            **point,
            "deployment": deployment_result["summary"],
            "training_route": training_result["summary"],
            "route_comparison": comparison,
        }
        tensors["points"][point_name] = {
            "deployment_main_flow": deployment_result["tensors"]["main_flow"].detach().cpu(),
            "deployment_mcp_flow": deployment_result["tensors"]["mcp_flow"].detach().cpu(),
            "training_main_flow": training_result["tensors"]["main_flow"].detach().cpu(),
            "training_mcp_flow": training_result["tensors"]["mcp_flow"].detach().cpu(),
            "deployment_future_embed": deployment_result["hook"]["future_embed"].detach().cpu(),
            "training_future_embed": training_result["hook"]["future_embed"].detach().cpu(),
            "pre_hook_summaries": {
                "deployment": deployment_result["hook"]["summary"],
                "training_route": training_result["hook"]["summary"],
            },
        }
    manifest = {
        "schema": FIRST_MCP_ROUTE_EQUIVALENCE_SCHEMA,
        "status": "PASS",
        "diagnostic_only": True,
        "non_deployable": True,
        "runtime_git_sha": str(runtime_git_sha),
        "training_checkpoint_git_sha": str(training_checkpoint_git_sha),
        "checkpoint": dict(checkpoint_summary),
        "common_inputs_fingerprint_sha256": str(common_inputs_fingerprint_sha256),
        "common_inputs": dict(common_inputs),
        "rng_plan_fingerprint_sha256": rng_plan["trace"][
            "rng_plan_fingerprint_sha256"
        ],
        "training_raw_support_contract": {
            "training_raw_min": 0,
            "training_raw_max_inclusive": DEFAULT_NUM_TRAIN_TIMESTEPS - 1,
            "source": "_sample_raw_chunk_timesteps high=num_train_timesteps is exclusive",
        },
        "route_contract": {
            "history_chunks": [HISTORY_CHUNK_INDEX],
            "current_chunk": CURRENT_CHUNK_INDEX,
            "future_chunk": FUTURE_CHUNK_INDEX,
            "current_start_frame": CURRENT_START_FRAME,
            "future_start_frame": FUTURE_START_FRAME,
            "points": [POINT_DEPLOYMENT_ENDPOINT, POINT_TRAINING_EDGE],
            "one_joint_forward_per_route_per_point": True,
            "solver_loop": False,
            "deployment_depths_used": [1],
            "training_route_full_anchor_contract": {
                "depth1_anchor_count": 6,
                "depth2_anchor_count": 5,
                "depth3_anchor_count": 4,
                "selected_training_output": "depth1_anchor1",
            },
        },
        "input_tensors": _input_tensor_provenance(source_noise, teacher_target),
        "points": points,
        "forbidden_features": {
            "mcp_depth2_interpreted": False,
            "mcp_depth3_interpreted": False,
            "video_decode": False,
            "target_refinement": False,
            "verifier": False,
            "dmd": False,
            "routing": False,
            "self_rollout_training": False,
            "speed_benchmark": False,
        },
        "interpretation_contract": {
            "training_deployment_route_mismatch_supported": None,
            "deployment_endpoint_extrapolation_supported": None,
            "mcp_model_objective_failure_supported": None,
            "backbone_feature_route_mismatch_supported": None,
            "mcp_input_embedding_route_mismatch_supported": None,
            "case_descriptions": {
                "case_a_route_mismatch": (
                    "raw999 training route MCP flow is much better than "
                    "deployment route and route MCP outputs or taps differ: "
                    "train/deploy forward mismatch is the primary suspect."
                ),
                "case_b_endpoint_extrapolation": (
                    "raw999 is good on both routes but raw1000 is bad on both: "
                    "deployment extrapolation beyond training raw support is key."
                ),
                "case_c_model_objective": (
                    "raw999 training route itself remains bad and deployment is "
                    "similar: weights/objective/fusion/capacity/training is suspect."
                ),
                "case_d_backbone_context_route": (
                    "future embed/start/timestep match but four tap features differ "
                    "and training MCP is better: teacher-forced taps and KV-cache "
                    "taps are not equivalent."
                ),
                "case_e_mixed": "route, endpoint, and model errors can coexist.",
            },
        },
    }
    validate_first_mcp_route_equivalence_manifest(manifest)
    return FirstMCPRouteEquivalenceResult(manifest=manifest, tensors=tensors)


def validate_first_mcp_route_equivalence_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != FIRST_MCP_ROUTE_EQUIVALENCE_SCHEMA:
        raise RuntimeError("first MCP route equivalence schema mismatch")
    if manifest.get("status") != "PASS":
        raise RuntimeError("first MCP route equivalence status must be PASS")
    support = manifest.get("training_raw_support_contract")
    if not isinstance(support, Mapping):
        raise RuntimeError("training raw support contract missing")
    if int(support.get("training_raw_min", -1)) != 0:
        raise RuntimeError("training raw min mismatch")
    if int(support.get("training_raw_max_inclusive", -1)) != 999:
        raise RuntimeError("training raw max mismatch")
    points = manifest.get("points")
    if not isinstance(points, Mapping):
        raise RuntimeError("route equivalence points missing")
    if set(points.keys()) != {POINT_DEPLOYMENT_ENDPOINT, POINT_TRAINING_EDGE}:
        raise RuntimeError("route equivalence point set mismatch")
    if points[POINT_DEPLOYMENT_ENDPOINT].get("in_training_raw_support") is not False:
        raise RuntimeError("raw1000 must be outside training support")
    if points[POINT_TRAINING_EDGE].get("in_training_raw_support") is not True:
        raise RuntimeError("raw999 must be inside training support")
    for point_name, point in points.items():
        if point_name == POINT_DEPLOYMENT_ENDPOINT and int(point.get("raw_timestep")) != 1000:
            raise RuntimeError("deployment endpoint raw timestep mismatch")
        if point_name == POINT_TRAINING_EDGE and int(point.get("raw_timestep")) != 999:
            raise RuntimeError("training edge raw timestep mismatch")
        _validate_route_summary(point.get("deployment"), route="deployment")
        _validate_route_summary(point.get("training_route"), route="training_route")
        comparison = point.get("route_comparison")
        if not isinstance(comparison, Mapping):
            raise RuntimeError("route comparison missing")
        for field in (
            "current_state_route_sha_exact",
            "future_state_route_sha_exact",
            "main_timestep_route_exact",
            "mcp_timestep_route_exact",
            "future_grid_exact",
            "future_start_exact",
            "mcp_timestep_exact",
        ):
            if comparison.get(field) is not True:
                raise RuntimeError(f"{point_name} route comparison {field} must be true")
        if point_name == POINT_DEPLOYMENT_ENDPOINT:
            for route in ("deployment", "training_route"):
                summary = point.get(route)
                if not isinstance(summary, Mapping):
                    raise RuntimeError(f"{point_name} {route} summary missing")
                if summary.get("raw1000_current_state_equals_source") is not True:
                    raise RuntimeError(f"{point_name} {route} current state must equal source")
                if summary.get("raw1000_future_state_equals_source") is not True:
                    raise RuntimeError(f"{point_name} {route} future state must equal source")
    contract = manifest.get("route_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("route contract missing")
    if contract.get("solver_loop") is not False:
        raise RuntimeError("route equivalence audit must not run a solver loop")
    if contract.get("history_chunks") != [0]:
        raise RuntimeError("route equivalence audit must use only teacher chunk0 history")
    if int(contract.get("current_chunk", -1)) != CURRENT_CHUNK_INDEX:
        raise RuntimeError("route equivalence current chunk mismatch")
    if int(contract.get("future_chunk", -1)) != FUTURE_CHUNK_INDEX:
        raise RuntimeError("route equivalence future chunk mismatch")
    if int(contract.get("current_start_frame", -1)) != CURRENT_START_FRAME:
        raise RuntimeError("route equivalence current start frame mismatch")
    if int(contract.get("future_start_frame", -1)) != FUTURE_START_FRAME:
        raise RuntimeError("route equivalence future start frame mismatch")
    if contract.get("deployment_depths_used") != [1]:
        raise RuntimeError("route equivalence deployment must use depth1 only")
    full_anchor_contract = contract.get("training_route_full_anchor_contract")
    if not isinstance(full_anchor_contract, Mapping):
        raise RuntimeError("training route full anchor contract missing")
    if int(full_anchor_contract.get("depth1_anchor_count", -1)) != 6:
        raise RuntimeError("training route depth1 anchor count mismatch")
    if int(full_anchor_contract.get("depth2_anchor_count", -1)) != 5:
        raise RuntimeError("training route depth2 anchor count mismatch")
    if int(full_anchor_contract.get("depth3_anchor_count", -1)) != 4:
        raise RuntimeError("training route depth3 anchor count mismatch")
    if full_anchor_contract.get("selected_training_output") != "depth1_anchor1":
        raise RuntimeError("training route selected output contract mismatch")
    hypotheses = manifest.get("interpretation_contract")
    if not isinstance(hypotheses, Mapping):
        raise RuntimeError("interpretation contract missing")
    for field in (
        "training_deployment_route_mismatch_supported",
        "deployment_endpoint_extrapolation_supported",
        "mcp_model_objective_failure_supported",
        "backbone_feature_route_mismatch_supported",
        "mcp_input_embedding_route_mismatch_supported",
    ):
        if hypotheses.get(field) is not None:
            raise RuntimeError("route equivalence hypotheses must remain null")


def _run_deployment_route_point(
    *,
    runtime: deployment.DeploymentRuntime,
    main_scheduler: Any,
    mcp_scheduler: Any,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    rng_plan: Mapping[str, Any],
    point: Mapping[str, Any],
) -> dict[str, Any]:
    deployment_block_mask_before_is_none = _require_model_block_mask_none(
        runtime.generator,
        label="deployment route entry",
    )
    _recache_teacher_history0(
        runtime=runtime,
        source_noise=source_noise,
        teacher_target=teacher_target,
        conditional_dict=conditional_dict,
        rng_plan=rng_plan,
    )
    states = _build_point_states(
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        source_noise=source_noise,
        teacher_target=teacher_target,
        point=point,
    )
    current_state = states["current_state"]
    future_state = states["future_state"]
    main_timestep = _timestep(float(point["main_warped_timestep"]), current_state)
    mcp_timestep = _timestep(float(point["mcp_warped_timestep"]), future_state)
    snapshot = deployment.KVSnapshot.capture(runtime.kv_cache)
    kv_before = deployment.kv_boundary_summary(runtime.kv_cache)
    _require_model_block_mask_none(
        runtime.generator,
        label="deployment route before forward",
    )
    with _capture_mcp_pre_hook(runtime.generator.mcp) as capture:
        def call_joint():
            return runtime.generator(
                noisy_image_or_video=current_state,
                conditional_dict=dict(conditional_dict),
                timestep=main_timestep,
                kv_cache=runtime.kv_cache,
                crossattn_cache=runtime.crossattn_cache,
                current_start=CURRENT_START_FRAME * int(runtime.frame_seq_length),
                mcp_future_noises=[future_state],
                mcp_future_start_frames=[FUTURE_START_FRAME],
                mcp_timesteps=[mcp_timestep],
            )

        outputs, rng_guard = deployment._call_with_rng_guard(
            device=current_state.device,
            label="route_equivalence_deployment_joint_forward",
            fn=call_joint,
        )
    deployment_block_mask_after_is_none = _require_model_block_mask_none(
        runtime.generator,
        label="deployment route after forward",
    )
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise RuntimeError("deployment route must return one MCP output list")
    main_flow, returned_main_x0 = deployment._unpack_main_outputs(outputs)
    mcp_outputs = outputs[2]
    if not isinstance(mcp_outputs, (tuple, list)) or len(mcp_outputs) != 1:
        raise RuntimeError("deployment route must request depth1 only")
    mcp_flow = mcp_outputs[0]
    for name, tensor in (
        ("deployment_main_flow", main_flow),
        ("deployment_returned_main_x0", returned_main_x0),
        ("deployment_mcp_flow", mcp_flow),
    ):
        _require_finite_tensor(tensor, name=name)
    kv_temp = deployment.kv_boundary_summary(runtime.kv_cache)
    restored = snapshot.restore(runtime.kv_cache)
    if not restored:
        raise RuntimeError("deployment route KV rollback failed")
    kv_rollback = deployment.kv_boundary_summary(runtime.kv_cache)
    deployment._require_kv_rollback_matches(kv_before, kv_rollback)
    selected = _selected_mcp_call(capture, route="deployment")
    if selected["future_start_frames"] != [FUTURE_START_FRAME]:
        raise RuntimeError("deployment route MCP future_start_frames must be [6]")
    if len(selected["future_embeds"]) != 1:
        raise RuntimeError("deployment route must provide one future embed")
    exact = _exact_point_targets(
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        source_noise=source_noise,
        teacher_target=teacher_target,
        point=point,
    )
    main_x0 = _flow_to_x0_chunk(
        main_scheduler,
        flow=main_flow,
        state=current_state,
        timestep=main_timestep,
        name="deployment_main_x0",
    )
    mcp_x0 = _flow_to_x0_chunk(
        mcp_scheduler,
        flow=mcp_flow,
        state=future_state,
        timestep=mcp_timestep,
        name="deployment_mcp_x0",
    )
    summary = _route_summary(
        route="deployment",
        main_flow=main_flow,
        main_x0=main_x0,
        returned_main_x0=returned_main_x0,
        mcp_flow=mcp_flow,
        mcp_x0=mcp_x0,
        teacher_chunk1=_chunk(teacher_target, CURRENT_CHUNK_INDEX),
        teacher_chunk2=_chunk(teacher_target, FUTURE_CHUNK_INDEX),
        exact=exact,
        hook=selected,
        forward_rng=rng_guard,
        states=states,
        extra={
            "depths_used": [1],
            "mcp_call_count": len(capture.calls),
            "teacher_history_chunks": [HISTORY_CHUNK_INDEX],
            "current_start_frame": CURRENT_START_FRAME,
            "future_start_frame": FUTURE_START_FRAME,
            "deployment_block_mask_before_is_none": deployment_block_mask_before_is_none,
            "deployment_block_mask_after_is_none": deployment_block_mask_after_is_none,
            "kv_before": kv_before,
            "kv_temp": kv_temp,
            "kv_rollback": kv_rollback,
            "kv_rollback_exact": True,
        },
    )
    return {
        "summary": summary,
        "tensors": {
            "main_flow": main_flow,
            "mcp_flow": mcp_flow,
            "current_state": current_state,
            "future_state": future_state,
            "main_timestep": main_timestep,
            "mcp_timestep": mcp_timestep,
        },
        "hook": selected,
    }


def _run_training_route_point(
    *,
    generator: Any,
    main_scheduler: Any,
    mcp_scheduler: Any,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    point: Mapping[str, Any],
) -> dict[str, Any]:
    noisy_batch = _build_training_noisy_batch(
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        source_noise=source_noise,
        teacher_target=teacher_target,
        point=point,
    )
    anchors = build_full_sequence_mcp_anchor_inputs(noisy_batch)
    if sum(len(anchor["depths"]) for anchor in anchors) != 15:
        raise RuntimeError("training route must provide all 15 valid MCP anchors")
    model = _model_with_block_mask(generator)
    original_block_mask = model.block_mask
    training_block_mask_before_is_none = original_block_mask is None
    if not training_block_mask_before_is_none:
        raise RuntimeError("training route prior block_mask must be None")
    training_block_mask_created = False
    try:
        model.block_mask = None
        with _capture_mcp_pre_hook(generator.mcp) as capture:
            def call_training():
                with torch.no_grad():
                    return generator.forward_full_sequence_next_forcing(
                        noisy_image_or_video=noisy_batch.noisy_main,
                        clean_x=teacher_target,
                        conditional_dict=dict(conditional_dict),
                        timestep_main=noisy_batch.timestep_main,
                        mcp_anchor_inputs=anchors,
                    )

            outputs, rng_guard = deployment._call_with_rng_guard(
                device=teacher_target.device,
                label="route_equivalence_training_full_sequence_forward",
                fn=call_training,
            )
        training_block_mask_created = model.block_mask is not None
        if not training_block_mask_created:
            raise RuntimeError("training route did not create teacher-forcing block_mask")
    finally:
        model.block_mask = original_block_mask
    training_block_mask_restored_is_none = model.block_mask is None
    if not training_block_mask_restored_is_none:
        raise RuntimeError("training route failed to restore block_mask")
    main_flow_full = _output_field(outputs, "main_flow_pred")
    mcp_by_depth = tuple(_output_field(outputs, "mcp_flow_preds_by_depth"))
    if len(mcp_by_depth) != 3:
        raise RuntimeError("training route must return MCP depth1/2/3 predictions")
    main_flow = _chunk(main_flow_full, CURRENT_CHUNK_INDEX)
    mcp_flow = mcp_by_depth[0][:, CURRENT_CHUNK_INDEX]
    for name, tensor in (
        ("training_main_flow", main_flow),
        ("training_mcp_flow", mcp_flow),
    ):
        _require_finite_tensor(tensor, name=name)
    selected = _selected_mcp_call(capture, route="training_route")
    if selected["future_start_frames"][0] != FUTURE_START_FRAME:
        raise RuntimeError("training route selected MCP call must start with future frame 6")
    exact = _exact_point_targets(
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        source_noise=source_noise,
        teacher_target=teacher_target,
        point=point,
    )
    current_state = _chunk(noisy_batch.noisy_main, CURRENT_CHUNK_INDEX)
    future_state = noisy_batch.noisy_mcp_depths[0][:, CURRENT_CHUNK_INDEX]
    states = {
        "current_state": current_state,
        "future_state": future_state,
        "raw1000_current_state_equals_source": bool(
            torch.equal(current_state, _chunk(source_noise, CURRENT_CHUNK_INDEX))
        ),
        "raw1000_future_state_equals_source": bool(
            torch.equal(future_state, _chunk(source_noise, FUTURE_CHUNK_INDEX))
        ),
    }
    main_timestep = _timestep(float(point["main_warped_timestep"]), current_state)
    mcp_timestep = _timestep(float(point["mcp_warped_timestep"]), future_state)
    main_x0 = _flow_to_x0_chunk(
        main_scheduler,
        flow=main_flow,
        state=current_state,
        timestep=main_timestep,
        name="training_main_x0",
    )
    mcp_x0 = _flow_to_x0_chunk(
        mcp_scheduler,
        flow=mcp_flow,
        state=future_state,
        timestep=mcp_timestep,
        name="training_mcp_x0",
    )
    summary = _route_summary(
        route="training_route",
        main_flow=main_flow,
        main_x0=main_x0,
        returned_main_x0=None,
        mcp_flow=mcp_flow,
        mcp_x0=mcp_x0,
        teacher_chunk1=_chunk(teacher_target, CURRENT_CHUNK_INDEX),
        teacher_chunk2=_chunk(teacher_target, FUTURE_CHUNK_INDEX),
        exact=exact,
        hook=selected,
        forward_rng=rng_guard,
        states=states,
        extra={
            "forward_full_sequence_next_forcing_called": True,
            "mcp_call_count": len(capture.calls),
            "anchor_count": len(anchors),
            "flat_anchor_future_count": sum(len(anchor["depths"]) for anchor in anchors),
            "teacher_history_chunks": [HISTORY_CHUNK_INDEX],
            "current_start_frame": CURRENT_START_FRAME,
            "future_start_frame": FUTURE_START_FRAME,
            "training_block_mask_before_is_none": training_block_mask_before_is_none,
            "training_block_mask_created": training_block_mask_created,
            "training_block_mask_restored_is_none": training_block_mask_restored_is_none,
            "selected_output": {
                "depth": 1,
                "anchor_index": CURRENT_CHUNK_INDEX,
                "target_chunk_index": FUTURE_CHUNK_INDEX,
            },
            "main_flow_full": _tensor_record(main_flow_full),
            "depth_prediction_shapes": [
                [int(dim) for dim in tensor.shape] for tensor in mcp_by_depth
            ],
        },
    )
    return {
        "summary": summary,
        "tensors": {
            "main_flow": main_flow,
            "mcp_flow": mcp_flow,
            "current_state": current_state,
            "future_state": future_state,
            "main_timestep": main_timestep,
            "mcp_timestep": mcp_timestep,
        },
        "hook": selected,
    }


def _build_training_noisy_batch(
    *,
    main_scheduler: Any,
    mcp_scheduler: Any,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    point: Mapping[str, Any],
) -> NFSFFullSequenceNoisyBatch:
    main_t = _timestep(float(point["main_warped_timestep"]), teacher_target)
    noisy_main = _add_noise_chunk(
        main_scheduler,
        clean=teacher_target,
        noise=source_noise,
        timestep=main_t,
        name="training_noisy_main",
    )
    target_flow_main = _training_target_chunk(
        main_scheduler,
        clean=teacher_target,
        noise=source_noise,
        timestep=main_t,
        name="training_main_target_flow",
    )
    anchor_specs = build_full_sequence_mcp_anchor_specs()
    noisy_depths = []
    target_depths = []
    epsilon_depths = []
    raw_depths = []
    timestep_depths = []
    for depth in FULL_SEQUENCE_DEPTHS:
        chunks = []
        epsilons = []
        timesteps = []
        targets = []
        raw = torch.full(
            (teacher_target.shape[0], FULL_SEQUENCE_NUM_CHUNKS - int(depth)),
            int(point["raw_timestep"]),
            device=teacher_target.device,
            dtype=torch.int64,
        )
        for anchor_index in range(FULL_SEQUENCE_NUM_CHUNKS - int(depth)):
            target_chunk_index = anchor_index + int(depth)
            clean = _chunk(teacher_target, target_chunk_index)
            noise = _chunk(source_noise, target_chunk_index)
            timestep = _timestep(float(point["mcp_warped_timestep"]), clean)
            chunks.append(
                _add_noise_chunk(
                    mcp_scheduler,
                    clean=clean,
                    noise=noise,
                    timestep=timestep,
                    name="training_mcp_noisy_future",
                )
            )
            targets.append(
                _training_target_chunk(
                    mcp_scheduler,
                    clean=clean,
                    noise=noise,
                    timestep=timestep,
                    name="training_mcp_target_flow",
                )
            )
            epsilons.append(noise)
            timesteps.append(timestep)
        noisy_depths.append(torch.stack(chunks, dim=1))
        target_depths.append(torch.stack(targets, dim=1))
        epsilon_depths.append(torch.stack(epsilons, dim=1))
        raw_depths.append(raw)
        timestep_depths.append(torch.stack(timesteps, dim=1))
    raw_main = torch.full(
        (teacher_target.shape[0], FULL_SEQUENCE_NUM_CHUNKS),
        int(point["raw_timestep"]),
        device=teacher_target.device,
        dtype=torch.int64,
    )
    return NFSFFullSequenceNoisyBatch(
        clean_target=teacher_target,
        noisy_main=noisy_main,
        target_flow_main=target_flow_main,
        epsilon_main=source_noise,
        raw_timestep_main=raw_main,
        timestep_main=main_t,
        noisy_mcp_depths=tuple(noisy_depths),
        target_flow_mcp_depths=tuple(target_depths),
        epsilon_mcp_depths=tuple(epsilon_depths),
        raw_timestep_mcp_depths=tuple(raw_depths),
        timestep_mcp_depths=tuple(timestep_depths),
        anchor_specs=anchor_specs,
    )


def _build_point_states(
    *,
    main_scheduler: Any,
    mcp_scheduler: Any,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    point: Mapping[str, Any],
) -> dict[str, Any]:
    if int(point["raw_timestep"]) == RAW_DEPLOYMENT_ENDPOINT:
        current_state = _chunk(source_noise, CURRENT_CHUNK_INDEX).detach().clone()
        future_state = _chunk(source_noise, FUTURE_CHUNK_INDEX).detach().clone()
    else:
        current_state = _add_noise_chunk(
            main_scheduler,
            clean=_chunk(teacher_target, CURRENT_CHUNK_INDEX),
            noise=_chunk(source_noise, CURRENT_CHUNK_INDEX),
            timestep=_timestep(float(point["main_warped_timestep"]), _chunk(source_noise, CURRENT_CHUNK_INDEX)),
            name="deployment_raw999_current_state",
        )
        future_state = _add_noise_chunk(
            mcp_scheduler,
            clean=_chunk(teacher_target, FUTURE_CHUNK_INDEX),
            noise=_chunk(source_noise, FUTURE_CHUNK_INDEX),
            timestep=_timestep(float(point["mcp_warped_timestep"]), _chunk(source_noise, FUTURE_CHUNK_INDEX)),
            name="deployment_raw999_future_state",
        )
    for name, tensor in (("current_state", current_state), ("future_state", future_state)):
        _require_finite_tensor(tensor, name=name)
    return {
        "current_state": current_state,
        "future_state": future_state,
        "raw1000_current_state_equals_source": bool(
            torch.equal(current_state, _chunk(source_noise, CURRENT_CHUNK_INDEX))
        ),
        "raw1000_future_state_equals_source": bool(
            torch.equal(future_state, _chunk(source_noise, FUTURE_CHUNK_INDEX))
        ),
    }


def _exact_point_targets(
    *,
    main_scheduler: Any,
    mcp_scheduler: Any,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    point: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    main_timestep = _timestep(
        float(point["main_warped_timestep"]),
        _chunk(source_noise, CURRENT_CHUNK_INDEX),
    )
    mcp_timestep = _timestep(
        float(point["mcp_warped_timestep"]),
        _chunk(source_noise, FUTURE_CHUNK_INDEX),
    )
    return {
        "main_target": _training_target_chunk(
            main_scheduler,
            clean=_chunk(teacher_target, CURRENT_CHUNK_INDEX),
            noise=_chunk(source_noise, CURRENT_CHUNK_INDEX),
            timestep=main_timestep,
            name="exact_main_target_flow",
        ),
        "mcp_target": _training_target_chunk(
            mcp_scheduler,
            clean=_chunk(teacher_target, FUTURE_CHUNK_INDEX),
            noise=_chunk(source_noise, FUTURE_CHUNK_INDEX),
            timestep=mcp_timestep,
            name="exact_mcp_target_flow",
        ),
    }


def _route_summary(
    *,
    route: str,
    main_flow: torch.Tensor,
    main_x0: torch.Tensor,
    returned_main_x0: torch.Tensor | None,
    mcp_flow: torch.Tensor,
    mcp_x0: torch.Tensor,
    teacher_chunk1: torch.Tensor,
    teacher_chunk2: torch.Tensor,
    exact: Mapping[str, torch.Tensor],
    hook: Mapping[str, Any],
    forward_rng: Mapping[str, Any],
    states: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    summary = {
        "route": str(route),
        "main_flow": _tensor_record(main_flow),
        "main_x0": _tensor_record(main_x0),
        "returned_main_x0": None if returned_main_x0 is None else _tensor_record(returned_main_x0),
        "mcp_depth1_flow": _tensor_record(mcp_flow),
        "mcp_x0": _tensor_record(mcp_x0),
        "exact_main_target_flow": _tensor_record(exact["main_target"]),
        "exact_mcp_target_flow": _tensor_record(exact["mcp_target"]),
        "main_flow_mse_to_exact": _mse(main_flow, exact["main_target"]),
        "main_x0_mse_to_teacher": _mse(main_x0, teacher_chunk1),
        "mcp_flow_mse_to_exact": _mse(mcp_flow, exact["mcp_target"]),
        "mcp_x0_mse_to_teacher": _mse(mcp_x0, teacher_chunk2),
        "current_state": _tensor_record(states["current_state"]),
        "future_state": _tensor_record(states["future_state"]),
        "raw1000_current_state_equals_source": bool(
            states["raw1000_current_state_equals_source"]
        ),
        "raw1000_future_state_equals_source": bool(
            states["raw1000_future_state_equals_source"]
        ),
        "forward_rng": dict(forward_rng),
        "selected_mcp_pre_hook": hook["summary"],
    }
    summary.update(dict(extra))
    return summary


def _compare_route_results(
    deployment_result: Mapping[str, Any],
    training_result: Mapping[str, Any],
) -> dict[str, Any]:
    deployment_hook = deployment_result["hook"]
    training_hook = training_result["hook"]
    dep_main = deployment_result["tensors"]["main_flow"]
    train_main = training_result["tensors"]["main_flow"]
    dep_mcp = deployment_result["tensors"]["mcp_flow"]
    train_mcp = training_result["tensors"]["mcp_flow"]
    dep_current = deployment_result["tensors"]["current_state"]
    train_current = training_result["tensors"]["current_state"]
    dep_future = deployment_result["tensors"]["future_state"]
    train_future = training_result["tensors"]["future_state"]
    dep_main_t = deployment_result["tensors"]["main_timestep"]
    train_main_t = training_result["tensors"]["main_timestep"]
    dep_mcp_t = deployment_result["tensors"]["mcp_timestep"]
    train_mcp_t = training_result["tensors"]["mcp_timestep"]
    dep_features = deployment_hook["features"]
    train_features = training_hook["features"]
    if len(dep_features) != len(train_features):
        raise RuntimeError("route feature tap count mismatch")
    tap_mse = []
    tap_sha_exact = []
    tap_mean_abs = []
    tap_max_abs = []
    for left, right in zip(dep_features, train_features):
        tap_mse.append(_mse(left, right))
        tap_sha_exact.append(tensor_sha256(left.detach().cpu()) == tensor_sha256(right.detach().cpu()))
        delta = (left.detach().float() - right.detach().float()).abs()
        tap_mean_abs.append(float(delta.mean().item()))
        tap_max_abs.append(float(delta.max().item()))
    dep_embed = deployment_hook["future_embed"]
    train_embed = training_hook["future_embed"]
    embed_delta = (dep_embed.detach().float() - train_embed.detach().float()).abs()
    return {
        "current_state_route_sha_exact": tensor_sha256(dep_current.detach().cpu())
        == tensor_sha256(train_current.detach().cpu()),
        "future_state_route_sha_exact": tensor_sha256(dep_future.detach().cpu())
        == tensor_sha256(train_future.detach().cpu()),
        "main_timestep_route_exact": tensor_sha256(dep_main_t.detach().cpu())
        == tensor_sha256(train_main_t.detach().cpu()),
        "mcp_timestep_route_exact": tensor_sha256(dep_mcp_t.detach().cpu())
        == tensor_sha256(train_mcp_t.detach().cpu()),
        "main_flow_route_mse": _mse(dep_main, train_main),
        "main_flow_sha_exact": tensor_sha256(dep_main.detach().cpu())
        == tensor_sha256(train_main.detach().cpu()),
        "mcp_flow_route_mse": _mse(dep_mcp, train_mcp),
        "mcp_flow_sha_exact": tensor_sha256(dep_mcp.detach().cpu())
        == tensor_sha256(train_mcp.detach().cpu()),
        "tap_feature_mse_by_tap": tap_mse,
        "tap_feature_sha_exact_by_tap": tap_sha_exact,
        "tap_feature_mean_abs_by_tap": tap_mean_abs,
        "tap_feature_max_abs_by_tap": tap_max_abs,
        "future_embed_mse": _mse(dep_embed, train_embed),
        "future_embed_sha_exact": tensor_sha256(dep_embed.detach().cpu())
        == tensor_sha256(train_embed.detach().cpu()),
        "future_embed_mean_abs": float(embed_delta.mean().item()),
        "future_embed_max_abs": float(embed_delta.max().item()),
        "future_grid_exact": _json_fingerprint(deployment_hook["future_grid_record"])
        == _json_fingerprint(training_hook["future_grid_record"]),
        "future_start_exact": deployment_hook["future_start_frames"][0]
        == training_hook["future_start_frames"][0]
        == FUTURE_START_FRAME,
        "mcp_timestep_exact": tensor_sha256(deployment_hook["timestep"].detach().cpu())
        == tensor_sha256(training_hook["timestep"].detach().cpu()),
    }


@contextmanager
def _capture_mcp_pre_hook(mcp_module: Any):
    if mcp_module is None:
        raise RuntimeError("route equivalence audit requires generator.mcp")
    capture = _MCPPreHookCapture()
    handle = mcp_module.register_forward_pre_hook(capture.hook, with_kwargs=True)
    try:
        yield capture
    finally:
        handle.remove()


class _MCPPreHookCapture:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.selected: list[dict[str, Any]] = []

    def hook(self, _module: Any, _args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        starts = [None if value is None else int(value) for value in kwargs["future_start_frames"]]
        record = {
            "call_index": len(self.calls),
            "future_start_frames": starts,
            "future_count": len(starts),
        }
        self.calls.append(record)
        if starts and starts[0] == FUTURE_START_FRAME:
            selected = _clone_selected_mcp_inputs(kwargs, call_index=record["call_index"])
            self.selected.append(selected)


def _clone_selected_mcp_inputs(
    kwargs: Mapping[str, Any],
    *,
    call_index: int,
) -> dict[str, Any]:
    features = tuple(_clone_tensor(tensor, name="mcp_pre_hook_feature") for tensor in kwargs["features"])
    future_embeds = tuple(
        _clone_tensor(tensor, name="mcp_pre_hook_future_embed")
        for tensor in kwargs["future_embeds"]
    )
    timesteps = tuple(
        None if tensor is None else _clone_tensor(tensor, name="mcp_pre_hook_timestep")
        for tensor in kwargs["timesteps"]
    )
    future_start_frames = [
        None if value is None else int(value)
        for value in kwargs["future_start_frames"]
    ]
    future_grids = tuple(_clone_grid(value) for value in kwargs["future_grid_sizes"])
    if not future_embeds:
        raise RuntimeError("selected MCP call has no future embeds")
    if timesteps[0] is None:
        raise RuntimeError("selected MCP call has no timestep for depth1")
    selected = {
        "call_index": int(call_index),
        "features": features,
        "future_embeds": future_embeds,
        "future_embed": future_embeds[0],
        "future_grid_sizes": future_grids,
        "future_grid_record": _grid_record(future_grids[0]),
        "future_start_frames": future_start_frames,
        "timesteps": timesteps,
        "timestep": timesteps[0],
    }
    selected["summary"] = _mcp_hook_summary(selected)
    return selected


def _selected_mcp_call(capture: _MCPPreHookCapture, *, route: str) -> dict[str, Any]:
    if len(capture.selected) != 1:
        raise RuntimeError(
            f"{route} route must capture exactly one selected MCP call, got {len(capture.selected)}"
        )
    selected = capture.selected[0]
    if route == "deployment" and len(selected["future_start_frames"]) != 1:
        raise RuntimeError("deployment route must call MCP depth1 only")
    if route == "training_route" and selected["future_start_frames"][:3] != [6, 9, 12]:
        raise RuntimeError("training route selected anchor1 must expose starts [6, 9, 12]")
    return selected


def _mcp_hook_summary(selected: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "call_index": int(selected["call_index"]),
        "future_start_frames": list(selected["future_start_frames"]),
        "feature_summaries": [_tensor_record(tensor) for tensor in selected["features"]],
        "future_embed_summaries": [
            _tensor_record(tensor) for tensor in selected["future_embeds"]
        ],
        "future_grid_records": [
            _grid_record(grid) for grid in selected["future_grid_sizes"]
        ],
        "timestep_summaries": [
            None if tensor is None else _tensor_record(tensor)
            for tensor in selected["timesteps"]
        ],
    }


def _recache_teacher_history0(
    *,
    runtime: deployment.DeploymentRuntime,
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    rng_plan: Mapping[str, Any],
) -> None:
    _ = source_noise
    counts = {"clean_recache_forward_count": 0}
    deployment._clean_recache(
        runtime=runtime,
        conditional_dict=conditional_dict,
        rng_plan=rng_plan,
        counts=counts,
        clean_chunk=_chunk(teacher_target, HISTORY_CHUNK_INDEX),
        chunk_index=HISTORY_CHUNK_INDEX,
        start_frame=0,
        expected_before=None,
    )


def _validate_route_summary(summary: Any, *, route: str) -> None:
    if not isinstance(summary, Mapping):
        raise RuntimeError(f"{route} summary missing")
    rng = summary.get("forward_rng")
    if not isinstance(rng, Mapping) or rng.get("unchanged") is not True:
        raise RuntimeError(f"{route} RNG guard did not remain unchanged")
    if rng.get("state_before_hash") != rng.get("state_after_hash"):
        raise RuntimeError(f"{route} RNG guard hash mismatch")
    for key in (
        "main_flow",
        "main_x0",
        "mcp_depth1_flow",
        "mcp_x0",
        "exact_main_target_flow",
        "exact_mcp_target_flow",
    ):
        record = summary.get(key)
        if not isinstance(record, Mapping) or record.get("finite") is not True:
            raise RuntimeError(f"{route} {key} must be finite")
    if summary.get("teacher_history_chunks") != [HISTORY_CHUNK_INDEX]:
        raise RuntimeError(f"{route} must use only teacher chunk0 history")
    if int(summary.get("current_start_frame", -1)) != CURRENT_START_FRAME:
        raise RuntimeError(f"{route} current start frame mismatch")
    if int(summary.get("future_start_frame", -1)) != FUTURE_START_FRAME:
        raise RuntimeError(f"{route} future start frame mismatch")
    if route == "deployment":
        if summary.get("deployment_block_mask_before_is_none") is not True:
            raise RuntimeError("deployment route block_mask before forward must be None")
        if summary.get("deployment_block_mask_after_is_none") is not True:
            raise RuntimeError("deployment route block_mask after forward must be None")
        if summary.get("depths_used") != [1]:
            raise RuntimeError("deployment route must use depth1 only")
        if int(summary.get("mcp_call_count", -1)) != 1:
            raise RuntimeError("deployment route must call MCP exactly once")
        if summary.get("kv_rollback_exact") is not True:
            raise RuntimeError("deployment route KV rollback must be exact")
    if route == "training_route":
        if summary.get("training_block_mask_before_is_none") is not True:
            raise RuntimeError("training route block_mask before forward must be None")
        if summary.get("training_block_mask_created") is not True:
            raise RuntimeError("training route must create teacher-forcing block_mask")
        if summary.get("training_block_mask_restored_is_none") is not True:
            raise RuntimeError("training route must restore block_mask to None")
        if summary.get("forward_full_sequence_next_forcing_called") is not True:
            raise RuntimeError("training route must call forward_full_sequence_next_forcing")
        if int(summary.get("anchor_count", -1)) != 6:
            raise RuntimeError("training route must provide 6 depth1 anchors")
        if int(summary.get("flat_anchor_future_count", -1)) != 15:
            raise RuntimeError("training route must provide all 15 MCP futures")
        if int(summary.get("mcp_call_count", -1)) != 6:
            raise RuntimeError("training route must call MCP once per valid depth1 anchor")
        if summary.get("selected_output") != {
            "depth": 1,
            "anchor_index": CURRENT_CHUNK_INDEX,
            "target_chunk_index": FUTURE_CHUNK_INDEX,
        }:
            raise RuntimeError("training route selected output mismatch")


def _model_with_block_mask(generator: Any) -> Any:
    model = getattr(generator, "model", None)
    if model is None or not hasattr(model, "block_mask"):
        raise RuntimeError("route equivalence audit requires generator.model.block_mask")
    return model


def _require_model_block_mask_none(generator: Any, *, label: str) -> bool:
    model = _model_with_block_mask(generator)
    if model.block_mask is not None:
        raise RuntimeError(f"{label} block_mask must be None")
    return True


def _input_tensor_provenance(
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
) -> dict[str, Any]:
    return {
        "teacher_chunk0_sha256": tensor_sha256(
            _chunk(teacher_target, HISTORY_CHUNK_INDEX).detach().cpu()
        ),
        "teacher_chunk1_sha256": tensor_sha256(
            _chunk(teacher_target, CURRENT_CHUNK_INDEX).detach().cpu()
        ),
        "teacher_chunk2_sha256": tensor_sha256(
            _chunk(teacher_target, FUTURE_CHUNK_INDEX).detach().cpu()
        ),
        "source_noise_chunk1_sha256": tensor_sha256(
            _chunk(source_noise, CURRENT_CHUNK_INDEX).detach().cpu()
        ),
        "source_noise_chunk2_sha256": tensor_sha256(
            _chunk(source_noise, FUTURE_CHUNK_INDEX).detach().cpu()
        ),
        "source_noise_sha256": tensor_sha256(source_noise.detach().cpu()),
        "teacher_target_sha256": tensor_sha256(teacher_target.detach().cpu()),
    }


def _validate_source_and_teacher(
    source_noise: torch.Tensor,
    teacher_target: torch.Tensor,
) -> None:
    if not torch.is_tensor(source_noise) or not torch.is_tensor(teacher_target):
        raise TypeError("source_noise and teacher_target must be tensors")
    if tuple(source_noise.shape) != tuple(teacher_target.shape):
        raise RuntimeError("source_noise and teacher_target shapes must match")
    if source_noise.ndim != 5:
        raise RuntimeError("source_noise must have shape [B,21,C,H,W]")
    if int(source_noise.shape[1]) != FULL_SEQUENCE_FRAME_COUNT:
        raise RuntimeError("route equivalence audit requires 21 latent frames")
    _require_finite_tensor(source_noise, name="source_noise")
    _require_finite_tensor(teacher_target, name="teacher_target")


def _chunk(tensor: torch.Tensor, chunk_index: int) -> torch.Tensor:
    start = int(chunk_index) * FULL_SEQUENCE_CHUNK_FRAMES
    return tensor[:, start:start + FULL_SEQUENCE_CHUNK_FRAMES]


def _add_noise_chunk(
    scheduler: Any,
    *,
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
    name: str,
) -> torch.Tensor:
    _require_finite_tensor(clean, name=f"{name}_clean")
    _require_finite_tensor(noise, name=f"{name}_noise")
    _require_finite_tensor(timestep, name=f"{name}_timestep")
    original_shape = clean.shape
    value = scheduler.add_noise(
        clean.flatten(0, 1),
        noise.flatten(0, 1),
        timestep.flatten(0, 1),
    ).unflatten(0, original_shape[:2])
    value = value.to(device=clean.device, dtype=clean.dtype)
    _require_finite_tensor(value, name=name)
    return value


def _training_target_chunk(
    scheduler: Any,
    *,
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
    name: str,
) -> torch.Tensor:
    _require_finite_tensor(clean, name=f"{name}_clean")
    _require_finite_tensor(noise, name=f"{name}_noise")
    _require_finite_tensor(timestep, name=f"{name}_timestep")
    original_shape = clean.shape
    value = scheduler.training_target(
        clean.flatten(0, 1),
        noise.flatten(0, 1),
        timestep.flatten(0, 1),
    ).unflatten(0, original_shape[:2])
    value = value.to(device=clean.device, dtype=clean.dtype)
    _require_finite_tensor(value, name=name)
    return value


def _flow_to_x0_chunk(
    scheduler: Any,
    *,
    flow: torch.Tensor,
    state: torch.Tensor,
    timestep: torch.Tensor,
    name: str,
) -> torch.Tensor:
    _require_finite_tensor(flow, name=f"{name}_flow")
    _require_finite_tensor(state, name=f"{name}_state")
    _require_finite_tensor(timestep, name=f"{name}_timestep")
    original_shape = state.shape
    value = scheduler.step(
        flow.flatten(0, 1),
        timestep.flatten(0, 1),
        state.flatten(0, 1),
        to_final=True,
    ).unflatten(0, original_shape[:2])
    value = value.to(device=state.device, dtype=state.dtype)
    _require_finite_tensor(value, name=name)
    return value


def _timestep(value: float, target: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(torch.tensor(float(value), dtype=torch.float32)).item()):
        raise RuntimeError("route equivalence timestep must be finite")
    timestep = torch.full(
        target.shape[:2],
        float(value),
        device=target.device,
        dtype=torch.float32,
    )
    _require_finite_tensor(timestep, name="timestep")
    return timestep


def _tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    _require_finite_tensor(tensor, name="recorded_tensor")
    summary = tensor_summary(tensor.detach().cpu())
    value = tensor.detach().float()
    return {
        "shape": summary["shape"],
        "dtype": summary["dtype"],
        "finite": summary["finite"],
        "sha256": summary["sha256"],
        "mean_abs": float(value.abs().mean().item()),
        "max_abs": float(value.abs().max().item()),
    }


def _mse(left: torch.Tensor, right: torch.Tensor) -> float:
    _require_finite_tensor(left, name="mse_left")
    _require_finite_tensor(right, name="mse_right")
    if tuple(left.shape) != tuple(right.shape):
        raise RuntimeError("MSE tensor shape mismatch")
    value = (left.detach().float() - right.detach().float()).square().mean()
    _require_finite_tensor(value, name="mse_value")
    return float(value.item())


def _require_finite_tensor(tensor: torch.Tensor, *, name: str) -> None:
    deployment._ensure_finite_tensor(tensor, name=f"route_equivalence_{name}")


def _clone_tensor(tensor: Any, *, name: str) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a tensor")
    _require_finite_tensor(tensor, name=name)
    return tensor.detach().clone()


def _clone_grid(value: Any) -> Any:
    if torch.is_tensor(value):
        _require_finite_tensor(value, name="future_grid_size")
        return value.detach().clone()
    if isinstance(value, (list, tuple)):
        return tuple(_clone_grid(child) for child in value)
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    raise TypeError("unsupported future_grid_size value")


def _grid_record(value: Any) -> Any:
    if torch.is_tensor(value):
        return {
            "kind": "tensor",
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
            "sha256": tensor_sha256(value.detach().cpu()),
            "values": value.detach().cpu().tolist(),
        }
    if isinstance(value, tuple):
        return [_grid_record(child) for child in value]
    if isinstance(value, list):
        return [_grid_record(child) for child in value]
    return value


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _output_field(outputs: Any, name: str) -> Any:
    if isinstance(outputs, Mapping):
        return outputs[name]
    return getattr(outputs, name)


__all__ = [
    "CURRENT_CHUNK_INDEX",
    "FIRST_MCP_ROUTE_EQUIVALENCE_SCHEMA",
    "FIRST_MCP_ROUTE_EQUIVALENCE_TENSOR_SCHEMA",
    "FUTURE_CHUNK_INDEX",
    "HISTORY_CHUNK_INDEX",
    "POINT_DEPLOYMENT_ENDPOINT",
    "POINT_TRAINING_EDGE",
    "RAW_DEPLOYMENT_ENDPOINT",
    "RAW_TRAINING_EDGE",
    "FirstMCPRouteEquivalenceResult",
    "build_flow_match_scheduler",
    "build_route_equivalence_point",
    "raw_timestep_in_training_support",
    "run_first_mcp_route_equivalence_audit",
    "validate_first_mcp_route_equivalence_manifest",
]
