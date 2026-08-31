from dataclasses import dataclass
from typing import Iterable, Literal

import torch
import torch.nn.functional as F

from utils.nf_sf_tensors import (
    DEFAULT_S_MAIN,
    DEFAULT_S_MCP,
    FULL_SEQUENCE_CHUNK_FRAMES,
    FULL_SEQUENCE_DEPTHS,
    FULL_SEQUENCE_FRAME_COUNT,
    FULL_SEQUENCE_NUM_CHUNKS,
    FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION,
    LATENT_FRAME_AXIS,
    full_sequence_mcp_anchor_counts,
    sample_nf_sf_noise_and_timesteps,
    sample_nf_sf_full_sequence_noise_and_timesteps,
)


NFSFTrainMode = Literal["frozen", "joint"]
NFSFFullSequenceObjectiveMode = Literal["next_forcing_full", "main_only_full_control"]

FULL_SEQUENCE_TRAINER_SCHEMA = "nf_sf_full_sequence_next_forcing_trainer_v1"
FULL_SEQUENCE_RUN_KIND = "nf_sf_full_sequence_next_forcing_v1"
FULL_SEQUENCE_OBJECTIVE_VERSION = "nf_sf_full_sequence_next_forcing_objective_v1"
FULL_SEQUENCE_PAPER_REFERENCE = "Next Forcing arXiv:2606.11187"
FULL_SEQUENCE_FUTURE_EMBEDDING_ORDER = "depth_major"
FULL_SEQUENCE_FRAME_SEQ_LENGTH = 1560
FULL_SEQUENCE_CHUNK_TOKENS = FULL_SEQUENCE_CHUNK_FRAMES * FULL_SEQUENCE_FRAME_SEQ_LENGTH
FULL_SEQUENCE_DEPTH_WEIGHTS = (0.5, 0.2, 0.1)
FULL_SEQUENCE_CHECKPOINT_STEPS = (0, 500, 2000, 5000)
FULL_SEQUENCE_TARGET_GLOBAL_STEP = 5000
OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256 = (
    "a0413986d9734e02c09504e1520f5697ba6df731bb2f0f35577485e9cc8f56a3"
)


@dataclass(frozen=True)
class NFSFSelectedState:
    clean_history: torch.Tensor | None
    current_target: torch.Tensor
    future_targets: tuple[torch.Tensor, ...]
    future_valid_masks: tuple[torch.Tensor, ...] | None = None
    current_start_frame: int | None = None


@dataclass(frozen=True)
class NFSFNoisyBatch:
    state: NFSFSelectedState
    noisy_current: torch.Tensor
    noisy_futures: tuple[torch.Tensor, ...]
    timestep_main: torch.Tensor
    timestep_depths: tuple[torch.Tensor, ...]
    epsilon_main: torch.Tensor
    epsilon_depths: tuple[torch.Tensor, ...]
    target_flow_main: torch.Tensor
    target_flow_depths: tuple[torch.Tensor, ...]
    future_valid_masks: tuple[torch.Tensor, ...]
    future_start_frames: tuple[int, ...]


@dataclass(frozen=True)
class NFSFFullSequenceMCPAnchorSpec:
    depth: int
    anchor_index: int
    target_chunk_index: int
    future_start_frame: int
    flat_index: int


@dataclass(frozen=True)
class NFSFFullSequenceNoisyBatch:
    clean_target: torch.Tensor
    noisy_main: torch.Tensor
    target_flow_main: torch.Tensor
    epsilon_main: torch.Tensor
    raw_timestep_main: torch.Tensor
    timestep_main: torch.Tensor
    noisy_mcp_depths: tuple[torch.Tensor, ...]
    target_flow_mcp_depths: tuple[torch.Tensor, ...]
    epsilon_mcp_depths: tuple[torch.Tensor, ...]
    raw_timestep_mcp_depths: tuple[torch.Tensor, ...]
    timestep_mcp_depths: tuple[torch.Tensor, ...]
    anchor_specs: tuple[NFSFFullSequenceMCPAnchorSpec, ...]
    future_embedding_order: str = FULL_SEQUENCE_FUTURE_EMBEDDING_ORDER
    rng_draw_order_version: str = FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION


@dataclass(frozen=True)
class NFSFLossBreakdown:
    total_loss: torch.Tensor
    main_loss: torch.Tensor
    mcp_depth_losses: tuple[torch.Tensor, ...]

    def log_dict(self) -> dict[str, torch.Tensor]:
        log = {
            "total_loss": self.total_loss.detach(),
            "main_loss": self.main_loss.detach(),
        }
        for index, loss in enumerate(self.mcp_depth_losses, start=1):
            log[f"mcp_depth{index}_loss"] = loss.detach()
        return log


@dataclass(frozen=True)
class NFSFFullSequenceLossBreakdown:
    total_loss: torch.Tensor
    main_loss: torch.Tensor
    mcp_depth_losses: tuple[torch.Tensor, ...]
    main_chunk_losses: tuple[torch.Tensor, ...]
    mcp_anchor_losses: tuple[tuple[torch.Tensor, ...], ...]
    objective_mode: NFSFFullSequenceObjectiveMode

    def log_dict(self) -> dict[str, torch.Tensor]:
        log = {
            "total_loss": self.total_loss.detach(),
            "main_loss": self.main_loss.detach(),
        }
        for index, loss in enumerate(self.main_chunk_losses):
            log[f"main_chunk{index}_loss"] = loss.detach()
        for depth_index, loss in enumerate(self.mcp_depth_losses, start=1):
            log[f"mcp_depth{depth_index}_loss"] = loss.detach()
        for depth_index, anchor_losses in enumerate(self.mcp_anchor_losses, start=1):
            for anchor_index, loss in enumerate(anchor_losses):
                log[f"mcp_depth{depth_index}_anchor{anchor_index}_loss"] = loss.detach()
        return log


@dataclass(frozen=True)
class NFSFForwardResult:
    noisy_batch: NFSFNoisyBatch
    main_flow_pred: torch.Tensor
    mcp_flow_preds: tuple[torch.Tensor, ...]
    losses: NFSFLossBreakdown


@dataclass(frozen=True)
class NFSFFullSequenceForwardResult:
    noisy_batch: NFSFFullSequenceNoisyBatch
    main_flow_pred: torch.Tensor
    mcp_flow_preds_by_depth: tuple[torch.Tensor, ...]
    losses: NFSFFullSequenceLossBreakdown
    tap_shapes: tuple[tuple[int, ...], ...]
    anchor_token_slices: tuple[tuple[int, int], ...]
    main_backbone_forward_count: int
    future_embedding_order: str | None = None


@dataclass(frozen=True)
class NFSFMCP1GridPointResult:
    loss: torch.Tensor
    metadata: dict


@dataclass(frozen=True)
class NFSFParamAudit:
    name: str
    parameter_names: tuple[str, ...]
    tensor_count: int
    trainable_parameter_count: int
    requires_grad: bool
    in_optimizer: bool


@dataclass(frozen=True)
class NFSFOptimizerPlan:
    mode: NFSFTrainMode
    optimizer_param_groups: list[dict]
    audits: tuple[NFSFParamAudit, ...]


def validate_nf_sf_full_sequence_objective_mode(
    objective_mode: str,
) -> NFSFFullSequenceObjectiveMode:
    if objective_mode not in ("next_forcing_full", "main_only_full_control"):
        raise ValueError(
            "objective_mode must be 'next_forcing_full' or "
            "'main_only_full_control'"
        )
    return objective_mode  # type: ignore[return-value]


def full_sequence_anchor_token_slice(
    anchor_index: int,
    *,
    frame_seq_length: int = FULL_SEQUENCE_FRAME_SEQ_LENGTH,
    chunk_frames: int = FULL_SEQUENCE_CHUNK_FRAMES,
) -> slice:
    anchor = int(anchor_index)
    if anchor < 0:
        raise ValueError("anchor_index must be non-negative")
    if frame_seq_length <= 0:
        raise ValueError("frame_seq_length must be positive")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    chunk_tokens = frame_seq_length * chunk_frames
    return slice(anchor * chunk_tokens, (anchor + 1) * chunk_tokens)


def build_full_sequence_mcp_anchor_specs(
    *,
    num_chunks: int = FULL_SEQUENCE_NUM_CHUNKS,
    chunk_frames: int = FULL_SEQUENCE_CHUNK_FRAMES,
    depths: Iterable[int] = FULL_SEQUENCE_DEPTHS,
    future_embedding_order: str = FULL_SEQUENCE_FUTURE_EMBEDDING_ORDER,
) -> tuple[NFSFFullSequenceMCPAnchorSpec, ...]:
    if future_embedding_order != FULL_SEQUENCE_FUTURE_EMBEDDING_ORDER:
        raise ValueError(
            f"future_embedding_order must be {FULL_SEQUENCE_FUTURE_EMBEDDING_ORDER!r}"
        )
    if num_chunks <= 0:
        raise ValueError("num_chunks must be positive")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    normalized_depths = tuple(int(depth) for depth in depths)
    if normalized_depths != FULL_SEQUENCE_DEPTHS:
        raise ValueError("full-sequence v1 requires depths=(1, 2, 3)")

    specs: list[NFSFFullSequenceMCPAnchorSpec] = []
    flat_index = 0
    for depth in normalized_depths:
        valid_anchor_count = max(num_chunks - depth, 0)
        for anchor_index in range(valid_anchor_count):
            target_chunk_index = anchor_index + depth
            specs.append(
                NFSFFullSequenceMCPAnchorSpec(
                    depth=depth,
                    anchor_index=anchor_index,
                    target_chunk_index=target_chunk_index,
                    future_start_frame=target_chunk_index * chunk_frames,
                    flat_index=flat_index,
                )
            )
            flat_index += 1
    return tuple(specs)


def prepare_nf_sf_full_sequence_noisy_batch(
    clean_target: torch.Tensor,
    *,
    scheduler_main,
    scheduler_mcp,
    rng: torch.Generator,
    chunk_frames: int = FULL_SEQUENCE_CHUNK_FRAMES,
    depths: Iterable[int] = FULL_SEQUENCE_DEPTHS,
    s_main: float = DEFAULT_S_MAIN,
    s_mcp: float = DEFAULT_S_MCP,
) -> NFSFFullSequenceNoisyBatch:
    _validate_full_sequence_clean_target(clean_target, chunk_frames=chunk_frames)
    normalized_depths = tuple(int(depth) for depth in depths)
    if normalized_depths != FULL_SEQUENCE_DEPTHS:
        raise ValueError("full-sequence v1 requires depths=(1, 2, 3)")
    samples = sample_nf_sf_full_sequence_noise_and_timesteps(
        clean_target,
        chunk_frames=chunk_frames,
        depths=normalized_depths,
        s_main=s_main,
        s_mcp=s_mcp,
        generator=rng,
    )
    noisy_main = _add_noise_like_scheduler(
        scheduler_main,
        clean_target,
        samples.epsilon_main,
        samples.timestep_main,
    )
    target_flow_main = _training_target_like_scheduler(
        scheduler_main,
        clean_target,
        samples.epsilon_main,
        samples.timestep_main,
    )

    target_mcp_depths = _full_sequence_mcp_targets(
        clean_target,
        chunk_frames=chunk_frames,
        depths=normalized_depths,
    )
    noisy_mcp_depths = []
    target_flow_mcp_depths = []
    for target, epsilon, timestep in zip(
        target_mcp_depths,
        samples.epsilon_mcp_depths,
        samples.timestep_mcp_depths,
    ):
        noisy_mcp_depths.append(
            _add_noise_for_anchor_chunks(
                scheduler_mcp,
                target,
                epsilon,
                timestep,
            )
        )
        target_flow_mcp_depths.append(
            _training_target_for_anchor_chunks(
                scheduler_mcp,
                target,
                epsilon,
                timestep,
            )
        )

    return NFSFFullSequenceNoisyBatch(
        clean_target=clean_target,
        noisy_main=noisy_main,
        target_flow_main=target_flow_main,
        epsilon_main=samples.epsilon_main,
        raw_timestep_main=samples.raw_timestep_main,
        timestep_main=samples.timestep_main,
        noisy_mcp_depths=tuple(noisy_mcp_depths),
        target_flow_mcp_depths=tuple(target_flow_mcp_depths),
        epsilon_mcp_depths=samples.epsilon_mcp_depths,
        raw_timestep_mcp_depths=samples.raw_timestep_mcp_depths,
        timestep_mcp_depths=samples.timestep_mcp_depths,
        anchor_specs=build_full_sequence_mcp_anchor_specs(
            num_chunks=clean_target.shape[LATENT_FRAME_AXIS] // chunk_frames,
            chunk_frames=chunk_frames,
            depths=normalized_depths,
        ),
    )


def build_full_sequence_mcp_anchor_inputs(
    noisy_batch: NFSFFullSequenceNoisyBatch,
) -> tuple[dict, ...]:
    specs_by_anchor: dict[int, list[NFSFFullSequenceMCPAnchorSpec]] = {}
    for spec in noisy_batch.anchor_specs:
        specs_by_anchor.setdefault(spec.anchor_index, []).append(spec)

    anchors = []
    for anchor_index in range(FULL_SEQUENCE_NUM_CHUNKS):
        specs = tuple(sorted(specs_by_anchor.get(anchor_index, ()), key=lambda item: item.depth))
        if not specs:
            continue
        future_noises = []
        future_start_frames = []
        timesteps = []
        depths = []
        flat_indices = []
        for spec in specs:
            depth_offset = spec.depth - 1
            future_noises.append(
                noisy_batch.noisy_mcp_depths[depth_offset][:, spec.anchor_index]
            )
            future_start_frames.append(spec.future_start_frame)
            timesteps.append(
                noisy_batch.timestep_mcp_depths[depth_offset][:, spec.anchor_index]
            )
            depths.append(spec.depth)
            flat_indices.append(spec.flat_index)
        anchors.append(
            {
                "anchor_index": anchor_index,
                "depths": tuple(depths),
                "future_noises": tuple(future_noises),
                "future_start_frames": tuple(future_start_frames),
                "timesteps": tuple(timesteps),
                "flat_indices": tuple(flat_indices),
            }
        )
    return tuple(anchors)


def compute_nf_sf_full_sequence_losses(
    *,
    main_flow_pred: torch.Tensor,
    mcp_flow_preds_by_depth: tuple[torch.Tensor, ...],
    noisy_batch: NFSFFullSequenceNoisyBatch,
    depth_weights: tuple[float, ...] = FULL_SEQUENCE_DEPTH_WEIGHTS,
    objective_mode: NFSFFullSequenceObjectiveMode = "next_forcing_full",
) -> NFSFFullSequenceLossBreakdown:
    objective_mode = validate_nf_sf_full_sequence_objective_mode(objective_mode)
    if tuple(main_flow_pred.shape) != tuple(noisy_batch.target_flow_main.shape):
        raise ValueError(
            "main_flow_pred shape mismatch: "
            f"{tuple(main_flow_pred.shape)} != {tuple(noisy_batch.target_flow_main.shape)}"
        )
    clean = noisy_batch.clean_target
    chunk_frames = FULL_SEQUENCE_CHUNK_FRAMES
    num_frames = clean.shape[LATENT_FRAME_AXIS]
    num_chunks = num_frames // chunk_frames
    if num_chunks != FULL_SEQUENCE_NUM_CHUNKS:
        raise ValueError("full-sequence v1 requires exactly 7 chunks")
    main_pred_chunks = main_flow_pred.unflatten(1, (num_chunks, chunk_frames))
    main_target_chunks = noisy_batch.target_flow_main.unflatten(
        1,
        (num_chunks, chunk_frames),
    )
    main_chunk_losses = tuple(
        F.mse_loss(
            main_pred_chunks[:, index].float(),
            main_target_chunks[:, index].float(),
            reduction="mean",
        )
        for index in range(num_chunks)
    )
    main_loss = torch.stack(main_chunk_losses).mean()
    total_loss = main_loss

    if objective_mode == "main_only_full_control":
        if mcp_flow_preds_by_depth:
            raise ValueError("main_only_full_control must not receive MCP predictions")
        return NFSFFullSequenceLossBreakdown(
            total_loss=total_loss,
            main_loss=main_loss,
            mcp_depth_losses=(),
            main_chunk_losses=main_chunk_losses,
            mcp_anchor_losses=(),
            objective_mode=objective_mode,
        )

    if len(mcp_flow_preds_by_depth) != len(noisy_batch.target_flow_mcp_depths):
        raise ValueError("MCP prediction depth count mismatch")
    if len(depth_weights) < len(mcp_flow_preds_by_depth):
        raise ValueError("depth_weights must cover every MCP depth")

    mcp_depth_losses = []
    mcp_anchor_losses = []
    for depth_index, (pred, target) in enumerate(
        zip(mcp_flow_preds_by_depth, noisy_batch.target_flow_mcp_depths),
        start=1,
    ):
        if tuple(pred.shape) != tuple(target.shape):
            raise ValueError(
                f"MCP depth {depth_index} shape mismatch: "
                f"{tuple(pred.shape)} != {tuple(target.shape)}"
            )
        anchor_count = pred.shape[1]
        losses = tuple(
            F.mse_loss(
                pred[:, anchor_index].float(),
                target[:, anchor_index].float(),
                reduction="mean",
            )
            for anchor_index in range(anchor_count)
        )
        if not losses:
            depth_loss = pred.sum() * 0.0
        else:
            depth_loss = torch.stack(losses).mean()
        mcp_anchor_losses.append(losses)
        mcp_depth_losses.append(depth_loss)
        total_loss = total_loss + float(depth_weights[depth_index - 1]) * depth_loss

    return NFSFFullSequenceLossBreakdown(
        total_loss=total_loss,
        main_loss=main_loss,
        mcp_depth_losses=tuple(mcp_depth_losses),
        main_chunk_losses=main_chunk_losses,
        mcp_anchor_losses=tuple(mcp_anchor_losses),
        objective_mode=objective_mode,
    )


def run_nf_sf_full_sequence_forward_loss(
    generator,
    *,
    conditional_dict: dict,
    noisy_batch: NFSFFullSequenceNoisyBatch,
    depth_weights: Iterable[float] = FULL_SEQUENCE_DEPTH_WEIGHTS,
    objective_mode: NFSFFullSequenceObjectiveMode = "next_forcing_full",
) -> NFSFFullSequenceForwardResult:
    objective_mode = validate_nf_sf_full_sequence_objective_mode(objective_mode)
    mcp_anchor_inputs = (
        build_full_sequence_mcp_anchor_inputs(noisy_batch)
        if objective_mode == "next_forcing_full"
        else ()
    )
    outputs = generator.forward_full_sequence_next_forcing(
        noisy_image_or_video=noisy_batch.noisy_main,
        clean_x=noisy_batch.clean_target,
        conditional_dict=conditional_dict,
        timestep_main=noisy_batch.timestep_main,
        mcp_anchor_inputs=mcp_anchor_inputs,
    )
    main_flow_pred = _output_field(outputs, "main_flow_pred")
    mcp_flow_preds_by_depth = tuple(
        _output_field(outputs, "mcp_flow_preds_by_depth")
    )
    tap_shapes = tuple(_output_field(outputs, "tap_shapes"))
    anchor_token_slices = tuple(_output_field(outputs, "anchor_token_slices"))
    main_backbone_forward_count = int(
        _output_field(outputs, "main_backbone_forward_count")
    )
    future_embedding_order = _output_field(outputs, "future_embedding_order")
    losses = compute_nf_sf_full_sequence_losses(
        main_flow_pred=main_flow_pred,
        mcp_flow_preds_by_depth=mcp_flow_preds_by_depth,
        noisy_batch=noisy_batch,
        depth_weights=tuple(float(weight) for weight in depth_weights),
        objective_mode=objective_mode,
    )
    return NFSFFullSequenceForwardResult(
        noisy_batch=noisy_batch,
        main_flow_pred=main_flow_pred,
        mcp_flow_preds_by_depth=mcp_flow_preds_by_depth,
        losses=losses,
        tap_shapes=tap_shapes,
        anchor_token_slices=anchor_token_slices,
        main_backbone_forward_count=main_backbone_forward_count,
        future_embedding_order=future_embedding_order,
    )


def prepare_nf_sf_noisy_batch(
    state: NFSFSelectedState,
    *,
    scheduler_main,
    scheduler_mcp,
    rng: torch.Generator,
    chunk_frames: int = 3,
    depths: Iterable[int] = (1, 2, 3),
    s_main: float = DEFAULT_S_MAIN,
    s_mcp: float = DEFAULT_S_MCP,
) -> NFSFNoisyBatch:
    depths = tuple(int(depth) for depth in depths)
    _validate_selected_state(state, chunk_frames=chunk_frames, depths=depths)

    samples = sample_nf_sf_noise_and_timesteps(
        state.current_target,
        chunk_frames=chunk_frames,
        depths=depths,
        s_main=s_main,
        s_mcp=s_mcp,
        generator=rng,
    )
    noisy_current = _add_noise_like_scheduler(
        scheduler_main,
        state.current_target,
        samples.epsilon_main,
        samples.timestep_main,
    )
    target_flow_main = _training_target_like_scheduler(
        scheduler_main,
        state.current_target,
        samples.epsilon_main,
        samples.timestep_main,
    )

    noisy_futures = []
    target_flow_depths = []
    for target, epsilon, timestep in zip(
        state.future_targets,
        samples.epsilon_depths,
        samples.timestep_depths,
    ):
        noisy_futures.append(
            _add_noise_like_scheduler(scheduler_mcp, target, epsilon, timestep)
        )
        target_flow_depths.append(
            _training_target_like_scheduler(scheduler_mcp, target, epsilon, timestep)
        )

    valid_masks = _future_masks_for(state, chunk_frames=chunk_frames)
    current_start = _current_start_frame(state)
    return NFSFNoisyBatch(
        state=state,
        noisy_current=noisy_current,
        noisy_futures=tuple(noisy_futures),
        timestep_main=samples.timestep_main,
        timestep_depths=samples.timestep_depths,
        epsilon_main=samples.epsilon_main,
        epsilon_depths=samples.epsilon_depths,
        target_flow_main=target_flow_main,
        target_flow_depths=tuple(target_flow_depths),
        future_valid_masks=valid_masks,
        future_start_frames=tuple(
            current_start + depth * chunk_frames for depth in depths
        ),
    )


def run_nf_sf_forward_loss(
    generator,
    *,
    conditional_dict: dict,
    noisy_batch: NFSFNoisyBatch,
    depth_weights: Iterable[float] = (0.5, 0.2, 0.1),
) -> NFSFForwardResult:
    clean_history = noisy_batch.state.clean_history
    clean_kwargs = {}
    if clean_history is not None and clean_history.shape[1] > 0:
        clean_kwargs = {
            "clean_x": clean_history,
            "aug_t": torch.zeros_like(noisy_batch.timestep_main),
        }

    outputs = generator(
        noisy_image_or_video=noisy_batch.noisy_current,
        conditional_dict=conditional_dict,
        timestep=noisy_batch.timestep_main,
        mcp_future_noises=list(noisy_batch.noisy_futures),
        mcp_future_start_frames=list(noisy_batch.future_start_frames),
        mcp_timesteps=list(noisy_batch.timestep_depths),
        **clean_kwargs,
    )
    if len(outputs) != 3:
        raise RuntimeError("NF-SF M2 forward expected main flow, x0, and MCP flows")
    main_flow_pred, _, mcp_flow_preds = outputs
    losses = compute_nf_sf_losses(
        main_flow_pred=main_flow_pred,
        mcp_flow_preds=tuple(mcp_flow_preds),
        noisy_batch=noisy_batch,
        depth_weights=tuple(float(weight) for weight in depth_weights),
    )
    return NFSFForwardResult(
        noisy_batch=noisy_batch,
        main_flow_pred=main_flow_pred,
        mcp_flow_preds=tuple(mcp_flow_preds),
        losses=losses,
    )


def run_nf_sf_mcp1_grid_point_loss(
    generator,
    *,
    conditional_dict: dict,
    state: NFSFSelectedState,
    scheduler,
    epsilon_main: torch.Tensor,
    epsilon_future: torch.Tensor,
    timestep: torch.Tensor,
    chunk_frames: int = 3,
) -> NFSFMCP1GridPointResult:
    if chunk_frames != 3:
        raise ValueError("MCP-1 grid auxiliary loss requires chunk_frames=3")
    if state.clean_history is None:
        raise ValueError("MCP-1 grid auxiliary loss requires clean history")
    if len(state.future_targets) < 1:
        raise ValueError("MCP-1 grid auxiliary loss requires next1 target")
    current = state.current_target
    next1 = state.future_targets[0]
    _validate_chunk_tensor(current, chunk_frames=chunk_frames, name="current_target")
    _validate_chunk_tensor(next1, chunk_frames=chunk_frames, name="future1")
    if tuple(next1.shape) != tuple(current.shape):
        raise ValueError("future1 shape must match current_target")
    if state.clean_history.ndim != 5 or state.clean_history.shape[1] != chunk_frames:
        raise ValueError("MCP-1 grid auxiliary loss requires one clean history chunk")
    if tuple(epsilon_main.shape) != tuple(current.shape):
        raise ValueError("epsilon_main shape must match current_target")
    if tuple(epsilon_future.shape) != tuple(next1.shape):
        raise ValueError("epsilon_future shape must match next1 target")
    if epsilon_main.dtype != current.dtype:
        raise ValueError("epsilon_main dtype must match current_target")
    if epsilon_future.dtype != next1.dtype:
        raise ValueError("epsilon_future dtype must match next1 target")
    _ensure_finite_tensor(epsilon_main, name="epsilon_main")
    _ensure_finite_tensor(epsilon_future, name="epsilon_future")

    timestep_chunk = _expand_timestep_to_chunk(
        timestep,
        target=current,
        name="mcp1_grid_timestep",
    )
    noisy_current = _add_noise_like_scheduler(
        scheduler,
        current,
        epsilon_main,
        timestep_chunk,
    )
    noisy_future = _add_noise_like_scheduler(
        scheduler,
        next1,
        epsilon_future,
        timestep_chunk,
    )
    target_flow = _training_target_like_scheduler(
        scheduler,
        next1,
        epsilon_future,
        timestep_chunk,
    )
    _ensure_finite_tensor(noisy_current, name="mcp1_grid_noisy_current")
    _ensure_finite_tensor(noisy_future, name="mcp1_grid_noisy_future")
    _ensure_finite_tensor(target_flow, name="mcp1_grid_target_flow")

    current_start = _current_start_frame(state)
    future_start = current_start + chunk_frames
    if future_start != 6:
        raise ValueError("MCP-1 grid auxiliary loss requires future_start_frame=6")

    outputs = generator(
        noisy_image_or_video=noisy_current,
        conditional_dict=conditional_dict,
        timestep=timestep_chunk,
        clean_x=state.clean_history,
        aug_t=torch.zeros_like(timestep_chunk),
        mcp_future_noises=[noisy_future],
        mcp_future_start_frames=[future_start],
        mcp_timesteps=[timestep_chunk],
    )
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise RuntimeError("MCP-1 grid auxiliary loss expected generator output triple")
    mcp_outputs = outputs[2]
    if not isinstance(mcp_outputs, (tuple, list)) or len(mcp_outputs) != 1:
        count = "non-sequence" if not isinstance(mcp_outputs, (tuple, list)) else len(mcp_outputs)
        raise RuntimeError(
            "MCP-1 grid auxiliary loss expected exactly one MCP flow output, "
            f"got {count}"
        )
    flow_pred = mcp_outputs[0]
    if not torch.is_tensor(flow_pred):
        raise TypeError("MCP-1 grid auxiliary flow output must be a tensor")
    if tuple(flow_pred.shape) != tuple(target_flow.shape):
        raise ValueError(
            "MCP-1 grid auxiliary flow shape mismatch: "
            f"{tuple(flow_pred.shape)} != {tuple(target_flow.shape)}"
        )
    if flow_pred.dtype != target_flow.dtype:
        raise ValueError(
            "MCP-1 grid auxiliary flow dtype mismatch: "
            f"{flow_pred.dtype} != {target_flow.dtype}"
        )
    _ensure_finite_tensor(flow_pred, name="mcp1_grid_flow_pred")
    loss = F.mse_loss(flow_pred.float(), target_flow.float(), reduction="mean")
    if not bool(torch.isfinite(loss.detach()).all().item()):
        raise RuntimeError("MCP-1 grid auxiliary loss is non-finite")
    timestep_value = float(timestep_chunk.detach().float()[0, 0].item())
    return NFSFMCP1GridPointResult(
        loss=loss,
        metadata={
            "timestep": timestep_value,
            "future_start_frame": int(future_start),
            "loss": float(loss.detach().float().item()),
            "flow_shape": [int(dim) for dim in flow_pred.shape],
            "flow_dtype": str(flow_pred.dtype),
            "target_flow_dtype": str(target_flow.dtype),
            "finite": True,
        },
    )


def compute_nf_sf_losses(
    *,
    main_flow_pred: torch.Tensor,
    mcp_flow_preds: tuple[torch.Tensor, ...],
    noisy_batch: NFSFNoisyBatch,
    depth_weights: tuple[float, ...] = (0.5, 0.2, 0.1),
) -> NFSFLossBreakdown:
    if len(mcp_flow_preds) != len(noisy_batch.target_flow_depths):
        raise ValueError("mcp_flow_preds must align 1:1 with target_flow_depths")
    if len(depth_weights) < len(mcp_flow_preds):
        raise ValueError("depth_weights must cover every MCP depth")

    main_loss = F.mse_loss(
        main_flow_pred.float(),
        noisy_batch.target_flow_main.float(),
        reduction="mean",
    )
    mcp_losses = []
    total_loss = main_loss
    for index, (pred, target, mask) in enumerate(
        zip(
            mcp_flow_preds,
            noisy_batch.target_flow_depths,
            noisy_batch.future_valid_masks,
        )
    ):
        if pred.shape != target.shape:
            raise ValueError(
                f"MCP depth {index + 1} shape mismatch: "
                f"{tuple(pred.shape)} != {tuple(target.shape)}"
            )
        if mask.shape != (target.shape[1],):
            raise ValueError(
                f"MCP depth {index + 1} mask shape {tuple(mask.shape)} "
                f"does not match frame count {target.shape[1]}"
            )
        if bool(mask.any()):
            loss = F.mse_loss(
                pred[:, mask].float(),
                target[:, mask].float(),
                reduction="mean",
            )
        else:
            loss = pred.sum() * 0.0
        mcp_losses.append(loss)
        total_loss = total_loss + depth_weights[index] * loss

    return NFSFLossBreakdown(
        total_loss=total_loss,
        main_loss=main_loss,
        mcp_depth_losses=tuple(mcp_losses),
    )


def configure_nf_sf_optimizer_plan(
    generator,
    *,
    mode: NFSFTrainMode,
    lr: float | None = None,
    group_lrs: dict[str, float] | None = None,
) -> NFSFOptimizerPlan:
    if mode not in ("frozen", "joint"):
        raise ValueError("mode must be 'frozen' or 'joint'")
    if lr is None and group_lrs is None:
        raise ValueError("either lr or group_lrs must be provided")
    if getattr(generator, "mcp", None) is None:
        raise ValueError("generator must have MCP modules attached")

    generator.requires_grad_(False)
    groups = collect_nf_sf_parameter_groups(generator)
    trainable_names = {"mcp_fusion"}
    trainable_names.update(
        name for name in groups.keys() if name.startswith("mcp_depth")
    )
    if mode == "joint":
        trainable_names.update({"backbone", "patch_embedding"})

    optimizer_param_groups = []
    optimizer_param_ids = set()
    for name, named_params in groups.items():
        requires_grad = name in trainable_names
        for _, param in named_params:
            param.requires_grad_(requires_grad)
        if requires_grad:
            params = [param for _, param in named_params]
            optimizer_param_groups.append(
                {
                    "name": name,
                    "params": params,
                    "lr": _optimizer_group_lr(name, lr, group_lrs),
                }
            )
            optimizer_param_ids.update(id(param) for param in params)

    audits = []
    for name, named_params in groups.items():
        params = [param for _, param in named_params]
        param_ids = {id(param) for param in params}
        audits.append(
            NFSFParamAudit(
                name=name,
                parameter_names=tuple(param_name for param_name, _ in named_params),
                tensor_count=len(params),
                trainable_parameter_count=sum(
                    param.numel() for param in params if param.requires_grad
                ),
                requires_grad=all(param.requires_grad for param in params) if params else False,
                in_optimizer=bool(param_ids) and param_ids <= optimizer_param_ids,
            )
        )

    expected_ids = {
        id(param)
        for name, named_params in groups.items()
        if name in trainable_names
        for _, param in named_params
    }
    if optimizer_param_ids != expected_ids:
        raise RuntimeError("optimizer parameter set does not match trainable groups")

    return NFSFOptimizerPlan(
        mode=mode,
        optimizer_param_groups=optimizer_param_groups,
        audits=tuple(audits),
    )


def configure_nf_sf_full_sequence_optimizer_plan(
    generator,
    *,
    objective_mode: NFSFFullSequenceObjectiveMode,
    group_lrs: dict[str, float],
) -> NFSFOptimizerPlan:
    objective_mode = validate_nf_sf_full_sequence_objective_mode(objective_mode)
    if getattr(generator, "mcp", None) is None:
        raise ValueError("full-sequence v1 requires attached MCP modules")

    generator.requires_grad_(False)
    groups = collect_nf_sf_parameter_groups(generator)
    trainable_names = {"backbone", "patch_embedding"}
    if objective_mode == "next_forcing_full":
        trainable_names.add("mcp_fusion")
        trainable_names.update(
            name for name in groups.keys() if name.startswith("mcp_depth")
        )

    optimizer_param_groups = []
    optimizer_param_ids = set()
    for name, named_params in groups.items():
        requires_grad = name in trainable_names
        for _, param in named_params:
            param.requires_grad_(requires_grad)
        if requires_grad:
            params = [param for _, param in named_params]
            optimizer_param_groups.append(
                {
                    "name": name,
                    "params": params,
                    "lr": _optimizer_group_lr(name, None, group_lrs),
                }
            )
            optimizer_param_ids.update(id(param) for param in params)

    audits = []
    for name, named_params in groups.items():
        params = [param for _, param in named_params]
        param_ids = {id(param) for param in params}
        audits.append(
            NFSFParamAudit(
                name=name,
                parameter_names=tuple(param_name for param_name, _ in named_params),
                tensor_count=len(params),
                trainable_parameter_count=sum(
                    param.numel() for param in params if param.requires_grad
                ),
                requires_grad=all(param.requires_grad for param in params) if params else False,
                in_optimizer=bool(param_ids) and param_ids <= optimizer_param_ids,
            )
        )

    expected_ids = {
        id(param)
        for name, named_params in groups.items()
        if name in trainable_names
        for _, param in named_params
    }
    if optimizer_param_ids != expected_ids:
        raise RuntimeError("full-sequence optimizer parameter set mismatch")

    return NFSFOptimizerPlan(
        mode="joint",
        optimizer_param_groups=optimizer_param_groups,
        audits=tuple(audits),
    )


def audit_nf_sf_full_sequence_gradients(
    generator,
    *,
    objective_mode: NFSFFullSequenceObjectiveMode,
) -> dict[str, dict[str, int | float | bool]]:
    objective_mode = validate_nf_sf_full_sequence_objective_mode(objective_mode)
    expected = {"backbone", "patch_embedding"}
    if objective_mode == "next_forcing_full":
        expected.update({"mcp_fusion", "mcp_depth1", "mcp_depth2", "mcp_depth3"})
    groups = collect_nf_sf_parameter_groups(generator)
    use_shared_mcp_head = bool(
        getattr(generator, "official_shared_mcp_output_head", False)
    )
    report: dict[str, dict[str, int | float | bool]] = {}
    for name, named_params in groups.items():
        trainable_named_params = [
            (param_name, param)
            for param_name, param in named_params
            if param.requires_grad
        ]
        trainable_params = [param for _, param in trainable_named_params]
        allowed_missing = [
            param
            for param_name, param in trainable_named_params
            if (
                use_shared_mcp_head
                and name.startswith("mcp_depth")
                and ".head." in param_name
                and param.grad is None
            )
        ]
        missing = [
            param
            for param_name, param in trainable_named_params
            if param.grad is None
            and not (
                use_shared_mcp_head
                and name.startswith("mcp_depth")
                and ".head." in param_name
            )
        ]
        finite = True
        norm_sq = 0.0
        grad_tensors = 0
        for param in trainable_params:
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            grad_tensors += 1
            finite = finite and bool(torch.isfinite(grad).all().item())
            norm_sq += float(grad.square().sum().item())
        norm = norm_sq ** 0.5
        report[name] = {
            "expected_trainable": name in expected,
            "trainable_tensors": len(trainable_params),
            "grad_tensors": grad_tensors,
            "missing_grad_tensors": len(missing),
            "allowed_missing_grad_tensors": len(allowed_missing),
            "all_finite": finite,
            "aggregate_grad_norm": norm,
            "pass": (
                (name not in expected and not trainable_params)
                or (
                    name in expected
                    and len(trainable_params) > 0
                    and len(missing) == 0
                    and finite
                    and norm > 0.0
                )
            ),
        }
    return report


def nf_sf_full_sequence_gradient_audit_pass(report: dict[str, dict]) -> bool:
    return all(bool(item.get("pass")) for item in report.values())


def nf_sf_full_sequence_checkpoint_steps(
    target_global_step: int = FULL_SEQUENCE_TARGET_GLOBAL_STEP,
) -> tuple[int, ...]:
    if int(target_global_step) != FULL_SEQUENCE_TARGET_GLOBAL_STEP:
        raise ValueError("full-sequence v1 production target must be 5000")
    return FULL_SEQUENCE_CHECKPOINT_STEPS


def nf_sf_full_sequence_train_cursor(
    global_step: int,
    *,
    train_sample_count: int = 2048,
) -> dict[str, int | None]:
    step = int(global_step)
    count = int(train_sample_count)
    if count <= 0:
        raise ValueError("train_sample_count must be positive")
    if step < 0:
        raise ValueError("global_step must be non-negative")
    if step == 0:
        return {
            "global_step": 0,
            "sample_position": None,
            "cycle_index": 0,
            "next_sample_position": 0,
        }
    sample_position = (step - 1) % count
    cycle_index = (step - 1) // count
    return {
        "global_step": step,
        "sample_position": sample_position,
        "cycle_index": cycle_index,
        "next_sample_position": step % count,
    }


def build_nf_sf_full_sequence_provenance(
    *,
    objective_mode: NFSFFullSequenceObjectiveMode,
    reference_checkpoint_sha256: str = OFFICIAL_SELF_FORCING_CHECKPOINT_SHA256,
    git_sha: str | None = None,
) -> dict:
    objective_mode = validate_nf_sf_full_sequence_objective_mode(objective_mode)
    provenance = {
        "schema": FULL_SEQUENCE_TRAINER_SCHEMA,
        "run_kind": FULL_SEQUENCE_RUN_KIND,
        "objective_version": FULL_SEQUENCE_OBJECTIVE_VERSION,
        "objective_mode": objective_mode,
        "paper_reference": FULL_SEQUENCE_PAPER_REFERENCE,
        "paper_exact_reproduction": False,
        "reference_checkpoint_sha256": str(reference_checkpoint_sha256),
        "git_sha": git_sha,
        "rng_draw_order_version": FULL_SEQUENCE_RNG_DRAW_ORDER_VERSION,
        "future_embedding_order": FULL_SEQUENCE_FUTURE_EMBEDDING_ORDER,
        "adaptation_differences": {
            "base_model": "official Self-Forcing Wan2.1",
            "fixed_chunk_frames": FULL_SEQUENCE_CHUNK_FRAMES,
            "paper_random_chunk_size_not_used": True,
            "noisy_history_augmentation": False,
            "paper_exact_mcp_attention": False,
            "mcp_history_via_fused_main_features": True,
            "mcp_future_attention_single_chunk_only": True,
            "action_stream_absent": True,
        },
        "objective": {
            "full_sequence_teacher_forcing_main": True,
            "full_teacher_frames": FULL_SEQUENCE_FRAME_COUNT,
            "num_chunks": FULL_SEQUENCE_NUM_CHUNKS,
            "all_main_chunks_supervised": True,
            "mcp_anchor_counts": list(
                full_sequence_mcp_anchor_counts(
                    num_chunks=FULL_SEQUENCE_NUM_CHUNKS,
                    depths=FULL_SEQUENCE_DEPTHS,
                )
            ),
            "tail_loss_excluded": True,
            "main_shift": DEFAULT_S_MAIN,
            "mcp_shift": DEFAULT_S_MCP,
            "depth_weights": list(FULL_SEQUENCE_DEPTH_WEIGHTS),
            "joint_backbone": True,
            "shared_patch_embedding": True,
            "taps": [3, 11, 19, 29],
            "mcp_blocks_per_depth": 3,
            "self_rollout": False,
            "dmd": False,
            "generated_history": False,
            "noisy_history_augmentation": False,
        },
        "memory": {
            "gradient_checkpointing": True,
            "anchor_micro_loop": True,
            "main_backbone_forward_count_per_train_sample": 1,
        },
        "schedule": {
            "target_global_step": FULL_SEQUENCE_TARGET_GLOBAL_STEP,
            "checkpoint_steps": list(FULL_SEQUENCE_CHECKPOINT_STEPS),
            "validation_steps": list(FULL_SEQUENCE_CHECKPOINT_STEPS),
            "no_125_step_pilot": True,
        },
    }
    if objective_mode == "main_only_full_control":
        provenance["objective"].update(
            {
                "mcp_forward": False,
                "mcp_loss": False,
                "mcp_rng_draws_consumed_for_matched_main_trajectory": True,
            }
        )
    else:
        provenance["objective"].update({"mcp_forward": True, "mcp_loss": True})
    validate_nf_sf_full_sequence_provenance(provenance)
    return provenance


def validate_nf_sf_full_sequence_provenance(provenance: dict) -> None:
    if not isinstance(provenance, dict):
        raise TypeError("provenance must be a dict")
    if provenance.get("schema") != FULL_SEQUENCE_TRAINER_SCHEMA:
        raise ValueError("full-sequence provenance schema mismatch")
    if provenance.get("run_kind") != FULL_SEQUENCE_RUN_KIND:
        raise ValueError("full-sequence run_kind mismatch")
    if provenance.get("objective_version") != FULL_SEQUENCE_OBJECTIVE_VERSION:
        raise ValueError("full-sequence objective_version mismatch")
    if bool(provenance.get("paper_exact_reproduction")):
        raise ValueError("full-sequence v1 must not claim paper-exact reproduction")
    if "m5_formal" in provenance or "m5_formal_trainer" in provenance:
        raise ValueError("full-sequence provenance must not include M5 formal metadata")
    adaptation = provenance.get("adaptation_differences")
    if not isinstance(adaptation, dict):
        raise ValueError("provenance missing adaptation_differences")
    required_false = ("paper_exact_mcp_attention", "noisy_history_augmentation")
    for key in required_false:
        if bool(adaptation.get(key)):
            raise ValueError(f"adaptation_differences.{key} must be false")
    required_true = (
        "paper_random_chunk_size_not_used",
        "mcp_history_via_fused_main_features",
        "mcp_future_attention_single_chunk_only",
        "action_stream_absent",
    )
    for key in required_true:
        if not bool(adaptation.get(key)):
            raise ValueError(f"adaptation_differences.{key} must be true")


def require_nf_sf_full_sequence_runtime(
    *,
    config,
    generator,
    objective_mode: NFSFFullSequenceObjectiveMode,
) -> dict:
    objective_mode = validate_nf_sf_full_sequence_objective_mode(objective_mode)
    config_nfpb = int(getattr(config, "num_frame_per_block", 0))
    model_nfpb = int(getattr(generator.model, "num_frame_per_block", 0))
    if config_nfpb != FULL_SEQUENCE_CHUNK_FRAMES:
        raise ValueError("full-sequence v1 requires config.num_frame_per_block=3")
    if model_nfpb != FULL_SEQUENCE_CHUNK_FRAMES:
        raise ValueError("full-sequence v1 requires model.num_frame_per_block=3")
    if not bool(getattr(generator.model, "gradient_checkpointing", False)):
        raise ValueError("full-sequence v1 requires model.gradient_checkpointing=True")
    if objective_mode == "next_forcing_full" and getattr(generator, "mcp", None) is None:
        raise ValueError("next_forcing_full requires MCP modules")
    return {
        "config_num_frame_per_block": config_nfpb,
        "model_num_frame_per_block": model_nfpb,
        "gradient_checkpointing": True,
        "objective_mode": objective_mode,
    }


def _optimizer_group_lr(
    name: str,
    lr: float | None,
    group_lrs: dict[str, float] | None,
) -> float:
    if group_lrs is None:
        if lr is None:
            raise ValueError("lr must be provided when group_lrs is absent")
        return float(lr)
    if name in group_lrs:
        return float(group_lrs[name])
    if name.startswith("mcp_depth") and "mcp" in group_lrs:
        return float(group_lrs["mcp"])
    if name == "mcp_fusion" and "mcp" in group_lrs:
        return float(group_lrs["mcp"])
    raise ValueError(f"missing optimizer lr for parameter group {name!r}")


def collect_nf_sf_parameter_groups(
    generator,
) -> dict[str, tuple[tuple[str, torch.nn.Parameter], ...]]:
    patch_named = tuple(
        (f"model.patch_embedding.{name}", param)
        for name, param in generator.model.patch_embedding.named_parameters()
    )
    patch_ids = {id(param) for _, param in patch_named}
    backbone_named = tuple(
        (f"model.{name}", param)
        for name, param in generator.model.named_parameters()
        if id(param) not in patch_ids
    )
    groups = {
        "backbone": backbone_named,
        "patch_embedding": patch_named,
        "mcp_fusion": tuple(
            (f"mcp.fusion.{name}", param)
            for name, param in generator.mcp.fusion.named_parameters()
        ),
    }
    for index, module in enumerate(generator.mcp.mcp_modules, start=1):
        groups[f"mcp_depth{index}"] = tuple(
            (f"mcp.mcp_modules.{index - 1}.{name}", param)
            for name, param in module.named_parameters()
        )
    return groups


def _validate_selected_state(
    state: NFSFSelectedState,
    *,
    chunk_frames: int,
    depths: tuple[int, ...],
) -> None:
    current = state.current_target
    _validate_chunk_tensor(current, chunk_frames=chunk_frames, name="current_target")
    if len(state.future_targets) != len(depths):
        raise ValueError("future_targets must align 1:1 with depths")
    for index, target in enumerate(state.future_targets, start=1):
        _validate_chunk_tensor(target, chunk_frames=chunk_frames, name=f"future{index}")
        if target.shape != current.shape:
            raise ValueError(f"future{index} shape must match current_target")
    if state.clean_history is not None:
        history = state.clean_history
        if history.ndim != 5:
            raise ValueError("clean_history must have rank 5 [B, F, C, H, W]")
        if history.shape[0] != current.shape[0] or history.shape[2:] != current.shape[2:]:
            raise ValueError("clean_history shape must match current target except frames")
        if history.shape[1] % chunk_frames != 0:
            raise ValueError("clean_history frame count must be chunk-aligned")
        if history.shape[1] != chunk_frames:
            raise ValueError(
                "NF-SF M2 Wan teacher-forced harness supports exactly one clean "
                "history chunk"
            )
    if state.future_valid_masks is not None:
        if len(state.future_valid_masks) != len(depths):
            raise ValueError("future_valid_masks must align 1:1 with depths")
        for mask in state.future_valid_masks:
            if mask.shape != (chunk_frames,):
                raise ValueError("future_valid_masks must be frame masks for one chunk")
            if mask.dtype != torch.bool:
                raise ValueError("future_valid_masks must use bool dtype")
            if mask.device != current.device:
                raise ValueError("future_valid_masks must be on the target device")


def _validate_chunk_tensor(tensor: torch.Tensor, *, chunk_frames: int, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 5:
        raise ValueError(f"{name} must have rank 5 [B, F, C, H, W]")
    if tensor.shape[LATENT_FRAME_AXIS] != chunk_frames:
        raise ValueError(f"{name} must be exactly one selected target chunk")
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must use a floating dtype")


def _future_masks_for(
    state: NFSFSelectedState,
    *,
    chunk_frames: int,
) -> tuple[torch.Tensor, ...]:
    if state.future_valid_masks is not None:
        return state.future_valid_masks
    return tuple(
        torch.ones(chunk_frames, dtype=torch.bool, device=target.device)
        for target in state.future_targets
    )


def _current_start_frame(state: NFSFSelectedState) -> int:
    if state.current_start_frame is not None:
        if state.current_start_frame < 0:
            raise ValueError("current_start_frame must be non-negative")
        return int(state.current_start_frame)
    if state.clean_history is None:
        return 0
    return int(state.clean_history.shape[1])


def _add_noise_like_scheduler(
    scheduler,
    clean: torch.Tensor,
    epsilon: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    return scheduler.add_noise(
        clean.flatten(0, 1),
        epsilon.flatten(0, 1),
        timestep.flatten(0, 1),
    ).unflatten(0, clean.shape[:2])


def _training_target_like_scheduler(
    scheduler,
    clean: torch.Tensor,
    epsilon: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    return scheduler.training_target(
        clean.flatten(0, 1),
        epsilon.flatten(0, 1),
        timestep.flatten(0, 1),
    ).unflatten(0, clean.shape[:2])


def _validate_full_sequence_clean_target(
    clean_target: torch.Tensor,
    *,
    chunk_frames: int,
) -> None:
    if not isinstance(clean_target, torch.Tensor):
        raise TypeError("clean_target must be a torch.Tensor")
    if clean_target.ndim != 5:
        raise ValueError("clean_target must have rank 5 [B, F, C, H, W]")
    if clean_target.shape[LATENT_FRAME_AXIS] != FULL_SEQUENCE_FRAME_COUNT:
        raise ValueError("full-sequence v1 requires exactly 21 latent frames")
    if clean_target.shape[LATENT_FRAME_AXIS] % chunk_frames != 0:
        raise ValueError("clean_target frame count must be chunk-aligned")
    if clean_target.shape[LATENT_FRAME_AXIS] // chunk_frames != FULL_SEQUENCE_NUM_CHUNKS:
        raise ValueError("full-sequence v1 requires exactly 7 chunks")
    if not clean_target.is_floating_point():
        raise ValueError("clean_target must use a floating dtype")


def _full_sequence_mcp_targets(
    clean_target: torch.Tensor,
    *,
    chunk_frames: int,
    depths: tuple[int, ...],
) -> tuple[torch.Tensor, ...]:
    num_chunks = clean_target.shape[LATENT_FRAME_AXIS] // chunk_frames
    targets = []
    for depth in depths:
        valid_count = max(num_chunks - depth, 0)
        chunks = []
        for anchor_index in range(valid_count):
            target_chunk = anchor_index + depth
            chunks.append(
                clean_target[
                    :,
                    target_chunk * chunk_frames : (target_chunk + 1) * chunk_frames,
                ]
            )
        if chunks:
            targets.append(torch.stack(chunks, dim=1))
        else:
            targets.append(
                clean_target.new_empty(
                    (clean_target.shape[0], 0, chunk_frames, *clean_target.shape[2:])
                )
            )
    return tuple(targets)


def _add_noise_for_anchor_chunks(
    scheduler,
    clean: torch.Tensor,
    epsilon: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    _validate_anchor_chunk_batch(clean, epsilon, timestep, name="add_noise")
    bsz, anchors, frames = clean.shape[:3]
    flat_shape = (bsz * anchors * frames, *clean.shape[3:])
    return scheduler.add_noise(
        clean.reshape(flat_shape),
        epsilon.reshape(flat_shape),
        timestep.reshape(bsz * anchors * frames),
    ).reshape_as(clean)


def _training_target_for_anchor_chunks(
    scheduler,
    clean: torch.Tensor,
    epsilon: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    _validate_anchor_chunk_batch(clean, epsilon, timestep, name="training_target")
    bsz, anchors, frames = clean.shape[:3]
    flat_shape = (bsz * anchors * frames, *clean.shape[3:])
    return scheduler.training_target(
        clean.reshape(flat_shape),
        epsilon.reshape(flat_shape),
        timestep.reshape(bsz * anchors * frames),
    ).reshape_as(clean)


def _validate_anchor_chunk_batch(
    clean: torch.Tensor,
    epsilon: torch.Tensor,
    timestep: torch.Tensor,
    *,
    name: str,
) -> None:
    if clean.ndim != 6:
        raise ValueError(f"{name} clean tensor must have shape [B, A, F, C, H, W]")
    if tuple(epsilon.shape) != tuple(clean.shape):
        raise ValueError(f"{name} epsilon shape must match clean")
    if tuple(timestep.shape) != tuple(clean.shape[:3]):
        raise ValueError(f"{name} timestep shape must be [B, A, F]")


def _output_field(outputs, name: str):
    if isinstance(outputs, dict):
        return outputs[name]
    if hasattr(outputs, name):
        return getattr(outputs, name)
    raise TypeError(f"full-sequence model output missing {name}")


def _expand_timestep_to_chunk(
    timestep: torch.Tensor,
    *,
    target: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if not torch.is_tensor(timestep):
        raise TypeError(f"{name} must be a torch.Tensor")
    value = timestep.to(device=target.device, dtype=torch.float32)
    if value.numel() == 1:
        return torch.full(
            target.shape[:2],
            float(value.reshape(-1)[0].item()),
            device=target.device,
            dtype=torch.float32,
        )
    if tuple(value.shape) == tuple(target.shape[:2]):
        return value
    if value.numel() == int(target.shape[0] * target.shape[1]):
        return value.reshape(target.shape[:2])
    raise ValueError(
        f"{name} shape {tuple(timestep.shape)} cannot broadcast to "
        f"{tuple(target.shape[:2])}"
    )


def _ensure_finite_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must be a floating tensor")
    if not bool(torch.isfinite(tensor.detach().float()).all().item()):
        raise RuntimeError(f"{name} contains non-finite values")
