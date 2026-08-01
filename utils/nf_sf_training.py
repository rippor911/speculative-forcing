from dataclasses import dataclass
from typing import Iterable, Literal

import torch
import torch.nn.functional as F

from utils.nf_sf_tensors import (
    DEFAULT_S_MAIN,
    DEFAULT_S_MCP,
    LATENT_FRAME_AXIS,
    sample_nf_sf_noise_and_timesteps,
)


NFSFTrainMode = Literal["frozen", "joint"]


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
class NFSFForwardResult:
    noisy_batch: NFSFNoisyBatch
    main_flow_pred: torch.Tensor
    mcp_flow_preds: tuple[torch.Tensor, ...]
    losses: NFSFLossBreakdown


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
