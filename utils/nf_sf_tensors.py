from dataclasses import dataclass
from typing import Iterable

import torch


LATENT_FRAME_AXIS = 1
MAX_FUTURE_DEPTH = 3
DEFAULT_S_MAIN = 5.0
DEFAULT_S_MCP = 10.0
DEFAULT_NUM_TRAIN_TIMESTEPS = 1000


def make_generator(seed: int, device: torch.device | str) -> torch.Generator:
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("NF-SF random sampling supports CPU and CUDA tensors only")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


@dataclass(frozen=True)
class FutureChunkTarget:
    depth: int
    target: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class NFSFTensorSamples:
    epsilon_main: torch.Tensor
    epsilon_depths: tuple[torch.Tensor, ...]
    timestep_main: torch.Tensor
    timestep_depths: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class NFSFTensorInputs:
    main_target: torch.Tensor
    future_targets: tuple[FutureChunkTarget, ...]
    samples: NFSFTensorSamples


def make_cpu_generator(seed: int) -> torch.Generator:
    return make_generator(seed, "cpu")


def shift_future_chunks(
    latent: torch.Tensor,
    *,
    chunk_frames: int,
    depth: int,
    chunk_axis: int = LATENT_FRAME_AXIS,
) -> FutureChunkTarget:
    """Shift [B, F, C, H, W] latent chunks by `depth` future chunks.

    The returned tensor keeps the input shape by replicating the last legal chunk.
    The returned frame mask marks only chunks with an in-range future target as valid.
    """
    num_frames = _validate_latent_chunks(latent, chunk_frames, chunk_axis)
    _validate_depth(depth)

    num_chunks = num_frames // chunk_frames
    pieces = []
    for chunk_index in range(num_chunks):
        source_chunk = min(chunk_index + depth, num_chunks - 1)
        pieces.append(
            latent.narrow(
                chunk_axis,
                source_chunk * chunk_frames,
                chunk_frames,
            )
        )

    target = torch.cat(pieces, dim=chunk_axis)
    valid_mask = future_valid_mask(
        num_frames=num_frames,
        chunk_frames=chunk_frames,
        depth=depth,
        device=latent.device,
    )
    return FutureChunkTarget(depth=depth, target=target, valid_mask=valid_mask)


def future_valid_mask(
    *,
    num_frames: int,
    chunk_frames: int,
    depth: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    _validate_chunk_frames(chunk_frames)
    _validate_depth(depth)
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if num_frames % chunk_frames != 0:
        raise ValueError(
            f"num_frames ({num_frames}) must be divisible by chunk_frames "
            f"({chunk_frames})"
        )

    num_chunks = num_frames // chunk_frames
    valid_chunks = max(num_chunks - depth, 0)
    mask = torch.zeros(num_frames, dtype=torch.bool, device=device)
    mask[: valid_chunks * chunk_frames] = True
    return mask


def flow_match_shift_timesteps(
    timestep: torch.Tensor,
    *,
    shift: float,
    num_train_timesteps: int = DEFAULT_NUM_TRAIN_TIMESTEPS,
) -> torch.Tensor:
    """Apply the same flow-match timestep warp as FlowMatchScheduler."""
    if shift <= 0:
        raise ValueError("shift must be positive")
    if num_train_timesteps <= 0:
        raise ValueError("num_train_timesteps must be positive")

    sigma = timestep.to(dtype=torch.float32) / float(num_train_timesteps)
    shifted_sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return shifted_sigma * float(num_train_timesteps)


def sample_chunk_timesteps(
    *,
    batch_size: int,
    chunk_frames: int,
    shift: float,
    generator: torch.Generator | None = None,
    device: torch.device | str = "cpu",
    min_timestep: int = 0,
    max_timestep: int = DEFAULT_NUM_TRAIN_TIMESTEPS,
    num_train_timesteps: int = DEFAULT_NUM_TRAIN_TIMESTEPS,
) -> torch.Tensor:
    """Sample one raw timestep per batch sample and repeat it within a target chunk."""
    _validate_chunk_frames(chunk_frames)
    device = torch.device(device)
    _validate_random_device(device=device, generator=generator)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if min_timestep < 0:
        raise ValueError("min_timestep must be non-negative")
    if max_timestep <= min_timestep:
        raise ValueError("max_timestep must be greater than min_timestep")

    raw = torch.randint(
        min_timestep,
        max_timestep,
        (batch_size, 1),
        device=device,
        dtype=torch.int64,
        generator=generator,
    )
    raw = raw.repeat(1, chunk_frames)
    return flow_match_shift_timesteps(
        raw,
        shift=shift,
        num_train_timesteps=num_train_timesteps,
    )


def sample_nf_sf_noise_and_timesteps(
    target_chunk: torch.Tensor,
    *,
    chunk_frames: int,
    depths: Iterable[int] = (1, 2, 3),
    s_main: float = DEFAULT_S_MAIN,
    s_mcp: float = DEFAULT_S_MCP,
    generator: torch.Generator | None = None,
    chunk_axis: int = LATENT_FRAME_AXIS,
    num_train_timesteps: int = DEFAULT_NUM_TRAIN_TIMESTEPS,
) -> NFSFTensorSamples:
    _validate_target_chunk(target_chunk, chunk_frames, chunk_axis)
    _validate_random_device(device=target_chunk.device, generator=generator)
    normalized_depths = _normalize_depths(depths)
    if not target_chunk.is_floating_point():
        raise ValueError("target_chunk must use a floating dtype to sample noise")

    batch_size = target_chunk.shape[0]
    epsilon_main = _randn_like(target_chunk, generator=generator)
    epsilon_depths = tuple(
        _randn_like(target_chunk, generator=generator)
        for _ in normalized_depths
    )

    timestep_main = sample_chunk_timesteps(
        batch_size=batch_size,
        chunk_frames=chunk_frames,
        shift=s_main,
        generator=generator,
        device=target_chunk.device,
        num_train_timesteps=num_train_timesteps,
    )
    timestep_depths = tuple(
        sample_chunk_timesteps(
            batch_size=batch_size,
            chunk_frames=chunk_frames,
            shift=s_mcp,
            generator=generator,
            device=target_chunk.device,
            num_train_timesteps=num_train_timesteps,
        )
        for _ in normalized_depths
    )

    return NFSFTensorSamples(
        epsilon_main=epsilon_main,
        epsilon_depths=epsilon_depths,
        timestep_main=timestep_main,
        timestep_depths=timestep_depths,
    )


def prepare_nf_sf_tensor_inputs(
    main_target: torch.Tensor,
    *,
    future_target_chunks: Iterable[torch.Tensor],
    chunk_frames: int,
    depths: Iterable[int] = (1, 2, 3),
    s_main: float = DEFAULT_S_MAIN,
    s_mcp: float = DEFAULT_S_MCP,
    generator: torch.Generator | None = None,
    chunk_axis: int = LATENT_FRAME_AXIS,
    num_train_timesteps: int = DEFAULT_NUM_TRAIN_TIMESTEPS,
) -> NFSFTensorInputs:
    _validate_target_chunk(main_target, chunk_frames, chunk_axis)
    normalized_depths = _normalize_depths(depths)
    future_chunks = tuple(future_target_chunks)
    if len(future_chunks) != len(normalized_depths):
        raise ValueError("future_target_chunks must align 1:1 with depths")

    future_targets = []
    for depth, target in zip(normalized_depths, future_chunks):
        _validate_target_chunk(target, chunk_frames, chunk_axis)
        if target.shape[0] != main_target.shape[0] or target.shape[2:] != main_target.shape[2:]:
            raise ValueError("future target chunk shape must match main_target")
        future_targets.append(
            FutureChunkTarget(
                depth=depth,
                target=target,
                valid_mask=torch.ones(
                    chunk_frames,
                    dtype=torch.bool,
                    device=target.device,
                ),
            )
        )

    samples = sample_nf_sf_noise_and_timesteps(
        main_target,
        chunk_frames=chunk_frames,
        depths=normalized_depths,
        s_main=s_main,
        s_mcp=s_mcp,
        generator=generator,
        chunk_axis=chunk_axis,
        num_train_timesteps=num_train_timesteps,
    )
    return NFSFTensorInputs(
        main_target=main_target,
        future_targets=tuple(future_targets),
        samples=samples,
    )


def _randn_like(
    latent: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    return torch.randn(
        latent.shape,
        dtype=latent.dtype,
        device=latent.device,
        generator=generator,
    )


def _normalize_depths(depths: Iterable[int]) -> tuple[int, ...]:
    normalized = tuple(int(depth) for depth in depths)
    if not normalized:
        raise ValueError("depths must not be empty")
    for depth in normalized:
        _validate_depth(depth)
    return normalized


def _validate_latent_chunks(
    latent: torch.Tensor,
    chunk_frames: int,
    chunk_axis: int,
) -> int:
    if not isinstance(latent, torch.Tensor):
        raise TypeError("latent must be a torch.Tensor")
    if latent.ndim != 5:
        raise ValueError(
            "NF-SF latent tensor must have rank 5 with layout [B, F, C, H, W]"
        )
    if chunk_axis != LATENT_FRAME_AXIS:
        raise ValueError(
            "NF-SF latent layout requires chunk_axis=1 for the frame dimension"
        )
    _validate_chunk_frames(chunk_frames)

    num_frames = latent.shape[chunk_axis]
    if num_frames <= 0:
        raise ValueError("latent frame dimension must be positive")
    if num_frames % chunk_frames != 0:
        raise ValueError(
            f"latent frame count ({num_frames}) must be divisible by "
            f"chunk_frames ({chunk_frames})"
        )
    return num_frames


def _validate_target_chunk(
    target_chunk: torch.Tensor,
    chunk_frames: int,
    chunk_axis: int,
) -> None:
    num_frames = _validate_latent_chunks(target_chunk, chunk_frames, chunk_axis)
    if num_frames != chunk_frames:
        raise ValueError(
            "noise/timestep sampling expects one selected target chunk with "
            f"{chunk_frames} frames, got {num_frames}"
        )


def _validate_random_device(
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> None:
    if device.type not in ("cpu", "cuda"):
        raise ValueError("NF-SF random sampling supports CPU and CUDA tensors only")
    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator")
        if generator.device != device:
            raise ValueError("generator device must match tensor device")


def _validate_chunk_frames(chunk_frames: int) -> None:
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")


def _validate_depth(depth: int) -> None:
    if depth <= 0:
        raise ValueError("depth must be positive")
    if depth > MAX_FUTURE_DEPTH:
        raise ValueError(f"depth must be <= {MAX_FUTURE_DEPTH}")
