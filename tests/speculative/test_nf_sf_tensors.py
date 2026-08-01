import pytest
import torch

from utils.nf_sf_tensors import (
    DEFAULT_S_MAIN,
    DEFAULT_S_MCP,
    future_valid_mask,
    flow_match_shift_timesteps,
    make_generator,
    make_cpu_generator,
    prepare_nf_sf_tensor_inputs,
    sample_nf_sf_noise_and_timesteps,
    shift_future_chunks,
)
from utils.scheduler import FlowMatchScheduler


def _latent_values(num_frames: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.arange(num_frames, dtype=dtype).reshape(1, num_frames, 1, 1, 1)


def _expected_shift_values(num_frames: int, chunk_frames: int, depth: int) -> torch.Tensor:
    values = []
    num_chunks = num_frames // chunk_frames
    for chunk_index in range(num_chunks):
        source_chunk = min(chunk_index + depth, num_chunks - 1)
        start = source_chunk * chunk_frames
        values.extend(range(start, start + chunk_frames))
    return torch.tensor(values, dtype=torch.float32)


def _expected_mask(num_frames: int, chunk_frames: int, depth: int) -> torch.Tensor:
    valid_frames = max(num_frames - depth * chunk_frames, 0)
    mask = torch.zeros(num_frames, dtype=torch.bool)
    mask[:valid_frames] = True
    return mask


def _legacy_ode_shift(latent: torch.Tensor, chunk_frames: int, depth: int) -> torch.Tensor:
    num_frames = latent.shape[1]
    if num_frames % chunk_frames != 0:
        raise ValueError
    num_chunks = num_frames // chunk_frames
    pieces = []
    for index in range(num_chunks):
        source = min(index + depth, num_chunks - 1)
        pieces.append(latent[:, source * chunk_frames:(source + 1) * chunk_frames])
    return torch.cat(pieces, dim=1)


def _legacy_ode_mask(num_frames: int, chunk_frames: int, depth: int, device) -> torch.Tensor:
    num_chunks = num_frames // chunk_frames
    mask = torch.zeros(num_frames, dtype=torch.bool, device=device)
    valid_chunks = max(num_chunks - depth, 0)
    mask[: valid_chunks * chunk_frames] = True
    return mask


def _storage_ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def test_future_shift_exact_values_masks_and_chunk_frame_sizes() -> None:
    for chunk_frames in (1, 2, 3, 4):
        num_frames = chunk_frames * 5
        latent = _latent_values(num_frames)

        for depth in (1, 2, 3):
            result = shift_future_chunks(
                latent,
                chunk_frames=chunk_frames,
                depth=depth,
            )

            assert torch.equal(
                result.target.flatten(),
                _expected_shift_values(num_frames, chunk_frames, depth),
            )
            assert torch.equal(
                result.valid_mask,
                _expected_mask(num_frames, chunk_frames, depth),
            )
            assert result.target.shape == latent.shape
            assert result.target.dtype == latent.dtype
            assert result.target.device == latent.device


def test_non_divisible_frame_count_errors_instead_of_truncating() -> None:
    latent = _latent_values(5)

    with pytest.raises(ValueError, match="divisible"):
        shift_future_chunks(latent, chunk_frames=2, depth=1)


def test_same_seed_reproducible_and_input_not_modified() -> None:
    main = _latent_values(3)
    futures = (_latent_values(3) + 10, _latent_values(3) + 20, _latent_values(3) + 30)
    before = main.clone()

    first = prepare_nf_sf_tensor_inputs(
        main,
        future_target_chunks=futures,
        chunk_frames=3,
        generator=make_cpu_generator(17),
    )
    second = prepare_nf_sf_tensor_inputs(
        main,
        future_target_chunks=futures,
        chunk_frames=3,
        generator=make_cpu_generator(17),
    )

    assert torch.equal(main, before)
    assert torch.equal(first.main_target, second.main_target)
    for left, right in zip(first.future_targets, second.future_targets):
        assert left.depth == right.depth
        assert torch.equal(left.target, right.target)
        assert torch.equal(left.valid_mask, right.valid_mask)

    assert torch.equal(first.samples.epsilon_main, second.samples.epsilon_main)
    assert torch.equal(first.samples.timestep_main, second.samples.timestep_main)
    for left, right in zip(first.samples.epsilon_depths, second.samples.epsilon_depths):
        assert torch.equal(left, right)
    for left, right in zip(first.samples.timestep_depths, second.samples.timestep_depths):
        assert torch.equal(left, right)


def test_noise_and_timestep_are_independent_and_follow_generator_path() -> None:
    target_chunk = torch.zeros((2, 2, 2, 1, 1), dtype=torch.float32)
    samples = sample_nf_sf_noise_and_timesteps(
        target_chunk,
        chunk_frames=2,
        s_main=5.0,
        s_mcp=10.0,
        generator=make_cpu_generator(1234),
    )

    expected_generator = make_cpu_generator(1234)
    expected_noises = [
        torch.randn(target_chunk.shape, dtype=target_chunk.dtype, generator=expected_generator)
        for _ in range(4)
    ]
    expected_raw_timesteps = [
        torch.randint(0, 1000, (2, 1), dtype=torch.int64, generator=expected_generator)
        for _ in range(4)
    ]
    expected_main_timestep = flow_match_shift_timesteps(
        expected_raw_timesteps[0].repeat(1, 2),
        shift=5.0,
    )
    expected_depth_timesteps = [
        flow_match_shift_timesteps(raw.repeat(1, 2), shift=10.0)
        for raw in expected_raw_timesteps[1:]
    ]

    assert torch.equal(samples.epsilon_main, expected_noises[0])
    for actual, expected in zip(samples.epsilon_depths, expected_noises[1:]):
        assert torch.equal(actual, expected)
    assert torch.equal(samples.timestep_main, expected_main_timestep)
    for actual, expected in zip(samples.timestep_depths, expected_depth_timesteps):
        assert torch.equal(actual, expected)

    all_noises = (samples.epsilon_main, *samples.epsilon_depths)
    for index, left in enumerate(all_noises):
        for right in all_noises[index + 1:]:
            assert _storage_ptr(left) != _storage_ptr(right)
    all_timesteps = (samples.timestep_main, *samples.timestep_depths)
    for index, left in enumerate(all_timesteps):
        for right in all_timesteps[index + 1:]:
            assert _storage_ptr(left) != _storage_ptr(right)

    wrong_main_shift = flow_match_shift_timesteps(
        expected_raw_timesteps[0].repeat(1, 2),
        shift=10.0,
    )
    wrong_mcp_shift = flow_match_shift_timesteps(
        expected_raw_timesteps[1].repeat(1, 2),
        shift=5.0,
    )
    assert not torch.allclose(samples.timestep_main, wrong_main_shift)
    assert not torch.allclose(samples.timestep_depths[0], wrong_mcp_shift)


def test_selected_target_chunk_timestep_granularity() -> None:
    target_chunk = torch.zeros((2, 4, 1, 1, 1), dtype=torch.float32)
    generator = make_cpu_generator(2024)
    expected_generator = make_cpu_generator(2024)

    actual = sample_nf_sf_noise_and_timesteps(
        target_chunk,
        chunk_frames=4,
        generator=generator,
    )
    for _ in range(4):
        torch.randn(target_chunk.shape, dtype=target_chunk.dtype, generator=expected_generator)
    raw_timesteps = [
        torch.randint(0, 1000, (2, 1), dtype=torch.int64, generator=expected_generator)
        for _ in range(4)
    ]

    expected_main = flow_match_shift_timesteps(raw_timesteps[0].repeat(1, 4), shift=5.0)
    expected_depth1 = flow_match_shift_timesteps(raw_timesteps[1].repeat(1, 4), shift=10.0)
    assert actual.timestep_main.shape == (2, 4)
    assert actual.timestep_depths[0].shape == (2, 4)
    assert torch.equal(actual.timestep_main, expected_main)
    assert torch.equal(actual.timestep_depths[0], expected_depth1)
    assert torch.equal(actual.timestep_main[:, :1].expand(-1, 4), actual.timestep_main)
    assert torch.equal(
        actual.timestep_depths[0][:, :1].expand(-1, 4),
        actual.timestep_depths[0],
    )


def test_timestep_shift_matches_flow_match_scheduler_output_for_edge_values() -> None:
    unshifted = torch.tensor([0.0, 1.0, 500.0, 999.0, 1000.0])
    scheduler_indices = torch.tensor([1000, 999, 500, 1, 0])

    for shift in (1.0, DEFAULT_S_MAIN, DEFAULT_S_MCP):
        scheduler = FlowMatchScheduler(
            num_inference_steps=1001,
            num_train_timesteps=1000,
            shift=shift,
            sigma_min=0.0,
        )
        actual = flow_match_shift_timesteps(unshifted, shift=shift)

        assert torch.allclose(actual, scheduler.timesteps[scheduler_indices])


def test_dtype_and_cpu_device_are_preserved() -> None:
    main = torch.zeros((1, 2, 1, 1, 1), dtype=torch.float64)
    futures = (main + 1, main + 2, main + 3)
    prepared = prepare_nf_sf_tensor_inputs(
        main,
        future_target_chunks=futures,
        chunk_frames=2,
        generator=make_cpu_generator(99),
    )

    assert prepared.future_targets[0].target.dtype == torch.float64
    assert prepared.future_targets[0].target.device.type == "cpu"
    assert prepared.future_targets[0].valid_mask.dtype == torch.bool
    assert prepared.future_targets[0].valid_mask.device.type == "cpu"
    assert prepared.samples.epsilon_main.dtype == torch.float64
    assert prepared.samples.epsilon_main.device.type == "cpu"
    assert prepared.samples.timestep_main.dtype == torch.float32
    assert prepared.samples.timestep_main.device.type == "cpu"


def test_random_sampling_rejects_non_cpu_target_chunks() -> None:
    target_chunk = torch.empty((1, 2, 1, 1, 1), dtype=torch.float32, device="meta")

    with pytest.raises(ValueError, match="CPU and CUDA"):
        sample_nf_sf_noise_and_timesteps(
            target_chunk,
            chunk_frames=2,
            generator=make_cpu_generator(1),
        )


def test_make_generator_matches_cpu_helper() -> None:
    target_chunk = torch.zeros((1, 2, 1, 1, 1), dtype=torch.float32)

    from_generic = sample_nf_sf_noise_and_timesteps(
        target_chunk,
        chunk_frames=2,
        generator=make_generator(7, target_chunk.device),
    )
    from_cpu_helper = sample_nf_sf_noise_and_timesteps(
        target_chunk,
        chunk_frames=2,
        generator=make_cpu_generator(7),
    )

    assert torch.equal(from_generic.epsilon_main, from_cpu_helper.epsilon_main)
    assert torch.equal(from_generic.timestep_main, from_cpu_helper.timestep_main)


def test_random_sampling_rejects_wrong_generator_type() -> None:
    target_chunk = torch.zeros((1, 2, 1, 1, 1), dtype=torch.float32)

    with pytest.raises(TypeError, match="torch.Generator"):
        sample_nf_sf_noise_and_timesteps(
            target_chunk,
            chunk_frames=2,
            generator=object(),
        )


def test_ode_shift_and_mask_match_legacy_reference_logic() -> None:
    latent = _latent_values(15)

    for depth in (1, 2, 3):
        actual = shift_future_chunks(latent, chunk_frames=3, depth=depth)
        expected_target = _legacy_ode_shift(latent, chunk_frames=3, depth=depth)
        expected_mask = _legacy_ode_mask(15, chunk_frames=3, depth=depth, device=latent.device)

        assert torch.equal(actual.target, expected_target)
        assert torch.equal(actual.target.flatten(), _expected_shift_values(15, 3, depth))
        assert actual.target.shape == latent.shape
        assert actual.valid_mask.dtype == torch.bool
        assert actual.valid_mask.device == latent.device
        assert torch.equal(actual.valid_mask, expected_mask)
        assert torch.equal(
            future_valid_mask(
                num_frames=15,
                chunk_frames=3,
                depth=depth,
                device=latent.device,
            ),
            expected_mask,
        )


def test_invalid_parameters_raise_clear_errors() -> None:
    latent = _latent_values(8)

    with pytest.raises(ValueError, match="chunk_frames"):
        shift_future_chunks(latent, chunk_frames=0, depth=1)
    with pytest.raises(ValueError, match="depth"):
        shift_future_chunks(latent, chunk_frames=2, depth=0)
    with pytest.raises(ValueError, match="depth"):
        shift_future_chunks(latent, chunk_frames=2, depth=4)
    with pytest.raises(ValueError, match="rank 5"):
        shift_future_chunks(torch.zeros((1, 8, 1, 1)), chunk_frames=2, depth=1)
    with pytest.raises(ValueError, match="chunk_axis=1"):
        shift_future_chunks(latent, chunk_frames=2, depth=1, chunk_axis=2)
    with pytest.raises(ValueError, match="one selected target chunk"):
        sample_nf_sf_noise_and_timesteps(latent, chunk_frames=2)
