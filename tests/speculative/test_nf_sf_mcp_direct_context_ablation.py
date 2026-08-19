from __future__ import annotations

import importlib.util
import math
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from torch import nn

from utils.nf_sf_mcp_direct_context_ablation import (
    NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP,
    NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA,
    NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_TARGET_STEP,
    NF_SF_MCP_DIRECT_CLEAN_KV_FIXED_PROBE_SCHEMA,
    NF_SF_MCP_DIRECT_CLEAN_KV_PARENT_CHECKPOINT_SHA256,
    PRIMARY_FIXED_PROBE_ANCHOR_INDEX,
    PRIMARY_FIXED_PROBE_DEPTH,
    PRIMARY_FIXED_PROBE_RAW_TIMESTEP,
    PRIMARY_FIXED_PROBE_TARGET_CHUNK,
    ablation_decision_rule_metadata,
    ablation_step_numbers,
    build_ablation_provenance,
    build_ablation_run_plan,
    build_ablation_smoke_plan,
    direct_clean_context_kv_enabled,
    parameter_key_tuple,
    run_fixed_raw999_probe_for_ablation,
    run_nf_sf_full_sequence_forward_loss_for_ablation,
    validate_ablation_real_run_guards,
)
from utils.nf_sf_tensors import DEFAULT_S_MAIN, DEFAULT_S_MCP


ROOT = Path(__file__).resolve().parents[2]


class _SimpleRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * scale * self.weight


class _SimpleCausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.head = nn.Linear(dim, self.out_dim * math.prod(self.patch_size), bias=False)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        frames = int(e.shape[1])
        frame_seqlen = x.shape[1] // frames
        return self.head(x).unflatten(1, (frames, frame_seqlen))


def _sinusoidal_embedding_1d(dim: int, t: torch.Tensor) -> torch.Tensor:
    t = t.float().reshape(-1, 1)
    freqs = torch.arange(dim, device=t.device, dtype=torch.float32).reshape(1, -1)
    return torch.sin(t / 1000.0 + freqs).to(t.dtype)


def _scaled_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = float(q.shape[-1]) ** -0.5
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    probs = scores.softmax(dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v)


@contextmanager
def _load_mcp_with_stubs():
    names = (
        "wan",
        "wan.modules",
        "wan.modules.attention",
        "wan.modules.causal_model",
        "wan.modules.model",
    )
    sentinel = object()
    old = {name: sys.modules.get(name, sentinel) for name in names}
    wan = types.ModuleType("wan")
    wan_modules = types.ModuleType("wan.modules")
    attention = types.ModuleType("wan.modules.attention")
    causal_model = types.ModuleType("wan.modules.causal_model")
    model = types.ModuleType("wan.modules.model")
    attention.attention = _scaled_attention
    causal_model.CausalHead = _SimpleCausalHead
    causal_model.causal_rope_apply = lambda x, grid_sizes, freqs, start_frame=0: x
    model.WanLayerNorm = nn.LayerNorm
    model.WanRMSNorm = _SimpleRMSNorm
    model.sinusoidal_embedding_1d = _sinusoidal_embedding_1d
    sys.modules["wan"] = wan
    sys.modules["wan.modules"] = wan_modules
    sys.modules["wan.modules.attention"] = attention
    sys.modules["wan.modules.causal_model"] = causal_model
    sys.modules["wan.modules.model"] = model
    module_name = f"_mcp_direct_context_under_test_{id(object())}"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "wan" / "modules" / "mcp.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)
        for name in names:
            if old[name] is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old[name]


def test_mcp_direct_clean_kv_no_new_parameter_keys_and_anchor0_exact() -> None:
    with _load_mcp_with_stubs() as mcp:
        torch.manual_seed(1)
        stack = mcp.MCPStack(
            dim=4,
            ffn_dim=8,
            num_heads=2,
            out_dim=1,
            patch_size=(1, 1, 1),
            freq_dim=4,
            num_modules=1,
            num_layers=1,
            tap_layers=(0, 1),
            qk_norm=True,
        )
        before_keys = parameter_key_tuple(stack)
        features = (torch.randn(1, 3, 4), torch.randn(1, 3, 4))
        future = [torch.randn(1, 3, 4)]
        grid = [torch.tensor([[3, 1, 1]], dtype=torch.long)]
        timestep = [torch.full((1, 3), 1000.0)]
        control = stack(
            features=features,
            future_embeds=future,
            future_grid_sizes=grid,
            future_start_frames=[3],
            timesteps=timestep,
            freqs=torch.ones(32, 2),
        )[0]
        treatment_anchor0 = stack(
            features=features,
            future_embeds=future,
            future_grid_sizes=grid,
            future_start_frames=[3],
            timesteps=timestep,
            freqs=torch.ones(32, 2),
            direct_clean_context_kv=True,
            clean_context_features=(features[0][:, :0], features[1][:, :0]),
            clean_context_grid_sizes=torch.tensor([[0, 1, 1]], dtype=torch.long),
            clean_context_start_frame=0,
        )[0]

        torch.testing.assert_close(treatment_anchor0, control, rtol=0.0, atol=0.0)
        assert parameter_key_tuple(stack) == before_keys


def test_target_query_direct_clean_kv_shapes_and_separate_rope_metadata() -> None:
    with _load_mcp_with_stubs() as mcp:
        module = mcp.MCPSelfAttention(dim=4, num_heads=2, qk_norm=False)
        attention_calls = []
        rope_calls = []

        def fake_attention(q, k, v):
            attention_calls.append(
                {
                    "q_len": q.shape[1],
                    "k_len": k.shape[1],
                    "v_len": v.shape[1],
                }
            )
            return torch.zeros_like(q)

        def fake_rope(x, grid_sizes, freqs, start_frame=0):
            rope_calls.append(
                {
                    "seq_len": x.shape[1],
                    "frames": int(grid_sizes[0, 0].item()),
                    "start_frame": int(start_frame),
                }
            )
            return x

        module_attention = fake_attention
        module_rope = fake_rope
        mcp.attention = module_attention
        mcp.causal_rope_apply = module_rope

        x = torch.randn(1, 3, 4)
        clean = torch.randn(1, 15, 4)
        module(
            x,
            torch.tensor([[3, 1, 1]], dtype=torch.long),
            torch.ones(32, 2),
            start_frame=18,
            clean_context=clean,
            clean_context_grid_sizes=torch.tensor([[15, 1, 1]], dtype=torch.long),
            clean_context_start_frame=0,
        )

        assert attention_calls == [{"q_len": 3, "k_len": 18, "v_len": 18}]
        assert rope_calls == [
            {"seq_len": 3, "frames": 3, "start_frame": 18},
            {"seq_len": 3, "frames": 3, "start_frame": 18},
            {"seq_len": 15, "frames": 15, "start_frame": 0},
        ]


def test_depth1_only_direct_context_and_clean_tap_gradient() -> None:
    with _load_mcp_with_stubs() as mcp:
        torch.manual_seed(2)
        stack = mcp.MCPStack(
            dim=4,
            ffn_dim=8,
            num_heads=2,
            out_dim=1,
            patch_size=(1, 1, 1),
            freq_dim=4,
            num_modules=3,
            num_layers=1,
            tap_layers=(0, 1),
            qk_norm=False,
        )
        for module in stack.mcp_modules:
            nn.init.ones_(module.head.head.weight)

        clean_features = (
            torch.randn(1, 3, 4, requires_grad=True),
            torch.randn(1, 3, 4, requires_grad=True),
        )
        features = (torch.randn(1, 3, 4), torch.randn(1, 3, 4))
        future = [torch.randn(1, 3, 4) for _ in range(3)]
        grids = [torch.tensor([[3, 1, 1]], dtype=torch.long) for _ in range(3)]
        starts = [6, 9, 12]
        timesteps = [torch.full((1, 3), 1000.0) for _ in range(3)]
        calls = []

        for index, module in enumerate(stack.mcp_modules):
            original = module.forward

            def wrapped_forward(*args, _index=index, _original=original, **kwargs):
                calls.append((_index, bool(kwargs.get("direct_clean_context_kv", False))))
                return _original(*args, **kwargs)

            module.forward = wrapped_forward

        preds = stack(
            features=features,
            future_embeds=future,
            future_grid_sizes=grids,
            future_start_frames=starts,
            timesteps=timesteps,
            freqs=torch.ones(32, 2),
            direct_clean_context_kv=True,
            clean_context_features=clean_features,
            clean_context_grid_sizes=torch.tensor([[3, 1, 1]], dtype=torch.long),
            clean_context_start_frame=0,
        )
        sum(pred.sum() for pred in preds).backward()

        assert calls == [(0, True), (1, False), (2, False)]
        assert all(tensor.grad is not None for tensor in clean_features)
        assert sum(float(tensor.grad.abs().sum()) for tensor in clean_features) > 0.0


def test_wrapper_treatment_clean_context_prefix_counts_and_anchor0_control() -> None:
    from tests.speculative.test_nf_sf_full_sequence_next_forcing import _loss_batch
    from utils.nf_sf_training import build_full_sequence_mcp_anchor_inputs
    from utils.wan_wrapper import (
        FULL_SEQUENCE_CHUNK_FRAMES,
        FULL_SEQUENCE_FRAME_SEQ_LENGTH,
        WanDiffusionWrapper,
    )

    class FakeBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_frame_per_block = 3
            self.freqs = torch.ones(32, 2)

        def forward(self, x, **kwargs):
            batch = x.shape[0]
            features = tuple(torch.full((batch, 32760, 2), float(i)) for i in range(4))
            aux = {
                "features": features,
                "mcp_embeds": tuple(torch.zeros((batch, 4680, 2)) for _ in kwargs["mcp_patch_inputs"]),
                "mcp_grid_sizes": tuple(
                    torch.tensor([[3, 60, 26]], dtype=torch.long)
                    for _ in kwargs["mcp_patch_inputs"]
                ),
            }
            if kwargs.get("return_feature_halves", False):
                clean = tuple(torch.full((batch, 32760, 2), 10.0 + i) for i in range(4))
                noisy = features
                aux["clean_features"] = clean
                aux["noisy_features"] = noisy
                assert all(torch.equal(torch.cat([c, n], dim=1)[:, 32760:], n) for c, n in zip(clean, noisy))
            return torch.zeros((batch, 1, 21, 1, 1)), aux

    class FakeMCP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = []

        def forward(self, **kwargs):
            clean_context = kwargs.get("clean_context_features")
            clean_tokens = 0 if clean_context is None else int(clean_context[0].shape[1])
            clean_grid = kwargs.get("clean_context_grid_sizes")
            self.calls.append(
                {
                    "direct": bool(kwargs.get("direct_clean_context_kv", False)),
                    "clean_tokens": clean_tokens,
                    "clean_grid_frames": None if clean_grid is None else int(clean_grid[0, 0].item()),
                    "future_start_frames": tuple(kwargs["future_start_frames"]),
                    "feature_tokens": int(kwargs["features"][0].shape[1]),
                }
            )
            return [
                torch.zeros((1, 1, 3, 1, 1))
                for _ in kwargs["future_embeds"]
            ]

    wrapper = WanDiffusionWrapper.__new__(WanDiffusionWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = FakeBackbone()
    wrapper.mcp = FakeMCP()
    wrapper.mcp_tap_layers = (3, 11, 19, 29)
    wrapper.uniform_timestep = False
    wrapper.seq_len = 32760

    batch = _loss_batch()
    anchors = build_full_sequence_mcp_anchor_inputs(batch)
    wrapper.forward_full_sequence_next_forcing(
        noisy_image_or_video=batch.noisy_main,
        clean_x=batch.clean_target,
        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 2))},
        timestep_main=batch.timestep_main,
        mcp_anchor_inputs=anchors,
        direct_clean_context_kv=True,
    )

    chunk_tokens = FULL_SEQUENCE_CHUNK_FRAMES * FULL_SEQUENCE_FRAME_SEQ_LENGTH
    assert wrapper.mcp.calls[0]["direct"] is False
    assert wrapper.mcp.calls[0]["clean_tokens"] == 0
    assert wrapper.mcp.calls[1]["direct"] is True
    assert wrapper.mcp.calls[1]["clean_tokens"] == chunk_tokens
    assert wrapper.mcp.calls[1]["clean_grid_frames"] == 3
    assert wrapper.mcp.calls[1]["future_start_frames"][0] == 6
    assert wrapper.mcp.calls[5]["clean_tokens"] == 5 * chunk_tokens
    assert wrapper.mcp.calls[5]["clean_grid_frames"] == 5 * FULL_SEQUENCE_CHUNK_FRAMES
    assert wrapper.mcp.calls[5]["future_start_frames"][0] == 18
    assert all(call["feature_tokens"] == chunk_tokens for call in wrapper.mcp.calls)


class _ProbeFakeModel:
    def __init__(self) -> None:
        self.block_mask = None


class _ProbeFakeMCP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fusion = nn.Linear(16, 4, bias=False)
        nn.init.ones_(self.fusion.weight)
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(
            {
                "direct": bool(kwargs.get("direct_clean_context_kv", False)),
                "future_start_frames": tuple(kwargs["future_start_frames"]),
                "clean_tokens": 0
                if kwargs.get("clean_context_features") is None
                else int(kwargs["clean_context_features"][0].shape[1]),
            }
        )
        return [
            torch.zeros((1, 1, 3, 1, 1))
            for _ in kwargs["future_embeds"]
        ]


class _ProbeFakeGenerator(nn.Module):
    def __init__(self, *, create_block_mask: bool = True) -> None:
        super().__init__()
        self.model = _ProbeFakeModel()
        self.mcp = _ProbeFakeMCP()
        self.grad_enabled = []
        self.create_block_mask = bool(create_block_mask)
        self.created_block_mask = object()
        self.forward_saw_block_mask_none = []

    def forward_full_sequence_next_forcing(self, **kwargs):
        self.grad_enabled.append(torch.is_grad_enabled())
        self.forward_saw_block_mask_none.append(self.model.block_mask is None)
        if self.create_block_mask:
            self.model.block_mask = self.created_block_mask
        direct = bool(kwargs.get("direct_clean_context_kv", False))
        noisy = kwargs["noisy_image_or_video"]
        batch, _frames, channels, height, width = noisy.shape
        depth_outputs = [
            torch.zeros((batch, 6, 3, channels, height, width), dtype=noisy.dtype),
            torch.zeros((batch, 5, 3, channels, height, width), dtype=noisy.dtype),
            torch.zeros((batch, 4, 3, channels, height, width), dtype=noisy.dtype),
        ]
        for anchor in kwargs["mcp_anchor_inputs"]:
            anchor_index = int(anchor["anchor_index"])
            depths = tuple(int(value) for value in anchor["depths"])
            starts = tuple(int(value) for value in anchor["future_start_frames"])
            features = tuple(
                torch.full((batch, 4680, 4), float(anchor_index + offset))
                for offset in range(4)
            )
            future_embeds = [
                torch.full((batch, 4680, 4), float(depth), dtype=noisy.dtype)
                for depth in depths
            ]
            grids = [
                torch.tensor([[3, 60, 26]], dtype=torch.long)
                for _ in depths
            ]
            mcp_kwargs = {}
            if direct and anchor_index > 0:
                mcp_kwargs = {
                    "direct_clean_context_kv": True,
                    "clean_context_features": tuple(
                        torch.full((batch, anchor_index * 4680, 4), 10.0 + offset)
                        for offset in range(4)
                    ),
                    "clean_context_grid_sizes": torch.tensor(
                        [[anchor_index * 3, 60, 26]],
                        dtype=torch.long,
                    ),
                    "clean_context_start_frame": 0,
                }
            self.mcp(
                features=features,
                future_embeds=future_embeds,
                future_grid_sizes=grids,
                future_start_frames=list(starts),
                timesteps=list(anchor["timesteps"]),
                freqs=torch.ones(32, 2),
                **mcp_kwargs,
            )
        return types.SimpleNamespace(
            main_flow_pred=torch.zeros_like(noisy),
            mcp_flow_preds_by_depth=tuple(depth_outputs),
            tap_shapes=((batch, 32760, 4),) * 4,
            anchor_token_slices=tuple((i * 4680, (i + 1) * 4680) for i in range(7)),
            main_backbone_forward_count=1,
            future_embedding_order="depth_major",
        )


def _fixed_probe_inputs():
    import utils.nf_sf_first_mcp_route_equivalence as route_eq

    source_noise = torch.linspace(0.0, 1.0, 21).reshape(1, 21, 1, 1, 1)
    teacher_target = torch.linspace(1.0, 2.0, 21).reshape(1, 21, 1, 1, 1)
    main_scheduler = route_eq.build_flow_match_scheduler(shift=DEFAULT_S_MAIN, device="cpu")
    mcp_scheduler = route_eq.build_flow_match_scheduler(shift=DEFAULT_S_MCP, device="cpu")
    return source_noise, teacher_target, main_scheduler, mcp_scheduler


def _run_probe(generator: _ProbeFakeGenerator, *, arm: str = "control") -> dict:
    source_noise, teacher_target, main_scheduler, mcp_scheduler = _fixed_probe_inputs()
    return run_fixed_raw999_probe_for_ablation(
        generator,
        arm=arm,
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        source_noise=source_noise,
        teacher_target=teacher_target,
        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 4))},
        sample_identity="fixed-validation-identity",
    )


def test_fixed_raw999_probe_contract_control_and_treatment() -> None:
    source_noise, teacher_target, main_scheduler, mcp_scheduler = _fixed_probe_inputs()

    control_generator = _ProbeFakeGenerator()
    treatment_generator = _ProbeFakeGenerator()
    control = run_fixed_raw999_probe_for_ablation(
        control_generator,
        arm="control",
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        source_noise=source_noise,
        teacher_target=teacher_target,
        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 4))},
        sample_identity="fixed-validation-identity",
    )
    treatment = run_fixed_raw999_probe_for_ablation(
        treatment_generator,
        arm="direct_clean_kv",
        main_scheduler=main_scheduler,
        mcp_scheduler=mcp_scheduler,
        source_noise=source_noise,
        teacher_target=teacher_target,
        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 4))},
        sample_identity="fixed-validation-identity",
    )

    assert control["schema"] == NF_SF_MCP_DIRECT_CLEAN_KV_FIXED_PROBE_SCHEMA
    assert control["raw_timestep"] == PRIMARY_FIXED_PROBE_RAW_TIMESTEP == 999
    assert control["depth"] == PRIMARY_FIXED_PROBE_DEPTH == 1
    assert control["anchor_index"] == PRIMARY_FIXED_PROBE_ANCHOR_INDEX == 1
    assert control["target_chunk"] == PRIMARY_FIXED_PROBE_TARGET_CHUNK == 2
    assert control["solver_loop"] is False
    assert control["direct_clean_context_kv"] is False
    assert control["clean_context_token_count"] == 0
    assert control["block_mask_before_was_none"] is True
    assert control["block_mask_recreated_for_probe"] is True
    assert control["block_mask_restored_exact"] is True
    assert control["block_mask_policy"] == "reset_to_none_recreate_teacher_forcing_then_restore"
    assert treatment["direct_clean_context_kv"] is True
    assert treatment["clean_context_token_count"] == 4680
    assert treatment["clean_context_grid_frames"] == 3
    assert treatment["clean_context_start_frame"] == 0
    assert treatment["target_start_frame"] == 6
    assert treatment["selected_mcp_future_start_frames"] == [6, 9, 12]
    assert treatment["paper_exact_reproduction"] is False
    assert treatment["canonical_training_eligible"] is False
    assert treatment["canonical_deployment_eligible"] is False
    for key in (
        "source_noise_sha256",
        "target_sha256",
        "exact_flow_sha256",
        "target_mcp_input_sha256",
    ):
        assert control[key] == treatment[key]
    assert "mcp1_flow_mse_to_exact" in treatment
    assert "main_flow_mse_to_exact" in treatment
    assert all(value is False for value in control_generator.grad_enabled)
    assert all(value is False for value in treatment_generator.grad_enabled)


def test_fixed_raw999_probe_resets_cached_block_mask_and_restores_identity() -> None:
    generator = _ProbeFakeGenerator()
    original = object()
    generator.model.block_mask = original

    record = _run_probe(generator)

    assert generator.forward_saw_block_mask_none == [True]
    assert generator.model.block_mask is original
    assert record["block_mask_before_was_none"] is False
    assert record["block_mask_recreated_for_probe"] is True
    assert record["block_mask_restored_exact"] is True


def test_fixed_raw999_probe_fresh_block_mask_restores_none() -> None:
    generator = _ProbeFakeGenerator()

    record = _run_probe(generator)

    assert generator.forward_saw_block_mask_none == [True]
    assert generator.model.block_mask is None
    assert record["block_mask_before_was_none"] is True
    assert record["block_mask_recreated_for_probe"] is True
    assert record["block_mask_restored_exact"] is True


def test_fixed_raw999_probe_fails_closed_if_block_mask_not_recreated_and_restores() -> None:
    generator = _ProbeFakeGenerator(create_block_mask=False)
    original = object()
    generator.model.block_mask = original

    with pytest.raises(RuntimeError, match="did not recreate teacher-forcing block_mask"):
        _run_probe(generator)

    assert generator.forward_saw_block_mask_none == [True]
    assert generator.model.block_mask is original


def test_ablation_plan_provenance_decision_rules_and_guards() -> None:
    steps = ablation_step_numbers()
    assert steps[0] == 6501
    assert steps[-1] == 7000
    assert len(steps) == 500
    assert NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_PARENT_STEP == 6500
    assert NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_TARGET_STEP == 7000

    control = build_ablation_run_plan("control")
    treatment = build_ablation_run_plan("direct_clean_kv")
    assert control.validation_steps == (7000,)
    assert control.checkpoint_steps == (7000,)
    assert control.no_validation6500 is True
    assert control.direct_clean_context_kv is False
    assert treatment.direct_clean_context_kv is True
    assert treatment.depth1_direct_context_only is True
    assert direct_clean_context_kv_enabled("direct_clean_kv") is True

    provenance = build_ablation_provenance(
        arm="direct_clean_kv",
        runtime_git_sha="c3f89888bf6da31b48650f0a680dd6534943f56f",
        semantic_lock_fingerprint="abc",
    )
    assert provenance["schema"] == NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA
    assert provenance["canonical_training_eligible"] is False
    assert provenance["canonical_deployment_eligible"] is False
    assert provenance["new_trainable_parameters"] is False
    assert provenance["objective_changed"] is False
    assert provenance["optimizer_changed"] is False
    assert provenance["data_changed"] is False
    assert provenance["rng_changed"] is False
    assert provenance["main_attention_changed"] is False
    assert provenance["depth2_depth3_direct_context_changed"] is False
    assert provenance["opd_changed"] is False
    assert provenance["on_policy_changed"] is False
    assert provenance["noisy_history_augmentation_changed"] is False

    decision = ablation_decision_rule_metadata()
    assert decision["auto_declare_go"] is False
    assert decision["baseline_step6500"]["raw999_mcp1"] == pytest.approx(0.11986814439296722)

    outside = Path("D:/nf_sf_direct_clean_kv_ablation_outside_repo")
    validate_ablation_real_run_guards(
        arm="control",
        parent_step=6500,
        parent_checkpoint_sha256=NF_SF_MCP_DIRECT_CLEAN_KV_PARENT_CHECKPOINT_SHA256,
        output_dir=outside,
        repo_root=ROOT,
        staged_changes=False,
        tracked_changes=False,
    )
    with pytest.raises(RuntimeError, match="outside the repo"):
        validate_ablation_real_run_guards(
            arm="control",
            parent_step=6500,
            parent_checkpoint_sha256=NF_SF_MCP_DIRECT_CLEAN_KV_PARENT_CHECKPOINT_SHA256,
            output_dir=ROOT / "inside",
            repo_root=ROOT,
            staged_changes=False,
            tracked_changes=False,
        )


def test_ablation_forward_control_and_treatment_switch_only_path() -> None:
    from tests.speculative.test_nf_sf_full_sequence_next_forcing import _loss_batch

    class FakeGenerator:
        def __init__(self) -> None:
            self.calls = []

        def forward_full_sequence_next_forcing(self, **kwargs):
            direct = bool(kwargs.get("direct_clean_context_kv", False))
            self.calls.append(direct)
            batch = kwargs["noisy_image_or_video"].shape[0]
            return types.SimpleNamespace(
                main_flow_pred=torch.zeros((batch, 21, 1, 1, 1)),
                mcp_flow_preds_by_depth=(
                    torch.zeros((batch, 6, 3, 1, 1, 1)),
                    torch.zeros((batch, 5, 3, 1, 1, 1)),
                    torch.zeros((batch, 4, 3, 1, 1, 1)),
                ),
                tap_shapes=((batch, 32760, 2),) * 4,
                anchor_token_slices=tuple(
                    (i * 4680, (i + 1) * 4680)
                    for i in range(7)
                ),
                main_backbone_forward_count=1,
                future_embedding_order="depth_major",
            )

    batch = _loss_batch()
    generator = FakeGenerator()
    run_nf_sf_full_sequence_forward_loss_for_ablation(
        generator,
        arm="control",
        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 2))},
        noisy_batch=batch,
    )
    run_nf_sf_full_sequence_forward_loss_for_ablation(
        generator,
        arm="direct_clean_kv",
        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 2))},
        noisy_batch=batch,
    )
    assert generator.calls == [False, True]


class _Sample:
    def __init__(self, target: torch.Tensor) -> None:
        self.target_latent = target
        self.source_noise = torch.zeros_like(target)


class _Acquire:
    def __init__(self, value) -> None:
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _TinyTeacherStore:
    validation_identities = ("val0",)

    def __init__(self, target: torch.Tensor) -> None:
        self.target = target

    def train_identity_for_step(self, step: int) -> str:
        assert int(step) == 6501
        return "train0"

    def acquire(self, identity: str):
        return _Acquire(_Sample(self.target.clone()))


class _TinyConditionalStore:
    def acquire(self, identity: str):
        return _Acquire({"prompt_embeds": torch.zeros((1, 1, 1))})


class _ValidationFakeGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.grad_enabled = []

    def forward_full_sequence_next_forcing(self, **kwargs):
        self.grad_enabled.append(torch.is_grad_enabled())
        clean = kwargs["clean_x"]
        batch = clean.shape[0]
        return types.SimpleNamespace(
            main_flow_pred=torch.zeros_like(clean),
            mcp_flow_preds_by_depth=(
                torch.zeros((batch, 6, 3, 1, 1, 1)),
                torch.zeros((batch, 5, 3, 1, 1, 1)),
                torch.zeros((batch, 4, 3, 1, 1, 1)),
            ),
            tap_shapes=((batch, 32760, 2),) * 4,
            anchor_token_slices=tuple((i * 4680, (i + 1) * 4680) for i in range(7)),
            main_backbone_forward_count=1,
            future_embedding_order="depth_major",
        )


def _cpu_rng_helpers():
    def capture_global_rng_state(device):
        assert torch.device(device).type == "cpu"
        return {"cpu": torch.get_rng_state().detach().cpu().clone()}

    def assert_global_rng_equal(before, after):
        assert before.keys() == after.keys()
        for key in before:
            assert torch.equal(before[key], after[key])

    return {
        "capture_global_rng_state": capture_global_rng_state,
        "assert_global_rng_equal": assert_global_rng_equal,
        "target_latent_from_sample": lambda sample: sample.target_latent,
    }


def test_validation_runs_no_grad_and_preserves_rng() -> None:
    from scripts import train_nf_sf_mcp_direct_context_ablation as runner
    from tests.speculative.test_nf_sf_full_sequence_next_forcing import _scheduler

    target = torch.zeros((1, 21, 1, 1, 1), dtype=torch.float32)
    generator = _ValidationFakeGenerator()
    train_rng = torch.Generator(device="cpu")
    train_rng.manual_seed(123)
    before = train_rng.get_state().clone()
    result = runner.run_ablation_validation(
        helpers=_cpu_rng_helpers(),
        arm="direct_clean_kv",
        generator=generator,
        scheduler_main=_scheduler(DEFAULT_S_MAIN),
        scheduler_mcp=_scheduler(DEFAULT_S_MCP),
        teacher_store=_TinyTeacherStore(target),
        conditional_store=_TinyConditionalStore(),
        validation_seed=5,
        train_rng=train_rng,
        device=torch.device("cpu"),
        dtype=torch.float32,
        global_step=7000,
    )

    assert result["schema"].endswith("_validation_v1")
    assert generator.grad_enabled == [False]
    assert torch.equal(before, train_rng.get_state())


def test_control_train_step_delegates_to_canonical_runner() -> None:
    from scripts import train_nf_sf_mcp_direct_context_ablation as runner

    calls = []

    def canonical_train_step(**kwargs):
        calls.append(kwargs)
        return {
            "global_step": kwargs["global_step"],
            "sample_identity": "train0",
            "sample_cursor": {"global_step": kwargs["global_step"]},
            "total_loss": 1.0,
        }

    train_rng = torch.Generator(device="cpu")
    train_rng.manual_seed(1)
    record = runner.run_ablation_train_step(
        helpers={"run_full_sequence_train_step": canonical_train_step},
        arm="control",
        generator=object(),
        optimizer=object(),
        scheduler_main=object(),
        scheduler_mcp=object(),
        teacher_store=object(),
        conditional_store=object(),
        train_rng=train_rng,
        global_step=6501,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert len(calls) == 1
    assert calls[0]["objective_mode"] == "next_forcing_full"
    assert calls[0]["smoke"] is False
    assert record["canonical_control_path"] is True
    assert record["direct_clean_context_kv"] is False


class _TrainStepFakeGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward_full_sequence_next_forcing(self, **kwargs):
        self.calls.append(bool(kwargs.get("direct_clean_context_kv", False)))
        clean = kwargs["clean_x"]
        batch = clean.shape[0]
        main = self.weight * torch.ones_like(clean)
        return types.SimpleNamespace(
            main_flow_pred=main,
            mcp_flow_preds_by_depth=(
                self.weight * torch.ones((batch, 6, 3, 1, 1, 1)),
                self.weight * torch.ones((batch, 5, 3, 1, 1, 1)),
                self.weight * torch.ones((batch, 4, 3, 1, 1, 1)),
            ),
            tap_shapes=((batch, 32760, 2),) * 4,
            anchor_token_slices=tuple((i * 4680, (i + 1) * 4680) for i in range(7)),
            main_backbone_forward_count=1,
            future_embedding_order="depth_major",
        )


def test_treatment_train_step_asserts_rng_after_step_and_clears_grads() -> None:
    from scripts import train_nf_sf_mcp_direct_context_ablation as runner
    from tests.speculative.test_nf_sf_full_sequence_next_forcing import _scheduler

    assert_calls = []

    def capture_global_rng_state(device):
        return {"cpu": torch.get_rng_state().detach().cpu().clone()}

    def assert_global_rng_equal(before, after):
        assert_calls.append((before, after))
        for key in before:
            assert torch.equal(before[key], after[key])

    def assert_finite_loss(loss, name):
        assert name == "total_loss"
        assert torch.isfinite(loss).all()

    helpers = {
        "capture_global_rng_state": capture_global_rng_state,
        "assert_global_rng_equal": assert_global_rng_equal,
        "assert_finite_loss": assert_finite_loss,
        "has_nonfinite_grad": lambda module: False,
        "loss_breakdown_to_floats": lambda losses: {
            "total_loss": float(losses.total_loss.detach().item()),
            "main_loss": float(losses.main_loss.detach().item()),
            "mcp_depth_losses": [
                float(loss.detach().item()) for loss in losses.mcp_depth_losses
            ],
            "mcp_anchor_losses": [],
        },
        "target_latent_from_sample": lambda sample: sample.target_latent,
    }
    target = torch.zeros((1, 21, 1, 1, 1), dtype=torch.float32)
    generator = _TrainStepFakeGenerator()
    optimizer = torch.optim.SGD(generator.parameters(), lr=0.1)
    train_rng = torch.Generator(device="cpu")
    train_rng.manual_seed(17)
    record = runner.run_ablation_train_step(
        helpers=helpers,
        arm="direct_clean_kv",
        generator=generator,
        optimizer=optimizer,
        scheduler_main=_scheduler(DEFAULT_S_MAIN),
        scheduler_mcp=_scheduler(DEFAULT_S_MCP),
        teacher_store=_TinyTeacherStore(target),
        conditional_store=_TinyConditionalStore(),
        train_rng=train_rng,
        global_step=6501,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert generator.calls == [True]
    assert len(assert_calls) == 3
    assert generator.weight.grad is None
    assert record["direct_clean_context_kv"] is True
    assert record["canonical_control_path"] is False


def test_smoke_plan_dry_run_and_log_interval_guard() -> None:
    from scripts import train_nf_sf_mcp_direct_context_ablation as runner

    smoke = build_ablation_smoke_plan("direct_clean_kv")
    assert smoke["engineering_smoke"] is True
    assert smoke["first_step"] == 6501
    assert smoke["target_step"] == 6501
    assert smoke["update_count"] == 1
    assert smoke["validation_steps"] == ()
    assert smoke["checkpoint_steps"] == ()
    assert len(ablation_step_numbers()) == 500

    args = runner.parse_args(
        [
            "--arm",
            "direct_clean_kv",
            "--engineering_smoke_one_step",
            "--parent_checkpoint",
            "checkpoint_step006500.pt",
            "--expected_runtime_git_sha",
            "c3f89888bf6da31b48650f0a680dd6534943f56f",
            "--output_dir",
            "D:/nf_sf_direct_clean_kv_ablation_outside_repo",
        ]
    )
    summary = runner.run_ablation(args)
    assert summary["dry_run"] is True
    assert summary["engineering_smoke"] is True
    assert summary["run_plan"]["update_count"] == 1
    assert summary["run_plan"]["validation_steps"] == ()

    bad_args = runner.parse_args(
        [
            "--arm",
            "control",
            "--parent_checkpoint",
            "checkpoint_step006500.pt",
            "--expected_runtime_git_sha",
            "c3f89888bf6da31b48650f0a680dd6534943f56f",
            "--output_dir",
            "D:/nf_sf_direct_clean_kv_ablation_outside_repo",
            "--log_interval",
            "0",
        ]
    )
    with pytest.raises(ValueError, match="log_interval"):
        runner.run_ablation(bad_args)


def test_source_guards_for_no_detach_and_noncanonical_checkpoint_schema() -> None:
    from utils.nf_sf_training import FULL_SEQUENCE_TRAINER_SCHEMA

    causal_text = (ROOT / "wan" / "modules" / "causal_model.py").read_text(encoding="utf-8")
    halves_snippet = causal_text[
        causal_text.index("if features is not None and return_feature_halves"):
        causal_text.index("x = self.head", causal_text.index("if features is not None and return_feature_halves"))
    ]
    assert ".detach" not in halves_snippet
    assert ".clone" not in halves_snippet
    assert ".cpu" not in halves_snippet
    assert 'aux["clean_features"]' in causal_text
    assert 'aux["noisy_features"]' in causal_text
    assert NF_SF_MCP_DIRECT_CLEAN_KV_ABLATION_SCHEMA != FULL_SEQUENCE_TRAINER_SCHEMA

    runner_text = (
        ROOT / "scripts" / "train_nf_sf_mcp_direct_context_ablation.py"
    ).read_text(encoding="utf-8")
    assert "canonical_training_eligible\": False" in runner_text
    assert "canonical_deployment_eligible\": False" in runner_text
