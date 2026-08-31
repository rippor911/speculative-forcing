from __future__ import annotations

import copy
import importlib.util
import math
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from torch import nn


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
        self.patch_size = tuple(int(value) for value in patch_size)
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
    module_name = f"_mcp_paper_fidelity_under_test_{id(object())}"
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


def _parameter_key_tuple(module: nn.Module) -> tuple[str, ...]:
    return tuple(name for name, _ in module.named_parameters())


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _grad_l1(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().abs().sum().item())
    return total


def _tiny_case_tensors(mcp, *, num_layers: int = 1, seed: int = 11):
    torch.manual_seed(seed)
    module = mcp.MCPModule(
        dim=6,
        ffn_dim=12,
        num_heads=2,
        num_layers=num_layers,
        out_dim=1,
        patch_size=(1, 1, 1),
        freq_dim=6,
        qk_norm=True,
    )
    clean_frames = 6
    target_frames = 3
    chunk_tokens = 3
    total_tokens = clean_frames + target_frames
    x = (
        torch.arange(total_tokens * module.dim, dtype=torch.float32)
        .reshape(1, total_tokens, module.dim)
        .div(17.0)
        .sub(1.0)
    )
    target_grid = torch.tensor([[target_frames, 1, 1]], dtype=torch.long)
    clean_grid = torch.tensor([[clean_frames, 1, 1]], dtype=torch.long)
    target_start = 6
    sequence_timestep = torch.cat(
        [
            torch.zeros((1, clean_frames), dtype=torch.float32),
            torch.full((1, target_frames), 777.0, dtype=torch.float32),
        ],
        dim=1,
    )
    e = module.time_embedding(
        mcp.sinusoidal_embedding_1d(module.freq_dim, sequence_timestep.flatten()).type_as(x)
    )
    e = (
        module.time_projection(e)
        .unflatten(1, (6, module.dim))
        .unflatten(dim=0, sizes=sequence_timestep.shape)
    )
    return {
        "module": module,
        "x": x,
        "e": e,
        "sequence_timestep": sequence_timestep,
        "clean_token_count": clean_frames,
        "target_token_count": target_frames,
        "chunk_tokens": chunk_tokens,
        "target_grid": target_grid,
        "clean_grid": clean_grid,
        "target_start": target_start,
        "freqs": torch.ones(16, 2),
    }


def _logical_rope_positions(x: torch.Tensor, grid_sizes: torch.Tensor, start_frame: int) -> torch.Tensor:
    frames = int(grid_sizes[0, 0].item())
    if frames <= 0 or x.shape[1] % frames != 0:
        raise ValueError("tiny rope test expects positive frame-aligned token counts")
    frame_seqlen = x.shape[1] // frames
    return torch.arange(
        int(start_frame),
        int(start_frame) + frames,
        dtype=x.dtype,
        device=x.device,
    ).repeat_interleave(frame_seqlen)


def _logical_rope_apply(x, grid_sizes, freqs, start_frame=0, *, records=None):
    positions = _logical_rope_positions(x, grid_sizes, int(start_frame))
    if records is not None:
        records.append(
            {
                "seq_len": int(x.shape[1]),
                "positions": [int(value) for value in positions.detach().cpu().tolist()],
            }
        )
    phase = positions.view(1, -1, 1, 1) / 29.0
    dims = torch.arange(x.shape[-1], dtype=x.dtype, device=x.device).view(1, 1, 1, -1)
    return x * torch.cos(phase + dims / 11.0) + torch.sin(phase + dims / 13.0)


def _dense_appendix_a_mask(clean_tokens: int, target_tokens: int, chunk_tokens: int, device) -> torch.Tensor:
    total = clean_tokens + target_tokens
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    for q_idx in range(total):
        for kv_idx in range(total):
            if q_idx < clean_tokens:
                mask[q_idx, kv_idx] = (
                    kv_idx < clean_tokens
                    and (kv_idx // chunk_tokens) <= (q_idx // chunk_tokens)
                )
            else:
                mask[q_idx, kv_idx] = kv_idx < clean_tokens or clean_tokens <= kv_idx < total
    return mask


def _dense_masked_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scale = float(q.shape[-1]) ** -0.5
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    scores = scores.masked_fill(~mask.view(1, 1, *mask.shape), -torch.finfo(scores.dtype).max)
    probs = scores.softmax(dim=-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, v)


def _dense_paper_self_attention_output(mcp, self_attn, x, case):
    batch, seq_len = x.shape[:2]
    heads = self_attn.num_heads
    head_dim = self_attn.head_dim
    target_tokens = case["target_token_count"]
    clean_tokens = case["clean_token_count"]
    q = self_attn.norm_q(self_attn.q(x)).view(batch, seq_len, heads, head_dim)
    k = self_attn.norm_k(self_attn.k(x)).view(batch, seq_len, heads, head_dim)
    v = self_attn.v(x).view(batch, seq_len, heads, head_dim)
    q_clean, q_target = q[:, :clean_tokens], q[:, clean_tokens:]
    k_clean, k_target = k[:, :clean_tokens], k[:, clean_tokens:]
    v_clean, v_target = v[:, :clean_tokens], v[:, clean_tokens:]
    roped_q = torch.cat(
        [
            mcp.causal_rope_apply(q_clean, case["clean_grid"], case["freqs"], start_frame=0).type_as(v),
            mcp.causal_rope_apply(
                q_target,
                case["target_grid"],
                case["freqs"],
                start_frame=case["target_start"],
            ).type_as(v),
        ],
        dim=1,
    )
    roped_k = torch.cat(
        [
            mcp.causal_rope_apply(k_clean, case["clean_grid"], case["freqs"], start_frame=0).type_as(v),
            mcp.causal_rope_apply(
                k_target,
                case["target_grid"],
                case["freqs"],
                start_frame=case["target_start"],
            ).type_as(v),
        ],
        dim=1,
    )
    mask = _dense_appendix_a_mask(
        clean_tokens,
        target_tokens,
        case["chunk_tokens"],
        x.device,
    )
    y = _dense_masked_attention(roped_q, roped_k, torch.cat([v_clean, v_target], dim=1), mask)
    return self_attn.o(y.flatten(2))


def _block_attention_input(block, x, e):
    num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
    e_chunks = (block.modulation.unsqueeze(1) + e).chunk(6, dim=2)
    attn_input = (
        block.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
        * (1 + e_chunks[1])
        + e_chunks[0]
    ).flatten(1, 2)
    return attn_input, e_chunks, num_frames, frame_seqlen


def _dense_paper_block_forward(mcp, block, x, e, case):
    attn_input, e_chunks, num_frames, frame_seqlen = _block_attention_input(block, x, e)
    attn_output = _dense_paper_self_attention_output(mcp, block.self_attn, attn_input, case)
    x = x + (attn_output.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e_chunks[2]).flatten(1, 2)
    ffn_input = (
        block.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
        * (1 + e_chunks[4])
        + e_chunks[3]
    ).flatten(1, 2)
    y = block.ffn(ffn_input)
    x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e_chunks[5]).flatten(1, 2)
    return x, attn_output


def _sparse_paper_block_forward(block, x, e, case):
    return block(
        x,
        e,
        case["target_grid"],
        case["freqs"],
        case["target_start"],
        paper_fidelity_mcp1_mask=True,
        target_token_count=case["target_token_count"],
        target_grid_sizes=case["target_grid"],
        target_start_frame=case["target_start"],
        clean_prefix_grid_sizes=case["clean_grid"],
        clean_prefix_start_frame=0,
    )


def _max_abs_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach() - right.detach()).abs().max().item())


def _assert_close_with_error(left: torch.Tensor, right: torch.Tensor) -> float:
    torch.testing.assert_close(left, right, rtol=1.0e-5, atol=1.0e-6)
    return _max_abs_error(left, right)


def compute_chunk_sparse_equivalence_metrics() -> dict[str, float | int]:
    with _load_mcp_with_stubs() as mcp:
        rope_records = []
        mcp.causal_rope_apply = lambda x, grid_sizes, freqs, start_frame=0: _logical_rope_apply(
            x,
            grid_sizes,
            freqs,
            start_frame=start_frame,
            records=rope_records,
        )
        mcp.attention = _scaled_attention
        case = _tiny_case_tensors(mcp, num_layers=3, seed=21)
        module = case["module"]
        before_count = _parameter_count(module)
        block = module.blocks[0]
        attn_input, _, _, _ = _block_attention_input(block, case["x"], case["e"])
        dense_attn = _dense_paper_self_attention_output(mcp, block.self_attn, attn_input, case)
        sparse_attn = block.self_attn(
            attn_input,
            case["target_grid"],
            case["freqs"],
            case["target_start"],
            paper_fidelity_mcp1_mask=True,
            target_token_count=case["target_token_count"],
            target_grid_sizes=case["target_grid"],
            target_start_frame=case["target_start"],
            clean_prefix_grid_sizes=case["clean_grid"],
            clean_prefix_start_frame=0,
        )
        dense_one, dense_attn_again = _dense_paper_block_forward(mcp, block, case["x"], case["e"], case)
        sparse_one = _sparse_paper_block_forward(block, case["x"], case["e"], case)

        dense_three = case["x"]
        sparse_three = case["x"]
        for next_block in module.blocks:
            dense_three, _ = _dense_paper_block_forward(mcp, next_block, dense_three, case["e"], case)
            sparse_three = _sparse_paper_block_forward(next_block, sparse_three, case["e"], case)

        metrics = {
            "one_block_attn_max_error": max(
                _assert_close_with_error(dense_attn, sparse_attn),
                _assert_close_with_error(dense_attn_again, sparse_attn),
            ),
            "one_block_clean_max_error": _assert_close_with_error(
                dense_one[:, : case["clean_token_count"]],
                sparse_one[:, : case["clean_token_count"]],
            ),
            "one_block_target_max_error": _assert_close_with_error(
                dense_one[:, case["clean_token_count"] :],
                sparse_one[:, case["clean_token_count"] :],
            ),
            "three_block_clean_max_error": _assert_close_with_error(
                dense_three[:, : case["clean_token_count"]],
                sparse_three[:, : case["clean_token_count"]],
            ),
            "three_block_target_max_error": _assert_close_with_error(
                dense_three[:, case["clean_token_count"] :],
                sparse_three[:, case["clean_token_count"] :],
            ),
            "parameter_count_before": before_count,
            "parameter_count_after": _parameter_count(module),
        }
        expected_clean_positions = [0, 1, 2, 3, 4, 5]
        expected_target_positions = [6, 7, 8]
        metrics["rope_call_count"] = len(rope_records)
        metrics["rope_positions_match"] = int(
            all(
                (
                    record["seq_len"] == case["clean_token_count"]
                    and record["positions"] == expected_clean_positions
                )
                or (
                    record["seq_len"] == case["target_token_count"]
                    and record["positions"] == expected_target_positions
                )
                for record in rope_records
            )
        )
        metrics["clean_timestep_sum"] = float(
            case["sequence_timestep"][:, : case["clean_token_count"]].sum().item()
        )
        metrics["target_timestep_min"] = float(
            case["sequence_timestep"][:, case["clean_token_count"] :].min().item()
        )
        metrics["target_timestep_max"] = float(
            case["sequence_timestep"][:, case["clean_token_count"] :].max().item()
        )
        return metrics


def compute_chunk_sparse_gradient_metrics() -> dict[str, float]:
    with _load_mcp_with_stubs() as mcp:
        mcp.causal_rope_apply = _logical_rope_apply
        mcp.attention = _scaled_attention
        dense_case = _tiny_case_tensors(mcp, num_layers=1, seed=31)
        sparse_case = dict(dense_case)
        sparse_case["module"] = copy.deepcopy(dense_case["module"])
        sparse_case["e"] = dense_case["e"].detach().clone()
        dense_module = dense_case["module"]
        sparse_module = sparse_case["module"]
        x_dense = dense_case["x"].detach().clone().requires_grad_(True)
        x_sparse = dense_case["x"].detach().clone().requires_grad_(True)
        dense_out, _ = _dense_paper_block_forward(
            mcp,
            dense_module.blocks[0],
            x_dense,
            dense_case["e"],
            dense_case,
        )
        sparse_out = _sparse_paper_block_forward(
            sparse_module.blocks[0],
            x_sparse,
            sparse_case["e"],
            sparse_case,
        )
        target_weight = torch.linspace(
            -0.3,
            0.4,
            dense_case["target_token_count"] * dense_module.dim,
            dtype=torch.float32,
        ).reshape(1, dense_case["target_token_count"], dense_module.dim)
        (dense_out[:, -dense_case["target_token_count"] :] * target_weight).sum().backward()
        (sparse_out[:, -sparse_case["target_token_count"] :] * target_weight).sum().backward()
        metrics = {
            "clean_input_grad_max_error": _assert_close_with_error(
                x_dense.grad[:, : dense_case["clean_token_count"]],
                x_sparse.grad[:, : sparse_case["clean_token_count"]],
            ),
            "target_input_grad_max_error": _assert_close_with_error(
                x_dense.grad[:, dense_case["clean_token_count"] :],
                x_sparse.grad[:, sparse_case["clean_token_count"] :],
            ),
        }
        for name in ("q", "k", "v", "o"):
            dense_linear = getattr(dense_module.blocks[0].self_attn, name)
            sparse_linear = getattr(sparse_module.blocks[0].self_attn, name)
            metrics[f"{name}_weight_grad_max_error"] = _assert_close_with_error(
                dense_linear.weight.grad,
                sparse_linear.weight.grad,
            )
            metrics[f"{name}_bias_grad_max_error"] = _assert_close_with_error(
                dense_linear.bias.grad,
                sparse_linear.bias.grad,
            )
        metrics["overall_gradient_max_error"] = max(metrics.values())
        return metrics


def test_paper_fidelity_mcp1_mask_allows_appendix_a_rows() -> None:
    with _load_mcp_with_stubs() as mcp:
        allows = mcp.paper_fidelity_mcp1_mask_allows
        kwargs = {"clean_token_count": 4, "target_token_count": 2, "chunk_tokens": 2}

        assert [allows(0, kv, **kwargs) for kv in range(6)] == [
            True,
            True,
            False,
            False,
            False,
            False,
        ]
        assert [allows(2, kv, **kwargs) for kv in range(6)] == [
            True,
            True,
            True,
            True,
            False,
            False,
        ]
        assert [allows(4, kv, **kwargs) for kv in range(6)] == [
            True,
            True,
            True,
            True,
            True,
            True,
        ]
        assert allows(-1, 0, **kwargs) is False
        assert allows(0, 6, **kwargs) is False
        with pytest.raises(ValueError, match="chunk-aligned"):
            allows(0, 0, clean_token_count=3, target_token_count=2, chunk_tokens=2)


def test_one_block_dense_vs_chunk_sparse_forward_equivalence() -> None:
    metrics = compute_chunk_sparse_equivalence_metrics()

    assert metrics["one_block_attn_max_error"] <= 1.0e-6
    assert metrics["one_block_clean_max_error"] <= 1.0e-6
    assert metrics["one_block_target_max_error"] <= 1.0e-6
    assert metrics["clean_timestep_sum"] == 0.0
    assert metrics["target_timestep_min"] == 777.0
    assert metrics["target_timestep_max"] == 777.0
    assert metrics["rope_positions_match"] == 1


def test_three_block_dense_vs_chunk_sparse_forward_equivalence() -> None:
    metrics = compute_chunk_sparse_equivalence_metrics()

    assert metrics["three_block_clean_max_error"] <= 1.0e-6
    assert metrics["three_block_target_max_error"] <= 1.0e-6
    assert metrics["parameter_count_before"] == metrics["parameter_count_after"]


def test_target_only_loss_gradient_dense_vs_chunk_sparse_equivalence() -> None:
    metrics = compute_chunk_sparse_gradient_metrics()

    assert metrics["clean_input_grad_max_error"] <= 1.0e-6
    assert metrics["target_input_grad_max_error"] <= 1.0e-6
    for name in ("q", "k", "v", "o"):
        assert metrics[f"{name}_weight_grad_max_error"] <= 1.0e-6
        assert metrics[f"{name}_bias_grad_max_error"] <= 1.0e-6
    assert metrics["overall_gradient_max_error"] <= 1.0e-6


def test_target_attention_uses_pre_block_clean_not_same_block_updated_clean() -> None:
    with _load_mcp_with_stubs() as mcp:
        mcp.causal_rope_apply = _logical_rope_apply
        mcp.attention = _scaled_attention
        case = _tiny_case_tensors(mcp, num_layers=1, seed=41)
        block = case["module"].blocks[0]
        clean_tokens = case["clean_token_count"]
        target_tokens = case["target_token_count"]
        attn_input, _, _, _ = _block_attention_input(block, case["x"], case["e"])
        correct_y = block.self_attn(
            attn_input,
            case["target_grid"],
            case["freqs"],
            case["target_start"],
            paper_fidelity_mcp1_mask=True,
            target_token_count=target_tokens,
            target_grid_sizes=case["target_grid"],
            target_start_frame=case["target_start"],
            clean_prefix_grid_sizes=case["clean_grid"],
            clean_prefix_start_frame=0,
        )
        correct_block, _ = _dense_paper_block_forward(mcp, block, case["x"], case["e"], case)
        mixed_residual = case["x"].clone()
        mixed_residual[:, :clean_tokens] = correct_block[:, :clean_tokens]
        updated_clean_attn_input, _, _, _ = _block_attention_input(
            block,
            mixed_residual,
            case["e"],
        )

        def target_y_with_clean_source(clean_attn_source: torch.Tensor) -> torch.Tensor:
            clean_target_attn_input = torch.cat(
                [clean_attn_source, attn_input[:, clean_tokens:]],
                dim=1,
            )
            batch, seq_len = clean_target_attn_input.shape[:2]
            heads = block.self_attn.num_heads
            head_dim = block.self_attn.head_dim
            q = block.self_attn.norm_q(block.self_attn.q(attn_input)).view(
                batch,
                seq_len,
                heads,
                head_dim,
            )
            k = block.self_attn.norm_k(block.self_attn.k(clean_target_attn_input)).view(
                batch,
                seq_len,
                heads,
                head_dim,
            )
            v = block.self_attn.v(clean_target_attn_input).view(batch, seq_len, heads, head_dim)
            roped_q_target = mcp.causal_rope_apply(
                q[:, clean_tokens:],
                case["target_grid"],
                case["freqs"],
                start_frame=case["target_start"],
            ).type_as(v)
            roped_k_clean = mcp.causal_rope_apply(
                k[:, :clean_tokens],
                case["clean_grid"],
                case["freqs"],
                start_frame=0,
            ).type_as(v)
            roped_k_target = mcp.causal_rope_apply(
                k[:, clean_tokens:],
                case["target_grid"],
                case["freqs"],
                start_frame=case["target_start"],
            ).type_as(v)
            y = mcp.attention(
                roped_q_target,
                torch.cat([roped_k_clean, roped_k_target], dim=1),
                torch.cat([v[:, :clean_tokens], v[:, clean_tokens:]], dim=1),
            )
            return block.self_attn.o(y.flatten(2))

        leaky_target_y = target_y_with_clean_source(updated_clean_attn_input[:, :clean_tokens])

        torch.testing.assert_close(
            correct_y[:, clean_tokens:],
            target_y_with_clean_source(attn_input[:, :clean_tokens]),
            rtol=1.0e-5,
            atol=1.0e-6,
        )
        assert _max_abs_error(correct_y[:, clean_tokens:], leaky_target_y) > 1.0e-5


def test_paper_fidelity_attention_uses_chunk_sparse_calls_and_rope_offsets() -> None:
    with _load_mcp_with_stubs() as mcp:
        module = mcp.MCPSelfAttention(dim=4, num_heads=2, qk_norm=False)
        attention_calls = []
        rope_calls = []

        def fake_attention(q, k, v):
            attention_calls.append(
                {"q_len": q.shape[1], "k_len": k.shape[1], "v_len": v.shape[1]}
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

        mcp.attention = fake_attention
        mcp.causal_rope_apply = fake_rope

        out = module(
            torch.randn(1, 6, 4),
            torch.tensor([[1, 1, 2]], dtype=torch.long),
            torch.ones(8, 2),
            start_frame=6,
            paper_fidelity_mcp1_mask=True,
            target_token_count=2,
            target_grid_sizes=torch.tensor([[1, 1, 2]], dtype=torch.long),
            target_start_frame=6,
            clean_prefix_grid_sizes=torch.tensor([[2, 1, 2]], dtype=torch.long),
            clean_prefix_start_frame=0,
        )

        assert out.shape == (1, 6, 4)
        assert attention_calls == [
            {"q_len": 2, "k_len": 2, "v_len": 2},
            {"q_len": 2, "k_len": 4, "v_len": 4},
            {"q_len": 2, "k_len": 6, "v_len": 6},
        ]
        assert rope_calls == [
            {"seq_len": 4, "frames": 2, "start_frame": 0},
            {"seq_len": 4, "frames": 2, "start_frame": 0},
            {"seq_len": 2, "frames": 1, "start_frame": 6},
            {"seq_len": 2, "frames": 1, "start_frame": 6},
        ]


def test_paper_fidelity_module_updates_clean_residuals_and_uses_clean_zero_timestep() -> None:
    with _load_mcp_with_stubs() as mcp:
        torch.manual_seed(3)
        timestep_calls = []
        original_sinusoidal = mcp.sinusoidal_embedding_1d

        def capture_sinusoidal(dim, t):
            timestep_calls.append(t.detach().cpu().clone())
            return original_sinusoidal(dim, t)

        mcp.sinusoidal_embedding_1d = capture_sinusoidal
        module = mcp.MCPModule(
            dim=4,
            ffn_dim=8,
            num_heads=2,
            num_layers=1,
            out_dim=1,
            patch_size=(1, 1, 1),
            freq_dim=4,
            qk_norm=False,
        )
        nn.init.ones_(module.head.head.weight)
        captured = {}
        original_block = module.blocks[0].forward

        def wrapped_block(x, e, *args, **kwargs):
            out = original_block(x, e, *args, **kwargs)
            captured["input"] = x.detach()
            captured["output"] = out.detach()
            captured["paper"] = bool(kwargs.get("paper_fidelity_mcp1_mask", False))
            return out

        module.blocks[0].forward = wrapped_block
        upstream = torch.randn(1, 2, 4)
        future_tokens = torch.randn(1, 2, 4)
        clean_prefix = torch.randn(1, 4, 4)
        rng_before = torch.get_rng_state().clone()

        flow, hidden = module(
            upstream=upstream,
            future_tokens=future_tokens,
            grid_sizes=torch.tensor([[1, 1, 2]], dtype=torch.long),
            start_frame=6,
            timestep=torch.full((1, 1), 777.0),
            freqs=torch.ones(8, 2),
            paper_fidelity_mcp1_mask=True,
            clean_prefix=clean_prefix,
            clean_prefix_grid_sizes=torch.tensor([[2, 1, 2]], dtype=torch.long),
            clean_prefix_start_frame=0,
            clean_prefix_timestep=torch.zeros((1, 2)),
        )

        assert torch.equal(torch.get_rng_state(), rng_before)
        assert captured["paper"] is True
        assert captured["input"].shape == (1, 6, 4)
        assert hidden.shape == (1, 2, 4)
        assert flow.shape == (1, 1, 1, 1, 2)
        assert not torch.allclose(captured["input"][:, :4], captured["output"][:, :4])
        assert [call.tolist() for call in timestep_calls] == [[777.0], [0.0, 0.0, 777.0]]


def test_paper_fidelity_stack_gradient_reaches_clean_and_noisy_taps() -> None:
    with _load_mcp_with_stubs() as mcp:
        torch.manual_seed(4)
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
            qk_norm=False,
        )
        nn.init.ones_(stack.mcp_modules[0].head.head.weight)
        clean_features = (
            torch.randn(1, 4, 4, requires_grad=True),
            torch.randn(1, 4, 4, requires_grad=True),
        )
        noisy_features = (
            torch.randn(1, 2, 4, requires_grad=True),
            torch.randn(1, 2, 4, requires_grad=True),
        )
        future = [torch.randn(1, 2, 4, requires_grad=True)]

        pred = stack(
            features=noisy_features,
            future_embeds=future,
            future_grid_sizes=[torch.tensor([[1, 1, 2]], dtype=torch.long)],
            future_start_frames=[6],
            timesteps=[torch.full((1, 1), 777.0)],
            freqs=torch.ones(8, 2),
            paper_fidelity_mcp1_mask=True,
            paper_fidelity_clean_prefix_features=clean_features,
            paper_fidelity_clean_prefix_grid_sizes=torch.tensor([[2, 1, 2]], dtype=torch.long),
            paper_fidelity_clean_prefix_start_frame=0,
            paper_fidelity_clean_prefix_timestep=torch.zeros((1, 2)),
        )[0]
        pred.sum().backward()

        assert stack.mcp_modules[0].proj.weight.grad.abs().sum().item() > 0.0
        assert stack.fusion[0].weight.grad.abs().sum().item() > 0.0
        assert noisy_features[0].grad.abs().sum().item() > 0.0
        assert clean_features[0].grad.abs().sum().item() > 0.0
        assert future[0].grad.abs().sum().item() > 0.0


def test_paper_fidelity_anchor0_parity_and_no_new_trainable_parameters() -> None:
    with _load_mcp_with_stubs() as mcp:
        torch.manual_seed(5)
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
            qk_norm=False,
        )
        nn.init.ones_(stack.mcp_modules[0].head.head.weight)
        before_keys = _parameter_key_tuple(stack)
        before_count = _parameter_count(stack)
        features = (torch.randn(1, 2, 4), torch.randn(1, 2, 4))
        future = [torch.randn(1, 2, 4)]
        grid = [torch.tensor([[1, 1, 2]], dtype=torch.long)]
        timestep = [torch.full((1, 1), 1000.0)]

        control = stack(
            features=features,
            future_embeds=future,
            future_grid_sizes=grid,
            future_start_frames=[3],
            timesteps=timestep,
            freqs=torch.ones(8, 2),
        )[0]
        treatment = stack(
            features=features,
            future_embeds=future,
            future_grid_sizes=grid,
            future_start_frames=[3],
            timesteps=timestep,
            freqs=torch.ones(8, 2),
            paper_fidelity_mcp1_mask=True,
            paper_fidelity_clean_prefix_features=(features[0][:, :0], features[1][:, :0]),
            paper_fidelity_clean_prefix_grid_sizes=torch.tensor([[0, 1, 2]], dtype=torch.long),
            paper_fidelity_clean_prefix_start_frame=0,
            paper_fidelity_clean_prefix_timestep=torch.zeros((1, 0)),
        )[0]

        torch.testing.assert_close(treatment, control, rtol=0.0, atol=0.0)
        assert _parameter_key_tuple(stack) == before_keys
        assert _parameter_count(stack) == before_count


def test_official_shared_mcp_output_head_false_keeps_independent_head_route() -> None:
    with _load_mcp_with_stubs() as mcp:
        torch.manual_seed(15)
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
            qk_norm=False,
        )
        shared_head = mcp.CausalHead(4, 1, (1, 1, 1), 1.0e-6)
        before_keys = _parameter_key_tuple(stack)
        before_count = _parameter_count(stack)
        features = (torch.randn(1, 2, 4), torch.randn(1, 2, 4))
        future = [torch.randn(1, 2, 4)]
        grid = [torch.tensor([[1, 1, 2]], dtype=torch.long)]
        timestep = [torch.full((1, 1), 1000.0)]
        kwargs = {
            "features": features,
            "future_embeds": future,
            "future_grid_sizes": grid,
            "future_start_frames": [3],
            "timesteps": timestep,
            "freqs": torch.ones(8, 2),
        }

        default = stack(**kwargs)[0]
        explicit_false = stack(
            **kwargs,
            official_shared_mcp_output_head=False,
            main_output_head=shared_head,
        )[0]
        torch.testing.assert_close(explicit_false, default, rtol=0.0, atol=0.0)

        stack.zero_grad(set_to_none=True)
        shared_head.zero_grad(set_to_none=True)
        control = stack(
            **kwargs,
            official_shared_mcp_output_head=False,
            main_output_head=shared_head,
        )[0]
        control.sum().backward()
        assert stack.mcp_modules[0].head.head.weight.grad is not None
        assert stack.mcp_modules[0].head.head.weight.grad.abs().sum().item() > 0.0
        assert shared_head.head.weight.grad is None
        assert _parameter_key_tuple(stack) == before_keys
        assert _parameter_count(stack) == before_count


def test_official_shared_mcp_output_head_gradients_and_dormant_head() -> None:
    with _load_mcp_with_stubs() as mcp:
        torch.manual_seed(16)
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
            qk_norm=False,
        )
        shared_head = mcp.CausalHead(4, 1, (1, 1, 1), 1.0e-6)
        before_keys = _parameter_key_tuple(stack)
        before_count = _parameter_count(stack)
        shared_head_weight_id = id(shared_head.head.weight)
        features = (
            torch.randn(1, 2, 4, requires_grad=True),
            torch.randn(1, 2, 4, requires_grad=True),
        )
        future = [torch.randn(1, 2, 4, requires_grad=True)]

        pred = stack(
            features=features,
            future_embeds=future,
            future_grid_sizes=[torch.tensor([[1, 1, 2]], dtype=torch.long)],
            future_start_frames=[3],
            timesteps=[torch.full((1, 1), 1000.0)],
            freqs=torch.ones(8, 2),
            official_shared_mcp_output_head=True,
            main_output_head=shared_head,
        )[0]
        pred.square().mean().backward()

        assert id(shared_head.head.weight) == shared_head_weight_id
        assert shared_head.head.weight.grad is not None
        assert shared_head.head.weight.grad.abs().sum().item() > 0.0
        assert _grad_l1(stack.mcp_modules[0].blocks[0].parameters()) > 0.0
        assert stack.fusion[0].weight.grad.abs().sum().item() > 0.0
        assert stack.mcp_modules[0].proj.weight.grad.abs().sum().item() > 0.0
        assert _grad_l1(stack.mcp_modules[0].head.parameters()) == 0.0
        assert _parameter_key_tuple(stack) == before_keys
        assert _parameter_count(stack) == before_count


def test_paper_fidelity_stack_routes_depth1_only_and_rejects_direct_clean_mix() -> None:
    with _load_mcp_with_stubs() as mcp:
        torch.manual_seed(6)
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
        records = []
        for index, module in enumerate(stack.mcp_modules):
            original = module.forward

            def wrapped_forward(*args, _index=index, _original=original, **kwargs):
                clean_prefix = kwargs.get("clean_prefix")
                records.append(
                    {
                        "depth": _index + 1,
                        "paper": bool(kwargs.get("paper_fidelity_mcp1_mask", False)),
                        "direct": bool(kwargs.get("direct_clean_context_kv", False)),
                        "clean_tokens": 0 if clean_prefix is None else clean_prefix.shape[1],
                    }
                )
                return _original(*args, **kwargs)

            module.forward = wrapped_forward

        features = (torch.randn(1, 2, 4), torch.randn(1, 2, 4))
        clean_features = (torch.randn(1, 4, 4), torch.randn(1, 4, 4))
        kwargs = dict(
            features=features,
            future_embeds=[torch.randn(1, 2, 4) for _ in range(3)],
            future_grid_sizes=[torch.tensor([[1, 1, 2]], dtype=torch.long) for _ in range(3)],
            future_start_frames=[6, 9, 12],
            timesteps=[torch.full((1, 1), 1000.0) for _ in range(3)],
            freqs=torch.ones(8, 2),
            paper_fidelity_mcp1_mask=True,
            paper_fidelity_clean_prefix_features=clean_features,
            paper_fidelity_clean_prefix_grid_sizes=torch.tensor([[2, 1, 2]], dtype=torch.long),
            paper_fidelity_clean_prefix_start_frame=0,
            paper_fidelity_clean_prefix_timestep=torch.zeros((1, 2)),
        )

        stack(**kwargs)

        assert records == [
            {"depth": 1, "paper": True, "direct": False, "clean_tokens": 4},
            {"depth": 2, "paper": False, "direct": False, "clean_tokens": 0},
            {"depth": 3, "paper": False, "direct": False, "clean_tokens": 0},
        ]
        with pytest.raises(ValueError, match="mutually exclusive"):
            stack(**{**kwargs, "direct_clean_context_kv": True})


def test_wrapper_paper_fidelity_full_sequence_anchor_shapes_and_metadata() -> None:
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
            self.calls = []

        def forward(self, x, **kwargs):
            self.calls.append(kwargs)
            batch = x.shape[0]
            features = tuple(torch.full((batch, 32760, 2), float(i)) for i in range(4))
            aux = {
                "features": features,
                "mcp_embeds": tuple(
                    torch.zeros((batch, 4680, 2)) for _ in kwargs["mcp_patch_inputs"]
                ),
                "mcp_grid_sizes": tuple(
                    torch.tensor([[3, 60, 26]], dtype=torch.long)
                    for _ in kwargs["mcp_patch_inputs"]
                ),
            }
            if kwargs.get("return_feature_halves", False):
                aux["clean_features"] = tuple(
                    torch.full((batch, 32760, 2), 10.0 + float(i)) for i in range(4)
                )
                aux["noisy_features"] = features
            return torch.zeros((batch, 1, 21, 1, 1)), aux

    class FakeMCP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = []

        def forward(self, **kwargs):
            clean_prefix = kwargs.get("paper_fidelity_clean_prefix_features")
            clean_tokens = 0 if clean_prefix is None else int(clean_prefix[0].shape[1])
            clean_grid = kwargs.get("paper_fidelity_clean_prefix_grid_sizes")
            clean_timestep = kwargs.get("paper_fidelity_clean_prefix_timestep")
            feature_tokens = int(kwargs["features"][0].shape[1])
            self.calls.append(
                {
                    "paper": bool(kwargs.get("paper_fidelity_mcp1_mask", False)),
                    "direct": bool(kwargs.get("direct_clean_context_kv", False)),
                    "clean_tokens": clean_tokens,
                    "feature_tokens": feature_tokens,
                    "mcp_sequence_total": clean_tokens + feature_tokens,
                    "clean_grid_frames": None
                    if clean_grid is None
                    else int(clean_grid[0, 0].item()),
                    "clean_timestep_shape": None
                    if clean_timestep is None
                    else tuple(clean_timestep.shape),
                    "clean_timestep_sum": None
                    if clean_timestep is None
                    else float(clean_timestep.sum().item()),
                    "future_start_frames": tuple(kwargs["future_start_frames"]),
                }
            )
            batch = kwargs["features"][0].shape[0]
            return [
                torch.zeros((batch, 1, 3, 1, 1))
                for _ in kwargs["future_embeds"]
            ]

    def make_wrapper() -> WanDiffusionWrapper:
        wrapper = WanDiffusionWrapper.__new__(WanDiffusionWrapper)
        nn.Module.__init__(wrapper)
        wrapper.model = FakeBackbone()
        wrapper.mcp = FakeMCP()
        wrapper.mcp_tap_layers = (3, 11, 19, 29)
        wrapper.uniform_timestep = False
        wrapper.seq_len = 32760
        return wrapper

    batch = _loss_batch()
    anchors = build_full_sequence_mcp_anchor_inputs(batch)
    wrapper = make_wrapper()
    outputs = wrapper.forward_full_sequence_next_forcing(
        noisy_image_or_video=batch.noisy_main,
        clean_x=batch.clean_target,
        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 2))},
        timestep_main=batch.timestep_main,
        mcp_anchor_inputs=anchors,
        paper_fidelity_mcp1_mask=True,
    )

    chunk_tokens = FULL_SEQUENCE_CHUNK_FRAMES * FULL_SEQUENCE_FRAME_SEQ_LENGTH
    assert len(wrapper.model.calls) == 1
    assert wrapper.model.calls[0]["return_feature_halves"] is True
    assert len(wrapper.model.calls[0]["mcp_patch_inputs"]) == 15
    assert outputs.paper_exact_reproduction is False
    assert outputs.paper_fidelity_mcp1_mask is True
    assert outputs.mcp_path_kind == "paper_fidelity_clean_residual_mask"
    assert outputs.main_backbone_forward_count == 1
    assert [tuple(pred.shape) for pred in outputs.mcp_flow_preds_by_depth] == [
        (1, 6, 3, 1, 1, 1),
        (1, 5, 3, 1, 1, 1),
        (1, 4, 3, 1, 1, 1),
    ]
    assert [call["mcp_sequence_total"] for call in wrapper.mcp.calls] == [
        chunk_tokens,
        2 * chunk_tokens,
        3 * chunk_tokens,
        4 * chunk_tokens,
        5 * chunk_tokens,
        6 * chunk_tokens,
    ]
    for anchor_index, call in enumerate(wrapper.mcp.calls):
        clean_frames = anchor_index * FULL_SEQUENCE_CHUNK_FRAMES
        assert call["paper"] is True
        assert call["direct"] is False
        assert call["feature_tokens"] == chunk_tokens
        assert call["clean_tokens"] == anchor_index * chunk_tokens
        assert call["clean_grid_frames"] == clean_frames
        assert call["clean_timestep_shape"] == (1, clean_frames)
        assert call["clean_timestep_sum"] == 0.0
        assert call["future_start_frames"][0] == (anchor_index + 1) * FULL_SEQUENCE_CHUNK_FRAMES
    assert wrapper.mcp.calls[0]["future_start_frames"] == (3, 6, 9)
    assert wrapper.mcp.calls[5]["future_start_frames"] == (18,)

    canonical_wrapper = make_wrapper()
    canonical_outputs = canonical_wrapper.forward_full_sequence_next_forcing(
        noisy_image_or_video=batch.noisy_main,
        clean_x=batch.clean_target,
        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 2))},
        timestep_main=batch.timestep_main,
        mcp_anchor_inputs=anchors,
    )
    assert "return_feature_halves" not in canonical_wrapper.model.calls[0]
    assert canonical_outputs.paper_exact_reproduction is False
    assert canonical_outputs.paper_fidelity_mcp1_mask is False
    assert canonical_outputs.mcp_path_kind == "canonical_target_only"
    assert all(call["paper"] is False for call in canonical_wrapper.mcp.calls)
    assert all(call["direct"] is False for call in canonical_wrapper.mcp.calls)

    direct_wrapper = make_wrapper()
    direct_outputs = direct_wrapper.forward_full_sequence_next_forcing(
        noisy_image_or_video=batch.noisy_main,
        clean_x=batch.clean_target,
        conditional_dict={"prompt_embeds": torch.zeros((1, 1, 2))},
        timestep_main=batch.timestep_main,
        mcp_anchor_inputs=anchors,
        direct_clean_context_kv=True,
    )
    assert direct_wrapper.model.calls[0]["return_feature_halves"] is True
    assert direct_outputs.paper_exact_reproduction is False
    assert direct_outputs.paper_fidelity_mcp1_mask is False
    assert direct_outputs.mcp_path_kind == "direct_clean_static_kv"
    assert direct_wrapper.mcp.calls[0]["direct"] is False
    assert direct_wrapper.mcp.calls[1]["direct"] is True
    assert all(call["paper"] is False for call in direct_wrapper.mcp.calls)


def test_main_clean_zero_time_source_contract_is_still_present() -> None:
    text = (ROOT / "wan" / "modules" / "causal_model.py").read_text(encoding="utf-8")
    aug_default = text.index("if aug_t is None:")
    zero_assign = text.index("aug_t = torch.zeros_like(t)", aug_default)
    clean_embedding = text.index("e0_clean", zero_assign)
    concat_embedding = text.index("torch.cat([e0_clean, e0]", clean_embedding)

    assert aug_default < zero_assign < clean_embedding < concat_embedding
