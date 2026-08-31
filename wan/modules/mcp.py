"""Multi-Chunk Prediction (MCP) heads.

Next-Forcing style auxiliary heads that predict FUTURE video chunks from the
backbone's intermediate features, giving the causal backbone dense long-range
temporal supervision instead of next-chunk-only ("myopic") supervision.

Reference: "Next Forcing: Causal World Modeling with Multi-Chunk Prediction"
(arXiv 2606.11187). No official code was released; this is an adaptation of the
method described in the paper's architecture figure to this repo's Self-Forcing
rollout, with two deliberate differences documented in `MCPStack`.

Layout conventions inherited from `wan/modules/causal_model.py`:
  * hidden states are a flat token sequence [B, L, dim] with L = F * frame_seqlen
  * frame_seqlen = (H // 2) * (W // 2) for the default patch_size (1, 2, 2)
  * RoPE is applied at ABSOLUTE frame positions via `causal_rope_apply`, so a
    future chunk must be roped at its true start frame, not at 0.
"""

import math

import torch
import torch.nn as nn

from wan.modules.attention import attention
from wan.modules.causal_model import CausalHead, causal_rope_apply
from wan.modules.model import (
    WanLayerNorm,
    WanRMSNorm,
    sinusoidal_embedding_1d,
)

__all__ = [
    "MCPStack",
    "mcp_unpatchify",
    "MCP_INPUT_TIMESTEP",
    "paper_fidelity_mcp1_mask_allows",
]

# MCP always consumes the future chunk at sigma = 1 (pure noise).
# FlowMatchScheduler builds timesteps = sigmas * num_train_timesteps with
# sigmas[0] = shift * 1.0 / (1 + (shift - 1) * 1.0) = 1.0 for any shift, so
# sigma = 1 <=> timestep = 1000 regardless of the configured timestep_shift.
# At sigma = 1: x_t = (1 - sigma) * x0 + sigma * noise = noise, and the flow
# target v* = (x_t - x0) / sigma reduces exactly to `noise - x0`, i.e. the
# scheduler's own `training_target` (utils/scheduler.py:179). No sigma lookup,
# no division, no argmin snapping.
MCP_INPUT_TIMESTEP = 1000


def paper_fidelity_mcp1_mask_allows(
    q_idx: int,
    kv_idx: int,
    *,
    clean_token_count: int,
    target_token_count: int,
    chunk_tokens: int,
) -> bool:
    """Appendix-A clean/noisy block mask for one anchor-wise MCP1 sequence."""
    q = int(q_idx)
    kv = int(kv_idx)
    clean_tokens = int(clean_token_count)
    target_tokens = int(target_token_count)
    chunk = int(chunk_tokens)
    total = clean_tokens + target_tokens
    if q < 0 or kv < 0 or q >= total or kv >= total:
        return False
    if chunk <= 0 or target_tokens <= 0:
        raise ValueError("target_token_count and chunk_tokens must be positive")
    if clean_tokens % chunk != 0 or target_tokens != chunk:
        raise ValueError("paper-fidelity MCP1 expects chunk-aligned clean/target tokens")

    if q < clean_tokens:
        # Clean rows follow chunk-causal clean-only attention.
        return kv < clean_tokens and (kv // chunk) <= (q // chunk)
    # Target rows attend every clean-prefix token plus the same target chunk.
    return kv < clean_tokens or clean_tokens <= kv < total


def mcp_unpatchify(x, grid_sizes, patch_size, out_dim):
    """Mirror of `CausalWanModel.unpatchify` (causal_model.py:1011-1034).

    Duplicated rather than imported so MCP stays independent of the backbone
    instance. `x` is the 4-D CausalHead output [B, F, frame_seqlen, prod(patch_size)*out_dim];
    the leading `u[:math.prod(v)]` slice is a no-op for that layout, exactly as
    in the backbone.
    """
    c = out_dim
    out = []
    for u, v in zip(x, grid_sizes.tolist()):
        u = u[: math.prod(v)].view(*v, *patch_size, c)
        u = torch.einsum("fhwpqrc->cfphqwr", u)
        u = u.reshape(c, *[i * j for i, j in zip(v, patch_size)])
        out.append(u)
    return torch.stack(out)


class MCPSelfAttention(nn.Module):
    """Self-attention over the tokens of a single future chunk.

    This matches the paper's mask (Appendix A): "Noisy -> Noisy: Noisy tokens only
    attend to other noisy tokens within the same chunk (self-attention within the
    chunk being denoised)."

    The canonical/default path still does not grant Noisy -> Clean direct K/V:
    history reaches the head only through fused backbone features. The experimental
    `direct_clean_context_kv` ablation can opt in to target-query-only direct
    preceding-clean K/V for MCP depth1. That path tests the clean-context bottleneck
    without implementing the paper's full shared clean+noisy MCP mask.

    RoPE uses the chunk's true absolute start frame, i.e. the paper's
    RoPE(x_0^[k][i]) = RoPE(i + k) (Eq. 6).
    """

    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        grid_sizes,
        freqs,
        start_frame,
        *,
        clean_context=None,
        clean_context_grid_sizes=None,
        clean_context_start_frame=0,
        paper_fidelity_mcp1_mask=False,
        target_token_count=None,
        target_grid_sizes=None,
        target_start_frame=None,
        clean_prefix_grid_sizes=None,
        clean_prefix_start_frame=0,
    ):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        if paper_fidelity_mcp1_mask:
            return self._forward_paper_fidelity_mcp1(
                q,
                k,
                v,
                freqs,
                target_token_count=target_token_count,
                target_grid_sizes=target_grid_sizes,
                target_start_frame=target_start_frame,
                clean_prefix_grid_sizes=clean_prefix_grid_sizes,
                clean_prefix_start_frame=clean_prefix_start_frame,
            )

        roped_q = causal_rope_apply(q, grid_sizes, freqs, start_frame=start_frame).type_as(v)
        roped_k = causal_rope_apply(k, grid_sizes, freqs, start_frame=start_frame).type_as(v)

        if clean_context is not None and clean_context.shape[1] > 0:
            if clean_context_grid_sizes is None:
                raise ValueError("clean_context_grid_sizes is required with clean_context")
            if clean_context.shape[0] != b or clean_context.shape[2] != self.dim:
                raise ValueError("clean_context must have shape [B, L_clean, dim]")
            clean_s = clean_context.shape[1]
            clean_k = self.norm_k(self.k(clean_context)).view(b, clean_s, n, d)
            clean_v = self.v(clean_context).view(b, clean_s, n, d)
            roped_clean_k = causal_rope_apply(
                clean_k,
                clean_context_grid_sizes,
                freqs,
                start_frame=clean_context_start_frame,
            ).type_as(v)
            roped_k = torch.cat([roped_clean_k, roped_k], dim=1)
            v = torch.cat([clean_v, v], dim=1)

        x = attention(roped_q, roped_k, v)
        return self.o(x.flatten(2))

    def _forward_paper_fidelity_mcp1(
        self,
        q,
        k,
        v,
        freqs,
        *,
        target_token_count,
        target_grid_sizes,
        target_start_frame,
        clean_prefix_grid_sizes,
        clean_prefix_start_frame,
    ):
        if target_token_count is None or target_grid_sizes is None:
            raise ValueError("paper-fidelity MCP1 requires target token metadata")
        if target_start_frame is None:
            raise ValueError("paper-fidelity MCP1 requires target start-frame metadata")
        if clean_prefix_grid_sizes is None:
            raise ValueError("paper-fidelity MCP1 requires clean prefix grid metadata")
        target_tokens = int(target_token_count)
        clean_tokens = int(q.shape[1]) - target_tokens
        if clean_tokens < 0:
            raise ValueError("target_token_count exceeds sequence length")
        if target_tokens <= 0:
            raise ValueError("target_token_count must be positive")
        if clean_tokens % target_tokens != 0:
            raise ValueError("paper-fidelity MCP1 requires chunk-aligned clean prefix")

        q_clean, q_target = q[:, :clean_tokens], q[:, clean_tokens:]
        k_clean, k_target = k[:, :clean_tokens], k[:, clean_tokens:]
        v_clean, v_target = v[:, :clean_tokens], v[:, clean_tokens:]

        outputs = []
        if clean_tokens > 0:
            roped_q_clean = causal_rope_apply(
                q_clean,
                clean_prefix_grid_sizes,
                freqs,
                start_frame=clean_prefix_start_frame,
            ).type_as(v)
            roped_k_clean = causal_rope_apply(
                k_clean,
                clean_prefix_grid_sizes,
                freqs,
                start_frame=clean_prefix_start_frame,
            ).type_as(v)
            for clean_stop in range(target_tokens, clean_tokens + 1, target_tokens):
                outputs.append(
                    attention(
                        roped_q_clean[:, clean_stop - target_tokens:clean_stop],
                        roped_k_clean[:, :clean_stop],
                        v_clean[:, :clean_stop],
                    )
                )
        else:
            roped_k_clean = None

        roped_q_target = causal_rope_apply(
            q_target,
            target_grid_sizes,
            freqs,
            start_frame=int(target_start_frame),
        ).type_as(v)
        roped_k_target = causal_rope_apply(
            k_target,
            target_grid_sizes,
            freqs,
            start_frame=int(target_start_frame),
        ).type_as(v)
        if clean_tokens > 0:
            target_k = torch.cat([roped_k_clean, roped_k_target], dim=1)
            target_v = torch.cat([v_clean, v_target], dim=1)
        else:
            target_k = roped_k_target
            target_v = v_target
        outputs.append(attention(roped_q_target, target_k, target_v))
        return self.o(torch.cat(outputs, dim=1).flatten(2))


class MCPBlock(nn.Module):
    """A lightweight transformer block: self-attention + FFN, no cross-attention.

    The paper's MCP module is only "the fusion MLP, projection layers, and
    lightweight transformer blocks" (Sec. 4.6, Zero-Overhead Mode) and never
    describes cross-attention inside it. Conditioning reaches the head through
    h_fuse instead -- the backbone's own layers already cross-attended to it, and
    Sec. 4.4 says improvements propagate to the other stream "indirectly" via the
    main model's attention, not the MCP's.

    Dropping cross-attention also makes the paper's weight init possible: what
    remains (self_attn.{q,k,v,o,norm_q,norm_k}, ffn, modulation) is parameter-shape
    identical to CausalWanAttentionBlock (causal_model.py:243-335), so
    `MCPStack.init_from_backbone` can copy the main model's last layers straight in.
    Ablation (Table 2): 83.8 without weight init vs 85.8 with.
    """

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        qk_norm=True,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = MCPSelfAttention(dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        grid_sizes,
        freqs,
        start_frame,
        *,
        clean_context=None,
        clean_context_grid_sizes=None,
        clean_context_start_frame=0,
        paper_fidelity_mcp1_mask=False,
        target_token_count=None,
        target_grid_sizes=None,
        target_start_frame=None,
        clean_prefix_grid_sizes=None,
        clean_prefix_start_frame=0,
    ):
        r"""
        Args:
            x (Tensor): [B, L, C]
            e (Tensor): [B, F, 6, C]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        attn_input = (
            self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]
        ).flatten(1, 2)
        clean_attn_context = None if clean_context is None else self.norm1(clean_context)
        y = self.self_attn(
            attn_input,
            grid_sizes,
            freqs,
            start_frame,
            clean_context=clean_attn_context,
            clean_context_grid_sizes=clean_context_grid_sizes,
            clean_context_start_frame=clean_context_start_frame,
            paper_fidelity_mcp1_mask=paper_fidelity_mcp1_mask,
            target_token_count=target_token_count,
            target_grid_sizes=target_grid_sizes,
            target_start_frame=target_start_frame,
            clean_prefix_grid_sizes=clean_prefix_grid_sizes,
            clean_prefix_start_frame=clean_prefix_start_frame,
        )
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        y = self.ffn(
            (self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
        )
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
        return x


class MCPModule(nn.Module):
    """One MCP module: predicts the video chunk `k` blocks ahead of the current one.

    Data flow (matching the paper's figure):
        Embed(noisy future chunk) -> Concat(upstream feature) -> Linear Proj
        -> N transformer blocks -> CausalHead -> flow prediction
    """

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        num_layers,
        out_dim,
        patch_size,
        freq_dim,
        qk_norm=True,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.patch_size = patch_size
        self.out_dim = out_dim

        # No patch_embedding of our own: the paper embeds the noisy shifted target
        # "through the shared patch embedding layer" (Sec. 4.3), so the backbone's is
        # passed into forward(). Registering it here as well would duplicate the
        # parameter in the state_dict and hand it to the optimizer twice.

        # Eq. 8: z^[k] = W_k [h_prev^[k-1] ; Embed(x_tk^[k])], W_k in R^{d x 2d}.
        # The bracket is a CHANNEL concat -- token count stays L on both sides.
        self.proj = nn.Linear(dim * 2, dim)

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.blocks = nn.ModuleList(
            [MCPBlock(dim, ffn_dim, num_heads, qk_norm, eps) for _ in range(num_layers)]
        )
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        self.init_weights()

    def init_weights(self):
        """Fallback init. The blocks are normally OVERWRITTEN by
        MCPStack.init_from_backbone (the paper's "initialized from the last few
        layers of the main model"); this only covers proj / time_* / head, which
        have no backbone counterpart.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        # Zero-init the output projection, mirroring the backbone's head init
        # (causal_model.py:1058). The paper does not mention this; it is a
        # deliberate addition. The backbone's head is trained for the backbone's
        # final residual stream, which is not what the MCP blocks emit, so copying
        # it would be meaningless -- and a random head would blast the carefully
        # distilled backbone with a large MCP gradient on step 0. dL/dW is non-zero
        # even at W=0, so the head leaves zero after the first update.
        # (bias is already zeroed by the generic Linear loop above.)
        nn.init.zeros_(self.head.head.weight)

    def forward(
        self,
        upstream,
        future_tokens,
        grid_sizes,
        start_frame,
        timestep,
        freqs,
        *,
        direct_clean_context_kv=False,
        clean_context=None,
        clean_context_grid_sizes=None,
        clean_context_start_frame=0,
        paper_fidelity_mcp1_mask=False,
        clean_prefix=None,
        clean_prefix_grid_sizes=None,
        clean_prefix_start_frame=0,
        clean_prefix_timestep=None,
        official_shared_mcp_output_head=False,
        main_output_head=None,
    ):
        """
        Args:
            upstream (Tensor): [B, L, dim] fused backbone feature (module 1) or the
                previous module's transformer output (modules 2..K).
            future_tokens (Tensor): [B, L, dim] the target chunk's noisy latent,
                ALREADY embedded by the backbone's shared patch_embedding -- inside
                the backbone's own forward (causal_model.py), because that is the
                only place where the shared conv's parameters are guaranteed
                gathered and mixed-precision cast under FSDP. The sharing itself
                (Sec. 4.3) is unchanged: gradient still flows into the backbone's
                patch_embedding through these tokens.
            grid_sizes (Tensor): [B, 3] patchified (F, H, W) of the target chunk,
                for RoPE and unpatchify.
            start_frame (int): absolute frame index of the target chunk (for RoPE).
            timestep (Tensor): [B, F] diffusion timestep of the target chunk.
        Returns:
            flow_pred (Tensor): [B, C_out, F, H, W] flow-matching prediction.
            hidden (Tensor): [B, L, dim] this module's transformer output, to be
                chained into the next module.
        """
        if direct_clean_context_kv and paper_fidelity_mcp1_mask:
            raise ValueError("direct_clean_context_kv and paper_fidelity_mcp1_mask are mutually exclusive")

        x = future_tokens

        if x.shape[1] != upstream.shape[1]:
            raise ValueError(
                f"MCP token-count mismatch: future chunk has {x.shape[1]} tokens but the "
                f"upstream feature has {upstream.shape[1]}. MCP requires every chunk to have "
                f"the same frame count (num_frame_per_block)."
            )

        # Concat + Linear Proj
        target_token_count = x.shape[1]
        x = self.proj(torch.cat([x, upstream], dim=-1))
        use_paper_fidelity = bool(
            paper_fidelity_mcp1_mask
            and clean_prefix is not None
            and clean_prefix.shape[1] > 0
        )
        if use_paper_fidelity:
            if clean_prefix.shape[0] != x.shape[0] or clean_prefix.shape[2] != self.dim:
                raise ValueError("clean_prefix must have shape [B, L_clean, dim]")
            if clean_prefix_grid_sizes is None:
                raise ValueError("clean_prefix_grid_sizes is required for paper-fidelity MCP1")
            clean_frames = int(clean_prefix_grid_sizes[0, 0].item())
            if clean_frames <= 0:
                raise ValueError("paper-fidelity clean prefix must contain at least one frame")
            if clean_prefix_timestep is None:
                clean_prefix_timestep = torch.zeros(
                    (x.shape[0], clean_frames),
                    device=timestep.device,
                    dtype=timestep.dtype,
                )
            if tuple(clean_prefix_timestep.shape) != (x.shape[0], clean_frames):
                raise ValueError("clean_prefix_timestep must have shape [B, F_clean]")
            sequence_timestep = torch.cat(
                [
                    clean_prefix_timestep.to(device=timestep.device, dtype=timestep.dtype),
                    timestep,
                ],
                dim=1,
            )
            # PAPER_UNDERSPECIFIED: Eq. 8 does not define clean-row z initialization.
            # The paper-fidelity path uses fused clean hidden states as clean residuals.
            x = torch.cat([clean_prefix.to(device=x.device, dtype=x.dtype), x], dim=1)
        else:
            sequence_timestep = timestep

        # Time modulation for the target chunk
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x)
        )
        if use_paper_fidelity:
            e_sequence = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, sequence_timestep.flatten()).type_as(x)
            )
        else:
            e_sequence = e
        e0 = (
            self.time_projection(e_sequence)
            .unflatten(1, (6, self.dim))
            .unflatten(dim=0, sizes=sequence_timestep.shape)
        )

        block_clean_context = clean_context if direct_clean_context_kv else None
        for block in self.blocks:
            x = block(
                x,
                e0,
                grid_sizes,
                freqs,
                start_frame,
                clean_context=block_clean_context,
                clean_context_grid_sizes=clean_context_grid_sizes,
                clean_context_start_frame=clean_context_start_frame,
                paper_fidelity_mcp1_mask=use_paper_fidelity,
                target_token_count=target_token_count,
                target_grid_sizes=grid_sizes,
                target_start_frame=start_frame,
                clean_prefix_grid_sizes=clean_prefix_grid_sizes,
                clean_prefix_start_frame=clean_prefix_start_frame,
            )
        hidden = x[:, -target_token_count:, :] if use_paper_fidelity else x

        head_e = e.unflatten(dim=0, sizes=timestep.shape).unsqueeze(2)
        if official_shared_mcp_output_head:
            if main_output_head is None:
                raise ValueError(
                    "main_output_head is required for official shared MCP output head"
                )
            out = main_output_head(hidden, head_e)
        else:
            out = self.head(hidden, head_e)
        flow_pred = mcp_unpatchify(out, grid_sizes, self.patch_size, self.out_dim)
        return flow_pred, hidden


class MCPStack(nn.Module):
    """Multi-layer feature fusion + a chain of MCP modules.

    Two deliberate departures from the paper, both forced by this repo being a
    data-free DMD distillation rather than teacher-forced supervised training:

    1. The heads are TRAINED the way the backbone is trained here, not the way the
       paper trains its heads: coupled one-step x0 regression onto the ODE
       trajectory endpoint in the ODE-init stage (model/ode_regression.py), then
       distribution matching in the DMD stage, where the rollout itself simulates
       the MCP-accelerated inference and the drafts are part of the emitted video
       scored by the unmodified DMD loss (pipeline/self_forcing_training.py
       _inference_with_trajectory_mcp_accelerated). There is no ground-truth video
       to run the paper's Eq. 5/12 against, and a one-shot drafting head trained
       by uncoupled regression collapses to the conditional mean of the future.
    2. During the DMD rollout the future chunk is always fed at sigma=1 (pure
       noise, timestep=1000), because at that moment it genuinely IS untouched
       noise. This also matches the MCP-accelerated inference mode, where the head
       must draft the next chunk from noise while the backbone denoises the
       current one; the head's one-step draft there is x0 = eps - v.
    """

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        out_dim,
        patch_size,
        freq_dim,
        num_modules=1,
        num_layers=3,
        tap_layers=(3, 11, 19, 29),
        qk_norm=True,
        eps=1e-6,
    ):
        super().__init__()
        self.tap_layers = tuple(tap_layers)
        self.num_modules = num_modules
        self.num_layers = num_layers

        # Eq. 7: h_fuse = MLP([h_4; h_12; h_20; h_30]) -- "concatenated along the
        # feature dimension and compressed through a two-layer MLP".
        self.fusion = nn.Sequential(
            nn.Linear(dim * len(self.tap_layers), dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )
        for m in self.fusion.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.mcp_modules = nn.ModuleList(
            [
                MCPModule(
                    dim=dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_heads,
                    num_layers=num_layers,
                    out_dim=out_dim,
                    patch_size=patch_size,
                    freq_dim=freq_dim,
                    qk_norm=qk_norm,
                    eps=eps,
                )
                for _ in range(num_modules)
            ]
        )

    @torch.no_grad()
    def init_from_backbone(self, blocks):
        """Initialize every MCP block from the main model's LAST layers.

        Sec. 5.1: "MCP module weights are initialized from the last few layers of the
        main model." Table 2 measures the cost of skipping this: 83.8 vs 85.8.

        Possible only because MCPBlock carries no cross-attention, which leaves it
        parameter-shape identical to CausalWanAttentionBlock. Every MCP depth is
        seeded from the same last-N backbone layers; the paper does not say whether
        depths should differ, and they diverge immediately anyway.

        `proj`, `time_*`, `fusion` and `head` have no backbone counterpart and keep
        their init_weights() values.
        """
        n = self.num_layers
        if len(blocks) < n:
            raise ValueError(
                f"backbone has {len(blocks)} layers, cannot seed {n} MCP layers from its tail"
            )
        source = blocks[len(blocks) - n:]
        for module in self.mcp_modules:
            for dst, src in zip(module.blocks, source):
                for name in ("q", "k", "v", "o"):
                    getattr(dst.self_attn, name).weight.copy_(getattr(src.self_attn, name).weight)
                    getattr(dst.self_attn, name).bias.copy_(getattr(src.self_attn, name).bias)
                for name in ("norm_q", "norm_k"):
                    d, s = getattr(dst.self_attn, name), getattr(src.self_attn, name)
                    if hasattr(d, "weight") and hasattr(s, "weight"):
                        d.weight.copy_(s.weight)
                for i in (0, 2):  # ffn: Linear, GELU, Linear
                    dst.ffn[i].weight.copy_(src.ffn[i].weight)
                    dst.ffn[i].bias.copy_(src.ffn[i].bias)
                dst.modulation.copy_(src.modulation)
        return self

    def forward(
        self,
        features,
        future_embeds,
        future_grid_sizes,
        future_start_frames,
        timesteps,
        freqs,
        *,
        direct_clean_context_kv=False,
        clean_context_features=None,
        clean_context_grid_sizes=None,
        clean_context_start_frame=0,
        paper_fidelity_mcp1_mask=False,
        paper_fidelity_clean_prefix_features=None,
        paper_fidelity_clean_prefix_grid_sizes=None,
        paper_fidelity_clean_prefix_start_frame=0,
        paper_fidelity_clean_prefix_timestep=None,
        official_shared_mcp_output_head=False,
        main_output_head=None,
    ):
        """
        Args:
            features (List[Tensor]): tapped backbone hidden states, each [B, L, dim],
                in the order of `self.tap_layers`.
            future_embeds (List[Optional[Tensor]]): per-module target chunk tokens
                [B, L, dim], embedded by the backbone's shared patch_embedding
                inside the backbone's own forward (causal_model.py). A `None` entry
                (target chunk runs past the end of the video) stops the chain there
                -- the paper instead pads by replicating the last chunk and excludes
                it from the loss (Eq. 4, 12), which comes to the same thing.
            future_grid_sizes (List[Optional[Tensor]]): per-module patchified
                (F, H, W) of the target chunk.
            future_start_frames (List[int]): absolute start frame of each target chunk.
            timesteps (List[Tensor]): per-module [B, F] timestep of the noisy chunk.
        Returns:
            List[Tensor]: flow predictions [B, C_out, F, H, W], one per module that ran.
        """
        if len(features) != len(self.tap_layers):
            raise ValueError(
                f"MCPStack expected {len(self.tap_layers)} tapped features, got {len(features)}"
            )
        if direct_clean_context_kv and paper_fidelity_mcp1_mask:
            raise ValueError("direct_clean_context_kv and paper_fidelity_mcp1_mask are mutually exclusive")

        # h_prev^[0] = h_fuse  (Eq. 8)
        upstream = self.fusion(torch.cat(features, dim=-1))
        clean_context = None
        if direct_clean_context_kv and clean_context_features is not None:
            if len(clean_context_features) != len(self.tap_layers):
                raise ValueError(
                    f"MCPStack expected {len(self.tap_layers)} clean context features, "
                    f"got {len(clean_context_features)}"
                )
            if clean_context_features and clean_context_features[0].shape[1] > 0:
                clean_context = self.fusion(torch.cat(clean_context_features, dim=-1))
        clean_prefix = None
        if paper_fidelity_mcp1_mask and paper_fidelity_clean_prefix_features is not None:
            if len(paper_fidelity_clean_prefix_features) != len(self.tap_layers):
                raise ValueError(
                    f"MCPStack expected {len(self.tap_layers)} paper-fidelity clean "
                    f"prefix features, got {len(paper_fidelity_clean_prefix_features)}"
                )
            if (
                paper_fidelity_clean_prefix_features
                and paper_fidelity_clean_prefix_features[0].shape[1] > 0
            ):
                clean_prefix = self.fusion(torch.cat(paper_fidelity_clean_prefix_features, dim=-1))

        flow_preds = []
        for k, module in enumerate(self.mcp_modules):
            if k >= len(future_embeds) or future_embeds[k] is None:
                break
            flow_pred, upstream = module(
                upstream=upstream,
                future_tokens=future_embeds[k],
                grid_sizes=future_grid_sizes[k],
                start_frame=future_start_frames[k],
                timestep=timesteps[k],
                freqs=freqs,
                direct_clean_context_kv=bool(direct_clean_context_kv and k == 0 and clean_context is not None),
                clean_context=clean_context if k == 0 else None,
                clean_context_grid_sizes=clean_context_grid_sizes if k == 0 else None,
                clean_context_start_frame=clean_context_start_frame,
                paper_fidelity_mcp1_mask=bool(paper_fidelity_mcp1_mask and k == 0),
                clean_prefix=clean_prefix if k == 0 else None,
                clean_prefix_grid_sizes=paper_fidelity_clean_prefix_grid_sizes if k == 0 else None,
                clean_prefix_start_frame=paper_fidelity_clean_prefix_start_frame,
                clean_prefix_timestep=paper_fidelity_clean_prefix_timestep if k == 0 else None,
                official_shared_mcp_output_head=bool(official_shared_mcp_output_head),
                main_output_head=main_output_head,
            )
            flow_preds.append(flow_pred)
        return flow_preds
