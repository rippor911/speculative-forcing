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

__all__ = ["MCPStack", "mcp_unpatchify", "MCP_INPUT_TIMESTEP"]

# MCP always consumes the future chunk at sigma = 1 (pure noise).
# FlowMatchScheduler builds timesteps = sigmas * num_train_timesteps with
# sigmas[0] = shift * 1.0 / (1 + (shift - 1) * 1.0) = 1.0 for any shift, so
# sigma = 1 <=> timestep = 1000 regardless of the configured timestep_shift.
# At sigma = 1: x_t = (1 - sigma) * x0 + sigma * noise = noise, and the flow
# target v* = (x_t - x0) / sigma reduces exactly to `noise - x0`, i.e. the
# scheduler's own `training_target` (utils/scheduler.py:179). No sigma lookup,
# no division, no argmin snapping.
MCP_INPUT_TIMESTEP = 1000


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

    The paper's mask also grants Noisy -> Clean (each noisy token attends to all
    causally preceding clean context tokens). We cannot: in the Self-Forcing rollout
    the clean history lives in the MAIN model's KV cache, which these weights cannot
    read. History still reaches the head, but only through the fused backbone
    features -- whose tokens already attended over that KV cache. Per-token, not by
    the head's own attention.

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

    def forward(self, x, grid_sizes, freqs, start_frame):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        roped_q = causal_rope_apply(q, grid_sizes, freqs, start_frame=start_frame).type_as(v)
        roped_k = causal_rope_apply(k, grid_sizes, freqs, start_frame=start_frame).type_as(v)

        x = attention(roped_q, roped_k, v)
        return self.o(x.flatten(2))


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

    def forward(self, x, e, grid_sizes, freqs, start_frame):
        r"""
        Args:
            x (Tensor): [B, L, C]
            e (Tensor): [B, F, 6, C]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            grid_sizes,
            freqs,
            start_frame,
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

    def forward(self, upstream, future_noise, start_frame, timestep, freqs, patch_embedding):
        """
        Args:
            upstream (Tensor): [B, L, dim] fused backbone feature (module 1) or the
                previous module's transformer output (modules 2..K).
            future_noise (Tensor): [B, C_in, F, H, W] noisy latent of the target chunk.
            start_frame (int): absolute frame index of the target chunk (for RoPE).
            timestep (Tensor): [B, F] diffusion timestep of `future_noise`.
            patch_embedding (nn.Module): the BACKBONE's patch embedding, shared per
                Sec. 4.3. Passed in rather than owned, so the parameter stays
                registered once (on the backbone) and MCP's gradient flows into it.
        Returns:
            flow_pred (Tensor): [B, C_out, F, H, W] flow-matching prediction.
            hidden (Tensor): [B, L, dim] this module's transformer output, to be
                chained into the next module.
        """
        # Embed the noisy future chunk -> [B, L, dim]
        x = [patch_embedding(u.unsqueeze(0)) for u in future_noise]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = torch.cat([u.flatten(2).transpose(1, 2) for u in x])

        if x.shape[1] != upstream.shape[1]:
            raise ValueError(
                f"MCP token-count mismatch: future chunk has {x.shape[1]} tokens but the "
                f"upstream feature has {upstream.shape[1]}. MCP requires every chunk to have "
                f"the same frame count (num_frame_per_block)."
            )

        # Concat + Linear Proj
        x = self.proj(torch.cat([x, upstream], dim=-1))

        # Time modulation for the target chunk
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x)
        )
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=timestep.shape)

        for block in self.blocks:
            x = block(x, e0, grid_sizes, freqs, start_frame)
        hidden = x

        out = self.head(x, e.unflatten(dim=0, sizes=timestep.shape).unsqueeze(2))
        flow_pred = mcp_unpatchify(out, grid_sizes, self.patch_size, self.out_dim)
        return flow_pred, hidden


class MCPStack(nn.Module):
    """Multi-layer feature fusion + a chain of MCP modules.

    Two deliberate departures from the paper, both forced by this repo being a
    data-free DMD distillation rather than teacher-forced supervised training:

    1. The heads are TRAINED the way the backbone is trained here, not the way the
       paper trains its heads: coupled one-step x0 regression onto the ODE
       trajectory endpoint in the ODE-init stage (model/ode_regression.py), then
       distribution matching on draft-substituted videos in the DMD stage
       (model/dmd.py compute_mcp_loss). There is no ground-truth video to run the
       paper's Eq. 5/12 against, and a one-shot drafting head trained by uncoupled
       regression collapses to the conditional mean of the future.
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

    def forward(self, features, future_noises, future_start_frames, timesteps, freqs, patch_embedding):
        """
        Args:
            features (List[Tensor]): tapped backbone hidden states, each [B, L, dim],
                in the order of `self.tap_layers`.
            future_noises (List[Optional[Tensor]]): per-module noisy target chunk,
                [B, C_in, F, H, W]. A `None` entry (target chunk runs past the end of
                the video) stops the chain there -- the paper instead pads by
                replicating the last chunk and excludes it from the loss (Eq. 4, 12),
                which comes to the same thing.
            future_start_frames (List[int]): absolute start frame of each target chunk.
            timesteps (List[Tensor]): per-module [B, F] timestep of the noisy chunk.
            patch_embedding (nn.Module): the backbone's shared patch embedding.
        Returns:
            List[Tensor]: flow predictions [B, C_out, F, H, W], one per module that ran.
        """
        if len(features) != len(self.tap_layers):
            raise ValueError(
                f"MCPStack expected {len(self.tap_layers)} tapped features, got {len(features)}"
            )

        # h_prev^[0] = h_fuse  (Eq. 8)
        upstream = self.fusion(torch.cat(features, dim=-1))

        flow_preds = []
        for k, module in enumerate(self.mcp_modules):
            if k >= len(future_noises) or future_noises[k] is None:
                break
            flow_pred, upstream = module(
                upstream=upstream,
                future_noise=future_noises[k],
                start_frame=future_start_frames[k],
                timestep=timesteps[k],
                freqs=freqs,
                patch_embedding=patch_embedding,
            )
            flow_preds.append(flow_pred)
        return flow_preds
