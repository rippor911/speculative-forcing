import types
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel
from wan.modules.mcp import MCPStack, MCP_INPUT_TIMESTEP


REPO_ROOT = Path(__file__).resolve().parents[1]
WAN_MODELS_ROOT = REPO_ROOT / "wan_models"
FULL_SEQUENCE_FRAME_SEQ_LENGTH = 1560
FULL_SEQUENCE_CHUNK_FRAMES = 3
FULL_SEQUENCE_NUM_CHUNKS = 7
FULL_SEQUENCE_DEPTHS = (1, 2, 3)
FULL_SEQUENCE_FUTURE_EMBEDDING_ORDER = "depth_major"


@dataclass(frozen=True)
class FullSequenceNFSFModelOutputs:
    main_flow_pred: torch.Tensor
    mcp_flow_preds_by_depth: tuple[torch.Tensor, ...]
    tap_shapes: tuple[tuple[int, ...], ...]
    anchor_token_slices: tuple[tuple[int, int], ...]
    future_embedding_order: str = FULL_SEQUENCE_FUTURE_EMBEDDING_ORDER
    main_backbone_forward_count: int = 1
    paper_exact_reproduction: bool = False
    paper_fidelity_mcp1_mask: bool = False
    mcp_path_kind: str = "canonical_target_only"


def drop_mcp_weights(state_dict: dict) -> dict:
    """Strip MCP head weights from a generator state_dict.

    Checkpoints trained with mcp_num_modules > 0 contain mcp.* entries, because
    fsdp_state_dict walks the whole WanDiffusionWrapper. Eval-time loaders build a
    plain wrapper with no heads, so a strict load would abort on unexpected keys.

    Dropping them is the intended behaviour and not a workaround: this is Next
    Forcing's "zero-overhead" inference mode, where the MCP modules are removed and
    the main model runs exactly like the baseline. The heads stay in the checkpoint
    for MCP-accelerated (speculative) decoding, which must load them deliberately.
    """
    return {k: v for k, v in state_dict.items() if not k.startswith("mcp.")}


class WanTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(WAN_MODELS_ROOT / "Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=str(WAN_MODELS_ROOT / "Wan2.1-T2V-1.3B/google/umt5-xxl/"), seq_len=512, clean='whitespace')

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path=str(WAN_MODELS_ROOT / "Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"),
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0
    ):
        super().__init__()

        if is_causal:
            self.model = CausalWanModel.from_pretrained(
                str(WAN_MODELS_ROOT / model_name), local_attn_size=local_attn_size, sink_size=sink_size)
        else:
            self.model = WanModel.from_pretrained(str(WAN_MODELS_ROOT / model_name))
        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 32760  # [1, 21, 16, 60, 104]

        # Multi-Chunk Prediction heads; opt-in via add_mcp_modules().
        self.mcp = None
        self.mcp_tap_layers = None
        self.mcp_initialized_from_backbone = False

        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def add_mcp_modules(
        self,
        num_modules: int = 1,
        num_layers: int = 3,
        tap_layers=(3, 11, 19, 29),
    ) -> None:
        """Attach Next-Forcing style Multi-Chunk Prediction heads to this generator.

        Mirrors the `adding_cls_branch` pattern: the modules are created after the
        backbone is loaded from pretrained weights, so the backbone's registered
        config is untouched. They live on the wrapper (the FSDP root), which means
        `generator.parameters()` picks them up automatically for the optimizer
        (trainer/distillation.py:106) and EMA (trainer/distillation.py:147).

        Note they are NOT present in checkpoints/ode_init.pt, so the strict
        load_state_dict at trainer/distillation.py:168 must tolerate them.
        """
        if not isinstance(self.model, CausalWanModel):
            raise ValueError("MCP heads require the causal backbone (is_causal=True).")
        if self.model.model_type != "t2v":
            # For i2v the backbone's in_dim is the latent channels plus the image
            # conditioning channels, but MCP embeds a bare noise latent. Fail loudly
            # here instead of deep inside MCPModule's Conv3d.
            raise ValueError(
                f"MCP heads currently support t2v only (backbone model_type="
                f"{self.model.model_type!r}); its in_dim would not match a bare latent."
            )
        bad = [i for i in tap_layers if not (0 <= i < self.model.num_layers)]
        if bad:
            raise ValueError(
                f"mcp_tap_layers {bad} out of range for a {self.model.num_layers}-layer backbone."
            )
        # The backbone taps in ascending block order (causal_model.py's block loop appends
        # as it goes), so the fusion always sees features sorted by depth. Sort here too,
        # otherwise an unsorted tuple would silently permute the fusion's concat.
        tap_layers = tuple(sorted(set(tap_layers)))

        m = self.model
        self.mcp = MCPStack(
            dim=m.dim,
            ffn_dim=m.ffn_dim,
            num_heads=m.num_heads,
            out_dim=m.out_dim,
            patch_size=m.patch_size,
            freq_dim=m.freq_dim,
            num_modules=num_modules,
            num_layers=num_layers,
            tap_layers=tap_layers,
            qk_norm=m.qk_norm,
            eps=m.eps,
        )
        # Sec. 5.1: "MCP module weights are initialized from the last few layers of
        # the main model." Must run AFTER the backbone holds its pretrained weights
        # (WanDiffusionWrapper.__init__ has already called from_pretrained above) and
        # BEFORE the trainer FSDP-wraps the generator.
        self.mcp.init_from_backbone(m.blocks)
        self.mcp.requires_grad_(True)
        self.mcp_tap_layers = tuple(tap_layers)
        self.mcp_initialized_from_backbone = True

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward_full_sequence_next_forcing(
        self,
        *,
        noisy_image_or_video: torch.Tensor,
        clean_x: torch.Tensor,
        conditional_dict: dict,
        timestep_main: torch.Tensor,
        mcp_anchor_inputs=(),
        aug_t: Optional[torch.Tensor] = None,
        direct_clean_context_kv: bool = False,
        paper_fidelity_mcp1_mask: bool = False,
    ) -> FullSequenceNFSFModelOutputs:
        """Training-only full-sequence Next-Forcing route.

        The main teacher-forced backbone runs exactly once over clean21+noisy21.
        All valid MCP future chunks are embedded by the backbone's shared
        patch_embedding inside that same model forward, then this wrapper slices
        noisy-half taps and runs the existing single-chunk MCP chain per anchor.
        """
        if clean_x is None:
            raise ValueError("full-sequence Next-Forcing requires clean_x")
        if tuple(clean_x.shape) != tuple(noisy_image_or_video.shape):
            raise ValueError("clean_x and noisy_image_or_video must have the same shape")
        if noisy_image_or_video.ndim != 5:
            raise ValueError("full-sequence tensors must have shape [B, F, C, H, W]")
        if noisy_image_or_video.shape[1] != FULL_SEQUENCE_CHUNK_FRAMES * FULL_SEQUENCE_NUM_CHUNKS:
            raise ValueError("full-sequence Next-Forcing requires 21 latent frames")
        if tuple(timestep_main.shape) != tuple(noisy_image_or_video.shape[:2]):
            raise ValueError("timestep_main must have shape [B, 21]")
        if int(getattr(self.model, "num_frame_per_block", 0)) != FULL_SEQUENCE_CHUNK_FRAMES:
            raise ValueError("model.num_frame_per_block must be 3")

        anchors = tuple(mcp_anchor_inputs or ())
        run_mcp = bool(anchors)
        if direct_clean_context_kv and not run_mcp:
            raise ValueError("direct_clean_context_kv requires MCP anchor inputs")
        if paper_fidelity_mcp1_mask and not run_mcp:
            raise ValueError("paper_fidelity_mcp1_mask requires MCP anchor inputs")
        if direct_clean_context_kv and paper_fidelity_mcp1_mask:
            raise ValueError("direct_clean_context_kv and paper_fidelity_mcp1_mask are mutually exclusive")
        if run_mcp:
            if self.mcp is None:
                raise ValueError("MCP anchor inputs require add_mcp_modules()")
            if self.mcp_tap_layers is None:
                raise ValueError("mcp_tap_layers missing; call add_mcp_modules() first")

        if self.uniform_timestep:
            input_timestep = timestep_main[:, 0]
        else:
            input_timestep = timestep_main

        prompt_embeds = conditional_dict["prompt_embeds"]
        if aug_t is None:
            aug_t = torch.zeros_like(timestep_main)

        mcp_patch_inputs = None
        flat_mcp_entries = ()
        if run_mcp:
            flat_mcp_entries = self._flatten_full_sequence_mcp_anchor_inputs(anchors)
            mcp_patch_inputs = [
                entry["future_noise"].permute(0, 2, 1, 3, 4)
                for entry in flat_mcp_entries
            ]

        model_kwargs = {}
        if run_mcp:
            model_kwargs = {
                "return_features": self.mcp_tap_layers,
                "mcp_patch_inputs": mcp_patch_inputs,
            }
            if direct_clean_context_kv or paper_fidelity_mcp1_mask:
                model_kwargs["return_feature_halves"] = True

        out = self.model(
            noisy_image_or_video.permute(0, 2, 1, 3, 4),
            t=input_timestep,
            context=prompt_embeds,
            seq_len=self.seq_len,
            clean_x=clean_x.permute(0, 2, 1, 3, 4),
            aug_t=aug_t,
            **model_kwargs,
        )

        if run_mcp:
            main_flow_pred, aux = out
            mcp_flow_preds_by_depth, tap_shapes, anchor_slices = (
                self._run_full_sequence_anchor_mcp(
                    aux=aux,
                    anchors=anchors,
                    flat_mcp_entries=flat_mcp_entries,
                    direct_clean_context_kv=direct_clean_context_kv,
                    paper_fidelity_mcp1_mask=paper_fidelity_mcp1_mask,
                )
            )
        else:
            main_flow_pred = out
            mcp_flow_preds_by_depth = ()
            tap_shapes = ()
            anchor_slices = self._full_sequence_anchor_token_slices()

        return FullSequenceNFSFModelOutputs(
            main_flow_pred=main_flow_pred.permute(0, 2, 1, 3, 4),
            mcp_flow_preds_by_depth=mcp_flow_preds_by_depth,
            tap_shapes=tap_shapes,
            anchor_token_slices=anchor_slices,
            paper_exact_reproduction=False,
            paper_fidelity_mcp1_mask=bool(paper_fidelity_mcp1_mask),
            mcp_path_kind=(
                "paper_fidelity_clean_residual_mask"
                if paper_fidelity_mcp1_mask
                else (
                    "direct_clean_static_kv"
                    if direct_clean_context_kv
                    else "canonical_target_only"
                )
            ),
        )

    def _flatten_full_sequence_mcp_anchor_inputs(self, anchors) -> tuple[dict, ...]:
        entries = []
        for depth in FULL_SEQUENCE_DEPTHS:
            for anchor in anchors:
                anchor_index = int(self._anchor_value(anchor, "anchor_index"))
                depths = tuple(int(value) for value in self._anchor_value(anchor, "depths"))
                if depth not in depths:
                    continue
                local_index = depths.index(depth)
                future_noises = tuple(self._anchor_value(anchor, "future_noises"))
                future_start_frames = tuple(self._anchor_value(anchor, "future_start_frames"))
                timesteps = tuple(self._anchor_value(anchor, "timesteps"))
                future_noise = future_noises[local_index]
                timestep = timesteps[local_index]
                if future_noise.shape[1] != FULL_SEQUENCE_CHUNK_FRAMES:
                    raise ValueError("each MCP future noise must be one 3-frame chunk")
                if tuple(timestep.shape) != tuple(future_noise.shape[:2]):
                    raise ValueError("each MCP timestep must have shape [B, 3]")
                entries.append(
                    {
                        "anchor_index": anchor_index,
                        "depth": depth,
                        "future_noise": future_noise,
                        "future_start_frame": int(future_start_frames[local_index]),
                    }
                )
        expected = sum(FULL_SEQUENCE_NUM_CHUNKS - depth for depth in FULL_SEQUENCE_DEPTHS)
        if len(entries) != expected:
            raise ValueError(f"full-sequence MCP expected {expected} valid futures")
        for flat_index, entry in enumerate(entries):
            entry["flat_index"] = flat_index
        return tuple(entries)

    def _run_full_sequence_anchor_mcp(
        self,
        *,
        aux,
        anchors,
        flat_mcp_entries,
        direct_clean_context_kv: bool = False,
        paper_fidelity_mcp1_mask: bool = False,
    ):
        features = tuple(aux["features"])
        if len(features) != len(self.mcp_tap_layers):
            raise ValueError("MCP feature tap count mismatch")
        clean_features = None
        if direct_clean_context_kv or paper_fidelity_mcp1_mask:
            clean_features = tuple(aux["clean_features"])
            if len(clean_features) != len(self.mcp_tap_layers):
                raise ValueError("MCP clean feature tap count mismatch")
        embeds = tuple(aux["mcp_embeds"])
        grids = tuple(aux["mcp_grid_sizes"])
        if len(embeds) != len(flat_mcp_entries) or len(grids) != len(flat_mcp_entries):
            raise ValueError("embedded MCP future count mismatch")

        embed_index = {
            (int(entry["anchor_index"]), int(entry["depth"])): index
            for index, entry in enumerate(flat_mcp_entries)
        }
        flow_preds_by_depth = {depth: [] for depth in FULL_SEQUENCE_DEPTHS}
        for anchor in anchors:
            anchor_index = int(self._anchor_value(anchor, "anchor_index"))
            token_slice = self._full_sequence_anchor_token_slice(anchor_index)
            anchor_features = tuple(feature[:, token_slice, :] for feature in features)
            depths = tuple(int(value) for value in self._anchor_value(anchor, "depths"))
            starts = tuple(int(value) for value in self._anchor_value(anchor, "future_start_frames"))
            timesteps = tuple(self._anchor_value(anchor, "timesteps"))

            future_embeds = []
            future_grids = []
            for depth in depths:
                index = embed_index[(anchor_index, depth)]
                future_embeds.append(embeds[index])
                future_grids.append(grids[index])

            mcp_kwargs = {}
            if direct_clean_context_kv and anchor_index > 0:
                clean_stop = token_slice.start
                mcp_kwargs = {
                    "direct_clean_context_kv": True,
                    "clean_context_features": tuple(
                        feature[:, :clean_stop, :] for feature in clean_features
                    ),
                    "clean_context_grid_sizes": self._full_sequence_clean_context_grid_sizes(
                        future_grids[0],
                        anchor_index,
                    ),
                    "clean_context_start_frame": 0,
                }
            if paper_fidelity_mcp1_mask:
                clean_stop = token_slice.start
                clean_frame_count = int(anchor_index) * FULL_SEQUENCE_CHUNK_FRAMES
                mcp_kwargs = {
                    "paper_fidelity_mcp1_mask": True,
                    "paper_fidelity_clean_prefix_features": tuple(
                        feature[:, :clean_stop, :] for feature in clean_features
                    ),
                    "paper_fidelity_clean_prefix_grid_sizes": self._full_sequence_clean_prefix_grid_sizes(
                        future_grids[0],
                        clean_frame_count,
                    ),
                    "paper_fidelity_clean_prefix_start_frame": 0,
                    "paper_fidelity_clean_prefix_timestep": torch.zeros(
                        (timesteps[0].shape[0], clean_frame_count),
                        device=timesteps[0].device,
                        dtype=timesteps[0].dtype,
                    ),
                }

            flow_preds = self.mcp(
                features=anchor_features,
                future_embeds=future_embeds,
                future_grid_sizes=future_grids,
                future_start_frames=list(starts),
                timesteps=list(timesteps),
                freqs=self.model.freqs,
                **mcp_kwargs,
            )
            if len(flow_preds) != len(depths):
                raise RuntimeError("MCP chain output count mismatch for anchor")
            for depth, pred in zip(depths, flow_preds):
                flow_preds_by_depth[depth].append(pred.permute(0, 2, 1, 3, 4))

        stacked = []
        for depth in FULL_SEQUENCE_DEPTHS:
            preds = flow_preds_by_depth[depth]
            expected_count = FULL_SEQUENCE_NUM_CHUNKS - depth
            if len(preds) != expected_count:
                raise RuntimeError(
                    f"MCP depth {depth} expected {expected_count} anchors, got {len(preds)}"
                )
            stacked.append(torch.stack(preds, dim=1))
        tap_shapes = tuple(tuple(int(dim) for dim in feature.shape) for feature in features)
        anchor_slices = self._full_sequence_anchor_token_slices()
        return tuple(stacked), tap_shapes, tuple(anchor_slices)

    @staticmethod
    def _full_sequence_anchor_token_slice(anchor_index: int) -> slice:
        if not (0 <= int(anchor_index) < FULL_SEQUENCE_NUM_CHUNKS):
            raise ValueError("anchor_index out of range for full-sequence v1")
        chunk_tokens = FULL_SEQUENCE_FRAME_SEQ_LENGTH * FULL_SEQUENCE_CHUNK_FRAMES
        start = int(anchor_index) * chunk_tokens
        return slice(start, start + chunk_tokens)

    @staticmethod
    def _full_sequence_clean_context_grid_sizes(
        target_grid_sizes: torch.Tensor,
        anchor_index: int,
    ) -> torch.Tensor:
        if int(anchor_index) <= 0:
            raise ValueError("clean context requires anchor_index > 0")
        grid_sizes = target_grid_sizes.clone()
        grid_sizes[:, 0] = int(anchor_index) * FULL_SEQUENCE_CHUNK_FRAMES
        return grid_sizes

    @staticmethod
    def _full_sequence_clean_prefix_grid_sizes(
        target_grid_sizes: torch.Tensor,
        clean_frame_count: int,
    ) -> torch.Tensor:
        if int(clean_frame_count) < 0:
            raise ValueError("clean_frame_count must be non-negative")
        grid_sizes = target_grid_sizes.clone()
        grid_sizes[:, 0] = int(clean_frame_count)
        return grid_sizes

    @classmethod
    def _full_sequence_anchor_token_slices(cls) -> tuple[tuple[int, int], ...]:
        return tuple(
            (
                cls._full_sequence_anchor_token_slice(index).start,
                cls._full_sequence_anchor_token_slice(index).stop,
            )
            for index in range(FULL_SEQUENCE_NUM_CHUNKS)
        )

    @staticmethod
    def _anchor_value(anchor, key: str):
        if isinstance(anchor, dict):
            return anchor[key]
        return getattr(anchor, key)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        mcp_future_noises: Optional[List[Optional[torch.Tensor]]] = None,
        mcp_future_start_frames: Optional[List[int]] = None,
        mcp_timesteps: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        MCP (opt-in): pass `mcp_future_noises` (a list of [B, F, C, H, W] noisy target
        chunks, one per MCP depth, `None` to stop the chain) with their
        `mcp_future_start_frames` (the RoPE offset, i.e. Eq. 6's i+k). When given, the
        return becomes (flow_pred, pred_x0, mcp_flow_preds); otherwise the existing
        (flow_pred, pred_x0) contract is unchanged, which matters because ~20 call
        sites unpack exactly two values.

        `mcp_timesteps` supplies the per-depth [B, F] timestep of each noisy input.
        Omit it to default to sigma=1 (pure noise), the DMD rollout's drafting
        condition. The ODE-init stage passes the student's denoising timesteps with
        the trajectory's own latents (model/ode_regression.py), mirroring the
        backbone's distillation.

        Works on both the kv_cache (rollout) and the plain teacher-forced paths.
        """
        prompt_embeds = conditional_dict["prompt_embeds"]

        run_mcp = mcp_future_noises is not None
        if run_mcp:
            if self.mcp is None:
                raise ValueError("mcp_future_noises given but add_mcp_modules() was never called.")
            if classify_mode:
                raise ValueError("MCP and classify_mode cannot be combined.")
            if mcp_future_start_frames is None or len(mcp_future_start_frames) != len(mcp_future_noises):
                raise ValueError("mcp_future_start_frames must align 1:1 with mcp_future_noises.")
            if mcp_timesteps is not None and len(mcp_timesteps) != len(mcp_future_noises):
                raise ValueError("mcp_timesteps must align 1:1 with mcp_future_noises.")

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        mcp_flow_preds = None
        # [B, F, C, H, W] -> [B, C, F, H, W], matching the main path's permute. The
        # chunks are embedded by the backbone INSIDE its own forward (see
        # CausalWanModel._embed_mcp_chunks for why), so they must be handed to the
        # model call rather than to the heads directly.
        mcp_patch_inputs = None
        mcp_kwargs = {}
        if run_mcp:
            mcp_patch_inputs = [
                None if c is None else c.permute(0, 2, 1, 3, 4)
                for c in mcp_future_noises
            ]
            # Only the causal backbone knows these kwargs; the bidirectional score
            # models (WanModel) must not receive them at all.
            mcp_kwargs = {
                "return_features": self.mcp_tap_layers,
                "mcp_patch_inputs": mcp_patch_inputs,
            }
        # X0 prediction
        if kv_cache is not None:
            out = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                **mcp_kwargs
            )
            if run_mcp:
                flow_pred, aux = out
                flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                mcp_flow_preds = self._run_mcp(
                    aux=aux,
                    mcp_future_noises=mcp_future_noises,
                    mcp_future_start_frames=mcp_future_start_frames,
                    mcp_timesteps=mcp_timesteps
                )
            else:
                flow_pred = out.permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                out = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                    **mcp_kwargs
                )
                if run_mcp:
                    flow_pred, aux = out
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                    mcp_flow_preds = self._run_mcp(
                        aux=aux,
                        mcp_future_noises=mcp_future_noises,
                        mcp_future_start_frames=mcp_future_start_frames,
                        mcp_timesteps=mcp_timesteps
                    )
                else:
                    flow_pred = out.permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    out = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        **mcp_kwargs
                    )
                    if run_mcp:
                        # The ODE-init path: one teacher-forced forward over the whole
                        # sequence, which is where MCP can follow the paper exactly.
                        flow_pred, aux = out
                        flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                        mcp_flow_preds = self._run_mcp(
                            aux=aux,
                            mcp_future_noises=mcp_future_noises,
                            mcp_future_start_frames=mcp_future_start_frames,
                            mcp_timesteps=mcp_timesteps
                        )
                    else:
                        flow_pred = out.permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        if mcp_flow_preds is not None:
            return flow_pred, pred_x0, mcp_flow_preds

        return flow_pred, pred_x0

    def _run_mcp(self, aux, mcp_future_noises, mcp_future_start_frames, mcp_timesteps=None):
        """Run the MCP chain on the backbone's tapped features.

        Must be called from inside `forward` so it executes within the FSDP root's
        forward pass: MCP params are only gathered while the root is running, so
        invoking `generator.mcp(...)` from outside would read sharded garbage.

        With `mcp_timesteps=None` the inputs are assumed to be at sigma=1 (pure
        noise), the DMD rollout's drafting condition: the future chunk is untouched
        noise at that moment. The ODE-init stage passes the trajectory's own latents
        at the student's denoising timesteps instead (model/ode_regression.py),
        mirroring the backbone's distillation.
        """
        # Derive the batch size from the first chunk that is actually present rather
        # than mcp_future_noises[0]: the caller guarantees at least one non-None entry,
        # but not that it is the first one.
        first = next(c for c in mcp_future_noises if c is not None)
        batch_size = first.shape[0]
        starts, timesteps = [], []
        for k, (chunk, start) in enumerate(zip(mcp_future_noises, mcp_future_start_frames)):
            if chunk is None:
                starts.append(None)
                timesteps.append(None)
                continue
            starts.append(start)
            if mcp_timesteps is not None:
                timesteps.append(mcp_timesteps[k])
            else:
                timesteps.append(
                    torch.full(
                        (batch_size, chunk.shape[1]),
                        MCP_INPUT_TIMESTEP,
                        device=chunk.device,
                        dtype=torch.int64
                    )
                )

        # The chunk tokens were embedded by the backbone inside its own forward
        # (CausalWanModel._embed_mcp_chunks) so the shared patch_embedding was in
        # FSDP-gathered, mixed-precision-cast state. freqs is a buffer (never
        # flat-sharded), so reading it here is safe.
        flow_preds = self.mcp(
            features=aux["features"],
            future_embeds=aux["mcp_embeds"],
            future_grid_sizes=aux["mcp_grid_sizes"],
            future_start_frames=starts,
            timesteps=timesteps,
            freqs=self.model.freqs,
        )
        # [B, C, F, H, W] -> [B, F, C, H, W]
        return [p.permute(0, 2, 1, 3, 4) for p in flow_preds]

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
