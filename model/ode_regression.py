import torch.nn.functional as F
from typing import Tuple
import torch

from model.base import BaseModel
from utils.checkpoint import (
    extract_generator_state_dict,
    load_state_dict_allowing_mcp_mismatch,
)
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class ODERegression(BaseModel):
    def __init__(self, args, device):
        """
        Initialize the ODERegression module.
        This class is self-contained and compute generator losses
        in the forward pass given precomputed ode solution pairs.
        This class supports the ode regression loss for both causal and bidirectional models.
        See Sec 4.3 of CausVid https://arxiv.org/abs/2412.07772 for details
        """
        super().__init__(args, device)

        # Step 1: Initialize all models

        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Step 2: Initialize all hyperparameters
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)

        # Next-Forcing Multi-Chunk Prediction, initialized here the way the BACKBONE
        # is initialized here: the heads are trained by the same coupled one-step
        # regression onto the trajectory endpoint (see _prepare_mcp_inputs), shifted
        # k chunks ahead. The DMD stage then trains them through the accelerated
        # rollout itself (pipeline/self_forcing_training.py, the drafts are part of
        # the emitted video) -- mirrored distillation in both stages, NOT the
        # paper's supervised flow-matching recipe, which assumes ground-truth video
        # this pipeline does not have.
        #
        # Built here rather than in _initialize_models because ODERegression's ctor
        # reconstructs self.generator itself, so the heads must be attached AFTER
        # that, before loading generator_ckpt and before the trainer FSDP-wraps it.
        self.mcp_num_modules = int(getattr(args, "mcp_num_modules", 0))
        self.mcp_loss_weight = float(getattr(args, "mcp_loss_weight", 1.0))
        self.mcp_depth_weights = [
            float(w) for w in getattr(args, "mcp_depth_weights", (0.5, 0.2, 0.1))
        ]
        if self.mcp_num_modules > 0:
            if self.mcp_num_modules > len(self.mcp_depth_weights):
                raise ValueError(
                    f"mcp_num_modules={self.mcp_num_modules} exceeds the "
                    f"{len(self.mcp_depth_weights)} mcp_depth_weights provided"
                )
            # Rank-independent seed: the trainer calls set_seed(config.seed + rank)
            # and fsdp_wrap uses sync_module_states=False, so without this every rank
            # would build different heads. See model/base.py for the same guard.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(getattr(args, "mcp_init_seed", 0)))
                self.generator.add_mcp_modules(
                    num_modules=self.mcp_num_modules,
                    num_layers=int(getattr(args, "mcp_num_layers", 3)),
                    tap_layers=tuple(getattr(args, "mcp_tap_layers", (3, 11, 19, 29)))
                )

        if getattr(args, "generator_ckpt", False):
            print(f"Loading pretrained generator from {args.generator_ckpt}")
            checkpoint = torch.load(args.generator_ckpt, map_location="cpu")
            state_dict = extract_generator_state_dict(checkpoint)
            missing, unexpected = load_state_dict_allowing_mcp_mismatch(
                self.generator, state_dict
            )
            if missing:
                print(
                    f"MCP: {len(missing)} params not in checkpoint; "
                    "kept from initialization"
                )
            if unexpected:
                print(
                    f"MCP: {len(unexpected)} checkpoint params unused by this config"
                )

    def _initialize_models(self, args, device):
        # `device` is required: BaseModel.__init__ calls self._initialize_models(args,
        # device) (model/base.py:15). This override used to take only `args`, so
        # constructing ODERegression raised TypeError before reaching any of the code
        # below -- the ODE stage was unrunnable, which is consistent with upstream
        # shipping no ODE config and deferring the instructions.
        self.generator = WanDiffusionWrapper(**getattr(args, "model_kwargs", {}), is_causal=True)
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        # BaseModel.__init__ reads self.scheduler right after this returns, to warp
        # denoising_step_list.
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    @torch.no_grad()
    def _prepare_generator_input(self, ode_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories,
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - ode_latent: a tensor containing the whole ODE sampling trajectories [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
        Output:
            - noisy_input: a tensor containing the selected latent [batch_size, num_frames, num_channels, height, width].
            - timestep: a tensor containing the corresponding timestep [batch_size].
        """
        batch_size, num_denoising_steps, num_frames, num_channels, height, width = ode_latent.shape
        latent_device = ode_latent.device
        denoising_steps = self.denoising_step_list.to(latent_device)

        # Step 1: Randomly choose a timestep for each frame
        index = self._get_timestep(
            0,
            len(denoising_steps),
            batch_size,
            num_frames,
            self.num_frame_per_block,
            uniform_timestep=False
        ).to(latent_device)
        if self.args.i2v:
            index[:, 0] = len(denoising_steps) - 1

        noisy_input = torch.gather(
            ode_latent, dim=1,
            index=index.reshape(batch_size, 1, num_frames, 1, 1, 1).expand(
                -1, -1, -1, num_channels, height, width)
        ).squeeze(1)

        timestep = denoising_steps[index].to(self.device)

        # if self.extra_noise_step > 0:
        #     random_timestep = torch.randint(0, self.extra_noise_step, [
        #                                     batch_size, num_frames], device=self.device, dtype=torch.long)
        #     perturbed_noisy_input = self.scheduler.add_noise(
        #         noisy_input.flatten(0, 1),
        #         torch.randn_like(noisy_input.flatten(0, 1)),
        #         random_timestep.flatten(0, 1)
        #     ).detach().unflatten(0, (batch_size, num_frames)).type_as(noisy_input)

        #     noisy_input[timestep == 0] = perturbed_noisy_input[timestep == 0]

        return noisy_input, timestep

    def generator_loss(self, ode_latent: torch.Tensor, conditional_dict: dict) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noisy latents and compute the ODE regression loss.
        Input:
            - ode_latent: a tensor containing the ODE latents [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
            They are ordered from most noisy to clean latents.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - loss: a scalar tensor representing the generator loss.
            - log_dict: a dictionary containing additional information for loss timestep breakdown.
        """
        # Step 1: Run generator on noisy latents
        target_latent = ode_latent[:, -1]

        noisy_input, timestep = self._prepare_generator_input(
            ode_latent=ode_latent)

        mcp_inputs = self._prepare_mcp_inputs(ode_latent) if self.mcp_num_modules > 0 else None

        if mcp_inputs is not None:
            _, pred_image_or_video, mcp_flow_preds = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                mcp_future_noises=mcp_inputs["noisy"],
                mcp_future_start_frames=mcp_inputs["start_frames"],
                mcp_timesteps=mcp_inputs["timesteps"]
            )
        else:
            _, pred_image_or_video = self.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep
            )

        # Step 2: Compute the regression loss
        mask = timestep != 0

        loss = F.mse_loss(
            pred_image_or_video[mask], target_latent[mask], reduction="mean")

        log_dict = {
            "unnormalized_loss": F.mse_loss(pred_image_or_video, target_latent, reduction='none').mean(dim=[1, 2, 3, 4]).detach(),
            "timestep": timestep.float().mean(dim=1).detach(),
            "input": noisy_input.detach(),
            "output": pred_image_or_video.detach(),
        }

        # Step 3: Multi-Chunk Prediction auxiliary loss (Next Forcing).
        if mcp_inputs is not None:
            mcp_loss, mcp_log = self._compute_mcp_loss(mcp_inputs, mcp_flow_preds)
            log_dict.update(mcp_log)
            if mcp_loss is not None:
                loss = loss + self.mcp_loss_weight * mcp_loss

        return loss, log_dict

    def _shift_chunks(self, latent: torch.Tensor, k: int) -> torch.Tensor:
        """Eq. 4: x_0^[k][i] = x_0[min(i + k, F)], i.e. advance the video by k CHUNKS,
        replicating the last chunk past the boundary.

        `latent` is [B, num_frames, C, H, W]; chunks are num_frame_per_block frames.
        """
        m = self.num_frame_per_block
        num_frames = latent.shape[1]
        if num_frames % m != 0:
            raise ValueError(
                f"MCP chunk shifting needs num_frames ({num_frames}) divisible by "
                f"num_frame_per_block ({m})"
            )
        num_chunks = num_frames // m
        pieces = []
        for i in range(num_chunks):
            j = min(i + k, num_chunks - 1)
            pieces.append(latent[:, j * m:(j + 1) * m])
        return torch.cat(pieces, dim=1)

    def _mcp_valid_frame_mask(self, num_frames: int, k: int, device) -> torch.Tensor:
        """Eq. 12: "the last k padded chunks are excluded from the loss computation"."""
        m = self.num_frame_per_block
        num_chunks = num_frames // m
        mask = torch.zeros(num_frames, dtype=torch.bool, device=device)
        valid_chunks = max(num_chunks - k, 0)
        mask[: valid_chunks * m] = True
        return mask

    @torch.no_grad()
    def _prepare_mcp_inputs(self, ode_latent: torch.Tensor) -> dict:
        """Build MCP inputs exactly the way the backbone's own distillation does.

        The backbone's recipe in this stage (`_prepare_generator_input` +
        `generator_loss`): pick one of the student's noise levels, take the
        TRAJECTORY'S OWN latent at that level, and one-step regress the x0
        prediction onto the trajectory endpoint. The heads get the identical
        treatment, shifted k chunks ahead:

            input   shift_chunks(ode_latent[:, i], k)   trajectory point at level i
            label   denoising_step_list[i]
            target  shift_chunks(ode_latent[:, -1], k)  trajectory endpoint

        Because (input, target) lie on the same teacher ODE trajectory they are
        deterministically coupled, so the ONE-STEP regression is well-posed -- the
        same argument as ODE init for the backbone (CausVid Sec 4.3). This replaces
        the paper's Eq. 5 (fresh noise at a random t_k under s_mcp): that recipe
        trains a multi-step denoiser, the right object in the paper's from-scratch
        supervised setting, but an UNCOUPLED one-step regression collapses to the
        conditional mean -- the wrong object for a one-shot drafting head. At level
        0 the input is the chunk's pure starting noise (sigma = 1), the literal
        inference-time drafting condition.
        """
        batch_size, _, num_frames = ode_latent.shape[:3]
        device, dtype = ode_latent.device, ode_latent.dtype
        target_latent = ode_latent[:, -1]

        denoising_steps = self.denoising_step_list.to(device)
        num_levels = len(denoising_steps)
        scheduler_timesteps = self.scheduler.timesteps.to(device)
        scheduler_sigmas = self.scheduler.sigmas.to(device)

        noisy, clean, timesteps, sigmas, masks, starts = [], [], [], [], [], []
        for k in range(1, self.mcp_num_modules + 1):
            # One level per sample per depth, mirroring the backbone's random pick.
            index = torch.randint(0, num_levels, (batch_size,), device=device)
            x_t = ode_latent[torch.arange(batch_size, device=device), index]
            t_k = denoising_steps[index].float()

            # sigma_t via the same argmin lookup as _convert_flow_pred_to_x0, so the
            # x0 conversion in the loss is consistent with the wrapper's.
            timestep_id = torch.argmin(
                (scheduler_timesteps.unsqueeze(0) - t_k.unsqueeze(1)).abs(), dim=1)
            sigma_k = scheduler_sigmas[timestep_id].reshape(batch_size, 1, 1, 1, 1)

            noisy.append(self._shift_chunks(x_t, k).to(dtype))
            clean.append(self._shift_chunks(target_latent, k))
            # The wrapper needs a [B, F] timestep; the level is constant across the
            # sequence for this depth.
            timesteps.append(t_k.unsqueeze(1).expand(batch_size, num_frames))
            sigmas.append(sigma_k)
            masks.append(self._mcp_valid_frame_mask(num_frames, k, device))
            # Eq. 6: RoPE(i + k). A uniform offset of k chunks over the sequence gives
            # every token at frame j the position j + k*M, which is exactly i+k in
            # chunk terms. causal_rope_apply(start_frame=0) == rope_apply, so this is
            # consistent with the main model's own positions.
            starts.append(k * self.num_frame_per_block)

        return {
            "noisy": noisy, "clean": clean, "timesteps": timesteps,
            "sigmas": sigmas, "masks": masks, "start_frames": starts,
        }

    def _compute_mcp_loss(self, mcp_inputs, mcp_flow_preds):
        """One-step x0 regression onto the ODE endpoint, mirroring the backbone.

        The head predicts flow; convert with x0 = x_t - sigma_t * v (the identity in
        WanDiffusionWrapper._convert_flow_pred_to_x0) and take the MSE in x0 space,
        the space the backbone's own ODE loss above lives in. Combined across depths
        as sum_k w_k * L_k (Eq. 13, w = [0.5, 0.2, 0.1]).
        """
        if not mcp_flow_preds:
            return None, {}

        mcp_loss = 0.0
        log = {}
        for k, flow_pred in enumerate(mcp_flow_preds):
            frame_mask = mcp_inputs["masks"][k]
            if not bool(frame_mask.any()):
                continue
            x0_pred = (
                mcp_inputs["noisy"][k].float()
                - mcp_inputs["sigmas"][k].float() * flow_pred.float()
            )
            l_k = F.mse_loss(
                x0_pred[:, frame_mask],
                mcp_inputs["clean"][k][:, frame_mask].float().detach(),
                reduction="mean"
            )
            mcp_loss = mcp_loss + self.mcp_depth_weights[k] * l_k
            log[f"mcp_loss_depth{k + 1}"] = l_k.detach()

        if not log:
            return None, {}
        log["mcp_loss"] = mcp_loss.detach()
        return mcp_loss, log
