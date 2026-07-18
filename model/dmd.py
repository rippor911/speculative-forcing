from pipeline import SelfForcingTrainingPipeline
import torch.nn.functional as F
from typing import Optional, Tuple
import torch

from model.base import SelfForcingModel


class DMD(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: SelfForcingTrainingPipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        # Multi-Chunk Prediction (Next Forcing). self.mcp_num_modules is set by
        # BaseModel._initialize_models, which also builds the heads.
        self.mcp_loss_weight = float(getattr(args, "mcp_loss_weight", 1.0))
        # Per-depth loss weights, Eq. 13 / Sec. 5.1: w = [0.5, 0.2, 0.1]. Each depth
        # gets its own hybrid video and KL pass (compute_mcp_loss); these weight the
        # per-depth losses, exactly as they weight the ODE stage's regressions.
        self.mcp_depth_weights = [
            float(w) for w in getattr(args, "mcp_depth_weights", (0.5, 0.2, 0.1))
        ]
        if self.mcp_num_modules > len(self.mcp_depth_weights):
            raise ValueError(
                f"mcp_num_modules={self.mcp_num_modules} exceeds the "
                f"{len(self.mcp_depth_weights)} mcp_depth_weights provided"
            )

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep
        )

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_image - pred_real_image)

        # TODO: Change the normalizer for causal teacher
        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Unroll generator to obtain fake videos
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent
        )

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to
        )

        # Step 3: Multi-Chunk Prediction loss (Next Forcing) -- as DISTRIBUTION
        # MATCHING, mirroring the backbone. The drafts the heads produced inside the
        # rollout are substituted into the video and scored by the same
        # frozen-teacher/critic KL gradient as the backbone's own DMD loss. The heads
        # thus get the identical two-stage treatment the backbone got (coupled ODE
        # regression init -> DMD), instead of the paper's supervised flow-matching
        # loss, which has no non-circular target in this data-free pipeline and, at
        # one-shot sigma=1 use, collapses to a conditional mean.
        if self.mcp_num_modules > 0:
            mcp_loss, mcp_log_dict = self.compute_mcp_loss(
                pred_image=pred_image,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                denoised_timestep_from=denoised_timestep_from,
                denoised_timestep_to=denoised_timestep_to
            )
            dmd_log_dict.update(mcp_log_dict)
            if mcp_loss is not None:
                return dmd_loss + self.mcp_loss_weight * mcp_loss, dmd_log_dict

        return dmd_loss, dmd_log_dict

    def _build_mcp_hybrid(self, video: torch.Tensor, depth: int):
        """Substitute depth-`depth` drafts into `video` at alternating chunk positions.

        Returns (hybrid, frame_mask, used_ids, num_chunks), or None when this depth
        has no draft in the scored window. At sigma=1 a head's one-step draft is
        x0 = eps - v (x_t = eps and x0 = x_t - sigma * v), differentiable through v.

        ALTERNATING (odd) chunks only: the paper's 2x inference mode emits
        main-model chunks at even positions and drafts at odd positions, so this is
        the joint layout the score models should judge -- every draft sits between
        main-model chunks, exactly as deployed. One hybrid is built PER DEPTH and
        each gets its own KL pass (compute_mcp_loss), so every head receives direct
        distribution-matching gradient every step -- the deliberate quality-over-
        compute choice; see the COST NOTE in configs/self_forcing_dmd_mcp.yaml.

        Deterministic and identical across ranks (the rollout's block layout is
        broadcast-synced), so no collectives are needed here.
        """
        records = getattr(self.inference_pipeline, "last_mcp_records", None)
        if not records:
            return None

        offset = getattr(self, "_mcp_frame_offset", 0)
        first_valid = getattr(self, "_mcp_first_valid_frame", 0)
        num_frames = video.shape[1]
        chunk_size = self.num_frame_per_block

        selected = []
        for record in records:
            if record["depth"] != depth:
                continue
            start = record["start_frame"] - offset
            end = start + record["num_frames"]
            if start < first_valid or end > num_frames:
                # Draft frames fall outside the returned last-21 window.
                continue
            if (start // chunk_size) % 2 == 0:
                # Even chunk: the main model's position in the 2x pattern.
                continue
            selected.append((start, record))
        if not selected:
            return None

        hybrid = video.detach().clone()
        frame_mask = torch.zeros_like(hybrid, dtype=torch.bool)
        used_ids = set()
        for start, record in selected:
            draft = record["noise"] - record["flow_pred"]
            hybrid[:, start:start + record["num_frames"]] = draft.to(hybrid.dtype)
            frame_mask[:, start:start + record["num_frames"]] = True
            used_ids.add(id(record))
        return hybrid, frame_mask, used_ids, len(selected)

    def compute_mcp_loss(
        self,
        pred_image: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0
    ):
        """Distribution-matching loss for the MCP heads: the backbone's own DMD
        loss, pointed at a draft-substituted video.

        The rollout video with draft chunks swapped in is scored by the SAME frozen
        teacher and critic through compute_distribution_matching_loss. The
        gradient_mask confines the KL gradient to the drafted chunks, and every
        non-draft chunk is detached, so the heads (and, through their feature taps,
        the backbone) are the only parameters this loss reaches. The backbone's own
        chunks keep receiving their gradient from the main dmd_loss -- nothing is
        double-counted.

        Why not a regression (the paper's Eq. 12)? This pipeline has no ground-truth
        future chunk, and a one-shot sigma=1 head trained by uncoupled regression
        learns the conditional MEAN of the future -- blurry whenever the future is
        multimodal. Distribution matching is what turned the backbone from its
        (equally blurry) ODE init into a sharp one-shot generator; the heads need
        the same medicine for the same reason.

        The critic learns to score these hybrid videos in its own diet
        (critic_loss), keeping the fake score on-distribution.

        One hybrid and one KL pass PER DEPTH, combined as Eq. 13's
        sum_k w_k * L_k: every head gets direct distribution-matching gradient
        every step, exactly as every head gets direct regression gradient every
        step in the ODE stage. Cost scales with mcp_num_modules (see the COST
        NOTE in the config); this is the deliberate quality-over-compute choice.
        """
        records = getattr(self.inference_pipeline, "last_mcp_records", None)
        if not records:
            return None, {}

        mcp_loss = 0.0
        log = {}
        used_ids = set()
        for depth in range(self.mcp_num_modules):
            built = self._build_mcp_hybrid(pred_image, depth)
            if built is None:
                continue
            hybrid, frame_mask, depth_used, num_chunks = built
            l_d, kl_log = self.compute_distribution_matching_loss(
                image_or_video=hybrid,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                gradient_mask=frame_mask,
                denoised_timestep_from=denoised_timestep_from,
                denoised_timestep_to=denoised_timestep_to
            )
            mcp_loss = mcp_loss + self.mcp_depth_weights[depth] * l_d
            used_ids |= depth_used
            log[f"mcp_loss_depth{depth + 1}"] = l_d.detach()
            log[f"mcp_drafts_depth{depth + 1}"] = torch.tensor(
                float(num_chunks), device=pred_image.device)

        if not log:
            return None, {}

        # Ghost term: records never substituted in any depth-hybrid (even-chunk
        # targets) enter the loss times 0.0, so every MCP parameter participates
        # in backward on every rank and FSDP's per-parameter reduce hooks fire
        # uniformly (a param whose hook never fires stalls the collective).
        ghost = 0.0
        for record in records:
            if id(record) not in used_ids and record["flow_pred"].requires_grad:
                ghost = ghost + record["flow_pred"].sum() * 0.0

        log["mcp_loss"] = mcp_loss.detach()
        return mcp_loss + ghost, log

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent
            )

        # Step 2: Compute the fake prediction
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True
        )

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )

        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        # Step 4: MCP hybrid diet. The generator's MCP loss queries the fake score on
        # draft-substituted videos, one hybrid PER DEPTH (compute_mcp_loss), so the
        # critic must track every one of those distributions -- a fake score that has
        # never seen a draft chunk scores it off-distribution and the KL gradient it
        # produces is noise. The records come detached out of the critic's own
        # no_grad rollout, so this trains the critic only, never the heads. The final
        # loss averages the pure video and all per-depth hybrids, at the cost of one
        # fake_score pass per depth per critic step.
        if self.mcp_num_modules > 0:
            def hybrid_denoising_loss_fn(hybrid):
                hybrid_noise = torch.randn_like(hybrid)
                noisy_hybrid = self.scheduler.add_noise(
                    hybrid.flatten(0, 1),
                    hybrid_noise.flatten(0, 1),
                    critic_timestep.flatten(0, 1)
                ).unflatten(0, image_or_video_shape[:2])
                _, pred_fake_hybrid = self.fake_score(
                    noisy_image_or_video=noisy_hybrid,
                    conditional_dict=conditional_dict,
                    timestep=critic_timestep
                )
                if self.args.denoising_loss_type == "flow":
                    from utils.wan_wrapper import WanDiffusionWrapper
                    hybrid_flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                        scheduler=self.scheduler,
                        x0_pred=pred_fake_hybrid.flatten(0, 1),
                        xt=noisy_hybrid.flatten(0, 1),
                        timestep=critic_timestep.flatten(0, 1)
                    )
                    hybrid_fake_noise = None
                else:
                    hybrid_flow_pred = None
                    hybrid_fake_noise = self.scheduler.convert_x0_to_noise(
                        x0=pred_fake_hybrid.flatten(0, 1),
                        xt=noisy_hybrid.flatten(0, 1),
                        timestep=critic_timestep.flatten(0, 1)
                    ).unflatten(0, image_or_video_shape[:2])
                return self.denoising_loss_func(
                    x=hybrid.flatten(0, 1),
                    x_pred=pred_fake_hybrid.flatten(0, 1),
                    noise=hybrid_noise.flatten(0, 1),
                    noise_pred=hybrid_fake_noise,
                    alphas_cumprod=self.scheduler.alphas_cumprod,
                    timestep=critic_timestep.flatten(0, 1),
                    flow_pred=hybrid_flow_pred
                )

            hybrid_losses = []
            for depth in range(self.mcp_num_modules):
                built = self._build_mcp_hybrid(generated_image, depth)
                if built is not None:
                    hybrid_losses.append(hybrid_denoising_loss_fn(built[0]))
            if hybrid_losses:
                denoising_loss = (denoising_loss + sum(hybrid_losses)) / (1 + len(hybrid_losses))

        # Step 5: Debugging Log
        critic_log_dict = {
            "critic_timestep": critic_timestep.detach()
        }

        return denoising_loss, critic_log_dict
