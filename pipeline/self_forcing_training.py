from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from contextlib import nullcontext
from typing import List, Optional
import torch
import torch.distributed as dist


class SelfForcingTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 memory_gap_blocks: int = 0,
                 memory_gap_sample_mode: str = "fixed",
                 memory_gap_min_blocks: int = 0,
                 memory_gap_max_blocks: int = 0,
                 mcp_num_modules: int = 0,
                 mcp_accel_depths: int = 0,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length
        self.memory_gap_blocks = int(memory_gap_blocks)
        self.memory_gap_sample_mode = memory_gap_sample_mode
        self.memory_gap_min_blocks = int(memory_gap_min_blocks)
        self.memory_gap_max_blocks = int(memory_gap_max_blocks)

        self.mcp_num_modules = int(mcp_num_modules)
        # Heads deployed at inference. When > 0 the rollout simulates the
        # MCP-accelerated inference loop itself (_inference_with_trajectory_mcp_
        # accelerated): the emitted video contains the heads' drafts, so the plain
        # DMD loss trains backbone and heads jointly -- no auxiliary MCP loss.
        self.mcp_accel_depths = int(mcp_accel_depths) if mcp_accel_depths else self.mcp_num_modules
        if self.mcp_num_modules > 0:
            if not (0 < self.mcp_accel_depths <= self.mcp_num_modules):
                raise ValueError(
                    f"mcp_accel_depths={self.mcp_accel_depths} must be in "
                    f"[1, mcp_num_modules={self.mcp_num_modules}]"
                )
            if self.memory_gap_blocks > 0 or self.memory_gap_max_blocks > 0:
                raise ValueError(
                    "MCP-accelerated rollout and memory_gap are not composable yet: "
                    "the accelerated rollout already defines its own KV commit "
                    "schedule (drafts committed before the next anchor)."
                )

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        if dist.is_initialized():
            dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def sample_memory_gap_blocks(self, device):
        if self.memory_gap_sample_mode == "fixed":
            return self.memory_gap_blocks

        if self.memory_gap_sample_mode != "uniform":
            raise ValueError(f"Unsupported memory_gap_sample_mode: {self.memory_gap_sample_mode}")

        min_gap = self.memory_gap_min_blocks
        max_gap = self.memory_gap_max_blocks
        if max_gap < min_gap:
            raise ValueError("memory_gap_max_blocks must be >= memory_gap_min_blocks")

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            gap = torch.randint(min_gap, max_gap + 1, (1,), device=device, dtype=torch.long)
        else:
            gap = torch.empty(1, device=device, dtype=torch.long)
        if dist.is_initialized():
            dist.broadcast(gap, src=0)
        return int(gap.item())

    def _mcp_future_chunks(self, block_index, block_starts, all_num_frames, noise, num_input_frames):
        """Pure-noise latents and absolute start frames of the chunks MCP must predict.

        At block i, MCP module k predicts block i+1+k. Those blocks have not been
        touched by the rollout yet, so `noise` still holds their pristine sigma=1
        latents -- exactly the input MCP is trained on. A `None` entry stops the
        MCP chain at that point (see MCPStack.forward).
        """
        cur_num_frames = all_num_frames[block_index]
        noises, starts = [], []
        for k in range(self.mcp_num_modules):
            j = block_index + 1 + k
            valid = j < len(all_num_frames)
            if valid:
                num_frames = all_num_frames[j]
                lo = block_starts[j] - num_input_frames
                hi = lo + num_frames
                # MCP concatenates the future chunk's tokens with the current
                # chunk's feature, so their frame counts must match. They only
                # differ for the independent-first-frame block.
                valid = num_frames == cur_num_frames and lo >= 0 and hi <= noise.shape[1]
            if not valid:
                noises.append(None)
                starts.append(None)
                continue
            noises.append(noise[:, lo:hi])
            starts.append(block_starts[j])
        return noises, starts

    def _inference_with_trajectory_mcp_accelerated(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        """Simulate the MCP-accelerated inference loop (Self Forcing's principle
        applied to the accelerated generator).

        Deployment advances P = mcp_accel_depths + 1 chunks per autoregressive
        step: the backbone denoises the ANCHOR chunk while the deployed heads
        one-shot draft the next P-1 chunks from their pure noise (draft = eps - v
        at sigma=1); anchor and drafts are committed to the KV cache in order and
        the next anchor conditions on them. This rollout reproduces that loop, so
        the emitted video IS the deployed distribution: the plain DMD loss on it
        trains the backbone (through anchor chunks) and the heads (through draft
        chunks) jointly -- no auxiliary MCP loss, no post-hoc video assembly.

        The backbone denoises only ceil(num_blocks / P) blocks per rollout instead
        of all of them, so this rollout is also cheaper than the vanilla one.
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if self.independent_first_frame or initial_latent is not None:
            raise NotImplementedError(
                "MCP-accelerated rollout currently supports the plain t2v setup "
                "(no independent first frame, no initial latent)."
            )
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
        num_output_frames = num_frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )

        m = self.num_frame_per_block
        period = self.mcp_accel_depths + 1
        anchor_blocks = list(range(0, num_blocks, period))
        all_num_frames = [m] * num_blocks
        block_starts = self._block_start_frames(0, all_num_frames)
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(
            len(anchor_blocks), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21

        def commit_to_cache(latent, start_frame):
            # The same context-noise recache the vanilla rollout does per block
            # (its "Step 3.3"); at deployment every emitted chunk -- anchor or
            # draft -- enters the cache exactly like this.
            context_timestep = torch.ones(
                [batch_size, m], device=noise.device, dtype=torch.int64
            ) * self.context_noise
            noised = self.scheduler.add_noise(
                latent.flatten(0, 1),
                torch.randn_like(latent.flatten(0, 1)),
                context_timestep.flatten(0, 1)
            ).unflatten(0, latent.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=noised,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=start_frame * self.frame_seq_length
                )

        # Zero-valued gradient thread for drafts the chain computed but the
        # deployment does not emit (depths beyond mcp_accel_depths): every MCP
        # parameter must participate in backward on every rank or FSDP's
        # per-parameter reduce hooks stall.
        ghost = None

        for anchor_index, anchor_block in enumerate(anchor_blocks):
            current_start_frame = block_starts[anchor_block]
            noisy_input = noise[:, current_start_frame:current_start_frame + m]

            mcp_noises, mcp_flow_preds = None, None
            # Step 1: denoise the anchor chunk; the heads draft at the exit step,
            # whose backbone features carry the autograd graph.
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[anchor_index])
                timestep = torch.ones(
                    [batch_size, m], device=noise.device, dtype=torch.int64
                ) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * m], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    mcp_noises, mcp_starts = self._mcp_future_chunks(
                        anchor_block, block_starts, all_num_frames, noise, 0
                    )
                    use_grad = current_start_frame >= start_gradient_frame_index
                    with (nullcontext() if use_grad else torch.no_grad()):
                        if any(n is not None for n in mcp_noises):
                            _, denoised_pred, mcp_flow_preds = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                                mcp_future_noises=mcp_noises,
                                mcp_future_start_frames=mcp_starts
                            )
                        else:
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                    break

            # Step 2: place the anchor chunk and its deployed drafts in the video.
            output[:, current_start_frame:current_start_frame + m] = denoised_pred

            draft_latents = []
            if mcp_flow_preds is not None:
                for k, flow_pred in enumerate(mcp_flow_preds):
                    if k >= self.mcp_accel_depths:
                        contribution = flow_pred.sum() * 0.0
                        ghost = contribution if ghost is None else ghost + contribution
                        continue
                    # sigma = 1 one-step draft: x_t = eps, x0 = x_t - sigma * v.
                    draft = mcp_noises[k] - flow_pred
                    draft_start = block_starts[anchor_block + 1 + k]
                    output[:, draft_start:draft_start + m] = draft
                    draft_latents.append((draft_start, draft))

            # Step 3: commit to the KV cache in deployment order -- anchor first,
            # then each draft -- so the next anchor conditions on all of them.
            commit_to_cache(denoised_pred, current_start_frame)
            for draft_start, draft in draft_latents:
                commit_to_cache(draft, draft_start)

        if ghost is not None:
            # Numerically zero; only threads undeployed drafts into the graph.
            output = output + ghost

        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    @staticmethod
    def _block_start_frames(first_start_frame, all_num_frames):
        starts, frame = [], first_start_frame
        for num_frames in all_num_frames:
            starts.append(frame)
            frame += num_frames
        return starts

    def _kv_index_snapshot(self):
        return [
            (
                int(cache["global_end_index"].item()),
                int(cache["local_end_index"].item())
            )
            for cache in self.kv_cache1
        ]

    def _restore_kv_index_snapshot(self, snapshot):
        for cache, (global_end_index, local_end_index) in zip(self.kv_cache1, snapshot):
            cache["global_end_index"].fill_(global_end_index)
            cache["local_end_index"].fill_(local_end_index)

    def _cache_start(self):
        return int(self.kv_cache1[0]["global_end_index"].item())

    def _commit_context_block(self, latent, start_frame, conditional_dict):
        batch_size, current_num_frames = latent.shape[:2]
        context_timestep = torch.ones(
            [batch_size, current_num_frames],
            device=latent.device,
            dtype=torch.int64
        ) * self.context_noise
        context_latent = self.scheduler.add_noise(
            latent.flatten(0, 1),
            torch.randn_like(latent.flatten(0, 1)),
            context_timestep.flatten(0, 1)
        ).unflatten(0, latent.shape[:2])

        with torch.no_grad():
            self.generator(
                noisy_image_or_video=context_latent,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=start_frame * self.frame_seq_length,
                cache_start=self._cache_start()
            )

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        if self.mcp_num_modules > 0:
            return self._inference_with_trajectory_mcp_accelerated(
                noise=noise,
                initial_latent=initial_latent,
                return_sim_step=return_sim_step,
                **conditional_dict
            )

        memory_gap_blocks = self.sample_memory_gap_blocks(noise.device)
        if memory_gap_blocks > 0:
            return self._inference_with_trajectory_memory_gap(
                noise=noise,
                initial_latent=initial_latent,
                return_sim_step=return_sim_step,
                memory_gap_blocks=memory_gap_blocks,
                **conditional_dict
            )

        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        # if self.kv_cache1 is None:
        #     self._initialize_kv_cache(
        #         batch_size=batch_size,
        #         dtype=noise.dtype,
        #         device=noise.device,
        #     )
        #     self._initialize_crossattn_cache(
        #         batch_size=batch_size,
        #         dtype=noise.dtype,
        #         device=noise.device
        #     )
        # else:
        #     # reset cross attn cache
        #     for block_index in range(self.num_transformer_blocks):
        #         self.crossattn_cache[block_index]["is_init"] = False
        #     # reset kv cache
        #     for block_index in range(len(self.kv_cache1)):
        #         self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)
        #         self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21
        block_starts = self._block_start_frames(current_start_frame, all_num_frames)

        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                    if current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                    break

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _inference_with_trajectory_memory_gap(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            memory_gap_blocks: int = 1,
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )

        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21
        block_starts = self._block_start_frames(current_start_frame, all_num_frames)
        pending_context_blocks = []

        for block_index, current_num_frames in enumerate(all_num_frames):
            block_start_frame = current_start_frame
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            kv_snapshot = self._kv_index_snapshot()
            block_cache_start = self._cache_start()

            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            cache_start=block_cache_start
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    if current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                                cache_start=block_cache_start
                            )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            cache_start=block_cache_start
                        )
                    break

            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Remove current-block KV written during denoising from the visible memory.
            self._restore_kv_index_snapshot(kv_snapshot)
            pending_context_blocks.append(
                {
                    "latent": denoised_pred.detach(),
                    "start_frame": block_start_frame,
                }
            )

            while len(pending_context_blocks) > memory_gap_blocks:
                context_block = pending_context_blocks.pop(0)
                self._commit_context_block(
                    latent=context_block["latent"],
                    start_frame=context_block["start_frame"],
                    conditional_dict=conditional_dict
                )

            current_start_frame += current_num_frames

        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
