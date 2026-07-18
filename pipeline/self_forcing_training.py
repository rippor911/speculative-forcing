from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
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
        # Side channel for the MCP predictions made during the last rollout. The
        # rollout's return arity is consumed in several places (and varies with
        # return_sim_step), so predictions are handed to DMD.compute_mcp_loss via
        # this attribute instead. Reset at the top of every rollout so a stale
        # record can never leak into a later training step.
        self.last_mcp_records = []

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

    def _record_mcp(self, mcp_flow_preds, mcp_noises, mcp_starts):
        """Stash one block's MCP predictions with everything the loss will need.

        The raw `noise` is kept rather than a ready-made target because at sigma=1
        the flow target is exactly `noise - x0_teacher`, and x0_teacher does not
        exist until after the rollout, when DMD runs the frozen teacher.

        `mcp_flow_preds` is only as long as the number of modules that actually ran
        (the chain stops at the first None), so zip truncates to the right pairing.
        """
        for depth, (flow_pred, chunk_noise, start) in enumerate(
                zip(mcp_flow_preds, mcp_noises, mcp_starts)):
            self.last_mcp_records.append({
                "flow_pred": flow_pred,
                "noise": chunk_noise,
                "start_frame": start,
                "num_frames": chunk_noise.shape[1],
                # 0-based MCP depth (0 == next^1), for the per-depth loss weights.
                "depth": depth,
            })

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
        # Clear before dispatching so both rollout paths start from a clean slate and
        # a previous step's predictions can never be consumed by this step's loss.
        self.last_mcp_records = []

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
                        # Drafts are consumed by BOTH rollout owners: the generator's
                        # MCP loss substitutes them into the video for distribution
                        # matching, and the critic trains its fake score on the same
                        # hybrid construction (model/dmd.py compute_mcp_loss /
                        # critic_loss). Under the critic's no_grad rollout the head
                        # forward is cheap and its records come out detached, which
                        # is exactly what the critic diet needs.
                        run_mcp = self.mcp_num_modules > 0
                        mcp_noises, mcp_starts = self._mcp_future_chunks(
                            block_index, block_starts, all_num_frames, noise, num_input_frames
                        ) if run_mcp else (None, None)

                        if mcp_noises is not None and any(n is not None for n in mcp_noises):
                            # MCP only rides the gradient-carrying forward: this is the
                            # single step per block whose backbone features are attached
                            # to the autograd graph, which is the whole point of the
                            # auxiliary supervision.
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
                            self._record_mcp(mcp_flow_preds, mcp_noises, mcp_starts)
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
                        # Drafts are consumed by BOTH rollout owners: the generator's
                        # MCP loss substitutes them into the video for distribution
                        # matching, and the critic trains its fake score on the same
                        # hybrid construction (model/dmd.py compute_mcp_loss /
                        # critic_loss). Under the critic's no_grad rollout the head
                        # forward is cheap and its records come out detached, which
                        # is exactly what the critic diet needs.
                        run_mcp = self.mcp_num_modules > 0
                        mcp_noises, mcp_starts = self._mcp_future_chunks(
                            block_index, block_starts, all_num_frames, noise, num_input_frames
                        ) if run_mcp else (None, None)

                        if mcp_noises is not None and any(n is not None for n in mcp_noises):
                            _, denoised_pred, mcp_flow_preds = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                                cache_start=block_cache_start,
                                mcp_future_noises=mcp_noises,
                                mcp_future_start_frames=mcp_starts
                            )
                            self._record_mcp(mcp_flow_preds, mcp_noises, mcp_starts)
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
