from utils.distributed import launch_distributed_job
from utils.scheduler import FlowMatchScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from utils.dataset import TextDataset
import torch.distributed as dist
from tqdm import tqdm
import argparse
import torch
import math
import os


def init_model(device, timestep_shift):
    # What timestep_shift actually controls here is the LOCAL scheduler below: it sets
    # which timesteps the 48-step teacher ODE visits, i.e. how the steps are distributed
    # over the noise range. That is the value which must match the config consuming these
    # pairs (configs/ode_init.yaml) -- see the note there.
    #
    # It is also passed to the wrapper for consistency, but that matters far less than it
    # looks: scheduler.py:133 defines timesteps = sigmas * 1000, so looking a sigma up by
    # timestep returns ~t/1000 under any shift, and the flow->x0->flow round trip between
    # WanDiffusionWrapper.forward and _convert_x0_to_flow_pred cancels the sigma anyway.
    # Passing it just removes a ~1e-4 table-quantisation discrepancy.
    #
    # Upstream constructed this bare, inheriting the wrapper's 8.0 default
    # (utils/wan_wrapper.py:140) -- Wan's default rather than a deliberate choice. Every
    # Self-Forcing config trains at 5.0, so the default here is 5.0 to keep ODE init and
    # DMD on one schedule. Pass --timestep_shift 8.0 to reproduce upstream's schedule.
    model = WanDiffusionWrapper(timestep_shift=timestep_shift).to(device).to(torch.float32)
    encoder = WanTextEncoder().to(device).to(torch.float32)
    model.model.requires_grad_(False)

    scheduler = FlowMatchScheduler(
        shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(num_inference_steps=48, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)

    sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

    unconditional_dict = encoder(
        text_prompts=[sample_neg_prompt]
    )

    return model, encoder, scheduler, unconditional_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--caption_path", type=str)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument(
        "--timestep_shift", type=float, default=5.0,
        help="Noise-schedule shift for the teacher ODE. MUST equal timestep_shift in the "
             "config that trains on these pairs (configs/ode_init.yaml). Defaults to 5.0 to "
             "match every Self-Forcing config; upstream left this at the wrapper's 8.0 default."
    )

    args = parser.parse_args()

    # launch_distributed_job()
    launch_distributed_job()

    device = torch.cuda.current_device()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, encoder, scheduler, unconditional_dict = init_model(
        device=device, timestep_shift=args.timestep_shift)

    if dist.get_rank() == 0:
        print(f"[ode-pairs] timestep_shift={args.timestep_shift} guidance_scale={args.guidance_scale} "
              f"-> configs/ode_init.yaml MUST set timestep_shift: {args.timestep_shift}")

    dataset = TextDataset(args.caption_path)

    # if global_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

    for index in tqdm(range(int(math.ceil(len(dataset) / dist.get_world_size()))), disable=dist.get_rank() != 0):
        prompt_index = index * dist.get_world_size() + dist.get_rank()
        if prompt_index >= len(dataset):
            continue
        prompt = dataset[prompt_index]

        conditional_dict = encoder(text_prompts=prompt)

        latents = torch.randn(
            [1, 21, 16, 60, 104], dtype=torch.float32, device=device
        )

        noisy_input = []

        for progress_id, t in enumerate(tqdm(scheduler.timesteps)):
            timestep = t * \
                torch.ones([1, 21], device=device, dtype=torch.float32)

            noisy_input.append(latents)

            _, x0_pred_cond = model(
                latents, conditional_dict, timestep
            )

            _, x0_pred_uncond = model(
                latents, unconditional_dict, timestep
            )

            x0_pred = x0_pred_uncond + args.guidance_scale * (
                x0_pred_cond - x0_pred_uncond
            )

            flow_pred = model._convert_x0_to_flow_pred(
                scheduler=scheduler,
                x0_pred=x0_pred.flatten(0, 1),
                xt=latents.flatten(0, 1),
                timestep=timestep.flatten(0, 1)
            ).unflatten(0, x0_pred.shape[:2])

            latents = scheduler.step(
                flow_pred.flatten(0, 1),
                scheduler.timesteps[progress_id] * torch.ones(
                    [1, 21], device=device, dtype=torch.long).flatten(0, 1),
                latents.flatten(0, 1)
            ).unflatten(dim=0, sizes=flow_pred.shape[:2])

        noisy_input.append(latents)

        noisy_inputs = torch.stack(noisy_input, dim=1)

        noisy_inputs = noisy_inputs[:, [0, 12, 24, 36, -1]]

        stored_data = noisy_inputs

        torch.save(
            {prompt: stored_data.cpu().detach()},
            os.path.join(args.output_folder, f"{prompt_index:05d}.pt")
        )

    dist.barrier()


if __name__ == "__main__":
    main()
