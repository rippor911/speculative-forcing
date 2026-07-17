<p align="center">
<h1 align="center">Self Forcing</h1>
<h3 align="center">Bridging the Train-Test Gap in Autoregressive Video Diffusion</h3>
</p>
<p align="center">
  <p align="center">
    <a href="https://www.xunhuang.me/">Xun Huang</a><sup>1</sup>
    ·
    <a href="https://zhengqili.github.io/">Zhengqi Li</a><sup>1</sup>
    ·
    <a href="https://guandehe.github.io/">Guande He</a><sup>2</sup>
    ·
    <a href="https://mingyuanzhou.github.io/">Mingyuan Zhou</a><sup>2</sup>
    ·
    <a href="https://research.adobe.com/person/eli-shechtman/">Eli Shechtman</a><sup>1</sup><br>
    <sup>1</sup>Adobe Research <sup>2</sup>UT Austin
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2506.08009">Paper</a> | <a href="https://self-forcing.github.io">Website</a> | <a href="https://huggingface.co/gdhe17/Self-Forcing/tree/main">Models (HuggingFace)</a></h3>
</p>

---

Self Forcing trains autoregressive video diffusion models by **simulating the inference process during training**, performing autoregressive rollout with KV caching. It resolves the train-test distribution mismatch and enables **real-time, streaming video generation on a single RTX 4090** while matching the quality of state-of-the-art diffusion models.

---


https://github.com/user-attachments/assets/7548c2db-fe03-4ba8-8dd3-52d2c6160739


## Requirements
We tested this repo on the following setup:
* Nvidia GPU with at least 24 GB memory (RTX 4090, A100, and H100 are tested).
* Linux operating system.
* 64 GB RAM.

Other hardware setup could also work but hasn't been tested.

## Installation
Create a conda environment and install dependencies:
```
conda create -n self_forcing python=3.10 -y
conda activate self_forcing
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
python setup.py develop
```

## Quick Start
### Download checkpoints
```
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

### CLI Inference
Note: **Our model works better with long, detailed prompts** since it's trained with such prompts. We will integrate prompt extension into the codebase (similar to [Wan2.1](https://github.com/Wan-Video/Wan2.1/tree/main?tab=readme-ov-file#2-using-prompt-extention)) in the future. For now, it is recommended to use third-party LLMs (such as GPT-4o) to extend your prompt before providing to the model.

Example inference script using the chunk-wise autoregressive checkpoint trained with DMD:
```
python inference.py \
    --config_path configs/self_forcing_dmd.yaml \
    --output_folder videos/self_forcing_dmd \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --use_ema
```
Other config files and corresponding checkpoints can be found in [configs](configs) folder and our [huggingface repo](https://huggingface.co/gdhe17/Self-Forcing/tree/main/checkpoints).

## Training
### Download text prompts and ODE initialized checkpoint
```
huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .
huggingface-cli download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts
```
Note: Our training algorithm (except for the GAN version) is data-free (**no video data is needed**). For now, we directly provide the ODE initialization checkpoint and will add more instructions on how to perform ODE initialization in the future (which is identical to the process described in the [CausVid](https://github.com/tianweiy/CausVid) repo).

### ODE initialization from scratch (reconstructed)

Upstream never shipped these instructions or a config. `configs/ode_init.yaml` reconstructs
them from `scripts/generate_ode_pairs.py`, `model/ode_regression.py`, and CausVid's
`configs/wan_causal_ode.yaml`. Read the header comment in that file before running — it
documents a silent-failure trap.

This stage regresses a **causal 4-step student** onto the endpoint of the **bidirectional**
base model's 48-step CFG ODE trajectory, doing bidirectional→causal and 48→4 steps at once.
It is required, not an optional warmup: CausVid's Table 4 ablation reports frame quality
**64.4 with ODE init vs 48.1 without**.

```bash
# 1) Generate ODE trajectory pairs from the bidirectional base model.
#    The paper uses 16k pairs; CausVid used 1.5K.
torchrun --nproc_per_node=8 scripts/generate_ode_pairs.py \
    --output_folder ode_pairs/ \
    --caption_path prompts/vidprom_filtered_extended.txt \
    --guidance_scale 6.0 --timestep_shift 5.0

# 2) Convert to LMDB (ODERegressionLMDBDataset only reads LMDB).
python scripts/create_lmdb_iterative.py --data_path ode_pairs/ --lmdb_path ode_lmdb/

# 3) Causal ODE regression. CausVid trains this for 3000 iterations
#    (their released checkpoint is named ..._checkpoint_model_003000).
#    There is no max_iters key — stop the run yourself at ~3000.
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
    --rdzv_backend=c10d --rdzv_endpoint $MASTER_ADDR \
    train.py --config_path configs/ode_init.yaml \
    --logdir logs/ode_init --disable-wandb --no_visualize
```

**`--timestep_shift` and the config's `timestep_shift` must agree.** `model/ode_regression.py`
gathers `ode_latent[:, i]` and labels it `denoising_step_list[i]`, so the saved trajectory
points and the warped `denoising_step_list` must be the same noise levels. Nothing checks
this — a mismatch just teaches the model the wrong timestep for every sample. At 5.0 both
sides are `[1000.0, 937.5, 833.3, 625.0]`.

We pin **5.0** rather than upstream's 8.0. Upstream built the teacher as a bare
`WanDiffusionWrapper()`, inheriting the wrapper's 8.0 default (`utils/wan_wrapper.py:140`) —
Wan's default, not a deliberate choice — while every Self-Forcing config trains at 5.0. Using
5.0 keeps ODE init and DMD on one schedule.

Upstream never shipped an ODE config, so which shift produced the released `ode_init.pt` is
unknown. If it was 8.0, their pipeline simply had a cross-stage shift and evidently worked, so
8.0 is not wrong — 5.0 is just more consistent when the checkpoint exists to feed Self-Forcing.
The switch is low-risk: both are 48-step solves of the same ODE from the same noise, and shift
only redistributes step placement (which matters for few-step sampling, not a 48-step solve).
5.0 spends more steps at low noise than 8.0 (8 vs 5 below t=500), which if anything sharpens
the endpoint this stage regresses onto.

Gradient accumulation is rejected on this stage (`trainer/ode.py:98`), so scale batch size via
GPU count only.

### Self Forcing Training with DMD
```
torchrun --nnodes=8 --nproc_per_node=8 --rdzv_id=5235 \
  --rdzv_backend=c10d \
  --rdzv_endpoint $MASTER_ADDR \
  train.py \
  --config_path configs/self_forcing_dmd.yaml \
  --logdir logs/self_forcing_dmd \
  --disable-wandb
```
Our training run uses 600 iterations and completes in under 2 hours using 64 H100 GPUs. By implementing gradient accumulation, it should be possible to reproduce the results in less than 16 hours using 8 H100 GPUs.

## Acknowledgements
This codebase is built on top of the open-source implementation of [CausVid](https://github.com/tianweiy/CausVid) by [Tianwei Yin](https://tianweiy.github.io/) and the [Wan2.1](https://github.com/Wan-Video/Wan2.1) repo.

## Citation
If you find this codebase useful for your research, please kindly cite our paper:
```
@article{huang2025selfforcing,
  title={Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion},
  author={Huang, Xun and Li, Zhengqi and He, Guande and Zhou, Mingyuan and Shechtman, Eli},
  journal={arXiv preprint arXiv:2506.08009},
  year={2025}
}
```
