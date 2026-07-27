# F3 Standalone Speculative Inference — Real GPU Validation

## Status

Milestone F3 scripted GPU validation: **complete**.

Validated commit:

```text
01d83710b7e4ef6eb5d0a8b8706bc61ca5027a9f
```

Branch:

```text
feat/speculative-inference
```

This validation covers functional execution, control-flow semantics, commit ordering,
same-noise fallback metadata, final visible KV indices, output generation, and GPU
memory release.

It does not establish elementwise latent/KV equality, quality equivalence, or
production speedup.

## Environment

```text
GPU: 1 × NVIDIA A100-SXM4-80GB
Available GPUs: 4
PyTorch: 2.5.1+cu124
Python environment:
  /home/dataset-assist-0/luojy/efficiency/rippor/envs/speculative-forcing/bin/python

Config:
  configs/self_forcing_dmd_mcp.yaml

Checkpoint:
  checkpoints/self_forcing_dmd_plus_mcp_ode75.pt

Checkpoint restore:
  MCP_COMPLETE_STRICT_RESTORE

MCP tensor count:
  172
```

All runs used:

```text
prompt: A cat walking on grass.
seed: 0
fps: 16
device: cuda:0
resolution: 832 × 480
latent frames per block: 3
```

## 1. Always-accept parity

### Reference

```text
entry: inference_mcp.py
num_frames: 6
mcp_depth: 1
mode: mcp
policy: always_accept
elapsed: 120.68 s
peak GPU memory: 17289 MiB
exit code: 0
```

### F3 standalone path

```text
entry: inference_speculative.py
num_frames: 6
mcp_depth: 1
policy: always_accept
elapsed: 106.46 s
peak GPU memory: 19819 MiB
exit code: 0
```

### Control-flow result

```text
anchor block: 0
accepted draft: block 1 / depth 1
committed blocks: [0, 1]
fallback: none
invalidated: none
next anchor: none
final KV global/local: 9360 / 9360
reference RNG mode: frozen_mcp_always_accept
reference RNG draw count: 1
rollout setup flags: [0]
```

Control-flow, commit ordering, visible KV indices, and RNG setup metadata match the
frozen MCP reference semantics.

### Output comparison

Frozen reference vs F3 always-accept:

```text
average PSNR: 33.636183 dB
average SSIM: 0.954642
```

Frozen reference repeated with the same prompt and seed:

```text
average PSNR: 33.845863 dB
average SSIM: 0.958751
```

The F3/reference video difference is of the same order as the frozen reference's
own repeated-run variation in the current CUDA environment.

MP4 SHA256 and decoded frame hashes are not identical, including between two
frozen-reference runs. Therefore byte equality is not used as the pass criterion.

## 2. Always-reject parity

### Vanilla reference

```text
entry: inference_mcp.py --disable_mcp
num_frames: 6
mode: vanilla
effective mcp_depth: 0
elapsed: 125.62 s
exit code: 0
```

### F3 standalone path

```text
entry: inference_speculative.py
num_frames: 6
mcp_depth: 1
policy: always_reject
elapsed: 120.66 s
peak GPU memory: 19819 MiB
exit code: 0
```

### Control-flow result

```text
anchor block: 0
rejected draft: block 1 / depth 1
fallback block: 1
fallback source-noise reuse metadata: true
commits:
  block 0: anchor
  block 1: fallback
invalidated: none
committed blocks: [0, 1]
next anchor: none
final KV global/local: 9360 / 9360
reference RNG mode: vanilla_target_only
reference RNG draw count: 2
rollout setup flags: [0, 0]
```

This matches the expected target-only semantics for two blocks.

### Output comparison

Vanilla reference vs F3 always-reject:

```text
average PSNR: 34.296667 dB
average SSIM: 0.953421
```

This is also of the same order as the repeated frozen-reference variation.

## 3. Reject-at-depth control-flow validation

Run configuration:

```text
entry: inference_speculative.py
num_frames: 12
mcp_depth: 3
policy: reject_at_depth
reject_depth: 2
elapsed: 127.05 s
peak GPU memory: 23209 MiB
exit code: 0
```

Observed first window:

```text
anchor: block 0
accept: block 1 / depth 1
reject: block 2 / depth 2
fallback: block 2
fallback source-noise reuse metadata: true
invalidate: block 3 / depth 3
commits: [block 0 anchor, block 1 draft, block 2 fallback]
next anchor: block 3
KV after window: 14040 / 14040
```

Observed second window:

```text
anchor: block 3
max depth: 0
commits: [block 3 anchor]
next anchor: none
```

Final state:

```text
committed blocks: [0, 1, 2, 3]
commit start frames: [0, 3, 6, 9]
final KV global/local: 18720 / 18720
expected KV index: 18720
```

All automated semantic checks passed:

```text
two_windows: PASS
final_commits_0_1_2_3: PASS
first_anchor_0: PASS
accept_depth_1: PASS
reject_depth_2: PASS
fallback_block_2: PASS
invalidate_block_3: PASS
next_anchor_block_3: PASS
second_anchor_block_3: PASS
final_kv_matches_expected: PASS
overall: PASS
```

## GPU memory cleanup

All F3 runs returned resident GPU memory to approximately zero after process exit.

Recorded peak values are end-to-end smoke diagnostics and include model loading and
other components. They are not backend-only memory measurements.

## Conclusions

F3 standalone speculative inference passed real A100 functional and control-flow
validation for:

```text
always_accept
always_reject
reject_at_depth
```

Validated properties include:

- strict MCP checkpoint restore;
- real MCP proposal execution;
- longest-prefix acceptance;
- first-rejection termination;
- deeper-draft invalidation;
- same-noise fallback metadata;
- dynamic next-uncommitted anchor;
- commit ordering;
- final visible KV indices;
- final VAE decode and MP4 output;
- process-level GPU memory release.

## Remaining limitations

This validation does not prove:

- elementwise equality of latent tensors;
- elementwise equality or fingerprint equality of KV tensors;
- deterministic CUDA execution;
- quality equivalence;
- inference speedup;
- production latency;
- MCP depth 2/3 output parity against a dedicated frozen oracle;
- VAE candidate-decode transaction correctness;
- ImageReward or learned-verifier correctness.

The observed wall-clock times include model initialization, checkpoint loading,
text encoding, rollout, VAE decoding, and video encoding. They must not be used
as formal performance results.

Formal timing must exclude model loading and warmup and must report repeated-run
means and standard deviations.

## Artifacts

Validation outputs are stored locally under:

```text
f3_parity_01d8371/
```

This directory contains MP4 files, traces, logs, frame hashes, and metric files.
It is intentionally untracked and must not be committed.
