# NF-SF Mainline Guide

This guide records how to use the current NF-SF v1 workspace. It does not redefine the research plan and does not authorize changes to frozen legacy entrypoints.

## 1. Document Authority

- `docs/NF_SF_V1_LOCKED_PLAN.md` is the authority for what the project is doing.
- `docs/NF_SF_MAINLINE_GUIDE.md` is the operational guide for how to use the current workspace.
- Historical experiments under `experiments/` are evidence and provenance, not the mainline definition.
- `SPEC.md` contains older project history; any conflict with the locked plan or this guide should be treated as historical background.

## 2. Current Status Snapshot

- Branch: `next-forcing`
- Code baseline audited for this guide: `7a4796aa33d23d43243f867e20df24eea346bd86`
- The documentation commit may be a descendant of this code baseline. Always inspect the live branch, HEAD, and working tree before acting.
- Current unique mainline: NF-SF v1.
- Reference Teacher: permanently frozen.
- Stage A: formal 0 -> 500 completed.
- Stage B: formal 500 -> 2000 paused.
- Current unique next step: M6 depth-1 parallel multi-step inference.
- Correct M6 requires a new `inference_next_forcing.py`; do not modify or repurpose `inference_mcp.py`.

## 3. Model Roles

- Reference Teacher: the official `self_forcing_dmd.pt` checkpoint used to generate frozen reference latent payloads and reference oracles. It is not trained further.
- NF Main: the trainable main Wan/Self-Forcing generator path in formal NF-SF training.
- MCP1/2/3: future-chunk prediction heads trained jointly with NF Main. Formal Stage A trains all three depths; M6 first evaluates depth-1 parallel inference.
- VAE / text encoder: frozen support components for decoding and prompt conditioning. They are not part of formal Stage A optimization.

## 4. Entrypoint Matrix

| Entrypoint | Classification | Current role |
| --- | --- | --- |
| `inference.py` | REFERENCE ORACLE — CONDITIONAL | Official Self-Forcing pipeline semantics. It has exact Oracle A meaning only under the locked oracle inputs: official `self_forcing_dmd.pt`, locked four-step config, same prompt/conditioning, exact stored teacher `source_noise`, same latent shape/frame/chunk contract, and recorded raw/warped schedules. Using only the same seed is insufficient. The stock `inference.py` CLI output alone is not an exact M6 A/B oracle unless the exact stored teacher `source_noise` and the other locked inputs are supplied through a controlled oracle path. |
| `train.py` | SUPPORT | Upstream training dispatcher. Not the NF-SF v1 formal trainer. |
| `inference_mcp.py` | LEGACY/FROZEN | Historical MCP rollout entry. Hardcodes `ANCHOR_DENOISING_STEPS=[1000]`; one-step rollout only; must not evaluate NF-SF v1. |
| `inference_speculative.py` | LEGACY/FROZEN / SUPPORT | Historical speculative controller entry built on the frozen one-step MCP path. Not an NF-SF v1 quality entry. |
| `scripts/train_nf_sf_m3_overfit.py` | MAINLINE | Current NF-SF formal trainer and M3/M5 orchestration owner. |
| `scripts/eval_nf_sf_m3_overfit.py` | SUPPORT | Local reconstruction, restore, and decode evaluation support for fixed-window checkpoints. |
| `scripts/build_nf_sf_m5_formal_sample_plan.py` | SUPPORT | Builds the locked 2048/256 M5 formal sample plan. |
| `inference_next_forcing.py` | PLANNED / MISSING / MAINLINE | Planned M6 depth-1 parallel multi-step inference entry. This file is intentionally missing at this audited HEAD. |

## 5. Main Data Flows

### Reference Teacher Generation

```text
formal sample record
 -> prompt conditional + seeded source noise
 -> frozen official Self-Forcing teacher
 -> four-step warped reference rollout
 -> teacher payload with source_noise, target_latent, schedules, valid_anchor_blocks
 -> teacher manifest
```

The Reference Teacher is frozen permanently. `experiments/E0208C_teacher_rollout_formal/generate_teacher_shard.py` writes 21 latent frames with 3 latent frames per chunk and records `valid_anchor_blocks=[0,1,2,3]`.

Known material gap: `experiments/E0208C0_teacher_writer_reproduction/reproduce_teacher.py` imports `experiments/E0202_anchor_replay_audit/audit_replay.py`, but that file is absent in the current checkout. This is documented here and is not fixed by this documentation task.

### Formal Stage A Training

```text
formal sample plan identity
 -> lazy teacher payload
 -> fixed selected window
 -> lazy conditional artifact
 -> independent main/MCP noise and timestep sampling
 -> generator forward with mcp_timesteps
 -> independent Flow Matching losses
 -> optimizer step
 -> streaming validation and formal checkpoint
```

Formal Stage A is teacher-forced fixed-window Flow Matching. It is not the old DMD rollout. The fixed formal selection is:

```text
history 0:3
current 3:6
next1   6:9
next2   9:12
next3   12:15
```

### Local Reconstruction Evaluation

```text
M3/M5 checkpoint + selected teacher sample
 -> restore generator and probe contract
 -> reconstruct main current over the teacher solver schedule
 -> reconstruct MCP1 next over the solver schedule
 -> optional VAE decode for local visual inspection
```

This path evaluates fixed-window reconstruction and restore behavior. It is not a deployment inference path.

### Legacy `inference_mcp.py` Rollout

```text
config + checkpoint + prompt
 -> text encoder
 -> SelfForcingTrainingPipeline([1000], last_step_only=True)
 -> vanilla one-step x0 anchors or MCP one-step x0 drafts
 -> commit anchor/drafts
 -> decode video
```

`inference_mcp.py` is frozen. It hardcodes `ANCHOR_DENOISING_STEPS=[1000]`, ignores `config.denoising_step_list`, and must not be used to evaluate NF-SF v1.

### Intended M6 Multi-Step Rollout

```text
config + NF-SF checkpoint + prompt
 -> inference_next_forcing.py
 -> build aligned main and MCP solver schedules
 -> maintain current_state and next_state
 -> every denoising timestep updates both states
 -> prevent noisy intermediate KV cache commits
 -> commit clean current, then clean next
 -> advance by i+2 chunks
```

M6 must test depth-1 parallel multi-step inference before Stage B resumes.

#### Main/MCP Schedule Alignment

M6 uses raw denoising indices:

```text
[1000, 750, 500, 250]
```

The main scheduler uses shift 5, producing approximately:

```text
[1000, 937.5, 833.3333, 625.0]
```

The MCP scheduler uses shift 10, producing approximately:

```text
[1000, 967.7419, 909.0909, 769.2308]
```

Schedule alignment is by the same raw timestep index. The implementation should compute and record these values through the scheduler rather than scattering the floating-point constants through the code.

Forbidden M6 schedule shortcuts:

- Do not pass the main warped timestep directly to MCP.
- Do not let `current_state` and `next_state` share one warped schedule.
- Do not leave MCP at the default timestep 1000 during multi-step rollout.

Each raw timestep index has the following contract:

```text
main_t = main_schedule[index]
mcp_t  = mcp_schedule[index]

main predicts current flow at main_t
MCP1 predicts next flow at mcp_t

current_state advances with the main scheduler
next_state advances with the MCP scheduler
```

These state transitions must reuse the controlled Reference Oracle update recipe, including the exact rollout RNG and per-transition re-noise semantics where applicable. M6 must not replace that recipe with an implementation-defined generalized scheduler update unless an oracle explicitly proves equivalence.

#### Intermediate KV Visibility Contract

The minimum M6 KV lifecycle contract is:

```text
before every denoising forward, including the fourth/final solver forward:
    snapshot the visible KV boundaries

run the current/MCP forward

after every denoising forward:
    restore the visible KV boundaries

update current_state and next_state with their schedulers

do not expose the noisy solver-forward write as committed history
```

Only after all four forward/update pairs finish may the runtime permanently recache:

```text
permanently recache clean current
permanently recache clean next
```

The final clean commit order is:

```text
current -> next
```

Then the next round advances by two chunks.

Generator forward with KV cache may write cache contents and advance `global_end_index` / `local_end_index`. Noisy intermediate forwards must not become permanent history. Index snapshot/restore is the current minimal implementation candidate, but static audit cannot prove index-only restore is safe under every local-attention overwrite or eviction case. The real-model M6 oracles must validate this assumption before it becomes a confirmed implementation contract.

#### First-Block Policy

The first-block policy is unresolved.

Formal Stage A always trained a current chunk with a non-empty clean-history window and did not train an empty-history MCP case. M6 must not silently enable MCP for block 0. A main-only bootstrap is a candidate design, not yet a validated conclusion. Any first-block MCP deployment requires a dedicated oracle or matching training coverage.

## 6. Checkpoints and Data Roles

- Official `self_forcing_dmd.pt`: frozen Reference Teacher checkpoint and initialization source. It is not an NF-SF v1 evaluation result.
- Teacher payload: torch-save dict containing source noise, target latent, schedules, and provenance. It provides fixed teacher windows for formal training and validation.
- Formal sample plan: locked 2048 train / 256 validation plan. Its identity currently includes sample index/id/split/split index/prompt hash and does not include a temporal anchor. `valid_anchor_blocks` is not part of formal sample identity.
- Step0 checkpoint: formal initialization checkpoint used by zero-overhead and resume/probe oracles.
- Step500 checkpoint: completed Stage A checkpoint and the parent for Stage B when Stage B resumes.
- Server-specific temporary output directories are not permanent contracts and should not be written into mainline docs as required paths.

## 7. M6 Oracles

### Common Locked Inputs

Oracle A/B/C/D must use the same:

- teacher payload identity
- exact stored `source_noise`, not only the same seed
- rollout_seed or restored global rollout RNG state
- exact per-transition re-noise recipe
- RNG draw ordering
- context-recache RNG-consumption contract
- prompt
- prompt conditioning / embedding contract
- conditional artifact or prompt-embedding hash
- latent shape
- latent frame count
- chunk size
- dtype
- device/runtime contract
- raw denoising-step indices
- context-noise / clean-recache contract
- config provenance
- code Git SHA

Each oracle must use and record its oracle-specific locked checkpoint SHA256:

- A: official reference checkpoint SHA256
- B: formal step0 checkpoint SHA256
- C/D: formal step500 checkpoint SHA256

C and D must use the same step500 checkpoint. Other Common Locked Inputs remain shared across A/B/C/D.

Every oracle output must record:

- raw schedule
- main warped schedule
- MCP warped schedule, if MCP is enabled
- source-noise identity/hash
- rollout seed / initial RNG-state hash
- transition-noise hashes, or a reproducible draw contract
- prompt embedding / conditional artifact hash
- checkpoint hash
- commit events
- KV visible-boundary events
- output latent/video artifact hashes

Server-specific temporary output directories are not part of the mainline contract.

- A. official reference 4-step: a controlled Oracle-A harness reusing `inference.py` / `CausalInferencePipeline` semantics, with the exact stored `source_noise` and locked rollout RNG contract. The stock CLI alone is insufficient.
- B. step0 zero-overhead 4-step: new M6 entry with MCP disabled, full four-step main denoising, and step0 weights.
- C. step500 zero-overhead 4-step: new M6 entry with MCP disabled, full four-step main denoising, and the completed Stage A step500 checkpoint.
- D. step500 depth1 parallel 4-step: new M6 entry with MCP1 enabled, aligned multi-step current/next denoising, clean current then clean next commit order, and i+2 advancement.

### Oracle Pass Criteria

The oracles separate protocol correctness from visual quality.

#### Oracle A: Official Reference Four-Step

`protocol_pass` requires at least:

- official reference checkpoint
- exact stored source_noise
- complete four-step raw/warped schedule
- four denoising updates per chunk
- final clean recache
- MCP disabled
- finite outputs
- complete provenance

Oracle A should be compared against the teacher payload target latent under controlled conditions. The numeric tolerance must be explicitly recorded by the oracle implementation and report; this guide does not hardcode an unverified tolerance. A normal stock CLI run that successfully generates a video is not an Oracle A pass.

#### Oracle B: Step0 Zero-Overhead Four-Step

`protocol_pass` requires at least:

- the same Common Locked Inputs
- step0 checkpoint
- MCP disabled
- complete four-step main denoising
- schedule, recache, and commit semantics matching Oracle A
- no MCP calls
- finite outputs
- comparison with Oracle A within explicitly reported bf16 tolerance

If Oracle B cannot reproduce Oracle A within the controlled tolerance, treat the failure first as an M6 entry or checkpoint-restore problem and do not proceed to Oracle C/D.

#### Oracle C: Step500 Zero-Overhead Four-Step

Record separate `protocol_pass` and `main_quality_pass` fields.

`protocol_pass` requires at least:

- step500 checkpoint
- MCP disabled
- complete four-step main denoising
- the same Common Locked Inputs as A/B
- rollback after every denoising forward, including the final solver forward, before any clean recache
- correct clean recache
- no MCP calls
- finite outputs

The quality gate must compare step500 against step0 in both latent and video space, report the differences, and declare the quality criterion in the experiment contract before the real run. A generated MP4 alone is not a quality pass. This guide does not claim from static audit that the step500 backbone has degraded or not degraded.

#### Oracle D: Step500 Depth1 Parallel Four-Step

Record separate `protocol_pass`, `visual_quality_pass`, and `runtime_measurement_status` fields.

`protocol_pass` requires at least:

- only MCP depth1 deployed
- `current_state` executes four main scheduler updates
- `next_state` executes four MCP scheduler updates
- main/MCP schedules aligned by the same raw timestep index with different warped schedules
- rollback after every denoising forward, including the final solver forward, before any clean recache
- noisy solver-forward KV writes remain invisible
- final clean commit order is current then next
- the next round advances by two chunks
- the runtime does not recompute an already accepted next chunk with main
- trace proves schedule, KV boundary, commit order, and cursor step by step
- finite outputs

`visual_quality_pass` separately checks MCP next-chunk clarity, chunk-boundary continuity, the stability of later anchors conditioned on MCP history, and long-trajectory drift. Protocol pass does not imply visual quality pass. M6 does not evaluate verifier, accept/reject routing, or refinement.

## 8. Confirmed and Open Issues

Confirmed implementation mismatch:

- `inference_mcp.py` is a historical one-step rollout path, not the M6 path.
- `inference_mcp.py` hardcodes `ANCHOR_DENOISING_STEPS=[1000]`.
- `inference_mcp.py` does not use `config.denoising_step_list`.
- Therefore `inference_mcp.py` cannot evaluate NF-SF v1.

Confirmed fixed-window coverage gap:

- Formal selection always uses history 0:3, current 3:6, next1 6:9, next2 9:12, next3 12:15.
- `valid_anchor_blocks` is recorded by teacher generation but does not enter formal sample identity.
- Formal train/validation currently uses a fixed teacher-forced window.

Likely history distribution gap:

- M6 will consume self-generated clean history during rollout, while Stage A training used teacher-forced history.
- This is a likely train-test history gap and must be verified by M6 oracles. It must not be written as a proven failure before those oracles run.

Unsupported conclusions:

- Visual quality, speedup, long-video stability, and first-block MCP safety are not established by static code audit.
- The first-block policy is unresolved; main-only bootstrap is a candidate, and block-0 MCP needs its own oracle or matching training coverage.
- Stage B must not resume merely because Stage A exists; M6 oracle evidence is required first.
- Missing `experiments/E0202_anchor_replay_audit/audit_replay.py` is a known material gap, not a reason to modify historical experiment directories in this task.

## 9. Local and Server Workflow

- Local workspace: code/document edits, review, static checks, and commits only when explicitly authorized.
- Server workspace: `git pull --ff-only`, tests, training, checkpoint generation, and video generation.
- Do not use `git add .`.
- Do not use `git reset`.
- Do not commit, push, or merge without explicit authorization.
- Do not rewrite or move historical experiment directories to make the current mainline look cleaner.

## 10. Workspace Hygiene

- `.mypy_cache/` is local-only cache and should stay out of Git.
- `.pytest_cache/` and `pytest-cache-files-*/` are local-only pytest cache/temp directories and should stay out of Git.
- `codex.log` is local session memory and should stay out of Git.
- `project_structure.txt` is local workspace inspection output and should stay out of Git.
- Large artifacts, checkpoints, decoded videos, profiler logs, tensor dumps, and server run outputs must not enter Git.
- Before deleting any local path, first confirm it is not tracked.
- Do not move or flatten `experiments/` history directories.
- Do not add broad ignore rules such as `*.json`, `*.pt`, or `*.mp4` for test fixture convenience. Existing historical ignore rules should be reviewed carefully before expansion.

## 11. New Session Reading Order

1. `docs/NF_SF_V1_LOCKED_PLAN.md`
2. `docs/NF_SF_MAINLINE_GUIDE.md`
3. Current `git branch --show-current`, `git rev-parse HEAD`, and `git status --short`
4. Current experiment checkpoint and artifact contract for the requested task

## 12. Stop Rules

- Do not enter Stage B before the M6 Oracle Gate issues an explicit GO decision.
- Stage B remains paused until: Oracle A and B protocol/reproduction gates pass; Oracle C protocol gate passes; Oracle D protocol gate passes; and Oracle C main-quality plus Oracle D visual-quality reviews issue an explicit GO decision. A quality FAIL does not permit Stage B under this guide unless the locked plan is explicitly revised. A generated MP4 alone is not an oracle pass. Runtime must be recorded, but speedup must not be claimed before the protocol gates pass.
- Do not use `inference_mcp.py` to evaluate NF-SF v1.
- Do not add verifier, DMD, self-rollout, direct history attention, refinement, or expanded data before M6 evidence requires it.
- Do not perform large repository refactors for M6.
- Do not modify `inference_mcp.py`; it is frozen legacy evidence.
