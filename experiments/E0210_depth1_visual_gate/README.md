# E0210 depth-1 visual gate

This package performs the fixed-teacher-history visual gate for the formal E0209 depth-1 MCP checkpoint.

## Suites

- `regression`: legacy teacher samples 004 and 005; 8 videos total when all four variants are available.
- `full`: the two legacy failures plus eight automatically selected formal-validation samples; 40 videos total.

Formal samples are selected from the E0209 best-validation state metrics at the easiest, p10, p25, median, p75, p90, largest anchor-spread, and worst positions.

## Variants

- `target`: official teacher latent.
- `step32`: E0203 step-32 MCP baseline.
- `e0207c`: previous small-data depth-1 checkpoint.
- `e0209`: formal 2048-prompt depth-1 best checkpoint.

For MCP variants, target blocks 1–4 are replaced, but each draft is computed using the correct teacher-generated history. This is not a closed-loop always-accept test.

## Install

Copy the directory into:

```text
experiments/E0210_depth1_visual_gate/
```

Then:

```bash
chmod +x experiments/E0210_depth1_visual_gate/*.sh
```

## Recommended execution

Run the two legacy failures first:

```bash
GPU=0 bash experiments/E0210_depth1_visual_gate/run_regression.sh
```

Inspect:

```bash
bash experiments/E0210_depth1_visual_gate/inspect.sh regression
```

The main review page is:

```text
experiments/E0210_depth1_visual_gate/regression/review.html
```

After manually confirming the regression samples, run the full suite:

```bash
GPU=0 bash experiments/E0210_depth1_visual_gate/run_full.sh
bash experiments/E0210_depth1_visual_gate/inspect.sh full
```

Package results for upload:

```bash
bash experiments/E0210_depth1_visual_gate/package_results.sh regression
bash experiments/E0210_depth1_visual_gate/package_results.sh full
```

## Decision

- Fixed-history visual failure: do not enter depth-2. Audit the loss target, latent splice semantics, or decoding path.
- Fixed-history visual pass: proceed to E0211 depth-1 always-accept closed-loop evaluation.
