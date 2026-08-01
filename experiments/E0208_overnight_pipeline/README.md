# E0208 overnight pipeline

This bundle performs one uninterrupted pipeline:

1. Resume-safe four-GPU generation of 2,304 official Self-Forcing teacher samples.
2. Validation and merge of 36 shard manifests into one training manifest.
3. Fresh MCP depth-1 training using the established E0207A trainer for three deterministic epochs.

Install the folder at:

`experiments/E0208_overnight_pipeline/`

Run from the repository root inside one tmux session:

```bash
chmod +x experiments/E0208_overnight_pipeline/*.sh
bash experiments/E0208_overnight_pipeline/run_overnight.sh
```

Detach with `Ctrl-b d`. Inspect later with:

```bash
bash experiments/E0208_overnight_pipeline/inspect_results.sh
```

The generator is resume-safe. Re-running the overnight script reuses completed teacher shards. The depth-1 trainer starts fresh if a prior nonempty E0209 output directory exists; the old directory is archived with a timestamp.
