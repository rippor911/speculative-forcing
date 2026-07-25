# Verifier Framework

Milestone F1 adds only the model-free verifier assembly needed to close the
current speculative loop:

```text
CandidateDecoder -> CandidateScorer -> ScoreAggregator
  -> CompositeCandidateEvaluator -> AcceptancePolicy
  -> SpeculativeController
```

`SPEC.md` remains the canonical project specification.

## Boundaries

- `CandidateDecoder` converts one `DraftCandidate` into scorer input. The F1
  identity decoder returns the candidate latent unchanged. It does not score,
  decide, call fallback, call a generator, or mutate KV/output/controller state.
- `CandidateScorer` consumes decoded payloads and returns raw per-frame scores.
  The F1 scripted scorer reads only an explicit `scores_by_depth` table. It does
  not aggregate scores, read thresholds, or decide acceptance.
- `ScoreAggregator` converts finite per-frame scores into one block score. F1
  provides `min_frame` and `mean_frame`. Aggregators do not read thresholds and
  keep no call history.
- `CompositeCandidateEvaluator` runs decoder, scorer, and aggregator in order,
  then returns the existing `Evaluation` object with a `ScoreResult` in
  `Evaluation.value`. It does not decide, fallback, commit, or rollback.
- `FixedThresholdPolicy` reads only `ScoreResult.block_score` and its explicit
  threshold. It accepts exactly when `block_score >= threshold`.

Policy objects must not call generators or fallback because the controller owns
longest-prefix routing and fallback timing. Letting a policy mutate generation
state would bypass rollback and could produce `accept, reject, accept` style
state transitions that the controller is designed to forbid.

The controller does not know ImageReward, VAE, scorer names, or threshold
values. It only calls `evaluator.evaluate(candidate)` and
`policy.decide(evaluation)`.

## Adding Components

To add a scorer, implement `score(decoded) -> RawScoreResult`, validate all
scores as finite numbers, avoid thresholds and generator calls, then register it
in `SCORER_FACTORIES` with an explicit config `type`.

To add an aggregator, implement `aggregate(scores) -> float`, reject empty or
non-finite scores, avoid thresholds and mutable state, then register it in
`AGGREGATOR_FACTORIES`.

Factories consume already parsed mappings only. They do not read YAML files, do
not support dynamic import paths, and fail fast with field paths such as
`speculative.acceptance.threshold`.

## Scope

The F1 fake decoder and scripted scorer only validate composition and routing.
They do not represent real Wan, VAE decode, ImageReward, or learned verifier
quality. The next stages are wiring this framework to the real MCP backend and
scripted parity entrypoint.
