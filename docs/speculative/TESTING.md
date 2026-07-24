# Speculative Core Testing

The tests are pure `unittest` tests and do not load Wan, CUDA, checkpoints,
datasets, ImageReward, or videos.

Run:

```bash
python -m unittest discover -s tests/speculative -p "test_*.py" -v
```

The controller tests use fake implementations of:

- `ProposalSource`
- `Evaluator`
- `FallbackGenerator`
- `Committer`

Covered behavior:

- always-accept commits anchor and all drafts in order;
- always-reject commits anchor then fallback only;
- reject-at-depth commits the accepted prefix then fallback;
- duplicate committed blocks are rejected;
- out-of-order draft proposals do not commit;
- discontinuous frame ranges are rejected;
- cross-window frame ranges must remain continuous;
- fallback uses the rejected draft's source noise;
- deeper drafts are not evaluated after the first reject;
- exceptions roll back controller and committer transaction state;
- rollback failure reports both original and rollback exceptions;
- begin failure is exception-atomic and does not trigger rollback;
- trace events are emitted in deterministic order.

The trace schema tests verify event shape, strict integer/string field typing,
sequence numbering, JSON-safe metadata, deep metadata immutability,
`allow_nan=False` serialization, metadata copy isolation, and rejection of
unknown event names.

All fake tensor/payload values in tests are treated as read-only borrowed
objects. The tests assert identity preservation for `source_noise`; no fake
adapter mutates payloads in place.
