# VAL-001 — Shared Held-Out Validation Scoring

- Roadmap ticket: `VAL-001`
- Branch: `codex/val-001-held-out-validation`
- Draft PR: pending
- Experiment owner: implementation agent; exact runtime model/reasoning are not exposed
- Status: planned
- Started (UTC): 2026-07-13
- Last updated (UTC): 2026-07-13
- Model-run provenance: `docs/model-runs/VAL-001-held-out-validation.md`

## Predeclared question and decision rule

- Hypothesis (one falsifiable claim): one token-weighted causal-LM scorer can
  produce identical held-out NLL and perplexity from the training-time and
  standalone checkpoint paths while retaining immutable checkpoint, manifest,
  corpus, and evaluated-token identities.
- Expected result: the two paths agree exactly on a deterministic CPU fixture;
  known logits match analytically calculated NLL; Japanese, English, and
  aggregate denominators reconcile; overlap and memorization misuse fail before
  scoring.
- Success condition: every VAL-001 acceptance criterion and its ticket-required
  known-logit, parity, cadence, overlap, and checkpoint-milestone check passes.
- Failure condition / stop condition: stop on any score-path divergence, identity
  omission, denominator mismatch, train/validation overlap, or result that labels
  same-corpus memorization as held-out validation.
- Relevant baseline commit and run: DATA-004 stacked head
  `e1d4ed8af98de84a3393cd0f6e517f9daf649138`; DATA-004 evidence is recorded in
  `docs/experiments/DATA-004-pinned-baseline-mixture.md`.
- Baseline metrics and evidence link: existing training-time validation behavior
  and tests on the stacked base; no standalone VAL-001 score exists yet.
- Smallest run capable of answering the question: deterministic CPU fixture
  checkpoint evaluation plus focused and full network-free tests.

## Planned budget

| Resource | Limit | Derivation / measurement source |
| --- | --- | --- |
| Elapsed time on target hardware | N/A — correctness ticket uses CPU fixture | Ticket acceptance criteria |
| Training tokens | Fixture only; no consequential pretraining run | Scoring parity, not model quality |
| Optimizer steps | Small deterministic fixture checkpoint only | Checkpoint milestone acceptance |
| Evaluation work and cadence | One fixed bounded validation pass per trigger | VAL-001 scope |
| Checkpoint count and bytes | Minimum fixture checkpoint(s) | Standalone/training parity |
| Local / external / W&B storage | Local JSON; W&B disabled in evidence run | Optional summary remains off by default |

## Attempt 1 — implementation and fixture evidence pending

### Launch identity

- Started / ended (UTC): pending
- Outcome: pending
- Exact command: pending
- Fully resolved Hydra configuration or immutable path to it: pending
- Git commit SHA: pending
- Worktree state: pending
- Dependency lock identity (path and checksum): pending
- Container/image identity or `N/A` with reason: pending

### Scientific identity

- Model architecture/config identity and parameter count: pending fixture evidence
- Initialization / pretrained-weight check: random-initialized project model only
- Tokenizer name, immutable revision/checksum, and special-token contract: pending
- Training manifest identity/checksum: pending
- Validation manifest identity/checksum: pending
- Train/validation disjointness evidence: pending
- Random seeds (Python, NumPy, PyTorch CPU/CUDA, loader/sampler as applicable): pending
- Hardware identity: CPU fixture; exact evidence pending
- Software/runtime identity: pending
- Precision and numerical controls: FP32 deterministic fixture

### Counters, evidence, and integrity

- Actual elapsed time, optimizer steps, training tokens, examples, and target tokens: pending
- Training/validation/evaluation metrics with denominators: pending
- System metrics (throughput, memory, utilization, I/O, pauses as applicable): N/A for bounded correctness fixture
- W&B run/project/artifact IDs, or reasoned `N/A`: W&B disabled for local evidence
- Checkpoint IDs/paths/checksums and retention status, or reasoned `N/A`: pending
- Logs, tables, plots, or other immutable evidence links: pending
- Integrity checks: pending
- Comparison with the predeclared baseline using the same protocol: pending

### Attempt interpretation

- Result against success/failure conditions: pending
- Failure or anomaly: pending
- Most likely cause and supporting evidence: pending
- Alternatives ruled out and how: pending
- What remains uncertain: pending

## Conclusion

- Hypothesis result: inconclusive
- Evidence-backed conclusion: implementation and independent review are pending.
- Uncertainty and limitations: VAL-001 is stacked on unmerged DATA-004 and must
  be rebased or retargeted after its dependency merges.
- Exactly one next step or next question: implement the shared scorer and prove
  training-time/standalone parity on the fixed fixture.
