# EXP-001 — Experiment-Handoff Fixture Dry Run

> Documentation-only R0 fixture. This record tests whether the handoff contract
> is complete. It did not train or evaluate a model and is not scientific
> evidence for model, data, system, or performance claims.

- Roadmap ticket: `EXP-001`
- Branch: `codex/exp-001-review-record`
- Draft PR: `pending` (replace after the initial documentation commit is pushed)
- Experiment owner: implementation agent; exact runtime identity not exposed
- Status: concluded fixture
- Started (UTC): 2026-07-11
- Last updated (UTC): 2026-07-11
- Model-run provenance: `docs/model-runs/EXP-001-review-record.md`

## Predeclared question and decision rule

- Hypothesis: a fresh agent can reconstruct a complete positive or negative
  experiment handoff from the template without asking for omitted run identity.
- Expected result: the fixture contains all consequential-run fields required by
  `PHILOSOPHY.md`, preserves a negative attempt, and traces branch through review.
- Success condition: every EXP-001 acceptance field is visibly represented and
  the negative attempt retains its attempted command, resolved config, and
  evidence.
- Failure condition: any attempted config/evidence can be omitted, or the
  baseline/next question cannot be found by a fresh agent.
- Relevant baseline commit and run: `origin/main` at
  `a05eb1de5656643757a1c3d98047c98dedea8bfa`; run baseline is
  `none — readiness gates unmet`.
- Baseline metrics and evidence link: `ROADMAP.md` AS-IS snapshot and readiness
  decision; no decision-grade run metrics exist.
- Smallest run capable of answering the question: documentation inspection plus
  repository validation; no training launch.

## Planned budget

| Resource | Limit | Derivation / measurement source |
| --- | --- | --- |
| Elapsed time on target hardware | 15 minutes of CPU-only inspection | R0 documentation fixture; DGX execution is unnecessary |
| Training tokens | 0 | No model execution is in scope |
| Optimizer steps | 0 | No model execution is in scope |
| Evaluation work and cadence | One field-coverage scan after editing | Sufficient to test the record surface |
| Checkpoint count and bytes | 0 checkpoints / 0 bytes | No trainer invocation |
| Local / external / W&B storage | Repository text only; no W&B upload | Avoid false run lineage and quota use |

## Attempt 1 — deliberately negative configuration dry run

This attempt is intentionally invalid. It demonstrates that a negative launch
cannot disappear and that its attempted Hydra configuration and evidence remain
reviewable.

### Launch identity

- Started / ended (UTC): 2026-07-11T14:58:41Z / 2026-07-11T14:59:42Z
- Outcome: negative configuration finding; composition succeeded and training was
  deliberately not launched because both streaming source lists were empty
- Exact command:
  ```text
  uv run python src/train.py --cfg job data.mode=streaming training.epochs=1 wandb.enabled=false
  ```
- Fully resolved Hydra configuration printed by that command:
  ```yaml
  data:
    mode: streaming
    train: data/inputLearnText.txt
    val: data/inputLearnText.txt
    streaming:
      cache:
        dir: data/stream_loader_cache
        max_size_bytes: 750000000000
      prefetch:
        enabled: true
        buffer_size: 16
      train:
        max_tokens: 1000000
        add_eos: true
        seed: 42
        sources: []
      validation:
        max_tokens: 100000
        add_eos: true
        seed: 43
        sources: []
  training:
    sequence_length: 64
    epochs: 1
    batch_size: 64
    shuffle: true
    optimizer:
      _target_: torch.optim.AdamW
      lr: 0.001
      weight_decay: 0.0
    scheduler:
      enabled: false
      interval: epoch
      _target_: torch.optim.lr_scheduler.CosineAnnealingLR
      T_max: ${training.epochs}
      eta_min: 0.0
  model:
    embed_size: 384
    num_heads: 6
    num_layers: 6
    dropout: 0.1
  artifacts:
    checkpoints_dir: artifacts/checkpoints
    tokenizers_dir: artifacts/tokenizers
    tokenizer_filename: tokenizer.json
  wandb:
    enabled: false
    project: llm-scratch
    entity: sunday-research
    name: null
    mode: online
    log_model_every_n_epoch: null
  ```
  The command exited successfully after printing the composed job config. The
  empty source lists are the negative result; no training command was launched.
- Git commit SHA: `pending` (fixture placeholder for the eventual PR commit)
- Worktree state: dirty; this documentation diff is the evidence under review
- Dependency lock identity: `uv.lock`, SHA-256
  `f010398f7f1520d2ba4fafa5e6b9319d1c637fbb9c8c51b9e4fb469faa641a70`
- Container/image identity: `N/A — no process or container was launched`

### Scientific identity

- Model architecture/config identity and parameter count: `N/A — rejected
  before model construction`
- Initialization / pretrained-weight check: `N/A — no weights loaded`
- Tokenizer identity: `N/A — rejected before tokenizer construction`
- Training manifest identity/checksum: `N/A — attempted source list was empty`
- Validation manifest identity/checksum: `N/A — attempted source list was empty`
- Train/validation disjointness evidence: failed precondition; neither manifest
  existed
- Random seeds: `N/A — no process launched`
- Hardware identity: `N/A — no training process launched`
- Software/runtime identity: `uv run` environment selected by the checksummed
  `uv.lock`; Hydra/Python composed the config, but Torch training was not started
- Precision and numerical controls: `N/A — no model computation`

### Counters, evidence, and integrity

- Counters: 0 training seconds, 0 optimizer steps, 0 training/example/target
  tokens; configuration command completed in 0.86 wall-clock seconds
- Metrics: `N/A — no training, validation, or evaluation`
- System metrics: `N/A — only Hydra configuration composition ran`
- W&B IDs: `N/A — W&B disabled and no launch occurred`
- Checkpoint IDs: `N/A — no trainer execution`
- Evidence: exit code 0 and the complete composed config above, plus
  `ROADMAP.md`'s confirmed blocker that the production-looking streaming switch
  has no configured sources
- Integrity checks: passed no-run integrity (no data read, benchmark access,
  pretrained weights, teacher outputs, synthetic targets, metrics, or
  checkpoints); failed manifest-presence precondition as intended
- Baseline comparison: `N/A — no scientific baseline exists and no metrics were
  produced`

### Attempt interpretation

- Result against conditions: expected negative result; the record preserves the
  attempted config and evidence rather than presenting an empty failure line
- Failure: empty training and validation source manifests make the attempt
  non-runnable
- Most likely cause: the illustrative override selects a streaming mode without
  a canonical composed source profile, directly evidenced by both empty lists
- Alternatives ruled out: CUDA, tokenizer, numerical stability, and W&B were not
  reached, so none can explain this preflight failure
- Remaining uncertainty: whether a future CFG-001 profile will resolve to a safe
  runnable source configuration

## Branch-to-review handoff trace

1. Branch: `codex/exp-001-review-record`.
2. Draft PR: `pending` (replace with the real URL when created).
3. Experiment record: this never-deleted fixture, linked from the PR.
4. Model-execution record: `docs/model-runs/EXP-001-review-record.md`.
5. Independent heavy review: `pending`; no verdict is claimed.
6. Merge authority: human reviewer only.

The trace uses `pending` placeholders intentionally until a commit and draft PR
exist; neither placeholder is evidence of completed review.

## Conclusion

- Hypothesis result: supported for template field coverage, pending independent
  review
- Evidence-backed conclusion: the record contract can retain a negative
  attempted configuration, reasoned N/A fields, evidence, and handoff links
- Uncertainty and limitations: this R0 fixture proves documentation shape only,
  not repeatability of a real run or any scientific/system outcome
- Exactly one next step: independently review EXP-001 against `PHILOSOPHY.md`,
  its acceptance criteria, and applicable `CHECK.md` sections 8.1, 8.3, and 7
