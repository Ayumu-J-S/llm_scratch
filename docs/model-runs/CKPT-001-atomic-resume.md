# CKPT-001 - Atomic rotating full-state resume

- PR: draft pending creation by the coordinating session
- Branch: `codex/ckpt-001-atomic-resume`
- Ticket: `CKPT-001`
- Hypothesis: a small direct checkpoint boundary can preserve the full
  single-process training state, verify durable recovery files before rotation,
  and resume an exact stream suffix without adding distributed or W&B policy.
- Experiment record: `N/A` — this is correctness/recovery infrastructure, not
  a model-quality experiment.
- Started: 2026-07-12
- Current verdict: implementation and repair complete; independent review
  pending.
- Record owner: implementation sub-agent `/root/ckpt001_implementation`

## Scope and decision context

- Goal: satisfy `ROADMAP.md` ticket `CKPT-001`.
- In scope: model/optimizer/scheduler/precision/counter/RNG/stream-cursor
  state; resolved config plus run/data/tokenizer identity; temp-write,
  read-back verification, atomic replacement, recovery rotation, best/final/
  milestone separation, explicit Hydra resume selection, compatibility checks,
  and the required invariant fixtures.
- Out of scope: W&B retention/upload policy, inference formats, conversion,
  distributed/cross-architecture recovery, or throughput optimization.
- Relevant `PHILOSOPHY.md` principles: one-machine operations remain direct and
  inspectable; recovery retention is small and verified before deletion; runs
  preserve identity and evidence; a bounded smoke must not be presented as a
  long-run storage or throughput result.
- Relevant `CHECK.md` selection: 5.4 disk/resource headroom, 6.3 boundaries,
  and 9.1 checkpoint/resume. Other GPU performance, evaluation, benchmark, and
  W&B-policy sections are unchanged or explicitly out of scope.
- Baseline commit: `19dce46b134569de4e29686570fa5bd69560a556`.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input / requested work | Outcome | Observable findings and evidence |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | implementation | not exposed by runtime | not exposed by runtime | Baseline `19dce46`; requested Luna / Extra High; CKPT-001, `PHILOSOPHY.md`, selected CHECK 5.4/6.3/9.1 | implemented, then full-suite regression found | Added `CheckpointManager`, full-state trainer payload/restore, compatibility identity, rotating recovery files, retained best/final/milestones, Hydra `resume_path`, per-save size/write/verification/pause metrics, and fixture coverage. Initial focused run: 32 passed. Reviewer full suite then found `tests/test_data_manifests.py::test_streaming_dataset_preflights_only_once_across_two_epochs`: 240 passed, 1 skipped, 1 failed. |
| 2 | repair | not exposed by runtime | not exposed by runtime | Failed full-suite handoff: terminal stream cursor was reloaded on normal next epoch, setting StreamLoader resume-pending and returning an empty epoch | implemented | Kept the observable terminal cursor but introduced an explicit resume-pending bit: only interruption/config/checkpoint restore reloads it. Natural iteration starts under the original stream policy. Targeted repair evidence: 47 passed. Reviewer confirmation: 9 targeted passed; full suite 241 passed, 1 skipped. |
| 3 | repair | not exposed by runtime | not exposed by runtime | Post-instrumentation two-step CPU smoke | implemented | The smoke exposed a duplicated milestone write at step 2: normal event handling and epoch-end handling both selected the same step. Added an independently persisted `_last_milestone_step` guard so resuming preserves de-duplication. Targeted evidence: 48 passed; corrected smoke emitted exactly milestones 1 and 2. |
| 4 | repair | not exposed by runtime | not exposed by runtime | Resume-safety audit of the local memorization path | implemented | Local map-style smoke loading has no persisted sampler cursor. Rather than silently replay a prefix, exact resume now requires a cursor-aware streaming train loader and rejects a checkpoint without `stream_cursor` before train loader creation. |

## Runtime provenance block

Requested/default values are distinct from the runtime display. The active
runtime exposes Codex / GPT-5 but not an exact deployment identifier or
reasoning mode.

| Namespace | Product | Displayed family | Exact model identifier | Reasoning mode | Source / unavailable reason |
| --- | --- | --- | --- | --- | --- |
| requested | Codex | Luna | not exposed by runtime | Extra High | explicit implementation request |
| actual | Codex | GPT-5 | not exposed by runtime | not exposed by runtime | runtime did not display an exact deployment ID or reasoning mode |

- Machine-readable capture: `docs/model-runs/evidence/CKPT-001-implementation-provenance.json`.
- Capture command: `uv run python scripts/capture_model_provenance.py --repo . --phase implementation --role implementation --task-path /root/ckpt001_implementation --requested-model Luna --requested-reasoning-mode 'Extra High' --actual-product Codex --actual-model-family GPT-5 --output docs/model-runs/evidence/CKPT-001-implementation-provenance.json`.
- Privacy: the capture contains no prompts, hidden chain-of-thought, token
  counts, secrets, or raw thread identifier.

## Failed-review handoff

This was an implementation-time full-suite failure, not a passing independent
review. It remains here for the required repair trail.

- From cycle: 1 implementation.
- Failed check: `tests/test_data_manifests.py::test_streaming_dataset_preflights_only_once_across_two_epochs`.
- Why: `StreamingTokenDataset` retained a completed cursor and unconditionally
  fed it to a new `StreamLoader`; its resume-pending semantic correctly means
  “no suffix remains,” but is wrong for a natural next epoch.
- Reproduction/evidence: reviewer full suite reported `240 passed, 1 skipped,
  1 failed`; the failure is retained rather than overwritten.
- Constraints: preserve exact interrupted suffixes, including threaded
  prefetch; do not duplicate StreamLoader cursor logic; normal data iteration
  must remain re-iterable.
- Selected repair: same available implementation runtime, requested Luna /
  Extra High; this was narrow local state wiring rather than a reason to alter
  model/data semantics.
- Repair request: distinguish explicit/interrupted resume from a naturally
  completed iterator, then rerun targeted stream/checkpoint/trainer tests and
  the full suite before review.

## Repair result

- Repair cycle: 1.
- Input handoff: terminal-cursor full-suite failure above, plus existing
  StreamLoader `resume_cursor_pending` contract.
- Changes: dataset owns `_resume_cursor_pending`; config/checkpoint cursor
  loads set it, first restored iteration clears it, and an interrupted iterator
  leaves it set. Natural completion records the cursor for inspection but does
  not reload it on a normal later epoch.
- Deliberately unchanged: StreamLoader cursor format, packing order, model,
  optimizer, data manifests, W&B policy, and distributed behavior.
- Local evidence: `uv run pytest -q tests/test_checkpoint.py tests/test_trainer.py tests/test_data_manifests.py` — 47 passed.
- Independent re-review: pending formal CKPT-001 review. Reviewer's repair
  confirmation was 9 targeted passed and `241 passed, 1 skipped` for the full
  suite, but no independent ticket verdict has been requested yet.

### Repair cycle 2

- Input handoff: corrected CPU smoke observed two writes for milestone step 2
  because the existing trainer independently invokes cadence logic at epoch end.
- Changes: `_last_milestone_step` is part of the trainer event state and full
  checkpoint payload; a milestone save only occurs once for an optimizer step.
- Local evidence: `uv run pytest -q tests/test_checkpoint.py tests/test_trainer.py tests/test_data_manifests.py` — 48 passed, including a regression asserting exactly one milestone per step across epoch end.
- Re-review: formal independent CKPT-001 review pending after the final full suite.

### Repair cycle 3

- Input handoff: local map-style loaders can checkpoint model state but cannot
  prove their next shuffled sample on a new process.
- Changes: `require_exact_stream_resume_state` rejects a checkpoint lacking a
  stream cursor during `src/train.py` preflight, before a train loader is built;
  Trainer applies the same guard before mutation.
- Local evidence: focused fixture asserts rejection of a missing cursor; exact
  resume equality uses the cursor-aware stream fixture.

## Final evidence (implementation state; review pending)

- Resolved checkpoint command/config: relative `artifacts.checkpoints_dir`
  lives under the Hydra run directory; `artifacts.resume_path` accepts `latest`,
  a directory, or a file. Non-`latest` relative paths resolve from the configured
  checkpoint directory, not the process CWD. Exact resume requires the
  cursor-aware streaming training path; a map-style loader without a persisted
  next-sample cursor is rejected rather than silently replaying a prefix.
  `resume_path` alone is excluded
  from the compatibility hash; model/data/tokenizer/resolved-config changes are
  rejected before train loader creation.
- Required fixture coverage: uninterrupted/resumed equality including
  Python/NumPy/Torch RNG and scheduler/optimizer/model state; corrupt-newest
  fallback; write/read-back failure preserving the prior recovery; verified
  replacement before rotation deletion; protected best/final/milestone files;
  identity mismatch rejection; relative resume-path behavior; and the exact
  next prefetched streaming batch.
- CPU bounded real-path smoke after the per-save metric and milestone repair:
  `uv run python src/train.py profile=smoke_overfit runtime.device=cpu training.max_steps=2 training.epochs=1 training.batch_size=2 training.checkpoint_every_n_steps=1 training.validation_every_n_steps=null training.milestone_every_n_steps=1 model.embed_size=16 model.num_heads=2 model.num_layers=1 model.dropout=0 wandb.enabled=false artifacts.checkpoints_dir=checkpoints hydra.run.dir=/tmp/ckpt001-smoke-run`.
  It completed two updates and produced verified 20.10 MB recovery, milestone,
  best, and final full-state files containing model, optimizer, scheduler,
  precision, counters, RNG, stream cursor, resolved config, and identity.
  Recovery write/verification/pause was 12.18/2.33/16.01 ms at step 2
  (20,104,777 bytes); final write/verification/pause was 12.64/2.21/15.98 ms
  (20,103,221 bytes). Exactly two milestone events were emitted, at steps 1
  and 2.
- Measurement contract: each checkpoint event now records
  `checkpoint/size_bytes`, write time, read-back verification time, total pause,
  and write bytes/s locally. The tiny CPU smoke is intentionally not a claim
  about full-model DGX save pause, UMA peak, or sustained storage throughput;
  those need a later bounded measured R2 run if they become decision-critical.
- Final implementation validation: `uv run pytest -q` — 243 passed, 1 skipped;
  focused checkpoint/trainer/config/stream tests — 58 passed; `uv run ruff
  check` over changed source/tests, `ruff format --check`, `git diff --check`,
  and `uv lock --check` passed.
- Known trade-offs: recovery saves are synchronous and avoid a deep copy of
  `model.state_dict()` to avoid an avoidable full-model UMA peak. Atomicity is
  the local-filesystem temp-file/read-back/`os.replace` contract, not a network
  filesystem guarantee.
- Unresolved risks: formal independent review, final exact-head checks, a
  any DGX-scale pause/memory result remain pending.
- Human decision requested: none; next step is independent CKPT-001 review.

## Merge authority and final audit

- Merge path: guarded agent self-merge only after the independent review and
  all exact-head gates; pending.
- Human authorization: user instruction on 2026-07-12, “これからはとりあえず
  全部セルフマージしていいよ / とりあえずロードマップ完成させよう”.
- Authorization covers bounded roadmap series: yes, subject to every guarded
  gate in `PHILOSOPHY.md` and `docs/agent-model-workflow.md`.
- Exact independently reviewed head / verdict / threads / checks / base /
  mergeability / final audit: pending. No head will be marked ready or merged
  until a heavier independent review is recorded for the exact final head.
- Prohibited self-merge categories: expected clear for ordinary source/test/doc
  work; final audit must re-confirm no secrets, private data, paid resource,
  destructive action, licensing uncertainty, release/deployment, or account
  change.

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | implementation and repair | Kept persistence direct, covered atomic failure/fallback/rotation and exact RNG-stream trajectory, and avoided deep-copy checkpoint memory | Initial stream wrapper conflated a terminal normal epoch with an explicit resume, causing one full-suite regression | ticket acceptance criteria, StreamLoader cursor contract, reviewer failure reproduction | repair complete; independent review pending |

## Ledger update

- [x] Added the in-progress ticket row to `docs/model-runs/README.md`.
- [x] Retained the failed full-suite cycle and repair handoff.
- [x] Recorded requested vs actual runtime provenance without inference.
- [ ] Add the formal independent review, exact-head final audit, and merge result.
- [x] Confirmed this is not the guarded-self-merge policy bootstrap PR.
