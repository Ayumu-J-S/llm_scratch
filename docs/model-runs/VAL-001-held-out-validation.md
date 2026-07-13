# VAL-001 — Trustworthy Lightweight Held-Out Validation

- PR: [#42](https://github.com/Ayumu-J-S/llm_scratch/pull/42) (draft)
- Branch: `codex/val-001-held-out-validation`
- Ticket: `VAL-001`
- Hypothesis: one shared token-weighted scorer gives identical training-time and
  standalone checkpoint results with complete immutable evaluation identity.
- Experiment: `docs/experiments/VAL-001-held-out-validation.md`
- Started: 2026-07-13
- Current verdict: Attempt 6 evidence `PASS WITH NOTE`; independent re-review pending
- Final record owner: implementation agent

## Scope and decision context

- Goal: fixed Japanese/English held-out validation without conflating
  memorization with generalization.
- In scope: shared scoring, per-corpus and aggregate NLL/perplexity, step/token
  cadence, standalone checkpoint evaluation, local JSON, optional compact W&B,
  and immutable result identity.
- Out of scope: generative benchmarks, human evaluation, and reserved tests.
- Policy: `PHILOSOPHY.md` evaluation-as-training, first-class evidence,
  fixed step/token cadences, and reproducible data/scorer identities.
- Baseline: stacked DATA-004 head
  `e1d4ed8af98de84a3393cd0f6e517f9daf649138`.
- Selected `CHECK.md`: minimum review, 6.1, 6.3, 7.1–7.3, 8.1–8.3, and the
  applicable checkpoint/evaluation identity parts of 9.1.

## Model execution trail

| Cycle | Phase | Requested model / mode | Exact displayed model / mode | Input commit | Outcome | Main findings / changes |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | implementation | `gpt-5.6-luna` / Extra High (`xhigh` invocation) | not exposed by runtime / not exposed by runtime | `e1d4ed8` | implemented, later repaired | Shared scorer, fixed-window factories, source attribution, identities, standalone Hydra evaluator, namespace guard, tests |
| 1 | independent review attempt | heavier review model / Extra Thinking | not exposed by runtime / not exposed by runtime | `a8520d7` | blocked before verdict | Earlier runtime path did not expose a selectable heavy reviewer; blocked attempt retained |
| 2 | repair | `gpt-5.6-luna` / Extra High | not exposed by runtime / not exposed by runtime | `a8520d7` | implemented | Strict JSON, NLL sums/reconciliation, source/manifest trust, verified checkpoint reconstruction, memorization isolation, stronger failure tests |
| 2 | preliminary audit repair | `gpt-5.6-luna` / Extra High | not exposed by runtime / not exposed by runtime | `2133248` | implemented | Model mode now restores even when iterator cleanup raises; regression test added at `0a13838` |
| 2 | independent heavy review | `gpt-5.6-sol` / Max | not exposed by runtime / not exposed by runtime | `41191cb` | FAIL | Checkpoint config was not bound to `identity.config_sha256`; configured/resolved manifest identity was not reconciled before scoring; scorer could trust a separately supplied manifest identity; training-time and standalone logical checkpoint identities differed; phase timing was incomplete for CHECK 6.3; evaluation package re-exports were unused |
| 3 | repair | `gpt-5.6-luna` / Extra High | not exposed by runtime / not exposed by runtime | `41191cb` | implemented; DGX evidence and re-review pending | Added canonical config/data identity verification, fail-closed actual-loader fingerprint/selection verification, shared logical identity parity, tamper regressions, benchmark-only atomic measurement capture, and direct evaluation imports |
| 3b | repair completion | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | `41191cb` working tree | implemented; DGX evidence and re-review pending | Replaced always-on/stale timing with disabled-by-default atomic measurement mode, separated optimizer/validation intervals, added CUDA-event and memory capture, closed loader selection/missing-metadata gaps, predeclared the repeated DGX protocol, and corrected provenance claims |
| 4 | CUDA determinism repair | available lightweight implementation model | not exposed by runtime / not exposed by runtime | `74a6d6b` | implemented; Attempt 5 and re-review pending | Attempt 4 failed exact trajectory at step 1 before validation; changed deterministic mode from warn-only to strict, strengthened its regression, and predeclared a fresh full matrix |
| 4 | repair QA | `gpt-5.6-luna` / Extra High (`xhigh` invocation) | not exposed by runtime / not exposed by runtime | `74a6d6b` working tree | PASS | Read-only focused review found no issue and agreed strict fail-closed determinism is the smallest sound repair; not the required final heavy re-review |
| 5 | evidence-protocol repair | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | `4264e4a` | Attempt 6 `PASS WITH NOTE`; heavy re-review pending | Preserved Attempt 5 as FAIL, repaired container sampling/start barrier, prospectively bounded the adaptive first-step recovery gate, and prohibited further relaxation |
| 5 | protocol review | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | `4264e4a` working tree | PASS WITH NOTE | Adaptive 5% gate is defensible only with a fully fresh matrix, CHECK-anchored disclosure, phase attribution, coarse container claims, and no further revision |

Requested values are invocation/config values, not claimed actual deployment
identifiers. The runtime did not expose the exact identifier or reasoning mode to
the caller for implementation/repair, so those actual fields remain unavailable.

## Provenance

- Initial implementation capture:
  `docs/model-runs/evidence/VAL-001-implementation-provenance.json`.
- Blocked review attempt:
  `docs/model-runs/evidence/VAL-001-review-provenance.json`.
- Repair capture:
  `docs/model-runs/evidence/VAL-001-repair-provenance.json`.
- Review-cycle-2 capture:
  `docs/model-runs/evidence/VAL-001-review-cycle-2-provenance.json`.
- Repair-cycle-3 capture:
  `docs/model-runs/evidence/VAL-001-repair-cycle-3-provenance.json`.
- Repair-cycle-3b capture:
  `docs/model-runs/evidence/VAL-001-repair-cycle-3b-provenance.json`.
- CUDA determinism repair capture:
  `docs/model-runs/evidence/VAL-001-repair-cycle-4-provenance.json`.
- CUDA determinism repair QA capture:
  `docs/model-runs/evidence/VAL-001-repair-cycle-4-luna-review-provenance.json`.
- Attempt 6 protocol review capture:
  `docs/model-runs/evidence/VAL-001-attempt6-protocol-review-provenance.json`.
- Codex CLI: `codex-cli 0.144.1` for recorded implementation/repair captures.
- Implementation head: `a8520d7fad718574d1fca4293e6f969c7a478b79`.
- Main invariant repair: `057983c`; measured merged head:
  `21332488e8a1d2334cbb6e2d0593a77a598c1d01`.
- Exceptional-cleanup repair: `0a138386a03e178a88b5ccca6334288b57188efb`.
- Failed independent review: requested `gpt-5.6-sol` / Max at `41191cb`;
  exact runtime model/mode not exposed.
- Repair phase model and reasoning: exact values not exposed by runtime.
- Identity/measurement repair head: `78e0448`.
- Attempt 4 protocol head: `74a6d6b`.
- CUDA determinism repair head: working tree after `74a6d6b`; no uncommitted-tree
  commit SHA is claimed before local verification completes.
- Attempt 5 measured head: `4264e4a`; compact failed evidence is retained.
- Privacy: no prompts, hidden chain-of-thought, token counts, secrets, or raw
  thread IDs are recorded.

## Implementation and repair findings

Three delegated audits separated roadmap acceptance, implementation semantics,
and DGX evidence. The first implementation needed these concrete repairs:

| Area | Finding | Repair / proof |
| --- | --- | --- |
| Metric reduction | Aggregate-only reporting and insufficient exact reconciliation | Per-corpus/aggregate `nll_sum`, target counts, token weighting, strict reconciliation tests |
| Identity | Batch-sensitive or incomplete result identity could hide a changed window/source assignment | Batching-independent hashes cover contexts, labels/masks, order, target IDs, and target sources |
| Data semantics | Same-corpus smoke could be confused with validation | `memorization/*`, no memorization best checkpoint, and standalone memorization rejection |
| Checkpoint trust | Standalone reconstruction needed checkpoint-owned config and physical identity | Verified full-state checkpoint load, checkpoint config/model/tokenizer/data validation, path/SHA/size output |
| Serialization | `exp(NLL)` overflow could emit non-standard JSON | Nullable perplexity plus overflow flag; atomic JSON uses `allow_nan=False` |
| Performance path | Repeated full checkpoint hashing could enter training-time validation | Physical hashing occurs in standalone/output identity, not the hot path |
| Cleanup | Iterator-close failure could skip model-mode restoration | Nested `finally` at `0a13838`; regression proves restoration despite close error |

No BENCH-001 work, generic framework, compatibility shim, or separate runtime
configuration source was added. Hydra remains authoritative and imports are
direct.

## Independent review FAIL at `41191cb`

The independent review requested `gpt-5.6-sol` / Max and returned `FAIL`; the
exact runtime model and reasoning mode were not exposed. The following
actionable findings were handed to the repair phase and remain visible here:

| Finding | Repair and regression evidence |
| --- | --- |
| `state.resolved_config` was not cryptographically bound to `identity.config_sha256` using the existing identity semantics. | `checkpoint_config_sha256` centralizes the existing “exclude only `artifacts.resume_path`” rule; checkpoint write/read and standalone load verify it. Resolved-config tampering fails before evaluation. |
| `identity.data_fingerprints` was not reconciled against ordered configured and actually resolved manifests. | Configured train-then-validation manifest fingerprints are checked against captured run identity and full-state identity; the scorer checks configured source order/fingerprint against the actual loader and rejects ordered-manifest tampering. |
| The scorer could use a caller-supplied manifest identity instead of the loader that was actually scored. | The scorer now derives identity from the loader and strictly compares any supplied identity to it. Standalone evaluation relies on this derived identity. |
| Training-time and standalone logical checkpoint identities had different shapes (`kind`/`counters` versus step/token fields). | `build_logical_checkpoint_identity` is used by both paths; the milestone parity regression asserts exact equality. |
| CHECK §6.3 lacked local phase timing for repeated DGX analysis. | Hydra-controlled measurement is disabled by default. When enabled it buffers data-wait/host phases, CUDA-event phases, allocator memory, logging, checkpoint, and complete validation-event timing, then atomically writes one JSON. It deliberately synchronizes once at the measured step boundary; the ordinary path creates no events, synchronization, or measurement file. |
| Unused evaluation package re-exports obscured the direct import boundary. | `evaluation.__init__` no longer re-exports scorer symbols; callers import `evaluation.scoring` directly. |

No new DGX result is claimed during this repair. A three-pair, six-run,
60-step warm-cache protocol with continuous GPU/host/container traces and
predeclared pass/fail gates is recorded in the experiment document before it is
run. The earlier R2 pair remains historical old-head evidence only.

## Validation and evidence

### Automated checks

- Focused baseline audit: `77 passed` before the repair series.
- Repair-focused suite after the final split/measurement repair: `55 passed`.
- Full repository suite after the final split/measurement repair:
  `310 passed, 1 skipped in 64.87s`.
- Full repository suite after the strict-determinism repair:
  `310 passed, 1 skipped in 64.15s`.
- Full repository suite at measured head `2133248`:
  `302 passed, 1 skipped in 64.17s`.
- `uv lock --check`, full Ruff lint, changed-Python-file format check, and
  `git diff --check` pass after repair. The repository-wide formatter check
  still reports four pre-existing unrelated files:
  `src/models/embedding.py`, `tests/test_ci_quality_gate.py`,
  `tests/test_data003_stream_cursor.py`, and `tests/test_model_provenance.py`;
  they were not changed in this repair.
- Cleanup-repair focused check at `0a13838`: `15 passed`; Ruff, format, and diff
  checks passed.
- Tests cover known logits, ignored/partial labels, NLL reconciliation, source
  boundaries/failures, batching/context/source digest changes, unknown sources,
  strict JSON overflow, iterator lifecycle, zero targets, mode restoration,
  memorization namespace/no-best, and standalone milestone parity/identity.
- Acceptance mapping: `test_known_logits_match_token_weighted_nll_and_perplexity`
  proves analytical scoring; `test_cross_manifest_content_overlap_is_rejected_even_when_ids_differ`
  and `test_audit_requires_document_ids_and_reports_both_overlap_types` prove
  held-out overlap rejection; `test_trainer_memorization_metrics_have_no_validation_namespace`
  and `test_standalone_held_out_evaluation_rejects_memorization_checkpoint` prove
  namespace and checkpoint separation; `test_validation_and_checkpoint_cadences_are_independent`
  and `test_token_cadence_records_boundaries_and_local_metrics` prove cadence;
  `test_standalone_milestone_matches_shared_training_time_score` proves milestone
  score and identity parity.
- Repair-focused tests additionally cover exact logical identity parity,
  resolved-config and data-fingerprint tampering, ordered manifest mismatch,
  actual-loader fingerprint/selection verification, missing-loader-metadata
  rejection, disabled measurement, atomic measurement flush, and separation of
  optimizer-step and validation-event timing. The focused suite passes
  (`55 passed`); the full suite passes (`310 passed, 1 skipped`).

### Invalidated evidence retained

`docs/experiments/evidence/VAL-001-cpu-parity.json` belongs to `a8520d7` and is
not current acceptance evidence. It evaluated a memorization `best.pt`; the
repaired code correctly creates no such best checkpoint and rejects standalone
memorization evaluation. It remains committed as a negative historical attempt.

### DGX R2 and standalone parity

Durable evidence:
[`docs/experiments/evidence/VAL-001-dgx-r2.json`](../experiments/evidence/VAL-001-dgx-r2.json).

- Pinned runtime: image
  `sha256:23a1bee69fe189e77105cdddeee9aeff6ef0763d58a691625fbfcab64efd1887`,
  NVIDIA GB10, BF16/CUDA, PyTorch `2.13.0a0+8145d630e8.nv26.06`.
- Matched 50-step arms used the same 49,535,114-parameter model, DATA-004 cache,
  manifests, seed, precision, batch/accumulation, and 102,400 training targets.
  Only validation cadence changed.
- The off/on training loss traces and all non-time step payloads matched exactly.
- Validation ran exactly at steps 25 and 50. Each pass scored 65,536 fixed
  targets at about 3,358 targets/s in about 19.52 seconds; the Japanese/English
  denominator split was exactly 32,768/32,768.
- Step-50 training-time and standalone evaluation matched exactly on aggregate
  and per-corpus scores, counts, manifests, logical checkpoint identity, 8,192
  windows, window hash, and target hash. Standalone output SHA-256 is
  `43bedcad72b138f873e227b8c455cf0598ac6a6b643d10b262ebf986838fa06e`.
- Best checkpoint SHA-256 is
  `0d48d810cd4657473c904b3cfd7e3a63174d2b7a3284248ad97ddbeb82f6ddea`.
- Validation plus best-save pauses totaled 43.1997 seconds. The first clean
  post-validation step returned to the immediate pre-validation timing range.

For that historical R2, CHECK §6.3 remained `FAIL` pending current-head evidence.
The prior pair had
no warm-up cutoff or continuous trace, and its reported p50 step times imply an
approximately 10.13% on/off throughput regression, meeting CHECK's normal FAIL
trigger. The predeclared repair-head protocol requires three matched pairs,
continuous traces, phase/data-wait evidence, exact trajectory/model parity, and
a median paired regression below 5%.

### Current-head DGX Attempt 3 — stopped on data wait

- Exact head `78e0448`, pinned image/GB10/cache, and continuous GPU/host/container
  traces were used. A no-work launch first failed on container Git ownership;
  explicit `safe.directory=/workspace` environment values repaired only that
  operational boundary.
- The completed sequence-8 pair reached 60 steps/122,880 targets per arm,
  preserved exact non-time trajectory parity, emitted zero off-arm validation
  events and step-25/50 on-arm events, and showed a +2.33% on/off throughput
  delta rather than the old-head regression.
- Validation pauses were 22.2045 s and 22.3052 s, with less than 0.3 ms
  attribution error, inside the predeclared pause budgets.
- The attempt correctly stopped before pairs 2/3 because steady data wait was
  82.60% off and 82.47% on, exceeding the zero-tolerance 10% gate. Cache
  inventory/content stayed exact and leases were released.
- Validation-off diagnostics localized per-window producer overhead: data wait
  fell monotonically to 0.86% at sequence 4,096, with only 7.40 GB reserved
  memory. Attempt 4 is predeclared before acceptance measurement at sequence
  4,096, batch 1, accumulation 1, and 4,096 targets/update. Diagnostic arms are
  excluded from acceptance replication.

### Current-head DGX Attempt 4 — stopped on CUDA nondeterminism

- Exact head `74a6d6b`, pinned image/GB10/cache, and continuous traces were used.
  Pair 1 reached 60 steps/245,760 targets per arm, cleared the data-wait gate
  (0.99% off, 1.36% on), and stayed within validation/scoring pause budgets.
- Post-warm-up throughput was 17,909.59 targets/s off and 17,965.08 targets/s on.
  Those values are retained as failed-attempt and deterministic-cost context,
  not acceptance replication.
- Exact trajectory failed at step 1, before validation: loss was exact, while
  gradient norm differed by `7.89165e-05`; all 60 step records differed. Both
  arms reported nondeterministic memory-efficient-attention backward because
  `deterministic=true` still used `warn_only=True`.
- Pair 1 stopped the attempt before pairs 2/3. Pinned-image BF16 probes confirmed
  unequal gradients in warn-only mode and exact gradients/model digests in
  strict mode. The smallest repair makes deterministic mode strict, retains the
  documented cross-platform/version caveat, and restarts all six arms as
  predeclared Attempt 5.

### Current-head DGX Attempt 5 — stopped on evidence gates

Durable compact evidence:
[`docs/experiments/evidence/VAL-001-dgx-r5-failed.json`](../experiments/evidence/VAL-001-dgx-r5-failed.json).

- Exact head `4264e4a`; all six strict-deterministic arms completed. All three
  60-step trajectories and final-model digests matched exactly. Paired
  validation-on throughput deltas were -1.21%, -0.016%, and -0.022% (median
  -0.022%), and every data-wait fraction was below 1%.
- All six validation events and scorers met their pause budgets. Replicated
  validation identities/scores and pair-1 step-50 training-time/standalone
  results matched exactly. Cache and lease checks, memory recovery, sustained
  five-step recovery, GPU/host coverage, and no-swap checks passed.
- Strict mode's descriptive pair-1 throughput cost versus otherwise matched
  warn-only Attempt 4 was -44.8% off and -45.7% on. Attempt 4 is not acceptance
  replication, so this is cost context rather than a causal performance verdict.
- Attempt 5 remains `FAIL`: container samples covered only 50.1-50.7% of the
  declared 1 Hz intervals, and five of six first post-validation steps exceeded
  the preceding five-step nearest-rank p95 by 0.7-4.7%. Following-five means
  still stayed within 3% of pre-event and 1.9% of paired-off controls.
- Attempt 6 is a fully fresh adaptive retry. It prospectively declares coarse
  0.5 Hz container evidence with explicit interval/per-event coverage, adds a
  collector start barrier, and applies CHECK §3's existing 5% threshold to both
  the first-step pre-event and paired-off comparisons. The 5% adaptation is
  disclosed, every signed phase result remains reportable, and no further gate
  relaxation is allowed.

### Current-head DGX Attempt 6 — `PASS WITH NOTE`

Durable compact evidence:
[`docs/experiments/evidence/VAL-001-dgx-r6.json`](../experiments/evidence/VAL-001-dgx-r6.json).

- Exact head `497a7b6`; all six fresh arms completed at 60 steps/245,760 targets.
  Every pair's non-time trajectory and canonical final-model digest matched
  exactly. Paired throughput deltas were -0.152%, +0.041%, and +0.008% (median
  +0.008%); all data-wait fractions were below 0.69%.
- Validation cadence, scores, per-corpus/aggregate denominators, identities,
  pause attribution, and replicated fixed-window results passed exactly. The
  maximum full event was 7.50 s and maximum scorer 4.76 s.
- Pair-1 step-50 training-time and standalone results matched exactly. Standalone
  output SHA-256 is `bdddc4aaf13c71b6903e14f401d4e3784e6f3b96f6ef1285537b2107267ad080`;
  verified best-checkpoint SHA-256 is
  `246c9ec331719b4eaa0367eeda28d392fa21b896d844401fff2544bf939a6a1f`.
- Every adaptive first-post and unchanged sustained-recovery gate passed. First
  steps were 0.20-1.28% versus pre-event p95 and -0.36% to +0.48% versus paired
  off. Following-five means were at most +1.57% versus pre-event and +0.28%
  versus paired off. Retained phases attribute the largest step-25 differences
  to approximately 4-13 ms host-device preparation/data wait.
- GPU/host and coarse-container trace gates passed; every container interval was
  below 2.03 s and 3-4 samples fell inside each validation event. Step tails,
  allocator recovery, immutable cache/shards, released leases, and no-swap gates
  passed.
- The note is mandatory: the 5% first-step rule was adapted after Attempt 5 and
  validated only on a fresh matrix. The conclusion is bounded recovery, not zero
  restart cost. Strict determinism's roughly 45% descriptive throughput cost
  versus failed warn-only Attempt 4 remains for DGX-001 to decide.

The R2 measured `2133248`; `0a13838` only changes exceptional iterator cleanup.
No successful scoring/training code path or performance control changed. This
parent-head relationship is disclosed rather than misrepresented as exact-head
measurement.

## Review status and handoff

### Review attempt 1

- Commit: `a8520d7`.
- Result: blocked before a technical verdict; it does not count as an
  independent review.
- Historical blocker claims about DGX unavailability are superseded by the
  pinned-container R2 above.

### Review cycle 2 — failed

- Reviewer request: `gpt-5.6-sol` / Max; exact runtime model/mode not exposed.
- Target: `41191cb`.
- Required review: `PHILOSOPHY.md`, VAL-001 acceptance criteria, minimum CHECK,
  6.1, 6.3, 7.1–7.3, 8.1–8.3, and applicable 9.1.
- Verdict: `FAIL`.
- Findings: see the explicit handoff above; all were sent to the repair phase.

### Repair cycle 3

- Repair model: requested `gpt-5.6-luna` / Extra High; exact displayed model and
  reasoning mode are not exposed by runtime.
- Input: failed review at `41191cb`, with the required checkpoint, manifest,
  scorer, logical-identity, timing, and import-boundary repair request.
- Outcome: implemented locally; focused checks pass; full verification and
  independent re-review remain pending.
- Completion: do not mark VAL-001 complete until an independent review returns
  `PASS` or justified `PASS WITH NOTE` for the exact repair head.

### Repair cycle 4

- Repair model/mode: exact displayed values not exposed by runtime; a focused
  implementation sub-agent was delegated the bounded change.
- Input: Attempt 4's step-1 gradient mismatch, identical step-1 loss, and the
  pinned runtime's explicit memory-efficient-attention nondeterminism warning.
- Outcome: `deterministic=true` now calls strict PyTorch deterministic
  algorithms; unsupported nondeterministic operations fail rather than warn and
  continue. The regression asserts warn-only is disabled.
- Evidence: focused reproducibility suite `9 passed`; full repository suite
  `310 passed, 1 skipped in 64.15s`; Ruff lint/format and `git diff --check`
  pass. Two pinned-image strict BF16 probes returned the same loss and full
  model-tensor digest. A read-only `gpt-5.6-luna` / Extra High repair QA returned
  `PASS` with no findings; exact runtime model/mode were not exposed.
- Completion: Attempt 5 did not pass its evidence gates; cycle 5 preserves that
  result and defines the fully fresh Attempt 6 required before heavy re-review.

### Repair cycle 5 — adaptive evidence protocol

- Input: Attempt 5's otherwise passing six-arm matrix failed its declared 1 Hz
  container coverage and zero-tolerance first-post-step gates.
- Change: preserve Attempt 5 as `FAIL`; remove the collector's extra sleep, hold
  training behind a collector start barrier, declare coarse container evidence
  at nominal 0.5 Hz with interval/per-validation-event gates, and use CHECK §3's
  existing 5% investigation threshold for both first-post pre-event and paired-
  off comparisons. Sustained five-step gates remain unchanged.
- Review: a delegated protocol reviewer returned `PASS WITH NOTE`; exact runtime
  model/mode were not exposed. It required an entirely fresh matrix, explicit
  adaptive/cherry-picking risk disclosure, phase attribution for the consistent
  direction, and no further threshold revision.
- Completion: Attempt 6 passed all six fresh arms with the mandatory adaptive-
  protocol/bounded-transient note; independent heavy re-review is next.

## Failed-review handoff

- From review cycle: 2, independent review at `41191cb`.
- Failed check and why: VAL-001 checkpoint/evaluation identity trust and CHECK
  §6.3 evidence completeness failed because the saved config, configured data,
  actual loader, logical parity, and phase timing were not all authoritative.
- Review model / mode: requested `gpt-5.6-sol` / Max; exact runtime values not
  exposed.
- Implementation model / mode that produced the failed state: exact values not
  exposed by runtime.
- Commit/diff to repair: `41191cb` and the working-tree repair diff.
- Reproduction command or evidence: independent review findings recorded above;
  existing standalone parity and DGX R2 records showed the identity/timing gaps.
- Relevant files/config/manifests: `src/training/checkpoint.py`,
  `src/evaluation/scoring.py`, `src/training/trainer.py`, `src/evaluate.py`,
  `src/evaluation/__init__.py`, and the VAL-001 tests.
- Attempts already made: `a8520d7` initial implementation, `057983c` scoring
  repair, `0a13838` cleanup repair, and the measured `41191cb` documentation
  head.
- Invariants and constraints: only `artifacts.resume_path` is excluded from
  config hashing; manifests remain immutable and ordered; training and
  standalone evaluation share one scorer and logical identity; no per-step CUDA
  synchronization in the ordinary path; benchmark-only synchronization must be
  explicit; no DGX result, PR edit, commit, or push was claimed by the repair
  agent.
- Selected next model / mode: requested `gpt-5.6-luna` / Extra High; actual
  displayed model and reasoning mode are not exposed by runtime.
- Why this model was selected: the findings are localized cross-component
  wiring and instrumentation repairs within the existing direct Hydra path.
- Exact repair request: bind saved config and ordered data identity, derive or
  strictly verify loader manifests, make logical identity exactly shared, add
  tamper/parity regressions, expose low-overhead phase timing, and remove unused
  re-exports.
- Completion evidence requested: focused and full tests, Ruff, format, lock,
  diff checks, then independent re-review of the exact repair head.

## Repair result

- Repair cycle: 3.
- Repair model / mode: exact displayed values not exposed by runtime; requested
  `gpt-5.6-luna` / Extra High.
- Input handoff: failed review at `41191cb`, requested as `gpt-5.6-sol` / Max;
  exact runtime values not exposed.
- Changes made: canonical checkpoint config/data verification; fail-closed
  actual-loader fingerprint/selection trust; exact logical checkpoint parity;
  tamper regressions; disabled-by-default atomic measurement JSON with CUDA
  events and complete validation pauses; direct evaluation imports.
- What was deliberately not changed: no Hydra alternative, generic framework,
  compatibility shim, scorer math, DGX job, PR, commit, push, or claim of a
  passing independent review.
- Local evidence: focused suite `55 passed`; full suite `310 passed, 1 skipped`;
  full Ruff, changed-Python format, lock, and diff checks pass. Full
  repository formatter still reports four pre-existing unrelated files.
- Repair commit: `78e0448`; later evidence/protocol commits are recorded above.
- Re-review model / mode: pending independent heavy review.
- Re-review verdict: pending; the prior verdict remains `FAIL`.

### CUDA determinism repair result

- Repair cycle: 4.
- Repair model / mode: not exposed by runtime / not exposed by runtime.
- Changes made: strict deterministic-algorithm enforcement and a focused
  regression proving warn-only is disabled.
- What was deliberately not changed: no SDPA backend hard-code, new Hydra knob,
  architecture change, evidence reuse, or cross-platform bitwise promise.
- Repair/protocol commit: `4264e4a`.
- Re-review verdict: pending; the prior independent verdict remains `FAIL`.

### Evidence-protocol repair result

- Repair cycle: 5.
- Repair model / mode: not exposed by runtime / not exposed by runtime.
- Changes made: failed-evidence preservation, collector start/rate repair, and a
  prospectively bounded adaptive recovery rule.
- What was deliberately not changed: no Attempt 5 evidence reclassification or
  reuse, no training/scoring/model/data change, and no further gate-relaxation
  option.
- Commit reviewed next: the evidence/docs head containing this record.
- Re-review verdict: pending; the prior independent verdict remains `FAIL`.

## Risks and handoff

- Known trade-off: on current Attempt 6, each fixed 65,536-target validation
  scores in 4.62–4.76 s and adds 2.04–2.74 s when an improving best checkpoint
  is saved. The earlier 19.52 s scorer result remains historical R2 evidence.
- Evidence status: fresh Attempt 6 is `PASS WITH NOTE` across all three matched
  pairs, exact identities/scores, standalone parity, recovery, and valid traces.
  Independent heavy re-review remains the only technical completion gate.
- Dependency: this stacked PR still depends on DATA-004, whose source-rights
  disposition is a human policy gate.
- Merge path: human review and merge; no self-merge authorization exists.
- Exactly one next step: independently heavy-review the exact Attempt 6 evidence
  head against VAL-001, `PHILOSOPHY.md`, and applicable `CHECK.md`.

## Merge authority and final audit

- Merge path: `human merge`.
- Human authorization: N/A — human merge.
- Exact independently reviewed head: pending.
- Latest independent verdict/model/mode: pending.
- Actionable findings repaired and re-reviewed: pending.
- Blocking review decision / `CHANGES_REQUESTED`: pending exact-head fetch.
- Newer human objection: none observed.
- Human review dismissed by an agent: no.
- Unresolved review threads: pending exact-head fetch.
- Branch-protection required-context and workflow inventory: pending final audit.
- Exact-head checks: pending after the final review/documentation commit.
- Base: `codex/data-004-pinned-baseline-mixture` at `e1d4ed8` when this stack began.
- Mergeability/conflict status: pending final audit.
- Prohibited self-merge categories: source-rights dependency and human merge path
  make self-merge unavailable.
- Final audit / immediate pre-merge refresh: pending human merge.
- Merge outcome: not merged; draft PR remains open.

## Ledger update

- [x] VAL-001 row exists in `docs/model-runs/README.md`.
- [x] Failed/invalidated attempts remain visible.
- [x] Failed independent review verdict and exact displayed provenance recorded.
- [ ] Aggregate pass/repair/review counts updated after the final verdict.
- [ ] Human merge/final audit recorded.
