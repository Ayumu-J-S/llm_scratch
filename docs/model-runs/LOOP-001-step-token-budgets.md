# LOOP-001 - Step/token budgets and correct metrics

- PR: [#25](https://github.com/Ayumu-J-S/llm_scratch/pull/25) (merged at `ea0873d7bee8b3796092bb4be4cdb9ad6d2b7ecb`); follow-up repair [#26](https://github.com/Ayumu-J-S/llm_scratch/pull/26) (merged at `c0bdfed2e618d73c0a0c262053fc842b0594db68`); audit follow-up: [#27](https://github.com/Ayumu-J-S/llm_scratch/pull/27) (draft)
- Branch: `codex/loop-001-step-token-budgets`
- Ticket: LOOP-001
- Hypothesis: A trainer whose stopping, scheduling, and event decisions use
  explicit optimizer-step and target-token counters will make local and
  streaming runs equivalent while preserving token-weighted objective metrics.
- Experiment record: `N/A` — this pass is a trainer correctness fixture; no
  research run was launched.
- Started: 2026-07-12
- Final verdict: PASS WITH NOTE
- Final record owner: implementation sub-agent `/root/loop001_implementation_retry`; merge audit `/root/loop001_audit`

## Scope and decision context

- Goal: implement step/token/time budgets, independent event cadences, and
  numerically correct local metrics.
- In scope: `src/training/trainer.py`, Hydra training-budget keys, config
  preflight, focused boundary and metric tests.
- Out of scope: checkpoint payload/resume, mixed precision, accumulation,
  distributed training, and W&B service behavior.
- Relevant `PHILOSOPHY.md` principles: direct readable training logic, explicit
  measurable work, token-weighted objective, and offline evidence parity.
- Baseline commit: `fbdb08606435f038f11aa1efd673105acd91cf84`
- Intended evidence: 210 passed / 1 skipped full CPU suite, 8 focused trainer
  fixtures, Hydra CPU smoke, Ruff, lock, and diff checks; follow-up metrics
  repair adds 211 passed / 1 skipped, 9 focused fixtures, and exact merge-gate
  observations for PRs #25 and #26.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `fbdb086` | Requested Luna / Extra High implementation pass | complete | Replaced epoch-only averaging with authoritative step/token/time counters, token-weighted NLL, exact token-budget truncation, independent event cadences, scheduler-after-update ordering, finite/empty guards, and W&B-independent JSONL metrics; corrected metric-free scheduler boundaries to avoid forcing validation outside its cadence. | `uv run --group dev pytest -q`: 206 passed, 1 skipped; focused trainer: 4 passed; Ruff and `git diff --check` pass; implementation lineage `a024f24` |
| 1 | review | not exposed by runtime | not exposed by runtime | `5de45e7` | Requested heavier independent Extra Thinking review | FAIL | Found missing post-backward/optimizer non-finite guards, no token-based event cadences, no guaranteed aggregate train/loss/perplexity under step logging, and fractional budget handling that could reach zero-token division. | independent review handoff from `/root/loop001_review` |
| 2 | repair | not exposed by runtime | not exposed by runtime | `5de45e7` | Repair every actionable finding without broadening scope | complete | Added gradient/parameter finite checks with contextual local failure records, direct `*_every_n_tokens` cadences, epoch aggregate loss/perplexity records, strict integer step/token budgets, and zero-token boundary handling. | `e972864`; full suite 209 passed, 1 skipped; focused trainer 7 passed |
| 2 | re-review | not exposed by runtime | not exposed by runtime | `f505ed7` | Requested independent re-review at exact repair head | FAIL | Four prior blockers were repaired, but CHECK 6.2 still found that non-finite validation loss raised without a persisted local event/context record. | reviewer `/root/loop001_review`; full 209 passed, 1 skipped; focused 7; Hydra CPU smoke; Ruff/lock/diff pass |
| 3 | repair | not exposed by runtime | not exposed by runtime | `f505ed7` | Record validation non-finite failures with batch/step/checkpoint context | complete | Validation now checks per-batch losses, persists `nonfinite_validation` JSONL evidence with batch/step/elapsed and preceding checkpoint when available, restores train mode in a `finally`, and has a regression fixture. | implementation head `0d09af8`; full 210 passed, 1 skipped; focused 8; Ruff/diff pass |
| 3 | re-review | not exposed by runtime | not exposed by runtime | `94da0d4` | Re-review exact validation-guard repair | PASS WITH NOTE | Validation now persists non-finite loss context; all LOOP-001 acceptance criteria and CHECK 6.1–6.3/7.1–7.4 evidence pass. Notes: token cadence is batch-boundary based and epoch-summary LR is pre-scheduler. | reviewer `/root/loop001_review`; full 210 passed, 1 skipped; focused 8; canonical streaming smoke; Ruff/lock/diff pass |
| 4 | follow-up repair | not exposed by runtime | not exposed by runtime | `ea0873d7` | Repair the post-merge P2 metrics-lifecycle finding from PR #25 | complete | `Trainer.fit()` now clears in-memory metrics and truncates `metrics.jsonl` at each fresh run boundary; a same-directory two-run regression prevents stale W&B-off evidence from mixing. | normative implementation `a332f46`; full 211 passed, 1 skipped; focused 9; Ruff/lock/diff pass |
| 4 | follow-up re-review | not exposed by runtime | not exposed by runtime | `98f609f` | Independent review of the exact metrics-lifecycle repair head | PASS WITH NOTE | P2 repair is scoped and correct. Note: repeated `fit()` on one Trainer does not reset counters; fresh Trainer instances define run boundaries and CKPT-001 owns resume lifecycle. | review `4679826123`; exact docs head `e21dc10`; full 211 passed, 1 skipped; focused 9 |
| 5 | merge audit / record finalization | not exposed by runtime | not exposed by runtime | `c0bdfed2` | Record PR #25 and PR #26 merges, resolved P2 thread, exact-head parity, and roadmap state | complete | PR #25 merged at `ea0873d7`; PR #26 merged at expected head `e21dc10` with merge `c0bdfed2`; no unresolved threads, workflow runs, or exact-head statuses were observed. | docs-only audit PR #27; local diff/lock/Ruff checks and merged PR evidence |

## Runtime provenance block

Requested/default values are not actual runtime identity. The current runtime
display does not expose the exact deployment model or reasoning mode.

| Namespace | Product | Displayed family | Exact model identifier | Reasoning mode | Source / unavailable reason |
| --- | --- | --- | --- | --- | --- |
| requested | Codex | Luna | not exposed by runtime | Extra High | user/AGENTS implementation request |
| actual | Codex | GPT-5 | not exposed by runtime | not exposed by runtime | runtime does not expose exact deployment or reasoning display |

- Capture file/evidence: `N/A` — no raw runtime capture is available in the
  delegated implementation session; values above follow repository provenance
  rules.
- Codex CLI version: not exposed by runtime
- Branch/commit: implementation `codex/loop-001-step-token-budgets`; independently reviewed implementation anchor `94da0d4`; merged PR #25 docs head `283fc0913d7fe8c8295ad03074d9cdf8b8b0bbb1`; follow-up PR #26 normative repair `a332f46`, independently reviewed docs head `98f609f`, and merged docs head `e21dc10d8e58a2407cd455b2e0a48a97c356fecf`.
- Phase/role/task path: implementation / LOOP-001 / delegated retry
- Privacy confirmation: no prompts, hidden chain-of-thought, token counts,
  secrets, or raw thread IDs recorded.

## Check selection and verdicts

### Review cycle 1 (initial implementation)

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: `5de45e7`
- Selected `CHECK.md` sections: 6.1 (objective, weighting, scheduler), 6.2
  (finite guards), 6.3 (cadence and synchronization), 7.1–7.4 (change
  surface/configuration).
- Major sections marked N/A and why: CHECK 4–5 and 6.4 are N/A until a DGX
  pilot exists; LOOP-001 does not claim throughput or long-run stability.
- Ticket acceptance result: FAIL — see failed-review handoff below
- Philosophy alignment: direct scope retained, but numerical and token-boundary gaps blocked acceptance
- Complexity / change-surface result: FAIL (historical; repaired in cycle 2)
- ML-system result: CPU fixture only; DGX evidence pending later STAB-001
- Verdict: FAIL (historical; repaired in cycle 2)

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| high | numerical health | finite loss alone did not prevent NaN gradients/parameters | review of `5de45e7`; CHECK 6.2 | Guard gradients and parameters, record batch/step, stop before counters advance |
| high | event boundaries | only step cadences existed; token cadence acceptance was unimplemented | review of `5de45e7`; LOOP-001 | Add independent `*_every_n_tokens` keys and boundary tests |
| medium | metrics | step logging could omit token-weighted aggregate train/loss/perplexity | review of `5de45e7`; CHECK 6.1 | Emit epoch aggregate metrics independently of step logging |
| medium | config/stop | fractional max token/step values could truncate and hit zero-token division | review of `5de45e7` | Require positive integers and guard no-remaining-token path |

### Review cycle 2 (repair re-review)

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: `f505ed7` (normative implementation plus docs lineage)
- Selected `CHECK.md` sections: 6.1–6.3 and 7.1–7.4; CHECK 4–5 and 6.4
  remain N/A because this ticket makes no DGX throughput or long-run claim.
- Ticket acceptance result: FAIL — validation non-finite failures lacked local
  persisted context even though training-side guards passed.
- Philosophy alignment: PASS — direct trainer/config changes, no checkpoint or
  distributed scope creep, and explicit offline metrics.
- Complexity / change-surface result: PASS WITH NOTE — token cadence is checked
  at completed batch boundaries; no mid-batch event machinery was introduced.
- ML-system result: PASS WITH NOTE — CPU/Hydra smoke only; DGX behavior remains
  a later STAB-001 concern.
- Verdict: FAIL (historical; repaired in cycle 3)

#### Re-review findings

| Severity | Area | Finding | Evidence | Action |
| --- | --- | --- | --- | --- |
| note | token cadence | A cadence threshold crossed by one batch is recorded at the completed batch token count; multiple crossed thresholds coalesce to one event/update. | reviewer report against `f505ed7` | Accepted as explicit batch-boundary semantics; no mid-batch side effects |
| note | scheduler observability | Epoch-summary LR is recorded before an epoch scheduler advances. | reviewer report against `f505ed7` | Accepted; scheduler ordering itself is tested/explicit and next-step metrics observe the updated LR |

### Review cycle 3 (repair re-review)

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: `94da0d4`
- Selected `CHECK.md` sections: 6.2, plus regression of 6.1–6.3.
- Ticket acceptance result: PASS — budgets, weighted metrics, independent
  cadences, scheduler ordering, and numeric guards are covered.
- Philosophy alignment: PASS — direct Hydra/trainer changes with no checkpoint,
  AMP, or distributed scope creep.
- Complexity / change-surface result: PASS WITH NOTE — token events are emitted
  after completed batch threshold crossings; epoch-summary LR is pre-scheduler.
- ML-system result: PASS WITH NOTE — CPU and canonical streaming smoke only;
  no DGX/long-run claim.
- Verdict: PASS WITH NOTE

## Failed-review handoff

### Failed-review handoff — cycle 1

- From review cycle: 1
- Failed check and why: CHECK 6.1/6.2/6.3 and LOOP-001 acceptance were not
  satisfied for numeric state, token event boundaries, aggregate metrics, and
  fractional budget safety.
- Review model / mode: not exposed by runtime / not exposed by runtime
- Implementation model / mode that produced the failed state: not exposed by
  runtime / not exposed by runtime
- Commit/diff to repair: `5de45e7`
- Reproduction command or evidence: independent review findings; focused tests
  lacked non-finite-gradient and token-cadence coverage.
- Relevant files/config/manifests: `src/training/trainer.py`,
  `src/runtime/config.py`, `config/train.yaml`, `tests/test_trainer.py`.
- Attempts already made: initial implementation at `5c537bd`, scheduler-cadence
  repair/docs at `a024f24` and `5de45e7`.
- Invariants and constraints: token-weighted NLL, exact authoritative counters,
  no counter advancement on unsafe updates, W&B-independent local evidence,
  Hydra-only configuration, no checkpoint/resume scope expansion.
- Selected next model / mode: not exposed by runtime / not exposed by runtime
- Why this model was selected: delegated repair with the complete independent
  findings and ticket acceptance context.
- Exact repair request: implement all four findings, add focused regressions,
  run full CPU suite/lint, and re-review the exact final head.
- Completion evidence requested: finite-gradient/parameter guard fixture,
  token cadence boundary fixture, aggregate train/loss/perplexity fixture,
  strict integer budget tests, full suite and Ruff.

### Failed-review handoff — cycle 2

- From review cycle: 2
- Failed check and why: CHECK 6.2 required non-finite validation failures to
  leave a persisted event with batch/step context and a preceding-checkpoint
  reference when available; the implementation raised without that record.
- Review model / mode: not exposed by runtime / not exposed by runtime
- Implementation model / mode that produced the failed state: not exposed by
  runtime / not exposed by runtime
- Commit/diff to repair: `f505ed7`
- Reproduction command or evidence: independent re-review of `f505ed7`.
- Relevant files/config/manifests: `src/training/trainer.py`,
  `tests/test_trainer.py`, and `docs/model-runs/LOOP-001-step-token-budgets.md`.
- Attempts already made: cycle-1 repair at `e972864`; cycle-2 re-review PASS
  WITH NOTE except for this validation evidence gap.
- Invariants and constraints: stop safely on non-finite validation, preserve
  local W&B-off evidence, restore training mode, and keep LOOP-001 scope.
- Selected next model / mode: not exposed by runtime / not exposed by runtime
- Why this model was selected: delegated focused repair with the exact review
  finding and CHECK 6.2 context.
- Exact repair request: persist a `nonfinite_validation` event with optimizer
  step, target tokens, elapsed time, validation batch index, and checkpoint
  context before raising; add a regression and re-review exact head.
- Completion evidence requested: full 210/1 suite, focused validation fixture,
  Ruff, and independent PASS/PASS WITH NOTE.

## Repair result

### Repair cycle 2

- Repair model / mode: not exposed by runtime / not exposed by runtime
- Input handoff: cycle-1 FAIL at `5de45e7`
- Changes made: finite gradient/parameter checks with persisted failure events;
  direct step/token event cadence support; epoch aggregate metrics; strict
  positive-integer max step/token validation and no-remaining-token guard;
  regression tests for each finding.
- What was deliberately not changed: checkpoint payload/resume, AMP,
  accumulation, distributed execution, and W&B service semantics.
- Local evidence: full `uv run --group dev pytest -q` 209 passed, 1 skipped;
  focused trainer 7 passed; Ruff and diff checks pass.
- Commit reviewed next: `f505ed7` (normative repair head; cycle-2 validation repair follows)
- Re-review model / mode: not exposed by runtime / not exposed by runtime
- Re-review verdict: FAIL — validation evidence gap; handed off as cycle 2.

### Repair cycle 3

- Repair model / mode: not exposed by runtime / not exposed by runtime
- Input handoff: cycle-2 FAIL at `f505ed7`
- Changes made: validation now detects non-finite per-batch losses, records a
  `nonfinite_validation` JSONL event with batch/step/elapsed and preceding
  checkpoint context when available, restores train mode in `finally`, and
  includes a regression test.
- What was deliberately not changed: training objective, cadence semantics,
  scheduler ordering, checkpoint payload/resume, AMP, and distributed scope.
- Local evidence: full `uv run --group dev pytest -q` 210 passed, 1 skipped;
  focused trainer 8 passed; Ruff and diff checks pass.
- Commit reviewed next: `94da0d4` (docs-only finalization descendant of implementation repair `0d09af8`)
- Re-review model / mode: not exposed by runtime / not exposed by runtime
- Re-review verdict: PASS WITH NOTE

## Final evidence

- Resolved Hydra command/config: composed canonical Hydra profile with CPU
  smoke preflight; config keys added to `config/train.yaml` and accepted by
  `runtime.config` preflight.
- Data/tokenizer/model identity: no data/model training run; fixed-logit CPU
  fixture only.
- Validation and measurements: PR #25 full suite 210 passed, 1 skipped with
  trainer fixture 8 passed; PR #26 full suite 211 passed, 1 skipped with
  trainer fixture 9 passed; canonical streaming Hydra CPU smoke passed; Ruff,
  lock, and diff checks pass for both.
- Performance/resource result if applicable: N/A.
- Failed attempts retained at: cycle-1 and cycle-2 review findings above;
  repaired at `e972864` and `0d09af8`; cycle-3 re-review PASS WITH NOTE at
  `94da0d4`; the post-merge P2 metrics-lifecycle finding is retained in
  `LOOP-001-metrics-audit.md` and repaired at `a332f46`.
- Known trade-offs: one optimizer update per loader batch; accumulation and
  checkpoint resume remain later tickets.
- Unresolved risks: token cadence is batch-boundary based; no DGX or long-run
  stability evidence is claimed in LOOP-001.
- Human decision requested: review the docs-only audit PR #27; PRs #25 and #26
  are already merged under the explicitly authorized bounded roadmap process.

## Merge authority and final audit

- Merge path: guarded agent self-merge after exact-head audit
- Human authorization: user explicitly authorized self-merge for the bounded
  roadmap goal on 2026-07-12; this implementation sub-agent does not merge.
- Authorization evidence location: parent session and PR #25/#26 bodies
- Authorization covers this named PR or bounded ticket/goal series: yes — LOOP-001 within the roadmap goal
- Exact independently reviewed head SHA: PR #25 implementation anchor `94da0d4`; PR #26 repair docs head `98f609f`
- Latest independent verdict / model / mode: PASS WITH NOTE / not exposed by runtime / not exposed by runtime (review `4679826123`)
- All actionable findings repaired and independently re-reviewed: yes — PR #25's P2 metrics-lifecycle finding was repaired in PR #26 and independently re-reviewed.
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: none observed on PR #25 or #26
- Newer human objections since authorization/review: none observed
- Human review dismissed by an agent: no
- Unresolved review threads at final audit: zero — PR #25 thread `PRRT_kwDORqx5mc6QMBn5` is resolved; PR #26 has no inline threads.
- Branch-protection required-context inventory: not exposed by the connected GitHub surface; no required-context inventory returned.
- Applicable configured workflow/check inventory: no workflow runs returned for exact PR #25 head `283fc0913d7fe8c8295ad03074d9cdf8b8b0bbb1` or PR #26 docs head `e21dc10d8e58a2407cd455b2e0a48a97c356fecf`.
- Observed exact-head check statuses: empty combined status for both reviewed heads.
- Expected checks absent, pending, skipped, cancelled, or non-successful: none observed; no-check evidence is limited to the connector's empty inventories and does not infer repository policy.
- No-check evidence when both inventories are empty: recorded with the connector evidence limitation above.
- Target branch and base SHA at final audit: PR #25 `main` / `fbdb08606435f038f11aa1efd673105acd91cf84`; PR #26 `main` / `ea0873d7bee8b3796092bb4be4cdb9ad6d2b7ecb`; audit PR #27 `main` / `c0bdfed2e618d73c0a0c262053fc842b0594db68`.
- Up-to-date, conflict-free, and mergeable evidence: PR #25 and PR #26 were closed/merged by the connector at the expected reviewed heads; audit branch is based directly on current `origin/main` `c0bdfed2` and is docs-only.
- Record, ledger, PR trail, validation, and risks parity: complete for PRs #25 and #26; this audit updates the roadmap and both LOOP records.
- Prohibited self-merge categories: clear; no secrets/security/deployment action
- Admin/bypass/force/disabled-check requirement: no
- Final audit PR body/comment location: PR #27 (draft; exact final head recorded in live PR metadata)
- Final audit changed reviewed head: no — docs-only follow-up after PR #25 and PR #26 merges
- Immediate pre-merge re-fetch/compare observation location: parent final refresh before merging PR #27
- Immediate refresh compared authorization, head, base, review decision/objections, threads, expected checks/statuses, and mergeability: PR #25/#26 observations recorded above; PR #27 refresh remains parent merge gate.
- Drift found: none between PR #25/#26 reviewed heads and their merge outcomes; no implementation drift in audit branch.
- Merge outcome: PR #25 `ea0873d7...` and PR #26 `c0bdfed2...` merged; PR #27 audit pending.

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | implementation + repair | Scoped direct Trainer redesign, failure repair, and invariant fixtures | Exact runtime identity/reasoning mode unavailable; token cadence remains batch-boundary based | LOOP-001, PHILOSOPHY, CHECK sections 6–7, independent FAIL handoffs | PASS WITH NOTE |
| not exposed by runtime / not exposed by runtime | independent review | Found four initial blockers and one validation-evidence gap, then verified all repairs | Exact runtime identity/reasoning mode unavailable; no DGX claim | Exact `94da0d4`, 210/1 suite, focused8, canonical streaming smoke | PASS WITH NOTE |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated per-model attempt, pass, repair, and review counts after cycle-3 review.
- [x] Confirmed the initial execution trail separates requested/default from actual runtime values.
- [x] Recorded complete guarded self-merge final audit evidence for PR #25 and follow-up PR #26; docs-only audit PR #27 remains the current handoff.
- [x] Confirmed that this bootstrap policy rule was not used before a human merged it.
