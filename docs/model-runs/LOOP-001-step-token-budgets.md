# LOOP-001 - Step/token budgets and correct metrics

- PR: [#25](https://github.com/Ayumu-J-S/llm_scratch/pull/25) (draft; pending parent merge)
- Branch: `codex/loop-001-step-token-budgets`
- Ticket: LOOP-001
- Hypothesis: A trainer whose stopping, scheduling, and event decisions use
  explicit optimizer-step and target-token counters will make local and
  streaming runs equivalent while preserving token-weighted objective metrics.
- Experiment record: `N/A` — this pass is a trainer correctness fixture; no
  research run was launched.
- Started: 2026-07-12
- Final verdict: in progress (cycle-3 repair)
- Final record owner: implementation sub-agent `/root/loop001_implementation_retry`

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
  fixtures, Hydra CPU smoke, Ruff, lock, and diff checks.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `fbdb086` | Requested Luna / Extra High implementation pass | complete | Replaced epoch-only averaging with authoritative step/token/time counters, token-weighted NLL, exact token-budget truncation, independent event cadences, scheduler-after-update ordering, finite/empty guards, and W&B-independent JSONL metrics; corrected metric-free scheduler boundaries to avoid forcing validation outside its cadence. | `uv run --group dev pytest -q`: 206 passed, 1 skipped; focused trainer: 4 passed; Ruff and `git diff --check` pass; implementation lineage `a024f24` |
| 1 | review | not exposed by runtime | not exposed by runtime | `5de45e7` | Requested heavier independent Extra Thinking review | FAIL | Found missing post-backward/optimizer non-finite guards, no token-based event cadences, no guaranteed aggregate train/loss/perplexity under step logging, and fractional budget handling that could reach zero-token division. | independent review handoff from `/root/loop001_review` |
| 2 | repair | not exposed by runtime | not exposed by runtime | `5de45e7` | Repair every actionable finding without broadening scope | complete | Added gradient/parameter finite checks with contextual local failure records, direct `*_every_n_tokens` cadences, epoch aggregate loss/perplexity records, strict integer step/token budgets, and zero-token boundary handling. | `e972864`; full suite 209 passed, 1 skipped; focused trainer 7 passed |
| 2 | re-review | not exposed by runtime | not exposed by runtime | `f505ed7` | Requested independent re-review at exact repair head | FAIL | Four prior blockers were repaired, but CHECK 6.2 still found that non-finite validation loss raised without a persisted local event/context record. | reviewer `/root/loop001_review`; full 209 passed, 1 skipped; focused 7; Hydra CPU smoke; Ruff/lock/diff pass |
| 3 | repair | not exposed by runtime | not exposed by runtime | `f505ed7` | Record validation non-finite failures with batch/step/checkpoint context | in progress | Validation now checks per-batch losses, persists `nonfinite_validation` JSONL evidence with batch/step/elapsed and preceding checkpoint when available, restores train mode in a `finally`, and has a regression fixture. | implementation head `0d09af8`; full 210 passed, 1 skipped; focused 8; Ruff/diff pass |
| 3 | re-review | not exposed by runtime | not exposed by runtime | pending final docs head | Re-review exact validation-guard repair | pending | Must verify CHECK 6.2 and retain the earlier PASS WITH NOTE findings. | pending |

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
- Branch/commit: `codex/loop-001-step-token-budgets` / repair head `0d09af8`; final docs head pending.
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
- Complexity / change-surface result: FAIL pending repair
- ML-system result: CPU fixture only; DGX evidence pending later STAB-001
- Verdict: FAIL; repair cycle 2 in progress

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
- Verdict: FAIL; cycle-3 repair in progress

#### Re-review findings

| Severity | Area | Finding | Evidence | Action |
| --- | --- | --- | --- | --- |
| note | token cadence | A cadence threshold crossed by one batch is recorded at the completed batch token count; multiple crossed thresholds coalesce to one event/update. | reviewer report against `f505ed7` | Accepted as explicit batch-boundary semantics; no mid-batch side effects |
| note | scheduler observability | Epoch-summary LR is recorded before an epoch scheduler advances. | reviewer report against `f505ed7` | Accepted; scheduler ordering itself is tested/explicit and next-step metrics observe the updated LR |

### Review cycle 3 (repair re-review)

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: pending exact final head (implementation repair `0d09af8`)
- Selected `CHECK.md` sections: 6.2, plus regression of 6.1–6.3.
- Ticket acceptance result: pending
- Philosophy alignment: pending
- Complexity / change-surface result: pending
- ML-system result: CPU/Hydra smoke only; no DGX claim.
- Verdict: pending independent re-review

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
- Commit reviewed next: `0d09af8` (docs finalization will create a descendant)
- Re-review model / mode: not exposed by runtime / not exposed by runtime
- Re-review verdict: pending

## Final evidence

- Resolved Hydra command/config: composed canonical Hydra profile with CPU
  smoke preflight; config keys added to `config/train.yaml` and accepted by
  `runtime.config` preflight.
- Data/tokenizer/model identity: no data/model training run; fixed-logit CPU
  fixture only.
- Validation and measurements: full suite 210 passed, 1 skipped; trainer
  fixture 8 passed; Hydra CPU smoke passed; Ruff, lock, and diff checks pass.
- Performance/resource result if applicable: N/A.
- Failed attempts retained at: cycle-1 and cycle-2 review findings above;
  repaired at `e972864` and `0d09af8`; cycle-3 independent review pending.
- Known trade-offs: one optimizer update per loader batch; accumulation and
  checkpoint resume remain later tickets.
- Unresolved risks: cycle-3 independent review pending; token cadence is
  batch-boundary based; no DGX or long-run stability evidence is claimed.
- Human decision requested: parent may mark PR #25 ready after final audit and
  merge under the explicitly authorized bounded roadmap process.

## Merge authority and final audit

- Merge path: guarded agent self-merge after parent final audit
- Human authorization: user explicitly authorized self-merge for the bounded
  roadmap goal on 2026-07-12; this implementation sub-agent does not merge.
- Authorization evidence location: parent session and PR #25 body
- Authorization covers this named PR or bounded ticket/goal series: yes — LOOP-001 within the roadmap goal
- Exact independently reviewed head SHA: pending cycle-3 re-review (`0d09af8` implementation repair)
- Latest independent verdict / model / mode: pending / not exposed by runtime / not exposed by runtime
- All actionable findings repaired and independently re-reviewed: pending cycle-3 review
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: cycle-2 validation evidence gap repaired; re-review pending
- Newer human objections since authorization/review: none observed
- Human review dismissed by an agent: no
- Unresolved review threads at final audit: pending parent refresh
- Branch-protection required-context inventory: pending parent GitHub audit
- Applicable configured workflow/check inventory: pending parent GitHub audit
- Observed exact-head check statuses: pending
- Expected checks absent, pending, skipped, cancelled, or non-successful: pending
- No-check evidence when both inventories are empty: pending
- Target branch and base SHA at final audit: `main` / `fbdb086` at review; parent must refresh against current main before merge
- Up-to-date, conflict-free, and mergeable evidence: pending parent refresh
- Record, ledger, PR trail, validation, and risks parity: cycle-3 docs/validation pending re-review
- Prohibited self-merge categories: clear; no secrets/security/deployment action
- Admin/bypass/force/disabled-check requirement: no
- Final audit PR body/comment location: PR #25 body and parent final audit comment
- Final audit changed reviewed head: no
- Immediate pre-merge re-fetch/compare observation location: N/A — sub-agent does not merge
- Immediate refresh compared authorization, head, base, review decision/objections, threads, expected checks/statuses, and mergeability: pending parent
- Drift found: no drift at reviewed head; parent refresh required for base changes
- Merge outcome: pending parent guarded merge

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | implementation + repair | Scoped direct Trainer redesign, failure repair, and invariant fixtures | Exact runtime identity/reasoning mode unavailable; validation evidence gap required cycle-3 repair | LOOP-001, PHILOSOPHY, CHECK sections 6–7, independent FAIL handoffs | pending cycle-3 review |
| not exposed by runtime / not exposed by runtime | independent review | Found four initial blockers and one validation-evidence gap; repair cycles are explicit | Exact runtime identity/reasoning mode unavailable; no DGX claim | Exact `f505ed7`, cycle-3 repair `0d09af8`, 210/1 suite, Hydra smoke | pending cycle-3 review |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [ ] Updated per-model attempt, pass, repair, and review counts after cycle-3 review.
- [x] Confirmed the initial execution trail separates requested/default from actual runtime values.
- [ ] Recorded complete guarded self-merge final audit evidence (parent merge pending).
- [x] Confirmed that this bootstrap policy rule was not used before a human merged it.
