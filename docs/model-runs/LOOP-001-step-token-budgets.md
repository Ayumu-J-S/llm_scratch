# LOOP-001 - Step/token budgets and correct metrics

- PR: draft / pending creation
- Branch: `codex/loop-001-step-token-budgets`
- Ticket: LOOP-001
- Hypothesis: A trainer whose stopping, scheduling, and event decisions use
  explicit optimizer-step and target-token counters will make local and
  streaming runs equivalent while preserving token-weighted objective metrics.
- Experiment record: `N/A` — this pass is a trainer correctness fixture; no
  research run was launched.
- Started: 2026-07-12
- Final verdict: in progress
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
- Intended evidence: 206 passed / 1 skipped full CPU suite, focused trainer
  fixtures, Ruff, and diff checks.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `fbdb086` | Requested Luna / Extra High implementation pass | in progress | Replaced epoch-only averaging with authoritative step/token/time counters, token-weighted NLL, exact token-budget truncation, independent event cadences, scheduler-after-update ordering, finite/empty guards, and W&B-independent JSONL metrics. | `uv run --group dev pytest -q`: 206 passed, 1 skipped; focused trainer: 4 passed; Ruff and `git diff --check` pass |
| 1 | review | not exposed by runtime | not exposed by runtime | pending exact implementation head | Requested heavier independent Extra Thinking review | pending | Review must cover LOOP-001 acceptance criteria and CHECK.md sections 6.1–6.3, 7.1–7.4. | pending |

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
- Branch/commit: `codex/loop-001-step-token-budgets` / implementation head pending
- Phase/role/task path: implementation / LOOP-001 / delegated retry
- Privacy confirmation: no prompts, hidden chain-of-thought, token counts,
  secrets, or raw thread IDs recorded.

## Check selection and verdicts

### Review cycle 1

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: pending
- Selected `CHECK.md` sections: 6.1 (objective, weighting, scheduler), 6.2
  (finite guards), 6.3 (cadence and synchronization), 7.1–7.4 (change
  surface/configuration).
- Major sections marked N/A and why: CHECK 4–5 and 6.4 are N/A until a DGX
  pilot exists; LOOP-001 does not claim throughput or long-run stability.
- Ticket acceptance result: pending independent review
- Philosophy alignment: pending independent review
- Complexity / change-surface result: pending independent review
- ML-system result: CPU fixture only; DGX evidence pending later STAB-001
- Verdict: pending

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| — | — | No independent review findings yet | pending | Run independent review against exact head |

## Failed-review handoff

N/A — no failed review occurred.

## Repair result

N/A — no repair occurred.

## Final evidence

- Resolved Hydra command/config: not launched; config keys added to canonical
  `config/train.yaml` and accepted by `runtime.config` preflight.
- Data/tokenizer/model identity: no data/model training run; fixed-logit CPU
  fixture only.
- Validation and measurements: full suite 206 passed, 1 skipped; trainer
  fixture 4 passed; Ruff and diff checks pass.
- Performance/resource result if applicable: N/A.
- Failed attempts retained at: none.
- Known trade-offs: one optimizer update per loader batch; accumulation and
  checkpoint resume remain later tickets.
- Unresolved risks: independent CHECK review and any integration training run.
- Human decision requested: review the draft PR; merge authority remains with
  the parent goal process.

## Merge authority and final audit

- Merge path: human merge / guarded agent self-merge only after parent audit
- Human authorization: bounded roadmap-series authorization recorded by parent
  session; this PR is not merged by this sub-agent.
- Authorization evidence location: parent session / PR body (pending)
- Authorization covers this named PR or bounded ticket/goal series: pending PR
- Exact independently reviewed head SHA: pending
- Latest independent verdict / model / mode: pending
- All actionable findings repaired and independently re-reviewed: pending
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: pending
- Newer human objections since authorization/review: none observed
- Human review dismissed by an agent: no
- Unresolved review threads at final audit: pending
- Branch-protection required-context inventory: pending parent GitHub audit
- Applicable configured workflow/check inventory: pending parent GitHub audit
- Observed exact-head check statuses: pending
- Expected checks absent, pending, skipped, cancelled, or non-successful: pending
- No-check evidence when both inventories are empty: pending
- Target branch and base SHA at final audit: `main` / `fbdb086`
- Up-to-date, conflict-free, and mergeable evidence: pending PR creation
- Record, ledger, PR trail, validation, and risks parity: pending
- Prohibited self-merge categories: clear; no secrets/security/deployment action
- Admin/bypass/force/disabled-check requirement: no
- Final audit PR body/comment location: pending
- Final audit changed reviewed head: no
- Immediate pre-merge re-fetch/compare observation location: N/A — sub-agent does not merge
- Immediate refresh compared authorization, head, base, review decision/objections, threads, expected checks/statuses, and mergeability: pending parent
- Drift found: pending
- Merge outcome: pending

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | implementation | Scoped direct Trainer redesign and focused invariant fixtures | Exact runtime identity/reasoning mode unavailable; independent review not yet run | LOOP-001, PHILOSOPHY, CHECK sections 6–7, current Trainer/config | in progress |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [ ] Updated per-model attempt, pass, repair, and review counts after review.
- [x] Confirmed the initial execution trail separates requested/default from actual runtime values.
- [ ] Recorded human merge or complete guarded self-merge authority/audit evidence.
- [x] Confirmed that this bootstrap policy rule was not used before a human merged it.
