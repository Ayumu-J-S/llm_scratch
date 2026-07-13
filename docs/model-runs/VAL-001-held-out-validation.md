# VAL-001 - Trustworthy Lightweight Held-Out Validation

- PR: draft / pending URL
- Branch: `codex/val-001-held-out-validation`
- Ticket: `VAL-001`
- Hypothesis: one shared token-weighted scorer gives identical training-time and
  standalone checkpoint results with complete immutable evaluation identity.
- Experiment record: `docs/experiments/VAL-001-held-out-validation.md`
- Started: 2026-07-13
- Final verdict: in progress
- Final record owner: implementation agent

## Scope and decision context

- Goal: implement fixed Japanese/English held-out validation without conflating
  memorization with generalization.
- In scope: shared scoring, per-corpus and aggregate NLL/perplexity, step/token
  cadence, standalone checkpoint evaluation, local JSON, optional compact W&B
  summary, and immutable result identity.
- Out of scope: generative benchmarks, human evaluation, and reserved tests.
- Relevant `PHILOSOPHY.md` principles: Evaluation is part of training;
  experiments are first-class artifacts; fixed step/token cadences; reproducible
  data and scorer identities.
- Baseline commit/run: stacked DATA-004 head
  `e1d4ed8af98de84a3393cd0f6e517f9daf649138`.
- Intended evidence: known-logit NLL, scoring parity, cadence, overlap rejection,
  checkpoint milestones, exact identities, focused/full tests, and applicable
  `CHECK.md` 6.1, 8.2, and 6.3 review.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `e1d4ed8` plus VAL-001 ticket | Implement the smallest coherent VAL-001 change | in progress | Audit and live-PR scaffold started | This record and draft PR |

## Runtime provenance block

| Namespace | Product | Displayed family | Exact model identifier | Reasoning mode | Source / unavailable reason |
| --- | --- | --- | --- | --- | --- |
| requested | Codex | GPT-5.6-class implementation model | not exposed by runtime | Extra High or higher requested | User and repository workflow |
| actual | Codex | not exposed by runtime | not exposed by runtime | not exposed by runtime | Collaboration runtime exposes no model/mode selector or display |

- Capture file/evidence: pending
- Codex CLI version: unavailable; `gh` is also unavailable, so PR operations use the connected GitHub integration
- Branch/commit: `codex/val-001-held-out-validation` / pending scaffold commit
- Phase/role/task path: implementation / primary and delegated audits
- Privacy confirmation: no prompts, hidden chain-of-thought, token counts, secrets, or raw thread ID recorded.

## Check selection and verdicts

### Review cycle 1

- Review model / mode: pending independent heavier review
- Commit reviewed: pending
- Selected `CHECK.md` sections: 6.1, 8.2, 6.3
- Major sections marked N/A and why: pending final review
- Ticket acceptance result: pending
- Philosophy alignment: pending
- Complexity / change-surface result: pending
- ML-system result: pending
- Verdict: pending

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| pending | pending | Independent review has not run | pending | Complete implementation first |

## Failed-review handoff

N/A — first review has not run.

## Repair result

N/A — no review repair cycle yet.

## Final evidence

- Resolved Hydra command/config: pending
- Data/tokenizer/model identity: pending
- Validation and measurements: pending
- Performance/resource result if applicable: bounded CPU fixture; pending cadence/pause observation
- Failed attempts retained at: this record and experiment record
- Known trade-offs: stacked PR depends on DATA-004 and will require retarget/rebase after dependency merge
- Unresolved risks: implementation and independent review pending
- Human decision requested: human review and merge after technical PASS/PASS WITH NOTE

## Merge authority and final audit

- Merge path: `human merge`
- Human authorization: N/A — human merge
- Authorization evidence location: N/A
- Authorization covers this named PR or bounded ticket/goal series: N/A
- Exact independently reviewed head SHA: pending
- Latest independent verdict / model / mode: pending
- All actionable findings repaired and independently re-reviewed: pending
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: pending
- Newer human objections since authorization/review: none observed
- Human review dismissed by an agent: no
- Unresolved review threads at final audit: pending
- Branch-protection required-context inventory: pending
- Applicable configured workflow/check inventory: pending
- Observed exact-head check statuses: pending
- Expected checks absent, pending, skipped, cancelled, or non-successful: pending
- No-check evidence when both inventories are empty: N/A
- Target branch and base SHA at final audit: stacked on `codex/data-004-pinned-baseline-mixture` / `e1d4ed8`
- Up-to-date, conflict-free, and mergeable evidence: pending
- Record, ledger, PR trail, validation, and risks parity: pending
- Prohibited self-merge categories: human merge selected
- Admin/bypass/force/disabled-check requirement: no
- Final audit PR body/comment location: pending
- Final audit changed reviewed head: pending
- Immediate pre-merge re-fetch/compare observation location: pending human merge
- Immediate refresh compared authorization, head, base, review decision/objections, threads, expected checks/statuses, and mergeability: no — pending
- Drift found: pending
- Merge outcome: pending

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | implementation | in progress | in progress | VAL-001 ticket, philosophy, applicable CHECK sections | in progress |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [ ] Updated per-model attempt, pass, repair, and review counts.
- [ ] Confirmed that the PR execution trail matches this record.
- [ ] Recorded human merge or complete guarded self-merge authority/audit evidence.
- [x] Confirmed that the bootstrap policy rule was not used before a human merged it.
