# <TICKET> - <Short Title>

- PR: draft / URL / unavailable
- Branch:
- Ticket:
- Hypothesis:
- Started:
- Final verdict: in progress / PASS / PASS WITH NOTE / FAIL / blocked
- Final record owner:

## Scope and decision context

- Goal:
- In scope:
- Out of scope:
- Relevant `PHILOSOPHY.md` principles:
- Baseline commit/run:
- Intended evidence:

## Execution timeline

One row represents one model invocation or one clearly bounded phase. Never
delete a failed row.

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | implementation |  | Extra High or actual |  |  |  |  |  |
| 1 | review |  | Extra Thinking or actual |  |  |  |  |  |

Allowed phases: `implementation`, `review`, `repair`, `re-review`, and `handoff`.

## Check selection and verdicts

### Review cycle 1

- Review model / mode:
- Commit reviewed:
- Selected `CHECK.md` sections:
- Major sections marked N/A and why:
- Ticket acceptance result:
- Philosophy alignment:
- Complexity / change-surface result:
- ML-system result:
- Verdict:

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |

## Failed-review handoff

Duplicate this section after every `FAIL`. Write `N/A` when no failed review occurred.

- From review cycle:
- Failed check and why:
- Review model / mode:
- Implementation model / mode that produced the failed state:
- Commit/diff to repair:
- Reproduction command or evidence:
- Relevant files/config/manifests:
- Attempts already made:
- Invariants and constraints:
- Selected next model / mode:
- Why this model was selected:
- Exact repair request:
- Completion evidence requested:

## Repair result

Duplicate this section for every repair.

- Repair cycle:
- Repair model / mode:
- Input handoff:
- Changes made:
- What was deliberately not changed:
- Local evidence:
- Commit reviewed next:
- Re-review model / mode:
- Re-review verdict:

## Final evidence

- Resolved Hydra command/config:
- Data/tokenizer/model identity:
- Validation and measurements:
- Performance/resource result if applicable:
- Failed attempts retained at:
- Known trade-offs:
- Unresolved risks:
- Human decision requested:

## Merge authority and final audit

- Merge path: `human merge` / `guarded agent self-merge`
- Human authorization: exact instruction, scope, date/context, or `N/A — human merge`
- Authorization evidence location:
- Authorization covers this named PR or bounded ticket/goal series: yes / no / N/A
- Exact independently reviewed head SHA:
- Latest independent verdict / model / mode:
- All actionable findings repaired and independently re-reviewed:
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence:
- Newer human objections since authorization/review: none / details
- Human review dismissed by an agent: no / yes (yes blocks self-merge)
- Unresolved review threads at final audit: zero / count
- Branch-protection required-context inventory:
- Applicable configured workflow/check inventory:
- Observed exact-head check statuses:
- Expected checks absent, pending, skipped, cancelled, or non-successful: zero / details
- No-check evidence when both inventories are empty: evidence / N/A
- Target branch and base SHA at final audit:
- Up-to-date, conflict-free, and mergeable evidence:
- Record, ledger, PR trail, validation, and risks parity:
- Prohibited self-merge categories: clear / blocked (state why)
- Admin/bypass/force/disabled-check requirement: no / yes
- Final audit PR body/comment location:
- Final audit changed reviewed head: no / yes (if yes, re-review is required)
- Immediate pre-merge re-fetch/compare observation location:
- Immediate refresh compared authorization, head, base, review decision/objections, threads, expected checks/statuses, and mergeability: yes / no
- Drift found: no / yes (yes aborts merge and requires appropriate update/revalidation/re-review)
- Merge outcome: pending / human merged / agent merged / not merged

## Model assessment from this ticket

Record observable outcomes, not hidden chain-of-thought.

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |

## Ledger update

- [ ] Added the PR/ticket row to `docs/model-runs/README.md`.
- [ ] Updated per-model attempt, pass, repair, and review counts.
- [ ] Confirmed that the PR execution trail matches this record.
- [ ] Recorded human merge or complete guarded self-merge authority/audit evidence.
- [ ] Confirmed that this bootstrap policy rule was not used before a human merged it.
