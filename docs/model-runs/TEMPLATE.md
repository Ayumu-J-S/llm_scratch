# Historical Review Record Template

The `docs/model-runs/` directory name is historical. New work does not record
agent runtime identity or reasoning settings. The pull request is the required
live handoff; this template may be used only when a separate durable review and
repair record is useful.

# <TICKET> — <Short Title>

- PR: draft / URL / unavailable
- Branch:
- Ticket:
- Hypothesis:
- Experiment record: `docs/experiments/<ticket>-<slug>.md` / `N/A` with reason
- Started:
- Final verdict: in progress / PASS / PASS WITH NOTE / FAIL / blocked
- Record owner:

## Scope and decision context

- Goal:
- In scope:
- Out of scope:
- Relevant `PHILOSOPHY.md` principles:
- Baseline commit/run:
- Intended evidence:

## Implementation and review timeline

Never delete a failed or blocked row.

| Cycle | Phase | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- |

Allowed phases: `implementation`, `review`, `repair`, `re-review`, and `handoff`.

## Check selection and verdicts

### Review cycle <N>

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

Duplicate this section after every `FAIL`. Write `N/A` when no failed review
occurred.

- From review cycle:
- Failed check and why:
- Reproduction command and evidence:
- Relevant repository context and resolved Hydra config:
- Invariants and constraints to preserve:
- Previous repair attempts:
- Exact repair request:
- Required completion evidence:

## Repair cycle <N>

- Finding addressed:
- Change made:
- Validation rerun:
- Remaining risk:

## Re-review

- Commit reviewed:
- Prior findings disposition:
- New findings:
- Verdict:
- Evidence:

## Merge authority and guarded audit

- Merge path: human merge / guarded agent self-merge
- Human authorization and scope, or `N/A — human merge`:
- Exact reviewed head:
- Final review verdict:
- Actionable findings repaired and re-reviewed:
- Blocking review decision / newer human objection:
- Unresolved review threads:
- Required-context and configured-workflow inventory:
- Exact-head check statuses:
- Current base and mergeable evidence:
- PR trail, validation, risks, and authorization parity:
- Prohibited self-merge categories: clear / blocked (state why)
- Admin/bypass/force/disabled-check requirement: no / yes
- Final audit PR body/comment location:
- Immediate pre-merge refresh location and drift result:
- Merge outcome: pending / human merged / agent merged / not merged
