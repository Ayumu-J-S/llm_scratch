## Ticket and hypothesis

- Roadmap ticket:
- Experiment record: `docs/experiments/<ticket>-<slug>.md` / `N/A` with reason
- Hypothesis:
- Expected result:
- Predeclared success, failure, and stop conditions:
- Predeclared elapsed-time budget:
- Why this is the smallest coherent change:

## Scope

- In scope:
- Out of scope:
- Baseline commit/run:

## Implementation, review, and repair trail — required

Never delete a failed or blocked cycle.

| Cycle | Phase | Input commit/context | Outcome | Important finding or change | Evidence |
| ---: | --- | --- | --- | --- | --- |

- [ ] Every implementation, review, repair, and re-review cycle is listed.
- [ ] Failed cycles and their handoffs remain visible.
- [ ] The latest review covers the exact proposed head.

## Implementation

- What changed:
- Important design decisions:
- Failed attempts:
- Resolved Hydra command/config:

For every consequential attempt, including negative and aborted attempts:

| Attempt | Exact command | Fully resolved Hydra config/path | Evidence link | Outcome |
| ---: | --- | --- | --- | --- |

- [ ] Failed/negative attempts retain their attempted config and evidence.

## Post-implementation review — required

- Commit reviewed:
- Relevant `PHILOSOPHY.md` principles:
- Selected `CHECK.md` sections:
- Important sections marked N/A and why:
- Ticket acceptance result:
- Complexity / change-surface result:
- ML-system result:
- Final verdict: `PASS` / `PASS WITH NOTE` / `FAIL` / `blocked`

## Review failures and repairs

For every failed review, state:

- what was wrong and where;
- evidence or reproduction path;
- context handed to the repair pass;
- resulting change; and
- re-review verdict and evidence.

Write `N/A — first review passed` only when no repair cycle occurred.

## Validation and evidence

The `CHECK.md` review does not automatically require new generic tests. List
the ticket-required checks, real ML-system observations, measurements, and any
tests that were actually necessary.

- Commands:
- Acceptance evidence:
- Performance/resource evidence if applicable:
- W&B/checkpoint/trace/log identifiers:
- Train/validation/data integrity evidence if applicable:
- Research-integrity checks (data separation, leakage, weight/target provenance):
- Conclusion against the predeclared conditions:

## Risks and handoff

- Known trade-offs:
- Unresolved uncertainty:
- Human decision requested:
- Exactly one next question or step:

- [ ] The PR is not marked ready while the latest review is `FAIL` or `blocked`.

## Merge authority and final audit — required

- Merge path: `human merge` / `guarded agent self-merge`
- Human authorization: exact instruction, scope, date/context, or `N/A — human merge`
- Authorization covers this named PR or bounded ticket/goal series: yes / no / N/A
- Exact reviewed head SHA:
- Latest review verdict:
- Actionable findings repaired and re-reviewed:
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence:
- Newer human objections since authorization/review: none / details
- Human review dismissed by an agent: no / yes (yes blocks self-merge)
- Unresolved review threads: zero / count
- Branch-protection required-context inventory:
- Applicable configured workflow/check inventory:
- Observed exact-head check statuses:
- Expected checks absent, pending, skipped, cancelled, or non-successful: zero / details
- No-check evidence when both inventories are empty: evidence / N/A
- Target branch and current base SHA:
- Up-to-date, conflict-free, and mergeable evidence:
- PR trail, validation evidence, risks, and authorization agree:
- Prohibited self-merge categories reviewed: clear / blocked (state why)
- Admin, protection bypass, force merge, or disabled checks required: no / yes
- Final audit recorded at (PR body/comment URL or `pending human merge`):
- Immediate pre-merge re-fetch/compare observation: authorization, head, base, review decision/objections, threads, expected checks/statuses, and mergeability; URL / pending
- Drift found by immediate pre-merge refresh: no / yes (yes aborts merge and requires appropriate update/revalidation/re-review)

- [ ] Human merge remains the default unless the recorded authorization is explicit and in scope.
- [ ] Tool access, authorship, and self-review were not treated as authorization.
- [ ] No human review was dismissed by an agent to clear a merge gate.
- [ ] No absent, pending, skipped, cancelled, failed, or otherwise non-successful expected check was waived.
- [ ] The final audit did not create an unreviewed head commit.
