## Ticket and hypothesis

- Roadmap ticket:
- Hypothesis / expected result:
- Why this is the smallest coherent change:

## Scope

- In scope:
- Out of scope:
- Baseline commit/run:

## Model execution trail — required

- Detailed record: `docs/model-runs/<ticket>-<slug>.md`

| Cycle | Phase | Exact model identifier | Reasoning mode | Outcome | Important finding or change |
| ---: | --- | --- | --- | --- | --- |

- [ ] Every implementation, review, repair, and re-review model is listed.
- [ ] Model IDs and modes are copied from the runtime, not inferred. Unavailable values say `not exposed by runtime`.
- [ ] Failed cycles and their handoffs remain in the detailed record.
- [ ] `docs/model-runs/README.md` summary and aggregate counts are updated.

## Implementation

- What changed:
- Important design decisions:
- Failed attempts:
- Resolved Hydra command/config:

## Post-implementation review — required

- Review model / mode:
- Commit reviewed:
- Relevant `PHILOSOPHY.md` principles:
- Selected `CHECK.md` sections:
- Important sections marked N/A and why:
- Ticket acceptance result:
- Complexity / change-surface result:
- ML-system result:
- Final verdict: `PASS` / `PASS WITH NOTE` / `FAIL`

## Review failures and repairs

For every failed review, state:

- what was wrong and where
- evidence or reproduction path
- implementation model that produced it
- repair model selected and why
- context handed to the repair model
- resulting change
- independent re-review model and verdict

Write `N/A — first review passed` only when no repair cycle occurred.

## Validation and evidence

The `CHECK.md` review does not automatically require new generic tests. List the ticket-required checks,
real ML-system observations, measurements, and any tests that were actually necessary.

- Commands:
- Acceptance evidence:
- Performance/resource evidence if applicable:
- W&B/checkpoint/trace/log identifiers:
- Train/validation/data integrity evidence if applicable:

## Risks and handoff

- Known trade-offs:
- Unresolved uncertainty:
- Human decision requested:
- Next useful hypothesis:

- [ ] The PR is not marked ready while the latest model review is `FAIL`.
- [ ] A human remains the merge authority.
