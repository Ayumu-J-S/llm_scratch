# <TICKET> - <Short Title>

- PR: draft / URL / unavailable
- Branch:
- Ticket:
- Hypothesis:
- Experiment record: `docs/experiments/<ticket>-<slug>.md` / `N/A` with reason
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

The experiment record is the run/scientific evidence; this file is the model
implementation and independent-review provenance. Cross-link both whenever a
PR contains a consequential run.

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

## Model assessment from this ticket

Record observable outcomes, not hidden chain-of-thought.

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |

## Ledger update

- [ ] Added the PR/ticket row to `docs/model-runs/README.md`.
- [ ] Updated per-model attempt, pass, repair, and review counts.
- [ ] Confirmed that the PR execution trail matches this record.
