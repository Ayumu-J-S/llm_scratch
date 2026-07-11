# EXP-001 — Experiment and PR Handoff Contract

- Current PR: [#10](https://github.com/Ayumu-J-S/llm_scratch/pull/10)
- Related process PR: [#9](https://github.com/Ayumu-J-S/llm_scratch/pull/9)
- Branch: `codex/exp-001-review-record`
- Ticket: `EXP-001`
- Hypothesis: one compact, append-only experiment record can make positive and
  negative run handoffs reconstructible by a fresh agent
- Experiment record: `docs/experiments/EXP-001-review-record.md`
- Started: 2026-07-11
- Final verdict: in progress — independent review pending
- Final record owner: current implementation agent; exact identity not exposed

## Scope and decision context

- Goal: define the one-ticket/branch/hypothesis convention, experiment record,
  fixture dry run, and PR linkage required by EXP-001.
- In scope: documentation and templates only.
- Out of scope: runtime code, Hydra profile implementation, roadmap edits, merge
  automation, and claims about model quality.
- Relevant philosophy: experiments are first-class software artifacts;
  negative results are retained; budgets and identities are predeclared; agents
  produce a human-legible handoff; a human remains merge authority.
- Baseline: `none — readiness gates unmet`, evidenced by `origin/main` at
  `a05eb1de5656643757a1c3d98047c98dedea8bfa` and the roadmap readiness decision.
- Intended evidence: field-coverage scan, fixture negative-attempt audit,
  `git diff --check`, and an independent review of applicable `CHECK.md`
  sections 8.1, 8.3, and 7.

## Process deviation inherited by this work

PR [#9](https://github.com/Ayumu-J-S/llm_scratch/pull/9) introduced the broader
agent review workflow and was merged as commit
`a05eb1de5656643757a1c3d98047c98dedea8bfa` before the required independent
heavy review completed. That merge-before-review sequence deviated from the
workflow now recorded in `AGENTS.md`. This record preserves the deviation; it
does not retroactively claim a passing verdict or merge authority.

## Execution timeline

Runtime model identifiers and reasoning modes were not exposed. They are
reported exactly as unavailable rather than inferred.

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | handoff (planning) | not exposed by runtime | not exposed by runtime | `origin/main` / EXP-001 ticket | Plan with requested `gpt-5.6-sol` / `ultra`, grounded in philosophy | completed | Runtime did not expose the requested model/mode; plan identified the separate experiment-record and fixture gaps | Planner handoff retained in the parent task |
| 1 | review | not exposed by runtime | not exposed by runtime | PR #9 / review workflow | Independent heavy review attempt 1 | blocked | Invocation returned no verdict and was interrupted after repeated bounded waits | No verdict produced; blocker retained here |
| 1 | review | not exposed by runtime | not exposed by runtime | PR #9 / review workflow | Independent heavy review attempt 2 | blocked | Replacement invocation also returned no verdict and was interrupted | No verdict produced; blocker retained here |
| 2 | implementation | not exposed by runtime | not exposed by runtime | `a05eb1d` plus planner handoff | First delegated implementation repair | blocked | Completed read-only inspection but made no edits before interruption | Clean worktree observed after the attempt |
| 3 | implementation | not exposed by runtime | not exposed by runtime | `a05eb1d` plus exact patch request | Replacement delegated implementation | implemented; interrupted before handoff | Added experiment guidance/template/fixture, PR fields, cross-links, and ledger entry | Current branch diff |
| 3 | repair | not exposed by runtime | not exposed by runtime | Replacement delegate's working-tree diff | Reconcile interrupted edits and validate exact evidence | completed locally; review pending | Matched record basename to branch slug, replaced illustrative config with actual Hydra composition output, and passed R0 field coverage | `uv run python src/train.py --cfg job ...`; `git diff --check`; `uv lock --check`; field scan |
| 3 | re-review | pending | pending | pending implementation commit | Independent heavy review against philosophy, ticket, and selected checks | pending | No review has run and no verdict is claimed | pending |

Allowed outcome interpretation: the two blocked attempts are not reviews
performed and are not passing reviews. The pending row is a handoff marker, not
a model invocation.

## Check selection and verdicts

### Review cycle 1

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: none; both attempts blocked before review
- Selected `CHECK.md` sections: 8.1, 8.3, and 7
- Major sections marked N/A: performance, DGX execution, data supply, numerical
  stability, and checkpoint runtime behavior; EXP-001 is an R0 documentation
  contract and changes no ML runtime
- Ticket acceptance result: pending
- Philosophy alignment: pending independent assessment
- Complexity / change-surface result: pending
- ML-system result: `N/A` pending reviewer confirmation
- Verdict: blocked; no `PASS`, `PASS WITH NOTE`, or `FAIL` was issued

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| blocker | review availability | Two delegated review invocations returned no verdict before interruption | Execution timeline | Run a fresh independent review against the final commit |
| process | merge order | Related PR #9 merged before required review completed | PR #9 and merge commit `a05eb1d` | Preserve deviation and do not repeat it for this ticket |

## Failed-review handoff

`N/A — no review returned FAIL. Two review invocations were blocked before a
verdict; their context and evidence are retained above.`

## Repair result

- Repair cycle: current EXP-001 documentation implementation
- Repair model / mode: not exposed by runtime / not exposed by runtime
- Input handoff: ticket acceptance criteria, philosophy, applicable checks, and
  the requested missing fields
- Changes made: added the experiment contract, filled R0 fixture including a
  retained negative attempt, tightened PR/model-run links, and updated ledger
- Deliberately not changed: `ROADMAP.md`, runtime code, Hydra config, or run data
- Local evidence: actual Hydra composition output (exit 0), `git diff --check`,
  `uv lock --check`, and explicit field-coverage scan all passed
- Commit reviewed next: `fd2f09886c83332670f3de5da924e0e818efddb5`
- Re-review model / mode: pending / pending
- Re-review verdict: pending; no passing verdict claimed

## Final evidence

- Resolved Hydra command/config: `N/A — documentation-only ticket`; the fixture
  preserves its deliberately invalid attempted config
- Data/tokenizer/model identity: `N/A — no scientific run`
- Validation and measurements: Hydra composition exit 0; `git diff --check`,
  `uv lock --check`, and the explicit R0 field-coverage scan passed; independent
  review pending
- Performance/resource result: `N/A — R0 documentation change`
- Failed attempts retained at: execution timeline and fixture Attempt 1
- Known trade-offs: prose completeness adds review surface, but prevents missing
  negative-run identity
- Unresolved risks: the contract has not yet been exercised by a real
  consequential run
- Human decision requested: review and merge only after independent review
  returns an acceptable verdict

## Model assessment from this ticket

No model-quality conclusion can be drawn because exact model identity/mode was
hidden and the independent review attempts were blocked.

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | planning | Produced a philosophy-grounded scoped plan | Requested Sol/Ultra identity could not be verified | Exact ticket, philosophy, and R0 checks | completed |
| not exposed by runtime / not exposed by runtime | implementation and repair | Produced a scoped documentation repair | First delegate stalled; replacement was interrupted before reporting validation | Exact planner handoff and patch request | implemented locally; validation pending |
| not exposed by runtime / not exposed by runtime | review attempts | N/A | Returned no verdict before interruption | Philosophy, ticket, and selected checks | blocked twice |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated applicable attempt counts without counting blocked reviews as
  performed or successful.
- [x] Confirmed the draft PR execution trail matches this record; final review
  fields remain pending.
