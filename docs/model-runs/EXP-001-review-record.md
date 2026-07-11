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
| 3 | re-review | not exposed by runtime | not exposed by runtime | `f0faf8466189cb8dd8fecdfca431e3de2bcbcee5` | Independent heavy review against philosophy, ticket, and selected checks | FAIL | The fixture called its Hydra YAML fully resolved while retaining `${training.epochs}`; the model-run record also named a stale re-review target SHA | Reviewer findings received 2026-07-11; exact failed-review handoff below |
| 4 | repair | not exposed by runtime | not exposed by runtime | Failed-review findings against `f0faf8466189cb8dd8fecdfca431e3de2bcbcee5` | Narrow Luna/Extra High repair requested: recapture resolved Hydra evidence, validate interpolation absence, and correct the re-review target | completed in `df1acf62a05266cfd8f80bd86c96d932d1d6c67e`; re-review pending | Ran the exact `--resolve` command, captured `T_max: 1`, added semantic no-interpolation validation, and replaced the stale target | Command exit 0; resolved-block validation exit 0; `git diff --check` |
| 4 | re-review | pending | pending | repair commit `df1acf62a05266cfd8f80bd86c96d932d1d6c67e` plus the current metadata-only branch head | Independent heavy re-review of the exact failed findings | pending | No passing re-review has run and no verdict is claimed | pending |

Allowed outcome interpretation: the two blocked attempts are not reviews
performed and are not passing reviews. The pending row is a handoff marker, not
a model invocation.

## Check selection and verdicts

### Review cycle 1 — blocked attempts

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

### Review cycle 2 — returned FAIL

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: `f0faf8466189cb8dd8fecdfca431e3de2bcbcee5`
- Selected `CHECK.md` sections: applicable documentation and reproducibility
  checks from 8.1, 8.3, and 7
- Verdict: `FAIL`; no ticket completion or passing verdict is claimed

#### Findings

| Severity | Area | What was wrong | Evidence | Required action |
| --- | --- | --- | --- | --- |
| major | reproducibility | The experiment fixture described its Hydra YAML as fully resolved, but `T_max` remained `${training.epochs}` | `docs/experiments/EXP-001-review-record.md` Attempt 1 | Run the exact command with `--resolve`, replace the block with its exact output including `T_max: 1`, and validate that the captured resolved block contains no interpolation |
| major | review trace | The recorded next-review target named a stale commit SHA | Previous Repair result entry | Point the next review at the future repaired commit, leaving it pending until that commit exists |

## Failed-review handoff

- Failed verdict: `FAIL` against
  `f0faf8466189cb8dd8fecdfca431e3de2bcbcee5`.
- Exact repair scope: run
  `uv run python src/train.py --cfg job --resolve data.mode=streaming training.epochs=1 wandb.enabled=false`;
  replace the fixture block with the exact resolved output; update its evidence,
  command, and timestamps; semantically verify the block has no `${...}`;
  preserve both review findings here; and correct ledger counts and the pending
  re-review target.
- Deliberate model selection: Luna at Extra High was requested for this narrow
  documentation/config-evidence repair. The collaboration runtime cannot expose
  or select model identity or reasoning mode, so both are recorded as `not
  exposed by runtime`; no substitute identity is inferred.
- Constraints carried forward: do not edit `ROADMAP.md` or runtime code; do not
  claim `PASS`; do not commit or push; retain the earlier blocked attempts.
- Required re-review: independently verify the repaired resolved block and
  review trace on the future repaired commit before ticket completion.

## Repair result

- Repair cycle: 4, narrow failed-review repair
- Repair model / mode: not exposed by runtime / not exposed by runtime
- Input handoff: the cycle-2 `FAIL` findings and complete failed-review handoff
  above
- Changes made: recaptured exact resolved Hydra output with `--resolve`, changed
  `T_max` to `1`, added semantic interpolation validation, preserved the FAIL,
  and corrected the next-review target and ledger counts
- Deliberately not changed: `ROADMAP.md`, runtime code, Hydra config, or run data
- Local evidence: exact resolved Hydra command exited 0; a `uv run python`
  validation extracted and parsed the captured YAML, recursively asserted that
  no string contains `${`, and asserted
  `training.scheduler.T_max == training.epochs == 1`; it printed `PASS: parsed
  captured YAML; no unresolved interpolation; T_max == epochs == 1` and exited
  0; `git diff --check` passed
- Commit reviewed next: repair commit
  `df1acf62a05266cfd8f80bd86c96d932d1d6c67e` plus the current metadata-only branch head
- Re-review model / mode: pending / pending
- Re-review verdict: pending; no passing verdict claimed

## Final evidence

- Resolved Hydra command/config: `N/A — documentation-only ticket`; the fixture
  preserves its deliberately invalid attempted config
- Data/tokenizer/model identity: `N/A — no scientific run`
- Validation and measurements: resolved Hydra composition exit 0; captured
  resolved-block no-interpolation scan exit 0; `git diff --check` passed;
  independent re-review pending
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
| not exposed by runtime / not exposed by runtime | implementation and repair | Produced scoped documentation repairs and exact resolved-config evidence | First delegate stalled; initial fixture captured unresolved interpolation | Exact planner handoff, failed findings, and narrow repair request | repair completed locally; re-review pending |
| not exposed by runtime / not exposed by runtime | review attempts | Detected unresolved evidence and a stale review target | Two earlier invocations returned no verdict before interruption | Philosophy, ticket, and selected checks | blocked twice; later review returned FAIL |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated applicable attempt counts without counting blocked reviews as
  performed or successful.
- [x] Confirmed the draft PR execution trail matches this record; final review
  fields remain pending.
