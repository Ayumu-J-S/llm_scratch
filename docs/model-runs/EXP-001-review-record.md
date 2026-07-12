# EXP-001 — Experiment and PR Handoff Contract

- Current PR: [#10](https://github.com/Ayumu-J-S/llm_scratch/pull/10)
- Related process PR: [#9](https://github.com/Ayumu-J-S/llm_scratch/pull/9)
- Branch: `codex/exp-001-review-record`
- Ticket: `EXP-001`
- Hypothesis: one compact, append-only experiment record can make positive and
  negative run handoffs reconstructible by a fresh agent
- Experiment record: `docs/experiments/EXP-001-review-record.md`
- Started: 2026-07-11
- Final verdict: `PASS WITH NOTE`
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
| 4 | re-review | not exposed by runtime | not exposed by runtime | repair commit `df1acf62a05266cfd8f80bd86c96d932d1d6c67e` and metadata head `46bd85837b211eeeb9632980ecfb12a09d1372ce` | Independent heavy re-review of the exact failed findings and full R0 acceptance | FAIL | Original resolved-config and SHA defects passed, but the live draft PR body still contained the pre-repair command, SHA, and execution trail | Reviewer findings received 2026-07-11; PR body inspection |
| 5 | repair | not exposed by runtime | not exposed by runtime | Failed live-handoff finding against PR #10 and `46bd85837b211eeeb9632980ecfb12a09d1372ce` | Update committed provenance and replace the live PR body after the final metadata commit | completed; re-review pending | Recorded the second FAIL and replaced the live body after the final metadata push with both FAIL/repair cycles and exact final head | This record and PR #10 body |
| 5 | re-review | not exposed by runtime | not exposed by runtime | `be26321bc57eebad88c63e2cae3b7641b5c0e533` plus updated PR #10 body | Independent heavy re-review of local records and live handoff parity | PASS WITH NOTE | All EXP-001 acceptance criteria and selected CHECK sections passed; real consequential-run usability remains unexercised | Reviewer validation: exact Hydra stdout equality, no interpolation, field scan 37/37, local/remote/PR head parity, lock and diff checks |
| 6 | handoff | not exposed by runtime | not exposed by runtime | Passing review at `be26321bc57eebad88c63e2cae3b7641b5c0e533` and the synchronized live PR #10 body | Commit the final verdict and publish the matching live handoff | completed at `176326c07ccb0ce69ae43adc8be808c662895e30` | Recorded the final `PASS WITH NOTE`, retained both earlier `FAIL` verdicts, and synchronized the live PR execution trail | Commit `176326c07ccb0ce69ae43adc8be808c662895e30`; PR #10 body |

Allowed outcome interpretation: the two blocked attempts are not reviews
performed and are not passing reviews. Historical `pending` text records the
state of a repair handoff at that point in time; it is not the final ticket
state. Two later performed reviews returned `FAIL` and drove repairs, and the
cycle-5 independent re-review then returned the final `PASS WITH NOTE` without
erasing those earlier states.

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

### Review cycle 3 — returned FAIL

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: `46bd85837b211eeeb9632980ecfb12a09d1372ce`
- Repair commit verified: `df1acf62a05266cfd8f80bd86c96d932d1d6c67e`
- Selected `CHECK.md` sections: 1, 7, 8.1, 8.3, and 11 EXP-001
- Ticket acceptance result: local contract criteria passed; live branch-to-PR
  handoff failed
- Verdict: `FAIL`; PR #10's body remained stale even though comments and local
  records contained the repair

#### Findings

| Severity | Area | What was wrong | Evidence | Required action |
| --- | --- | --- | --- | --- |
| major | live PR handoff | PR #10 body still named the command without `--resolve`, reviewed SHA `fd2f098a`, a cycle-3 pending review, and omitted the returned FAIL/repair | Live draft PR body inspected against `46bd85837b211eeeb9632980ecfb12a09d1372ce` | Replace the body after the final metadata commit with the exact resolved command, both FAIL cycles, repair SHA, final head, validation, and pending re-review state |

### Review cycle 4 — PASS WITH NOTE

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: `be26321bc57eebad88c63e2cae3b7641b5c0e533`
- Selected `CHECK.md` sections: 1, 7, 8.1, 8.3, and 11 EXP-001
- Major sections marked N/A: 3–6, 8.2, and 9–10 because no comparison,
  data path, GPU/model/numerical behavior, checkpoint/W&B behavior, or long-run
  operation changed or ran
- Ticket acceptance result: PASS; all roadmap acceptance fields and the live
  fixture handoff were demonstrated
- Philosophy alignment: PASS; hypothesis, budget, identities, negative result,
  integrity, conclusion, uncertainty, and next step remain human-legible
- Complexity / change-surface result: PASS; documentation/templates only, with
  separate experiment and model-provenance responsibilities
- ML-system result: N/A; no ML runtime changed or ran
- Verdict: `PASS WITH NOTE`

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| note | operational exercise | Static fixture and live PR handoff satisfy EXP-001, but the contract has not yet been used by a consequential training run | Fixture limitation, final reviewer validation | Exercise the contract during the next in-scope consequential run; this does not block EXP-001 |

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

### Failed-review handoff — cycle 3

- Failed verdict: `FAIL` against the live handoff at
  `46bd85837b211eeeb9632980ecfb12a09d1372ce`.
- Exact repair scope: update this record and ledger, push the metadata commit,
  then replace PR #10's body so its command, commits, execution timeline,
  validation, and verdict state match the repository records.
- Deliberate model selection: this is a bounded metadata synchronization repair;
  the current agent is used to avoid handing credentials/state to another
  execution path. Exact model identity and mode are not exposed by runtime.
- Constraints: preserve both FAIL findings, keep the PR draft, claim no passing
  verdict, and do not alter runtime code or `ROADMAP.md`.
- Completion evidence: live PR body and final branch head agree, followed by an
  independent re-review.

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
- Historical handoff state when the repair was committed: re-review pending; no
  passing verdict was claimed at that point
- Re-review model / mode: not exposed by runtime / not exposed by runtime
- Re-review verdict: `FAIL` at metadata head
  `46bd85837b211eeeb9632980ecfb12a09d1372ce`; the resolved-config repair passed,
  but the live PR body was stale

## Repair result — cycle 5 live handoff

- Repair model / mode: not exposed by runtime / not exposed by runtime
- Input handoff: review-cycle-3 live-PR mismatch finding
- Changes made: recorded the failed review and replaced PR #10's body after the
  final metadata commit was pushed
- Deliberately not changed: experiment contract, runtime code, Hydra config,
  `ROADMAP.md`, or the already-validated resolved config
- Local evidence: final branch head plus matching live PR #10 body
- Re-review target: final branch head plus live PR #10 body
- Re-review verdict: `PASS WITH NOTE` at
  `be26321bc57eebad88c63e2cae3b7641b5c0e533`

## Final evidence

- Resolved Hydra command/config: `N/A — documentation-only ticket`; the fixture
  preserves its deliberately invalid attempted config
- Data/tokenizer/model identity: `N/A — no scientific run`
- Validation and measurements: resolved Hydra composition exit 0; captured
  stdout byte-identical to fresh output; no unresolved interpolation;
  `T_max == epochs == 1`; field scan 37/37; `uv lock --check` and
  `git diff --check` passed; local/remote/PR heads matched
- Performance/resource result: `N/A — R0 documentation change`
- Failed attempts retained at: execution timeline and fixture Attempt 1
- Known trade-offs: prose completeness adds review surface, but prevents missing
  negative-run identity
- Unresolved risks: the contract has not yet been exercised by a real
  consequential run; accepted as the explicit review note for this R0 ticket
- Human decision requested: review the evidence and decide whether to merge;
  model review is not merge authority

## Model assessment from this ticket

No quality conclusion can be attributed to a named model or reasoning mode
because the runtime hid both. The observable process outcomes remain useful:
two invocations blocked before review, two performed reviews returned `FAIL`
with actionable findings, and a later independent review returned `PASS WITH
NOTE` after both repairs.

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | planning | Produced a philosophy-grounded scoped plan | Requested Sol/Ultra identity could not be verified | Exact ticket, philosophy, and R0 checks | completed |
| not exposed by runtime / not exposed by runtime | implementation and repair | Produced scoped documentation repairs and exact resolved-config evidence | First delegate stalled; initial fixture captured unresolved interpolation; first repair left the live PR body stale | Exact planner handoff, failed findings, and narrow repair request | repairs completed; first re-review failed on live-PR parity; later re-review passed with note |
| not exposed by runtime / not exposed by runtime | review attempts | Detected unresolved evidence, stale review trace, and stale live PR body; verified final repairs | Two earlier invocations returned no verdict before interruption | Philosophy, ticket, selected checks, exact failed-review handoffs, and live PR state | blocked twice; two FAIL verdicts; final PASS WITH NOTE |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated applicable attempt counts without counting blocked reviews as
  performed or successful.
- [x] Reconfirmed this record and the live PR body agree on the cycle-5
  independent `PASS WITH NOTE` and cycle-6 metadata handoff.
