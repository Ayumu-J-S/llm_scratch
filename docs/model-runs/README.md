# Model Run Ledger

This directory tracks the models used for roadmap implementation and their outcomes.

Create one `<ticket>-<short-slug>.md` file per PR from `TEMPLATE.md`. Append all
implementation, review, repair, and re-review phases for that PR to the same
file in chronological order. Never delete or overwrite failed cycles.

## PR and ticket summary

Add one row in every PR. Use `draft` before a PR exists and `unavailable` when a
PR URL cannot be created.

| Record | Ticket | PR | Initial implementation model / mode | First review model / mode | Repair cycles | Final verdict | Main failure tags |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| [EXP-001 review record](EXP-001-review-record.md) | EXP-001 | [#10](https://github.com/Ayumu-J-S/llm_scratch/pull/10); related process PR [#9](https://github.com/Ayumu-J-S/llm_scratch/pull/9) | not exposed by runtime / not exposed by runtime | blocked twice, then FAIL; model/mode not exposed | 2 | PASS WITH NOTE | reproducibility, stale-review-target, stale-pr-body, review-unavailable, merge-order |

Use short, stable values in `Main failure tags` so results can be aggregated,
for example `data-starvation`, `cuda-fallback`, `ml-semantics`,
`numerical-stability`, `resume`, `over-abstraction`, `scope-creep`,
`reproducibility`, and `logging-overhead`.

## Per-model aggregate

Update the applicable row when the PR finishes. Counts are sufficient until
there are enough observations for meaningful rates.

| Exact model / mode | Implementation attempts | First-review passes | Repair attempts | Successful repairs | Reviews performed | Important strengths observed | Recurring failure modes | Last updated |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | 2 | 0 | 3 | 1 | 3 | Produced a scoped EXP-001 plan, documentation repair, exact resolved-config recapture, and live-handoff repair; final review verified local/live parity | One implementation stalled; two review attempts returned no verdict; initial evidence retained an unresolved interpolation; PR body lagged repository records; exact model attribution impossible | 2026-07-11 |

### Counting rules

- `First-review passes` counts an independent first review that returned
  `PASS` or `PASS WITH NOTE` immediately after the initial implementation.
- `Successful repairs` counts a repair followed by a passing independent review.
- Treat different reasoning modes of the same model as separate rows.
- When the runtime hides the exact model identifier, write
  `not exposed by runtime / <mode>` rather than guessing.
- Record blocked runs and tool failures as attempts in the detailed record, but
  do not count them as successes.
- Count `Reviews performed` only when a reviewer actually returns findings and a
  verdict; a blocked pre-review invocation remains only in the detailed record.
- Do not judge model quality from counts alone. Include ticket type, change
  size, and failure tags.

## Questions this ledger should answer

As evidence accumulates, it should show:

- which model/mode tends to pass first review on small implementations
- which models struggle with data, GPU, training-loop, or design work
- which problems heavy review models detect
- whether a repair model succeeded with the failed-review context
- whether a higher reasoning mode justified its additional time or cycles

Use this as model-selection evidence, not as a leaderboard that ignores ticket difficulty.
