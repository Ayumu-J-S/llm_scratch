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
| [DATA-001 causal boundaries](DATA-001-causal-boundaries.md) | DATA-001 | [#11](https://github.com/Ayumu-J-S/llm_scratch/pull/11) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 1 | PASS WITH NOTE | ml-semantics, stale-accounting, boundary-policy |
| [DATA-002 immutable manifests](DATA-002-immutable-manifests.md) | DATA-002 | [#13](https://github.com/Ayumu-J-S/llm_scratch/pull/13) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 3 | PASS WITH NOTE | benchmark-authority, split-leakage, tokenizer-compatibility, workflow-integrity, merge-order |
| [POLICY-001 guarded agent self-merge](POLICY-001-agent-self-merge.md) | POLICY-001 | [#16](https://github.com/Ayumu-J-S/llm_scratch/pull/16) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 1 | PASS WITH NOTE | authorization, review-state, checks, drift |
| [EXP-001 review record](EXP-001-review-record.md) | EXP-001 | [#10](https://github.com/Ayumu-J-S/llm_scratch/pull/10); related process PR [#9](https://github.com/Ayumu-J-S/llm_scratch/pull/9) | not exposed by runtime / not exposed by runtime | blocked twice, then FAIL; model/mode not exposed | 5 | PASS WITH NOTE | reproducibility, stale-review-target, stale-pr-body, stale-final-status, stale-integration-verdict, provenance-target, review-unavailable, merge-order |
| [MODEL-001 baseline invariants](MODEL-001-baseline-invariants.md) | MODEL-001 | [#14](https://github.com/Ayumu-J-S/llm_scratch/pull/14); audit [#19](https://github.com/Ayumu-J-S/llm_scratch/pull/19) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 2 | PASS WITH NOTE | accepted-plan-coverage, canonical-integration, merge-order, runtime-identity |
| [TOK-001 canonical tokenizer](TOK-001-canonical-tokenizer.md) | TOK-001 | [#12](https://github.com/Ayumu-J-S/llm_scratch/pull/12) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 5 | PASS WITH NOTE | reproducibility, ml-semantics, tokenizer-cost, offline-identity, merge-order |
| [PROV-001 model provenance](PROV-001-model-provenance.md) | PROV-001 | [#17](https://github.com/Ayumu-J-S/llm_scratch/pull/17) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 1 | PASS WITH NOTE | runtime-identity, source-separation, redaction, final-head-parity |
| [ENV-001 DGX Spark CUDA](ENV-001-dgx-spark-cuda.md) | ENV-001 | [#15](https://github.com/Ayumu-J-S/llm_scratch/pull/15) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 2 | PASS WITH NOTE | cuda-runtime, reproducibility, dgx-spark, evidence, provenance-separation |
| [CFG-001 canonical Hydra](CFG-001-canonical-hydra.md) | CFG-001 | [#20](https://github.com/Ayumu-J-S/llm_scratch/pull/20); audit [#21](https://github.com/Ayumu-J-S/llm_scratch/pull/21) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 1 | PASS WITH NOTE | config-preflight, profile-composition, command-parity, evaluation-fallback, merge-audit |
| [REP-001 reproducibility](REP-001-reproducibility.md) | REP-001 | [#22](https://github.com/Ayumu-J-S/llm_scratch/pull/22) (merged); repair [#24](https://github.com/Ayumu-J-S/llm_scratch/pull/24) (merged); audit [#23](https://github.com/Ayumu-J-S/llm_scratch/pull/23) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 2 | PASS WITH NOTE | reproducibility, manifest-integrity, global-seeding, guard-ordering, dirty-verification, merge-audit |
| [LOOP-001 step/token budgets](LOOP-001-step-token-budgets.md) | LOOP-001 | [#25](https://github.com/Ayumu-J-S/llm_scratch/pull/25) (merged); follow-up [#26](https://github.com/Ayumu-J-S/llm_scratch/pull/26) draft | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 2 | PASS WITH NOTE | training-loop, token-weighting, cadence, numerical-stability, validation-evidence, runtime-identity |
| [LOOP-001 metrics audit](LOOP-001-metrics-audit.md) | LOOP-001 | [#26](https://github.com/Ayumu-J-S/llm_scratch/pull/26) (draft) | not exposed by runtime / not exposed by runtime | pending | 0 | in progress | metrics-lifecycle, offline-evidence, p2-repair, runtime-identity |

Use short, stable values in `Main failure tags` so results can be aggregated,
for example `data-starvation`, `cuda-fallback`, `ml-semantics`,
`numerical-stability`, `resume`, `over-abstraction`, `scope-creep`,
`reproducibility`, and `logging-overhead`.

## Per-model aggregate

Update the applicable row when the PR finishes. Counts are sufficient until
there are enough observations for meaningful rates.

| Exact model / mode | Implementation attempts | First-review passes | Repair attempts | Successful repairs | Reviews performed | Important strengths observed | Recurring failure modes | Last updated |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | 12 | 2 | 23 | 16 | 34 | Produced scoped EXP-001, DATA-001, DATA-002, MODEL-001, ENV-001, CFG-001, REP-001, LOOP-001, guarded-policy, canonical-tokenizer, and provenance changes; localized causal-boundary, immutable-manifest, tokenizer, model-invariant, CUDA-boundary, profile/preflight, source-separation, run-identity, global-seeding, training-loop, token-weighting, and final-head parity semantics; reviews found authority, lifecycle, identity, evidence, guard-ordering, numerical-health, validation-evidence, and workflow defects and independently reproduced CPU fixtures | One implementation stalled; two review attempts blocked; LOOP-001 initial review found missing non-finite-state guards, token cadences, aggregate metrics, and strict budget safety, then a follow-up found validation evidence context missing before repair; REP-001 follow-up review found missing dirty-worktree verification after the initial guard-ordering repair; CFG-001 first review found and repair closed an evaluation execution-path contradiction and profile coupling; earlier passes missed accounting, benchmark authority, lifecycle, tokenizer/workflow integration, exact accepted-plan assertions, provider/evidence labels, and provenance source precedence; exact model attribution remains impossible | 2026-07-12 |

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
