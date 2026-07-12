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
| [GATE-001 post-merge finalization](GATE-001-postmerge-finalize.md) | GATE-001 | draft | not exposed by runtime / not exposed by runtime | pending | 0 | in progress | post-merge-status, dependency-reconciliation, audit-parity |
| [GATE-001 bilingual overfit proof](GATE-001-bilingual-overfit-proof.md) | GATE-001 | [#39](https://github.com/Ayumu-J-S/llm_scratch/pull/39) (merged `2e2c4f4`) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime — PASS WITH NOTE `4680390444` | 4 | PASS WITH NOTE — final review `4680802947`; guarded audit and squash merge complete | memorization, resume, reproducibility, generation, fixture-sizing, sampling-audit, formatting, determinism-scope, branch-protection-inventory |
| [DATA-001 causal boundaries](DATA-001-causal-boundaries.md) | DATA-001 | [#11](https://github.com/Ayumu-J-S/llm_scratch/pull/11) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 1 | PASS WITH NOTE | ml-semantics, stale-accounting, boundary-policy |
| [DATA-002 immutable manifests](DATA-002-immutable-manifests.md) | DATA-002 | [#13](https://github.com/Ayumu-J-S/llm_scratch/pull/13) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 3 | PASS WITH NOTE | benchmark-authority, split-leakage, tokenizer-compatibility, workflow-integrity, merge-order |
| [POLICY-001 guarded agent self-merge](POLICY-001-agent-self-merge.md) | POLICY-001 | [#16](https://github.com/Ayumu-J-S/llm_scratch/pull/16) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 1 | PASS WITH NOTE | authorization, review-state, checks, drift |
| [EXP-001 review record](EXP-001-review-record.md) | EXP-001 | [#10](https://github.com/Ayumu-J-S/llm_scratch/pull/10); related process PR [#9](https://github.com/Ayumu-J-S/llm_scratch/pull/9) | not exposed by runtime / not exposed by runtime | blocked twice, then FAIL; model/mode not exposed | 5 | PASS WITH NOTE | reproducibility, stale-review-target, stale-pr-body, stale-final-status, stale-integration-verdict, provenance-target, review-unavailable, merge-order |
| [MODEL-001 baseline invariants](MODEL-001-baseline-invariants.md) | MODEL-001 | [#14](https://github.com/Ayumu-J-S/llm_scratch/pull/14); audit [#19](https://github.com/Ayumu-J-S/llm_scratch/pull/19) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 2 | PASS WITH NOTE | accepted-plan-coverage, canonical-integration, merge-order, runtime-identity |
| [TOK-001 canonical tokenizer](TOK-001-canonical-tokenizer.md) | TOK-001 | [#12](https://github.com/Ayumu-J-S/llm_scratch/pull/12) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 5 | PASS WITH NOTE | reproducibility, ml-semantics, tokenizer-cost, offline-identity, merge-order |
| [PROV-001 model provenance](PROV-001-model-provenance.md) | PROV-001 | [#17](https://github.com/Ayumu-J-S/llm_scratch/pull/17) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 1 | PASS WITH NOTE | runtime-identity, source-separation, redaction, final-head-parity |
| [ENV-001 DGX Spark CUDA](ENV-001-dgx-spark-cuda.md) | ENV-001 | [#15](https://github.com/Ayumu-J-S/llm_scratch/pull/15) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 2 | PASS WITH NOTE | cuda-runtime, reproducibility, dgx-spark, evidence, provenance-separation |
| [ROADMAP status refresh](ROADMAP-status-refresh.md) | ROADMAP-MAINT | unavailable — direct `main` push explicitly requested | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 0 | PASS — final docs-only no-drift confirmation pending | stale-roadmap-state, dependency-reconciliation, direct-push |
| [CI-001 network-free quality gate](CI-001-network-free-quality-gate.md) | CI-001 | [#37](https://github.com/Ayumu-J-S/llm_scratch/pull/37) (draft) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime — PASS WITH NOTE `4680271011` | 0 | PASS WITH NOTE — docs-only no-drift confirmation and guarded audit pending | ci-parity, offline-smoke, lock-drift, workflow-separation |
| [GEN-001 base-model continuation](GEN-001-base-continuation.md) | GEN-001 | [#38](https://github.com/Ayumu-J-S/llm_scratch/pull/38) (draft) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime — PASS WITH NOTE `4680305424` | 0 | PASS WITH NOTE — docs-only no-drift confirmation and guarded audit pending | checkpoint-reconstruction, deterministic-sampling, generation-boundaries, model-identity-parity |
| [STAB-001 stable training](STAB-001-stable-training.md) | STAB-001 | [#34](https://github.com/Ayumu-J-S/llm_scratch/pull/34) (draft) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 2 | PASS WITH NOTE — review `4680137991`; docs-only no-drift confirmation pending | numerical-stability, accumulation, bf16, scheduler, clipping-calibration, iterator-shutdown |
| [CKPT-001 atomic full-state resume](CKPT-001-atomic-resume.md) | CKPT-001 | [#36](https://github.com/Ayumu-J-S/llm_scratch/pull/36) (draft) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime — FAIL `4680194134`, `4680225857`; PASS WITH NOTE `4680237382` | 6 | PASS WITH NOTE `4680237382`; docs-only no-drift confirmation pending | resume, atomic-write, rotation, compatibility, stream-cursor, full-suite-regression, milestone-duplication, prefix-replay-guard, terminal-cursor, preview-lifecycle, run-identity |
| [DATA-003 packed resume repair](DATA-003-packed-resume-repair.md) | DATA-003 | [#31](https://github.com/Ayumu-J-S/llm_scratch/pull/31) (merged `cf827016`) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 2 | PASS WITH NOTE — implementation `4680026587`, no-drift `4680031289` / `4680036491`, guarded audit merged | resume, packed-cursor, token-accounting, completed-cursor |
| [DATA-003 P5 post-merge audit](DATA-003-p5-post-merge-audit.md) | DATA-003 | [#32](https://github.com/Ayumu-J-S/llm_scratch/pull/32) (merged `29c6b925`) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 3 | PASS WITH NOTE — final confirmations, guarded audit, and squash merge complete | post-merge-audit, merge-evidence, stale-roadmap-state, validation-command, audit-completeness |
| [DATA-003 P5 final post-merge audit](DATA-003-p5-final-postmerge-audit.md) | DATA-003 | [#33](https://github.com/Ayumu-J-S/llm_scratch/pull/33) (merged `f73626ce`) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 0 | PASS WITH NOTE — final reviews, guarded audit, and squash merge complete | post-merge-ledger, final-audit, merge-evidence |
| [DATA-003 P5 P33 post-merge ledger audit](DATA-003-p5-p33-postmerge-ledger-audit.md) | DATA-003 | [#35](https://github.com/Ayumu-J-S/llm_scratch/pull/35) (draft) | not exposed by runtime / not exposed by runtime | pending | 0 | in progress | post-merge-ledger, final-audit, merge-evidence |
| [CFG-001 canonical Hydra](CFG-001-canonical-hydra.md) | CFG-001 | [#20](https://github.com/Ayumu-J-S/llm_scratch/pull/20); audit [#21](https://github.com/Ayumu-J-S/llm_scratch/pull/21) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 1 | PASS WITH NOTE | config-preflight, profile-composition, command-parity, evaluation-fallback, merge-audit |
| [REP-001 reproducibility](REP-001-reproducibility.md) | REP-001 | [#22](https://github.com/Ayumu-J-S/llm_scratch/pull/22) (merged); repair [#24](https://github.com/Ayumu-J-S/llm_scratch/pull/24) (merged); audit [#23](https://github.com/Ayumu-J-S/llm_scratch/pull/23) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 2 | PASS WITH NOTE | reproducibility, manifest-integrity, global-seeding, guard-ordering, dirty-verification, merge-audit |
| [LOOP-001 step/token budgets](LOOP-001-step-token-budgets.md) | LOOP-001 | [#25](https://github.com/Ayumu-J-S/llm_scratch/pull/25) (merged); follow-up [#26](https://github.com/Ayumu-J-S/llm_scratch/pull/26) (merged); follow-up [#28](https://github.com/Ayumu-J-S/llm_scratch/pull/28) (merged); audit [#27](https://github.com/Ayumu-J-S/llm_scratch/pull/27) (draft) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 4 | PASS WITH NOTE | training-loop, token-weighting, cadence, numerical-stability, validation-evidence, metrics-lifecycle, p2-repair, merge-audit, runtime-identity |
| [LOOP-001 metrics audit](LOOP-001-metrics-audit.md) | LOOP-001 | [#26](https://github.com/Ayumu-J-S/llm_scratch/pull/26) (merged); [#28](https://github.com/Ayumu-J-S/llm_scratch/pull/28) (merged); finalization audit [#27](https://github.com/Ayumu-J-S/llm_scratch/pull/27) (draft) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 1 | PASS WITH NOTE | metrics-lifecycle, offline-evidence, p2-repair, merge-gate, runtime-identity |
| [DATA-003 deterministic stream cursor](DATA-003-stream-cursor.md) | DATA-003 | [#29](https://github.com/Ayumu-J-S/llm_scratch/pull/29) (merged `57266e1`; reopened by P2); P5 repair [#31](https://github.com/Ayumu-J-S/llm_scratch/pull/31) (merged `cf827016`) | not exposed by runtime / not exposed by runtime | not exposed by runtime / not exposed by runtime | 5 | PASS WITH NOTE — post-merge P2 repaired, audited, and merged | deterministic-order, cursor-resume, packed-cursor-resume, completed-cursor, post-merge-reopen, repeat-policy, prefetch-equivalence, async-ack, pass-completion, process-cursor-propagation, process-reuse |

Use short, stable values in `Main failure tags` so results can be aggregated,
for example `data-starvation`, `cuda-fallback`, `ml-semantics`,
`numerical-stability`, `resume`, `over-abstraction`, `scope-creep`,
`reproducibility`, and `logging-overhead`.

## Per-model aggregate

Update the applicable row when the PR finishes. Counts are sufficient until
there are enough observations for meaningful rates.

| Exact model / mode | Implementation attempts | First-review passes | Repair attempts | Successful repairs | Reviews performed | Important strengths observed | Recurring failure modes | Last updated |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | 21 | 8 | 43 | 25 | 59 | Produced scoped EXP-001, DATA-001, DATA-002, MODEL-001, ENV-001, CFG-001, REP-001, LOOP-001, STAB-001, CKPT-001, CI-001, GEN-001, GATE-001, guarded-policy, canonical-tokenizer, provenance, roadmap-state maintenance, and documentation-audit changes; localized causal-boundary, immutable-manifest, tokenizer, model-invariant, CUDA-boundary, profile/preflight, source-separation, run-identity, global-seeding, training-loop, token-weighting, BF16 accumulation/clipping/schedule, bounded-stream shutdown, full-state recovery, rotation, offline-CI parity, checkpoint-owned deterministic generation, fixed-fixture end-to-end evidence, and dependency-state semantics | One implementation stalled; two review attempts blocked; previous review findings remain as recorded above. CKPT-001 initially conflated completed stream iteration with explicit resume, causing one full-suite failure before a cursor-pending repair; a CPU smoke then found duplicate epoch-end milestone saves before a persisted event-state repair; an exact-resume audit rejected cursorless map-style loaders rather than silently replaying a prefix. Formal review `4680194134` found terminal completed-cursor resume returned no next pass; formal re-review `4680225857` found recorded run Git/lock/experiment identity omitted from resume compatibility. The run-identity repair then passed independent re-review `4680237382` with a bounded-evidence note. CI-001 review retained a documented process-scope limitation for its socket guard. GEN-001 required a pre-review same-shape model-identity parity guard before passing independent review `4680305424`. GATE-001 initially needed fixture sizing, a non-terminal recovery point, unambiguous predeclared samples, and one formatter-only repair; its review retained an environment-scoped determinism note. ROADMAP-MAINT passed its first independent R0 review with no findings. Exact model attribution remains impossible | 2026-07-12 |

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
