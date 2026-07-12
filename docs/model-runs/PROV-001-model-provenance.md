# PROV-001 - Make Codex model provenance visible

- PR: [#17](https://github.com/Ayumu-J-S/llm_scratch/pull/17) (draft)
- Branch: `codex/prov-001-model-provenance`
- Ticket: PROV-001 (repository provenance contract requested after the first roadmap wave)
- Hypothesis: a small, redaction-safe capture command that separates requested/default model settings from explicitly supplied runtime display will make each agent phase auditable without guessing hidden model identity or leaking prompts, tokens, or secrets.
- Experiment record: `N/A` — documentation/tooling provenance change; no ML experiment
- Started: 2026-07-12
- Final verdict: PASS WITH NOTE
- Final record owner: implementation agent; exact runtime identity is not exposed

## Scope and decision context

- Goal: record which Codex model and reasoning mode were requested/defaulted versus which values the runtime explicitly displayed for a phase.
- In scope: a stdlib-only JSON/Markdown capture command, a stable schema, tests, operator documentation, and this model-run/ledger record.
- Out of scope: inferring hidden model identity, capturing prompts/hidden chain-of-thought/tokens/secrets, changing runtime selection, or changing historical records and merge policy.
- Relevant `PHILOSOPHY.md` principles: experiments are first-class artifacts; record observable evidence rather than hidden chain-of-thought; preserve privacy and research integrity; keep the smallest coherent change.
- Baseline commit/run: `8a6f94b` (`origin/main`, post DATA-002 integration); no ML run.
- Intended evidence: CLI JSON and Markdown output, source-precedence and separation tests, redaction tests, graceful missing-git/CLI behavior, and focused/full repository gates.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | planning | not exposed by runtime | not exposed by runtime | `8a6f94b`; requested Sol / Ultra planning | Define a minimal provenance schema and safe capture boundary | completed | Separate requested/default fields from explicit runtime display; never infer exact ID or mode | Planner handoff 2026-07-12 |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `8a6f94b`; requested Luna / Extra High | Implement PROV-001 and start the live draft PR | completed | Added redaction-safe stdlib capture, schema docs, template/workflow guidance, and focused tests; exact active ID/mode remain unavailable | PR #17 head and focused test run |
| 1 | review | not exposed by runtime | not exposed by runtime | `c77c8c939b8c68f7fc9e1da995a16ded1743342f`; requested heavier / Extra Thinking | Independently review ticket, philosophy, and applicable `CHECK.md` sections | PASS WITH NOTE | R0 passed: separation, unavailable identity handling, privacy, docs, and tests are sound; source precedence is documented but caller-enforced and JSON-only output is narrower than the initial record wording | Independent review handoff 2026-07-12 |
| 1 | re-review | not exposed by runtime | not exposed by runtime | `4d6306d555c275e07cbf376bb1cff8c26a74b2f0`; requested heavier / Extra Thinking | Re-review documentation finalization and exact-head provenance parity | PASS WITH NOTE | Confirmed only model-run/ledger finalization changed; record/ledger parity and exact final head were correct after re-review | Independent re-review handoff 2026-07-12 |
| 2 | re-review | not exposed by runtime | not exposed by runtime | `552b74c80643178b346f128dd9ce90679be85f0f`; requested heavier / Extra Thinking | Re-review repair for the stale exact-head field identified by automated P2 review | PASS WITH NOTE | Confirmed the repair changes only the model-run record, aligns the final SHA, and preserves implementation parity; focused tests, full suite, Ruff, and CLI smoke remain passing | Independent repair re-review handoff 2026-07-12 |

Allowed phases: `implementation`, `review`, `repair`, `re-review`, and `handoff`.

## Check selection and verdicts

### Review cycle 1

- Review model / mode: not exposed by runtime / not exposed by runtime (requested Extra Thinking)
- Commit reviewed: `c77c8c939b8c68f7fc9e1da995a16ded1743342f`
- Selected `CHECK.md` sections: R0 documentation/reproducibility, experiment identity, research-integrity/privacy, and change-surface review
- Major sections marked N/A and why: data, tokenizer, model, optimizer, CUDA, performance, checkpoint, and W&B runtime behavior are N/A unless the reviewer finds an integration effect; this ticket adds only stdlib capture tooling and documentation.
- Ticket acceptance result: PASS — requested/default and actual namespaces are distinct; missing exact ID/mode have explicit reasons; privacy-safe context is captured; focused/full tests and lint pass.
- Philosophy alignment: PASS — observable evidence, no hidden-CoT claims, smallest coherent stdlib change, and human-legible handoff.
- Complexity / change-surface result: PASS WITH NOTE — no ML path changed; source precedence is a documented caller contract rather than machine-enforced config discovery.
- ML-system result: N/A — documentation/capture tooling only; no data, tokenizer, model, optimizer, CUDA, DGX, performance, checkpoint, or W&B behavior changed.
- Verdict: PASS WITH NOTE

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| note | source precedence | The implementation trusts explicit `--actual-*` arguments and does not read `~/.codex/config.toml`; requested/default source is intentionally caller-provided rather than inferred | `scripts/capture_model_provenance.py`, independent review | Documented follow-up; do not infer active identity |
| note | output surface | The command emits JSON only; an early record phrase mentioned JSON/Markdown, but ROADMAP acceptance requires a versioned capture schema and does not require Markdown output | CLI/docs review | Keep JSON canonical; render Markdown in model-run records |

## Failed-review handoff

N/A — no failed independent review has occurred.

## Repair result

### Repair cycle 1 — stale final-head provenance

- Repair model / mode: not exposed by runtime / not exposed by runtime
- Input handoff: automated GitHub P2 review on PR #17 identifying the stale `c77c8c9` exact-head field in the committed model-run record
- Changes made: changed the merge-authority field to the exact final head, added explicit re-review rows and final-head model assessment, and retained the original normative `c77c8c9` review entry
- What was deliberately not changed: implementation, tests, CLI schema, privacy behavior, and runtime/training code
- Local evidence: `git diff 4d6306d..552b74c` is record-only; focused tests `4 passed`, full suite `144 passed, 1 skipped`, Ruff passed, and CLI smoke remained valid
- Commit reviewed next: `552b74c80643178b346f128dd9ce90679be85f0f`
- Re-review model / mode: not exposed by runtime / not exposed by runtime
- Re-review verdict: PASS WITH NOTE

## Final evidence

- Resolved Hydra command/config: `N/A` — no Hydra/runtime training path changed.
- Data/tokenizer/model identity: `N/A` — no scientific run.
- Validation and measurements: focused provenance tests `4 passed`; full repository `144 passed, 1 skipped`; Ruff focused check passed; CLI smoke emitted valid JSON with actual `Codex`/`GPT-5` and unavailable exact ID/mode.
- Performance/resource result if applicable: `N/A` — small local stdlib command.
- Failed attempts retained at: execution timeline and future failed-review sections.
- Known trade-offs: requested/default values from `~/.codex/config.toml` are useful context but are not evidence of the active model; actual fields remain unavailable unless explicitly passed by the runtime display.
- Unresolved risks: runtime display injection depends on the caller; the capture command cannot discover hidden runtime state.
- Human decision requested: review the schema and decide whether the capture contract is sufficient for future PRs.

## Merge authority and final audit

- Merge path: `human merge`
- Human authorization: `N/A — human merge remains the default`
- Authorization evidence location: `N/A`
- Authorization covers this named PR or bounded ticket/goal series: N/A
- Exact independently reviewed head SHA: `552b74c80643178b346f128dd9ce90679be85f0f`
- Latest independent verdict / model / mode: PASS WITH NOTE / not exposed by runtime / not exposed by runtime (requested Extra Thinking)
- All actionable findings repaired and independently re-reviewed: yes; two non-blocking notes retained as documented follow-ups
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: pending human review
- Newer human objections since authorization/review: pending
- Human review dismissed by an agent: no
- Unresolved review threads at final audit: pending final GitHub refresh
- Branch-protection required-context inventory: pending
- Applicable configured workflow/check inventory: pending
- Observed exact-head check statuses: pending
- Expected checks absent, pending, skipped, cancelled, or non-successful: pending
- No-check evidence when both inventories are empty: pending
- Target branch and base SHA at final audit: `main` / pending refresh
- Up-to-date, conflict-free, and mergeable evidence: pending
- Record, ledger, PR trail, validation, and risks parity: yes for reviewed head; final PR state refresh pending
- Prohibited self-merge categories: clear for this documentation/tooling change; human merge remains default
- Admin/bypass/force/disabled-check requirement: no
- Final audit PR body/comment location: PR #17 review comment and body (after Ready transition)
- Final audit changed reviewed head: pending
- Immediate pre-merge re-fetch/compare observation location: pending human merge
- Immediate refresh compared authorization, head, base, review decision/objections, threads, expected checks/statuses, and mergeability: pending human merge
- Drift found: pending
- Merge outcome: not merged

## Model assessment from this ticket

Record observable outcomes, not hidden chain-of-thought.

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | planning | Scoped explicit runtime-display provenance and redaction boundaries | Exact planning identity unavailable | Ticket, PHILOSOPHY.md, workflow, and template | completed |
| not exposed by runtime / not exposed by runtime | implementation | Kept requested/default and actual runtime namespaces separate; added safe capture and tests | Exact deployment ID and reasoning mode unavailable; system Python lacked pytest and the project uv dev group was used | Ticket, docs, and explicit runtime display values | completed |
| not exposed by runtime / not exposed by runtime | independent review | Confirmed R0 acceptance, privacy, no inference, docs/ledger parity, and no ML-system impact | Source precedence is caller-enforced; JSON-only output is narrower than an early record phrase | Normative code head `c77c8c9` | PASS WITH NOTE |
| not exposed by runtime / not exposed by runtime | independent re-review | Confirmed final documentation/ledger parity and no semantic drift | None beyond retained caller-enforced source-precedence note | Exact final head `4d6306d555c275e07cbf376bb1cff8c26a74b2f0` | PASS WITH NOTE |
| not exposed by runtime / not exposed by runtime | repair re-review | Confirmed stale-head repair, exact final SHA parity, and no implementation drift | None beyond retained caller-enforced source-precedence note | Exact final head `552b74c80643178b346f128dd9ce90679be85f0f` | PASS WITH NOTE |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated per-model attempt, pass, repair, and review counts after implementation/review.
- [x] Confirmed that the PR execution trail matches this record.
- [ ] Recorded human merge or complete guarded self-merge authority/audit evidence.
- [x] Confirmed that this change does not alter the self-merge policy or historical records.
