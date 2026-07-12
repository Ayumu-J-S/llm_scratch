# PROV-001 - Make Codex model provenance visible

- PR: [#17](https://github.com/Ayumu-J-S/llm_scratch/pull/17) (draft)
- Branch: `codex/prov-001-model-provenance`
- Ticket: PROV-001 (repository provenance contract requested after the first roadmap wave)
- Hypothesis: a small, redaction-safe capture command that separates requested/default model settings from explicitly supplied runtime display will make each agent phase auditable without guessing hidden model identity or leaking prompts, tokens, or secrets.
- Experiment record: `N/A` — documentation/tooling provenance change; no ML experiment
- Started: 2026-07-12
- Final verdict: in progress
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
| 1 | review | not exposed by runtime | not exposed by runtime | pending implementation head; requested heavier / Extra Thinking | Independently review ticket, philosophy, and applicable `CHECK.md` sections | pending | Must verify schema separation, redaction, and live handoff parity | Pending |

Allowed phases: `implementation`, `review`, `repair`, `re-review`, and `handoff`.

## Check selection and verdicts

### Review cycle 1

- Review model / mode: pending / pending
- Commit reviewed: pending
- Selected `CHECK.md` sections: R0 documentation/reproducibility, experiment identity, research-integrity/privacy, and change-surface review
- Major sections marked N/A and why: data, tokenizer, model, optimizer, CUDA, performance, checkpoint, and W&B runtime behavior are N/A unless the reviewer finds an integration effect; this ticket adds only stdlib capture tooling and documentation.
- Ticket acceptance result: pending
- Philosophy alignment: pending
- Complexity / change-surface result: pending
- ML-system result: N/A pending reviewer confirmation
- Verdict: pending

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| pending | pending | pending | pending | pending |

## Failed-review handoff

N/A — no failed independent review has occurred.

## Repair result

N/A — no repair cycle has occurred.

## Final evidence

- Resolved Hydra command/config: `N/A` — no Hydra/runtime training path changed.
- Data/tokenizer/model identity: `N/A` — no scientific run.
- Validation and measurements: `uv run --project /tmp/llm_scratch-provenance-001 --group dev pytest tests/test_model_provenance.py -q` → `4 passed`; CLI smoke capture emitted valid JSON with requested `gpt-5.6-sol`/`xhigh` separate from actual `Codex`/`GPT-5` and unavailable exact ID/mode.
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
- Exact independently reviewed head SHA: pending
- Latest independent verdict / model / mode: pending
- All actionable findings repaired and independently re-reviewed: pending
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: pending human review
- Newer human objections since authorization/review: pending
- Human review dismissed by an agent: no
- Unresolved review threads at final audit: pending
- Branch-protection required-context inventory: pending
- Applicable configured workflow/check inventory: pending
- Observed exact-head check statuses: pending
- Expected checks absent, pending, skipped, cancelled, or non-successful: pending
- No-check evidence when both inventories are empty: pending
- Target branch and base SHA at final audit: `main` / pending refresh
- Up-to-date, conflict-free, and mergeable evidence: pending
- Record, ledger, PR trail, validation, and risks parity: pending
- Prohibited self-merge categories: clear for this documentation/tooling change; human merge remains default
- Admin/bypass/force/disabled-check requirement: no
- Final audit PR body/comment location: pending
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

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [ ] Updated per-model attempt, pass, repair, and review counts after implementation/review.
- [ ] Confirmed that the PR execution trail matches this record.
- [ ] Recorded human merge or complete guarded self-merge authority/audit evidence.
- [x] Confirmed that this change does not alter the self-merge policy or historical records.
