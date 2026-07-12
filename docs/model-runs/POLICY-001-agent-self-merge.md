# POLICY-001 - Guarded Agent Self-Merge

- PR: [#16](https://github.com/Ayumu-J-S/llm_scratch/pull/16) (draft)
- Branch: `codex/policy-001-agent-self-merge`
- Ticket: POLICY-001 (explicit repository-policy request; not a `ROADMAP.md` ticket)
- Hypothesis: Explicit, bounded human authorization plus independent review and fail-closed merge gates can permit routine agent self-merge without turning tool access or a passing self-review into merge authority.
- Started: 2026-07-12
- Final verdict: in progress
- Final record owner: implementation agent

## Scope and decision context

- Goal: Replace the unconditional human-only merge rule with a guarded, auditable self-merge policy while retaining human review as the default.
- In scope: `AGENTS.md`, `PHILOSOPHY.md`, the agent workflow, PR and model-run templates, this record, and the ledger.
- Out of scope: historical model-run records, `ROADMAP.md`, `CHECK.md`, repository rulesets, CI configuration, and any product or ML implementation.
- Relevant `PHILOSOPHY.md` principles: agent-native/human-legible operation, explicit authority, research integrity, reviewable evidence, safe reversible action, and avoiding destructive or externally consequential work.
- Baseline commit/run: `a05eb1de5656643757a1c3d98047c98dedea8bfa`; documentation-only R0 policy change, so no training run applies.
- Intended evidence: consistent normative gates across all instruction surfaces; a template dry run that cannot authorize this bootstrap PR; Markdown/diff validation; independent policy review before readiness.

## Execution timeline

One row represents one model invocation or one clearly bounded phase. Never
delete a failed row.

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `a05eb1de5656643757a1c3d98047c98dedea8bfa`; requested Luna / Extra High | Implement the explicitly requested guarded self-merge policy and bootstrap it under the current human-only rule | in progress | Started the live record and draft PR before policy edits | Initial record commit `210a433`; draft PR #16 |

Allowed phases: `implementation`, `review`, `repair`, `re-review`, and `handoff`.

## Check selection and verdicts

### Review cycle 1

- Review model / mode: pending independent review
- Commit reviewed: pending
- Selected `CHECK.md` sections: pending; expected R0 documentation/policy review only
- Major sections marked N/A and why: pending
- Ticket acceptance result: pending
- Philosophy alignment: pending
- Complexity / change-surface result: pending
- ML-system result: pending
- Verdict: pending

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| N/A | Pending | Independent review has not run | Pending implementation commit | Run independent review before readiness |

## Failed-review handoff

N/A — no review has run.

## Repair result

N/A — no repair cycle has run.

## Final evidence

- Resolved Hydra command/config: N/A — policy-only documentation change.
- Data/tokenizer/model identity: N/A — no ML system or data path changed.
- Validation and measurements: pending.
- Performance/resource result if applicable: N/A — R0 documentation-only change.
- Failed attempts retained at: this timeline and any later failed-review sections.
- Known trade-offs: self-merge reduces the human handoff bottleneck but raises the cost of unclear authorization or incomplete evidence, so every gate is fail-closed.
- Unresolved risks: policy text and templates are not implemented or independently reviewed yet.
- Human decision requested: review and merge this bootstrap policy PR; it cannot authorize its own merge.

## Model assessment from this ticket

Record observable outcomes, not hidden chain-of-thought.

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | implementation | Pending | Pending | Explicit guarded-policy contract and exact allowed files | in progress |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [ ] Updated per-model attempt, pass, repair, and review counts.
- [ ] Confirmed that the PR execution trail matches this record.
