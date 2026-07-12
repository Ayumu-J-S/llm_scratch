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
| 1 | implementation | not exposed by runtime | not exposed by runtime | `a05eb1de5656643757a1c3d98047c98dedea8bfa`; requested Luna / Extra High | Implement the explicitly requested guarded self-merge policy and bootstrap it under the current human-only rule | implemented | Added one consistent fail-closed gate contract, updated the workflow and templates, and preserved human merge for this bootstrap PR | Initial record `210a433`; draft PR #16; implementation diff based on `063f57b` |

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
- Validation and measurements: `git diff --check` passed; targeted `rg` found every required gate on the normative and template surfaces; template dry-run cases route unscoped authorization, any failed/missing required check, a prohibited category, and this bootstrap PR to human merge, while an explicitly authorized routine PR reaches self-merge only after every exact-head gate and final PR audit.
- Performance/resource result if applicable: N/A — R0 documentation-only change.
- Failed attempts retained at: this timeline and any later failed-review sections.
- Known trade-offs: self-merge reduces the human handoff bottleneck but raises the cost of unclear authorization or incomplete evidence, so every gate is fail-closed.
- Unresolved risks: policy text is not enforceable CI and still depends on an agent performing the recorded audit honestly; independent review is pending.
- Human decision requested: review and merge this bootstrap policy PR; it cannot authorize its own merge.

## Merge authority and final audit

- Merge path: `human merge`
- Human authorization: N/A — the explicit request to change policy does not override the bootstrap rule or authorize this PR to merge itself.
- Authorization evidence location: user request in the task that produced PR #16; recorded here and in the PR body as bootstrap-only human merge.
- Authorization covers this named PR or bounded ticket/goal series: N/A — human merge required.
- Exact independently reviewed head SHA: pending.
- Latest independent verdict / model / mode: pending.
- All actionable findings repaired and independently re-reviewed: pending.
- Unresolved review threads at final audit: pending.
- Required/configured checks at exact head: pending.
- Target branch and base SHA at final audit: `main` / pending refresh.
- Up-to-date, conflict-free, and mergeable evidence: pending.
- Record, ledger, PR trail, validation, and risks parity: pending.
- Prohibited self-merge categories: blocked — this PR introduces the self-merge governance control and is explicitly subject to the bootstrap rule.
- Admin/bypass/force/disabled-check requirement: no.
- Final audit PR body/comment location: pending human handoff.
- Final audit changed reviewed head: no changes permitted after review without re-review.
- Merge outcome: pending human merge.

## Model assessment from this ticket

Record observable outcomes, not hidden chain-of-thought.

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | implementation | Started the live PR before edits; kept the change on seven named documentation surfaces; handled the exact-head audit without creating a commit/review loop | Independent review still required | Explicit guarded-policy contract, bootstrap constraint, and exact allowed files | implemented |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated per-model attempt, pass, repair, and review counts.
- [ ] Confirmed that the PR execution trail matches this record.
- [x] Recorded human merge or complete guarded self-merge authority/audit evidence.
- [x] Confirmed that this bootstrap policy rule was not used before a human merged it.
