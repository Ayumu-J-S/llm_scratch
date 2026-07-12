# POLICY-001 - Guarded Agent Self-Merge

- PR: [#16](https://github.com/Ayumu-J-S/llm_scratch/pull/16) (draft)
- Branch: `codex/policy-001-agent-self-merge`
- Ticket: POLICY-001 (explicit repository-policy request; not a `ROADMAP.md` ticket)
- Hypothesis: Explicit, bounded human authorization plus independent review and fail-closed merge gates can permit routine agent self-merge without turning tool access or a passing self-review into merge authority.
- Started: 2026-07-12
- Final verdict: PASS WITH NOTE
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
| 1 | review | not exposed by runtime | not exposed by runtime | `00590be31e50c958b274f1effb4a83ea0125986e`; independent POLICY-001 review | Review authorization, review state, check state, mutation races, prohibited categories, workflow, and templates | FAIL | Found four merge-safety gaps: blocking review/objection state, expected-check inventory, immediate drift refresh, and overbroad external-communication prohibition | Independent review handoff received 2026-07-12 |
| 1 | repair | not exposed by runtime | not exposed by runtime | Failed review of `00590be`; same seven-file scope | Repair all four findings and add adversarial validation cases without readying, reviewing, or merging PR #16 | implemented | Added fail-closed review-decision, expected-check inventory, just-before-merge refresh, and narrowed non-routine external-action rules | Exact seven-file repair audit and adversarial cases passed |
| 1 | re-review | not exposed by runtime | not exposed by runtime | `fb7280837d32295f378f9eed16e3931bab404261`; requested heavier model / Extra Thinking | Independently re-review all repaired gates against POLICY-001, `PHILOSOPHY.md`, and applicable R0 `CHECK.md` guidance | PASS WITH NOTE | All four findings are resolved with no actionable findings; documentation controls still depend on honest execution, and authenticated branch-protection inventory was unavailable | Independent re-review received 2026-07-12 |
| 1 | handoff | not exposed by runtime | not exposed by runtime | Passing normative head `fb72808` and re-review result | Finalize provenance, ledger, human-bootstrap merge fields, and live PR trail without changing normative policy files | implemented | Final documentation-only handoff prepared; exact-head parity audit remains before readiness | Final provenance diff |

Allowed phases: `implementation`, `review`, `repair`, `re-review`, and `handoff`.

## Check selection and verdicts

### Review cycle 1 — FAIL

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: `00590be31e50c958b274f1effb4a83ea0125986e`
- Selected `CHECK.md` sections: R0 documentation/policy surface, workflow integrity, review evidence, and fail-closed handoff behavior
- Major sections marked N/A and why: data, tokenizer, model, optimizer, CUDA, performance, W&B, checkpoint, and training sections are N/A because the PR changes only governance documentation.
- Ticket acceptance result: FAIL — four required guarded-merge states were incomplete or overbroad.
- Philosophy alignment: FAIL — ordinary repository collaboration was unintentionally included in a protected-action phrase, while mutable review/check state was under-specified.
- Complexity / change-surface result: PASS — the change remained on the seven allowed documentation surfaces.
- ML-system result: N/A — no runtime, data, model, configuration, or training behavior changed.
- Verdict: FAIL

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| High | Review decision | The gate did not independently require a clear GitHub review decision, absence of `CHANGES_REQUESTED` and newer human objections, or prohibit agent dismissal of human reviews | `AGENTS.md`, `PHILOSOPHY.md`, workflow and templates at `00590be` | Add explicit, separately evidenced fail-closed review-decision gates |
| High | Expected checks | An empty observed status list could be treated as the no-check case without inventorying protected required contexts and applicable configured workflows | Workflow and template no-check wording at `00590be` | Separate expected inventories from exact-head observations; block every absent or non-success expected check |
| High | State drift | The final audit lacked an immediate just-before-merge re-fetch/compare of mutable authorization and GitHub state | Workflow after final-audit node at `00590be` | Add and record an immediate refresh; abort and revalidate/re-review on drift |
| Medium | Protected scope | `external communication` was broad enough to prohibit routine PR comments and evidence coordination | `PHILOSOPHY.md` and workflow prohibited-category lists at `00590be` | Narrow to non-routine external action outside ordinary repository collaboration |

### Review cycle 2 — PASS WITH NOTE

- Review model / mode: not exposed by runtime / not exposed by runtime (requested heavier model / Extra Thinking)
- Commit reviewed: `fb7280837d32295f378f9eed16e3931bab404261`
- Selected `CHECK.md` sections: R0 documentation/policy surface, minimum review, experiment handoff, and changeability
- Major sections marked N/A and why: data, tokenizer, model, optimizer, CUDA, performance, W&B, checkpoint, and training sections remain N/A because no runtime or ML-system behavior changed.
- Ticket acceptance result: PASS — every requested guarded-merge and bootstrap condition is present on the normative and template surfaces.
- Philosophy alignment: PASS — explicit authority remains bounded and fail-closed, high-impact actions remain human-only, and ordinary repository collaboration remains allowed.
- Complexity / change-surface result: PASS — the repair remained within the seven allowed documentation surfaces and added no enforcement framework.
- ML-system result: N/A — no runtime, data, model, configuration, or training behavior changed.
- Verdict: PASS WITH NOTE — controls are documentation-enforced and depend on honest execution. Repository API inspection reported no workflows, rulesets, or exact-head statuses, but authenticated branch-protection inventory was unavailable; therefore no claim is made that no required contexts exist. This uncertainty does not affect readiness for a human merge of the bootstrap PR.

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| Note | Enforcement | The repaired policy resolves all four failed-review findings; enforcement remains procedural rather than CI-backed | Independent review of `fb72808` and adversarial policy cases | None for this PR; retain as an explicit trade-off |
| Note | Check inventory | Repository APIs reported no workflows, rulesets, or statuses, but authenticated branch-protection required-context inventory was unavailable | Independent re-review environment | Do not infer or record a no-required-context conclusion; a future authorized self-merge must obtain the required inventory |

## Failed-review handoff

- From review cycle: 1
- Failed check and why: Four fail-closed policy requirements were absent or overbroad: blocking review/objection state, expected-check discovery, immediate mutable-state refresh, and protected external-action scope.
- Review model / mode: not exposed by runtime / not exposed by runtime
- Implementation model / mode that produced the failed state: not exposed by runtime / not exposed by runtime
- Commit/diff to repair: `00590be31e50c958b274f1effb4a83ea0125986e`
- Reproduction command or evidence: read the guarded gates and dry-run a `CHANGES_REQUESTED` review with resolved threads, an expected workflow absent from the status list, state drift after audit, and a routine PR comment.
- Relevant files/config/manifests: `AGENTS.md`, `PHILOSOPHY.md`, `docs/agent-model-workflow.md`, PR/model-run templates, this record, and the ledger; no Hydra or ML manifests apply.
- Attempts already made: initial targeted `rg` and template dry run covered authorization, generic failed/missing checks, prohibited categories, and bootstrap behavior, but overclaimed complete gate coverage.
- Invariants and constraints: human default; explicit bounded authorization; exact-head independent review; no bypass; prohibited high-impact categories; bootstrap remains human-only; exactly seven allowed files; historical records, `ROADMAP.md`, and `CHECK.md` unchanged.
- Selected next model / mode: not exposed by runtime / not exposed by runtime
- Why this model was selected: the defects are bounded policy-wiring and evidence-template omissions with an exact repair contract.
- Exact repair request: implement the four review findings verbatim, keep PR #16 draft, and do not mutate reviews, threads, readiness, or merge state.
- Completion evidence requested: consistent normative/template wording, adversarial dry runs for each failure mode, exact seven-file scope, Markdown fence parity, and `git diff --check` before fresh independent re-review.

## Repair result

- Repair cycle: 1
- Repair model / mode: not exposed by runtime / not exposed by runtime
- Input handoff: independent FAIL of `00590be` and the four exact findings above.
- Changes made: separated blocking review decisions and human objections from unresolved threads; prohibited agent dismissal of human reviews; split required/configured check inventories from exact-head observations; made every expected non-success state blocking; added immediate pre-merge drift refresh; narrowed protected external actions to non-routine actions outside ordinary repository collaboration.
- What was deliberately not changed: human-merge default, bounded authorization model, bootstrap rule, high-impact prohibited categories, `ROADMAP.md`, `CHECK.md`, historical records, code, CI, and runtime behavior.
- Local evidence: exact seven-file scope; `ROADMAP.md`, `CHECK.md`, and historical records untouched; targeted normative/template `rg`; Markdown fence parity; nine adversarial decisions; and `git diff --check` all passed.
- Commit reviewed next: repair head produced by this cycle; exact SHA is recorded in the PR trail and fresh review handoff after push.
- Re-review model / mode: not exposed by runtime / not exposed by runtime (requested heavier model / Extra Thinking).
- Re-review verdict: PASS WITH NOTE at `fb7280837d32295f378f9eed16e3931bab404261`; no actionable findings.

## Final evidence

- Resolved Hydra command/config: N/A — policy-only documentation change.
- Data/tokenizer/model identity: N/A — no ML system or data path changed.
- Validation and measurements: Initial validation passed `git diff --check` and its stated cases but overclaimed full guarded-gate coverage. Repair validation passed exact seven-file scope, unchanged `ROADMAP.md`/`CHECK.md`/historical records, targeted normative and separately evidenced template fields, Markdown fence parity, and `git diff --check`. Adversarial dry runs now block `CHANGES_REQUESTED` despite zero threads, a newer human objection, an agent-dismissed human review, every absent/pending/skipped/cancelled/non-successful expected check, an empty observation without required/configured/expected inventory evidence, any immediate-refresh drift, releases/deployments/account-permission changes, and this bootstrap PR; routine PR comments and evidence coordination remain allowed.
- Performance/resource result if applicable: N/A — R0 documentation-only change.
- Failed attempts retained at: this timeline and any later failed-review sections.
- Known trade-offs: self-merge reduces the human handoff bottleneck but raises the cost of unclear authorization or incomplete evidence, so every gate is fail-closed.
- Unresolved risks: policy text is not enforceable CI and still depends on an agent performing the recorded audit honestly. Authenticated branch-protection inventory was unavailable during review, so the absence of reported workflows, rulesets, and statuses is not evidence that no required contexts exist.
- Human decision requested: review and merge this bootstrap policy PR; it cannot authorize its own merge.

## Merge authority and final audit

- Merge path: `human merge`
- Human authorization: N/A — the explicit request to change policy does not override the bootstrap rule or authorize this PR to merge itself.
- Authorization evidence location: user request in the task that produced PR #16; recorded here and in the PR body as bootstrap-only human merge.
- Authorization covers this named PR or bounded ticket/goal series: N/A — human merge required.
- Exact independently reviewed head SHA: `fb7280837d32295f378f9eed16e3931bab404261` (normative policy head); the following finalization changes provenance/ledger only and requires exact-head parity audit before readiness.
- Latest independent verdict / model / mode: PASS WITH NOTE / not exposed by runtime / not exposed by runtime (requested heavier model / Extra Thinking).
- All actionable findings repaired and independently re-reviewed: yes; no actionable findings remain.
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: not asserted as an agent-merge gate; this bootstrap PR requires human merge.
- Newer human objections since authorization/review: none reported to the independent reviewer; human remains the merge authority.
- Human review dismissed by an agent: no; this PR is human-merge-only.
- Unresolved review threads at final audit: not asserted as an agent-merge gate; pending human review/handoff inspection.
- Branch-protection required-context inventory: unknown — authenticated branch-protection inventory was unavailable; no absence claim is made.
- Applicable configured workflow/check inventory: repository API reported no workflows or rulesets during independent review; this does not substitute for authenticated branch-protection inventory.
- Observed exact-head check statuses: repository API reported no statuses for `fb72808`; not characterized as a passing no-check state.
- Expected checks absent, pending, skipped, cancelled, or non-successful: not determined and not waived; agent self-merge is prohibited for this bootstrap PR.
- No-check evidence when both inventories are empty: insufficient to claim the no-check case; N/A to human-bootstrap readiness.
- Target branch and base SHA at final audit: `main` / `a05eb1de5656643757a1c3d98047c98dedea8bfa` at normative re-review.
- Up-to-date, conflict-free, and mergeable evidence: local fetch showed `origin/main` at `a05eb1d` and an ancestor of the branch; PR API reported mergeable after repair. Primary parity audit must refresh this after finalization.
- Record, ledger, PR trail, validation, and risks parity: finalized here and in the live PR; primary exact-head parity audit pending.
- Prohibited self-merge categories: blocked — this PR introduces the self-merge governance control and is explicitly subject to the bootstrap rule.
- Admin/bypass/force/disabled-check requirement: no.
- Final audit PR body/comment location: PR #16 body records the human-bootstrap path; no agent self-merge audit applies.
- Final audit changed reviewed head: no changes permitted after review without re-review.
- Immediate pre-merge re-fetch/compare observation location: N/A — human merge required.
- Immediate refresh compared authorization, head, base, review decision/objections, threads, expected checks/statuses, and mergeability: N/A — human merge required.
- Drift found: N/A — human merge required.
- Merge outcome: pending human merge.

## Model assessment from this ticket

Record observable outcomes, not hidden chain-of-thought.

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | implementation/repair | Started the live PR before edits; kept both implementation and repair on seven named documentation surfaces; accepted a precise failed-review handoff | Initial pass missed distinct blocking-review state, expected-check discovery, immediate state refresh, and overbroad external-action wording | Explicit guarded-policy contract, adversarial review findings, bootstrap constraint, and exact allowed files | successful repair |
| not exposed by runtime / not exposed by runtime | review/re-review | Found four subtle authority/state-race gaps, then verified every repair without expanding the change surface | Authenticated branch-protection inventory was unavailable, correctly retained as uncertainty rather than a no-check claim | Exact heads, accepted policy contract, normative files, templates, and failed-review handoff | FAIL, then PASS WITH NOTE |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated per-model attempt, pass, repair, and review counts.
- [x] Confirmed that the PR execution trail matches this record after finalization.
- [x] Recorded human merge or complete guarded self-merge authority/audit evidence.
- [x] Confirmed that this bootstrap policy rule was not used before a human merged it.
