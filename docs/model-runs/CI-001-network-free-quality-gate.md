# CI-001 - Network-free CPU quality gate

- PR: draft / pending creation
- Branch: `codex/ci-001-network-free-quality-gate`
- Ticket: `CI-001`
- Hypothesis: one canonical local command can prove the pull-request foundation
  (locked dependencies, lint, unit tests, Hydra composition, lock parity, and a
  network-blocked CPU smoke) without model/data downloads or credentials after
  dependency installation.
- Experiment record: `N/A` — this is an operations-quality ticket, not a model
  experiment.
- Started: 2026-07-12
- Final verdict: in progress
- Final record owner: `/root/ci001_implementation`

## Scope and decision context

- Goal: make every pull request demonstrate that the local, offline CPU path is
  still healthy.
- In scope: matching Make and GitHub Actions commands; frozen dependency sync;
  Ruff, pytest, Hydra composition, lock verification, and a tiny socket-blocked
  offline smoke; a separately-triggered network integration workflow.
- Out of scope: DGX performance, W&B credentials, full training, benchmarks,
  model/data changes, or changing Hydra's runtime/training configuration.
- Relevant `PHILOSOPHY.md` principles: executable canonical workflows; one
  reproducible dependency environment; small coherent changes; retained
  evidence; and no implicit online fallback.
- Baseline commit/run: `246e712127f9e7dca46a5c199440edb4d4ab55c6`.
- Intended evidence: local `make ci-cpu`, a forced lock-drift negative command,
  blocked-network smoke, and a pull-request workflow run.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `246e712`; CI-001, AGENTS, `PHILOSOPHY.md`, `CHECK.md` 1/7.3/8.1 | Requested Luna / Extra High; build the smallest matching local and PR gate | in progress | Selected a post-sync `--no-sync` command surface, a subprocess socket guard for the tiny local-manifest smoke, and separately-triggered network integration | implementation provenance below |

## Runtime provenance block

Requested/default values remain separate from actual runtime display.

| Namespace | Product | Displayed family | Exact model identifier | Reasoning mode | Source / unavailable reason |
| --- | --- | --- | --- | --- | --- |
| requested | Codex | Luna | Luna | Extra High | explicit task request |
| actual | Codex | GPT-5 | not exposed by runtime | not exposed by runtime | active runtime showed product/family only; exact deployment ID and reasoning setting were unavailable |

- Capture file/evidence: implementation capture at `2026-07-12T14:58:05Z` from `scripts/capture_model_provenance.py`; Codex CLI `codex-cli 0.144.1`; input branch/commit as above.
- Phase/role/task path: implementation / implementation / `/root/ci001_implementation`.
- Privacy confirmation: no prompts, hidden chain-of-thought, token counts,
  secrets, or raw thread ID are recorded.

## Check selection and verdicts

### Review cycle 1

- Review model / mode: pending independent reviewer; requested heavier reviewer /
  Extra Thinking, actual runtime identity must be recorded without inference.
- Commit reviewed: pending.
- Selected `CHECK.md` sections: 1, 7.3, 8.1 (CI-001 routing).
- Major sections marked N/A and why: model/data/training/GPU/performance are
  unchanged; the smoke is wiring evidence, not a performance or quality claim.
- Ticket acceptance result: pending.
- Philosophy alignment: pending.
- Complexity / change-surface result: pending.
- ML-system result: pending.
- Verdict: pending.

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| N/A | implementation | No review has run yet. | This live draft record | Obtain independent review before Ready/merge. |

## Failed-review handoff

N/A — no review has run.

## Repair result

N/A — no repair has run.

## Final evidence

- Resolved Hydra command/config: pending implementation.
- Data/tokenizer/model identity: the smoke must use the committed
  `smoke_overfit` manifest and canonical tokenizer, without online source access.
- Validation and measurements: pending.
- Performance/resource result if applicable: R0/R1 only; no DGX performance
  claim is in scope.
- Failed attempts retained at: this record.
- Known trade-offs: the socket guard will demonstrate Python-network isolation
  for the smoke; it is not a host firewall or a claim about external native code.
- Unresolved risks: hosted Actions behavior and the exact workflow run are
  pending PR creation.
- Human decision requested: none while implementation/review are in progress.

## Merge authority and final audit

- Merge path: guarded agent self-merge, only after the explicit bounded roadmap
  authorization and every current policy gate pass.
- Human authorization: user instruction “これからはとりあえず全部セルフマージしていいよ” for the bounded roadmap implementation series, recorded in the primary task context.
- Authorization covers this named PR or bounded ticket/goal series: yes — CI-001
  is within the stated roadmap goal.
- Exact independently reviewed head SHA: pending.
- Latest independent verdict / model / mode: pending.
- All actionable findings repaired and independently re-reviewed: pending.
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: pending.
- Newer human objections since authorization/review: none known.
- Human review dismissed by an agent: no.
- Unresolved review threads at final audit: pending.
- Branch-protection required-context inventory: pending.
- Applicable configured workflow/check inventory: pending.
- Observed exact-head check statuses: pending.
- Expected checks absent, pending, skipped, cancelled, or non-successful: pending.
- No-check evidence when both inventories are empty: N/A.
- Target branch and base SHA at final audit: pending.
- Up-to-date, conflict-free, and mergeable evidence: pending.
- Record, ledger, PR trail, validation, and risks parity: pending.
- Prohibited self-merge categories: clear only if final audit confirms no secrets,
  security-control change, paid resource, destructive action, release, deployment,
  account/permission change, or unresolved licensing issue.
- Admin/bypass/force/disabled-check requirement: must be no.
- Final audit PR body/comment location: pending.
- Final audit changed reviewed head: pending.
- Immediate pre-merge re-fetch/compare observation location: pending.
- Immediate refresh compared authorization, head, base, review decision/objections,
  threads, expected checks/statuses, and mergeability: pending.
- Drift found: pending.
- Merge outcome: pending.

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | implementation | In progress | Pending review | CI-001 acceptance criteria, existing Make/config paths, and `CHECK.md` route | pending |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md` as draft/pending.
- [ ] Updated per-model attempt, pass, repair, and review counts after outcome.
- [ ] Confirmed that the PR execution trail matches this record.
- [ ] Recorded complete guarded self-merge authority/audit evidence.
- [x] Confirmed that this is not the bootstrap policy PR.
