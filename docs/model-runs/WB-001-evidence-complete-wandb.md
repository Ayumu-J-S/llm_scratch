# WB-001 — Evidence-complete, quota-safe W&B

- PR: unavailable during delegated implementation; complete body at
  `/tmp/WB-001-pr-body.md`
- Branch: `codex/wb-001-evidence-safe-wandb`
- Ticket: `WB-001`
- Hypothesis: optional W&B tracking can remain compact and useful without
  becoming training authority, bulk storage, or a local failure mode.
- Experiment record: `docs/experiments/WB-001-evidence-complete-wandb.md`
- Started: 2026-07-13
- Final verdict: in progress — independent heavy review pending
- Final record owner: lead roadmap agent

## Scope and decision context

- Goal: retain useful experiment evidence while keeping W&B Free-plan safe.
- In scope: modes, existing scalar cadence, watch off by default, compact
  training/validation/system schema and lineage, explicit artifact policy,
  authenticated viewer/entity plus fresh visible usage preflight, projected
  size/headroom, deduplication, offline/disabled operation, local failure
  evidence, documentation, and predeclared target comparison.
- Out of scope: raw dataset upload, hard-coded quota, private API, mass deletion,
  W&B checkpoint backup, online credentials/service mutation, a long R3 pilot,
  or unrelated model/data/training changes.
- Relevant policy: `PHILOSOPHY.md` W&B experiment evidence without bulk
  storage; best/final/explicit milestone retention; current usage/retention
  check; agent-native local evidence and failure preservation.
- Baseline commit: `74d9e24c251b62b23892b11ba0c1c9c723cd8a12`.
- Intended evidence: strict test doubles and config tests, artifact matrix,
  missing/stale/auth/quota/upload-failure paths, scalar cadence/schema,
  identity exclusion, and a real socket-blocked disabled/offline CPU smoke.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `74d9e24` plus WB-001 ticket/research | Requested `gpt-5.6-luna` / Extra High (`xhigh`) or higher; implement smallest direct Hydra WB-001 path | implemented; review pending | Added failure-isolated tracker, strict disabled/offline/online config, compact existing-cadence schema, watch opt-in/teardown, sanitized dataset references, fail-closed artifact preflight/upload evidence, operational identity exclusion, docs/tests, and predeclared R2 | implementation provenance JSON; focused tests; socket-blocked smoke |
| 1 | review | not exposed by runtime | not exposed by runtime | uncommitted implementation snapshot | Read-only precommit acceptance audit; not the mandatory heavy review | FAIL | Undefined upload summary caused contradictory evidence; SDK string-team auth failed; artifact kind/identity/step, runtime/run/artifact IDs, non-finite events, socket coverage, and DGX evidence were incomplete | Reproductions and ordered findings from `/root/wb001_precommit_audit` |
| 1 | repair | not exposed by runtime | not exposed by runtime | failed precommit audit plus official W&B 0.25.1 behavior | Repair every acceptance-blocking audit finding and expand invariant tests | repaired; mandatory review pending | Bound artifacts to verified repository checkpoints, fixed team auth and summaries, bounded finish/evaluation isolation, exact validation cadence, stability/runtime evidence, socket guard, and 31 artifact tests | focused `115 passed`; full `361 passed, 1 skipped`; offline smoke pass |
| 1 | repair | not exposed by runtime | not exposed by runtime | predeclared R2 protocol and applicable CHECK | Implement transparent exact-head DGX runner and verifier before measurement | implemented; run pending | Added fixed cache-prime plus 9-run Latin-square, network/read-only/image guards, raw samplers/hashes, exact trajectory/checkpoint/lifecycle/resource/storage and paired gates | Bash syntax, Ruff/format, and 10 verifier tests pass |
| 2 | review | pending | pending | exact integration/evidence head pending | Independent heavy review against philosophy, ticket, and applicable CHECK | pending | pending | pending |

Requested values are recorded separately from actual runtime display. The
delegated runtime did not expose the actual model identifier or reasoning mode;
they are not inferred from the request.

## Runtime provenance block

| Namespace | Product | Displayed family | Exact model identifier | Reasoning mode | Source / unavailable reason |
| --- | --- | --- | --- | --- | --- |
| requested | Codex | GPT-5.6 Luna-equivalent implementation tier | `gpt-5.6-luna` | Extra High (`xhigh`) or higher | explicit task request |
| actual | not exposed by runtime | not exposed by runtime | not exposed by runtime | not exposed by runtime | active delegated runtime did not display these values |

- Capture files:
  `docs/model-runs/evidence/WB-001-implementation-provenance.json`,
  `docs/model-runs/evidence/WB-001-precommit-review-provenance.json`,
  `docs/model-runs/evidence/WB-001-repair-provenance.json`, and
  `docs/model-runs/evidence/WB-001-r2-harness-provenance.json`
- Codex CLI version: `codex-cli 0.144.1`
- Branch/commit: `codex/wb-001-evidence-safe-wandb` / input `74d9e24`
- Phase/role/task path: implementation / implementation /
  `/root/wb001_implementation`
- Privacy confirmation: no prompts, hidden chain-of-thought, token counts,
  secrets, or raw thread ID are recorded.

## Check selection and verdicts

### Review cycle 1 — precommit QA (not the mandatory heavy review)

- Review model / mode: actual model and reasoning mode not exposed by runtime
- Commit reviewed: uncommitted implementation snapshot
- Selected `CHECK.md` sections: minimum review; shared comparison conditions;
  §5.4 memory/storage; §6.3 logging cadence/synchronization; §7 changeability;
  §8.1 run identity; §9.2 W&B/logging.
- Major sections marked N/A and why: data/tokenizer/model math and benchmark
  quality do not change; real GPU performance remains predeclared R2 rather than
  an implementation-pass claim.
- Ticket acceptance result: FAIL before repair.
- Philosophy alignment: local-authority direction was sound, but contradictory
  artifact evidence and missing identity/stability fields violated the evidence
  contract.
- Complexity / change-surface result: direct repair retained one tracking
  module and existing trainer boundaries.
- ML-system result: R1 correctness passed after repair; R2 remained pending.
- Verdict: FAIL; repaired before the mandatory heavy review.

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| P0 | artifact outcome | Successful committed upload called an undefined summary helper and was then returned as failed | reproduced focused failure | implement safe summary persistence and retest success/failure outcomes |
| P1 | auth | W&B 0.25.1 returns team names as strings | real-shaped reproduction | accept supported string teams and retain entity check |
| P1 | checkpoint trust | Caller reason was attached to arbitrary bytes | raw-byte success tests | verify checkpoint schema, internal kind, run identity, and step |
| P1 | evidence | Run/artifact/runtime identity and non-finite events were incomplete | static audit | persist compact identities and stability events |
| P2 | offline claim | `connect_ex` and `sendto` escaped the guard | live socket reproduction | intercept both and narrow documented claim |
| P1 | target evidence | Required R2 comparison had not run | experiment record | commit harness and execute exact-head matrix before heavy review |

## Failed-review handoff

- From review cycle: precommit QA cycle 1.
- Failed check and why: artifact outcome consistency, public auth shape,
  checkpoint trust, evidence identity/stability, offline isolation, and target
  evidence were incomplete.
- Review model / mode: not exposed by runtime / not exposed by runtime.
- Implementation model / mode that produced the failed state: requested
  `gpt-5.6-luna` / Extra High; actual not exposed.
- Commit/diff to repair: uncommitted WB-001 integration diff.
- Reproduction command or evidence: focused test failure plus direct W&B
  0.25.1 viewer/socket probes retained in the audit handoff.
- Invariants and constraints: Hydra only; local evidence authoritative; no raw
  corpus; no private quota API; no online upload; no compatibility shim.
- Selected next model / mode: lead integration repair, actual runtime values not
  exposed, followed by requested `gpt-5.6-sol` / Max independent review.
- Exact repair request: close all ordered findings and prove them with
  repository checkpoint, SDK-shaped, failure-isolation, and socket tests.
- Completion evidence requested: focused/full tests, offline smoke, R2, and
  mandatory exact-head re-review.

## Repair result

- Repair cycle: 1.
- Repair model / mode: not exposed by runtime / not exposed by runtime.
- Input handoff: complete precommit QA findings plus official W&B 0.25.1
  research.
- Changes made: checkpoint binding; team-string support; non-contradictory
  summaries; run/artifact/runtime/stability identity; bounded finish;
  evaluation failure isolation; exact validation logging; expanded socket
  guard; invariant tests.
- What was deliberately not changed: model, objective, data, optimizer,
  training cadence, online service state, quota, and artifact default.
- Local evidence: focused `115 passed`; full `361 passed, 1 skipped`; real
  disabled/offline smoke pass.
- Commit reviewed next: pending exact evidence head.
- Re-review model / mode: requested `gpt-5.6-sol` / Max.
- Re-review verdict: pending.

## Final evidence

- Resolved Hydra controls: `wandb.mode=disabled|offline|online`, watch nested
  config, and artifact `none|best|final|milestone` with operator usage path,
  freshness, and reserve bytes; full example in `docs/wandb.md`.
- Data/tokenizer/model identity: unchanged. W&B config receives only manifest
  and external-reference metadata, never inline documents. Compact summary uses
  checkpoint-owned experiment/Git/config/lock/tokenizer/data fingerprints.
- Validation and measurements: focused integration `115 passed in 11.26s` plus
  `10 passed` for the R2 verifier; full repository suite `361 passed, 1 skipped
  in 68.20s`. Canonical disabled
  plus offline smoke passed with credentials removed and name resolution,
  `connect`, `connect_ex`, and `sendto` blocked. Ruff, changed-file format,
  lock, and diff checks pass.
- Performance/resource result: R1 only. No throughput or DGX claim. Three-arm,
  three-repetition, 100-step DGX protocol is predeclared before measurement.
- Failed attempts retained: first offline smoke placed its W&B directory under
  the repository; it was removed and `WANDB_DIR` now points at the smoke temp
  root. No training failure occurred.
- Known trade-offs: online artifact confirmation waits outside optimizer work at
  explicit retention boundaries; fresh quota visibility requires an operator
  Billing UI/CSV snapshot because the public Python API does not expose current
  storage usage.
- Unresolved risks: real online auth/service behavior and DGX overhead are not
  exercised; both fail closed or remain predeclared rather than claimed.
- Human decision requested: human review/merge after a passing independent
  review; no self-merge authorization exists.

## Merge authority and final audit

- Merge path: human merge
- Human authorization: `N/A — human merge`
- Authorization evidence location: N/A
- Authorization covers this named PR or bounded ticket/goal series: N/A
- Exact independently reviewed head SHA: pending
- Latest independent verdict / model / mode: pending
- All actionable findings repaired and independently re-reviewed: pending
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: pending
- Newer human objections since authorization/review: none observed
- Human review dismissed by an agent: no
- Unresolved review threads at final audit: pending
- Branch-protection required-context inventory: pending lead-agent final audit
- Applicable configured workflow/check inventory: pending lead-agent final audit
- Observed exact-head check statuses: pending
- Expected checks absent, pending, skipped, cancelled, or non-successful: pending
- No-check evidence when both inventories are empty: N/A
- Target branch and base SHA at final audit: stacked branch; refresh pending
- Up-to-date, conflict-free, and mergeable evidence: pending
- Record, ledger, PR trail, validation, and risks parity: pending
- Prohibited self-merge categories: clear for ordinary repo implementation, but
  no authorization exists and online credentials/quota were not used
- Admin/bypass/force/disabled-check requirement: no
- Final audit PR body/comment location: pending human merge
- Final audit changed reviewed head: no final audit yet
- Immediate pre-merge re-fetch/compare observation location: pending
- Immediate refresh compared authorization, head, base, review
  decision/objections, threads, expected checks/statuses, and mergeability: no
- Drift found: pending
- Merge outcome: not merged

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| actual not exposed / actual mode not exposed; requested `gpt-5.6-luna` / Extra High | implementation | Kept the design within one Hydra surface, isolated external failures, used public SDK behavior, and added policy/identity/cadence tests | First offline smoke used W&B's default repository-local run directory; repaired by an explicit temporary `WANDB_DIR` | WB-001 scope, official SDK research, existing trainer/checkpoint/VAL contracts, CHECK router | implemented; independent review pending |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated implementation attempt count; review/pass/repair counts remain
  unchanged until a review returns.
- [ ] Confirmed that the final PR execution trail matches this record.
- [x] Recorded human merge as the default.
- [x] Confirmed this is not the bootstrap policy PR.
