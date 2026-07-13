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
| 2 | repair | not exposed by runtime | not exposed by runtime | failed DGX Attempt 2 at `049acf7` | Diagnose retained R2 evidence and repair only demonstrated measurement/default defects | repaired; retry pending | Preserved failed projection; raised compute intensity to sequence 64; restored official 1,000-batch watch interval; made Docker stats truly streaming; excluded random container hostname from stable hardware equality | `docs/experiments/evidence/WB-001-dgx-r2-failed.json`; exact trajectories/checkpoints passed while data wait, sampler coverage, and watch overhead failed |
| 2 | review | not exposed by runtime | not exposed by runtime | clean adaptive head `ee4d41d` | Read-only prelaunch protocol audit; not the mandatory heavy review | FAIL; Attempt 3 aborted before measured arms | Default 1,000-batch watch would never emit in 400 batches; sequence 64 changed model/work despite contrary record; streaming stats lost timestamps | delegated audit plus retained cache-prime-only `/tmp/wb001-r3-ee4d41d0afe403f50031ffa53d907ff13cf5ba91` |
| 3 | repair | not exposed by runtime | not exposed by runtime | failed prelaunch audit | Repair protocol without weakening performance, exactness, or resource gates | repaired; re-audit failed | Run 300 steps/1,200 backward batches with 30-step warm-up; decode and structurally validate local watch histograms; timestamp container samples and gate temporal gaps; correct cross-attempt claims | inspector/verifier regression tests and retained prior evidence |
| 3 | review | not exposed by runtime | not exposed by runtime | uncommitted repaired protocol snapshot | Read-only prelaunch protocol re-audit; not the mandatory heavy review | FAIL before launch | Prior three findings closed; verifier incorrectly expected validations at steps 100/200/300 although this profile validates only at epoch end; planned step budget was stale | delegated re-audit; trainer/profile inspection; 15 focused tests |
| 4 | repair | not exposed by runtime | not exposed by runtime | failed prelaunch re-audit | Bind cadence gate to the actual unchanged profile and correct resource budget | repaired; final prelaunch audit pending | Require the sole epoch-end validation at `MAX_STEPS=300`; record 2,700 measured optimizer steps | focused tests, lint, format, shell syntax, diff check |
| 4 | review | not exposed by runtime | not exposed by runtime | uncommitted protocol snapshot | Protocol-focused prelaunch audit; not the mandatory heavy review | PASS, later superseded | Shell, verifier, inspector, cadence, budget, and focused tests appeared consistent | delegated protocol audit; real W&B 0.25.1 watch smoke |
| 4 | review | not exposed by runtime | not exposed by runtime | same uncommitted protocol snapshot | Independent implementation/data-horizon audit | FAIL; supersedes preceding PASS | Profile retained an 8,192-token cap, yielding only 16 updates and no watch emission; GPU/host coverage included pre-start samples; short evidence crashed the verifier | real composed streaming-loader count plus sampler/verifier inspection |
| 5 | repair | not exposed by runtime | not exposed by runtime | failed data-horizon audit | Make the exact loader reach every declared update and time-bound all resource samplers | repaired; final prelaunch review pending | Set stream cap 153,728 for exactly 1,200 microbatches/153,600 targets; filter GPU/host/Docker to start/end with gap gates; persist structured FAIL on incomplete evidence | exact containerized loader composition; focused tests |
| 5 | review | not exposed by runtime | not exposed by runtime | uncommitted horizon/sampler repair | Independent final prelaunch audits | PASS, later superseded | Both audits confirmed exact 1,200/300/153,600 loader work, time-windowed samplers, structured verifier failure, and focused checks | `/root/wb001_implementation` and `/root/wb_dgx_protocol_harness` |
| 5 | validation | not exposed by runtime | not exposed by runtime | exact redirected Docker streaming command | Exercise actual host sampler output before commit | FAIL before launch | Streaming CLI emitted ANSI screen-refresh rows and burst duplicates, so apparent 1 Hz rows were not independent samples | retained console reproduction; no measured arm launched |
| 6 | repair | not exposed by runtime | not exposed by runtime | failed host sampler validation | Use an honest, parseable container sampler cadence | repaired; final prelaunch review pending | Restored timestamped `docker stats --no-stream` polling at observed ~0.5 Hz; require 90% count, 3.5 s endpoints, and 4.5 s maximum gap | 7-second live Docker smoke produced three clean eight-field samples |
| 6 | review | not exposed by runtime | not exposed by runtime | uncommitted cycle-6 sampler repair | Final prelaunch audit before measurement | PASS | No blocker in polling lifecycle, clean eight-field rows, parser, or 2 s/90%/3.5 s/4.5 s gates | delegated protocol audit; shell, Ruff, diff, and focused tests pass |
| 7 | review | pending | pending | exact integration/evidence head pending | Mandatory heavy review against philosophy, ticket, and applicable CHECK after evidence | pending | pending | pending |

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
  `docs/model-runs/evidence/WB-001-repair-provenance.json`,
  `docs/model-runs/evidence/WB-001-r2-harness-provenance.json`,
  `docs/model-runs/evidence/WB-001-r2-repair-provenance.json`,
  `docs/model-runs/evidence/WB-001-r3-protocol-review-provenance.json`,
  `docs/model-runs/evidence/WB-001-r4-protocol-reviews-provenance.json`,
  `docs/model-runs/evidence/WB-001-r5-horizon-review-provenance.json`, and
  `docs/model-runs/evidence/WB-001-r6-container-sampler-provenance.json`
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

### DGX evidence repair

- Repair cycle: 2, following measured Attempt 2 `FAIL` at `049acf7`.
- Repair model / mode: not exposed by runtime / not exposed by runtime.
- Input handoff: durable failed projection plus raw `/tmp` evidence for all nine
  arms.
- Changes made: sequence length 8 → 64 for the comparison only; watch interval
  100 → official/default 1,000 batches; streaming Docker stats; stable hardware
  projection no longer includes randomized container hostname.
- What was deliberately not changed: seed, data order, arm order, artifact
  policy, decision thresholds, exactness gates, or failed evidence. Sequence 64
  did change the positional-encoding buffer and per-step work relative to
  Attempt 2; the earlier contrary statement was rejected by review.
- Local evidence: every Attempt 2 process exited 0 and exact config/trajectory,
  model, resume, and cursor gates passed; the run remained `FAIL` on the declared
  measurement gates.
- Commit reviewed next: `ee4d41d`, which failed prelaunch protocol audit.
- Re-review verdict: pending.

### DGX protocol repair after aborted Attempt 3

- Repair cycle: 3, following prelaunch audit `FAIL` at `ee4d41d`.
- Review/repair model and mode: not exposed by runtime / not exposed by runtime.
- Attempt disposition: interrupted after cache prime and before all measured
  arms; its partial root is retained and makes no performance claim.
- Changes made: 300 optimizer steps and 30 warm-up steps yield 1,200 backward
  batches; a network-isolated W&B 0.25.1 inspector copies each local record,
  validates datastore CRC/protobuf structure plus complete finite histogram
  values/bins, and requires at least one watch-on emission; Docker stats rows
  carry nanosecond timestamps with start/end/max-gap gates.
- Comparison scope: all arms within Attempt 4 use the same sequence-64 model and
  work. No throughput result is compared across Attempt 2 and Attempt 4.
- What was deliberately not changed: production watch default, seed, data
  order, Latin-square order, artifact policy, exactness gates, 5% investigation
  threshold, 10% failure threshold, memory/swap gates, or failed evidence.
- Prelaunch re-audit: `FAIL`; all three original findings were closed, but the
  verifier expected validations at 100/200/300 while the unchanged profile has
  no step cadence and therefore validates once at epoch end. The budget also
  still said 900 steps.
- Repair cycle 4: require validation only at `MAX_STEPS=300` and record the
  actual 2,700-step matrix budget. No runtime configuration or decision gate
  changed.
- A protocol-focused re-review returned `PASS`, but an independent
  implementation/data audit superseded it with `FAIL`: the profile's unchanged
  8,192-token stream ended after 63 microbatches/16 updates, and GPU/host sample
  counts included pre-start observations. Short evidence would also raise
  instead of persisting a structured verifier failure.
- Repair cycle 5: set `data.streaming.train.max_tokens=153728`. Exact
  network-isolated composition of the real loader produced 1,200 microbatches,
  300 full accumulation groups, and 153,600 targets. Filter GPU, vmstat, and
  Docker samples to the exact training window with sampler-specific endpoint
  and maximum-gap gates. Catch evidence-verification exceptions only to persist
  a structured `FAIL` summary.
- Two delegated re-audits returned `PASS`, but a direct host smoke then showed
  that redirected streaming `docker stats` emits ANSI screen-refresh rows and
  burst duplicates. Those rows cannot prove independent 1 Hz samples.
- Repair cycle 6: use timestamped `docker stats --no-stream` polling at its
  observed approximately 0.5 Hz cadence. The verifier now expects a 2 s
  interval, at least 90% count coverage, endpoint gaps at most 3.5 s, and no
  inter-sample gap above 4.5 s. A 7-second live smoke produced three clean
  eight-field rows; GPU remains 5 Hz and vmstat 1 Hz.
- Commit reviewed next: pending clean Attempt 4 head after final prelaunch audit.

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
- Performance/resource result: DGX Attempt 2 retained a `FAIL` at exact commit
  `049acf7`: all nine processes and exact trajectory/checkpoint gates passed,
  but data wait reached 12.68%, container coverage was about 48%, and watch-on
  regressed 26.26% versus offline/watch-off at the unsafe 100-batch interval.
  Attempt 3 was aborted before measured arms after its protocol audit failed.
  Audited Attempt 4 is predeclared before retry; no positive throughput claim
  is made from Attempts 2 or 3.
- Failed attempts retained: first offline smoke placed its W&B directory under
  the repository; it was removed and `WANDB_DIR` now points at the smoke temp
  root. No training failure occurred.
- Known trade-offs: online artifact confirmation waits outside optimizer work at
  explicit retention boundaries; fresh quota visibility requires an operator
  Billing UI/CSV snapshot because the public Python API does not expose current
  storage usage.
- Unresolved risks: real online auth/service behavior remains unexercised and
  fail-closed; DGX overhead remains unresolved until adaptive Attempt 4 passes.
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
