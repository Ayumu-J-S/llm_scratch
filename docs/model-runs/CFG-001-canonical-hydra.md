# CFG-001 - Canonical Hydra profiles and commands

- PR: [#20](https://github.com/Ayumu-J-S/llm_scratch/pull/20) (draft)
- Branch: `codex/cfg-001-canonical-hydra`
- Ticket: CFG-001
- Hypothesis: Explicit Hydra profiles, one preflight boundary, and one documented command path will make smoke and real-stream runs composable and reject unsafe data configurations before tokenization or training.
- Experiment record: `N/A` — this ticket validates configuration and fixture wiring; it does not claim a model-quality experiment.
- Started: 2026-07-12
- Final verdict: PASS WITH NOTE
- Final record owner: CFG-001 implementation agent

## Scope and decision context

- Goal: Replace ambiguous training commands with explicit `smoke_overfit`, `pretrain_streaming`, and future `evaluation` profiles.
- In scope: Hydra profile composition, static critical-key preflight, real stream batch preflight, resolved-config snapshots, importable/console commands, README/Makefile agreement, and focused tests.
- Out of scope: benchmark implementation, trainer redesign, evaluation execution, and compatibility shims.
- Relevant `PHILOSOPHY.md` principles: Hydra is the single runtime/training configuration system; executable canonical workflows; explicit errors and recovery paths; no hidden data or device fallback; small, direct boundaries.
- Baseline commit: `ed83c09634c9b0e11938ecdfba3a281274186e5d` (origin/main)
- Intended evidence: Hydra composition for each profile, preflight rejection tests, resolved-config files, CPU smoke, and manifest-backed train/validation batches.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `ed83c09`, CFG-001 and project policy | Requested Luna / Extra High implementation pass | complete | Added canonical profiles, strict preflight, snapshots, console wrappers, Make/README commands, and tests. | `f73cc2a`; composition and smoke commands below |
| 1 | handoff | not exposed by runtime | not exposed by runtime | `f73cc2a` / PR #20 | Prepare exact-head independent review | complete | Draft PR opened with model-run provenance and ledger. | PR #20 |
| 1 | review | not exposed by runtime | not exposed by runtime | `76b92c2` / PR #20 | Independent Extra Thinking review | FAIL | `evaluation.yaml` and README described evaluation as composition-only, but `train.py` accepted it and trained; profile name/mode/purpose could also be mixed by overrides. | reviewer handoff; exact-head review |
| 2 | repair | not exposed by runtime | not exposed by runtime | `76b92c2` / failed review | Reject evaluation and mismatched canonical profiles before tokenizer/data work | complete | `validate_training_config` now rejects evaluation purpose/task and enforces name↔mode↔purpose pairs; regression tests cover both guards. | `6e874ac`; focused suite |
| 2 | re-review | not exposed by runtime | not exposed by runtime | `52b89a9` | Independent re-review of evaluation guard and full CFG-001 | PASS WITH NOTE | Review ID `4679636006`: evaluation guard and canonical profile pairs pass; no DGX R2 claim is made for this configuration-wiring ticket. | exact reviewed head `52b89a95a31d7e929ca302c11de0326672b8679c` |

## Runtime provenance block

The active session visibly identifies the product/family as Codex / GPT-5. The
runtime did not expose a deployment identifier or reasoning-mode display, so
those actual fields remain unavailable. Requested values are recorded separately
and are not treated as actual runtime provenance.

| Namespace | Product | Displayed family | Exact model identifier | Reasoning mode | Source / unavailable reason |
| --- | --- | --- | --- | --- | --- |
| requested | Codex | Luna (requested) | not exposed by runtime | Extra High (requested) | user/agent implementation request |
| actual | Codex | GPT-5 | not exposed by runtime | not exposed by runtime | active runtime display does not expose exact deployment or reasoning mode |

- Capture file/evidence: runtime-visible session label; no hidden runtime metadata was available.
- Codex CLI version: not exposed by runtime
- Branch/commit: `codex/cfg-001-canonical-hydra` / implementation `6e874ac06f1936f2756590cb3626d1578371c1ec`, independently reviewed docs descendant `52b89a95a31d7e929ca302c11de0326672b8679c`
- Phase/role/task path: implementation / CFG-001 implementation sub-agent
- Privacy confirmation: no prompts, hidden chain-of-thought, token counts, secrets, or raw thread ID.

## Check selection and verdicts

### Review cycle 1

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: `52b89a95a31d7e929ca302c11de0326672b8679c` (normative repair `6e874ac`; docs-only descendant)
- Selected `CHECK.md` sections: R0 configuration/changeability and R1 smoke; no DGX R2 claim because this ticket changes profile wiring and command boundaries only.
- Major sections marked N/A and why: DGX performance, long-run stability, checkpoint/resume, and benchmark integrity are outside CFG-001.
- Ticket acceptance result: cycle 1 FAIL; evaluation fallback and profile mismatch repaired; cycle 2 PASS WITH NOTE
- Philosophy alignment: implementation uses Hydra profiles and direct validation; no `config.py`, compatibility aliases, or second training path.
- Complexity / change-surface result: PASS — one Hydra config system and one training path; evaluation remains composition-only.
- ML-system result: PASS WITH NOTE — R0/R1 profile and fixture evidence pass; no DGX R2 performance claim is made.
- Verdict: PASS WITH NOTE

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| info | scope | Profiles and preflight are isolated to CFG-001; evaluation remains composition-only. | `config/profile/*.yaml`, `src/runtime/config.py` | Independent reviewer to confirm no hidden execution fallback. |
| high | execution path | Evaluation profile was accepted by the training entrypoint despite documentation saying it was composition-only. | Cycle 1 independent review; `train.py` reached training branch for `profile=evaluation`. | Reject evaluation purpose/task during preflight and prove tokenizer/data are untouched. |
| high | execution path | Evaluation purpose/task is now rejected before tokenizer initialization. | `src/runtime/config.py`, `tests/test_config_profiles.py` | Closed by cycle 2 PASS WITH NOTE. |
| high | profile integrity | Profile name, mode, and purpose could be overridden into an unsafe mixed profile. | Cycle 1 independent review; no canonical pair invariant. | Enforce canonical pairs for `smoke_overfit` and `pretrain_streaming`, and reject unknown training profile names. |
| high | profile integrity | Canonical profile pairs are now enforced and unknown training profiles are rejected. | `src/runtime/config.py`, mismatch regression tests. | Closed by cycle 2 PASS WITH NOTE. |
| note | evidence scope | The ticket's real stream fixture and CPU smoke pass; no target-DGX performance claim is implied. | 194 passed, 1 skipped; profile preflight and `make smoke`. | Keep DGX R2/R3 evidence scoped to environment/performance tickets. |

## Failed-review handoff

- From review cycle: 1
- Failed check and why: CHECK R0/R1 execution-path consistency; evaluation was documented as composition-only but the training entrypoint accepted and executed it, and profile name/mode/purpose could be mixed through overrides.
- Review model / mode: not exposed by runtime / not exposed by runtime
- Implementation model / mode that produced the failed state: not exposed by runtime / not exposed by runtime
- Commit/diff to repair: `76b92c2` CFG-001 profile/preflight implementation
- Reproduction command or evidence: `uv run python src/train.py profile=evaluation` passed validation and proceeded into streaming training setup before repair.
- Relevant files/config/manifests: `config/profile/evaluation.yaml`, `README.md`, `src/train.py`, `src/runtime/config.py`
- Attempts already made: one implementation pass; one independent review returned FAIL.
- Invariants and constraints: evaluation remains composition-only; training must reject it before tokenizer/data initialization; no fallback to another profile.
- Selected next model / mode: not exposed by runtime / not exposed by runtime
- Why this model was selected: independent re-review of the exact repair head is required by AGENTS.md.
- Exact repair request: reject evaluation purpose/task in preflight, enforce canonical profile name↔mode↔purpose pairs, reject unknown training profile names, and add regressions proving the wrapper does not touch tokenizer/data.
- Completion evidence requested: focused test, full suite, lint/lock checks, and independent PASS or justified PASS WITH NOTE.

## Repair result

- Repair cycle: 2
- Repair model / mode: not exposed by runtime / not exposed by runtime
- Input handoff: cycle 1 FAIL above
- Changes made: `validate_training_config` rejects `purpose=evaluation` or `task=evaluate_checkpoint`, enforces canonical profile pairs, rejects unknown profile names; tests invoke `main.__wrapped__` and cover evaluation and mismatch guards before tokenizer initialization.
- What was deliberately not changed: evaluation composition file remains available for future evaluation work; no benchmark/trainer implementation added.
- Local evidence: focused CFG-001 suites 8 + 6 passed; full suite 194 passed, 1 skipped; Ruff, lock, diff, profile preflight, console preflight, and `make smoke` passed.
- Commit reviewed next: `52b89a95a31d7e929ca302c11de0326672b8679c`
- Re-review model / mode: not exposed by runtime / not exposed by runtime
- Re-review verdict: PASS WITH NOTE (review ID `4679636006`; exact head `52b89a95a31d7e929ca302c11de0326672b8679c`)

## Final evidence

- Resolved Hydra command/config: `uv run python scripts/config_check.py profile=pretrain_streaming`; snapshot `runs/pretrain_streaming/<timestamp>/resolved_config.yaml`.
- Data/tokenizer/model identity: stream profile uses bilingual immutable manifest fingerprint `47cca88c...4ffc19`; tokenizer remains canonical pinned config; no model-quality claim.
- Validation and measurements:
  - `uv run pytest -q`: 194 passed, 1 skipped on the repair head (focused repair suite 14 passed).
  - `uv run ruff check .`: passed.
  - `uv lock --check`: passed.
  - `uv run python scripts/config_check.py profile=pretrain_streaming`: passed and built train/validation batches from distinct manifest selections.
  - `uv run python scripts/config_check.py profile=smoke_overfit`: passed.
  - `uv run python src/train.py profile=smoke_overfit runtime.device=cpu training.epochs=1 training.batch_size=2 wandb.enabled=false`: passed; resolved config was written before tokenizer/data/model work.
  - `uv run llm-scratch-config-check profile=pretrain_streaming`: passed after editable install.
  - `make smoke`: passed with finite train/validation loss and resolved snapshot.
- Failed attempts retained at: cycle 1 independent review and repair handoff above; failure is preserved rather than overwritten.
- Known trade-offs: fixture stream profile uses sequence length 8 so the committed tiny fixture yields complete train and validation batches; production sequence length remains a deliberate later profile decision.
- Unresolved risks: future evaluation command is composition-only and now rejected by training; merge-gate branch-protection/check inventory still requires final refresh before merge.
- Human decision requested: review the draft PR and model-run trail; merge only after all guarded gates pass.

## Merge authority and final audit

- Merge path: guarded agent self-merge only after independent PASS WITH NOTE and exact-head gate audit; review is complete, final merge-gate audit remains.
- Human authorization: “これからはとりあえず全部セルフマージしていいよ” / user explicitly authorized self-merge for the bounded roadmap goal on 2026-07-12.
- Authorization evidence location: parent-session user instruction; final PR audit comment to be added by root before merge.
- Authorization covers this named PR or bounded ticket/goal series: bounded roadmap goal, yes.
- Exact independently reviewed head SHA: `52b89a95a31d7e929ca302c11de0326672b8679c`
- Latest independent verdict / model / mode: PASS WITH NOTE / not exposed by runtime / not exposed by runtime
- All actionable findings repaired and independently re-reviewed: yes
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: none in cycle 2; final refresh required
- Newer human objections since authorization/review: none observed
- Human review dismissed by an agent: no
- Unresolved review threads at final audit: zero observed in cycle 2; root must refresh immediately before merge
- Branch-protection required-context inventory: connector inventory unavailable in this review context; root must refresh
- Applicable configured workflow/check inventory: no workflow runs returned at review time; root must refresh
- Observed exact-head check statuses: no combined statuses returned for reviewed head; evidence limitation, root must refresh
- Expected checks absent, pending, skipped, cancelled, or non-successful: not established; root must inventory before merge
- No-check evidence when both inventories are empty: not established; root must inventory before merge
- Target branch and base SHA at final audit: `main` / `ed83c09634c9b0e11938ecdfba3a281274186e5d` at review; refresh required before merge
- Up-to-date, conflict-free, and mergeable evidence: pending final refresh
- Record, ledger, PR trail, validation, and risks parity: pending finalization
- Prohibited self-merge categories: clear — no secrets, security controls, publication, paid resources, destructive actions, release, deployment, account/permission change, or legal/licensing decision.
- Admin/bypass/force/disabled-check requirement: no
- Final audit PR body/comment location: pending
- Final audit changed reviewed head: no
- Immediate pre-merge re-fetch/compare observation location: root final audit comment
- Immediate refresh compared authorization, head, base, review decision/objections, threads, expected checks/statuses, and mergeability: required before merge
- Drift found: none in cycle 2; root must abort if refresh differs
- Merge outcome: pending

## Model assessment from this ticket

Record observable outcomes, not hidden chain-of-thought.

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| Codex / GPT-5; exact ID and mode not exposed | implementation (requested Luna / Extra High) | Kept profile-specific values in Hydra, added explicit preflight and run snapshots, made console/Make/README paths agree, and repaired evaluation fallback/profile mismatch after review. | First review missed the evaluation execution contradiction and name/mode/purpose coupling; independent re-review caught no further actionable defect. | CFG-001 acceptance, existing stream loader, PHILOSOPHY, CHECK R0/R1, failed-review handoff. | PASS WITH NOTE |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated per-model attempt/pass/review counts (cycle 1 FAIL and cycle 2 PASS WITH NOTE recorded).
- [x] Confirmed that the PR execution trail matches this record.
- [ ] Recorded human merge or complete guarded self-merge authority/audit evidence.
- [x] Confirmed that this bootstrap policy rule was not used before a human merged it.
