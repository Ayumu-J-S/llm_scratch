# CFG-001 - Canonical Hydra profiles and commands

- PR: [#20](https://github.com/Ayumu-J-S/llm_scratch/pull/20) (draft)
- Branch: `codex/cfg-001-canonical-hydra`
- Ticket: CFG-001
- Hypothesis: Explicit Hydra profiles, one preflight boundary, and one documented command path will make smoke and real-stream runs composable and reject unsafe data configurations before tokenization or training.
- Experiment record: `N/A` — this ticket validates configuration and fixture wiring; it does not claim a model-quality experiment.
- Started: 2026-07-12
- Final verdict: in progress
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
| 1 | implementation | not exposed by runtime | not exposed by runtime | `ed83c09`, CFG-001 and project policy | Requested Luna / Extra High implementation pass | in progress | Added canonical profiles, strict preflight, snapshots, console wrappers, Make/README commands, and tests. | `f73cc2a`; composition and smoke commands below |
| 1 | handoff | not exposed by runtime | not exposed by runtime | `f73cc2a` / PR #20 | Prepare exact-head independent review | pending | Draft PR opened; heavy review must inspect PHILOSOPHY, CFG-001 acceptance, and CHECK R0/R1. | PR #20 |

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
- Branch/commit: `codex/cfg-001-canonical-hydra` / `f73cc2a691fc0b8567012690807339d82916cbf4`
- Phase/role/task path: implementation / CFG-001 implementation sub-agent
- Privacy confirmation: no prompts, hidden chain-of-thought, token counts, secrets, or raw thread ID.

## Check selection and verdicts

### Review cycle 1

- Review model / mode: pending independent review; runtime exact model/mode not exposed
- Commit reviewed: `f73cc2a691fc0b8567012690807339d82916cbf4`
- Selected `CHECK.md` sections: R0 configuration/changeability and R1 smoke; no DGX R2 claim because this ticket changes profile wiring and command boundaries only.
- Major sections marked N/A and why: DGX performance, long-run stability, checkpoint/resume, and benchmark integrity are outside CFG-001.
- Ticket acceptance result: pending independent review
- Philosophy alignment: implementation uses Hydra profiles and direct validation; no `config.py`, compatibility aliases, or second training path.
- Complexity / change-surface result: pending independent review
- ML-system result: pending independent review
- Verdict: pending

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| info | scope | Profiles and preflight are isolated to CFG-001; evaluation remains composition-only. | `config/profile/*.yaml`, `src/runtime/config.py` | Independent reviewer to confirm no hidden execution fallback. |

## Failed-review handoff

N/A — no independent review has returned `FAIL` yet.

## Repair result

N/A — no repair cycle yet.

## Final evidence

- Resolved Hydra command/config: `uv run python scripts/config_check.py profile=pretrain_streaming`; snapshot `runs/pretrain_streaming/<timestamp>/resolved_config.yaml`.
- Data/tokenizer/model identity: stream profile uses bilingual immutable manifest fingerprint `47cca88c...4ffc19`; tokenizer remains canonical pinned config; no model-quality claim.
- Validation and measurements:
  - `uv run pytest -q`: 190 passed, 1 skipped before final record-only changes (the one pre-existing direct YAML assertion was repaired by retaining canonical stream shape in `config/train.yaml`; focused rerun 6 passed).
  - `uv run ruff check .`: passed.
  - `uv lock --check`: passed.
  - `uv run python scripts/config_check.py profile=pretrain_streaming`: passed and built train/validation batches from distinct manifest selections.
  - `uv run python src/train.py profile=smoke_overfit runtime.device=cpu training.epochs=1 training.batch_size=2 wandb.enabled=false`: passed; resolved config was written before tokenizer/data/model work.
  - `uv run llm-scratch-config-check profile=pretrain_streaming`: passed after editable install.
- Failed attempts retained at: command output in the implementation handoff; no failed review cycle.
- Known trade-offs: fixture stream profile uses sequence length 8 so the committed tiny fixture yields complete train and validation batches; production sequence length remains a deliberate later profile decision.
- Unresolved risks: future evaluation command is composition-only; heavy independent review and exact branch-protection inventory remain pending.
- Human decision requested: review the draft PR and model-run trail; merge only after all guarded gates pass.

## Merge authority and final audit

- Merge path: guarded agent self-merge only after independent PASS/PASS WITH NOTE and exact-head gate audit; currently pending review.
- Human authorization: user explicitly authorized self-merge for the bounded roadmap goal on 2026-07-12; final audit must cite the exact instruction in the PR.
- Authorization evidence location: parent-session user instruction and final PR audit comment (pending).
- Authorization covers this named PR or bounded ticket/goal series: bounded roadmap goal, yes.
- Exact independently reviewed head SHA: pending
- Latest independent verdict / model / mode: pending
- All actionable findings repaired and independently re-reviewed: pending
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: pending inventory
- Newer human objections since authorization/review: none observed
- Human review dismissed by an agent: no
- Unresolved review threads at final audit: pending
- Branch-protection required-context inventory: pending
- Applicable configured workflow/check inventory: pending
- Observed exact-head check statuses: pending
- Expected checks absent, pending, skipped, cancelled, or non-successful: pending
- No-check evidence when both inventories are empty: pending
- Target branch and base SHA at final audit: `main` / `ed83c096...` at PR creation; refresh required before merge
- Up-to-date, conflict-free, and mergeable evidence: pending final refresh
- Record, ledger, PR trail, validation, and risks parity: pending finalization
- Prohibited self-merge categories: clear — no secrets, security controls, publication, paid resources, destructive actions, release, deployment, account/permission change, or legal/licensing decision.
- Admin/bypass/force/disabled-check requirement: no
- Final audit PR body/comment location: pending
- Final audit changed reviewed head: no
- Immediate pre-merge re-fetch/compare observation location: pending
- Immediate refresh compared authorization, head, base, review decision/objections, threads, expected checks/statuses, and mergeability: pending
- Drift found: pending
- Merge outcome: pending

## Model assessment from this ticket

Record observable outcomes, not hidden chain-of-thought.

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| Codex / GPT-5; exact ID and mode not exposed | implementation (requested Luna / Extra High) | Kept profile-specific values in Hydra, added explicit preflight and run snapshots, and made console/Make/README paths agree. | Independent review not yet performed. | CFG-001 acceptance, existing stream loader, PHILOSOPHY, CHECK R0/R1. | implementation complete; review pending |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated per-model attempt/pass/review counts (implementation attempt; final counts await review).
- [x] Confirmed that the PR execution trail matches this record.
- [ ] Recorded human merge or complete guarded self-merge authority/audit evidence.
- [x] Confirmed that this bootstrap policy rule was not used before a human merged it.
