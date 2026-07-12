# REP-001 — Reproducible run identity and global seeding

- PR: [#22](https://github.com/Ayumu-J-S/llm_scratch/pull/22) (draft)
- Branch: `codex/rep-001-reproducibility`
- Ticket: REP-001
- Hypothesis: Capturing immutable code/config/input identity and deriving every
  RNG stream from Hydra makes bounded CPU fixture runs reproducible without W&B.
- Experiment record: `N/A — this ticket validates a fixture and run metadata,
  not a consequential model-quality experiment`
- Started: 2026-07-12
- Final verdict: PASS WITH NOTE
- Final record owner: Codex implementation agent

## Scope and decision context

- Goal: identify a run from its run directory and prove same-seed CPU batches and losses.
- In scope: Python/NumPy/Torch/CUDA/DataLoader/model-init seeds; config/lock/Git/environment/tokenizer/data evidence; immutable-input and dirty-tree guards.
- Out of scope: bitwise determinism for every GPU kernel, raw dataset uploads, and W&B policy.
- Relevant principles: reproducibility and experiment records are first-class artifacts; no hidden fallback for real training.
- Baseline commit: `a6c65cd0f535abc8d83686c83a671ea92054656f`
- Intended evidence: focused determinism/mutation tests, CPU smoke run manifest, full tests/lint/lock.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `a6c65cd` + REP-001/PHILOSOPHY/CHECK | Requested Luna / Extra High; implement scoped identity, immutable manifests, and global seeding | completed | Added `runtime.reproducibility`, Hydra seed config, DataLoader generators/worker seeding, run snapshots, dirty/mutable-input guards, and focused tests | `bb407cb`; initial focused tests, Ruff, diff check; CPU smoke run manifest |
| 1 | review | not exposed by runtime | not exposed by runtime | `bb407cb` | Requested heavier review at Extra Thinking; inspect REP acceptance and CHECK sections 1, 2, 6 | PASS WITH NOTE | Same-seed CPU batches/losses, run manifest, mutation and mutable-input guards pass; found guard ordering and deterministic toggle follow-ups | PR #22 review `4679671936`; broader full suite 198 passed, 1 skipped; Ruff, lock, diff pass |
| 2 | repair | not exposed by runtime | not exposed by runtime | review `4679671936` | Move dirty/mutable guard before tokenizer/data/model initialization; add regression; retain exact model provenance | completed | Run manifest now writes immediately after config resolution/global seeding, before tokenizer/data setup; dirty real-run regression proves tokenizer is untouched; lock/Git verification and deterministic=False reset added; external resolved configs copied into run directory | `2745b67`, then `deb0c1f`; REP-focused 6 passed |
| 2 | re-review | not exposed by runtime | not exposed by runtime | `deb0c1f` | Re-run independent review against exact repair head | PASS WITH NOTE | Acceptance remains satisfied; no blocking ML/system issue; only note is that GPU bitwise determinism remains out of scope and validation is CPU/R1 | review `4679685771`; exact head `deb0c1f`; full 200 passed, 1 skipped; focused integration 20 passed plus REP-focused 6; Ruff, lock, diff pass |

## Runtime provenance block

| Namespace | Product | Displayed family | Exact model identifier | Reasoning mode | Source / unavailable reason |
| --- | --- | --- | --- | --- | --- |
| requested | Codex | Luna | Luna | Extra High | user/task request |
| actual | Codex | GPT-5 | not exposed by runtime | not exposed by runtime | active runtime does not display exact deployment ID or reasoning mode |

- Capture file/evidence: active Codex context; no exact runtime capture surface was exposed.
- Codex CLI version: not exposed by runtime
- Branch/commit: `codex/rep-001-reproducibility` / `deb0c1f5cf71a1966804b0269b2f51e77c784bb1`
- Phase/role/task path: implementation / REP-001 / delegated agent
- Privacy confirmation: no prompts, hidden chain-of-thought, token counts, secrets, or raw thread ID recorded.

## Check selection and verdicts

### Review cycles 1 and 2

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: `deb0c1f5cf71a1966804b0269b2f51e77c784bb1` (cycle 2; cycle 1 was `67d9247`)
- Selected CHECK.md sections: 1 minimum review; 2 R1 smoke; 6 experiment integrity/changeability
- Major sections marked N/A and why: CHECK 4/5/7/8/9 — no data throughput, CUDA performance, optimizer, checkpoint, or production-run change.
- Ticket acceptance result: PASS — same-seed CPU batches/losses reproduce and run directory captures identity/input evidence.
- Philosophy alignment: PASS — no mutable real input or silent CPU fallback; exact unavailable values are not inferred.
- Complexity / change-surface result: PASS WITH NOTE — localized runtime/config/train/data-loader changes; no compatibility shim.
- ML-system result: PASS WITH NOTE — CPU/R1 evidence only; no GPU bitwise determinism claim.
- Verdict: PASS WITH NOTE (cycle 1 at `4679671936`; cycle 2 exact-head re-review at `4679685771`)

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| note | determinism scope | GPU bitwise determinism is intentionally not claimed; deterministic CPU fixture proof is exact | CHECK 2/R1, REP tests | retain as ticket out-of-scope note |

## Failed-review handoff

Cycle 1 review returned PASS WITH NOTE. The repair moved dirty/mutable guards
before tokenizer/data initialization and made the deterministic toggle explicit.
Cycle 2 re-review at `deb0c1f` returned PASS WITH NOTE.

## Final evidence

- Resolved Hydra command/config: `uv run python src/train.py profile=smoke_overfit wandb.enabled=false training.epochs=1`; after repair the run manifest is written before tokenizer/data/model initialization.
- Data/tokenizer/model identity: canonical tokenizer fingerprint `12ccbc02...`; memorization manifest fingerprint `00c3797a...`; config and lock SHA-256 recorded in `run_manifest.json`.
- Validation and measurements: same-seed fixture reproduced initial batches and the complete loss sequence exactly; REP-focused tests `6 passed`; broader focused integration `20 passed`; full suite `200 passed, 1 skipped`; CPU smoke completed one epoch.
- Performance/resource result: N/A — REP-001 is an identity/seeding ticket; no DGX throughput claim.
- Failed attempts retained at: N/A.
- Known trade-offs: deterministic Torch algorithms use `warn_only=True` so unsupported GPU kernels do not silently claim bitwise determinism.
- Unresolved risks: no blocking ticket risk; GPU bitwise determinism remains intentionally out of scope.
- Human decision requested: review and merge only after independent verdict and exact-head audit.

## Merge authority and final audit

- Merge path: human merge / guarded agent self-merge only after explicit authorization
- Human authorization: bounded roadmap self-merge authorization from user, if parent agent applies it
- Authorization evidence location: PR body and parent model-run audit
- Authorization covers this named PR or bounded ticket/goal series: pending parent audit
- Exact independently reviewed head SHA: pending
- Latest independent verdict / model / mode: pending
- All actionable findings repaired and independently re-reviewed: pending
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: pending
- Newer human objections since authorization/review: none observed
- Human review dismissed by an agent: no
- Unresolved review threads at final audit: pending
- Branch-protection required-context inventory: pending
- Applicable configured workflow/check inventory: pending
- Observed exact-head check statuses: pending
- Expected checks absent, pending, skipped, cancelled, or non-successful: pending
- No-check evidence when both inventories are empty: pending
- Target branch and base SHA at final audit: `main` / `a6c65cd` at PR creation
- Up-to-date, conflict-free, and mergeable evidence: PR created from current main; final refresh pending
- Record, ledger, PR trail, validation, and risks parity: pending final review
- Prohibited self-merge categories: clear — no secrets/security/deployment/permission changes
- Admin/bypass/force/disabled-check requirement: no
- Final audit PR body/comment location: PR #22
- Final audit changed reviewed head: no
- Immediate pre-merge re-fetch/compare observation location: pending
- Immediate refresh compared authorization, head, base, review decision/objections, threads, expected checks/statuses, and mergeability: pending
- Drift found: pending
- Merge outcome: pending

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | implementation/review | Kept scope localized; independently reproduced CPU fixture determinism and manifest guards | Exact runtime deployment and reasoning are unavailable; no GPU bitwise claim | REP-001, CFG-001 config shape, existing manifest/tokenizer APIs, CHECK R1 | PASS WITH NOTE; exact reviewed head `deb0c1f` |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated per-model attempt, pass, repair, and review counts: implementation, one repair, and two independent review cycles recorded.
- [x] Confirmed that the PR execution trail matches this record.
- [ ] Recorded human merge or complete guarded self-merge authority/audit evidence.
- [x] Confirmed that this bootstrap policy rule was not used before a human merged it.
