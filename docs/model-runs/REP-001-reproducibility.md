# REP-001 — Reproducible run identity and global seeding

- PR: [#22](https://github.com/Ayumu-J-S/llm_scratch/pull/22) (merged); repair [#24](https://github.com/Ayumu-J-S/llm_scratch/pull/24) (draft)
- Branch: `codex/rep-001-dirty-verify-fix` (repair)
- Audit: [#23](https://github.com/Ayumu-J-S/llm_scratch/pull/23) (draft)
- Ticket: REP-001
- Hypothesis: Capturing immutable code/config/input identity and deriving every
  RNG stream from Hydra makes bounded CPU fixture runs reproducible without W&B.
- Experiment record: `N/A — this ticket validates a fixture and run metadata,
  not a consequential model-quality experiment`
- Started: 2026-07-12
- Final verdict: PASS WITH NOTE (repair PR #24; final merge-gate audit pending)
- Final record owner: Codex implementation agent

## Scope and decision context

- Goal: identify a run from its run directory and prove same-seed CPU batches and losses.
- In scope: Python/NumPy/Torch/CUDA/DataLoader/model-init seeds; config/lock/Git/environment/tokenizer/data evidence; immutable-input and dirty-tree guards.
- Out of scope: bitwise determinism for every GPU kernel, raw dataset uploads, and W&B policy.
- Relevant principles: reproducibility and experiment records are first-class artifacts; no hidden fallback for real training.
- Baseline commit: `d5806b2119c03fe4ef8f14c2523748018bd3315c` (merged REP-001 baseline)
- Intended evidence: focused determinism/mutation tests, CPU smoke run manifest, full tests/lint/lock.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `a6c65cd` + REP-001/PHILOSOPHY/CHECK | Requested Luna / Extra High; implement scoped identity, immutable manifests, and global seeding | completed | Added `runtime.reproducibility`, Hydra seed config, DataLoader generators/worker seeding, run snapshots, dirty/mutable-input guards, and focused tests | `bb407cb`; initial focused tests, Ruff, diff check; CPU smoke run manifest |
| 1 | review | not exposed by runtime | not exposed by runtime | `bb407cb` | Requested heavier review at Extra Thinking; inspect REP acceptance and CHECK sections 1, 2, 6 | PASS WITH NOTE | Same-seed CPU batches/losses, run manifest, mutation and mutable-input guards pass; found guard ordering and deterministic toggle follow-ups | PR #22 review `4679671936`; broader full suite 198 passed, 1 skipped; Ruff, lock, diff pass |
| 2 | repair | not exposed by runtime | not exposed by runtime | review `4679671936` | Move dirty/mutable guard before tokenizer/data/model initialization; add regression; retain exact model provenance | completed | Run manifest now writes immediately after config resolution/global seeding, before tokenizer/data setup; dirty real-run regression proves tokenizer is untouched; lock/Git verification and deterministic=False reset added; external resolved configs copied into run directory | `2745b67`, then `deb0c1f`; REP-focused 6 passed |
| 2 | re-review | not exposed by runtime | not exposed by runtime | `deb0c1f` | Re-run independent review against exact repair head | PASS WITH NOTE | Acceptance remains satisfied; no blocking ML/system issue; only note is that GPU bitwise determinism remains out of scope and validation is CPU/R1 | review `4679685771`; exact head `deb0c1f`; full 200 passed, 1 skipped; focused integration 20 passed plus REP-focused 6; Ruff, lock, diff pass |
| 3 | independent review | not exposed by runtime | not exposed by runtime | merged PR #22 head `e2153a2` / merge `d5806b2` | Re-check `verify_run_manifest(..., root_dir=...)` against the captured Git worktree state | FAIL | Verification compared only `HEAD`; a clean captured manifest could pass in a dirty checkout with the same commit, so source state was not fully bound to the run record | PR #22 thread `PRRT_kwDORqx5mc6QLvLd`, P2 at `src/runtime/reproducibility.py:305` |
| 3 | repair | not exposed by runtime | not exposed by runtime | `d5806b2` + review thread `PRRT_kwDORqx5mc6QLvLd` | Compare recorded Git dirty/status fields during source verification; add clean-pass and clean-to-dirty regression tests | completed | `verify_run_manifest` now rejects dirty-state or status drift when `root_dir` is supplied; focused tests cover a matching clean worktree and dirty verification failure | repair PR #24 exact head `8f3307a`; 8 focused, 202 full/1 skipped; Ruff, lock, diff pass |
| 3 | re-review | not exposed by runtime | not exposed by runtime | `8f3307a` / PR #24 | Independent exact-head re-review of the dirty/status verification repair | PASS WITH NOTE | Clean verification succeeds; clean-to-dirty verification fails; no new ML claim; CPU/R1 only | review `4679720242`; exact head `8f3307a`; 8 focused, 202 full/1 skipped |
| 4 | merge audit / record finalization | not exposed by runtime | not exposed by runtime | PR #24 merge `792cb89`; audit PR #23 | Record the repair merge, resolved PR22 thread, exact reviewed heads, and final GitHub evidence without changing implementation | completed | PR #24 is closed and merged; PR22 P2 thread is resolved; `8f3307a` remains the independently reviewed repair head and `7553b34` is its docs descendant | merge `792cb895e04760dea4f4b23c36a934a5ab6589f9`; review IDs `4679720242`, `4679723227`; resolved thread `PRRT_kwDORqx5mc6QLvLd` |

## Runtime provenance block

| Namespace | Product | Displayed family | Exact model identifier | Reasoning mode | Source / unavailable reason |
| --- | --- | --- | --- | --- | --- |
| requested | Codex | Luna | Luna | Extra High | user/task request |
| actual | Codex | GPT-5 | not exposed by runtime | not exposed by runtime | active runtime does not display exact deployment ID or reasoning mode |

- Capture file/evidence: active Codex context; no exact runtime capture surface was exposed.
- Codex CLI version: not exposed by runtime
- Branch/commit: `codex/rep-001-dirty-verify-fix` / `8f3307a8b6d390c09405029efefd7a67adfad8e9`
- Phase/role/task path: repair / REP-001 / delegated agent
- Privacy confirmation: no prompts, hidden chain-of-thought, token counts, secrets, or raw thread ID recorded.

## Check selection and verdicts

### Review cycles 1–3

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: `8f3307a8b6d390c09405029efefd7a67adfad8e9` (repair re-review; cycle 1 was `67d9247`, cycle 2 was `deb0c1f`)
- Selected CHECK.md sections: 1 minimum review; 2 R1 smoke; 6 training loop/numerical health; 7.1 change surface; 7.3 repository policy; 8.1 experiment handoff and run-record integrity
- Major sections marked N/A and why: CHECK 4/5/7.2/7.4/8.2-8.4/9 — no data throughput, CUDA performance, optimizer, checkpoint, W&B, dependency-replacement, or next-change thought-experiment claim; CHECK 8.1 is applicable because this ticket creates the run record.
- Ticket acceptance result: PASS — same-seed CPU batches/losses reproduce and run directory captures identity/input evidence.
- Philosophy alignment: PASS — no mutable real input or silent CPU fallback; exact unavailable values are not inferred.
- Complexity / change-surface result: PASS WITH NOTE — localized runtime/config/train/data-loader changes; no compatibility shim.
- ML-system result: PASS WITH NOTE — CPU/R1 evidence only; no GPU bitwise determinism claim.
- Experiment-handoff result (CHECK 8.1): PASS — `run_manifest.json` records experiment ID, Git SHA/dirty/status state, resolved config and `uv.lock` hashes, hardware/software identity, tokenizer snapshot/hash, and data manifest snapshot/hash; `verify_run_manifest` detects captured-file mutation and verifies source lock, Git SHA, and worktree state when `root_dir` is supplied.
- Verdict: PASS WITH NOTE (cycle 1 at `4679671936`; cycle 2 at `4679685771`; repair re-review at `4679720242`)

Cycle 3 failed review (PR #22 thread `PRRT_kwDORqx5mc6QLvLd`) because
`verify_run_manifest(..., root_dir=...)` checked the recorded commit SHA but not
the captured dirty/status state. A same-HEAD checkout with uncommitted edits
could therefore pass verification. Repair PR #24 compares the recorded and
current Git dirty/status fields and adds focused clean-pass and dirty-failure
regressions; its exact-head re-review returned PASS WITH NOTE.

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| note | determinism scope | GPU bitwise determinism is intentionally not claimed; deterministic CPU fixture proof is exact | CHECK 2/R1, REP tests | retain as ticket out-of-scope note |

## Failed-review handoff

Cycle 1 review returned PASS WITH NOTE. The repair moved dirty/mutable guards
before tokenizer/data initialization and made the deterministic toggle explicit.
Cycle 2 re-review at `deb0c1f` returned PASS WITH NOTE.
Cycle 3 review found the dirty-worktree verification gap; repair PR #24 and its
exact-head re-review (`4679720242`) closed that finding with PASS WITH NOTE.

## Final evidence

- Resolved Hydra command/config: `uv run python src/train.py profile=smoke_overfit wandb.enabled=false training.epochs=1`; after repair the run manifest is written before tokenizer/data/model initialization.
- Data/tokenizer/model identity: canonical tokenizer fingerprint `12ccbc02...`; memorization manifest fingerprint `00c3797a...`; config and lock SHA-256 recorded in `run_manifest.json`.
- Validation and measurements: same-seed fixture reproduced initial batches and the complete loss sequence exactly; repair-focused reproducibility tests pass (`8 passed`); full suite passes (`202 passed, 1 skipped`); exact-head PR #24 re-review is PASS WITH NOTE.
- Performance/resource result: N/A — REP-001 is an identity/seeding ticket; no DGX throughput claim.
- Failed attempts retained at: N/A.
- Known trade-offs: deterministic Torch algorithms use `warn_only=True` so unsupported GPU kernels do not silently claim bitwise determinism.
- Unresolved risks: no blocking ticket risk; GPU bitwise determinism remains intentionally out of scope.
- Human decision requested: review and merge PR #24 only after an independent PASS/PASS WITH NOTE verdict and exact-head audit; PR #22 is already merged.

## Merge authority and final audit

- Merge path: human merge / guarded agent self-merge only after explicit authorization
- Human authorization: bounded roadmap self-merge authorization from the user on 2026-07-12
- Authorization evidence location: user goal instruction, PR #22 merge audit, and this follow-up record
- Authorization covers this named PR or bounded ticket/goal series: yes — the user's 2026-07-12 instruction authorizes self-merge for the bounded roadmap completion series; recorded in the parent goal/PR audit and this follow-up
- Exact independently reviewed implementation head SHA: `deb0c1f5cf71a1966804b0269b2f51e77c784bb1`
- Record-only docs head SHA: `e2153a2c7d50f72e2805eaebf19876d5a7691dd4`
- Latest independent verdict / model / mode: PASS WITH NOTE / not exposed by runtime / not exposed by runtime (docs head is record-only; no implementation drift)
- All actionable findings repaired and independently re-reviewed: yes — review IDs `4679685771`, `4679697456`, and `4679696518` record the exact-head and docs-scope passes
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: none observed; listed reviews are comments with PASS WITH NOTE findings only
- Newer human objections since authorization/review: none observed
- Human review dismissed by an agent: no
- Unresolved review threads at final audit: none — `github_list_pull_request_review_threads` returned an empty list for PR #22
- Branch-protection required-context inventory: not exposed by the connected GitHub surface; no required-context inventory was returned
- Applicable configured workflow/check inventory: none returned for `deb0c1f`, `e2153a2`, or merge `d5806b2`; `github_fetch_commit_workflow_runs` returned an empty list for each
- Observed exact-head check statuses: none returned for `deb0c1f`, `e2153a2`, or merge `d5806b2`; `github_get_commit_combined_status` returned an empty status list for each
- Expected checks absent, pending, skipped, cancelled, or non-successful: none observed; no expected contexts were exposed by the available inventory
- No-check evidence when both inventories are empty: yes, with the connector evidence limitation above recorded
- Target branch and base SHA at final audit: `main` / `a6c65cd0f535abc8d83686c83a671ea92054656f` at PR #22 creation; merge commit `d5806b2119c03fe4ef8f14c2523748018bd3315c`
- Up-to-date, conflict-free, and mergeable evidence: PR #22 was refreshed and merged by GitHub at `d5806b2`; the final merged state is closed/merged and no implementation drift exists in the docs audit
- Record, ledger, PR trail, validation, and risks parity: complete — PR body, exact reviewed heads, review IDs, empty checks/workflows/threads, and this record agree
- Prohibited self-merge categories: clear — no secrets/security/deployment/permission changes
- Admin/bypass/force/disabled-check requirement: no
- Final audit PR body/comment location: PR #22 merge body and this follow-up audit PR
- Final audit changed reviewed head: no
- Immediate pre-merge re-fetch/compare observation location: PR #22 final merge-gate audit in its PR body; merged head `e2153a2c7d50f72e2805eaebf19876d5a7691dd4`
- Immediate refresh compared authorization, head, base, review decision/objections, threads, expected checks/statuses, and mergeability: yes; no drift was found before GitHub created merge commit `d5806b2`
- Drift found: none
- Merge outcome: PR #22 closed/merged; merge commit `d5806b2119c03fe4ef8f14c2523748018bd3315c`

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | implementation/review | Kept scope localized; independently reproduced CPU fixture determinism and manifest guards | Exact runtime deployment and reasoning are unavailable; no GPU bitwise claim | REP-001, CFG-001 config shape, existing manifest/tokenizer APIs, CHECK R1 | PASS WITH NOTE; exact reviewed head `deb0c1f` |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated per-model attempt, pass, repair, and review counts: implementation, one repair, and two independent review cycles recorded.
- [x] Confirmed that the PR execution trail matches this record.
- [x] Recorded complete guarded self-merge authority/audit evidence for PR #22; the user authorized the bounded roadmap series and the exact-head merge evidence is retained above.
- [x] Confirmed that this bootstrap policy rule was not used before a human merged it.
