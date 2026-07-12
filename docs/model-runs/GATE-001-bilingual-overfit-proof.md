# GATE-001 - Bilingual Overfit Proof

- PR: [#39](https://github.com/Ayumu-J-S/llm_scratch/pull/39) (draft)
- Branch: `codex/gate-001-bilingual-overfit-proof`
- Ticket: `GATE-001`
- Hypothesis: a bounded random-initialized run can memorize a fixed bilingual fixture, resume exactly, and generate checkpoint-backed base-model continuations.
- Experiment record: `docs/experiments/GATE-001-bilingual-overfit-proof.md`
- Started: 2026-07-12
- Final verdict: candidate complete; independent review pending
- Final record owner: implementation agent

## Scope and decision context

- Goal: demonstrate the entire fixed-fixture learning chain before any real-pretraining claim.
- In scope: versioned tiny Japanese/English fixture, one bounded canonical run, full-state resume, local evidence, and checkpoint-backed continuation samples.
- Out of scope: held-out validation/generalization claims, production data, benchmark scores, architecture experiments, and online W&B.
- Relevant `PHILOSOPHY.md` principles: random initialization; Japanese and English first; evaluation boundaries; one-machine, bounded evidence; reproducible and human-legible experiment records.
- Baseline commit/run: `origin/main` at `7f9c1728098f5e0dc18653b1660e07e5b36788ce`; no GATE-001 evidence exists.
- Intended evidence: predeclared loss/budget/stop conditions; two independent same-seed traces; split/resume suffix equality; verified checkpoint identities and model digests; labeled JP/EN base-model continuations; complete local records.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `7f9c172`; requested Luna/lightweight Extra High | Implement the bounded fixed-fixture memorization/resume/generation gate. | in progress | Actual product/family displayed as Codex / GPT-5; exact deployment ID and reasoning mode are not displayed. | This record and linked predeclared experiment record |
| 1 | repair | not exposed by runtime | not exposed by runtime | candidate through `8db0b6b`; requested lightweight Extra High | Retain failed bounded attempts, keep the original loss/budget gate, and predeclare the smallest fixture-only retry. | in progress | Host CPU-Torch failure, insufficient windows, terminal-step resume, and diluted English memorization were all retained. Retry uses two repeated JP/EN documents with 11 batches/pass so step 100 has a suffix; no threshold relaxation. | Experiment-record retry predeclaration dated 2026-07-12 |
| 2 | repair | not exposed by runtime | not exposed by runtime | attempt-6 evidence at candidate `66ec702`; requested lightweight Extra High | Preserve the successful loss/trajectory result but repair the failed full-suffix sampling audit without changing training. | in progress | Retry uses longer, fixed in-fixture prefixes to remove cross-language first-token ambiguity; model/data/optimizer/seed/budget/threshold remain unchanged. | Experiment-record sampling-audit retry predeclaration dated 2026-07-12 |
| 2 | implementation | not exposed by runtime | not exposed by runtime | exact run head `3d0f4fbdc7c8ad40d30b9e5eb03e448e712d2e2e`; requested Luna/lightweight Extra High | Run the new predeclared prompt selection through the complete reference/repeat/resume proof. | candidate complete | Attempt 7 PASS: all three 200-step traces hashed identically; verified step-100 resume and GEN-001 JP/EN samples passed. | `reports/gate-001/attempt-7/gate_record.json`; experiment record |
| 2 | handoff | not exposed by runtime | not exposed by runtime | candidate `3d0f4fbdc7c8ad40d30b9e5eb03e448e712d2e2e`; requested heavier reviewer Extra Thinking | Hand off exact implementation evidence for independent PHILOSOPHY/ticket/CHECK review. | ready for review | No hidden reasoning recorded; request is limited to ticket acceptance, selected checks 6/8/9.1/R2, and change surface. | provenance capture 2026-07-12T16:23:21Z |
| 3 | repair | not exposed by runtime | not exposed by runtime | independent pre-review of `5e33d91`; requested lightweight Extra High | Apply the sole actionable finding: repository formatter layout in the proof runner. | repaired | `ruff format` changed seven insertions/thirteen deletions of line wrapping only; fixture, commands, evidence, and outcomes are unchanged. | `82e1ece`; `ruff format --check`, Ruff lint, 28 focused tests passed |

## Runtime provenance block

| Namespace | Product | Displayed family | Exact model identifier | Reasoning mode | Source / unavailable reason |
| --- | --- | --- | --- | --- | --- |
| requested | Codex | Luna or available lightweight model | not exposed by runtime | Extra High | task request |
| actual | Codex | GPT-5 | not exposed by runtime | not exposed by runtime | runtime exposes product/family but not deployment ID or reasoning mode |

- Capture file/evidence: `scripts/capture_model_provenance.py` stdout captured at 2026-07-12T16:23:21Z for implementation and handoff; exact fields below.
- Codex CLI version: `codex-cli 0.144.1`
- Branch/commit: `codex/gate-001-bilingual-overfit-proof` / `3d0f4fbdc7c8ad40d30b9e5eb03e448e712d2e2e`
- Phase/role/task path: implementation / `/root/gate001_implementation`
- Privacy confirmation: no prompts, hidden chain-of-thought, token counts, secrets, or raw thread IDs are recorded.

```json
{"phase":"implementation","role":"agent","task_path":"/root/gate001_implementation","requested":{"model":"Luna or available lightweight model","reasoning_mode":"Extra High"},"actual":{"product":"Codex","displayed_model_family":"GPT-5","exact_model_identifier":"not exposed by runtime","reasoning_mode":"not exposed by runtime"},"environment":{"codex_cli_version":"codex-cli 0.144.1","branch":"codex/gate-001-bilingual-overfit-proof","commit":"3d0f4fbdc7c8ad40d30b9e5eb03e448e712d2e2e","thread_id":"not recorded (privacy)"}}
```

## Check selection and verdicts

### Review cycle 1

- Review model / mode: pending independent review; requested heavier Extra Thinking, actual fields must be captured by that runtime.
- Commit reviewed: pending; handoff source is `3d0f4fbdc7c8ad40d30b9e5eb03e448e712d2e2e` plus this documentation-only evidence commit.
- Selected `CHECK.md` sections: 6, 8, 9.1 and GATE-001 R2.
- Major sections marked N/A and why: performance optimization and 15-60-minute thermal pilot are N/A; this is a bounded correctness/memorization gate, not a throughput claim.
- Ticket acceptance result: pending
- Philosophy alignment: pending
- Complexity / change-surface result: pending
- ML-system result: pending
- Verdict: pending

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| N/A | pending | No independent review has run. | N/A | Run after candidate evidence is committed. |

## Failed-review handoff

N/A — no independent review has run, so no review failure handoff exists.

## Repair result

Implementation repairs before review (all retained in the experiment record):

- Cycle 1: replace too-short/terminal fixture passes with a versioned 11-batch
  pass and reject a recovery point that has no suffix.
- Cycle 2: concentrate the fixture on its two declared JP/EN continuations;
  this preserved the NLL threshold rather than relaxing it.
- Cycle 3: predeclare unambiguous fixed in-fixture continuation prefixes after
  retaining the short-prompt sampling failure; rerun the full comparison.
- Cycle 4: independent pre-review found only `ruff format --check` drift in
  `scripts/run_gate_overfit.py`. Commit `82e1ece` is formatter-only; focused
  static/28-test validation passed and requires exact-head re-review.

## Final evidence

- Resolved Hydra command/config: exact Docker command and profile values in the linked experiment record; `profile=gate_overfit`, CUDA BF16, seed 42, 200 updates, step-100 recovery, W&B disabled.
- Data/tokenizer/model identity: versioned fixture manifest `eb607b8156d75987032d213e62dbb8c76bf61c7521d0a8db51b35c88c57804e9`; tokenizer fingerprint `12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b`; random 3,299,754-parameter decoder.
- Validation and measurements: this is same-corpus memorization, explicitly not validation. Reference/repeat/resume each: 200 steps, 3,200 targets, NLL `10.8711 -> 0.164456`; common full-trace SHA-256 `8758e770ae67e408c47b6a32e862513cf5b79668d1e45742f73e61ea298c7350`.
- Performance/resource result if applicable: R2 GB10 target smoke completed; no throughput, thermal, or performance claim is made.
- Failed attempts retained at: experiment record inventory and ignored local `reports/gate-001/attempt-{1..6}`.
- Known trade-offs: fixed-fixture loss and samples are intentionally non-generalizing evidence.
- Unresolved risks: GPU memory-efficient attention warned that its algorithm is non-deterministic under the current `warn_only` policy; exact equality is demonstrated only on this current environment. Independent review remains required.
- Human decision requested: perform the independent exact-head review; merge only after the later guarded audit.

## Merge authority and final audit

- Merge path: guarded agent self-merge, only after the later exact-head audit.
- Human authorization: user instruction in this bounded roadmap series: “これからはとりあえず全部セルフマージしていいよ”; later AGENTS policy requires full guarded gates.
- Authorization evidence location: parent task context and eventual PR final-audit comment.
- Authorization covers this named PR or bounded ticket/goal series: yes — roadmap completion series, subject to all guarded gates.
- Exact independently reviewed head SHA: pending
- Latest independent verdict / model / mode: pending
- All actionable findings repaired and independently re-reviewed: pending
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: pending audit
- Newer human objections since authorization/review: none known at implementation start
- Human review dismissed by an agent: no
- Unresolved review threads at final audit: pending
- Branch-protection required-context inventory: pending
- Applicable configured workflow/check inventory: pending
- Observed exact-head check statuses: pending
- Expected checks absent, pending, skipped, cancelled, or non-successful: pending
- No-check evidence when both inventories are empty: pending
- Target branch and base SHA at final audit: `main` / pending
- Up-to-date, conflict-free, and mergeable evidence: pending
- Record, ledger, PR trail, validation, and risks parity: pending
- Prohibited self-merge categories: pending final audit; expected clear (no secrets, paid resource, deployment, or release).
- Admin/bypass/force/disabled-check requirement: no
- Final audit PR body/comment location: pending
- Final audit changed reviewed head: no
- Immediate pre-merge re-fetch/compare observation location: pending
- Immediate refresh compared authorization, head, base, review decision/objections, threads, expected checks/statuses, and mergeability: pending
- Drift found: pending
- Merge outcome: not merged — implementation agent cannot Ready or merge.

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | implementation | Built a small direct runner on the canonical training/checkpoint/generation paths; retained failures; converged to exact same-seed/recovery evidence | Initial fixture sizing and sampling-prefix choices needed empirical repair; native CPU Torch could not provide CUDA evidence | Ticket, manifest/cursor contracts, predeclared retry records, and GB10 measurements | candidate complete; independent review pending |

## Ledger update

- [x] Added the draft PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated per-model attempt, pass, repair, and review counts for the completed implementation candidate; review counts await the independent verdict.
- [x] Confirmed that the PR execution trail is ready to be updated from this record.
- [ ] Recorded complete guarded self-merge authority/audit evidence.
- [x] Confirmed that this is not the bootstrap self-merge policy PR.
