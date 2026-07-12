# DATA-004 - Pinned Baseline Mixture

- PR: [#41](https://github.com/Ayumu-J-S/llm_scratch/pull/41) (draft)
- Branch: `codex/data-004-pinned-baseline-mixture`
- Ticket: `DATA-004`
- Hypothesis: exact shard manifests, content-disjoint splits, target-token scheduling, and bounded QA can provide a trustworthy 50/50 Japanese/English baseline stream.
- Experiment record: `docs/experiments/DATA-004-pinned-baseline-mixture.md`
- Started: 2026-07-12
- Final verdict: in progress
- Final record owner: implementation agent

## Scope and decision context

- Goal: implement and measure the pinned bilingual corpus required before the first real baseline.
- In scope: exact source research/pins/licenses, shard inventories, token mixture, non-empty train/validation, aggregate QA, cache/headroom safety, and bounded live/training evidence.
- Out of scope: adaptive/model-based filtering, benchmarks, W&B corpus artifacts, model/run-budget selection, full operations framework, or perfect-quality claims.
- Relevant `PHILOSOPHY.md` principles: Japanese/English first, train the claimed model, data/benchmark separation, one-machine boundary, evidence-first experiments, bounded storage, and retained failures.
- Baseline commit/run: `main@7648316`; GATE-001 and its finalization merged.
- Intended evidence: predeclared success conditions, primary-source inventories, fixture QA, live cold/warm reports, bounded stream proof, and independent CHECK review.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `main@7648316`; requested Luna/lightweight Extra High | Implement complete DATA-004 without VAL/DGX/OPS scope expansion. | in progress | Predeclared source candidates, 50/50 target ratio, integrity gates, disk reserve, budgets, and stop conditions. | experiment/model-run records and live draft PR |
| 1 | source audit | not exposed by runtime | not exposed by runtime | exact official Hugging Face metadata and dataset/model terms | Verify immutable inventories, provenance, licenses, terms, and safer alternatives before live downloads. | candidate pivot complete | Rejected FineWeb-Edu English because Llama 3 output lineage carries an explicit downstream-LLM restriction; rejected the Japanese Edu derivative because current permissive DeepSeek terms do not prove historical annotation terms. Selected direct FineWeb/FineWeb-2 revisions; official APIs reproduced 15 EN and 175 JA train shards with aggregate rows/bytes and per-file LFS identities. | primary-source audit retained in experiment record; no corpus shard downloaded |
| 1 | review | pending | pending | future stable candidate | Review PHILOSOPHY, acceptance, CHECK all 4/5.3/5.4/8.2 and applicable R2/R3. | pending | pending | pending |

## Runtime provenance block

| Namespace | Product | Displayed family | Exact model identifier | Reasoning mode | Source / unavailable reason |
| --- | --- | --- | --- | --- | --- |
| requested | Codex | Luna or available lightweight model | not exposed by runtime | Extra High | project default |
| actual | Codex | GPT-5 | not exposed by runtime | not exposed by runtime | runtime does not expose deployment ID or reasoning mode |

- Capture: stdout at `2026-07-12T21:43:54.292076Z`.
- Codex CLI: `codex-cli 0.144.1`.
- Branch/commit: `codex/data-004-pinned-baseline-mixture` / initial candidate `3715c15e3462679e14de76a5ff2d61ed19ea9a9e`; baseline `7648316d1ae6d503fde89661728074f963321506`.
- Phase/role/task path: implementation / `/root`.
- Privacy: no prompts, hidden chain-of-thought, token counts, secrets, or raw thread ID.

```json
{"schema_version":"1.0","captured_at":"2026-07-12T21:43:54.292076Z","phase":"implementation","role":"agent","task_path":"/root","requested":{"model":{"value":"Luna or available lightweight model","source":"explicit invocation/config default","status":"observed"},"reasoning_mode":{"value":"Extra High","source":"explicit invocation/config default","status":"observed"}},"actual":{"product":{"value":"Codex","source":"active runtime display","status":"observed"},"displayed_model_family":{"value":"GPT-5","source":"active runtime display","status":"observed"},"exact_model_identifier":{"value":"not exposed by runtime","source":"active runtime display","status":"unavailable"},"reasoning_mode":{"value":"not exposed by runtime","source":"active runtime display","status":"unavailable"}},"environment":{"codex_cli_version":"codex-cli 0.144.1","branch":"codex/data-004-pinned-baseline-mixture","commit":"7648316d1ae6d503fde89661728074f963321506","thread_id":"not recorded (privacy)"},"privacy":{"raw_thread_id_recorded":false,"prompts_recorded":false,"hidden_chain_of_thought_recorded":false,"token_counts_recorded":false,"secrets_recorded":false}}
```

## Check selection and verdicts

- Review model/mode and commit: pending.
- Selected sections: all 4, 5.3, 5.4, 8.2, comparison rules in 3, applicable R2/R3.
- Other major sections: N/A unless touched by implementation.
- Ticket/Philosophy/complexity/ML-system verdicts: pending.

## Failed-review handoff

Pending independent review.

## Repair result

N/A — no review has run.

## Final evidence

- Resolved Hydra config, source inventories, QA, measurements: pending.
- Failed attempts: retained in experiment record and aggregate evidence.
- Known trade-off: web-corpus QA reports limitations rather than claiming perfection.
- Failed source candidates: educational FineWeb derivatives were rejected before
  live access because their model-output provenance could not meet this ticket's
  conservative licensing gate. This did not relax any success condition.
- Risks: license/terms, filtering provenance, leakage, long tails, network variance, cache temp space.

## Merge authority and final audit

- Guarded agent self-merge only after exact-head PASS/PASS WITH NOTE and all gates.
- Bounded roadmap-series authorization remains in scope.
- Review/check/thread/protection/mergeability fields: pending final audit.
- Target: `main@7648316` initially.
- Unresolved licensing questions block self-merge until source review closes them.
- No admin/bypass/force path; this is not the bootstrap policy PR.

## Ledger update

- [x] Added draft PR row and implementation count.
- [ ] Review/repair counts and execution trail synchronized.
- [ ] Guarded merge evidence complete.
