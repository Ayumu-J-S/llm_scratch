# DATA-004 - Pinned Baseline Mixture

- PR: [#41](https://github.com/Ayumu-J-S/llm_scratch/pull/41) (ready for human review)
- Branch: `codex/data-004-pinned-baseline-mixture`
- Ticket: `DATA-004`
- Hypothesis: exact shard manifests, content-disjoint splits, target-token scheduling, and bounded QA can provide a trustworthy 50/50 Japanese/English baseline stream.
- Experiment record: `docs/experiments/DATA-004-pinned-baseline-mixture.md`
- Started: 2026-07-12
- Current verdict: PASS WITH NOTE `4681064118`; no blocking or actionable
  technical findings remain; agent self-merge rights-policy blocked
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
| 1 | implementation | not exposed by runtime | not exposed by runtime | `main@7648316`; requested Luna/lightweight Extra High | Implement complete DATA-004 without VAL/DGX/OPS scope expansion. | candidate complete | Added lazy immutable Parquet manifests, deterministic quality/split policy, native cursor, exact trained-target mixture, bounded cache safety, real profiles, aggregate QA, and tests. | implementation heads through `10c7eb1`; 276 passed, 1 skipped |
| 1 | implementation | not exposed by runtime | not exposed by runtime | exact official Hugging Face metadata and dataset/model terms | Audit immutable inventories, provenance, licenses, terms, and safer alternatives before live downloads. | candidate pivot complete | Rejected FineWeb-Edu English because Llama 3 output lineage carries an explicit downstream-LLM restriction; rejected the Japanese Edu derivative because current permissive DeepSeek terms do not prove historical annotation terms. Selected direct FineWeb/FineWeb-2 revisions; official APIs reproduced 15 EN and 175 JA train shards with aggregate rows/bytes and per-file LFS identities. | primary-source audit retained in experiment record; no corpus shard downloaded |
| 1 | repair | not exposed by runtime | not exposed by runtime | first bounded live integration on uncommitted core | Repair active cache leases exposed by early bounded iterator close. | PASS locally | Added idempotent native Parquet iterator close and cross-instance eviction regression; formal evidence exits with zero leases. | failed report `/tmp/tmp.VVORwKA74Y/data_preflight.{json,md}`; 17 focused tests |
| 1 | handoff | not exposed by runtime | not exposed by runtime | cold `10c7eb1`; warm `8548c4e`; exact pinned manifests/tokenizer | Run bounded real train/held-out streams, cold/warm cache path, and exact 262,144-target mixture. | PASS pending review | 4,096 accepted documents per split, overlap 0, exact 131,072/131,072 targets, 1.269 GB cold/zero-byte warm, zero final leases, disk admission PASS. | `reports/data/DATA-004/live-preflight-{cold,warm}.{json,md}` |
| 1 | review | not exposed by runtime | not exposed by runtime | exact `51aa6e239f8cd40c6e1a1b9279d2526cbd3404a9`; strongest GPT-5.6-class Extra Thinking requested | Review PHILOSOPHY, acceptance, CHECK all 4/5.3/5.4/8.2 and applicable 3/R2/R3. | FAIL `4680931313` | Content split did not guarantee/report document-ID disjointness; §4/R2/R3 throughput evidence absent; quota truncation unreported; live config hashes not reproducible. Underlying page-rights caveat blocks agent self-merge. | GitHub review `4680931313`; exact-head Actions `29212016075` success |
| 2 | repair | not exposed by runtime | not exposed by runtime | failed review `4680931313` at `51aa6e2`; strongest available GPT-5.6-class Extra High requested | Repair every technical finding without changing source pins, mixture gates, cache bounds, or VAL/DGX/OPS scope. | repair complete at `fee0f1a` | Content-bound schema-v2 IDs; ID and content overlap gates; production read/tokenizer/row/missing and process metrics; cursor-persisted quota-truncation accounting; complete resolved-config/argv retention; opt-in performance instrumentation; repeat summarizer; regression coverage. | `fee0f1a231e24957cee86568d9ef89f04eb4e27d`; 287 passed, 1 skipped; lock/lint/format/diff/config checks pass |
| 2 | handoff | not exposed by runtime | not exposed by runtime | code `fee0f1a`; evidence commit `458dbdc`; exact pinned manifests/tokenizer and fixed three-shard cache | Rerun cold/warm, three repeated warm observations, representative R3, and real-data CUDA/BF16 R2; retain reproducible commands/configs and failures. | scoped evidence PASS; re-review pending | Cold/warm membership, IDs, accounting and 50/50 targets match; repeats bound variance/resources; R3 provides an 18.34-minute loader observation; R2 provides 50 finite optimizer steps and checkpoint evidence. The web-page rights question remains a separate human gate. | `reports/data/DATA-004/`; `458dbdc4b08ae255cc740f1596f7ef041d0f1476` |
| 2 | independent re-review | not exposed by runtime | not exposed by runtime | exact `7542b3d1156b6474729021ed960f5090aa9dc959`; strongest GPT-5.6-class Extra Thinking requested | Re-review all four original failures, ROADMAP acceptance, PHILOSOPHY, applicable CHECK sections, and scoped evidence. | PASS WITH NOTE `4681064118` | All four technical failures are repaired; no blocking or actionable DATA-004 finding remains. R3 is loader-only and R2 is intentionally short, so long-duration GPU thermals, end-to-end/model-only headroom, and final UMA sizing remain DGX-001 scope. The separate source-rights policy gate remains open. | GitHub review `4681064118`; exact-head Actions `29214355192` success; 287 passed, 1 skipped; source recapture and all report hashes reproduced |

## Runtime provenance block

| Namespace | Product | Displayed family | Exact model identifier | Reasoning mode | Source / unavailable reason |
| --- | --- | --- | --- | --- | --- |
| requested-initial | Codex | Luna or available lightweight model | not exposed by runtime | Extra High | project default |
| requested-repair | Codex | strongest available GPT-5.6-class implementation model | not exposed by runtime | Extra High | explicit repair request |
| actual | Codex | GPT-5 | not exposed by runtime | not exposed by runtime | runtime does not expose deployment ID or reasoning mode |

- Capture: stdout at `2026-07-12T21:43:54.292076Z`; sub-agent runtimes
  independently reported the same exact-model/mode unavailability.
- Codex CLI: `codex-cli 0.144.1`.
- Branch/commit: `codex/data-004-pinned-baseline-mixture` / initial candidate `3715c15e3462679e14de76a5ff2d61ed19ea9a9e`; baseline `7648316d1ae6d503fde89661728074f963321506`.
- Phase/role/task path: implementation / `/root`.
- Privacy: no prompts, hidden chain-of-thought, token counts, secrets, or raw thread ID.

```json
{"schema_version":"1.0","captured_at":"2026-07-12T21:43:54.292076Z","phase":"implementation","role":"agent","task_path":"/root","requested":{"model":{"value":"Luna or available lightweight model","source":"explicit invocation/config default","status":"observed"},"reasoning_mode":{"value":"Extra High","source":"explicit invocation/config default","status":"observed"}},"actual":{"product":{"value":"Codex","source":"active runtime display","status":"observed"},"displayed_model_family":{"value":"GPT-5","source":"active runtime display","status":"observed"},"exact_model_identifier":{"value":"not exposed by runtime","source":"active runtime display","status":"unavailable"},"reasoning_mode":{"value":"not exposed by runtime","source":"active runtime display","status":"unavailable"}},"environment":{"codex_cli_version":"codex-cli 0.144.1","branch":"codex/data-004-pinned-baseline-mixture","commit":"7648316d1ae6d503fde89661728074f963321506","thread_id":"not recorded (privacy)"},"privacy":{"raw_thread_id_recorded":false,"prompts_recorded":false,"hidden_chain_of_thought_recorded":false,"token_counts_recorded":false,"secrets_recorded":false}}
```

## Check selection and verdicts

- Re-review model/mode and commit: not exposed / not exposed at exact
  `7542b3d1156b6474729021ed960f5090aa9dc959`.
- Selected sections: all 4, 5.3, 5.4, 8.2, comparison rules in 3, applicable R2/R3.
- Other major sections: N/A unless touched by implementation.
- Review `4680931313` verdict: FAIL; four actionable evidence/identity/
  accounting/reproducibility findings, with implementation strengths retained.
- Repair status: all four technical findings have code, tests, and rerun
  evidence at `fee0f1a` / `458dbdc`; independent review `4681064118` returned
  PASS WITH NOTE with no blocking or actionable DATA-004 finding.
- Policy status: the review's unresolved underlying web-page rights finding is
  separate from the technical repair and prohibits agent self-merge pending a
  human rights/policy disposition.

## Failed-review handoff

From review cycle 1 / GitHub review `4680931313`:

- Failed checks: ROADMAP document-ID disjointness and truncation QA;
  PHILOSOPHY consequential-run config retention; CHECK §4 and R2/R3 throughput,
  RSS, source-read/missing-data, and representative observation.
- Review model/mode: not exposed by runtime / not exposed by runtime; strongest
  appropriate GPT-5.6-class / Extra Thinking requested.
- Implementation model/mode: not exposed by runtime / not exposed by runtime.
- Repair diff: exact head `51aa6e239f8cd40c6e1a1b9279d2526cbd3404a9`.
- Reproduction: full suite and source capture pass; inspect review `4680931313`,
  reports, and preflight/identity/loader code paths.
- Invariants: keep shared content-hash split, exact target attribution, source
  pins, no raw-text retention, bounded downloads/cache, and VAL/DGX/OPS scope.
- Selected next model/mode: strongest appropriate GPT-5.6-class implementation
  model at Extra High or higher; exact runtime values must not be inferred.
- Exact repair request: content-bind v2 IDs and report both overlap types; add
  elapsed/rate/RSS/read/missing-data and repeated/R3 evidence; count quota
  truncation end to end; commit complete redacted resolved configs and exact
  commands; rerun cold/warm evidence and independent re-review.

## Repair result

Pre-review implementation repair:

- Input: bounded live smoke with `active_leases=2`.
- Change: explicit idempotent Parquet iterator close plus regression coverage.
- Evidence: formal cold/warm reports both show `active_leases=0`; focused cursor,
  cache, and early-close tests pass.
- Independent review: the later review still failed on separate findings.

Formal review repair cycle 2:

- Input: failed review `4680931313` anchored to `51aa6e239f8cd40c6e1a1b9279d2526cbd3404a9`.
- Implementation request: strongest available GPT-5.6-class model at Extra
  High; exact model identifier and actual reasoning mode are not exposed by
  the runtime.
- Code head: `fee0f1a231e24957cee86568d9ef89f04eb4e27d`.
- Evidence handoff: `458dbdc4b08ae255cc740f1596f7ef041d0f1476`.
- Identity repair: schema-v2 IDs always bind normalized content while v1
  fixture semantics remain unchanged; preflight now independently fails and
  reports document-ID and normalized-content overlap.
- Evidence repair: source-read, production-tokenization, loader, row, byte,
  missing-data, RSS/swap/page-fault, cache-rate, and whole-run measurements;
  three warm repeats; an 18.34-minute loader-only R3; and a real CUDA/BF16 R2.
- Accounting repair: target-quota truncated fragments and removed tokens are
  cursor-persisted and preserved across sync/thread/process execution; packed
  truncation without EOS fails closed.
- Reproducibility repair: schema-v2 reports retain exact original/effective
  argv, complete secret-checked resolved Hydra config, and a reproducible hash.
- Network-free evidence: `uv run pytest -q` returned 287 passed, 1 skipped;
  `uv lock --check`, Ruff lint/changed-file format, runtime requirements,
  `git diff --check`, and metadata-only config composition passed.
- Review status: independent re-review `4681064118` returned PASS WITH NOTE;
  this repair now counts as successful. The note defers long-duration GPU
  thermals, model-only/end-to-end headroom, and final UMA sizing to DGX-001.
- Merge status: the source-rights policy question remains unresolved and blocks
  agent self-merge regardless of the eventual technical verdict.

## Final evidence

- Evidence inventory: `reports/data/DATA-004/live-preflight-{cold,warm}.{json,md}`;
  `warm-repeat-{1,2,3}.{json,md}` and `warm-repeat-summary.json`;
  `representative-r3.{json,md}`; `r2-summary.{json,md}`;
  `r2-resolved-config.yaml`; `r2-run-manifest.json`; and `r2-metrics.jsonl`.
- Exact cold command, retained as original/effective argv and resolved config in
  the report:

  ```bash
  /home/ayumu/Documents/Proj/llm_scratch/.venv/bin/python3 scripts/preflight_data.py profile=pretrain_streaming data.streaming.cache.dir=/tmp/llm-scratch-data004-repair-cold-fee0f1a +preflight.max_documents_per_split=4096 +preflight.output_dir=/tmp/data004-repair-evidence-fee0f1a +preflight.report_stem=live-preflight-cold-v2 hydra.run.dir=/tmp/data004-repair-hydra-cold-fee0f1a --cold
  ```

- Exact warm command:

  ```bash
  /home/ayumu/Documents/Proj/llm_scratch/.venv/bin/python3 scripts/preflight_data.py profile=pretrain_streaming data.streaming.cache.dir=/tmp/llm-scratch-data004-repair-cold-fee0f1a +preflight.max_documents_per_split=4096 +preflight.output_dir=/tmp/data004-repair-evidence-fee0f1a +preflight.report_stem=live-preflight-warm-v2 hydra.run.dir=/tmp/data004-repair-hydra-warm-fee0f1a
  ```

- Repeated warm commands use the same interpreter/profile/cache, set
  `data.streaming.train.max_target_tokens=65536`,
  `+preflight.max_documents_per_split=1024`, and unique exact stems/run dirs
  `warm-repeat-{1,2,3}` / `data004-repair-repeat-{1,2,3}-fee0f1a`; each complete
  command is retained in its corresponding JSON report.
- Exact representative R3 command:

  ```bash
  /home/ayumu/Documents/Proj/llm_scratch/.venv/bin/python3 scripts/preflight_data.py profile=pretrain_streaming data.streaming.cache.dir=/tmp/llm-scratch-data004-repair-cold-fee0f1a data.streaming.train.max_target_tokens=4194304 +preflight.max_documents_per_split=4096 +preflight.output_dir=/tmp/data004-repair-evidence-fee0f1a +preflight.report_stem=representative-r3 hydra.run.dir=/tmp/data004-repair-r3-fee0f1a
  ```

- Exact digest-pinned R2 command:

  ```bash
  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/ayumu/Documents/Proj/llm_scratch:/workspace:ro -v /tmp/llm-scratch-data004-repair-cold-fee0f1a:/cache -v /tmp/data004-r2-container-fee0f1a:/evidence -w /workspace -e PYTHONPATH=/workspace/src -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/workspace llm-scratch:env-001 python src/train.py profile=pretrain_streaming data.streaming.cache.dir=/cache training.max_steps=50 training.max_tokens=null training.max_time=null wandb.enabled=false artifacts.checkpoints_dir=/evidence/checkpoints hydra.run.dir=/evidence/hydra
  ```

- Data identities: Japanese manifest `2fc3eb60986c96fcb752b14d740dd5a3f7cea8b52bb5a13cb5834a1f805d6bba`;
  English manifest `626a1eb095e9089e5c62ee2df9c058ab7c6dfc54064eca5c13e4d84e65a8d60a`;
  tokenizer `12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b`.
- Validation: 287 passed, 1 skipped; lock, lint, changed-file format,
  runtime-requirement, diff, and metadata-only config checks pass.
- Cold/warm: 4,096 accepted documents per split; document-ID and content overlap
  both zero; 262,144 trained targets exactly 50/50; whole-run 359.483/227.570
  seconds; packing 5,492.52/5,613.36 targets/s; peak RSS 814,043,136/
  837,894,144 bytes; zero swap/major faults; cold 3 downloads and
  1,269,008,673 bytes, warm zero; zero active leases/retries/corruptions; quota
  truncation 2 fragments and 895 removed tokens; projected free with full cache
  plus largest temp 440,163,898,869/440,163,358,197 bytes against a
  256,000,000,000-byte reserve.
- Repeated warm: three observations at 1,024 documents/split and 65,536 targets;
  whole-run 56.981-58.185 seconds (median 57.359, spread 1.205); packing
  4,961.34-5,040.06 targets/s (median 4,997.48, spread 78.72); peak RSS
  695,148,544-723,161,088 bytes; zero swap/major faults.
- R3: 1,100.519 seconds (18.34 minutes), 4,194,304 exact 50/50 targets,
  4,594.71 packed targets/s, 819,585,024-byte peak RSS, and zero swap, major
  faults, downloads, retries, leases, overlap, or missing-data events. This is
  a loader-only observation, not model/GPU supply-headroom evidence.
- R2: repository image ID `sha256:23a1bee69fe189e77105cdddeee9aeff6ef0763d58a691625fbfcab64efd1887`
  from digest-pinned NGC base; NVIDIA GB10, BF16 CUDA 13.3, 50 optimizer steps;
  3,200 targets in 5.921 training seconds (540.44 targets/s), 93.10 ms median/
  95.92 ms p95 step, finite loss/gradients, five clipped updates, zero
  non-finite values, validation loss 9.67945, and verified best/recovery/final
  checkpoints totaling 1,783,745,333 bytes. Samples covered 36-41 C,
  4.45-19.45 W, 208-2,411 MHz, and 0-68% utilization. The resolved-config,
  run-manifest, and metrics hashes are
  `af4e2221e87fbb7ded4d798123c180b11ad1c9e7c6d0341ef3ed8a03518a1a70`,
  `1f56031b3c25e5c232696357081485d4454c0dac27157cc8a3805c7e62e2e397`,
  and `4633b0e9d4861797416efdacc09da407cbe662899e84703c7546e56df6cfe791`.
- Config reproduction: running `config_fingerprint` over each report's complete
  `reproduction.resolved_hydra_config` reproduces both its
  `fingerprints.config` and `reproduction.config_sha256`; the cold/warm hashes
  are `2399ae1bf1fbd1e8d3bac180280787fda10e43c9506d1585cfba45650cb9efc9`
  and `f4dd39e18620142e78e9809db0ea0223955727ad519ed1ce53a44fae6ebc39e3`.
- Failed attempts: lease leak and dirty-evidence warm attempt are retained at
  the `/tmp` paths in the experiment record. The host R2 attempt failed closed
  before data because host PyTorch is CPU-only and remains at
  `/tmp/data004-r2-fee0f1a`; the pinned container then ran the unchanged profile.
- Known trade-off: web-corpus QA reports limitations rather than claiming perfection.
- Failed source candidates: educational FineWeb derivatives were rejected before
  live access because their model-output provenance could not meet this ticket's
  conservative licensing gate. This did not relax any success condition.
- Risks: ordinary web-content rights, mixed-language hard-filter trade-offs,
  unobserved shards, row-group memory tails, network variance, and cache temp space.

## Merge authority and final audit

- Guarded agent self-merge only after exact-head PASS/PASS WITH NOTE and all gates.
- Bounded roadmap-series authorization remains in scope.
- Latest independent verdict is PASS WITH NOTE `4681064118`; the PR is ready
  for human review so that the separate source-rights disposition and merge
  decision can be made. Agent self-merge remains prohibited.
- Target: `main@7648316` initially.
- Review `4680931313` treats the disclosed underlying web-page rights caveat as
  an unresolved legal/licensing question that blocks agent self-merge. A human
  policy/rights disposition and permitted merge path are required even after a
  technical PASS.
- No admin/bypass/force path; this is not the bootstrap policy PR.

## Ledger update

- [x] Added draft PR row and implementation count.
- [x] Repair, passing re-review, and execution trail synchronized.
- [ ] Guarded merge evidence complete.
