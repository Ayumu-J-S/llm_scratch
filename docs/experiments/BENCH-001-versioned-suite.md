# BENCH-001 — versioned base-model benchmark suite

## Predeclared question and conditions

- Question: can fixed checkpoint evaluation track Japanese commonsense and
  general mathematical reasoning without exposing reserved tests or allowing
  benchmark data into training?
- Expected result: deterministic scores and identities from the same fixture
  checkpoint; routine commands read only deterministic development subsets;
  injected contamination blocks scoring and identifies its training document.
- Failure conditions: a Hydra override grants final access; checkpoint,
  tokenizer, prompt, scorer, data, or decoding identity is absent; a complete
  training scan is not proven; raw benchmark/model text reaches local evidence
  or W&B; or an external result can enter the repository checkpoint path.
- Scope: JCommonsenseQA v1.3 and GSM8K only, zero-shot, with a narrow in-repo
  adapter.
- Out of scope: leaderboard breadth, chat/SFT tasks, inference optimization,
  automated judges, and using external outputs for training.

## Implementation and evidence trail

| Cycle | Phase | Outcome | Important evidence |
| ---: | --- | --- | --- |
| 1 | Implementation | Complete | Pinned registry, deterministic development subsets, guarded final entrypoint, checkpoint scoring, complete contamination scan, atomic local JSON, compact W&B table, isolated external aggregate recorder |
| 2 | Focused validation | PASS | Benchmark, generation, and config-profile tests pass; canonical online sources verify to 256 selected development examples and the documented subset hashes |
| 3 | Independent `/review` | FAIL | Review of `db221fe` plus the complete working-tree diff passed 337 tests and Ruff, then found three integrity defects: context incompatibility was detected only after the complete corpus scan, GSM8K ignored checkpoint BF16 precision, and external records accepted caller-asserted protocol/partition identity |
| 4 | Repair | Complete | Added a prompt/continuation/full-generation context preflight before the scan, passed checkpoint precision into every GSM8K forward, and made the recorder attach and enforce the compiled development protocol, source, selection, and 128-example totals |
| 5 | Full validation | PASS | Official CPU gate: 339 passed, 1 skipped; Ruff, Hydra config preflight, lock drift, offline smoke, `uv lock --check`, changed-path format, and diff checks pass. Canonical sources reverify to 256 examples, the pinned hashes, and context requirements JCommonsenseQA=98/GSM8K=264 |
| 6 | Independent re-review | FAIL | Exact-head review of `6053008` reran 339 tests (1 skipped) successfully, then found that result identity bound the training commit but not the executable evaluator/runtime, and that fixture scoring asserted repeatability rather than pinned golden outputs |
| 7 | Repair | Complete | Added evaluator Git dirty/commit, dependency-lock, OS/Python/PyTorch/CUDA/device/container identity to the result hash; pinned both fixture task metrics, predictions, trace hashes, generation length/stop reason, and empty-completion hash |
| 8 | Full validation | PASS | Official CPU gate: 339 passed, 1 skipped; Ruff, Hydra config preflight, lock drift, offline smoke, `uv lock --check`, changed-path format, and diff checks pass |
| 9 | Independent re-review | FAIL | Exact-head review of `9722b49` reran 339 tests (1 skipped), then found the canonical 48-codepoint scan was repeated and prohibitively materialized for every checkpoint, the default benchmark cache polluted Git status, and dirty evaluator identity bound filenames but not changed bytes |
| 10 | Repair | Complete | Added a verified suite/corpus/normalizer/scanner-implementation-bound reusable scan artifact with a no-rescan milestone invariant, moved generated cache state under ignored output storage, and bound tracked diffs plus non-ignored untracked bytes into Git identity |
| 11 | Full validation | PASS | Official CPU gate: 340 passed, 1 skipped; Ruff, Hydra config preflight, lock drift, offline smoke, `uv lock --check`, changed-path format, and diff checks pass |
| 12 | Independent re-review | FAIL | Exact-head review of `e1b6be3` reran 340 tests (1 skipped), then found that the external recorder could overwrite a checkpoint path/inode, the first full scan still performed per-codepoint window SHA-256/materialization across the canonical corpus, and cached evidence omitted its full producer dependency identity |
| 13 | Repair | Complete | Confined external JSON to a dedicated non-checkpoint tree with path/symlink/hardlink rejection; replaced corpus-window SHA-256 with a collision-verified linear rolling matcher; bound cached evidence to scoped evaluator source bytes, lock, installed PyArrow, Python/platform, suite/task content, and manifest content/fingerprints |
| 14 | Focused validation | PASS | Benchmark suite and Ruff pass; a 1,000,048-codepoint scale invariant requires exactly one rolling update per codepoint, only one exact-candidate allocation/verification, and matcher storage proportional to the two unique fixture patterns. Canonical final matcher construction retains 1,029,282 unique patterns rather than a 36,972,934-node trie |
| 15 | Full validation | PASS | Official CPU gate: 341 passed, 1 skipped; Ruff, Hydra config preflight, lock drift, offline smoke, `uv lock --check`, changed-path format, and diff checks pass |
| 16 | Independent re-review | FAIL | Exact-head review of `cba8281` reran 341 tests (1 skipped), then found that JCommonsenseQA separately encoded choices and therefore scored a synthetic tokenizer prefix, unsupported CUDA BF16 failed only after the scan, and external results could claim a one-token context under the same protocol identity |
| 17 | Repair | Complete | Changed JCommonsenseQA to one exact joint encoding with tokenizer-offset suffix masking and a new scorer identity; added CUDA BF16 capability preflight before suite loading/scanning; required protocol-bound no-truncation evidence, per-task required context, and a fixed 129-token external minimum |
| 18 | Focused validation | PASS | Corrected fixture golden, joint-tokenization invariant, external context contract, CUDA BF16 preflight, and Ruff pass; canonical sources reverify to 256 examples, protocol hash `79cf8b2…`, registry fingerprint `39e658f…`, and context requirements JCommonsenseQA=97/GSM8K=264 |
| 19 | Full validation | PASS | Official CPU gate: 345 passed, 1 skipped; Ruff, Hydra config preflight, lock drift, offline smoke, `uv lock --check`, changed-path format, and diff checks pass |
| 20 | Independent re-review | FAIL | Exact-head review of `cd8fade` reran 345 tests (1 skipped), then found that the new offset API made ordinary training tokenization materialize discarded offsets for every token and that repository benchmark output could overwrite a sibling checkpoint in the selected run's checkpoint namespace |
| 21 | Repair | Complete | Restored the original ID-only canonical `encode` hot path while keeping offset extraction explicit to benchmark scoring; derived the checkpoint-owned root after verified checkpoint/config loading and rejected JSON output/temp placement, checkpoint namespaces, symlink aliases, and hardlinks before suite loading or scanning |
| 22 | Focused validation | PASS | ID-only training encode, sibling checkpoint, directory-symlink alias, hardlink alias, scoring golden, benchmark/tokenizer/generation/config suites, and Ruff pass (87 focused tests) |
| 23 | Full validation | PASS | Official CPU gate: 348 passed, 1 skipped; Ruff, Hydra config preflight, lock drift, offline smoke, `uv lock --check`, changed-path format, and diff checks pass |
| 24 | Independent re-review | FAIL | Exact-head review of `10f237d` reran the full 348-test gate (1 skipped), then found that overlapping shingles could materialize the same reference many times per document before final deduplication, the zero-weight generation golden did not uniquely bind special-token sequences that decode identically, and untracked symlink target bytes could change without changing evaluator identity |
| 25 | Repair | Complete | Deduplicated shingle matches by task/example/field identity inside each document; added a versioned SHA-256 over canonical-JSON generated token IDs without retaining raw IDs; rejected every tracked or untracked non-regular evaluator path before reading dirty content |
| 26 | Focused validation | PASS | Bounded per-document reference identity, corrected versioned token-sequence golden, tracked/untracked symlink rejection (including target mutation), 27 benchmark/reproducibility tests, and Ruff pass |
| 27 | Full validation | PASS | Official CPU gate: 351 passed, 1 skipped; Ruff, Hydra config preflight, lock drift, offline smoke, `uv lock --check`, changed-path format, and diff checks pass |
| 28 | Independent re-review | FAIL | Exact-head review of `974231e` reran the full 351-test gate (1 skipped), then found that the output guard treated every selected checkpoint parent as a checkpoint namespace and that the optimizer-bearing loaded checkpoint payload remained live throughout the complete contamination scan and scoring |
| 29 | Repair | Complete | Restricted namespace protection to checkpoint-configured/recognized roots and exact symlink/hardlink aliases without treating an arbitrary selected file's parent as a checkpoint tree; constructed the sampler in a tight helper scope and proved both the loaded wrapper and an injected optimizer-state tensor are reclaimed before suite loading/scanning |
| 30 | Focused validation and BENCH sweep | PASS | 29 benchmark/reproducibility tests, scoped Ruff/format, and diff checks pass. A systematic read-only sweep of runner lifetime/path isolation, protocol/result identity, large-scan retained state/complexity, W&B retention, and external aggregate isolation found no additional acceptance-blocking defect |
| 31 | Full validation | PASS | Official CPU gate: 353 passed, 1 skipped; Ruff, Hydra config preflight, lock drift, offline smoke, `uv lock --check`, changed-path format, and diff checks pass |
| 32 | Independent re-review | FAIL | Exact-head review of `57d6ac0` reran 353 tests (1 skipped), then proved that 126/128 verbatim selected JCommonsenseQA source records escaped contamination detection because canonical serialization changed source key order/spacing and their short individual fields produced no 48-codepoint shingle |
| 33 | Repair | Complete | Retained each selected example's exact pinned JSONL source record, bound its SHA-256 into selected-example identity, and added a protocol-versioned canonical JSON-object identity that is invariant to key order and insignificant whitespace while remaining linear per candidate document |
| 34 | Focused validation | PASS | 103 benchmark/config/tokenizer/generation/reproducibility tests and scoped Ruff/format/diff checks pass; a 256-example synthetic all-selected invariant detects 128/128 source-faithful and reordered/indented variants for each task. Canonical online acceptance independently detects 128/128 exact source records and 128/128 reordered JSON objects for both JCommonsenseQA and GSM8K; registry `adf433c…`, protocol `d56ffdb…`, JCommonsenseQA selection `37e39dc…`, GSM8K selection `03fa95e…` |
| 35 | Full validation | PASS | Official CPU gate: 354 passed, 1 skipped; Ruff, Hydra config preflight, lock drift, offline smoke, `uv lock --check`, changed-path format, and diff checks pass |
| 36 | Independent re-review | FAIL | Exact-head review of `6abee17` reran 354 tests (1 skipped), canonical development/final loading, context checks, and Ruff, then found three defects: structure-aware contamination did not compose with BOM/newline/NFC text normalization; external records omitted separately attested prompt/scorer component hashes; and GSM8K generation accepted non-finite logits as arbitrary tokens and a normal score |
| 37 | Repair | Complete | Applied the repository text-identity normalization before canonical JSON parsing and revised the cache identity; attached and independently attested prompt/scorer hashes in external records; rejected any non-finite generation logits before argmax or sampling can produce a token or result |
| 38 | Focused validation | PASS | 27 benchmark/generation tests plus scoped Ruff/format/diff checks pass. The all-selected synthetic invariant now combines BOM, outer whitespace, CRLF, key reordering, and JSON indentation; canonical online acceptance independently detects 128/128 such variants for each task, and tests prove all NaN/positive-infinity/negative-infinity generation paths fail closed |
| 39 | Full validation | PASS | Official CPU gate: 357 passed, 1 skipped; Ruff, Hydra config preflight, lock drift, offline smoke, `uv lock --check`, changed-path format, and diff checks pass |

## Resolved protocol

- Suite: `BENCH-001-suite-v1`
- Development selection: the first 128 examples by SHA-256 rank of canonical
  task/example ID, separately for each task.
- JCommonsenseQA: a fixed Japanese question/options/answer prompt; the exact
  prompt-plus-choice string is encoded once, tokenizer source offsets define
  the scoreable choice suffix, and boundary-crossing tokens are rejected.
  Length-normalized conditional log probability is primary and raw
  log-probability accuracy is retained as a secondary metric.
- GSM8K: fixed `Question`/`Answer` prompt, greedy continuation, 128-token cap,
  the checkpoint-owned evaluation precision, and the dataset repository's
  `####` answer regex.
- Final acknowledgement: `BENCHMARK_FINAL_ACK=BENCH-001-suite-v1`; checked
  outside Hydra.
- Contamination: complete checkpoint-owned train selections, exact/normalized
  whole-document identity, source-faithful record identity, text-normalized
  canonical JSON-object identity across key-order/whitespace variants, and
  normalized 48-codepoint shingles.

## Review selection

- `PHILOSOPHY.md`: train/test separation, intermediate checkpoint evaluation,
  fixed development/final boundaries, same-protocol external comparisons, and
  quota-safe W&B evidence.
- `CHECK.md` 8.2: development/final separation, contamination, external output
  exclusion, and complete evaluation identity.
- `CHECK.md` 8.3: no benchmark claim is made from the fixture proof.
- `CHECK.md` 9.2: W&B contains compact aggregates only; no artifact upload or
  raw task content.
- R1 is sufficient because this ticket changes evaluation behavior without
  changing training objective, optimizer, model, data order, or hot-path
  performance.

## Current conclusion

All ten independent failed reviews remain visible. Their twenty-five findings are
repaired without weakening the fixed protocol or complete contamination gate:
cheap context incompatibility precedes scanning, both tasks honor checkpoint
precision, external records are pinned, evaluator/runtime and dirty source
bytes are identity-bound, fixture outputs are golden, generated cache state is
ignored, external JSON cannot enter or alias checkpoint storage, the first
corpus scan uses a bounded linear rolling matcher, full scoped producer identity
invalidates stale scan evidence, and completed suite/corpus/producer-bound scan
evidence is reused across milestones. Choice scoring now uses an exact joint
encoding and offset-defined suffix, unsupported CUDA BF16 is rejected before
the training scan, and external records prove sufficient no-truncation context.
Normal training encoding remains ID-only, while local result paths cannot enter
or alias a configured/recognized checkpoint namespace or the selected
checkpoint inode without misclassifying a broad ad hoc parent. Per-document
contamination evidence is reference-deduplicated, generation traces bind
canonical token-sequence hashes, evaluator identity rejects symlinked or special
producer paths, and the full optimizer-bearing load is reclaimed before the
suite and corpus scan. Selected examples now retain and hash the pinned source
record representation, while text normalization composes with a
structure-normalized JSON identity to detect BOM, newline, Unicode,
key-order, and whitespace variants; canonical acceptance covers every selected
development record in both tasks. External comparisons separately attest the
compiled prompt and scorer hashes, and generation rejects non-finite logits
before any GSM8K token or score is accepted.
An extra
repository-wide format diagnostic identified four pre-existing, unrelated
files outside this ticket's diff; the configured Ruff lint gate and all changed
benchmark paths pass, so those files were not rewritten here. The tenth repair's
focused and full gates plus canonical online acceptance pass; exact-head independent
re-review remains. No
benchmark score from the zero-weight fixture is a model-quality result.
