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
| 40 | Independent re-review | FAIL | Exact-head review of `342037f` reran 357 tests (1 skipped), canonical source checks, Ruff, lock, and diff gates, then found two integrity gaps: an ordinary existing repository JSON input could be selected as benchmark output and destructively replaced; run-manifest verification recorded but did not compare the exact dirty-worktree content digest |
| 41 | Repair | Complete | Confined repository benchmark results to a dedicated configured root outside input/cache/checkpoint/artifact namespaces; made internal and external atomic publication exclusive and no-overwrite; rejected existing files, path escapes, symlinks, and hardlinks. Run-manifest verification now requires exact equality of the captured worktree-content digest |
| 42 | Focused validation | PASS | 52 benchmark/reproducibility/config tests plus scoped Ruff/format/diff checks pass. Regressions preserve the canonical registry bytes under a malicious output override, reject configured-root escape and repository-data roots before suite loading, prove existing internal/external results cannot be replaced, and reject same-status/same-path dirty tracked-byte mutation during manifest verification |
| 43 | Full validation | PASS | Official CPU gate: 360 passed, 1 skipped; Ruff, Hydra config preflight, lock drift, offline smoke, `uv lock --check`, changed-path format, and diff checks pass |
| 44 | Independent re-review | FAIL | Exact-head review of `8626c24` reran 360 tests (1 skipped) successfully, then found that the fixed, exclusive default `benchmark.json` allowed the canonical command to publish only once across all checkpoints/partitions and that JCommonsenseQA could accept non-finite raw vocabulary logits when its gathered target probabilities remained finite |
| 45 | Repair | Complete | Made the default result path `dev\|final-<evaluation_identity_sha256>.json`, binding access, physical checkpoint bytes, compiled suite/protocol, evaluator revision, lock, and runtime while retaining exclusive no-overwrite publication and explicit fresh-root/path control; rejected any non-finite raw choice-scoring logit before log-softmax or target extraction |
| 46 | Focused validation | PASS | 65 benchmark/generation/config/reproducibility tests plus scoped Ruff pass. Regressions prove the default filename equals the complete result identity, exact reruns fail before repeating the training scan, and non-target NaN/positive-infinity/negative-infinity vocabulary logits cannot produce a JCommonsenseQA score |
| 47 | Full validation | PASS | Official CPU gate: 364 passed, 1 skipped; Ruff, Hydra config preflight, lock drift, offline smoke, `uv lock --check`, changed-path format, and diff checks pass |
| 48 | Independent re-review | FAIL | Exact-head review of `4243f92` reran the full 364-test gate (1 skipped), then reproduced a selected short JCommonsenseQA record embedded in a larger training document whose reordered/pretty JSON defeated exact, shingle, and standalone-JSON matching. The review also demonstrated that extreme but finite raw logits can normalize to non-finite log probabilities |
| 49 | Current-main integration and repair | Complete | Replayed the BENCH-only history onto WB-001 main `8791bb7`; directly adopted its `mode=disabled\|offline\|online` W&B schema and bounded failure-isolated SDK calls. The contamination scanner now extracts disjoint innermost JSON objects from prose/JSON wrappers in linear document work, invalidates stale scan evidence, and choice scoring rejects non-finite normalized probabilities or sums |
| 50 | Focused validation | PASS | 76 benchmark/generation/config/reproducibility tests plus scoped Ruff, format, and diff checks pass. All 256 selected fixture records are detected after key reordering, pretty printing, CRLF/BOM normalization, and embedding inside both prose and an outer JSON object; W&B compact-table and post-commit failure isolation regressions pass |
| 51 | Full validation | PASS | Current-main official CPU gate: 480 passed, 1 skipped; repository Ruff, resolved Hydra config preflight, lock-drift detection, disabled/offline process-tree-isolated smoke, `uv lock --check`, changed-path format, and diff checks pass. The resolved benchmark profile contains only the direct WB-001 mode and timeout schema |
| 52 | Independent re-review | FAIL | Exact-head review of `97acc5c` reran 480 tests (1 skipped), 124 focused tests, canonical source/context checks, and adversarial contamination probes, then found that ASCII-escaped NFD JSON strings were decoded only after document-level NFC. Reordered/pretty variants therefore escaped every match path for 102/128 selected JCommonsenseQA records |
| 53 | Repair | Complete | Recursively NFC-normalized decoded JSON keys and values before canonical hashing, rejected keys that collide after normalization, and revised the scan/cache identity. The all-selected wrapper regression now emits every string through ASCII-escaped NFD before structured matching |
| 54 | Focused and canonical validation | PASS | 76 benchmark/generation/config/reproducibility tests plus scoped Ruff and diff checks pass. An adversarial run against the pinned canonical development suite detects 128/128 JCommonsenseQA and 128/128 GSM8K records after key reordering, pretty printing, ASCII-escaped NFD strings, and prose wrapping |
| 55 | Full validation | PASS | Repaired current-main official CPU gate: 480 passed, 1 skipped; repository Ruff, resolved Hydra preflight, lock-drift detection, and disabled/offline process-tree-isolated smoke pass |
| 56 | Exact-head independent `/review` | blocked | Review was invoked against repaired head `a8c0823` and base `8791bb7`, but the reviewer exited before analysis because the Codex account usage limit was reached. No verdict was produced; the draft PR retained the exact command, CLI/reviewer sessions, exit code, and re-run handoff |
| 57 | Supplemental independent audit | FAIL | The audit detected only 1/128 double-serialized JCommonsenseQA records and 0/128 minimal quoted/escaped-prose variants because decoded JSON string values were not recursively inspected. It also reproduced an uncaught `RecursionError` from a roughly 1,000-level array inside an otherwise balanced object candidate |
| 58 | Repair | Complete | Added per-document byte/node/structural-depth/decoded-string traversal budgets; iterative deduplicated rescanning of NFC-normalized decoded JSON string values; mapping identities inside nested objects/arrays; recursive normalized-key collision rejection; and safe handling of JSON parser, normalizer, and canonicalizer recursion/malformed failures. Revised the scan/cache identity |
| 59 | Focused and canonical validation | PASS | 79 benchmark/generation/config/reproducibility tests plus Ruff pass. Regressions cover all 128 examples per task through double serialization, nested object/array strings, and quoted prose with ASCII-escaped NFD; exact and one-unit-beyond byte/node/depth/string caps; recursive key collisions; and a 1,200-level hostile candidate followed by a valid record. Pinned canonical development acceptance independently detects 128/128 JCommonsenseQA and 128/128 GSM8K for all three wrapper modes |
| 60 | Full validation | PASS | Official CPU gate: 483 passed, 1 skipped; repository Ruff, resolved Hydra preflight, lock-drift detection, and disabled/offline process-tree-isolated smoke pass. Validation reused only the existing 4.2 MB benchmark cache with 426 GB free and did not launch a full training-corpus scan |
| 61 | Supplemental exact-head re-audit | FAIL | Audit of clean head `cf67dcc` passed 127 focused tests and Ruff, then showed that one shared decoded-JSON budget could be exhausted by 4,095 tiny prefix objects in a 41.4 KB document. The later selected record became a silent non-match while the worker still reported `scan_complete=true`; byte, node, and decoded-string small-cap variants reproduced the same integrity failure |
| 62 | Repair | Complete | Distinguished traversal-budget exhaustion from malformed/deep JSON non-matches. Byte, node, decoded-string, decode-depth, and candidate-count exhaustion now propagates as a source/document-scoped `ContaminationScanError` without raw text; the scan worker cannot construct a complete report or publish/cache an index. Bumped all scan/normalization/cache identities |
| 63 | Focused and canonical validation | PASS | 81 benchmark/generation/config/reproducibility tests plus Ruff pass. The exact 4,095-prefix default reproduction fails closed with document diagnostics, 4,000 prefixes detects the target, byte/node/string/decode-depth exhaustion fails closed, structural overdepth is a bounded non-match, and the worker writes no index before a successful bounded rerun. Pinned canonical development acceptance remains 128/128 for both tasks across double-serialized, nested object/array, quoted-prose, and additional escaped-prose modes |
| 64 | Full validation | PASS | Official CPU gate: 485 passed, 1 skipped; repository Ruff, resolved Hydra preflight, lock-drift detection, and disabled/offline process-tree-isolated smoke pass. Free disk remained 426 GB and no full training-corpus scan was launched |
| 65 | Supplemental exact-head re-audit | PASS | Clean head `18bf6d5` preserved the 4,095-prefix fail-closed result, 4,000-prefix detection control, sanitized byte/node/string diagnostics, no report/index/stale-cache reuse, 128/128 wrapper detection, and deep-candidate handling; 81 focused tests, Ruff, and diff checks passed with no finding |
| 66 | Exact-head GitHub validation | FAIL | Python 3.12 parsed the 1,200-level array candidate that Python 3.11 rejected with `RecursionError`; bounded normalization then surfaced a depth-exhaustion exception instead of treating the over-depth candidate as a non-match. Pytest failed 1 of 485 tests and correctly skipped downstream workflow steps |
| 67 | Repair | Complete | Added a quote/escape-aware lexical object-and-array depth preflight before `json.loads`, making over-depth candidate behavior independent of Python parser recursion limits; bumped the scanner, normalizer, and cache identities. Genuine byte/node/string/decode-depth work exhaustion remains fail-closed |
| 68 | Focused and full validation | PASS | Both 1,200-level array and object candidates are bounded non-matches followed by a detected valid record. The focused gate passes 81 tests; the local official CPU gate passes 485 tests with 1 skipped plus Ruff, Hydra, lock-drift, and disabled/offline process-tree smoke. Exact-head Python 3.12 GitHub validation remains pending on the new immutable commit |
| 69 | Supplemental exact-head re-audit | FAIL | Audit of `4c99976` showed that 32 enclosing object wrappers cleared the bounded extractor stack and ignored the selected innermost record. The full scan then published and cached `scan_complete=true`, `contaminated=false`; 31 wrappers detected the target, and deep array wrappers were unaffected |
| 70 | Repair | Complete | Replaced the object-start stack and overflow-discard branch with a single-pass constant-memory leaf extractor. It tracks only object depth plus the newest candidate start/depth, yielding safe innermost objects at local depth while never parsing an enclosing deep candidate. Bumped all scan/normalizer/cache identities |
| 71 | Focused, canonical, and full validation | PASS | Direct and full-scan regressions detect the selected record under 40 object, array, and mixed wrapper layers and prove the cached report is contaminated. The 1,200-level array/object and 4,095-prefix controls still pass; 83 focused tests pass. Pinned canonical 40-level object/mixed acceptance is 128/128 for both tasks. The official CPU gate passes 487 tests with 1 skipped plus Ruff, Hydra, lock-drift, and disabled/offline process-tree smoke |
| 72 | Supplemental exact-head re-audit | FAIL | Audit of `4d8a254` composed the accepted adversaries and found that a reordered, ASCII-escaped NFD selected record serialized as a JSON string inside 40 array layers was silently skipped. The full scan published and cached `scan_complete=true`, `contaminated=false`; depth 30 detected, depths 31-32 failed closed, and depth 33 or greater returned a false clean result |
| 73 | Repair | Complete | Made the constant-memory string-literal extractor mirror leaf-object handling: every complete literal is decoded at local depth independent of its physical outer wrapper, while recursive decoded-string depth and the shared byte/node/string budgets remain hard limits. Bumped all scan/normalizer/cache identities |
| 74 | Focused and full validation | PASS | Separate direct and full-scan regressions detect reversed-key NFD/ASCII records as raw JSON inside 40 object layers and after JSON-string serialization inside 40 array and mixed layers; every cached report remains contaminated. An 8,193-literal document fails closed on the shared string budget; unclosed-string/newline recovery, 1,200-level hostile wrappers, and the 4,095-prefix incomplete-scan control remain intact. The focused gate passes 86 tests and the official CPU/static gate passes 490 tests with 1 skipped; no broad training-corpus scan or GPU work ran |
| 75 | Exact-head independent `/review` | FAIL | Formal review of clean head `dcab48e` reran the full 490-test gate (1 skipped), then found that the default CUDA evaluator did not impose a deterministic execution policy. `nn.MultiheadAttention` could therefore select nondeterministic SDPA/cuDNN paths while the evaluation identity omitted the policy that produced the score |
| 76 | Repair | Complete | Applied a fixed policy before checkpoint loading: seed 0, strict deterministic algorithms, required pre-initialization cuBLAS workspace, math-only SDPA, deterministic/non-autotuned cuDNN, highest FP32 matmul precision, and TF32 off. The evaluator verifies the observed backend state, fails closed if CUDA was initialized under a conflicting workspace policy, and binds the exact policy/revision into evaluation identity |
| 77 | Focused validation | PASS | Four determinism and fixture-identity regressions plus scoped Ruff/format pass. Tests prove the policy precedes checkpoint loading, backend state is strict and observable, conflicting initialized-CUDA state fails closed, and identical fixture results retain the complete fixed policy in their hashed identity; full validation and exact-head re-review remain pending |
| 78 | Full validation | PASS | Official network-isolated CPU gate passes 493 tests with 1 skipped, repository Ruff, resolved smoke Hydra preflight, lock-drift rejection, and disabled/offline process-tree smoke. The broader BENCH/generation/config/reproducibility selection passes 89 tests; changed-file format and diff checks pass. No GPU, network dataset access, full-corpus scan, or large artifact was used; 426 GB remained free |
| 79 | Exact-head independent `/review` | FAIL | Formal review of clean head `7b60ab9` found that the scanner skipped every string literal inside a balanced object range even when the enclosing object could not be decoded. A reordered, ASCII-escaped NFD selected record inside a malformed object could therefore produce and cache false-clean complete-scan evidence |
| 80 | Repair | Complete | String literals are suppressed only inside object ranges that decoded successfully. Literals inside malformed or over-depth candidates remain independently bounded and inspectable, while successfully decoded objects retain the existing deduplicated recursive traversal |
| 81 | Focused validation | PASS | A full-scan regression reproduces the malformed balanced wrapper, detects the serialized selected record by training document ID, and verifies that the cached report is contaminated. The deep object/array/mixed wrapper controls and scoped Ruff/diff gates also pass; 90 focused tests pass |
| 82 | Full validation | PASS | Official network-isolated CPU gate passes 494 tests with 1 skipped, repository Ruff, resolved smoke Hydra preflight, lock-drift rejection, and disabled/offline process-tree smoke. The scan, normalization, and JSON-object revisions were advanced so prior false-clean cache evidence is ineligible for reuse; exact-head re-review remains pending |

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
- Result publication: the default path under `outputs/benchmark-results` is
  `<access>-<evaluation_identity_sha256>.json`; results are exclusive and are
  never replaced, so an exact rerun uses a fresh configured result root/path.
- Deterministic execution: before loading the checkpoint, evaluation fixes seed
  0, strict deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  math-only SDPA, deterministic cuDNN without autotuning, highest FP32 matmul
  precision, and TF32 off. Conflicting already-initialized CUDA state fails
  closed, and the complete applied policy is hashed into result identity.
- Contamination: complete checkpoint-owned train selections, exact/normalized
  whole-document identity, source-faithful record identity, text-normalized
  canonical JSON-object identity across key-order/whitespace variants, and
  normalized 48-codepoint shingles. Decoded JSON string values are recursively
  rescanned under fixed total-byte, node, structural-depth, and decoded-string
  limits, including nested object/array, double-serialized, and quoted-prose
  wrappers. Malformed or parser-overdepth candidates are bounded non-matches;
  exhaustion of any work limit fails the complete scan closed with source and
  document identity and cannot publish reusable complete evidence. A lexical
  container-depth preflight makes parser-overdepth behavior independent of the
  Python runtime's recursion threshold. A constant-memory leaf-object extractor
  and an analogous complete-string-literal extractor still recover safe
  innermost records from arbitrarily deep object, array, or mixed wrappers
  without parsing the over-depth envelope. Physical wrapper depth does not
  consume logical decoded-string depth, while the shared work budgets still
  fail the whole document scan closed.

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

All twenty failed review/audit cycles remain visible. Their thirty-eight findings are
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
key-order, whitespace, embedded wrapper, and ASCII-escaped decoded-NFD variants.
Decoded JSON strings are recursively inspected through object, array,
double-serialized, and quoted-prose wrappers under strict per-document
byte/node/depth/string caps; normalized-key collisions and parser recursion are
non-matches rather than scan crashes, while actual traversal-budget exhaustion
is an explicit incomplete-scan error that cannot become a cached PASS. Lexical
container-depth validation prevents Python-version-specific parser recursion
behavior from changing that distinction, while constant-memory leaf extraction
prevents deep enclosing objects or arrays from hiding a safe innermost benchmark
record or serialized record string;
canonical acceptance covers every selected development record in both tasks.
External comparisons separately attest the compiled prompt and scorer hashes,
and generation rejects non-finite logits before any GSM8K token or score is
accepted; choice scoring rejects non-finite raw logits before normalization or
extraction and non-finite normalized scores after log-softmax. Benchmark W&B calls use the shared
mode/timeout contract, are wall-clock bounded, and cannot invalidate committed
local evidence. Internal and external results are exclusive, no-overwrite
publications in dedicated output namespaces, and default checkpoint-owned
filenames bind the access partition plus complete evaluation identity so
milestones, final evaluation, and evaluator revisions can coexist without
replacement, while run-manifest verification compares the exact recorded
dirty-worktree bytes in addition to commit, dirty flag, and status paths.
Benchmark execution now establishes and verifies its fixed strict deterministic
CUDA/backend policy before checkpoint loading, and hashes that policy into every
result identity rather than treating deterministic generation settings as a
substitute for deterministic execution.
An extra repository-wide format diagnostic identified four pre-existing, unrelated
files outside this ticket's diff; the configured Ruff lint gate and all changed
benchmark paths pass, so those files were not rewritten here. Every repair's
focused and full gates pass; exact-head independent re-review remains pending
on the repaired successor and the PR remains draft. No
benchmark score from the zero-weight fixture is a model-quality result.
