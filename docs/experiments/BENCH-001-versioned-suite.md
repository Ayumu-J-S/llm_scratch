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
| 16 | Independent re-review | Pending | Repeat against the exact committed fourth-repair head and preserve the verdict in the pull request |

## Resolved protocol

- Suite: `BENCH-001-suite-v1`
- Development selection: the first 128 examples by SHA-256 rank of canonical
  task/example ID, separately for each task.
- JCommonsenseQA: a fixed Japanese question/options/answer prompt; each choice
  is tokenized at an explicit continuation boundary and scored by conditional
  log probability; length normalization is primary and raw log-probability
  accuracy is retained as a secondary metric.
- GSM8K: fixed `Question`/`Answer` prompt, greedy continuation, 128-token cap,
  the checkpoint-owned evaluation precision, and the dataset repository's
  `####` answer regex.
- Final acknowledgement: `BENCHMARK_FINAL_ACK=BENCH-001-suite-v1`; checked
  outside Hydra.
- Contamination: complete checkpoint-owned train selections, exact/normalized
  whole-document identity, and normalized 48-codepoint shingles.

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

All four independent failed reviews remain visible. Their eleven findings are
repaired without weakening the fixed protocol or complete contamination gate:
cheap context incompatibility precedes scanning, both tasks honor checkpoint
precision, external records are pinned, evaluator/runtime and dirty source
bytes are identity-bound, fixture outputs are golden, generated cache state is
ignored, external JSON cannot enter or alias checkpoint storage, the first
corpus scan uses a bounded linear rolling matcher, full scoped producer identity
invalidates stale scan evidence, and completed suite/corpus/producer-bound scan
evidence is reused across milestones. An extra
repository-wide format diagnostic identified four pre-existing, unrelated
files outside this ticket's diff; the configured Ruff lint gate and all changed
benchmark paths pass, so those files were not rewritten here. The fourth
repair's full gate passes and its exact-head independent review is pending. No
benchmark score from the zero-weight fixture is a model-quality result.
