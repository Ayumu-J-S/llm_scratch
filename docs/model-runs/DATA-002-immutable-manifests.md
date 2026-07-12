# DATA-002 — Immutable Manifests and Disjoint Splits

- PR: [#13](https://github.com/Ayumu-J-S/llm_scratch/pull/13) (draft during dependency integration)
- Branch: `codex/data-002-immutable-manifests`
- Ticket: `DATA-002`
- Hypothesis: immutable source manifests plus deterministic document identity
  and split assignment can make provenance and train/validation separation
  auditable before source access or training without adding per-sample hot-path
  checksum work.
- Started: 2026-07-11T17:38:46Z
- Current verdict: integration review pending; the earlier isolated head was PASS WITH NOTE
- Final record owner: primary task; exact runtime identity not exposed

## Scope and decision context

- Goal: make data identity, usage terms, checksums, document IDs/content hashes,
  split membership, and benchmark boundaries reproducible and fail closed.
- In scope: source/revision/config/split/text-ID/license/terms metadata; SHA-256
  for local/downloaded content; stable document IDs and normalized content
  hashes; deterministic split assignment/fingerprints; explicit memorization
  smoke fixture; benchmark access guard.
- Out of scope: final legal judgment, full semantic deduplication, benchmark
  scoring, raw-data W&B uploads, and unrelated training-loop redesign.
- Relevant `PHILOSOPHY.md` principles: visible text-to-training causal chain;
  data provenance and split construction recorded; training and benchmark test
  data separated; reproducible evidence; smallest coherent direct path; one-DGX
  boundary.
- Baseline commit/run: `a05eb1de5656643757a1c3d98047c98dedea8bfa`.
- Intended evidence: manifest/checksum mutation fails before source access;
  zero train/validation overlap by document ID and normalized content hash;
  split membership invariant under input reorder/prefetch; explicit same-corpus
  smoke only; benchmark guard; bounded R1 startup/hot-path measurement.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | handoff (planning attempt 1) | not exposed by runtime | not exposed by runtime | `e0705b0`, full parent context, DATA-002/philosophy/CHECK/current data paths | Requested `gpt-5.6-sol` / Ultra plan | interrupted | Produced no handoff after repeated finalize requests; stopped to avoid blocking the roadmap | collaboration attempt retained in parent task |
| 0 | handoff (planning attempt 2) | not exposed by runtime | not exposed by runtime | `e0705b0`, clean minimal context, same bounded request | Retry requested `gpt-5.6-sol` / Ultra plan | interrupted | Produced no handoff after repeated finalize requests; stopped without repository changes | collaboration attempt retained in parent task |
| 0 | handoff (planning attempt 3) | not exposed by runtime | not exposed by runtime | `e0705b0`, DATA-002/philosophy/CHECK, loader/cache/config/tests | Requested `gpt-5.6-sol` / Ultra plan with immediate bounded output | completed | Defined strict manifest/index schema, stable text identity, content-based deterministic split, preflight/hot-path separation, overlap/smoke/benchmark guards, source rules, modular merge seams, R1 evidence, and exact tests | planner handoff in parent task |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `be6470e`, accepted plan and live draft PR | Requested Luna / Extra High or maximum; bounded identity/manifest/split contract, fixture builder, one loader seam, tests, and R1 evidence | completed | Added strict canonical schemas/fingerprints, bounded Unicode identity, stable IDs, deterministic content-hash splits, purpose/benchmark guards, checksum-validating preflight, immutable resolved documents, manifest loader integration, and committed bilingual/smoke/benchmark fixtures | 57 passed, 3 explicit network integration skips; initial checks passed, but the precommit audit below found blocking authority/lifecycle/integration defects |
| 1 | precommit audit | not exposed by runtime | not exposed by runtime | uncommitted initial implementation | Review acceptance, philosophy, and applicable CHECK sections before implementation commit | FAIL | Hydra fields could grant reserved benchmark access; the PyTorch dataset repeated preflight each epoch; train/validation loaders lacked a cross-source overlap check; a purpose string authorized mutable local text; package paths could escape; URL cache identity/concurrency was not process-safe | Exact reproductions and repair handoff below |
| 2 | repair | not exposed by runtime | not exposed by runtime | uncommitted FAIL handoff and complete repository context | Close all six audit blockers and add exact regressions | completed | Training hardcodes benchmark denial; resolved manifests live on the dataset; cross-loader ID/content overlap fails before preview/model; default smoke is pinned; package paths fail closed; URL+SHA cache uses per-key Linux locks and preserves other-process temporaries | 62 passed, 3 explicit network skips; six exact repair regressions pass; Ruff/format/lock/diff/Hydra pass |
| 2 | independent `/review` | not exposed by runtime | not exposed by runtime | `b4bcd7f4b86a8477ff94670cff2f0b387bfb0da8` | Mandatory independent review against DATA-002, philosophy, and selected CHECK sections | FAIL | Training still honored `require_manifests: false` and empty resolved mappings; canonical smoke contained Japanese outside the committed English-project BPE; documented streaming budgets guaranteed fixture exhaustion | Review handoff supplied by primary task; exact model/mode unavailable |
| 3 | repair | not exposed by runtime | not exposed by runtime | stable review-failed commit `b4bcd7f4` and exact findings | Requested Luna / Extra High or maximum; close only the three review findings and run actual bounded workflows | completed | Training forces manifest-only streaming and rejects explicit false/empty mappings; smoke is pinned English compatible with the current BPE; fixture streaming budgets use `max`; obsolete mutable-path override removed | 64 passed, 3 explicit network skips; actual tokenizer+smoke and bilingual streaming one-epoch CPU commands completed offline |
| 3 | re-review | not exposed by runtime | not exposed by runtime | `96b2227054177fd245d7c3956392b224f542c6a7` | Independent DATA-002 `/review` of every prior finding and the complete ticket | PASS WITH NOTE | All authority, overlap, tokenizer-compatibility, workflow, split, lifecycle, and documentation findings closed; DATA-004 large-corpus work remains explicitly deferred | 64 passed, 3 explicit network skips; focused 5 passed; two actual bounded CPU workflows; quality/Hydra checks pass |
| 4 | dependency integration implementation | not exposed by runtime | not exposed by runtime | DATA-002 head `38bc94cfe2fb2a20d010174bc98d693324aa7101`; `origin/main` `073514455a6448c2471c75e433a0fc3c2021b2bc` after PRs #10-#12 | Requested Luna / Extra High; normally merge current main, preserve DATA-001/TOK-001 invariants, compose canonical tokenizer with manifest authority, rerun real bounded workflows, and leave draft for fresh independent review | completed; independent review pending | Normal merge result `94faa2175b940ea4f373a9cf7b03e4606f98b32c`; canonical-only tokenizer validation remains before manifest source access; DATA-001 stride, quota boundaries, accounting, and process state remain; DATA-002 adds one-time preflight, disjointness, cache identity, and thread-only manifest prefetch | 140 passed, 1 explicit HF opt-in skip; 20 exact integration regressions; resolved Hydra and debug; canonical memorization/streaming CPU epochs; bounded R1; Ruff/format/lock/diff pass |

## Current merge authorization

- On 2026-07-12, the human explicitly authorized the bounded set of repository
  PRs #10 through #15 for guarded agent self-merge after human-merging the
  bootstrap policy PR #16.
- Authorization alone is not a merge verdict. PR #13 remains a draft during
  this integration cycle and may not be marked ready or merged until the exact
  integrated head passes an independent review, all actionable findings are
  repaired and re-reviewed, expected-check and review-thread inventories are
  current, and the final no-drift audit required by `PHILOSOPHY.md` succeeds.

## Dependency integration evidence (cycle 4)

- Integration method: normal merge, not rebase or force push. The DATA-002
  branch started at `38bc94cfe2fb2a20d010174bc98d693324aa7101` and merged
  `origin/main` `073514455a6448c2471c75e433a0fc3c2021b2bc`, which already
  contains human-merged PRs #10, #11, and #12.
- Integration implementation commit:
  `94faa2175b940ea4f373a9cf7b03e4606f98b32c`.
- Conflict policy: preserve the canonical two-field tokenizer config and raw
  reserved-control rejection from TOK-001; preserve stride-`L` over `L+1`, EOS
  quota boundaries, unique target accounting, and parent-visible process
  prefetch state from DATA-001; add DATA-002 manifest identity only at the
  source/dataset/training seams.
- Offline suite: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
  uv run --group dev pytest -q` reported 140 passed and one explicit opt-in HF
  integration skip. The skipped test requires
  `RUN_HF_DATASET_INTEGRATION=1`; no manifest correctness test skipped.
- Exact integration regressions: 20 focused tests passed across manifest
  mutation/checksum/split failures, document-ID and normalized-content overlap,
  reserved benchmark and Hydra-authority denial, one preflight across two
  traversals, canonical-tokenizer-before-source ordering, reorder/prefetch
  membership, URL cache corruption, cross-process URL+SHA locking, manifest
  process-prefetch rejection, DATA-001 process accounting, stride/carry, quota
  boundary behavior, and a finite tiny decoder update.
- Resolved execution paths: default and `data.mode=streaming` Hydra jobs both
  resolve to the same canonical tokenizer fingerprint
  `12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b`.
  The standalone debug command emitted one `(1, 8)` inputs/labels batch with a
  true shift check from the train manifest selection.
- Actual canonical memorization workflow: a one-epoch CPU run at sequence
  length 4, batch size 1, width 16, one layer, W&B disabled completed 13 train
  and validation windows with finite `train_loss=10.722412` and
  `val_loss=10.485553`; its final checkpoint was 6,695,029 bytes under `/tmp`.
- Actual disjoint streaming workflow: the analogous one-epoch CPU run at
  sequence length 8 completed without quota exhaustion with finite
  `train_loss=10.946695` and `val_loss=10.821602`; its final checkpoint was
  6,695,285 bytes under `/tmp`. The run uses the committed 11-document train and
  9-document validation selections and the same canonical tokenizer.
- Bounded R1 fixture measurement: 50 train-manifest preflights had 0.375 ms
  median (0.367-0.798 ms). Dataset construction, including canonical tokenizer
  validation plus the sole manifest preflight, took 109.026 ms. Thirty complete
  canonical/thread-prefetched traversals had 123.965 ms median, 14 windows and
  112 next-token targets each; the identity-hash spy stayed at exactly 20 calls
  after construction and 32 traversals. Process `ru_maxrss` was 336,496 KiB.
  This tiny CPU fixture is lifecycle evidence only and makes no real-data, GPU,
  DGX, or throughput claim.
- Quality gates: the 11 Python files changed relative to current main pass Ruff
  check and format check; `uv lock --check`, staged/unstaged diff checks, and
  all three config/workflow commands pass. A repository-wide format check still
  identifies three pre-existing main-branch files; they are unchanged by this
  PR and were not reformatted out of scope.
- Current handoff: implementation is complete but no verdict is claimed for the
  integrated head. PR #13 stays draft for a fresh independent `/review` against
  DATA-002, `PHILOSOPHY.md`, and applicable `CHECK.md` sections.

## Check selection and verdicts

### Review cycle 1

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: uncommitted precommit implementation
- Selected `CHECK.md` sections: 1, 4.1, 4.4, 7, 8.2, and 11 DATA-002
- Major sections marked N/A and why: GPU/model/optimizer/checkpoint sections; this
  ticket changes data identity and startup/loader behavior only.
- Ticket acceptance result: FAIL before repair.
- Philosophy alignment: FAIL; config-controlled reserved benchmark access and
  mutable same-corpus authority violated research-integrity boundaries.
- Complexity / change-surface result: FAIL; identity was correct in isolation
  but not enforced at the actual dataset/training/cache lifecycle seams.
- ML-system result: FAIL; repeated epoch hashing and absent cross-loader overlap
  validation made the claimed runtime invariant false.
- Verdict: FAIL

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| P0 | benchmark authority | `StreamLoader` trusted Hydra `access` and `allow_reserved_benchmark`, allowing reserved text into the training loader | reserved fixture loaded when both fields were set | hardcode training access and reject authority fields |
| P0 | split isolation | independently valid train/validation loaders could select the same membership | no union-level ID/content comparison existed | validate all resolved train sources against all validation sources before preview/model |
| P1 | lifecycle | `StreamingTokenDataset.__iter__` constructed and preflighted a new loader every epoch | normalized-content hashing repeated on each traversal | retain immutable resolved manifests on dataset construction and inject them into iterators |
| P0 | smoke authority | a user-set purpose string authorized arbitrary mutable `data/inputLearnText.txt` | local-text path bypassed manifest identity | replace the mode with the pinned memorization manifest |
| P1 | package paths | absolute and traversal source/index/artifact paths were accepted | schema path strings were resolved without containment | require package-relative contained paths in loader and builder |
| P1 | cache integrity | URL-only keying, eager temp cleanup, and in-process coordination did not protect concurrent processes | same URL could alias content identities; a new process could delete another process's temp | key by URL+SHA, preserve foreign temporaries, and add a per-key Linux file lock |

## Failed-review handoff

Repair the six findings above without expanding into DATA-003/DATA-004. Preserve
the strict manifest core, add exact exploit/lifecycle regressions, and rerun the
full evidence suite before requesting independent re-review.

### Independent review cycle 2

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: `b4bcd7f4b86a8477ff94670cff2f0b387bfb0da8`
- Verdict: FAIL

| Severity | Area | Finding | Required repair |
| --- | --- | --- | --- |
| P0 | production authority | `require_manifests: false` remained a Hydra-controlled bypass and empty resolved train/validation maps passed the disjoint check | force manifests in training, reject explicit false, and reject empty production mappings |
| P1 | canonical smoke | pinned smoke text contained Japanese characters absent from the currently committed English-corpus BPE | preserve immutable smoke identity with text encodable by the canonical branch tokenizer |
| P1 | documented workflow | `data.mode=streaming` requested 1,000,000/100,000 tokens from a 20-document fixture and exhausted by construction | use a valid fixture horizon and execute the documented path with a compatible tokenizer |

### Independent re-review cycle 3

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: `96b2227054177fd245d7c3956392b224f542c6a7`
- Selected `CHECK.md` sections: 4.1, 4.4, 8.2, and 11 DATA-002
- Ticket acceptance result: PASS; every real training input is manifest-backed,
  train/validation identity sets are non-empty and disjoint, split membership is
  deterministic, and same-corpus use is restricted to the pinned smoke profile.
- Philosophy alignment: PASS; provenance and benchmark boundaries fail closed,
  failed attempts remain visible, and no deferred real-data claim is implied.
- Complexity / change-surface result: PASS; the manifest core and one existing
  loader/training seam remain within DATA-002.
- ML-system result: PASS WITH NOTE; actual bounded canonical smoke and streaming
  epochs completed with finite losses and checkpoints. Real HF inventory,
  bounded large-corpus loading, DGX behavior, and real-source throughput remain
  the explicit DATA-004 boundary.
- Verdict: PASS WITH NOTE

## Repair result

All six findings were repaired in cycle 2. Focused regressions reproduce the old
benchmark bypass, two-epoch hash repetition, same-selection overlap, mutable
smoke authority, path escape, and cross-process cache race; all now fail closed
or exhibit the required one-time behavior. Independent re-review remains
pending, so no PASS is claimed.

Cycle 3 repaired the independent findings. Manifest-only streaming is now code
authority, the fingerprinted CC0 smoke uses current-BPE-compatible English
text, and both fixture horizons are `max`. Actual offline one-epoch smoke and
streaming commands completed. Independent re-review returned PASS WITH NOTE;
the note is limited to work explicitly deferred to DATA-004.

## Isolated-head evidence (cycles 1-3)

The following evidence is retained as the historical record for the original
isolated branch. Cycle 4 above supersedes its tokenizer commands and dependency
merge assumptions for the current PR head.

- Resolved Hydra command/config: `uv run python src/train.py --cfg job
  --resolve`; the default `data.mode: memorization_smoke` names the committed
  manifest and fingerprint directly. Streaming uses manifest-backed, distinct
  train/validation selections.
- Data identity:
  - bilingual manifest:
    `47cca88c4a5595e27eb5d60d99918fb77c30b23f7c0ae98024153f25e14ffc19`;
    dataset:
    `d34675a2697a507b6ae2d499e772b1a6ded264000b619b42b94731555c335a2e`;
    train/validation:
    `1cf2ddecf55e08e38ab95e2bac0d1ce17ded7df6b80f4d8e9cf6864449e1fe7e` /
    `fec55d7aa28b1250e76e79d69d6047b5cb4fa909664fbeed6f39ded132bdf347`;
    11 train and 9 validation documents with zero ID/content-hash overlap.
  - memorization-smoke manifest:
    `00c3797a7d0eda13950fd699a60c45fcd388829f016479caaeb369438767bd31`;
    dataset:
    `21c82c527fb8fafbbba4e2ea2bdf7057aed48ec8ac995a369a356747c70cd05b`;
    source SHA-256:
    `15f042efbbb7e7b3fd31b8027cc7f4feb7e94967d494ea7da3e6f2690ee0181d`.
  - reserved benchmark fixture:
    `a12e307b8c3817efe956d8d37c55edcde56f30cf695e83a9e60155ff5949eb79`.
- Provenance: repository-authored fixtures are CC0. The intended
  `hotchpotch/fineweb-2-edu-japanese` source records its exact commit, ODC-By
  1.0 card license, Common Crawl terms, intended config/split, and missing
  139-training-shard inventory blocker in
  `data/manifests/fineweb-2-edu-japanese.blocked.md`; it is deliberately not
  runnable, and its test shard is excluded from the intended inventory.
- Validation: `uv run pytest -q` reported 64 passed and 3 explicit opt-in
  external-download skips. Manifest correctness tests do not skip. Ruff checked
  and format-checked every tracked/untracked changed Python file;
  `uv lock --check`, `git diff --check`, and Hydra resolution passed.
- Mutation/invariant evidence covers schema/fingerprint,
  revision/path/artifact/index/source/split changes, exact Unicode identity,
  duplicate ID/content, reorder, seed/prefetch membership, overlap, purpose and
  benchmark guards, URL cache corruption, unchanged text/metadata, preflight
  once, and existing loader/training behavior.
- Exact repair regressions: six focused tests passed for benchmark-authority
  denial, one preflight across two real dataset traversals, same-selection
  cross-loader rejection, pinned smoke identity, package path containment, and
  SHA-keyed cross-process cache locking.
- Independent-review regressions prove explicit `require_manifests: false`
  fails in `build_streaming_dataloader`, absent authority is forced true, empty
  production train/validation mappings fail before iteration/model, canonical
  fixture budgets are `max`, and the pinned smoke encodes under a project BPE
  trained from the committed tokenizer corpus.
- Actual bounded canonical smoke: `src/train_tokenizer.py` built the 512-token
  project BPE into a fresh `/tmp` directory. A one-epoch CPU run with sequence
  length 4, batch 1, 16-wide/1-layer model, W&B disabled, and temporary
  checkpoints completed 34 train/validation windows with finite losses
  (`train_loss=6.337600`, `val_loss=6.004064`) and wrote an 87,413-byte final
  checkpoint. The smoke manifest fingerprint was logged before tokenization.
- Actual documented streaming seam: a temporary 203-token project BPE trained
  only on the committed bilingual fixture, followed by a one-epoch CPU run with
  sequence length 8, batch 1, 16-wide/1-layer model and W&B disabled, completed
  without quota exhaustion (`train_loss=5.441960`, `val_loss=5.361721`) and
  wrote a 46,837-byte final checkpoint. All artifacts remained under `/tmp`.
- R1 bounded fixture measurement after repair: 50 train-selection preflights
  had 0.396 ms median (0.391-0.862 ms range) and process `ru_maxrss` 237,404
  KiB, dominated by imported Torch/runtime. Actual `StreamingTokenDataset`
  construction, including its sole preflight, took 0.452 ms. Thirty subsequent
  complete traversals had 0.363 ms median (0.355-0.765 ms), 41 windows, and
  903,574 next-token targets/s. The identity hash spy stayed at exactly 20
  source documents after construction and after both tested epochs. These
  tiny-fixture results make no real-data, GPU, or DGX performance claim.
- Failed attempt: the first builder command failed with
  `ModuleNotFoundError: data`; the repository is not installed as a package.
  The documented `PYTHONPATH=src` command and rerun succeeded.
- Trade-off: local/URL JSONL documents are materialized after one preflight so
  the hot path never rehashes or resplits. This bounded adapter is not the final
  large-corpus source path. HF manifests are identity-validatable but fail at
  runtime until DATA-004 adds the full inventory and bounded adapter.
- Cache boundary: manifested URL entries are keyed by URL plus expected SHA-256
  and installed atomically under a per-key Linux file lock. The repair does not
  claim global cross-key cache-capacity serialization.
- Merge seam at that time: `src/data/stream_loader/loader.py`, the two Hydra
  configs, and `src/train.py` overlapped then-unmerged DATA-001/TOK-001 sibling
  work. Cycle 4 resolves that seam while the portable core remains isolated in
  `src/data/{identity,manifests,splits}.py`.
- Unresolved risks: no R2/DGX or consequential real-source run; no real-data
  throughput claim. The canonical tokenizer and both bounded training seams
  were exercised with temporary artifacts. Legacy direct sources remain usable
  for component tests, while the training entrypoint forces manifest-only data.
- Current merge decision is governed by the explicit cycle-4 authorization and
  all guarded self-merge gates; the earlier model verdict alone is not merge
  authority for the integrated head.

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | planning attempts 1-2 | No observable planning output | Both attempts remained active without returning a handoff and were deliberately interrupted | Repeated explicit finalize requests | failed operationally; no repository mutation |
| not exposed by runtime / not exposed by runtime | planning attempt 3 | Produced an exact schema, source rules, deterministic content-based split, preflight/runtime boundary, modular integration plan, tests, and R1 review contract | Requested Sol/Ultra identity and mode were unavailable; real HF shard evidence remains to be verified during implementation | Minimal bounded context and explicit current-main merge constraint | plan accepted for implementation |
| not exposed by runtime / not exposed by runtime | initial implementation | Implemented the bounded stdlib core, fail-closed source identity, one loader seam, committed fixtures, mutation/invariance tests, and honest R1 evidence without taking later tickets | Requested Luna/Extra High identity and mode were unavailable; first builder invocation omitted `PYTHONPATH`; large HF streaming remains blocked | Accepted planner handoff, exact acceptance tests, narrow sibling-branch seam, and verified source facts | implementation completed; first reviews failed |
| not exposed by runtime / not exposed by runtime | precommit audit and repair | Found authority, lifecycle, cross-loader, mutable-smoke, package-path, and process-cache defects that unit-level identity tests missed; repaired each with exact regressions | Runtime did not expose an independent model/mode; the repair still missed a training-entrypoint bypass and runnable-workflow defects | Concrete exploit reproductions and actual two-epoch dataset behavior | initial review FAIL; repair required another cycle |
| not exposed by runtime / not exposed by runtime | independent review 2, repair 3, and re-review 3 | Identified remaining production authority, tokenizer compatibility, and documented-horizon defects; repair added training-entrypoint guards and actual offline workflow evidence; re-review closed every finding | Previous repair stopped at component tests and did not execute the canonical smoke/streaming commands | Stable reviewed commit, exact failing cases, current BPE corpus, bounded CPU commands, and full acceptance matrix | PASS WITH NOTE; DATA-004 boundary retained |
| not exposed by runtime / not exposed by runtime | dependency integration implementation | Reconciled the accepted manifest contract with current canonical-tokenizer and causal-stream semantics; retained all three tickets' invariants and executed both real bounded training modes | Exact requested Luna / Extra High identity and mode were not exposed; independent review is still required before any integrated verdict | Stable sibling merge commits, predeclared conflict policy, exact focused regressions, and actual canonical runs | implementation complete; integrated review pending |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated implementation, two failed reviews, three repair/integration
  attempt counts, and the earlier passing independent re-review.
- [ ] Reconcile the integrated independent-review verdict and exact reviewed
  head after cycle 4 review.
