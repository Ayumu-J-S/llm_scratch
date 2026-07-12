# TOK-001 — Canonical Japanese/English Tokenizer

- PR: [#12](https://github.com/Ayumu-J-S/llm_scratch/pull/12) (repair awaiting independent re-review)
- Branch: `codex/tok-001-canonical-tokenizer`
- Ticket: `TOK-001`
- Hypothesis: a pinned established Japanese/English tokenizer selected by frozen
  corpus evidence and integrated through one local manifest/wrapper will make
  token IDs, vocabulary, special tokens, offline training, streaming, debug,
  model construction, and future generation identity consistent.
- Experiment record: `reports/tokenizers/TOK-001/comparison.md` (CPU R1
  tokenizer selection evidence; no model-quality experiment)
- Started: 2026-07-11T16:27:49Z
- Final verdict: `PENDING RE-REVIEW`
- Final record owner: primary task; exact runtime identity not exposed

## Scope and decision context

- Goal: select, pin, vendor, and use one Japanese-capable tokenizer everywhere
  without importing pretrained model weights or chat behavior.
- In scope: frozen comparison corpus/report; licenses/revisions/file hashes;
  compression/latency/throughput/RSS/model-cost evidence; canonical manifest and
  wrapper; Hydra identity; local/stream/debug/model/process-prefetch integration;
  removal of project BPE training and unused tokenizer backends; offline tests.
- Out of scope: pretrained weights, chat templates, tokenizer research,
  generation implementation, data-split construction, trainer redesign, and
  compatibility aliases.
- Relevant `PHILOSOPHY.md`: random-initialized model capability only; Japanese
  and English first; one DGX Spark boundary; one inspectable component path;
  record provenance, failures, costs, and uncertainty; avoid duplicate backends.
- Baseline commit/run: `a05eb1de5656643757a1c3d98047c98dedea8bfa`;
  training hard-wires the English-only project BPE while standalone streaming
  names a remote Qwen tokenizer.
- Intended evidence: pinned four-candidate R1 report and Pareto decision;
  committed winner artifact/license/manifest; missing/mutable revision and
  missing/moved/mutated artifact failures; offline deterministic IDs/round
  trips; local and process streaming; debug/train parity; streamed batch
  through the conventional model with finite loss.

## Planned candidate set and decision rule

| Candidate | Immutable revision | Vocabulary | Planning status |
| --- | --- | ---: | --- |
| LLM-jp v1 | `llm-jp/llm-jp-13b-v1.0@c3134b3a958b56d443c1484a3d640502637cfbd2` | 50,570 | provisional front-runner; Apache-2.0, Japanese/English/code, byte fallback |
| rinna bilingual | `rinna/bilingual-gpt-neox-4b@803fb7671ac30766ffc6d21139d809b549ee26a3` | 65,536 | compare, but redistribution notice must be resolved |
| LLM-jp v3 | `llm-jp/llm-jp-3-13b@cd3823f4c1fcbb0ad2e2af46036ab1b0ca13192a` | 99,574 | compare improved compression against larger model cost |
| Qwen3 control | `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` | 151,669 | incumbent standalone control; tokenizer files only, no chat template |

Hard gates are immutable/offline load, clear redistribution, deterministic IDs,
valid Unicode encode/round-trip, deterministic malformed rejection, special-ID
and ID-range agreement, streamed finite model loss, and no hidden network/model
weight/chat dependency. Rank eligible candidates on Japanese/English tokens per
byte, throughput/latency/RSS, and vocabulary-driven parameters/logits/state.
Within measurement spread, select the smaller vocabulary.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | handoff (planning) | not exposed by runtime | not exposed by runtime | `a05eb1d`, TOK-001, philosophy/CHECK, current tokenizer paths, official candidate sources | Plan with requested `gpt-5.6-sol` / `ultra` | completed | Defined pinned four-candidate R1/R2 rule, vocabulary costs, artifact contract, direct removals, and offline integration; runtime hid requested identity/mode | Planner handoff in parent task |
| 1 | implementation (phase 1 comparison) | not exposed by runtime | not exposed by runtime | `0b695b8`, planner handoff, exact candidate revisions, TOK-001/philosophy/CHECK | Requested `gpt-5.6-luna` at Extra High/max; freeze corpus, verify allowlisted candidate artifacts/licenses, benchmark, and select without runtime integration | completed | Added 160-document corpus and malformed recipes, exact candidate lock, reproducible fetch/benchmark/report, and six comparison-contract tests; selected LLM-jp v1; rinna excluded before artifact/runtime addition; Qwen3 failed exact Unicode round-trip | `reports/tokenizers/TOK-001/`; `tests/fixtures/tokenizer_comparison/v1/`; `tests/test_tokenizer_comparison.py` |
| 2 | implementation (phase 2 integration) | not exposed by runtime | not exposed by runtime | `ddcef45`, phase-1 LLM-jp v1 decision/report, TOK-001/philosophy/CHECK, phase-2 handoff | Requested `gpt-5.6-luna` at Extra High/max; vendor LLM-jp v1, integrate one canonical offline path, remove old BPE/backends, and validate a streamed model batch | completed | Vendored pinned tokenizer/license/manifest/source notice; added strict manifest/hash/pipeline/probe wrapper; made Hydra train/stream/debug/model/process paths share it; removed BPE training and remote/backend branches/dependencies; added offline process-stream/model evidence | `assets/tokenizers/llm-jp-v1/`; `src/tokenizer/canonical.py`; `tests/test_canonical_tokenizer.py`; 21-test no-skip offline integration and 59-test full suite |
| 1 | precommit audit | not exposed by runtime | not exposed by runtime | phase-2 working tree, TOK-001/philosophy/CHECK | Audit contract and evidence before the implementation commit | FAIL | Found an unused, misleading `install_path` manifest field and under-specified EOS/PAD evidence; confirmed runtime parity/removals/process reconstruction | Audit handoff in parent task |
| 1 | repair | not exposed by runtime | not exposed by runtime | failed precommit audit and phase-2 working tree | Remove the nonportable location claim, test mutable revisions, and make EOS/PAD semantics executable and explicit | completed | Removed `install_path`; re-fingerprinted the package; added refingerprinted `main` rejection for both revisions; proved EOS targets and PAD masking without changing loss behavior | fingerprint `12ccbc02...`; focused `55 passed, 1 existing skip`; full `61 passed, 1 existing skip` |
| 2 | independent review | not exposed by runtime | not exposed by runtime | `759bf4b`, full TOK-001 diff, philosophy/CHECK, PR/model-run evidence | Independent TOK-001 `/review` | FAIL | Tokenizer acceptance and ML checks passed, but the claimed changed-file format check was false because the precommit file list omitted new untracked Python files | `uv run ruff format --check tests/test_canonical_tokenizer.py` failed at line 295 |
| 2 | repair | not exposed by runtime | not exposed by runtime | failed review at `759bf4b` and concrete repair handoff | Format the added test, rerun checks using a base-to-head file list, and preserve the failed verdict in every handoff surface | completed | Ruff reformatted one expression; the corrected base-to-head changed-file format check passes | repair diff in `tests/test_canonical_tokenizer.py`; validation below |
| 3 | independent re-review | not exposed by runtime | not exposed by runtime | `e041c48`, repaired test/record/ledger, current live PR | Fresh TOK-001 `/review` | PASS WITH NOTE | Formatting evidence and execution trail repaired; all acceptance criteria remain satisfied; CUDA/DGX R2 remains deferred without a performance claim | base-to-head format `11 files already formatted`; canonical `17 passed`; full `61 passed, 1 existing skip`; lint/lock/diff/Hydra parity pass |
| 4 | external GitHub review | not exposed by runtime | not exposed by runtime | `9d5cfeaac8bf1df5082ea68f182da586731932ab`, ready PR #12 | Automated review of the ready PR head | FAIL | Found that `add_special_tokens=False` still recognizes literal upstream special-token spellings, allowing corpus text to emit PAD/EOD and other reserved IDs; PAD would be attention-masked as if synthetic padding and raw EOD would be indistinguishable from loader boundaries | [unresolved P2 review thread](https://github.com/Ayumu-J-S/llm_scratch/pull/12#discussion_r3564723322); direct reproduction emitted PAD=4 and EOD=7 |
| 4 | repair | not exposed by runtime | not exposed by runtime | failed external review at `9d5cfea`, full manifest/wrapper/local/stream paths, TOK-001 and CHECK 4.2/4.3/6.1 | Requested Luna / Extra High; reject every manifest-reserved special ID after the one canonical raw-text encode without changing loader EOS insertion | completed; re-review pending | Derived all eight role/token/ID entries from the validated manifest; raw special IDs now fail with explicit context; clean corpus IDs are unchanged; local and serial/process stream paths fail before emitting a batch/sample; deliberate EOS append remains exactly once | canonical `43 passed`; focused offline `81 passed, 1 skipped`; full offline `87 passed, 1 skipped`; frozen wrapper digest `3c1078f7...`; artifact/license hashes unchanged |

## Check selection and verdicts

### Review cycle 2

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: `759bf4b6c9be3bd53ac2ba7c480d3c82d6d8b95e`
- Selected `CHECK.md` sections: 1, 4.1, 4.2, 5.4, 7, 8.2, and 11 TOK-001
- Major sections marked N/A and why: checkpoint/W&B/benchmark/trainer-loop
  behavior does not change; DGX R2 awaits ENV/CFG but no performance claim will
  be inferred from CPU R1.
- Ticket acceptance result: all TOK-001 acceptance criteria demonstrated
- Philosophy alignment: tokenizer selection/integration aligned; research-integrity
  handoff failed because a claimed check result was reproducibly false
- Complexity / change-surface result: direct canonical path and removals passed
- ML-system result: CPU R1 and bounded model smoke passed; CUDA/DGX R2 remains
  explicitly deferred with no performance claim
- Verdict: `FAIL`

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| Medium | Research integrity / formatting evidence | The new canonical test required Ruff formatting, but the record and PR claimed the changed-file format check passed. The precommit shell file list used `git diff` and silently omitted untracked added files. | `uv run ruff format --check tests/test_canonical_tokenizer.py` reported one file requiring formatting at line 295. | Format the file; build the check list from `a05eb1d...HEAD` plus worktree changes; rerun all checks; update record, ledger, and PR; obtain fresh independent re-review. |
| Note | TOK-001/CHECK | No substantive tokenizer correctness defect was found. Offline identity, revision/artifact mutation, Hydra parity, process reconstruction, special/vocab/model agreement, EOS/PAD semantics, and streamed finite loss passed. | Independent `61 passed, 1 skipped`, dedicated `17 passed`; artifact/license/fingerprint checks passed. | Preserve CUDA/DGX R2 deferral and do not make a performance claim. |

## Failed-review handoff

Repair the false formatting result without changing tokenizer semantics. The
handoff includes commit `759bf4b`, the exact failing command/file/line, all
passing acceptance evidence, the requirement to use a base-to-head changed-file
list that includes added files, and the requirement for a fresh independent
re-review.

The later external GitHub review of ready head `9d5cfea` returned `FAIL` for a
separate ML-semantics issue. Although implicit special-token insertion was
disabled, the upstream tokenizer still recognized literal spellings such as
`<pad|LLM-jp>` and emitted reserved IDs. The repair handoff requires one
underlying encode, rejection of all eight manifest-reserved IDs with clear
role/token/ID evidence, unchanged clean-text IDs, preserved explicit loader EOS
append behavior, local and process-stream failure propagation, and a fresh
independent review before restoring a passing verdict.

## Repair result

Precommit audit repair completed before the stable implementation commit. The
canonical manifest no longer claims a fixed filesystem install location;
immutable identity remains the expected manifest fingerprint, exact 40-hex
upstream revisions, and verified artifact bytes. Mutable revision names such as
`main` are rejected even when the mutated manifest is internally re-fingerprinted.
After the first independent review failed, Ruff reformatted
`tests/test_canonical_tokenizer.py`. The corrected formatter check enumerates
Python files from the complete base-to-head diff plus any worktree changes, so
newly added files cannot be silently omitted. The fresh re-review at `e041c48`
returned `PASS WITH NOTE` at that point. The later external cycle invalidated
that final status; cycle 4 is repaired but remains pending independent re-review.

### Re-review cycle 3

- Re-review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: `e041c48b5b020fe9902aa43862590130f9d0fa01`
- Failed finding repaired: yes; the complete base-to-head Python format check
  reports 11 files already formatted, and the PR/model-run preserve the failed
  review and repair trail accurately.
- Ticket acceptance result: pass
- Philosophy alignment: pass
- Complexity / change-surface result: pass
- ML-system result: pass for CPU R1/wiring/identity evidence; CUDA/DGX R2 is a
  documented prerequisite-ordered uncertainty, not an inferred result
- Verdict: `PASS WITH NOTE`
- Note: the 50,570-token vocabulary materially increases model cost, and DGX
  loader headroom, step cost, and UMA behavior remain unmeasured until ENV/CFG.

### External review cycle 4

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: `9d5cfeaac8bf1df5082ea68f182da586731932ab`
- Selected `CHECK.md` sections: 4.2 tokenization, 4.3 packing/boundaries/token
  accounting, and 6.1 objective semantics
- Ticket acceptance result: fail; raw corpus text could emit reserved PAD/EOD
  IDs and violate the recorded boundary and masking semantics
- Philosophy alignment: fail; the visible text-to-token causal chain did not
  distinguish raw content from loader-owned control tokens
- Complexity / change-surface result: repair must stay inside the canonical
  wrapper and must not add an escaping or parallel tokenizer path
- ML-system result: fail; real corpus content could be attention-masked as PAD,
  while raw EOD could masquerade as an explicit document boundary
- Verdict: `FAIL`

#### Finding and repair

| Severity | Area | What was wrong | Repair | Evidence after repair |
| --- | --- | --- | --- | --- |
| P2 | Tokenization / objective semantics | `Tokenizer.encode(..., add_special_tokens=False)` recognized all eight literal upstream special-token spellings. The wrapper range-checked but accepted their IDs, so PAD/EOD from raw documents entered the same path as synthetic padding and loader boundaries. | Build the complete reserved ID to role/token mapping from the already-validated manifest. After the single underlying raw-text encode and ordinary range validation, reject the first emitted reserved ID with role/token/ID context. Do not mutate text or change `StreamLoader`'s explicit `eos_token_id` append. | Every special spelling embedded in raw text fails; PAD/EOD fail at start/end/repeated; local text and serial/process-prefetched streams fail before output; clean Japanese/English/mixed/emoji and all 160 frozen documents emit no reserved IDs and retain digest `3c1078f72957170fd3c7ac94c9d3313b367f3bf243562a693588810f07dfe907`; clean loader output has exactly one appended EOD and no PAD. |

Repair implementation model / mode: not exposed by runtime / not exposed by
runtime (requested Luna / Extra High; the runtime did not display an exact
identifier or reasoning mode). Final verdict remains `PENDING RE-REVIEW`; this
implementation pass does not substitute for the required independent review.

## Final evidence

- Resolved Hydra command/config: phase-1 commands were `uv run python
  src/tokenizer/comparison.py --fetch --fetch-only`, `uv run python
  src/tokenizer/comparison.py`, and `uv run pytest
  tests/test_tokenizer_comparison.py -q`. Phase 2 resolved the training profile
  with `uv run python src/train.py --cfg job --resolve` and the standalone
  stream profile with Hydra `compose(config_name="stream_loader")`; both select
  `assets/tokenizers/llm-jp-v1/manifest.json` and expected fingerprint
  `12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b`.
- Data/tokenizer/model identity: frozen corpus has 160 documents, 20 in each of
  eight strata, SHA-256
  `16d2596017853928346bbf6270fc723d67799b8387080aaa172c42a19804a45a`.
  Candidate lock SHA-256 is
  `56cec8a5e01dac1eeda829057f615d618ce66f729f5d667ea1f7e801b5d7a8f7`.
  Selected `llm-jp/llm-jp-13b-v1.0` at
  `c3134b3a958b56d443c1484a3d640502637cfbd2`; tokenizer JSON SHA-256 is
  `fefc427dff3323dd8a2fd66f392b90a62896db3b11a031463ad0f4c70fb1de9c`.
- Validation and measurements: all eligible artifacts loaded from exact local
  files. Phase 2 vendors only `tokenizer.json` and its Apache-2.0 `LICENSE`, plus
  the repository manifest/source notice; no model weights, model config, or chat
  template are present. The manifest fingerprint is
  `12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b`,
  vocabulary is 50,570 with maximum ID 50,569, and UNK/BOS/PAD/EOS-EOD IDs are
  0/1/4/7. LLM-jp v1 and v3 passed all phase-1 hard gates; Qwen3 failed exact
  Unicode round-trip on 6/160 documents because its normalizer composed NFD
  input; rinna was excluded because the exact revision contains no
  repository-owned license file/tokenizer-specific redistribution notice.
  LLM-jp v1 measured 0.221068 Japanese and 0.210384 English tokens/UTF-8 byte,
  median 6.37M input bytes/s across five warm CPU passes, 50,570 vocabulary,
  80.6 MiB fresh-process VmHWM, and zero unknown tokens/exceptions/round-trip
  failures. LLM-jp v3 improved Japanese/English compression by 23.7%/7.25%,
  but its 99,574 vocabulary raises vocabulary-driven parameters from 38,888,330
  to 76,572,406, FP32 training state from 593.4 MiB to 1,168.4 MiB, BF16 batch
  logits from 395.1 MiB to 777.9 MiB, and its 5.73M bytes/s median lay below the
  full LLM-jp v1 five-pass range. The predeclared promotion rule therefore
  selected LLM-jp v1; both eligible candidates remain on the Pareto frontier.
  Phase-2 network-offline validation used `UV_NO_SYNC=1`, `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`, and `DATASETS_OFFLINE=1`: the focused canonical,
  stream-loader, streaming-dataset, and train-streaming integration passed 55
  tests with one existing opt-in remote-dataset skip. It includes fixed Japanese,
  English, mixed, symbols, and emoji IDs and round trips; missing and mutable
  revisions; missing/moved/mutated artifact bytes; strict UTF-8/range rejection;
  source-access guards; parent/process revalidation; local/debug/train ID parity;
  process-prefetched local JSONL; model vocabulary dimension; and finite
  causal-LM loss. The full offline suite passed 61 tests with the same one
  existing opt-in remote-dataset skip; the dedicated canonical integration file
  passed all 17 tests with no skips. `uv run ruff check .`, scoped changed-file
  `ruff format --check`, `uv lock --check`, and `git diff --check` passed.
  EOS/EOD ID 7 is appended at document boundaries and is an ordinary next-token
  target. Packed batches contain no padding; PAD ID 4 configures the embedding
  and attention padding mask if it appears, while current cross-entropy has no
  `ignore_index` behavior. A real bounded CPU entrypoint run used the canonical
  manifest, `tests/fixtures/tiny_corpus.jsonl`, sequence length 8, batch size 8,
  one epoch/12 training batches, W&B disabled, and a 16-wide one-layer model; it
  completed with finite train/validation losses 10.810500/10.592047.
  Cycle-4 repair adds an explicit raw/control-token boundary: all eight
  manifest special IDs are rejected from wrapper-encoded text, while
  `StreamLoader` still appends EOD ID 7 directly after successful clean-text
  encoding. Dedicated canonical validation now passes 43 tests; focused offline
  tokenizer/stream/data/train validation passes 81 tests with the one existing
  opt-in remote-dataset skip; the full offline suite passes 87 tests with that
  same skip. Frozen clean IDs retain SHA-256
  `3c1078f72957170fd3c7ac94c9d3313b367f3bf243562a693588810f07dfe907`.
- Performance/resource result: CPU R1 comparison required. Phase-2 smoke ran on
  Linux aarch64 with `torch 2.10.0+cpu`; CUDA was unavailable
  (`torch.version.cuda=None`, zero devices), so no CUDA/DGX R2 or performance
  claim is made. The
  default 384-wide untied embedding/head grows from 393,728 vocabulary-driven
  parameters at the former configured 512-token target to 38,888,330 at 50,570
  tokens, a delta of 38,494,602; the full current default model grows from
  11,040,512 to 49,535,114 parameters.
- Failed attempts retained at: execution timeline and comparison report. The
  first invocation using `uv run python -m tokenizer.comparison` failed because
  this non-packaged repository does not put `src` on the Python module path;
  the direct committed script command is used instead. In phase 2, the first
  focused command used bare `pytest` and failed because the executable is only
  available through `uv run`; the corrected command used `uv run pytest`. The
  next collection failed because `scripts` is not an import package; the parity
  test now loads the debug script with `runpy`. The first canonical-fixture pass
  had five failures from old one-character/one-token quota assumptions and an
  unbound method mock; the fixtures now use real canonical lengths/IDs and the
  direct serializable config. The first real-entrypoint smoke mistakenly used
  the full local corpus and a 6,877-batch epoch; it was manually stopped after
  3,550 batches because it was not a bounded smoke and yields no result claim.
  The corrected 12-batch fixture command completed, after which focused and full
  suites passed. During cycle 4, the first base-to-head formatting invocation
  included deleted Python paths and therefore failed before completing the
  gate; rerunning with `--diff-filter=ACMR` checked the 11 existing changed
  Python files successfully.
- Known trade-offs: the selected established tokenizer greatly enlarges the
  current untied embedding/LM head; its measured bilingual compression and
  smaller cost than the other eligible candidate justified selection, but the
  38.5M-parameter delta is material.
- Unresolved risks: CUDA/DGX R2 step cost and loader headroom remain unmeasured,
  and the existing opt-in public-Hugging-Face dataset integration was not run in
  the offline phase-2 validation. CPU R1 tokenization throughput and a bounded
  CPU model smoke are not claims about end-to-end DGX training supply.
- Human decision requested: none until a fresh independent review evaluates the
  cycle-4 repair; the PR must not return to ready/merge consideration on this
  implementation pass alone.

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | planning | Produced pinned candidates, primary-source links, cost model, selection gates, direct API/removal map, and test/R2 plan | Requested Sol/Ultra identity/mode unavailable; proposed R2 cannot precede ENV/CFG | Ticket order, philosophy, local cache, official repositories, current code | plan accepted with dependency-order adjustment |
| not exposed by runtime / not exposed by runtime | implementation phase 1 | Produced an exact artifact lock, frozen multilingual/Unicode corpus, operational and cost measurements, hard-gate evidence, deterministic selection, and focused tests | Requested Luna Extra High/max identity/mode was unavailable; initial module-form command did not match the repository's non-packaged `src` layout and was corrected to a direct script command | Planner handoff, candidate 40-hex revisions, official cards/source licenses, phase boundary excluding integration | comparison completed; LLM-jp v1 selected |
| not exposed by runtime / not exposed by runtime | implementation phase 2 | Reduced all runtime consumers to one strict offline wrapper/config, preserved pre-source failure ordering across process prefetch, removed obsolete backends/dependencies, and demonstrated fixed IDs plus finite streamed model loss | Requested Luna Extra High/max identity/mode was unavailable; first test commands/import and old character-token fixture assumptions required correction | Phase-1 winner/hash/license evidence, explicit direct-removal handoff, ticket acceptance criteria, real vendored artifact | integration completed; first independent review failed on handoff evidence, then repaired |
| not exposed by runtime / not exposed by runtime | independent review | Verified every substantive TOK-001 acceptance criterion, reran the real bounded entrypoint and offline suite, checked hashes/fingerprint/provenance, and retained the CUDA/DGX deferral | Caught that the changed-file formatting claim was false because new untracked files had been omitted from the precommit file list | Stable commit, full diff, philosophy/CHECK, live PR and model-run evidence | `FAIL`; concrete one-file repair and re-review handoff |
| not exposed by runtime / not exposed by runtime | repair | Applied the minimal formatter-only change and corrected the validation file-selection method without changing tokenizer semantics | Initial implementation validation used a worktree-only `git diff` list that cannot see untracked additions | Exact failed command/file/line and reviewer handoff | repair completed; re-review `PASS WITH NOTE` |
| not exposed by runtime / not exposed by runtime | independent re-review | Verified the formatter repair, full failed-review trail, complete base-to-head file list, all offline tests, lock/diff, Hydra parity, and clean worktree | Exact model/mode remained hidden; CUDA/DGX R2 remains unavailable in the current CPU runtime | Repair commit `e041c48`, failed-review handoff, live PR, philosophy/CHECK | `PASS WITH NOTE`; ready for human review after final docs parity audit |
| not exposed by runtime / not exposed by runtime | external GitHub review | Found that literal special-token spellings bypassed the intended raw/control-token boundary and connected the defect to PAD masking and EOS semantics | Exact model/mode was not exposed; the finding arrived only after the earlier passing re-review | Ready head `9d5cfea`, manifest special IDs, canonical encode line, model PAD behavior, loader EOS append | `FAIL`; concrete P2 repair required |
| not exposed by runtime / not exposed by runtime | repair cycle 4 | Kept one canonical encode path, derived the complete reserved mapping from the validated manifest, preserved explicit loader EOS, and added end-to-end local/process failure and frozen-corpus identity evidence | Requested Luna / Extra High identity/mode was unavailable; fresh independent review remains required | Exact external finding, manifest, wrapper, stream/local consumers, frozen digest | repair completed; verdict pending re-review |

## Ledger update

- [x] Added ticket/PR row to `docs/model-runs/README.md`.
- [x] Recorded both implementation invocations, both failed reviews, and all
  three repair passes under the runtime-exposed identity/mode values.
- [ ] Fresh independent review must evaluate cycle 4 and then update the live PR,
  final verdict, and docs parity before the PR can return to ready state.
