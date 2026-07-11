# TOK-001 — Canonical Japanese/English Tokenizer

- PR: [#12](https://github.com/Ayumu-J-S/llm_scratch/pull/12) (draft)
- Branch: `codex/tok-001-canonical-tokenizer`
- Ticket: `TOK-001`
- Hypothesis: a pinned established Japanese/English tokenizer selected by frozen
  corpus evidence and integrated through one local manifest/wrapper will make
  token IDs, vocabulary, special tokens, offline training, streaming, debug,
  model construction, and future generation identity consistent.
- Experiment record: `reports/tokenizers/TOK-001/comparison.md` (CPU R1
  tokenizer selection evidence; no model-quality experiment)
- Started: 2026-07-11T16:27:49Z
- Final verdict: in progress
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
  committed winner artifact/license/manifest; missing/moved/mutated failures;
  offline deterministic IDs/round trips; local and process streaming; debug/train
  parity; streamed batch through the conventional model with finite loss.

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
| 2 | implementation (phase 2 integration) | pending | pending | phase 1 winner and evidence | Vendor LLM-jp v1, integrate one canonical offline path, remove old BPE/backends, and validate streamed model batch | pending | Not started; deliberately excluded from phase 1 | pending |
| 1 | review | pending | pending | pending implementation commit | Independent TOK-001 review | pending | No verdict claimed | pending |

## Check selection and verdicts

### Review cycle 1

- Review model / mode: pending
- Commit reviewed: pending
- Selected `CHECK.md` sections: 1, 4.1, 4.2, 5.4, 7, 8.2, and 11 TOK-001
- Major sections marked N/A and why: checkpoint/W&B/benchmark/trainer-loop
  behavior does not change; DGX R2 awaits ENV/CFG but no performance claim will
  be inferred from CPU R1.
- Ticket acceptance result: pending
- Philosophy alignment: pending
- Complexity / change-surface result: pending
- ML-system result: pending
- Verdict: pending

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |

## Failed-review handoff

`N/A — review pending.`

## Repair result

`N/A — implementation/review pending.`

## Final evidence

- Resolved Hydra command/config: runtime integration remains pending; phase 1
  commands were `uv run python src/tokenizer/comparison.py --fetch --fetch-only`,
  `uv run python src/tokenizer/comparison.py`, and
  `uv run pytest tests/test_tokenizer_comparison.py -q`.
- Data/tokenizer/model identity: frozen corpus has 160 documents, 20 in each of
  eight strata, SHA-256
  `16d2596017853928346bbf6270fc723d67799b8387080aaa172c42a19804a45a`.
  Candidate lock SHA-256 is
  `56cec8a5e01dac1eeda829057f615d618ce66f729f5d667ea1f7e801b5d7a8f7`.
  Selected `llm-jp/llm-jp-13b-v1.0` at
  `c3134b3a958b56d443c1484a3d640502637cfbd2`; tokenizer JSON SHA-256 is
  `fefc427dff3323dd8a2fd66f392b90a62896db3b11a031463ad0f4c70fb1de9c`.
- Validation and measurements: all eligible artifacts loaded from exact local
  files. LLM-jp v1 and v3 passed all phase-1 hard gates; Qwen3 failed exact
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
  Validation after report generation: `uv lock --check` passed; `uv run ruff
  check .` passed; `uv run pytest -q` passed with 45 tests and 3 skips;
  `git diff --check` passed; scoped Ruff format check passed. Repository-wide
  `ruff format --check .` remains red on six pre-existing files outside this
  phase's diff (`scripts/debug_stream_loader.py`, model files, `bpe.py`,
  `train.py`, and `trainer.py`), which were not reformatted as unrelated churn.
- Performance/resource result: CPU R1 comparison required. R2 BF16/DGX cannot
  run until ENV/CFG establish CUDA and a real profile; no performance conclusion
  will be made without it.
- Failed attempts retained at: execution timeline and comparison report. The
  first invocation using `uv run python -m tokenizer.comparison` failed because
  this non-packaged repository does not put `src` on the Python module path;
  the direct committed script command is used instead.
- Known trade-offs: established multilingual tokenizers greatly enlarge the
  current untied embedding/LM head; compression must repay that cost
- Unresolved risks: phase-2 artifact vendoring and notices, integration
  correctness, real streamed-batch/model behavior, and DGX R2 step cost. CPU R1
  tokenization throughput is not a claim about end-to-end training supply.
- Human decision requested: review/merge after acceptable independent verdict;
  model review is not merge authority

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | planning | Produced pinned candidates, primary-source links, cost model, selection gates, direct API/removal map, and test/R2 plan | Requested Sol/Ultra identity/mode unavailable; proposed R2 cannot precede ENV/CFG | Ticket order, philosophy, local cache, official repositories, current code | plan accepted with dependency-order adjustment |
| not exposed by runtime / not exposed by runtime | implementation phase 1 | Produced an exact artifact lock, frozen multilingual/Unicode corpus, operational and cost measurements, hard-gate evidence, deterministic selection, and focused tests | Requested Luna Extra High/max identity/mode was unavailable; initial module-form command did not match the repository's non-packaged `src` layout and was corrected to a direct script command | Planner handoff, candidate 40-hex revisions, official cards/source licenses, phase boundary excluding integration | comparison completed; LLM-jp v1 selected |

## Ledger update

- [x] Added ticket/PR row to `docs/model-runs/README.md`.
- [x] Recorded phase-1 implementation attempt; aggregate counts remain pending
  until the ticket finishes.
- [ ] Confirm live PR execution trail matches this record.
