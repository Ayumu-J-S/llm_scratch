# TOK-001 — Canonical Japanese/English Tokenizer

- PR: draft pending initial record commit
- Branch: `codex/tok-001-canonical-tokenizer`
- Ticket: `TOK-001`
- Hypothesis: a pinned established Japanese/English tokenizer selected by frozen
  corpus evidence and integrated through one local manifest/wrapper will make
  token IDs, vocabulary, special tokens, offline training, streaming, debug,
  model construction, and future generation identity consistent.
- Experiment record: `reports/tokenizers/TOK-001/comparison.md` (planned
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
| 1 | implementation | pending | pending | initial record commit | Freeze corpus, compare candidates, select/package winner, integrate direct canonical path, remove old backends, validate | pending | No implementation started | pending |
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

- Resolved Hydra command/config: pending
- Data/tokenizer/model identity: candidate revisions above; final artifact and
  frozen corpus checksums pending measurement
- Validation and measurements: pending
- Performance/resource result: CPU R1 comparison required. R2 BF16/DGX cannot
  run until ENV/CFG establish CUDA and a real profile; no performance conclusion
  will be made without it.
- Failed attempts retained at: execution timeline and comparison report
- Known trade-offs: established multilingual tokenizers greatly enlarge the
  current untied embedding/LM head; compression must repay that cost
- Unresolved risks: exact winner, artifact redistribution, real DGX step cost,
  and final tokenizer/process RSS
- Human decision requested: review/merge after acceptable independent verdict;
  model review is not merge authority

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | planning | Produced pinned candidates, primary-source links, cost model, selection gates, direct API/removal map, and test/R2 plan | Requested Sol/Ultra identity/mode unavailable; proposed R2 cannot precede ENV/CFG | Ticket order, philosophy, local cache, official repositories, current code | plan accepted with dependency-order adjustment |

## Ledger update

- [x] Added ticket/PR row to `docs/model-runs/README.md`.
- [ ] Update model counts after implementation/review invocations.
- [ ] Confirm live PR execution trail matches this record.
