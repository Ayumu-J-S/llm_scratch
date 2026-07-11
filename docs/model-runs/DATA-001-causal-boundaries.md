# DATA-001 — Correct Packed Causal Boundaries

- PR: [#11](https://github.com/Ayumu-J-S/llm_scratch/pull/11)
- Branch: `codex/data-001-causal-boundaries`
- Ticket: `DATA-001`
- Hypothesis: packed `L+1` token windows advanced by stride `L`, with explicit
  quota-truncation boundaries, train every intended next-token transition
  exactly once without corrupting source spans or token counts.
- Experiment record: `N/A — correctness-invariant ticket with bounded fixture
  and tiny CPU smoke evidence; no consequential research run`
- Started: 2026-07-11T15:31:27Z
- Final verdict: in progress
- Final record owner: primary task; exact runtime identity not exposed

## Scope and decision context

- Goal: train every intended next-token transition exactly once.
- In scope: stride `L` over `L+1` windows, carried boundary token, explicit
  quota-truncation/EOS policy, packed source spans, and explicit packed target
  accounting.
- Out of scope: shuffle/cursor policy, multi-node or worker partitioning,
  production sources, tokenizer replacement, trainer counter integration,
  architecture changes, compilation, kernels, and general performance tuning.
- Relevant `PHILOSOPHY.md`: preserve the claimed objective; keep the data/model
  boundary clear; fail explicitly; make the smallest conventional change; do
  not add a configurable compatibility path for broken semantics.
- Baseline commit/run: `a05eb1de5656643757a1c3d98047c98dedea8bfa`;
  no credible training baseline exists because roadmap readiness gates remain
  unmet.
- Intended evidence: exact ticket sequence and transition multiset; property
  transitions; quota/EOS and span tests; explicit token accounting; tiny real
  dataloader/model forward-backward optimizer smoke; focused/full CPU checks.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | handoff (planning) | not exposed by runtime | not exposed by runtime | `a05eb1d`, DATA-001, philosophy, applicable CHECK | Plan with requested `gpt-5.6-sol` / `ultra` | completed | Located stride/truncation root causes and specified direct packing-layer invariants; runtime did not expose requested identity/mode | Planner handoff in parent task |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `a05eb1d`, accepted DATA-001 plan, ticket/philosophy/CHECK, loader/dataset/tests/README | Requested `gpt-5.6-luna` at Extra High/max; implement the smallest coherent DATA-001 fix and R1 evidence | completed | Packed windows now stride by `W-1`, quota truncation preserves or requires an explicit boundary, spans/counters follow the stride, and packed token bases are explicit | focused `58 passed, 3 skipped`; full `59 passed, 3 skipped`; Ruff, lock, diff, Hydra composition green |
| 1 | review | pending | pending | pending implementation commit | Independent review against ticket, philosophy, and selected CHECK sections | pending | No verdict claimed | pending |

## Check selection and verdicts

### Review cycle 1

- Review model / mode: pending
- Commit reviewed: pending
- Selected `CHECK.md` sections: 1, 4.1, 4.3, 6.1, 7, and 11 DATA-001
- Major sections marked N/A and why: tokenizer selection, manifests/splits,
  checkpointing, W&B, and architecture do not change; DGX R2 is unavailable
  until later ENV/TOK/CFG dependencies provide CUDA and a real profile.
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

`N/A — implementation and review pending.`

## Final evidence

- Resolved Hydra command/config: `uv run python src/train.py data.mode=streaming
  --cfg job --resolve` completed and resolved `data.mode: streaming`, context
  length 64, and the current streaming sections. The resolved train/validation
  source lists remain empty, the known CFG-001 blocker; no implicit fixture or
  local-text fallback was used.
- Data/tokenizer/model identity: deterministic in-memory character-level
  `tokenizers.WordLevel` fixtures with explicit IDs/EOS; current
  `SimpleDecoderTransformer` with one layer, 8-wide embeddings, two heads,
  context length 3, FP32 CPU for the optimizer smoke. No pretrained model
  weights or external data entered the test.
- Validation and measurements:
  - `uv run pytest -q tests/test_stream_loader.py tests/test_streaming_dataset.py`
    -> `58 passed, 3 skipped in 3.68s`; skipped tests are opt-in external
    tokenizer/dataset integrations.
  - `uv run pytest -q` -> `59 passed, 3 skipped in 3.96s`.
  - `uv run ruff check .` -> pass; `uv run ruff format --check` on all three
    changed Python files -> pass; `git diff --check` -> pass.
  - `uv lock --check` -> resolved 147 packages with no lock change.
  - Exact `[2..8]` windows/collation and repeated-ID `Counter` properties prove
    transition equality; quota exact/cut/EOS-only cases, unsafe no-EOS packed
    truncation, source ratios/spans, synchronous and process-prefetched tail
    accounting/reset, and a finite-loss, finite-nonzero-gradient optimizer step
    all pass.
  - A bounded 8,193-token document sanity emitted 128 overlapping windows and
    accounted for 8,192 targets with no dropped transition.
- Performance/resource result if applicable: R1 CPU correctness and bounded
  loader supply sanity passed. No throughput or resource-regression claim is
  made. Runtime identity was `aarch64`, `torch 2.10.0+cpu`, and
  `torch.cuda.is_available() == False`. R2/DGX validation is not runnable from
  this environment and the empty real streaming profile and remains sequenced
  behind ENV/TOK/CFG work.
- Failed attempts retained at: the first repo-wide `ruff format --check .`
  reported six pre-existing unformatted Python files outside this ticket. The
  changed Python files pass format checking; unrelated files were not rewritten.
- Known trade-offs: a carried token is materialized in two adjacent windows but
  contributes each transition once
- Unresolved risks: real-profile throughput and CUDA behavior remain for later
  dependency tickets
- Human decision requested: review/merge after independent acceptable verdict;
  a model review is not merge authority

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | planning | Produced exact stride, EOS, span, accounting, and smoke invariants | Overconstrained completion on an R2 run blocked by later roadmap dependencies | Ticket order, philosophy, source/tests, CHECK routing | plan accepted with sequencing adjustment |
| not exposed by runtime / not exposed by runtime | implementation | Kept the code change localized to packing/quota semantics and added direct invariant evidence through the real dataset/model boundary | The requested `gpt-5.6-luna` identity and Extra High/max mode could not be selected or verified in this collaboration runtime; no independent verdict is available from this pass | Accepted plan, exact acceptance sequence, explicit out-of-scope list, existing tests | implementation completed; awaiting independent heavy review |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated the per-model implementation attempt count; review/repair counts
  remain zero until those phases actually run.
- [ ] Confirm the PR execution trail matches this record.
