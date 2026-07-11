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
| 1 | review | not exposed by runtime | not exposed by runtime | `7193fb4` | Independent review against DATA-001, philosophy, and selected CHECK sections | FAIL | Process-prefetch counters were copied to the parent only after successful completion, so a later early-closed or failed iteration exposed stale totals from the prior success | Counter lifecycle inspection at `StreamLoader._iter_async` and process accounting marker path |
| 1 | repair | not exposed by runtime | not exposed by runtime | `7193fb4` plus review cycle 1 handoff | Requested `gpt-5.6-luna` at Extra High; reset async accounting before worker launch and cover repeat/close/error lifecycle | completed | Parent counters reset at each async iteration start; normal process completion still publishes final totals; early close and worker error retain safe current-iteration zeros | lifecycle regressions `3 passed`; focused `60 passed, 3 skipped`; full `61 passed, 3 skipped`; Ruff, lock, diff, Hydra green; re-review pending |
| 1 | re-review | pending | pending | pending repair commit | Independently verify the failed counter lifecycle and the unchanged DATA-001 invariants | pending | No verdict claimed | pending |

## Check selection and verdicts

### Review cycle 1

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: `7193fb4`
- Selected `CHECK.md` sections: 1, 4.1, 4.3, 6.1, 7, and 11 DATA-001
- Major sections marked N/A and why: tokenizer selection, manifests/splits,
  checkpointing, W&B, and architecture do not change; DGX R2 is unavailable
  until later ENV/TOK/CFG dependencies provide CUDA and a real profile.
- Ticket acceptance result: FAIL — final packed accounting was correct only on
  successful process completion; iteration-local accounting was false after a
  prior success followed by early close or worker failure.
- Philosophy alignment: FAIL for observable accounting integrity; stale totals
  could misrepresent which data and targets the failed iteration supplied.
- Complexity / change-surface result: otherwise localized; no trainer, Hydra,
  tokenizer, or compatibility path was introduced.
- ML-system result: exact transition, quota, EOS, source-span, and optimizer
  invariants passed, but process-prefetch failure lifecycle accounting failed.
- Verdict: FAIL

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| High | Process-prefetch accounting lifecycle | Parent-visible `token_counts` and `packed_token_counts` retained the previous successful iteration until the child completion marker arrived; early close or worker failure never delivered that marker | `7193fb4` reset counters only inside the child/output path, while the parent copied them only from `_ACCOUNTING_MARKER` | Reset parent accounting at the actual start of every async iteration and regress success→success, success→early-close, and success→worker-error |

## Failed-review handoff

- From review cycle: 1
- Failed check and why: CHECK 4.3 emitted/target/dropped accounting was not
  iteration-safe under process-prefetch close and error paths.
- Review model / mode: not exposed by runtime / not exposed by runtime
- Implementation model / mode that produced the failed state: not exposed by
  runtime / not exposed by runtime
- Commit/diff to repair: `7193fb4`
- Reproduction command or evidence: run one successful process-prefetched
  packed iteration, then either close the next iteration after its first sample
  or make its JSONL worker fail; the parent retained the first run's totals.
- Relevant files/config/manifests: `src/data/stream_loader/loader.py`,
  `tests/test_stream_loader.py`; in-memory and mutable local JSONL fixtures with
  process prefetch.
- Attempts already made: successful child totals were sent through a private
  completion marker, but parent totals were not cleared before worker launch.
- Invariants and constraints: completed process iteration publishes exact final
  totals; early/failed iteration must never expose a prior run; do not move
  counters into the trainer or add a configuration branch.
- Selected next model / mode: requested `gpt-5.6-luna` at Extra High; actual
  model identifier and reasoning mode not exposed by runtime.
- Why this model was selected: narrow repair requested by the primary task
  following the independent FAIL.
- Exact repair request: reset counters before each async worker launch and add
  repeated-success, early-close, and worker-error process-prefetch regressions.
- Completion evidence requested: focused/full pytest, Ruff, lock/diff, Hydra
  composition, updated README and model-run ledger; independent re-review.

## Repair result

- Repair cycle: 1
- Repair model / mode: not exposed by runtime / not exposed by runtime; requested
  `gpt-5.6-luna` at Extra High.
- Input handoff: review cycle 1 FAIL on target `7193fb4`.
- Changes made: centralized iteration accounting reset and invoked it before
  async worker launch; retained child-to-parent final accounting on successful
  completion; added repeated success, success→early-close, and
  success→worker-error process regression tests.
- What was deliberately not changed: packing stride, EOS/quota behavior,
  trainer integration, configuration, and the known repeated front-deletion
  implementation. The latter remains a documented scaling risk with no
  performance claim.
- Local evidence: process lifecycle regressions `3 passed`; focused
  `60 passed, 3 skipped`; full `61 passed, 3 skipped`; Ruff check and changed-file
  format check, lock check, Hydra streaming composition, and diff check passed.
- Commit reviewed next: pending repair commit.
- Re-review model / mode: pending independent heavy review.
- Re-review verdict: pending; no PASS claimed.

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
    -> `60 passed, 3 skipped in 7.14s`; skipped tests are opt-in external
    tokenizer/dataset integrations.
  - `uv run pytest -q` -> `61 passed, 3 skipped in 7.72s`.
  - `uv run ruff check .` -> pass; `uv run ruff format --check` on all three
    changed Python files -> pass; `git diff --check` -> pass.
  - `uv lock --check` -> resolved 147 packages with no lock change.
  - Exact `[2..8]` windows/collation and repeated-ID `Counter` properties prove
    transition equality; quota exact/cut/EOS-only cases, unsafe no-EOS packed
    truncation, source ratios/spans, synchronous accounting/reset, completed
    process totals, repeated process success, process early-close/error reset,
    and a finite-loss, finite-nonzero-gradient optimizer step all pass.
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
  contributes each transition once. Packed buffering still repeatedly deletes
  consumed prefixes from the front of a Python list; the bounded sanity proves
  correctness, not favorable long-document scaling.
- Unresolved risks: real-profile throughput, long-document front-deletion
  scaling, and CUDA behavior remain for later dependency tickets.
- Human decision requested: review/merge after independent acceptable verdict;
  a model review is not merge authority

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | planning | Produced exact stride, EOS, span, accounting, and smoke invariants | Overconstrained completion on an R2 run blocked by later roadmap dependencies | Ticket order, philosophy, source/tests, CHECK routing | plan accepted with sequencing adjustment |
| not exposed by runtime / not exposed by runtime | implementation | Kept the code change localized to packing/quota semantics and added direct invariant evidence through the real dataset/model boundary | The requested `gpt-5.6-luna` identity and Extra High/max mode could not be selected or verified in this collaboration runtime; no independent verdict is available from this pass | Accepted plan, exact acceptance sequence, explicit out-of-scope list, existing tests | implementation completed; awaiting independent heavy review |
| not exposed by runtime / not exposed by runtime | review | Detected stale parent accounting specifically on early-close and worker-error process paths despite green success-path tests | Exact heavier model identity/mode was not exposed | Target `7193fb4`, CHECK 4.3, process accounting marker lifecycle | FAIL; repair required |
| not exposed by runtime / not exposed by runtime | repair | Localized the lifecycle fix to one reset boundary and added direct failure-path regressions | Requested Luna Extra High identity/mode remained unverifiable; re-review is still required | Exact FAIL handoff and mutable local JSONL reproduction | repair completed; verdict remains in progress |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated implementation, review, and repair attempt counts; successful
  repair remains zero until an independent passing re-review.
- [ ] Confirm the PR execution trail matches this record.
