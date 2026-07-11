# MODEL-001 - Conventional Baseline Invariants

- PR: [#14](https://github.com/Ayumu-J-S/llm_scratch/pull/14) (draft)
- Branch: `codex/model-001-baseline-invariants`
- Ticket: `MODEL-001`
- Hypothesis: a compact invariant suite and bounded overfit proof can make the
  current conventional decoder-only Transformer a trustworthy reference without
  redesigning the architecture.
- Started: 2026-07-11T18:54:00Z
- Final verdict: in progress
- Final record owner: primary task; exact runtime identity not exposed

## Scope and decision context

- Goal: protect the current simple architecture before later model changes.
- In scope: output shape, maximum context, causal prefix invariance, finite
  loss/gradients, padding semantics, parameter count, deterministic tiny-batch
  overfit, and applicable R1/R2 measurements.
- Out of scope: RoPE, RMSNorm, SwiGLU, weight tying, new attention mechanisms,
  architecture redesign, trainer redesign, or optimization work.
- Relevant `PHILOSOPHY.md` principles: begin with a conventional, inspectable
  model; make the learning causal chain visible; prove tiny overfit and stable
  behavior before expanding complexity; preserve the smallest coherent change.
- Baseline commit/run: `a05eb1de5656643757a1c3d98047c98dedea8bfa`.
- Intended evidence: exact invariant tests, predeclared overfit threshold and
  seed, CPU timing/memory reference, and CUDA forward/backward smoke only when
  the ENV-001 runtime is available.

## Predeclared implementation contract

- The architecture math stays unchanged: conventional MHA, LayerNorm, GELU,
  sinusoidal positions, residual paths, and untied LM head.
- Token inputs must be rank-2, non-empty, `torch.long`, and no longer than the
  configured context. `pad_token_id` must be a non-boolean integer inside the
  vocabulary.
- Padding has one authority: the model derives it from `pad_token_id`; the
  public external-mask override is removed because no repository caller uses it.
  Padded query logits are zero and all-pad rows remain finite using tensor-only
  operations, without `.item()`, host boolean conversion, or CUDA synchronization.
  A zero-valid-target objective remains LOOP-001 scope.
- Tiny test model: vocab 8, width 16, 4 heads, one layer, feed-forward width 32,
  context 6, dropout 0, PAD 0. Independent exact parameter oracle: 2,488.
- Overfit gate fixed before implementation: seed 17, a fixed 4x6 cyclic batch
  over IDs 1-4, AdamW lr 0.02 and weight decay 0, exactly 30 updates, all losses
  finite, final below initial, and final cross-entropy at most 0.02. Planner
  calibration observed 0.002615; the acceptance threshold remains 0.02.
- R1: current default FP32 CPU model, batch 4/context 64, one thread, two warmups
  and seven forward/loss/backward samples with every time plus median/p95/max and
  process RSS recorded. R2 CUDA/BF16 is blocked until ENV-001 when CUDA is absent.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | handoff | not exposed by runtime | not exposed by runtime | baseline plus ticket/philosophy/CHECK/model context | Requested Sol / Ultra plan for a bounded implementation and evidence contract | completed | Froze architecture; defined metadata-only input validation, single-authority padding semantics, exact invariant suite, fixed overfit gate, parameter oracle, and R1/R2 evidence | Planner handoff retained in primary task; no repository mutation |
| 1 | implementation | not exposed by runtime | not exposed by runtime (requested Luna / Extra High) | `a186aa6ffa954c38bc88ca891bae34394f494737`, accepted predeclared contract | Implement the smallest invariant suite and required direct model fixes; run CPU validation and R1 | completed | Added metadata-only input validation, constructor PAD validation, inferred padding with an all-pad-safe tensor sentinel, zero padded-query hidden states/logits, and one bounded invariant test module; architecture math is unchanged | 20 focused tests; 59 passed/3 opt-in network skips full suite; fixed overfit CE 3.028204918 -> 0.002615080 in 30 updates; R1 retained below |
| 1 | repair | not exposed by runtime | not exposed by runtime (requested Luna / Extra High) | uncommitted implementation plus primary precommit audit | Complete exact accepted-plan assertions omitted by the first test pass without changing model code | completed | Switched shape evidence to the fixed 4x6 batch; strengthened causal suffix, nonzero-gradient, padded-objective/PAD-gradient, parameter utility, and external-mask-authority assertions | 20 focused tests; 59 passed/3 opt-in network skips; model implementation unchanged |
| 1 | review | pending | requested heavier / Extra Thinking | pending stable implementation commit | Independent `/review` against acceptance, philosophy, and applicable CHECK | pending | No verdict claimed; implementation evidence is not a review substitute | pending |

## Check selection and verdicts

### Review cycle 1

- Review model / mode: pending
- Commit reviewed: pending
- Selected `CHECK.md` sections: 5.2, 6.1, 6.2, 7, and 11 MODEL-001
- Major sections marked N/A and why: pending review
- Ticket acceptance result: pending
- Philosophy alignment: pending
- Complexity / change-surface result: pending
- ML-system result: pending
- Verdict: pending

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |

## Failed-review handoff

N/A - review pending.

## Repair result

N/A - review pending.

## Final evidence

- Resolved Hydra command/config: `uv run python src/train.py --cfg job
  --resolve` completed successfully; its output retained the current local-text
  baseline and model/training values described below.
- Data/tokenizer/model identity: resolved `uv run python src/train.py --cfg job
  --resolve`; current baseline uses local-text train/validation paths,
  sequence length 64, vocabulary 512, width 384, six heads, six layers, and
  dropout 0.1. MODEL-001 did not change data, tokenizer, objective, or model
  architecture.
- Validation and measurements: `uv run pytest -q
  tests/test_simple_decoder_transformer.py` -> 20 passed; `uv run pytest -q` ->
  59 passed and 3 existing opt-in network tests skipped; changed Python files
  pass Ruff formatting; `uv run ruff check .`, `uv lock --check`, and
  `git diff --check` pass. The fixed seed-17 4x6 tiny batch used exactly 30
  AdamW updates at lr 0.02/weight decay 0; CE moved from 3.028204918 to
  0.002615080, below the predeclared 0.02 threshold. Independent parameter
  oracles pass at 2,488 tiny and 11,040,512 default parameters.
- Performance/resource result if applicable: R1 ran on
  Linux 6.17.0-1021-nvidia aarch64, Python 3.11.15, PyTorch 2.10.0+cpu, FP32,
  one Torch thread, default 11,040,512-parameter model, batch 4/context 64,
  two warmups and seven forward/CE/backward samples. Warmups were 0.348251658
  and 0.197080849 seconds. Measured samples were 0.203491432, 0.197784442,
  0.193664565, 0.185591112, 0.188827135, 0.191766092, and 0.193504321
  seconds; median 0.193504321, nearest-rank p95 0.203491432, maximum
  0.203491432, and process `ru_maxrss` 450,468 KiB. This is a reference only;
  no speed comparison or improvement claim is made. Source inspection found
  no `.item()`, host boolean conversion, or tensor-conditioned Python branch in
  the changed model hot path.
- Failed attempts retained at: the first standalone R1 command omitted
  `PYTHONPATH=src` and failed with `ModuleNotFoundError: No module named
  'models'`; the corrected command was `PYTHONPATH=src uv run python ...`.
- Known trade-offs: padded query work is still computed inside attention and
  then zeroed; this favors a direct, inspectable invariant over a new attention
  implementation. R1 is a single bounded CPU reference, not a performance
  conclusion.
- Unresolved risks: CUDA/DGX validation depends on ENV-001 and must not be
  fabricated: this runtime exposes PyTorch 2.10.0+cpu and
  `torch.cuda.is_available() == False`, so R2 is blocked on ENV-001. Independent
  heavy review remains pending; no PASS verdict is claimed.
- Human decision requested: review only after independent model verdict; a human
  remains the sole merge authority.

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime (requested Luna / Extra High) | implementation and precommit repair | Kept the change to one model and one invariant test module; satisfied the predeclared causal, numerical, padding, parameter, and overfit checks | The first test pass omitted several exact accepted-plan assertions and the initial standalone R1 probe omitted `PYTHONPATH=src`; both were corrected before the stable implementation commit | Frozen parameter oracle, exact overfit gate, primary precommit audit, single padding authority, and explicit architecture exclusions | completed; review pending |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated per-model attempt, pass, repair, and review counts for the
  completed implementation phase; review counts remain zero while `/review` is
  pending.
- [ ] Confirmed that the PR execution trail matches this record.
