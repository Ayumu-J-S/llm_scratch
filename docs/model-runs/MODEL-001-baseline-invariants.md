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
| 1 | implementation | pending | requested Luna / Extra High | pending accepted plan | Implement the smallest invariant suite and any required direct model fixes | pending | No implementation claimed yet | pending |
| 1 | review | pending | requested heavier / Extra Thinking | pending stable implementation commit | Independent `/review` against acceptance, philosophy, and applicable CHECK | pending | No verdict claimed yet | pending |

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

- Resolved Hydra command/config: pending
- Data/tokenizer/model identity: pending
- Validation and measurements: pending
- Performance/resource result if applicable: pending
- Failed attempts retained at: this record
- Known trade-offs: pending
- Unresolved risks: CUDA/DGX validation depends on ENV-001 and must not be
  fabricated if the runtime remains CPU-only.
- Human decision requested: review only after independent model verdict; a human
  remains the sole merge authority.

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [ ] Updated per-model attempt, pass, repair, and review counts.
- [ ] Confirmed that the PR execution trail matches this record.
