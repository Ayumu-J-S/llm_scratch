# MODEL-001 - Conventional Baseline Invariants

- PR: draft
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

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | handoff | pending | requested Sol / Ultra | baseline plus ticket/philosophy/CHECK/model context | Produce a bounded implementation and evidence plan | pending | No plan claimed yet | pending |
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
