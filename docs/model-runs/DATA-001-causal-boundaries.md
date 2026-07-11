# DATA-001 — Correct Packed Causal Boundaries

- PR: draft pending initial record commit
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
| 1 | implementation | pending | pending | initial record commit | Implement the smallest coherent DATA-001 fix and R1 evidence | pending | No implementation has started | pending |
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

- Resolved Hydra command/config: pending
- Data/tokenizer/model identity: frozen in-memory token-ID fixture plus current
  conventional tiny model for smoke; exact identities pending implementation
- Validation and measurements: pending
- Performance/resource result if applicable: R1 loader sanity required; DGX R2
  cannot run on the current CPU-only PyTorch/empty-real-profile baseline and
  will be recorded without a performance claim
- Failed attempts retained at: execution timeline
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

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [ ] Update per-model implementation, repair, and review counts after those
  phases actually run.
- [ ] Confirm the PR execution trail matches this record.
