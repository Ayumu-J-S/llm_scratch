# VAL-001 - Trustworthy Lightweight Held-Out Validation

- PR: [#42](https://github.com/Ayumu-J-S/llm_scratch/pull/42) (draft)
- Branch: `codex/val-001-held-out-validation`
- Ticket: `VAL-001`
- Hypothesis: one shared token-weighted scorer gives identical training-time and
  standalone checkpoint results with complete immutable evaluation identity.
- Experiment record: `docs/experiments/VAL-001-held-out-validation.md`
- Started: 2026-07-13
- Final verdict: blocked — implementation evidence passes; independent heavy review and DGX §6.3 evidence unavailable
- Final record owner: implementation agent

## Scope and decision context

- Goal: implement fixed Japanese/English held-out validation without conflating
  memorization with generalization.
- In scope: shared scoring, per-corpus and aggregate NLL/perplexity, step/token
  cadence, standalone checkpoint evaluation, local JSON, optional compact W&B
  summary, and immutable result identity.
- Out of scope: generative benchmarks, human evaluation, and reserved tests.
- Relevant `PHILOSOPHY.md` principles: Evaluation is part of training;
  experiments are first-class artifacts; fixed step/token cadences; reproducible
  data and scorer identities.
- Baseline commit/run: stacked DATA-004 head
  `e1d4ed8af98de84a3393cd0f6e517f9daf649138`.
- Intended evidence: known-logit NLL, scoring parity, cadence, overlap rejection,
  checkpoint milestones, exact identities, focused/full tests, and applicable
  `CHECK.md` 6.1, 8.2, and 6.3 review.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `e1d4ed8` plus VAL-001 ticket and acceptance audit | Implement the smallest coherent VAL-001 change | implemented | Shared scorer, fixed-window factories, attribution, identities, standalone Hydra evaluator, namespace guard, and invariant tests; CPU parity passed | `a8520d7`; [`docs/experiments/evidence/VAL-001-cpu-parity.json`](../experiments/evidence/VAL-001-cpu-parity.json) |
| 1 | review | not exposed by runtime | not exposed by runtime | `a8520d7` implementation diff, `PHILOSOPHY.md`, ticket, CHECK 6.1/8.2/6.3 | Independent heavier review | blocked | Runtime exposes no selectable heavier model or Extra Thinking mode; full review handoff prepared. DGX §6.3 smoke also blocked by CPU-only Torch | This record, PR #42, environment diagnostic |

## Runtime provenance block

| Namespace | Product | Displayed family | Exact model identifier | Reasoning mode | Source / unavailable reason |
| --- | --- | --- | --- | --- | --- |
| requested | Codex | GPT-5.6-class implementation model | not exposed by runtime | Extra High or higher requested | User and repository workflow |
| actual | Codex | not exposed by runtime | not exposed by runtime | not exposed by runtime | Collaboration runtime exposes no model/mode selector or display |

- Capture file/evidence: `docs/model-runs/evidence/VAL-001-implementation-provenance.json`;
  blocked review attempt: `docs/model-runs/evidence/VAL-001-review-provenance.json`
- Codex CLI version: unavailable; `gh` is also unavailable, so PR operations use the connected GitHub integration
- Branch/commit: `codex/val-001-held-out-validation` / implementation
  `a8520d7fad718574d1fca4293e6f969c7a478b79`; evidence/handoff
  `ed70f37b7c76144ba7ddb13ebdb13539314d453f`
- Phase/role/task path: implementation / primary and delegated audits
- Privacy confirmation: no prompts, hidden chain-of-thought, token counts, secrets, or raw thread ID recorded.

## Check selection and verdicts

### Review cycle 1

- Review model / mode: blocked — exact model and reasoning mode not exposed; no selectable heavier reviewer or Extra Thinking runtime available
- Commit reviewed: `a8520d7fad718574d1fca4293e6f969c7a478b79`
- Selected `CHECK.md` sections: 6.1, 8.2, 6.3
- Major sections marked N/A and why: 6.2/6.4 are not selected for the CPU
  correctness pass; DGX-specific performance items remain blocked rather than
  marked N/A. BENCH-001 and human-evaluation sections are out of scope by ticket.
- Ticket acceptance result: CPU fixture acceptance evidence passes; final
  acceptance remains blocked on independent review and §6.3 real-path evidence.
- Philosophy alignment: implementation follows fixed intermediate evaluation,
  step/token cadences, one-manifest/config authority, and compact result lineage.
- Complexity / change-surface result: localized to scorer, validation-loader
  construction, stream metadata propagation, checkpoint identity, evaluator,
  and ticket-specific tests; no BENCH-001 or generic refactor added.
- ML-system result: token weighting, ignored labels, causal source attribution,
  model-mode restoration, fixed-window replay, and CPU parity pass. DGX BF16
  throughput/pause/memory evidence is blocked by runtime packaging.
- Verdict: blocked pending independent heavier review and DGX §6.3 smoke

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| blocked | review runtime | Required heavier model / Extra Thinking mode is unavailable in the active runtime | Runtime exposes no selector/display; implementation model cannot substitute for independent review | Run the required independent review when a heavier runtime is available |
| note | CHECK §6.3 | DGX Spark GB10 is visible but locked PyTorch is `2.10.0+cpu`, CUDA unavailable, BF16 unavailable | `scripts/diagnose_environment.py` and `nvidia-smi` output in experiment record | Run 50–200 real DATA-004 BF16 steps and record pause/targets/s/step p50-p95/data wait/memory/recovery A/B |

## Failed-review handoff

- From review cycle: 1, blocked before independent verdict
- Failed check and why: no technical FAIL was returned. The required heavier
  independent review could not run because this runtime exposes neither a
  selectable heavier model nor Extra Thinking mode. The real CHECK §6.3 smoke
  is separately blocked because Torch is CPU-only.
- Review model / mode: not exposed by runtime / not exposed by runtime
- Implementation model / mode that produced the current state: not exposed by
  runtime / not exposed by runtime; requested implementation setting was Luna
  or available lightweight model at Extra High or higher
- Commit/diff to repair or review: `a8520d7fad718574d1fca4293e6f969c7a478b79`
- Reproduction/evidence commands:
  - `make ci-cpu` → Ruff clean, `294 passed, 1 skipped`, config/lock/offline
    smoke all passed
  - CPU train/eval commands and exact result are in
    `docs/experiments/evidence/VAL-001-cpu-parity.json`
  - `uv run --no-sync python scripts/diagnose_environment.py` → aarch64 GB10,
    `torch 2.10.0+cpu`, CUDA unavailable, BF16 unavailable
- Relevant files/config/manifests: `src/evaluation/scoring.py`,
  `src/evaluate.py`, `src/training/trainer.py`, `src/train.py`,
  `src/data/streaming_dataset.py`, `src/training/checkpoint.py`,
  `config/profile/evaluation.yaml`, VAL-001 fixture manifests, and CHECK
  sections 6.1/8.2/6.3
- Attempts already made: fixed scorer/digest implementation; repaired scorer
  batch-index failure reporting; completed focused and full network-free suites;
  completed CPU training-time/standalone parity; attempted DGX precondition
  inspection and retained the blocked result.
- Invariants and constraints: Hydra remains the only runtime configuration
  surface; direct imports; no raw held-out text/tokens in results; fixed
  windows are rebuilt per score event; per-corpus metrics are token-weighted;
  same-corpus smoke is `memorization/*`; physical checkpoint identity is only
  attached to a saved matching state; BENCH-001 remains out of scope.
- Selected next model / mode: an available heavier independent reviewer at
  Extra Thinking, exact runtime identity recorded as displayed
- Why selected: required by repository workflow for an independent ML/data/GPU
  review; implementation self-review cannot be counted as a pass
- Exact repair/review request: review the exact head against PHILOSOPHY.md,
  VAL-001 acceptance criteria, and CHECK 6.1/8.2/6.3; specifically audit
  batching-independent digest semantics, ignored-label/source attribution,
  fixed-window parity, overlap/memorization separation, checkpoint physical
  identity timing, iterator closure, cadence/pause behavior, and real DATA-004
  DGX BF16 evidence. Return PASS, justified PASS WITH NOTE, or FAIL with
  actionable evidence.
- Completion evidence requested: independent verdict on the exact head, all
  actionable findings repaired and re-reviewed, plus the bounded 50–200-step
  DGX result (or an explicitly updated blocked/PASS WITH NOTE record if the
  CUDA-capable runtime/data remains unavailable).

## Repair result

N/A — no technical review FAIL or repair cycle; the implementation-only repair
of scorer failure context is retained in the implementation history and covered
by the passing focused/full tests.

## Final evidence

- Resolved Hydra command/config: CPU train/eval commands and config SHA in
  [`docs/experiments/evidence/VAL-001-cpu-parity.json`](../experiments/evidence/VAL-001-cpu-parity.json)
- Data/tokenizer/model identity: fixture manifest, tokenizer fingerprint, and
  logical/physical checkpoint identities are recorded in the evidence JSON
- Validation and measurements: known-logit, partial/ignored-label,
  source-boundary, fixed-window digest, cadence, milestone, namespace, mode
  restoration, iterator-close, and parity tests pass; full suite is 294/1
- Performance/resource result if applicable: CPU validation pause recorded;
  DGX §6.3 target is blocked, with required missing fields explicitly listed
- Failed attempts retained at: this record and experiment record
- Known trade-offs: stacked PR depends on DATA-004 and will require retarget/rebase after dependency merge
- Unresolved risks: independent heavy review is blocked by runtime capability;
  real DATA-004 BF16 pause/throughput/memory evidence is blocked by CPU-only
  Torch; stacked DATA-004 dependency still requires retarget/rebase
- Human decision requested: human review and merge after technical PASS/PASS WITH NOTE

## Merge authority and final audit

- Merge path: `human merge`
- Human authorization: N/A — human merge
- Authorization evidence location: N/A
- Authorization covers this named PR or bounded ticket/goal series: N/A
- Exact independently reviewed head SHA: none — review blocked before verdict
- Latest independent verdict / model / mode: blocked / not exposed by runtime / not exposed by runtime
- All actionable findings repaired and independently re-reviewed: pending
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: pending
- Newer human objections since authorization/review: none observed
- Human review dismissed by an agent: no
- Unresolved review threads at final audit: zero observed on PR #42 at handoff
- Branch-protection required-context inventory: pending
- Applicable configured workflow/check inventory: pending
- Observed exact-head check statuses: pending
- Expected checks absent, pending, skipped, cancelled, or non-successful: pending
- No-check evidence when both inventories are empty: N/A
- Target branch and base SHA at final audit: stacked on `codex/data-004-pinned-baseline-mixture` / `e1d4ed8`
- Up-to-date, conflict-free, and mergeable evidence: pending
- Record, ledger, PR trail, validation, and risks parity: synchronized in
  `ed70f37`; PR #42 body records the same implementation/evidence heads
- Prohibited self-merge categories: human merge selected
- Admin/bypass/force/disabled-check requirement: no
- Final audit PR body/comment location: PR #42 draft body; final guarded audit is not applicable before human review
- Final audit changed reviewed head: pending
- Immediate pre-merge re-fetch/compare observation location: pending human merge
- Immediate refresh compared authorization, head, base, review decision/objections, threads, expected checks/statuses, and mergeability: no — pending
- Drift found: pending
- Merge outcome: not merged — draft PR intentionally left for human review

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | implementation | Localized the shared scorer and evaluator; preserved Hydra/direct-import policy; fixed a validation failure-context regression; produced exact CPU parity and network-free evidence | Could not independently review its own implementation or provide DGX BF16 measurements from CPU-only Torch | VAL-001 ticket, acceptance audit, PHILOSOPHY.md, CHECK 6.1/8.2/6.3, DATA-004 manifests | implemented; review handoff blocked |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated per-model attempt, pass, repair, and review counts.
- [x] Confirmed that the PR execution trail matches this record.
- [ ] Recorded human merge or complete guarded self-merge authority/audit evidence.
- [x] Confirmed that the bootstrap policy rule was not used before a human merged it.
