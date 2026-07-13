# VAL-001 — Trustworthy Lightweight Held-Out Validation

- PR: [#42](https://github.com/Ayumu-J-S/llm_scratch/pull/42) (draft)
- Branch: `codex/val-001-held-out-validation`
- Ticket: `VAL-001`
- Hypothesis: one shared token-weighted scorer gives identical training-time and
  standalone checkpoint results with complete immutable evaluation identity.
- Experiment: `docs/experiments/VAL-001-held-out-validation.md`
- Started: 2026-07-13
- Current verdict: implementation/evidence complete; independent heavy review pending
- Final record owner: implementation agent

## Scope and decision context

- Goal: fixed Japanese/English held-out validation without conflating
  memorization with generalization.
- In scope: shared scoring, per-corpus and aggregate NLL/perplexity, step/token
  cadence, standalone checkpoint evaluation, local JSON, optional compact W&B,
  and immutable result identity.
- Out of scope: generative benchmarks, human evaluation, and reserved tests.
- Policy: `PHILOSOPHY.md` evaluation-as-training, first-class evidence,
  fixed step/token cadences, and reproducible data/scorer identities.
- Baseline: stacked DATA-004 head
  `e1d4ed8af98de84a3393cd0f6e517f9daf649138`.
- Selected `CHECK.md`: minimum review, 6.1, 6.3, 7.1–7.3, 8.1–8.3, and the
  applicable checkpoint/evaluation identity parts of 9.1.

## Model execution trail

| Cycle | Phase | Requested model / mode | Exact displayed model / mode | Input commit | Outcome | Main findings / changes |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | implementation | `gpt-5.6-luna` / Extra High (`xhigh` invocation) | not exposed by runtime / not exposed by runtime | `e1d4ed8` | implemented, later repaired | Shared scorer, fixed-window factories, source attribution, identities, standalone Hydra evaluator, namespace guard, tests |
| 1 | independent review attempt | heavier review model / Extra Thinking | not exposed by runtime / not exposed by runtime | `a8520d7` | blocked before verdict | Earlier runtime path did not expose a selectable heavy reviewer; blocked attempt retained |
| 2 | repair | `gpt-5.6-luna` / Extra High | not exposed by runtime / not exposed by runtime | `a8520d7` | implemented | Strict JSON, NLL sums/reconciliation, source/manifest trust, verified checkpoint reconstruction, memorization isolation, stronger failure tests |
| 2 | preliminary audit repair | `gpt-5.6-luna` / Extra High | not exposed by runtime / not exposed by runtime | `2133248` | implemented | Model mode now restores even when iterator cleanup raises; regression test added at `0a13838` |
| 2 | independent heavy review | `gpt-5.6-sol` / Max (Extra Thinking class) | pending runtime display | documented exact head | pending | Must return PASS, justified PASS WITH NOTE, or actionable FAIL |

Requested values are invocation/config values, not claimed actual deployment
identifiers. The runtime did not expose the exact identifier or reasoning mode to
the caller for implementation/repair, so those actual fields remain unavailable.

## Provenance

- Initial implementation capture:
  `docs/model-runs/evidence/VAL-001-implementation-provenance.json`.
- Blocked review attempt:
  `docs/model-runs/evidence/VAL-001-review-provenance.json`.
- Repair capture:
  `docs/model-runs/evidence/VAL-001-repair-provenance.json`.
- Codex CLI: `codex-cli 0.144.1` for recorded implementation/repair captures.
- Implementation head: `a8520d7fad718574d1fca4293e6f969c7a478b79`.
- Main invariant repair: `057983c`; measured merged head:
  `21332488e8a1d2334cbb6e2d0593a77a598c1d01`.
- Exceptional-cleanup repair: `0a138386a03e178a88b5ccca6334288b57188efb`.
- Privacy: no prompts, hidden chain-of-thought, token counts, secrets, or raw
  thread IDs are recorded.

## Implementation and repair findings

Three delegated audits separated roadmap acceptance, implementation semantics,
and DGX evidence. The first implementation needed these concrete repairs:

| Area | Finding | Repair / proof |
| --- | --- | --- |
| Metric reduction | Aggregate-only reporting and insufficient exact reconciliation | Per-corpus/aggregate `nll_sum`, target counts, token weighting, strict reconciliation tests |
| Identity | Batch-sensitive or incomplete result identity could hide a changed window/source assignment | Batching-independent hashes cover contexts, labels/masks, order, target IDs, and target sources |
| Data semantics | Same-corpus smoke could be confused with validation | `memorization/*`, no memorization best checkpoint, and standalone memorization rejection |
| Checkpoint trust | Standalone reconstruction needed checkpoint-owned config and physical identity | Verified full-state checkpoint load, checkpoint config/model/tokenizer/data validation, path/SHA/size output |
| Serialization | `exp(NLL)` overflow could emit non-standard JSON | Nullable perplexity plus overflow flag; atomic JSON uses `allow_nan=False` |
| Performance path | Repeated full checkpoint hashing could enter training-time validation | Physical hashing occurs in standalone/output identity, not the hot path |
| Cleanup | Iterator-close failure could skip model-mode restoration | Nested `finally` at `0a13838`; regression proves restoration despite close error |

No BENCH-001 work, generic framework, compatibility shim, or separate runtime
configuration source was added. Hydra remains authoritative and imports are
direct.

## Validation and evidence

### Automated checks

- Focused baseline audit: `77 passed` before the repair series.
- Full repository suite at measured head `2133248`:
  `302 passed, 1 skipped in 64.17s`.
- `uv lock --check`, full Ruff lint, changed-file format check, and
  `git diff --check` passed at that head.
- Cleanup-repair focused check at `0a13838`: `15 passed`; Ruff, format, and diff
  checks passed.
- Tests cover known logits, ignored/partial labels, NLL reconciliation, source
  boundaries/failures, batching/context/source digest changes, unknown sources,
  strict JSON overflow, iterator lifecycle, zero targets, mode restoration,
  memorization namespace/no-best, and standalone milestone parity/identity.

### Invalidated evidence retained

`docs/experiments/evidence/VAL-001-cpu-parity.json` belongs to `a8520d7` and is
not current acceptance evidence. It evaluated a memorization `best.pt`; the
repaired code correctly creates no such best checkpoint and rejects standalone
memorization evaluation. It remains committed as a negative historical attempt.

### DGX R2 and standalone parity

Durable evidence:
[`docs/experiments/evidence/VAL-001-dgx-r2.json`](../experiments/evidence/VAL-001-dgx-r2.json).

- Pinned runtime: image
  `sha256:23a1bee69fe189e77105cdddeee9aeff6ef0763d58a691625fbfcab64efd1887`,
  NVIDIA GB10, BF16/CUDA, PyTorch `2.13.0a0+8145d630e8.nv26.06`.
- Matched 50-step arms used the same 49,535,114-parameter model, DATA-004 cache,
  manifests, seed, precision, batch/accumulation, and 102,400 training targets.
  Only validation cadence changed.
- The off/on training loss traces and all non-time step payloads matched exactly.
- Validation ran exactly at steps 25 and 50. Each pass scored 65,536 fixed
  targets at about 3,358 targets/s in about 19.52 seconds; the Japanese/English
  denominator split was exactly 32,768/32,768.
- Step-50 training-time and standalone evaluation matched exactly on aggregate
  and per-corpus scores, counts, manifests, logical checkpoint identity, 8,192
  windows, window hash, and target hash. Standalone output SHA-256 is
  `43bedcad72b138f873e227b8c455cf0598ac6a6b643d10b262ebf986838fa06e`.
- Best checkpoint SHA-256 is
  `0d48d810cd4657473c904b3cfd7e3a63174d2b7a3284248ad97ddbeb82f6ddea`.
- Validation plus best-save pauses totaled 43.1997 seconds. The first clean
  post-validation step returned to the immediate pre-validation timing range.

CHECK §6.3 is partially satisfied: pause accounting, independent cadence,
trajectory isolation, counters, and recovery are demonstrated. This one A/B
pair does not support a general speed claim, lacks a continuous system trace,
and the trainer does not expose separate data-wait/forward/backward/etc. timing.
The independent reviewer must decide whether this is a justified note for this
validation ticket or requires added instrumentation/rerun.

The R2 measured `2133248`; `0a13838` only changes exceptional iterator cleanup.
No successful scoring/training code path or performance control changed. This
parent-head relationship is disclosed rather than misrepresented as exact-head
measurement.

## Review status and handoff

### Review attempt 1

- Commit: `a8520d7`.
- Result: blocked before a technical verdict; it does not count as an
  independent review.
- Historical blocker claims about DGX unavailability are superseded by the
  pinned-container R2 above.

### Required review cycle 2

- Requested reviewer: `gpt-5.6-sol`, Max reasoning (repository's independent
  Extra Thinking class).
- Target: exact documentation/code head after the R2 record is committed.
- Required review: `PHILOSOPHY.md`, VAL-001 acceptance criteria, minimum CHECK,
  6.1, 6.3, 7.1–7.3, 8.1–8.3, and applicable 9.1.
- Specific questions: scoring math and masks; batching-independent identities;
  source attribution; overlap/memorization separation; checkpoint physical and
  logical trust; iterator/mode lifecycle; cadence/pause isolation; whether the
  one-pair and missing phase breakdown are acceptable as `PASS WITH NOTE`.
- Completion: all actionable findings repaired and independently re-reviewed,
  then record exact runtime-displayed model/mode and verdict.

## Risks and handoff

- Known trade-off: the fixed 65,536-target validation pass costs about 19.52 s
  plus about 1.8–2.4 s when an improving best checkpoint is saved.
- Evidence limitation: one matched pair, snapshots rather than a continuous
  system trace, and no phase/data-wait timing.
- Dependency: this stacked PR still depends on DATA-004, whose source-rights
  disposition is a human policy gate.
- Merge path: human review and merge; no self-merge authorization exists.
- Exactly one next step: independent GPT-5.6 heavy review of the documented head.

## Merge authority and final audit

- Merge path: `human merge`.
- Human authorization: N/A — human merge.
- Exact independently reviewed head: pending.
- Latest independent verdict/model/mode: pending.
- Actionable findings repaired and re-reviewed: pending.
- Blocking review decision / `CHANGES_REQUESTED`: pending exact-head fetch.
- Newer human objection: none observed.
- Human review dismissed by an agent: no.
- Unresolved review threads: pending exact-head fetch.
- Branch-protection required-context and workflow inventory: pending final audit.
- Exact-head checks: pending after the final review/documentation commit.
- Base: `codex/data-004-pinned-baseline-mixture` at `e1d4ed8` when this stack began.
- Mergeability/conflict status: pending final audit.
- Prohibited self-merge categories: source-rights dependency and human merge path
  make self-merge unavailable.
- Final audit / immediate pre-merge refresh: pending human merge.
- Merge outcome: not merged; draft PR remains open.

## Ledger update

- [x] VAL-001 row exists in `docs/model-runs/README.md`.
- [x] Failed/invalidated attempts remain visible.
- [ ] Independent review verdict and exact displayed provenance recorded.
- [ ] Aggregate pass/repair/review counts updated after the final verdict.
- [ ] Human merge/final audit recorded.
