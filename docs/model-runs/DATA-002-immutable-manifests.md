# DATA-002 — Immutable Manifests and Disjoint Splits

- PR: draft
- Branch: `codex/data-002-immutable-manifests`
- Ticket: `DATA-002`
- Hypothesis: immutable source manifests plus deterministic document identity
  and split assignment can make provenance and train/validation separation
  auditable before source access or training without adding per-sample hot-path
  checksum work.
- Started: 2026-07-11T17:38:46Z
- Final verdict: in progress
- Final record owner: primary task; exact runtime identity not exposed

## Scope and decision context

- Goal: make data identity, usage terms, checksums, document IDs/content hashes,
  split membership, and benchmark boundaries reproducible and fail closed.
- In scope: source/revision/config/split/text-ID/license/terms metadata; SHA-256
  for local/downloaded content; stable document IDs and normalized content
  hashes; deterministic split assignment/fingerprints; explicit memorization
  smoke fixture; benchmark access guard.
- Out of scope: final legal judgment, full semantic deduplication, benchmark
  scoring, raw-data W&B uploads, and unrelated training-loop redesign.
- Relevant `PHILOSOPHY.md` principles: visible text-to-training causal chain;
  data provenance and split construction recorded; training and benchmark test
  data separated; reproducible evidence; smallest coherent direct path; one-DGX
  boundary.
- Baseline commit/run: `a05eb1de5656643757a1c3d98047c98dedea8bfa`.
- Intended evidence: manifest/checksum mutation fails before source access;
  zero train/validation overlap by document ID and normalized content hash;
  split membership invariant under input reorder/prefetch; explicit same-corpus
  smoke only; benchmark guard; bounded R1 startup/hot-path measurement.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | handoff (planning) | pending | pending | `a05eb1d`, DATA-002, philosophy/CHECK, current data paths | Requested `gpt-5.6-sol` / Ultra plan | pending | No plan or model identity claimed yet | pending delegated planner |
| 1 | implementation | pending | pending | pending plan and live draft PR | Self-contained manifest/split implementation | pending | No implementation outcome claimed | pending |
| 1 | review | pending | pending | pending implementation commit | Independent DATA-002 `/review` | pending | No verdict claimed | pending |

## Check selection and verdicts

### Review cycle 1

- Review model / mode: pending
- Commit reviewed: pending
- Selected `CHECK.md` sections: 1, 4.1, 4.4, 7, 8.2, and 11 DATA-002
- Major sections marked N/A and why: pending plan/review
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
- Data/tokenizer/model identity: pending
- Validation and measurements: pending
- Performance/resource result if applicable: pending R1 design
- Failed attempts retained at: execution timeline
- Known trade-offs: pending
- Unresolved risks: pending
- Human decision requested: review/merge only after independent verdict; model
  review is not merge authority

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [ ] Updated per-model attempt, pass, repair, and review counts.
- [ ] Confirmed that the PR execution trail matches this record.
