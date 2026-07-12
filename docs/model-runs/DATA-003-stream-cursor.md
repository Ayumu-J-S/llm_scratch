# DATA-003 - Deterministic stream horizon, shuffle, and exact cursor

- PR: draft / pending creation
- Branch: `codex/data-003-stream-cursor`
- Ticket: DATA-003
- Hypothesis: A bounded stream with explicit pass policy and serialized source/RNG cursor can reproduce an uninterrupted suffix while keeping prefetch an execution detail.
- Experiment record: `N/A` — loader fixture and invariant evidence are captured here; no research-quality model run is in scope.
- Started: 2026-07-12
- Final verdict: in progress
- Final record owner: `/root/data003_implementation`

## Scope and decision context

- Goal: prevent repeated-prefix passes and make streaming interruption/resume exact.
- In scope: explicit per-pass horizon, deterministic bounded shuffle, source RNG and cursor/buffer state, repeat accounting, and sync/thread/process ordering equivalence.
- Out of scope: multi-node sharding, adaptive mixtures, throughput optimization, and model checkpoint payloads.
- Relevant `PHILOSOPHY.md` principles: one-machine boundary; train the model we claim to train; explicit data identity and reproducible experiments; no hidden fallback.
- Baseline commit: `60a6d86482241fff891c8701b9242d2fc0817bb6` (merged LOOP-001 final audit).
- Intended evidence: deterministic fixture sequence, JSON-safe cursor interruption/resume, pass/repeat policy, and prefetch equivalence tests.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `60a6d864`; DATA-003, `PHILOSOPHY.md`, selected `CHECK.md` data/packing and prefetch sections | Requested Luna / Extra High implementation; keep one loader path and make cursor state serializable | completed; independent review pending | Added explicit horizon/repeat policy, deterministic bounded shuffle, source/global RNG state, raw-document cursor and packed-buffer state, `state_dict`/`load_state_dict`, and prefetch cursor markers | 6 DATA-003 tests; 218 passed, 1 skipped; Ruff, lock, diff checks pass |
| 1 | review | not exposed by runtime | not exposed by runtime | pending exact draft-PR head | Requested heavier independent Extra Thinking review against DATA-003 acceptance, `PHILOSOPHY.md`, and selected `CHECK.md` sections | pending | Must verify no prefix repeat without explicit repeat, exact interrupted suffix including packed windows, source/manifest identity, and sync/thread/process equivalence | pending |

## Runtime provenance block

```json
{
  "schema_version": "1.0",
  "captured_at": "2026-07-12T10:40:55.330424Z",
  "phase": "implementation",
  "role": "implementation",
  "task_path": "/root/data003_implementation",
  "requested": {
    "model": {"value": "Luna", "source": "explicit invocation/config default", "status": "observed"},
    "reasoning_mode": {"value": "Extra High", "source": "explicit invocation/config default", "status": "observed"}
  },
  "actual": {
    "product": {"value": "Codex", "source": "active runtime display", "status": "observed"},
    "displayed_model_family": {"value": "GPT-5", "source": "active runtime display", "status": "observed"},
    "exact_model_identifier": {"value": "not exposed by runtime", "source": "active runtime display", "status": "unavailable"},
    "reasoning_mode": {"value": "not exposed by runtime", "source": "active runtime display", "status": "unavailable"}
  },
  "environment": {
    "codex_cli_version": "codex-cli 0.144.1",
    "branch": "codex/data-003-stream-cursor",
    "commit": "60a6d86482241fff891c8701b9242d2fc0817bb6",
    "thread_id": "not recorded (privacy)"
  },
  "privacy": {
    "raw_thread_id_recorded": false,
    "prompts_recorded": false,
    "hidden_chain_of_thought_recorded": false,
    "token_counts_recorded": false,
    "secrets_recorded": false
  }
}
```

- Capture command: `uv run python scripts/capture_model_provenance.py --repo . --phase implementation --role implementation --task-path /root/data003_implementation --requested-model Luna --requested-reasoning-mode 'Extra High' --actual-product Codex --actual-model-family GPT-5 --actual-exact-model 'not exposed by runtime' --actual-reasoning-mode 'not exposed by runtime'`.
- Privacy confirmation: no prompts, hidden chain-of-thought, token counts, secrets, or raw thread IDs.

## Check selection and verdicts

### Review cycle 1

- Review model / mode: pending independent heavy review; actual exact model and mode must be recorded as displayed or `not exposed by runtime`.
- Commit reviewed: pending draft PR head.
- Selected `CHECK.md` sections: minimum review; 4.3 packing/cursor accounting; 4.4 source/cache identity; 7.1 training-loop synchronization only where loader ordering affects it; 8 experiment/reproducibility and 10 changeability.
- Major sections marked N/A and why: 5 DGX/GPU and 6 model/optimizer are unchanged; no performance claim or training recipe change is made.
- Ticket acceptance result: pending independent review.
- Philosophy alignment: deterministic source identity and explicit repeat policy are visible; prefetch does not alter order.
- Complexity / change-surface result: pending review of the single-loader state surface and bounded buffer implementation.
- ML-system result: fixture-level data semantics pass; no DGX claim.
- Verdict: in progress.

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| — | — | No independent findings yet. | Review pending. | Complete independent review before marking ready or merged. |

## Failed-review handoff

N/A — no failed independent review has been returned.

## Repair result

N/A — no repair cycle yet.

## Final evidence

- Resolved Hydra command/config: `config/stream_loader.yaml` now documents `horizon.repeat: false`, deterministic bounded shuffle, and buffer size; fixture tests exercise equivalent plain mappings.
- Data/tokenizer identity: canonical tokenizer manifest fingerprint `12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b`; fixture documents are immutable in-memory records for offline invariants.
- Validation and measurements: `uv run --group dev pytest -q` → `218 passed, 1 skipped`; DATA-003 focused file → `6 passed`; `uv run ruff check .`; `uv lock --check`; `git diff --check`.
- Performance/resource result if applicable: N/A; this ticket explicitly defers throughput optimization and DGX measurement.
- Failed attempts retained at: N/A.
- Known trade-offs: cursor stores bounded shuffle-buffer documents and Python RNG state so an interrupted stream can resume without source replay ambiguity; it is intentionally separate from CKPT-001 model state.
- Unresolved risks: independent review must confirm packed-window interruption semantics and process-prefetch cursor behavior on the exact PR head.
- Human decision requested: review the independent verdict and guarded merge audit after all checks are refreshed.

## Merge authority and final audit

- Merge path: guarded agent self-merge only if the parent goal authorization is recorded and all gates pass; otherwise human merge.
- Human authorization: parent task explicitly authorizes self-merge for the bounded roadmap goal on 2026-07-12; exact parent instruction must be copied into final PR audit.
- Authorization evidence location: parent task messages and final PR audit comment.
- Authorization covers this named PR or bounded ticket/goal series: pending final audit.
- Exact independently reviewed head SHA: pending.
- Latest independent verdict / model / mode: pending.
- All actionable findings repaired and independently re-reviewed: pending.
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: pending.
- Newer human objections since authorization/review: pending final refresh.
- Human review dismissed by an agent: no.
- Unresolved review threads at final audit: pending.
- Branch-protection required-context inventory: pending connector refresh.
- Applicable configured workflow/check inventory: pending connector refresh.
- Observed exact-head check statuses: pending connector refresh.
- Expected checks absent, pending, skipped, cancelled, or non-successful: pending.
- No-check evidence when both inventories are empty: pending.
- Target branch and base SHA at final audit: pending.
- Up-to-date, conflict-free, and mergeable evidence: pending.
- Record, ledger, PR trail, validation, and risks parity: pending PR creation/update.
- Prohibited self-merge categories: clear — ordinary repository data-loader code and tests only.
- Admin/bypass/force/disabled-check requirement: no.
- Final audit PR body/comment location: pending.
- Final audit changed reviewed head: no (must remain no).
- Immediate pre-merge re-fetch/compare observation location: pending.
- Immediate refresh compared authorization, head, base, review decision/objections, threads, expected checks/statuses, and mergeability: pending.
- Drift found: pending.
- Merge outcome: pending.

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| Codex / GPT-5; exact ID and mode not exposed | implementation | Localized deterministic source/cursor semantics, bounded shuffle, repeat accounting, and prefetch cursor propagation with focused invariants | Independent review and merge-gate evidence are not yet available | DATA-003 acceptance, loader internals, DATA-001/DATA-002 boundaries, and selected CHECK sections | implementation complete; review pending |

## Ledger update

- [x] Added the DATA-003 ticket record; PR URL and final verdict remain pending until the draft exists and independent review completes.
- [ ] Updated aggregate implementation/review counts after final verdict.
- [ ] Confirmed PR execution trail matches this record.
- [ ] Recorded complete guarded self-merge authority/audit or human merge evidence.
- [x] Confirmed no bootstrap policy self-merge rule is being used.
