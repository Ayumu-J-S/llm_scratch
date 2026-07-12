# DATA-003 - Deterministic stream horizon, shuffle, and exact cursor

- PR: [#29](https://github.com/Ayumu-J-S/llm_scratch/pull/29) (merged; PASS WITH NOTE; merge `57266e1e843be2d08e10ef5f387da8466b0c590f`)
- Branch: `codex/data-003-stream-cursor`
- Ticket: DATA-003
- Hypothesis: A bounded stream with explicit pass policy and serialized source/RNG cursor can reproduce an uninterrupted suffix while keeping prefetch an execution detail.
- Experiment record: `N/A` — loader fixture and invariant evidence are captured here; no research-quality model run is in scope.
- Started: 2026-07-12
- Final verdict: PASS WITH NOTE
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
| 1 | implementation | not exposed by runtime | not exposed by runtime | `60a6d864`; DATA-003, `PHILOSOPHY.md`, selected `CHECK.md` data/packing and prefetch sections | Requested Luna / Extra High implementation; keep one loader path and make cursor state serializable | completed; first review failed, then repair/re-review passed | Added explicit horizon/repeat policy, deterministic bounded shuffle, source/global RNG state, raw-document cursor and packed-buffer state, `state_dict`/`load_state_dict`, and prefetch cursor markers | 8 DATA-003 tests; 220 passed, 1 skipped; Ruff, lock, diff checks pass |
| 1 | review | not exposed by runtime | not exposed by runtime | `883f6d03e9f8fb763c5465715071eccb4038625b` (PR #29) | Requested heavier independent Extra Thinking review against DATA-003 acceptance, `PHILOSOPHY.md`, and selected `CHECK.md` sections | FAIL | Thread prefetch advanced the shared cursor ahead of a slow consumer; interrupted resume duplicated/omitted samples while process markers were consumer-safe | 10/10 bounded-memory reproductions with buffer 2 and delayed cursor capture |
| 2 | repair | not exposed by runtime | not exposed by runtime | review cycle 1 failure and `883f6d0` | Requested Luna / Extra High repair: make async cursor state consumer-acknowledged without changing sample order; preserve process behavior | completed; re-review PASS WITH NOTE | Added cursor ACK marker before every thread sample and a separate parent `_consumer_cursor`; `state_dict()` returns ACK state while async worker runs; added delayed thread/process interruption regressions | Focused DATA-003 tests 8 passed; full suite 220 passed, 1 skipped; static checks clean |
| 2 | re-review | not exposed by runtime | not exposed by runtime | `ea2c01e68ab4d120b10b3f8208d1388a0be7d19c` (PR #29) | Re-run independent review on exact repair head | PASS WITH NOTE | Consumer-ack cursor markers close the thread/process ahead-of-consumer defect; cursor buffering remains a documented memory trade-off | Review `4679913983`; focused 8 passed; full 220 passed, 1 skipped; static checks clean |
| 3 | review | not exposed by runtime | not exposed by runtime | `13d90f7e8921c0875a7c37ad1bf44a3147d94c09` (PR #29) | Exact-head refresh after ready-state docs update | FAIL | Thread reuse lost `pass_complete` on natural exhaustion; `load_state_dict` cursor was absent from spawned process config | Review `4679929272`; two P2 findings, no merge |
| 3 | repair | not exposed by runtime | not exposed by runtime | `13d90f7` plus review `4679929272` | Preserve completed async cursor and propagate explicit cursor into process worker; add regressions | completed; cycle-4 refresh found process reuse P2 | Final thread cursor marker preserves completed-pass state; `load_state_dict` updates serialized process config; thread reuse and process load-state tests added | Focused 10 passed; full 222 passed, 1 skipped; static checks clean; repair `bc8ebbc32db56436e37d32550c1ef6d11a56e66` |
| 3 | re-review | not exposed by runtime | not exposed by runtime | `bc8ebbc32db56436e37d32550c1ef6d11a56e66` | Independent exact-head review after P2 repairs | FAIL/continued | Process reuse remained unsafe on the exact refresh; handed to cycle 4 | Finding evidence `4679944167` |
| 4 | review | not exposed by runtime | not exposed by runtime | `04ca349e6257315580d225196cca658a134795ac` | Exact-head refresh of process-prefetch reuse | FAIL | Process prefetch serialized stale `self.config["cursor"]` on loader reuse and repeated the first pass | Review `4679944167`; repair `93132f7` |
| 4 | repair | not exposed by runtime | not exposed by runtime | `04ca349` plus cycle-4 finding | Sync acknowledged cursor into process config on each marker; add process same-loader reuse regression | completed; re-review PASS WITH NOTE | Config sync is gated to DATA-003 cursor mode so legacy process fixtures retain repeat behavior | Focused 11 passed; full 223 passed, 1 skipped; static checks clean |
| 4 | re-review | not exposed by runtime | not exposed by runtime | `9abfeb2e33be6e7e78bd3ac730544c9e29157d4c` | Independent exact-head review | PASS WITH NOTE | Process same-loader reuse and prior async cursor/resume paths pass; bounded cursor-buffer memory remains documented | Review `4679956834`; focused 11 passed; full 223 passed, 1 skipped; static checks clean |

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

- Review model / mode: actual exact model and reasoning mode not exposed by runtime (independent reviewer returned FAIL, then PASS WITH NOTE).
- Commit reviewed: `883f6d03e9f8fb763c5465715071eccb4038625b` (initial FAIL); repaired and re-reviewed at `ea2c01e68ab4d120b10b3f8208d1388a0be7d19c` (review `4679913983`).
- Selected `CHECK.md` sections: minimum review; 4.3 packing/cursor accounting; 4.4 source/cache identity; 7.1 training-loop synchronization only where loader ordering affects it; 8 experiment/reproducibility and 10 changeability.
- Major sections marked N/A and why: 5 DGX/GPU and 6 model/optimizer are unchanged; no performance claim or training recipe change is made.
- Ticket acceptance result: PASS WITH NOTE — all four acceptance invariants plus delayed async interruption regressions pass on code head `ea2c01e`.
- Philosophy alignment: deterministic source identity and explicit repeat policy are visible; prefetch does not alter order.
- Complexity / change-surface result: PASS WITH NOTE — protocol remains in the existing loader; bounded cursor buffers have documented memory cost.
- ML-system result: fixture-level data semantics pass; no DGX claim.
- Verdict: PASS WITH NOTE — cycles 3/4 findings were repaired and independently re-reviewed; bounded cursor-buffer memory remains the documented trade-off.

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| High | cursor/resume | Thread worker mutated the shared cursor while prefetching ahead; `state_dict()` could capture source position 7 after only one sample had been consumed, causing a resumed suffix mismatch. | Delayed-consumer reproduction with 100 memory documents, max_tokens 120, shuffle buffer 3, thread prefetch buffer 2; 10/10 failures. | Publish a consumer-acknowledged cursor before each sample and prevent worker state from replacing parent cursor. |

## Failed-review handoff

- From review cycle: 1
- Failed check and why: DATA-003 exact interruption/resume and CHECK 4.3/8 failed for thread prefetch because worker-ahead cursor state was externally visible.
- Review model / mode: actual exact identity and mode not exposed by runtime.
- Implementation model / mode that produced failed state: actual exact identity and mode not exposed by runtime; requested Luna / Extra High.
- Commit/diff to repair: `883f6d03e9f8fb763c5465715071eccb4038625b`.
- Reproduction command or evidence: consume one threaded-prefetched sample, sleep 50 ms, call `state_dict`, close, resume; source position was ahead and prefix+suffix differed in 10/10 trials.
- Relevant files/config/manifests: `src/data/stream_loader/loader.py`, `tests/test_data003_stream_cursor.py`, horizon shuffle fixture.
- Attempts already made: queue ordering was equivalent but parent read the worker's live cursor.
- Invariants and constraints: no sample-order change; process/thread/off must match; cursor must represent the last yielded sample; no hidden fallback.
- Selected next model / mode: requested Luna / Extra High repair; independent re-review remains required.
- Why this model was selected: localized loader lifecycle defect with a bounded queue protocol and no model/math changes.
- Exact repair request: add consumer-ack cursor markers before each async sample; keep worker cursor private from `state_dict`; add delayed thread and process resume regressions.
- Completion evidence requested: focused/full tests, exact-head independent re-review, and no unresolved blocking thread.

## Repair result

- Repair cycle: 2
- Repair model / mode: actual exact identity and mode not exposed by runtime; requested Luna / Extra High.
- Input handoff: review cycle 1's 10/10 delayed-thread cursor mismatch.
- Changes made: thread worker emits `_CURSOR_MARKER` immediately before each queued sample; parent tracks `_consumer_cursor`; async `state_dict()` returns acknowledged state, while process prefetch emits the same marker protocol. Added delayed thread interruption regression and retained process interruption coverage.
- What was deliberately not changed: source sampling, shuffle algorithm, manifest identity, packed residual semantics, process mode, or model/checkpoint code.
- Local evidence: DATA-003 focused tests 11 passed; full suite `223 passed, 1 skipped`; Ruff, lock, and diff checks pass.
- Commit reviewed next: `ea2c01e68ab4d120b10b3f8208d1388a0be7d19c` (code; current docs-only head `9c077d5553c8b2e9010b6b4f9e677ca52de25c1b`).
- Re-review model / mode: actual exact model and reasoning mode not exposed by runtime.
- Re-review verdict: PASS WITH NOTE (`4679913983`) on exact `ea2c01e`.

- Repair cycle 3: actual exact model and reasoning mode not exposed by runtime; requested Luna / Extra High.
- Input handoff: exact-head refresh `4679929272` found thread pass-completion cursor overwrite and missing process-worker cursor propagation from `load_state_dict`.
- Changes made: thread worker emits a final cursor marker after natural exhaustion; `load_state_dict` mirrors the cursor into serialized config; added same-loader thread reuse and process load-state resume regressions.
- Local evidence: DATA-003 focused 10 passed; full suite 222 passed, 1 skipped; Ruff, lock, and diff checks pass.
- Commit reviewed next: `bc8ebbc32db56436e37d32550c1ef6d11a56e66`.
- Re-review model / mode: actual exact model and reasoning mode not exposed by runtime.
- Re-review verdict: PASS WITH NOTE (`4679956834`) on exact head `9abfeb2`.

- Repair cycle 4: actual exact model and reasoning mode not exposed by runtime; requested Luna / Extra High.
- Input handoff: cycle-4 process-reuse finding `4679944167` on `04ca349`.
- Changes made: acknowledged cursor markers now synchronize `self.config["cursor"]` for DATA-003 cursor mode; process same-loader reuse regression added while preserving legacy process fixtures.
- Local evidence: DATA-003 focused 11 passed; full suite 223 passed, 1 skipped; Ruff, lock, and diff checks pass.
- Commit reviewed next: `93132f7f4103492e41357eee3d0f8a1277ccecb4` (current exact code/docs head).
- Re-review model / mode: actual exact model and reasoning mode not exposed by runtime.
- Re-review verdict: PASS WITH NOTE (`4679956834`) on exact head `9abfeb2`.

## Final evidence

- Resolved Hydra command/config: `config/stream_loader.yaml` now documents `horizon.repeat: false`, deterministic bounded shuffle, and buffer size; fixture tests exercise equivalent plain mappings.
- Data/tokenizer identity: canonical tokenizer manifest fingerprint `12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b`; fixture documents are immutable in-memory records for offline invariants.
- Validation and measurements: `uv run --group dev pytest -q` → `223 passed, 1 skipped`; DATA-003 focused file → `11 passed`; `uv run ruff check .`; `uv lock --check`; `git diff --check`.
- Performance/resource result if applicable: N/A; this ticket explicitly defers throughput optimization and DGX measurement.
- Failed attempts retained at: N/A.
- Known trade-offs: cursor stores bounded shuffle-buffer documents and Python RNG state so an interrupted stream can resume without source replay ambiguity; it is intentionally separate from CKPT-001 model state.
- Unresolved risks: bounded shuffle cursor stores buffered documents in memory by design; CKPT-001 still needs to persist this cursor alongside model state.
- Human decision requested: N/A — the recorded authorization and guarded audit permitted the ordinary repository self-merge.

## Merge authority and final audit

- Merge path: guarded agent self-merge only if the parent goal authorization is recorded and all gates pass; otherwise human merge.
- Human authorization: parent task explicitly authorizes self-merge for the bounded roadmap goal on 2026-07-12; exact parent instruction must be copied into final PR audit.
- Authorization evidence location: parent task messages and final PR audit comment.
- Authorization covers this named PR or bounded ticket/goal series: yes — DATA-003 within the bounded roadmap goal.
- Exact independently reviewed head SHA: `87a64b8a72604ddf67cf9536cb0661cff7a9a663` (docs-only descendant of repair code `93132f7`).
- Latest independent verdict / model / mode: PASS WITH NOTE `4679961413` (docs-only exact-head re-confirmation; implementation review `4679956834`); exact model and reasoning mode not exposed by runtime.
- All actionable findings repaired and independently re-reviewed: yes — cycles 1, 3, and 4 findings were re-reviewed on the exact head.
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: none; all submitted reviews are non-blocking comments and no `CHANGES_REQUESTED` review remains.
- Newer human objections since authorization/review: none observed in the immediate pre-merge refresh.
- Human review dismissed by an agent: no.
- Unresolved review threads at final audit: zero; both actionable threads were resolved before merge.
- Branch-protection required-context inventory: no required contexts reported by the connector.
- Applicable configured workflow/check inventory: no pull-request workflow runs reported for the exact head.
- Observed exact-head check statuses: empty (`github_get_commit_combined_status` for `87a64b8`).
- Expected checks absent, pending, skipped, cancelled, or non-successful: zero; the no-check state is evidenced by the empty status and workflow inventories.
- No-check evidence when both inventories are empty: final audit comment [`#issuecomment-4951026925`](https://github.com/Ayumu-J-S/llm_scratch/pull/29#issuecomment-4951026925) and exact-head connector refresh.
- Target branch and base SHA at final audit: `main` at `60a6d86482241fff891c8701b9242d2fc0817bb6`.
- Up-to-date, conflict-free, and mergeable evidence: final refresh recorded Ready, mergeable, and unchanged head before merge.
- Record, ledger, PR trail, validation, and risks parity: yes; this record and README now identify the merged PR and merge SHA.
- Prohibited self-merge categories: clear — ordinary repository data-loader code and tests only.
- Admin/bypass/force/disabled-check requirement: no.
- Final audit PR body/comment location: [`#issuecomment-4951026925`](https://github.com/Ayumu-J-S/llm_scratch/pull/29#issuecomment-4951026925) (post-merge completion; pre-merge refresh [`#issuecomment-4951019993`](https://github.com/Ayumu-J-S/llm_scratch/pull/29#issuecomment-4951019993)).
- Final audit changed reviewed head: no (must remain no).
- Immediate pre-merge re-fetch/compare observation location: [`#issuecomment-4951026925`](https://github.com/Ayumu-J-S/llm_scratch/pull/29#issuecomment-4951026925) (with the pre-merge audit at [`#issuecomment-4951019993`](https://github.com/Ayumu-J-S/llm_scratch/pull/29#issuecomment-4951019993)).
- Immediate refresh compared authorization, head, base, review decision/objections, threads, expected checks/statuses, and mergeability: yes; no drift found.
- Drift found: no.
- Merge outcome: merged to `main` as `57266e1e843be2d08e10ef5f387da8466b0c590f`.

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| Codex / GPT-5; exact ID and mode not exposed | implementation/review | Localized deterministic source/cursor semantics, bounded shuffle, repeat accounting, consumer-ack protocol, completion preservation, and process cursor synchronization; independent review closed all actionable findings | Exact deployment/model ID and reasoning mode unavailable; bounded cursor-buffer memory remains a documented trade-off | DATA-003 acceptance, loader internals, DATA-001/DATA-002 boundaries, selected CHECK sections, delayed-consumer and reuse/resume reproductions | PASS WITH NOTE `4679956834` |

## Ledger update

- [x] Added the DATA-003 ticket record and PR URL; final verdict is PASS WITH NOTE.
- [x] Updated aggregate implementation/review counts after final verdict.
- [x] Confirmed PR execution trail matches this record through cycle-4 review `4679956834` and repair `93132f7`.
- [x] Recorded complete guarded self-merge authority/audit or human merge evidence.
- [x] Confirmed no bootstrap policy self-merge rule is being used.
