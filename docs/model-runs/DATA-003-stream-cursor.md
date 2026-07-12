# DATA-003 - Deterministic stream horizon, shuffle, and exact cursor

- PR: [#29](https://github.com/Ayumu-J-S/llm_scratch/pull/29) (merged as `57266e1e843be2d08e10ef5f387da8466b0c590f`; post-merge P2 re-opened DATA-003); P5 repair [#31](https://github.com/Ayumu-J-S/llm_scratch/pull/31) (draft)
- Branch: `codex/data-003-stream-cursor`
- Ticket: DATA-003
- Hypothesis: A bounded stream with explicit pass policy and serialized source/RNG cursor can reproduce an uninterrupted suffix while keeping prefetch an execution detail.
- Experiment record: `N/A` — loader fixture and invariant evidence are captured here; no research-quality model run is in scope.
- Started: 2026-07-12
- Current verdict: in progress — P5 packed-cursor repair is implemented on PR #31; independent re-review pending
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
| 5 | post-merge review | not exposed by runtime | not exposed by runtime | merged `main` at `57266e1`; automated finding `4679969079`, independent audit review `4679980858` | Re-check exact packed-window interruption/resume invariant after merge | FAIL (P2) | `_packed_iter` yields the final partial window before clearing `_packed_cursor_buffer`, so the saved cursor retains already-emitted residual tokens and resume emits them again | Canonical tokenizer; seed 17; four memory docs `text="a"`; `packed_sequences`, `max_tokens=4`, `sequence_length=5`, `add_eos=false`, `drop_remainder=false`, `horizon={repeat:false, shuffle:false}`; full `[[311,311,311,311]]`, resume `[[311,311,311,311,311],[311,311,311,311]]` |
| 5 | documentation correction | not exposed by runtime | not exposed by runtime | PR #30 correction head `4da78859c99a8400ec6522f746eeee098fd40040`; failed audit `4679980858` | Reopen DATA-003, retain P2 evidence and repair handoff, and remove completion claims without changing code | completed; docs-only re-review PASS WITH NOTE | ROADMAP and ledger now mark DATA-003 In progress and scope link evidence correctly; no repair head or implementation claim added | Review `4679987639`; no source/test or unrelated-roadmap drift; inherited focused tests 11 passed and static checks remain clean |
| 6 | repair | not exposed by runtime | not exposed by runtime | post-merge P2 handoff; repair branch rebased on `9bf68b0`; [P5 record](DATA-003-packed-resume-repair.md) | Requested Luna / Extra High repair: consume the final partial packed residual before it can be checkpointed; retain sync/thread/process equivalence | implemented; independent re-review pending | PR #31 clears the terminal packed residual before yield and fixes thread prefetch's completed producer-cursor marker; it adds exact-suffix regressions across sync/thread/process plus the four-document P2 reproduction | Focused 15 passed; full 227 passed, 1 skipped; Ruff, lock, diff checks pass; final exact head pending |

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
- Historical validation before merge: `uv run --group dev pytest -q` → `223 passed, 1 skipped`; DATA-003 focused file → `11 passed`; `uv run ruff check .`; `uv lock --check`; `git diff --check`. Those checks did not exercise interruption after a final partial packed window.
- Documentation-audit link check: the three newly added PR/audit-comment links returned HTTP 200. A broad ROADMAP link sweep also found two pre-existing NVIDIA DGX URLs returning HTTP 404; those links are outside this DATA-003 correction, so this record makes no all-links-passing claim.
- Performance/resource result if applicable: N/A; this ticket explicitly defers throughput optimization and DGX measurement.
- Failed attempts retained at: post-merge packed-cursor reproduction below.
- Known trade-offs: cursor stores bounded shuffle-buffer documents and Python RNG state so an interrupted stream can resume without source replay ambiguity; it is intentionally separate from CKPT-001 model state.
- Unresolved risk: P5 repair PR #31 awaits independent review on its exact rebased head; until then DATA-003 remains in progress.
- Human decision requested: do not treat DATA-003 as complete or unblock its dependents until PR #31 has an independent PASS or justified PASS WITH NOTE.

## Post-merge P2 handoff

- Reviews: post-merge automated finding `4679969079` and independent audit FAIL `4679980858` on the merged DATA-003 lineage. Documentation-only correction review `4679987639` is PASS WITH NOTE on exact head `4da78859c99a8400ec6522f746eeee098fd40040`.
- Severity and violated acceptance criterion: P2; an interrupted and resumed stream no longer yields the exact uninterrupted suffix in `packed_sequences` output mode.
- Reproduction: canonical tokenizer; seed `17`; four memory documents with `text="a"`; `output_mode=packed_sequences`, `max_tokens=4`, `sequence_length=5`, `add_eos=false`, `drop_remainder=false`, and `horizon={repeat:false, shuffle:false}`.
- Expected: the uninterrupted run is `[[311, 311, 311, 311]]`; stopping after that partial output and resuming must produce no additional window.
- Observed: `state_dict()` retains `packed_buffer=[311, 311, 311, 311]`; resume emits `[[311, 311, 311, 311, 311], [311, 311, 311, 311]]`, so the consumed prefix plus resumed suffix differs from the uninterrupted run.
- Root cause: `_packed_iter` yields the final partial window before clearing `_packed_cursor_buffer`; the cursor therefore serializes residual tokens that have already been emitted.
- Repair handoff outcome: PR #31 implements the requested residual advance and
  adds sync/thread/process exact-suffix regressions. Its companion
  [per-PR record](DATA-003-packed-resume-repair.md) contains the runtime
  provenance and evidence; independent exact-head review remains pending.

## Historical PR #29 merge audit and current documentation-audit status

- Merge path: historical guarded self-merge of PR #29 only; it is not an authorization to merge a future repair without fresh gates.
- Human authorization: parent task explicitly authorized self-merge for the bounded roadmap goal on 2026-07-12; the authorization and audit apply to the historical #29 merge.
- Authorization evidence location: parent task messages and final PR #29 audit comment.
- Authorization covers a repair PR: pending fresh repair PR audit.
- Exact independently reviewed historical head SHA: `87a64b8a72604ddf67cf9536cb0661cff7a9a663` (docs-only descendant of repair code `93132f7`).
- Latest implementation verdict / model / mode: FAIL P2 (`4679980858`, following automated finding `4679969079`) on the merged lineage; exact model and reasoning mode not exposed by runtime. Documentation-only correction review `4679987639` is PASS WITH NOTE on `4da78859c99a8400ec6522f746eeee098fd40040` and does not repair the code. Historical PR #29 re-confirmation was PASS WITH NOTE `4679961413`.
- All actionable findings repaired and independently re-reviewed: no — PR #31 implements the post-merge packed-cursor P2 repair, but independent exact-head review remains pending.
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: packed-cursor P2 blocks ticket completion and any repair merge until PR #31 is independently re-reviewed.
- Newer objection/finding after the historical merge audit: yes — automated finding `4679969079`, independently confirmed as FAIL `4679980858`.
- Human review dismissed by an agent: no.
- Unresolved review threads at the historical audit: zero; the new P2 remains an open repair handoff.
- Branch-protection required-context inventory: no required contexts reported by the connector.
- Applicable configured workflow/check inventory: no pull-request workflow runs reported for the exact head.
- Observed exact-head check statuses: empty (`github_get_commit_combined_status` for `87a64b8`).
- Expected checks absent, pending, skipped, cancelled, or non-successful: zero; the no-check state is evidenced by the empty status and workflow inventories.
- No-check evidence when both inventories are empty: final audit comment [`#issuecomment-4951026925`](https://github.com/Ayumu-J-S/llm_scratch/pull/29#issuecomment-4951026925) and exact-head connector refresh.
- Target branch and base SHA at final audit: `main` at `60a6d86482241fff891c8701b9242d2fc0817bb6`.
- Up-to-date, conflict-free, and mergeable evidence: final refresh recorded Ready, mergeable, and unchanged head before merge.
- Record, ledger, PR trail, validation, and risks parity: corrected by PR #30 and appended with the PR #31 repair record; current status is in progress pending independent re-review.
- Prohibited self-merge categories: clear — ordinary repository data-loader code and tests only.
- Admin/bypass/force/disabled-check requirement: no.
- Historical final audit PR body/comment location: [`#issuecomment-4951026925`](https://github.com/Ayumu-J-S/llm_scratch/pull/29#issuecomment-4951026925) (post-merge completion; pre-merge refresh [`#issuecomment-4951019993`](https://github.com/Ayumu-J-S/llm_scratch/pull/29#issuecomment-4951019993)).
- Historical final audit changed reviewed head: no.
- Historical immediate pre-merge re-fetch/compare observation location: [`#issuecomment-4951026925`](https://github.com/Ayumu-J-S/llm_scratch/pull/29#issuecomment-4951026925) (with the pre-merge audit at [`#issuecomment-4951019993`](https://github.com/Ayumu-J-S/llm_scratch/pull/29#issuecomment-4951019993)).
- Historical refresh compared authorization, head, base, review decision/objections, threads, expected checks/statuses, and mergeability: yes; no drift found before merge.
- Historical merge outcome: PR #29 merged to `main` as `57266e1e843be2d08e10ef5f387da8466b0c590f`; the post-merge P2 re-opened the ticket.
- Current repair self-merge authorization: the user's general authorization for the bounded roadmap goal applies to PR #31, but its independent review, exact-head audit, and all guarded merge gates remain mandatory.

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| Codex / GPT-5; exact ID and mode not exposed | implementation/review | Localized deterministic source/cursor semantics, bounded shuffle, repeat accounting, consumer-ack protocol, completion preservation, process cursor synchronization, and the P5 residual repair | Earlier review coverage missed interruption after a final partial packed window; exact deployment/model ID and reasoning mode are unavailable | DATA-003 acceptance, loader internals, DATA-001/DATA-002 boundaries, selected CHECK sections, delayed-consumer and reuse/resume reproductions | P5 repair pending independent review (`4679980858` is the retained failure) |

## Ledger update

- [x] Added the DATA-003 ticket record and PR URLs; current verdict is in progress pending independent P5 re-review.
- [x] Updated the DATA-003 summary for the post-merge P2, documentation re-review, and PR #31 repair handoff.
- [x] Confirmed the execution trail retains cycles 1–4, post-merge finding `4679969079`, independent FAIL `4679980858`, documentation PASS WITH NOTE `4679987639`, and P5 repair cycle 6.
- [x] Retained the historical guarded self-merge evidence while recording that it does not clear the new P2.
- [x] Confirmed no bootstrap policy self-merge rule is being used.
