# DATA-003 - Packed residual resume repair

- PR: [#31](https://github.com/Ayumu-J-S/llm_scratch/pull/31) (draft)
- Branch: `codex/data-003-packed-resume-repair`
- Ticket: DATA-003
- Hypothesis: Once a final short packed window has been yielded, its residual
  tokens are consumed and must not remain in the serialized cursor; otherwise
  interruption and resume duplicates a prefix of that window.
- Experiment record: `N/A` — this is a deterministic loader-invariant repair,
  not a research-quality training run.
- Started: 2026-07-12
- Final verdict: in progress
- Final record owner: `/root/data003_p5_repair`

## Scope and decision context

- Goal: preserve DATA-003's exact uninterrupted suffix guarantee for a final
  `drop_remainder: false` packed window.
- In scope: packed residual cursor lifecycle, focused regression coverage, and
  model-run/PR evidence.
- Out of scope: source selection, shuffle algorithm, prefetch architecture,
  model/checkpoint payloads, distributed loading, and throughput optimization.
- Relevant `PHILOSOPHY.md` principles: train the model we claim to train;
  reproducible, inspectable experiments; direct readable implementations; no
  speculative compatibility paths.
- Baseline commit/run: `57266e1e843be2d08e10ef5f387da8466b0c590f` (merged
  DATA-003 PR #29); the repair is rebased on documentation-audit main
  `9bf68b0373022309e320db7e8674769a542cc511`.
- Intended evidence: the reported four one-token-document reproduction, plus
  an uninterrupted-versus-interrupted packed sequence equivalence assertion.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | repair | not exposed by runtime | not exposed by runtime | `57266e1` P2 reproduction; rebased on `9bf68b0`; `ROADMAP.md`, `PHILOSOPHY.md`, selected `CHECK.md` 4.1/4.3/9.1 | Requested Luna / Extra High repair: make packed `drop_remainder: false` resume exact without broadening DATA-003 | implemented; independent review pending | Clear the packed cursor residual before yielding a final short window. Thread prefetch now publishes the producer's completed cursor instead of constructing an empty source-state cursor after the producer closes. | Focused cursor suite: 15 passed; full suite: 227 passed, 1 skipped; Ruff, lock, and diff checks pass |
| 1 | re-review | not exposed by runtime | not exposed by runtime | exact rebased PR #31 head pending | Independent heavier review requested as Extra Thinking against DATA-003, philosophy, and selected checklist sections | pending | Must examine exact repair head and all guarded merge evidence before a ready/self-merge decision | pending |

## Runtime provenance block

```json
{
  "schema_version": "1.0",
  "captured_at": "2026-07-12T11:43:59.880721Z",
  "phase": "repair",
  "role": "repair",
  "task_path": "/root/data003_p5_repair",
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
    "branch": "codex/data-003-packed-resume-repair",
    "commit": "57266e1e843be2d08e10ef5f387da8466b0c590f",
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

- Capture command: `uv run python scripts/capture_model_provenance.py --repo . --phase repair --role repair --task-path /root/data003_p5_repair --requested-model Luna --requested-reasoning-mode 'Extra High' --actual-product Codex --actual-model-family GPT-5 --actual-exact-model 'not exposed by runtime' --actual-reasoning-mode 'not exposed by runtime'`.
- Privacy confirmation: no prompts, hidden chain-of-thought, token counts,
  secrets, or raw thread IDs are recorded.

## Check selection and verdicts

### Review cycle 1

- Review model / mode: pending independent re-review; actual exact model and
  reasoning mode are not exposed by runtime.
- Commit reviewed: pending repair head.
- Selected `CHECK.md` sections: minimum review; 4.1 data supply/prefetch
  order; 4.3 packing, transitions, and token accounting; 9.1 checkpoint/resume
  first-resumed-window behavior; 7.1 change surface; and 8.1 reproducibility.
- Major sections marked N/A and why: sections 5/6 (GPU, model, optimizer) are
  unchanged; this repair makes no performance claim and modifies no Hydra
  profile.
- Ticket acceptance result: repair evidence pending independent review.
- Philosophy alignment: pending independent review.
- Complexity / change-surface result: pending independent review.
- ML-system result: fixture-level cursor/order invariants pass; no DGX claim.
- Verdict: pending.

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| P2 | packed cursor | A final partial packed window is yielded while the cursor still contains the same residual tokens, so resume emits an extra full window followed by the original partial window. | Four one-token documents, `max_tokens=4`, `sequence_length=5`, `add_eos=false`, `drop_remainder=false`: full `[[311,311,311,311]]`; interrupted cursor has four residual tokens. | Clear/update the packed cursor before yielding the final short window, then demonstrate exact prefix-plus-resume equivalence. |

## Failed-review handoff

- From review cycle: post-merge P5 finding reported after DATA-003 PR #29
  merged; no new independent review verdict yet.
- Failed check and why: DATA-003's interruption/resume invariant fails for the
  final `drop_remainder: false` packed window because the cursor state says the
  window is still pending after it has been consumed.
- Review model / mode: finding supplied by the parent review flow; exact model
  and mode not exposed by runtime.
- Implementation model / mode that produced the failed state: exact model and
  reasoning mode not exposed by runtime; initial requested model/mode were Luna
  / Extra High.
- Commit/diff to repair: `57266e1e843be2d08e10ef5f387da8466b0c590f`.
- Reproduction command or evidence: fixture described in the finding; a focused
  pytest regression will retain it.
- Relevant files/config/manifests: `src/data/stream_loader/loader.py`,
  `tests/test_data003_stream_cursor.py`, canonical tokenizer fixture.
- Attempts already made: none for P5.
- Invariants and constraints: the cursor represents samples already yielded;
  `prefix + resumed` must equal uninterrupted output; sync and prefetch order
  must remain equivalent; no new execution path or compatibility shim.
- Selected next model / mode: requested Luna / Extra High repair.
- Why this model was selected: the defect is a localized packed-buffer
  lifecycle error with a deterministic reproduction.
- Exact repair request: ensure a final partial packed output consumes the
  cursor residual before `yield`, then add exact-suffix regression coverage.
- Completion evidence requested: focused and full tests, static checks, and an
  independent exact-head review.

## Repair result

- Repair cycle: 1.
- Repair model / mode: actual exact model and reasoning mode not exposed by
  runtime; requested Luna / Extra High.
- Input handoff: reported P5 four-token final-window reproduction.
- Changes made: `_packed_iter` clears `packed_buffer`/`packed_spans` before
  yielding a final short window. The same clear occurs after discarded
  remainders. Thread prefetch now uses an explicit producer-cursor capture;
  the old marker rebuilt a cursor with empty source state after the producer
  had naturally completed, which made final-window resume state incomplete.
  Added sync, thread, and process exact-suffix regressions plus the reported
  `max_tokens=4` explicit-repeat regression.
- What was deliberately not changed: source ordering, shuffle, prefetch protocol,
  quotas, model behavior, or checkpoint container design.
- Local evidence:
  - `uv run --group dev pytest -q tests/test_data003_stream_cursor.py` →
    `15 passed`.
  - `uv run --group dev pytest -q --junitxml=/tmp/data003-p5-results.xml` →
    `227 passed, 1 skipped` (228 collected; 0 failures/errors).
  - `uv run ruff check src/data/stream_loader/loader.py tests/test_data003_stream_cursor.py`,
    `uv lock --check`, and `git diff --check` → pass.
  - `uv run python scripts/debug_stream_loader.py --config config/stream_loader.yaml --limit 1`
    fails before the repair exercise because the committed fixture has a finite
    source but a five-billion-token horizon; it raises the pre-existing
    `Datasets exhausted before max_tokens quota was met` error. This PR neither
    changes that config nor claims that command as passing.
- Commit reviewed next: repair code `cae88a5855ed439ffb2265894a51bdc4306f5ec9`
  plus its current documentation descendant; independent review must name the
  exact final PR head it examines.
- Re-review model / mode: requested independent heavier Extra Thinking; actual
  exact identity/mode not exposed by runtime.
- Re-review verdict: pending.

## Final evidence

- Resolved Hydra command/config: `hydra.compose(config_name='stream_loader')`
  resolved the unchanged committed profile to `horizon.repeat=false`,
  `horizon.shuffle=true`, `horizon.shuffle_buffer_size=16`, and thread
  prefetch with buffer size 16. The focused fixture uses direct mappings to
  isolate the packed-cursor invariant.
- Data/tokenizer/model identity: existing canonical tokenizer fixture; no model
  or data manifest change.
- Validation and measurements: focused sync/thread/process exact-suffix
  regression is green (15 tests total); full repository suite is green (227
  passed, 1 skipped); static checks are green. The default debug command's
  unrelated finite-fixture/5B-horizon failure is retained above.
- Performance/resource result if applicable: N/A — no performance claim;
  CHECK R1 fixture semantics is the relevant validation size.
- Failed attempts retained at: this record's P5 finding and regression.
- Known trade-offs: final short windows are terminal within their pass, so their
  whole residual must be absent from a checkpoint cursor. Cursor capture remains
  bounded by the existing shuffle buffer; no new buffer or config branch was
  added.
- Unresolved risks: independent exact-head review and guarded merge audit remain
  required.
- Human decision requested: none before the independent review; self-merge is
  only eligible after every guarded gate passes.

## Merge authority and final audit

- Merge path: guarded agent self-merge only after the exact-head review and
  audit; otherwise human merge.
- Human authorization: user stated on 2026-07-12, `これからはとりあえず全部セルフマージしていいよ` (authorizing self-merge for the bounded roadmap-completion goal, after review).
- Authorization evidence location: conversation instruction; final PR audit
  must restate the scope and exact-head observation.
- Authorization covers this named PR or bounded ticket/goal series: yes —
  DATA-003 P5 is ordinary repository collaboration within the roadmap goal.
- Exact independently reviewed head SHA: pending.
- Latest independent verdict / model / mode: pending.
- All actionable findings repaired and independently re-reviewed: pending.
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: pending
  final refresh.
- Newer human objections since authorization/review: pending final refresh.
- Human review dismissed by an agent: no.
- Unresolved review threads at final audit: pending.
- Branch-protection required-context inventory: pending final refresh.
- Applicable configured workflow/check inventory: pending final refresh.
- Observed exact-head check statuses: pending final refresh.
- Expected checks absent, pending, skipped, cancelled, or non-successful: pending.
- No-check evidence when both inventories are empty: pending.
- Target branch and base SHA at final audit: pending.
- Up-to-date, conflict-free, and mergeable evidence: pending.
- Record, ledger, PR trail, validation, and risks parity: pending.
- Prohibited self-merge categories: clear so far — ordinary data-loader logic,
  tests, and documentation only; no secrets, security, paid resource, private
  data, deployment, release, or permission change.
- Admin/bypass/force/disabled-check requirement: no.
- Final audit PR body/comment location: pending.
- Final audit changed reviewed head: no (required).
- Immediate pre-merge re-fetch/compare observation location: pending.
- Immediate refresh compared authorization, head, base, review decision/objections,
  threads, expected checks/statuses, and mergeability: pending.
- Drift found: pending.
- Merge outcome: pending.

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| Codex / GPT-5; exact ID and mode not exposed | repair | Initial scoped reproduction and repair handoff | Pending implementation/review | DATA-003 P5 sequence, loader cursor state, selected `CHECK.md` sections | in progress |

## Ledger update

- [x] Added this per-PR record to `docs/model-runs/README.md`.
- [ ] Updated aggregate counts after independent review.
- [ ] Confirmed the PR execution trail matches this record.
- [ ] Recorded guarded self-merge audit or human merge evidence.
- [x] Confirmed this is not the bootstrap policy PR.
