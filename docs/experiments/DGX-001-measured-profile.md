# DGX-001 — Measured model profile and time/token budget

- Roadmap ticket: `DGX-001`
- Branch: `codex/dgx-001-final-integration`
- Draft PR: [#47](https://github.com/Ayumu-J-S/llm_scratch/pull/47)
- Status: pre-measurement protocol repaired; exact-head re-audit pending
- Baseline input: merged `WB-001` head
  `8791bb7237663b08c001b732393a76b240362476`

## Question and predeclared decision

Choose the conventional model depth, context, micro-batch, and initial run
duration from repeated GB10 measurements, while retaining unified-memory,
checkpoint, validation, and storage headroom.

The candidate matrix is declared in `config/dgx.yaml`. Each of its nine arms
uses the same pinned tokenizer/data mixture, strict seed/determinism, BF16 CUDA,
AdamW objective, 32,768 trained targets per optimizer update, ten warm-up
updates, twenty measured updates, and three repetitions. The only experimental
variables are decoder depth and the paired context/micro-batch choice.

Success requires all repetitions at one clean commit and image to pass the
resource/numerical/evidence gates. The conservative (slowest-repetition)
throughput must forecast at least one billion targets in seven days. The
deterministic rule then selects the deepest candidate within 20% of the fastest
passing arm and the longest context retaining at least 85% of that depth's
fastest arm. A non-unique result is rejected rather than broken by an undeclared
tie rule.

Any OOM, non-finite value, swap growth, monotonic allocator growth, temperature
above 80 C, sampler gap, missing CUDA timing, missing/failed checkpoint, or
insufficient disk/UMA evidence fails that arm. The runner does not reduce batch,
fall back to CPU, enable compilation, or retry a different configuration under
the same arm identity.

## Implementation

- `profile=dgx_candidate` is the repeated timed training path.
- `profile=pretrain_baseline` carries the provisional P70 shape only so the
  exact final matrix can enforce that the committed profile matches its result.
  It is not a selection claim. The profile is a one-hour cap with online W&B,
  watch off, artifact policy `none`, 5M-target validation, 2.5M-target rotating
  recovery, and 100M-target milestones.
- `scripts/measure_dgx.py` runs the canonical trainer, records physical
  config/manifest/checkpoint identities, samples host/GPU state out of band,
  enforces pilot watchdog limits, and binds the final checkpoint to the
  trainer's complete schema-v3 measurement chain. Pilot mode also records
  Japanese and English base-model continuations from that exact checkpoint.
- `scripts/measure_dgx_decomposition.py` measures three 10+20-update
  repetitions of the device-resident model path and real streaming loader path
  for the summary-selected profile.
- `scripts/run_dgx_measurements.py` checks clean source/image identity, executes
  the rotated triplicate matrix in the network-isolated pinned container as the
  host UID:GID, preserves commands and logs, hashes the immutable data cache
  before and after, and refuses auxiliary runs without a passing selection
  summary for the same commit. The pilot mounts host W&B authentication
  read-only without copying or serializing it.
- `scripts/summarize_dgx_measurements.py` gates every raw run and writes the
  repeat statistics, selection, pause-aware token/checkpoint/storage plan, and
  named bottleneck.

No compilation, native extension, custom kernel, architecture change, or
optimization ticket is introduced. The final summary will name the selected
profile's nearer measured ceiling, but optimization remains deferred until
`RUN-001` confirms that it matters in the first trustworthy baseline.

## Exact commands and evidence

Local implementation/config validation:

```text
uv run pytest -q tests/test_dgx_planning.py tests/test_dgx_runner.py \
  tests/test_config_profiles.py
uv run ruff check src/dgx scripts/measure_dgx.py \
  scripts/run_dgx_measurements.py scripts/summarize_dgx_measurements.py \
  tests/test_dgx_planning.py
PYTHONPATH=src uv run python scripts/run_dgx_measurements.py mode=plan
```

Target matrix, summary, and 30-minute pilot:

```text
HEAD=$(git rev-parse HEAD)
make dgx-measurements EXPECTED_COMMIT="$HEAD" \
  OUTPUT_ROOT="/tmp/dgx-001-$HEAD" \
  CACHE_ROOT="/absolute/path/to/hash-verified/stream_loader_cache"
make dgx-summarize OUTPUT_ROOT="/tmp/dgx-001-$HEAD"
make dgx-decompose EXPECTED_COMMIT="$HEAD" \
  OUTPUT_ROOT="/tmp/dgx-001-decomposition-$HEAD" \
  CACHE_ROOT="/absolute/path/to/hash-verified/stream_loader_cache" \
  SELECTED="<summary candidate_id>" \
  MATRIX_SUMMARY="/tmp/dgx-001-$HEAD/dgx-summary.json"
make dgx-pilot EXPECTED_COMMIT="$HEAD" \
  OUTPUT_ROOT="/tmp/dgx-001-pilot-$HEAD" \
  CACHE_ROOT="/absolute/path/to/hash-verified/stream_loader_cache" \
  SELECTED="<summary candidate_id>" \
  MATRIX_SUMMARY="/tmp/dgx-001-$HEAD/dgx-summary.json"
```

The target result section remains open until the exact-head matrix,
decomposition, and pilot complete. Historical matrix evidence is explicitly
`INCOMPLETE`; its provisional P70 rule result is not a selection. A smoke,
single sweep, or synthetic unit test is wiring evidence only and cannot be used
to claim that a measured profile has been selected.

## Review trail

| Cycle | Phase | Outcome | Important finding or change | Evidence |
| ---: | --- | --- | --- | --- |
| 1 | Implementation | in progress | Thin profiles/runner/summarizer over the canonical trainer and VAL/WB measurement hooks | Focused tests and plan composition |
| 2 | Target smoke attempt 1 | failed before data/model construction | Container Git rejected the read-only host worktree as dubious ownership; runner needs an ephemeral exact-path safe-directory setting | `/tmp/dgx-001-smoke-eb043b4/run.json` |
| 3 | Repair | implemented | Pass a container-only `safe.directory` value for the exact mounted worktree; clean status and exact commit remain required | Runner command and repeat target smoke |
| 4 | Target smoke attempt 2 | failed before model construction | Exact Git identity passed; the isolated worktree cache was empty and network isolation rejected an implicit corpus download | `/tmp/dgx-001-smoke-6a80d43/run.json` |
| 5 | Repair | implemented | Require an explicit existing `cache_root` and mount that hash-verified cache read/write into every matrix/pilot container | Runner preflight and repeat target smoke |
| 6 | Target smoke attempt 3 | passed | Real pinned JA/EN stream, 70,828,682 parameters, 12 BF16 CUDA updates/49,152 targets, verified checkpoints, finite validation, 12,891 post-warmup targets/s, 3.13% data wait, 3.88 GB allocator peak, 48 C max, zero swap | `docs/experiments/evidence/DGX-001-smoke-909e7b9.json` |
| 7 | Evidence projection repair | implemented | Use VAL-001's exact `full_event_pause_seconds` field so validation overhead is not understated | Smoke measurement validation row and focused test |
| 8 | Candidate matrix attempt 1 | stopped after complete first sweep | All nine model/context arms completed on exact head `d15f4a3`; final VAL announced schema-v2 segmented measurement/resume evidence, so repetitions 2-3 would not be final acceptance evidence. Repetition 2 had begun and was interrupted without a checkpoint. | `docs/experiments/evidence/DGX-001-matrix-r1-d15f4a3.json` |
| 9 | Projection repair | implemented | First-sweep P70 allocations showed a stable periodic low/high pattern. The monotonic-growth gate now compares the lower envelope of the first/last five measured steps instead of rejecting peak-to-trough oscillation. | Raw per-step allocator rows and regression test |
| 10 | Independent pre-measurement review | `FAIL` at `3abb62d` | Historical schema-v1 evidence could not authorize the final matrix; auxiliary authority, physical checkpoint binding, pause-aware throughput, full telemetry/watchdog, decomposition, storage, host ownership, and exact-run constraints needed to be made machine-verifiable before more GPU use | Draft PR review trail and repair handoff |
| 11 | Protocol repair | implemented; re-audit pending | Replaced the stale evidence parser with direct schema v3; bound plan/source/cache/config/manifest/checkpoint identities; declared the exact 9x3 10+20 matrix; added repeated decomposition and a representative online-W&B pilot; made throughput pause-aware; added telemetry coverage/health, repeatability, storage, and committed-profile gates; retained per-run logs | Focused planning/runner/config tests and exact commands above |

Independent `/review` will cover `PHILOSOPHY.md`, DGX-001 acceptance, and the
applicable `CHECK.md` minimum, comparison, data supply, DGX/UMA, training-health,
W&B, checkpoint/storage, and changeability sections at the exact proposed head.
