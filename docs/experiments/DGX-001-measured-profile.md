# DGX-001 — Measured model profile and time/token budget

- Roadmap ticket: `DGX-001`
- Branch: `codex/dgx-001-main-integration`
- Draft PR: deferred until the explicit `WB-001` dependency is merged
- Status: implementation and bounded target validation in progress
- Baseline input: stacked `WB-001` integration head
  `05afccd723df6a438458b0b4c448325a9811ed85`

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
deterministic rule then selects the deepest candidate within 20% of the fastest,
the longest context retaining at least 85% of that model's fastest arm, and the
lower-UMA arm within a 3% throughput tie.

Any OOM, non-finite value, swap growth, monotonic allocator growth, temperature
above 80 C, sampler gap, missing CUDA timing, missing/failed checkpoint, or
insufficient disk/UMA evidence fails that arm. The runner does not reduce batch,
fall back to CPU, enable compilation, or retry a different configuration under
the same arm identity.

## Implementation

- `profile=dgx_smoke` is a twelve-update real-data wiring proof.
- `profile=dgx_candidate` is the repeated timed training path.
- `profile=pretrain_baseline` is the selected-shape one-hour cap with online
  W&B, watch off, artifact policy `none`, 5M-target validation, 2.5M-target
  rotating recovery, and 100M-target milestones.
- `scripts/measure_dgx.py` runs the canonical trainer, records the resolved
  config/environment, samples host/GPU state out of band, and verifies the
  final checkpoint. Pilot mode also records Japanese and English base-model
  continuations from that exact checkpoint.
- `scripts/run_dgx_measurements.py` checks clean source/image identity, executes
  the rotated triplicate matrix in the network-isolated pinned container, and
  preserves exact commands.
- `scripts/summarize_dgx_measurements.py` gates every raw run and writes the
  repeat statistics, selection, token/checkpoint plan, and named bottleneck.

No compilation, native extension, custom kernel, architecture change, or
optimization ticket is introduced. The selected profile's largest phase is
named as the current bottleneck, but optimization remains deferred until
`RUN-001` confirms that it matters in the first trustworthy baseline.

## Exact commands and evidence

Local implementation/config validation:

```text
uv run pytest -q tests/test_dgx_planning.py tests/test_config_profiles.py
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
make dgx-pilot EXPECTED_COMMIT="$HEAD" \
  OUTPUT_ROOT="/tmp/dgx-001-pilot-$HEAD" \
  CACHE_ROOT="/absolute/path/to/hash-verified/stream_loader_cache" \
  SELECTED="<summary candidate_id>"
```

The target result section remains open until the exact-head matrix and pilot
complete. A smoke or synthetic unit test is wiring evidence only and cannot be
used to claim the measured profile is selected.

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

Independent `/review` will cover `PHILOSOPHY.md`, DGX-001 acceptance, and the
applicable `CHECK.md` minimum, comparison, data supply, DGX/UMA, training-health,
W&B, checkpoint/storage, and changeability sections at the exact proposed head.
