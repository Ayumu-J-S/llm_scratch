# DGX-001 — Measured model profile and time/token budget

- Roadmap ticket: `DGX-001`
- Branch: `codex/dgx-001-final-integration`
- Draft PR: [#47](https://github.com/Ayumu-J-S/llm_scratch/pull/47)
- Status: formal pre-measurement review findings repaired; fresh exact-head
  `/review` pending, with no target compute authorized before it passes
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

The human-required machine reserve is 100 GB. Preflight and active telemetry use
a stricter 120,000,000,000-byte free-disk stop threshold across both cache and
output filesystems. Preflight includes projected cache growth and retained
output growth; the final storage verdict separately requires at least
100,000,000,000 bytes free after the complete plan, in addition to the 2x
headroom rule. Output forecasting includes three rotating recovery checkpoints,
one atomic temporary, best, final, every 100M-target milestone, and retained
logs/evidence. No destructive cleanup is part of this ticket.

## Implementation

- `profile=dgx_candidate` is the repeated timed training path.
- `profile=pretrain_baseline` carries the provisional P70 shape only so the
  exact final matrix can enforce that the committed profile matches its result.
  It is not a selection claim. The profile keeps optimizer work to 3,480
  seconds plus a 120-second finalization reserve inside the one-hour wall
  budget, with online W&B, watch off, artifact policy `none`, 5M-target
  validation, 2.5M-target rotating recovery, and 100M-target milestones. A
  candidate passes only when twice its worst observed update/event tail plus
  final checkpoint fits that reserve; the pilot independently gates observed
  training plus final-checkpoint time against its 1,800-second wall budget.
- `scripts/measure_dgx.py` runs the canonical trainer, records physical
  config/manifest/checkpoint identities, samples host/GPU state out of band,
  enforces fail-closed watchdog limits for matrix and pilot roles, and binds the
  final checkpoint to the trainer's complete schema-v3 measurement chain. Pilot mode also records
  Japanese and English base-model continuations from that exact checkpoint.
- `scripts/measure_dgx_decomposition.py` measures three 10+20-update
  repetitions of the device-resident model path and real streaming loader path
  for the summary-selected profile. Both roles use the same fail-closed hard
  resource watchdog and preserve their failed evidence on interruption.
- Telemetry evidence I/O is itself fail-closed: open, sample, serialization,
  write, flush, fsync, close, and unexpected worker failures remain available
  through in-memory control state, interrupt the workload, and are surfaced by
  `stop()` even when the evidence file cannot describe its own failure.
- `scripts/run_dgx_measurements.py` checks clean source/image identity, executes
  the rotated triplicate matrix in the network-isolated pinned container as the
  host UID:GID, preserves commands and logs, hashes the immutable data cache
  before and after, and refuses auxiliary runs without a passing selection
  summary for the same commit. The pilot mounts host W&B authentication
  read-only without copying or serializing it.
- Every matrix, decomposition, and pilot role has a plan-authorized canonical
  hash of its complete resolved Hydra configuration plus the manifest's
  scientific experiment-config identity. Auxiliary runs also bind the physical
  matrix plan and summary, exact plan ID, protocol, selection rule, commit, and
  image rather than trusting a self-reported candidate.
- Every role also binds an exact parameter count and a conservative atomic
  checkpoint-write budget (128 bytes per parameter plus 4 GB fixed overhead).
  The operational disk floor is the greater of 120 GB and the required 100 GB
  reserve plus that budget; all current candidates stay below the static 20 GB
  buffer, while a larger future shape raises the floor automatically.
- `scripts/summarize_dgx_measurements.py` gates every raw run and writes the
  repeat statistics, selection, pause-aware token/checkpoint/storage plan, and
  named bottleneck. Data wait is divided only by optimizer-step wall time;
  projections charge scheduled scalar logging; decomposition compares isolated
  roles with the selected candidate's conservative compute throughput and uses
  the 1.2x threshold only for loader supply.

No compilation, native extension, custom kernel, architecture change, or
optimization ticket is introduced. The final summary will name the selected
profile's nearer measured ceiling, but optimization remains deferred until
`RUN-001` confirms that it matters in the first trustworthy baseline.

## Exact commands and evidence

Local implementation/config validation:

```text
uv run pytest -q tests/test_dgx_planning.py tests/test_dgx_runner.py \
  tests/test_dgx_telemetry.py tests/test_dgx_watchdog_roles.py \
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
| 12 | Independent pre-measurement re-review | `FAIL` at `63283cd` | Docker still launched a mutable tag after verifying its ID; measured rows did not require complete finite CUDA phases; cache/output capacity could be conflated across filesystems; telemetry scheduling was unnecessarily ambiguous after overruns; pilot W&B gates ignored scalar/summary circuit-breaker failures | Exact review findings retained in PR #47 |
| 13 | Protocol repair 2 | implemented | Launch by verified image digest; reject missing/non-finite/negative required CUDA phases; group cache/output plans by real filesystem device with separate headroom; schedule telemetry from collection completion; physically verify and parse W&B evidence with scalar/summary failure gates; make ample/low/split-filesystem tests deterministic | Focused and full validation recorded in PR #47 |
| 14 | Independent re-review | `blocked` | The required exact-head `/review` cannot run again until the shared review-service quota resets at 2026-07-25 23:43 UTC. No PASS is claimed, and target GPU work remains blocked. | PR #47 review handoff |
| 15 | Supplemental audit | `FAIL` at `a828b74` | A self-consistent mutation could evade subset-only config checks; auxiliary selection did not bind the physical source matrix plan/protocol strongly enough; clock, power, and utilization could be absent | Exact findings retained in PR #47 |
| 16 | Protocol repair 3 | implemented | Predeclare and verify complete per-role resolved/experiment config hashes; bind auxiliary evidence to the exact physical matrix plan and summary; require finite nonnegative clock/power/utilization coverage; add adversarial config and authority tests | Focused DGX tests |
| 17 | Human operational constraint | implemented | Preserve at least 100 GB machine availability using a 120 GB preflight/watchdog floor, projected cache/output growth grouped by filesystem, and an independent 100 GB post-plan reserve gate; no destructive cleanup | Same/split filesystem, low-space, and independent-reserve tests |
| 18 | Supplemental exact-head re-audit | `FAIL` at `e858bf3` | The live 120 GB watchdog interrupted only pilot; matrix/model-only/loader-only could record a 119 GB violation, return success, and allow later commands to continue | Auditor CPU reproduction and PR #47 trail |
| 19 | Protocol repair 4 | implemented; re-audit pending | Arm fail-closed interruption for every role, add decomposition hard preflight, preserve failed run/telemetry evidence, stop runner sequencing after any nonzero role, and bind the 100 GB reserve to a conservative per-role atomic-write budget that dynamically raises the live floor when needed | Role-level low-disk interruption, runner-incomplete, budget-boundary, and authority-tamper tests |
| 20 | Supplemental exact-head re-audit | `FAIL` at `016b323` | Evidence-file open/write/flush/fsync or malformed-sample failure could kill only the telemetry worker, leaving the workload un-interrupted and no machine-readable error | Exact `/dev/full` reproduction and PR #47 trail |
| 21 | Protocol repair 5 | implemented; re-audit pending | Contain the complete telemetry worker lifecycle, retain fatal state independently of the evidence file, interrupt every armed workload, and make `stop()` reject captured failure or unexpected thread death | Open/write/flush/malformed-sample/fsync/close/`/dev/full` adversarial tests |
| 22 | Exact-head CI | `FAIL` at `0fc8703` | The fsync adversarial test used host GPU sampling, so x86 CI correctly failed closed on missing `nvidia-smi` before reaching the intended fsync assertion | Actions run `29682842745` |
| 23 | CI fixture repair | implemented | Pin the fsync test to a synthetic safe sample so it exercises only the intended finalization failure on every platform | Full and focused local suites |
| 24 | Exact-head CI and supplemental audit | `PASS` at `0f13389` | The repaired telemetry lifecycle, all-role watchdog, storage reserve, exact config authority, and deterministic fixture passed; this supplemental pass did not replace formal `/review` | PR quality run `29683131461` and PR #47 trail |
| 25 | Formal pre-measurement `/review` | `FAIL` at `0f13389` | Data-wait and decomposition used pause-inclusive throughput; model-only was incorrectly subject to the loader threshold; projections omitted scheduled logs; the one-hour profile left its final checkpoint outside the cap; runtime pilot docs omitted `MATRIX_SUMMARY` | Exact review findings retained in PR #47 |
| 26 | Protocol repair 6 | implemented; re-review pending | Use optimizer-step wall for data wait, conservative compute throughput for decomposition, loader-only headroom gating, log-aware token budgets, a measured 2x-tail 120-second finalization reserve inside both wall plans, and a complete pilot command | Focused DGX/config suite: 84 passed; targeted planning/runner/watchdog suite: 49 passed |
| 27 | Full CPU validation | passed | Network-free full suite, lint, Hydra config preflight, lock-drift rejection, offline disabled/W&B smoke, exact 27-arm plan composition, and baseline resolved config all pass; no GPU or online W&B work ran | `make ci-cpu`: 503 passed, 1 skipped; about 425 GB free |

Independent `/review` will cover `PHILOSOPHY.md`, DGX-001 acceptance, and the
applicable `CHECK.md` minimum, comparison, data supply, DGX/UMA, training-health,
W&B, checkpoint/storage, and changeability sections at the exact proposed head.
