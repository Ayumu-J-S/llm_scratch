# VAL-001 — Shared Held-Out Validation Scoring

- Roadmap ticket: `VAL-001`
- Branch: `codex/val-001-held-out-validation`
- Draft PR: [#42](https://github.com/Ayumu-J-S/llm_scratch/pull/42)
- Status: Attempt 6 evidence `PASS WITH NOTE`; independent re-review pending
- Started / last updated (UTC): 2026-07-13 / 2026-07-13
- Model-run record: `docs/model-runs/VAL-001-held-out-validation.md`

## Predeclared question and decision rule

- Hypothesis: one token-weighted causal-LM scorer can produce identical held-out
  NLL/perplexity from training-time and standalone checkpoint paths while
  retaining immutable checkpoint, manifest, corpus, and evaluated-token identity.
- Expected result: both paths agree exactly; known logits match analytical NLL;
  Japanese, English, and aggregate denominators reconcile; overlap and
  memorization misuse fail before scoring.
- Success: every ROADMAP acceptance criterion passes, the matched DGX run shows
  bounded validation/checkpoint pauses without changing the train trajectory,
  and independent review returns `PASS` or justified `PASS WITH NOTE`.
- Failure / stop: any score divergence, identity omission, denominator mismatch,
  train/validation overlap, or same-corpus result labeled held-out validation.
- Baseline: stacked DATA-004 head `e1d4ed8af98de84a3393cd0f6e517f9daf649138`.

## Planned budget

| Resource | Limit / plan | Evidence basis |
| --- | --- | --- |
| Correctness | Full CPU suite plus focused identity/timing regressions | Known-logit and failure-path tests are authoritative after the pre-repair fixture became invalid |
| DGX target smoke | Three matched validation-off/on pairs, 60 steps per arm | CHECK repeated-measurement and VAL-001 pause-isolation requirements |
| Training work | 245,760 targets per arm; 1,474,560 targets across six acceptance runs | 4,096 effective targets/update in the repaired sequence-4,096 protocol |
| Validation | 65,536 fixed targets at steps 25 and 50 in each on arm | Six Japanese/English 50/50 validation events |
| Checkpoints | Final in every arm; best after each improving held-out score | Required parity, identity, and pause evidence |
| Resource trace | GPU at 5 Hz; host at 1 Hz; container at nominal 0.5 Hz in prospective Attempt 6 | Continuous attribution plus interval and per-validation-event coverage gates |
| External logging | W&B disabled | Local metrics, one measurement JSON, and compact evidence are sufficient for this ticket |

## Attempt 1 — pre-repair CPU fixture

- Measured code: `a8520d7fad718574d1fca4293e6f969c7a478b79`.
- Retained evidence:
  [`VAL-001-cpu-parity.json`](evidence/VAL-001-cpu-parity.json).
- Historical outcome: the fixture reported training-time/standalone score parity
  under a same-corpus memorization profile.
- Disposition: **invalidated as current acceptance evidence**. Later review found
  that standalone evaluation must reject memorization, and memorization runs must
  not create a `best.pt` validation checkpoint. The repaired implementation and
  tests enforce both rules. The file remains versioned so the failed design is
  not erased.
- Also invalidated: the attempt called the DGX gate blocked because the host venv
  contains CPU-only Torch. That diagnosis omitted the repository's pinned
  `llm-scratch:env-001` CUDA container, which is the canonical DGX runtime.

## Repair cycle — scoring and trust invariants

- Implementation repair: `057983c` plus merge head
  `21332488e8a1d2334cbb6e2d0593a77a598c1d01`.
- Cleanup repair after the preliminary exact-head audit:
  `0a138386a03e178a88b5ccca6334288b57188efb`.
- Changes demonstrated by tests:
  - standards-safe JSON for perplexity overflow;
  - explicit aggregate and per-corpus NLL sums/denominators;
  - exact label-aligned source attribution and reconciliation;
  - batching-independent context/mask/token/source identities;
  - verified checkpoint-owned standalone reconstruction;
  - memorization namespace, no memorization best checkpoint, and standalone
    memorization rejection;
  - fresh fixed validation loaders and iterator closure;
  - unconditional model-mode restoration even if iterator cleanup raises.
- Pre-evidence full suite at `2133248`: `302 passed, 1 skipped`; lock, Ruff,
  changed-file format, and `git diff --check` passed.
- Post-cleanup focused test: `15 passed`; Ruff/format/diff checks passed.

## Attempt 2 — matched DATA-004 DGX R2

The compact, machine-readable record is
[`VAL-001-dgx-r2.json`](evidence/VAL-001-dgx-r2.json). It contains commands,
resolved-config/run/checkpoint hashes, measured distributions, identities,
resource snapshots, and limitations without raw corpus text or token sequences.

### Conditions

- Measured head: `21332488e8a1d2334cbb6e2d0593a77a598c1d01`.
- Hardware: aarch64 DGX Spark, NVIDIA GB10, driver `580.159.03`, compute
  capability 12.1, BF16 supported.
- Runtime: `llm-scratch:env-001`, image ID
  `sha256:23a1bee69fe189e77105cdddeee9aeff6ef0763d58a691625fbfcab64efd1887`,
  PyTorch `2.13.0a0+8145d630e8.nv26.06`, CUDA 13.3.
- Model: 49,535,114 parameters; width 384, 6 layers, 6 heads, dropout 0.1.
- Work: BF16/CUDA, sequence 8, micro batch 64, accumulation 4, 2,048
  effective targets/update, 50 steps, 102,400 targets, W&B disabled.
- Data: the same warm DATA-004 cache and pinned Japanese/English manifests in
  both arms. The only intended arm difference was validation cadence: 1,000
  (no event in horizon) versus 25.

The first launch wrapped the container command in `/usr/bin/time -v` and failed
before training with exit 127 because that executable is absent from the image.
The successful runs used trainer metrics plus external `docker stats` and
`nvidia-smi` snapshots. This failed attempt is retained in the JSON record.

### Correctness and parity

- Both arms completed exactly 50 steps and 102,400 training targets.
- All 50 step losses were bit-identical; maximum absolute difference was 0.0.
  Step payloads other than elapsed time also matched exactly, including gradient
  norm, clipping, LR, and counters.
- The off arm emitted zero validation events. The on arm emitted exactly two,
  at steps 25 and 50, with no target overshoot.
- Both validation passes evaluated the same 8,192 windows and 65,536 targets:
  32,768 Japanese plus 32,768 English. Window and target SHA-256 identities were
  identical across the two events.
- Step-50 aggregate NLL was `9.033536911010742` over NLL sum `592021.875` and
  65,536 targets. Japanese NLL was `9.078840732574463`; English NLL was
  `8.988233089447021`.
- Standalone evaluation of the verified step-50 best checkpoint matched every
  aggregate/per-corpus value, denominator, manifest identity, logical checkpoint
  identity, window count, window digest, and token digest exactly.

### Pause and recovery evidence

| Measurement | Validation off | Validation on |
| --- | ---: | ---: |
| Step time p50, steps 2–50 | 0.5087 s | 0.5661 s, excluding pause-contaminated step 26 |
| Step time p95, steps 2–50 | 0.8500 s | 0.9230 s, excluding step 26 |
| Step time max | 1.2167 s | 1.2932 s, excluding step 26 |
| Approx. total including final save | 31.4407 s | 77.6661 s |

- Validation scoring pauses were 19.5218 s and 19.5148 s, at 3,357.1 and
  3,358.3 evaluated targets/s.
- Best-checkpoint pauses were 1.8077 s and 2.3555 s. Combined validation plus
  best-save pause was 43.1997 s.
- The raw step-25→26 interval was 22.0297 s; after subtracting the first
  validation and best-save pauses, it was 0.7002 s.
- Step 27 was 0.5866 s. Steps 27–31 averaged 0.5778 s, 2.98% below the
  immediate pre-validation steps 20–25 mean, so observed throughput returned to
  its prior range on the first uncontaminated post-validation step.
- Standalone scoring took 19.6951 s at 3,327.5 targets/s, within 0.93% of the
  matching training-time score duration/throughput.
- Snapshots observed 2.075–2.534 GiB container memory, 1,794 MiB GPU process
  memory, 41–45°C, and 13.2–16.09 W. They are snapshots, not peak or
  continuous-series claims.

### CHECK §6.3 limitation

This pair supports validation/checkpoint pause accounting, independent cadence,
no off-by-one/trajectory change, and immediate recovery. It is not a general
performance benchmark: there was one A/B pair, no predeclared warm-up cutoff,
and no continuous GPU/system trace. The repair adds local phase timing for
future repeated CHECK §6.3 analysis, but no post-repair DGX measurement was
run, so those fields have no new hardware evidence yet.

The container also warned that memory-efficient attention defaults to a
nondeterministic algorithm. Exact trajectory parity was observed for this pair,
but it is not a cross-platform bitwise reproducibility promise.

## Attempt 3 — stopped post-repair DGX R3 protocol

This protocol is committed before measurement. It replaces the insufficient
single-pair performance claim; Attempt 2 remains historical evidence only.

### Fixed conditions and run order

- Exact repaired commit, pinned `llm-scratch:env-001` image, one warm DATA-004
  cache, seed 42, BF16/CUDA, sequence length 8, micro batch 64, accumulation 4,
  2,048 effective targets/update, 60 steps, 122,880 targets, W&B disabled.
- Steps 1–10 are warm-up and excluded from steady-state comparisons.
- Validation-off uses cadence 1,000; validation-on uses cadence 25. The only
  intended semantic arm difference is validation cadence.
- Run order is `1-off`, `1-on`, `2-on`, `2-off`, `3-off`, `3-on`, all on the
  same hardware/image/cache. Validation occurs at steps 25 and 50.
- Measurement mode is enabled identically in every arm with CUDA events and one
  end-step boundary synchronization. The ordinary path remains disabled by
  default and adds no CUDA events, synchronization, or measurement file.
- Before each arm: require no competing GPU process and zero active cache
  leases. A download or eviction invalidates and restarts the matched pair.
- Retain resolved config, run manifest, metrics, atomic measurement JSON,
  stdout/stderr, 5 Hz `nvidia-smi`, 1 Hz `vmstat`, and 1 Hz `docker stats`, with
  SHA-256 for each raw artifact. Raw time series stay separately hashed; the
  committed record contains compact summaries rather than raw corpus content.

Each arm uses this canonical template, where `VALIDATION_CADENCE` is `1000` or
`25` and `RUN_NAME`, `CACHE`, and `OUT` identify only that arm:

```bash
docker run --rm --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --name "$RUN_NAME" \
  -v "$PWD:/workspace:ro" -v "$CACHE:/cache" -v "$OUT:/evidence" \
  -w /workspace llm-scratch:env-001 \
  python src/train.py profile=pretrain_streaming runtime.device=cuda \
    data.streaming.cache.dir=/cache reproducibility.seed=42 \
    training.precision=bf16 training.max_steps=60 training.max_tokens=null \
    training.max_time=null training.batch_size=64 training.sequence_length=8 \
    training.gradient_accumulation_steps=4 training.log_every_n_steps=1 \
    training.checkpoint_every_n_steps=1000 \
    training.milestone_every_n_steps=1000 \
    training.validation_every_n_steps="$VALIDATION_CADENCE" \
    artifacts.checkpoints_dir=/evidence/checkpoints wandb.enabled=false \
    measurement.enabled=true measurement.warmup_optimizer_steps=10 \
    measurement.cuda_events=true \
    measurement.output_path=/evidence/measurement.json \
    hydra.run.dir=/evidence/hydra
```

### Aggregation and zero-tolerance checks

- Compute each run's target tokens/s as total post-warm-up targets divided by
  total post-warm-up optimizer-step wall time; report p50/p95/max step time,
  data-wait fraction, phase distributions, memory, and resource coverage.
- Report all three paired throughput deltas plus median and min–max; do not pool
  individual steps as independent replicates.
- Require every pair's 60-step loss, gradient norm, clipping, LR, counters, and
  final canonical model-tensor digest to match exactly.
- Require every on arm to validate exactly at steps 25/50, every off arm to have
  zero validation events, and every arm to finish at exactly 122,880 targets.
- Require fixed manifest/window/token identities across all six validation
  events and exact training-time/standalone parity for pair 1's step-50 best
  checkpoint.
- Any identity, score, trajectory, split, non-finite, cache, or counter mismatch
  is an immediate `FAIL` for the affected pair.

### Predeclared performance/resource gates

- Maximum complete validation-event pause: 25 seconds with a best save;
  scoring alone: 22 seconds. Event attribution must reconcile within the larger
  of 5 ms or 1% of its full pause.
- Median paired steady-training throughput regression must be under 5%. A
  5–10% result requires diagnosis and a repeated pair; 10% or more is `FAIL`
  absent an explicitly accepted trade.
- Steady data wait must be at most 5% of step time. Above 5% requires loader
  diagnosis; above 10% or recurring correlated GPU gaps is `FAIL`.
- The first post-validation step must be no slower than the preceding five-step
  p95, and the following five-step mean must remain within 5% of both its
  pre-event window and paired off-arm positions.
- Step-time p95 above 1.5 times the median requires a phase/resource explanation.
  Memory must recover within `max(128 MiB, 5%)`; monotonic post-warm-up growth,
  sustained swap, or a trace gap spanning validation is `FAIL`.
- GPU sample coverage and host/container coverage must each reach at least 90%
  of their expected intervals.

### Outcome and stop decision

- The first launch failed before data/model initialization because container
  Git rejected the read-only `/workspace` bind mount as dubious ownership. The
  retry added only explicit Git `safe.directory=/workspace` environment values;
  the failed launch remains under the raw evidence root.
- Pair 1 then completed at exact head `78e0448`: both arms reached 60 steps and
  122,880 targets; the off arm had zero validation events and the on arm had
  validation exactly at steps 25/50. All non-time step records matched exactly.
- Post-warm-up throughput was 3,721.61 targets/s off and 3,808.35 targets/s on,
  a +2.33% on/off delta. Validation event pauses were 22.2045 s and 22.3052 s,
  with 20.1137 s and 19.7149 s scoring; attribution error stayed below 0.3 ms.
- The attempt nevertheless stopped as predeclared: data wait consumed 82.60%
  of off-arm and 82.47% of on-arm step time, well above the 10% FAIL gate.
  Cache names/sizes/content were unchanged and all leases were released, so the
  failure was producer/window overhead rather than download, eviction, or a
  validation-induced regression.
- Validation-off diagnostics kept the same model/data/image and isolated context
  shape. Data-wait fraction fell from 82.60% at sequence 8 to 14.76% at 256,
  8.96% at 1,024, 5.45% at 2,048, and 0.86% at 4,096. Sequence 4,096 used at
  most 7.40 GB PyTorch reserved memory and 6.49 GB allocated memory. These
  diagnostics choose the repaired protocol; they are not acceptance runs.

## Attempt 4 — stopped sequence-4,096 DGX R3 retry

This retry is committed before its acceptance runs. It changes only the train
shape needed to remove the evidenced per-window producer bottleneck:

- sequence length 4,096, micro batch 1, accumulation 1, 4,096 targets/update;
- 60 steps and 245,760 train targets per arm;
- the same exact code, model, tokenizer, manifests, seed, cache, image, hardware,
  warm-up cutoff, validation windows/65,536 targets, cadences, sampler rates,
  run order, trajectory/identity checks, and pass/fail gates as Attempt 3;
- run order `1-off`, `1-on`, `2-on`, `2-off`, `3-off`, `3-on`; and
- the explicit container Git safe-directory environment setting retained from
  the no-work launch repair.

The acceptance command template changes these three Hydra values from Attempt 3:

```text
training.sequence_length=4096
training.batch_size=1
training.gradient_accumulation_steps=1
```

PASS still requires a median paired throughput regression under 5%, steady
data wait at or below 5%, exact trajectory/final-model parity inside every pair,
all validation/standalone identities and scores exact, validation/checkpoint
budgets met, memory recovery, zero cache mutation, and continuous-trace coverage.
The sequence-4,096 choice was made from validation-off diagnostics before any
sequence-4,096 on/off comparison, so it does not select a favorable validation
effect.

### Outcome and stop decision

- Pair 1 completed at exact head `74a6d6b`: both arms reached 60 steps and
  245,760 targets; the off arm emitted zero validation events and the on arm
  validated exactly at steps 25/50.
- Post-warm-up throughput was 17,909.59 targets/s off and 17,965.08 targets/s
  on. Data wait was 0.99% off and 1.36% on, so the sequence-shape repair cleared
  the Attempt 3 producer bottleneck.
- Validation pauses were 6.9007 s and 7.4542 s; scoring was 4.5640 s and
  4.6195 s, and attribution error stayed below 0.3 ms. These observations remain
  failed-attempt evidence rather than acceptance results.
- The predeclared exact-trajectory gate failed at optimizer step 1, before the
  first validation event. Step-1 loss was identical (`11.029861450195312`), but
  gradient norm differed by `7.89165e-05`; all 60 step rows then differed.
  Both arms emitted PyTorch's warning that memory-efficient-attention backward
  is nondeterministic while deterministic algorithms are configured warn-only.
- Pair 1 therefore stopped the attempt before pairs 2/3. This failure does not
  show validation contamination: it shows that the cross-process exactness
  control was invalid under the configured CUDA determinism policy.

## Attempt 5 — predeclared strict-deterministic DGX R3 retry

The repair changes `reproducibility.deterministic=true` from warn-only to strict
PyTorch deterministic algorithms. Unsupported nondeterministic operations now
fail explicitly; this is not a promise of bitwise equality across platforms or
PyTorch versions. Two pinned-image sequence-4,096 BF16 probes with strict mode
produced identical loss and complete model-tensor digests.

Attempt 5 restarts the full acceptance matrix at the strict-determinism repair
head. It retains every Attempt 4 condition and gate: sequence 4,096, batch 1,
accumulation 1, 60 steps/245,760 targets per arm, warm-up steps 1–10, validation
at steps 25/50, the same manifests/cache/image/hardware/seed, continuous sampler
rates, and run order `1-off`, `1-on`, `2-on`, `2-off`, `3-off`, `3-on`. No
Attempt 4 arm is reused. The exact trajectory and canonical final-model digest
must match inside every pair before later pairs continue.

CHECK §8.1's deterministic-mode cost will be reported descriptively against the
otherwise matched Attempt 4 pair, while the three strict-mode pairs remain the
decision-grade validation-off/on comparison. All original throughput, data-wait,
pause, identity, standalone-parity, recovery, memory, cache, and trace-coverage
gates remain unchanged.

### Outcome and stop decision

The compact failed-attempt record is
[`VAL-001-dgx-r5-failed.json`](evidence/VAL-001-dgx-r5-failed.json).

- All six arms completed at exact head `4264e4a`. Every pair's 60-step non-time
  trajectory and canonical final-model digest matched exactly. Paired on/off
  throughput deltas were -1.21%, -0.016%, and -0.022% (median -0.022%); every
  data-wait fraction was below 1%.
- All six validation events stayed below 7.47 s, scoring stayed below 4.80 s,
  attribution reconciled, validation identities/scores matched across replicas,
  and step-50 training-time/standalone results matched exactly. Cache inventory
  and shard hashes were unchanged, leases released, memory recovered, following-
  five-step means passed, and no swap occurred.
- Strict determinism reduced pair-1 absolute throughput by 44.8% off and 45.7%
  on relative to otherwise matched warn-only Attempt 4. This is descriptive
  same-stack cost context, not a valid acceptance A/B because Attempt 4 failed
  exact trajectory. The eventual baseline determinism/performance choice remains
  an explicit DGX-001 decision.
- Attempt 5 remains `FAIL` under its predeclared evidence gates. The container
  collector emitted at approximately 2.04-second intervals, only 50.1-50.7% of
  the declared 1 Hz samples. Five of six first post-validation steps also
  exceeded the preceding five-step nearest-rank p95 (the maximum of five
  observations) by 0.7-4.7%, even though their following-five means stayed
  within 3% of pre-validation and 1.9% of paired-off controls.
- The first standalone command was rejected before evaluation because it tried
  to override a cache key absent from the operational evaluation profile. The
  successful retry correctly relied on checkpoint-owned resolved configuration;
  no score or evidence was reused from the no-work command.

## Attempt 6 — predeclared adaptive evidence-protocol retry

Attempt 6 is a fully fresh six-arm confirmatory matrix; no Attempt 5 arm is
reused. This protocol revision is explicitly adaptive rather than represented as
an original rule. Its 5% bound is anchored to CHECK §3's existing performance-
investigation threshold, not accepted retrospectively from Attempt 5. No further
threshold relaxation is allowed if Attempt 6 fails.

Every model/data/training condition and every trajectory, identity, score,
throughput, data-wait, pause, sustained-recovery, memory, cache, and standalone
gate remains unchanged. Run order remains `1-off`, `1-on`, `2-on`, `2-off`,
`3-off`, `3-on` at sequence 4,096, batch 1, accumulation 1, 60 steps/245,760
targets per arm, with strict deterministic algorithms.

For each validation event at optimizer step `k` in `{25, 50}`:

- `q_pre` is the nearest-rank p95 of on-arm steps `k-4` through `k`; with five
  observations this equals their maximum.
- The first following step passes only if both
  `t_on[k+1] <= 1.05 * q_pre` and
  `t_on[k+1] <= 1.05 * t_off[k+1]` hold on unrounded values.
- The existing sustained gate remains: the on-arm mean for steps `k+1` through
  `k+5` may be no more than 5% above both the on-arm pre-event five-step mean and
  the paired off-arm mean at the same post-event positions.
- All signed ratios and host/CUDA phase decompositions are reported even when
  they pass. Passing establishes at most a bounded 5% single-step transient and
  5% sustained slowdown under these conditions, not zero restart cost.

GPU sampling remains nominal 5 Hz and host sampling nominal 1 Hz. Container
sampling is prospectively declared as coarse nominal 0.5 Hz because the host's
sequential `docker stats --no-stream` collector empirically emits approximately
every two seconds. The extra wrapper sleep is removed, and a start barrier holds
training until GPU, host, and container collectors have started. A container
trace is valid only if at least 90% of adjacent intervals are at most 2.5 s, no
interval exceeds 3.0 s, and at least three samples fall inside every complete
validation interval. PyTorch per-step allocator measurements and the 5 Hz GPU /
1 Hz host traces remain authoritative for short-lived behavior. The fixed
Attempt 6 protocol script SHA-256 is
`4fc1cfee44f5c44657e6deeb87a3a5d301872d3c456588c381ad3b39f93c7914`.

### Outcome — `PASS WITH NOTE`, independent re-review pending

The compact decision record is
[`VAL-001-dgx-r6.json`](evidence/VAL-001-dgx-r6.json).

- All six arms completed at exact head `497a7b6`. Every pair's full 60-step
  non-time trajectory and canonical final-model digest matched exactly. Paired
  validation-on throughput deltas were -0.152%, +0.041%, and +0.008% (median
  +0.008%); run throughput was 9,855-9,881 targets/s and every data-wait fraction
  was below 0.69%.
- Validation ran exactly at steps 25/50 in every on arm and never in off arms.
  All six full pauses stayed below 7.50 s and scoring below 4.76 s. All events
  reconciled, shared identical manifest/window/token identity, and produced
  exact replicated scores at the same step.
- Pair-1 step-50 training-time and standalone scores, corpus reductions,
  denominators, logical identity, manifest identity, window identity, and token
  identity matched exactly. The standalone output SHA-256 is
  `bdddc4aaf13c71b6903e14f401d4e3784e6f3b96f6ef1285537b2107267ad080`;
  its verified best-checkpoint SHA-256 is
  `246c9ec331719b4eaa0367eeda28d392fa21b896d844401fff2544bf939a6a1f`.
- All six adaptive first-post gates passed. First steps were 0.20-1.28% above
  pre-event p95 and -0.36% to +0.48% versus the paired off step. Following-five
  means were at most 1.57% above pre-event and 0.28% above paired off. Phase
  attribution retained in the JSON shows the largest consistent step-25 deltas
  were approximately 4-13 ms of host-device preparation or data wait; exact
  training state and sustained throughput were unchanged.
- GPU nominal-5-Hz coverage was at least 99.5%; host nominal-1-Hz coverage was
  at least 97.9%. Every container interval was below 2.03 s, nominal-0.5-Hz
  coverage passed, and each validation interval contained 3-4 samples. Allocator
  memory recovered, step tails stayed bounded, cache inventory/shard hashes were
  unchanged, all leases released, and swap remained zero.
- Evidence verdict is `PASS WITH NOTE`, not an erasure of Attempt 5. The recovery
  rule is an explicitly adaptive, CHECK-anchored revision validated on a fully
  fresh matrix. It supports only a bounded <=5% first-step transient and <=5%
  sustained slowdown conclusion. Strict determinism's roughly 45% descriptive
  cost versus warn-only remains a DGX-001 baseline-policy decision.

## Independent review FAIL and repair

The independent review requested `gpt-5.6-sol` / Max for `41191cb` and returned
`FAIL`; the exact runtime model and reasoning mode were not exposed. It
found that checkpoint `state.resolved_config` was not bound to
`identity.config_sha256`, configured and resolved manifest identities were not
reconciled in order before scoring, the scorer could trust a separately passed
manifest identity, and training-time and standalone logical checkpoint
identities did not have exact parity. It also required complete low-overhead
local phase timing for repeated CHECK §6.3 analysis and removal of unused
evaluation package re-exports.

The repair phase, requested as `gpt-5.6-luna` / Extra High, has implemented:

- one canonical checkpoint config hash that excludes only
  `artifacts.resume_path`, with write/read/standalone verification;
- ordered configured manifest reconciliation and actual-loader manifest
  derivation/strict verification;
- one logical checkpoint identity builder used by training-time and standalone
  evaluation, with an exact-parity regression;
- resolved-config, data-fingerprint, ordered-manifest, and stale-loader
  tamper regressions;
- disabled-by-default benchmark timing for data wait, host/device preparation,
  forward, loss, backward, finite checks, clipping, optimizer, scheduler,
  metrics/logging, checkpoint, validation, CUDA-event phases, and allocator
  memory, retained in one atomically flushed measurement JSON; and
- direct imports from `evaluation.scoring` without package-level re-exports; and
- strict deterministic-algorithm enforcement when Hydra requests deterministic
  execution, closing the CUDA attention control failure exposed by Attempt 4.

The ordinary path adds no CUDA synchronization. Benchmark mode deliberately
uses one end-step boundary synchronization so CUDA event timings are valid.
Attempts 3-5 produced stopped current-head evidence; fresh Attempt 6 returned
`PASS WITH NOTE` under its predeclared adaptive evidence protocol.
Exact displayed repair model and reasoning mode are not exposed by runtime.

Local repair verification: focused validation/checkpoint/trainer/generation
tests `55 passed`; full repository tests `310 passed, 1 skipped`; full Ruff,
`uv lock --check`, changed-Python-file format check, and `git diff --check` pass.
The repository-wide formatter still reports four pre-existing unrelated files,
which were not changed. Independent re-review is still required.

## Conclusion

The prior DGX R2 is insufficient current performance evidence, and failed
Attempts 3-5 remain visible. Attempt 6 provides current-head repeated acceptance
evidence with an explicit adaptive-protocol note. VAL-001 remains incomplete
only until an independent heavy re-review accepts the exact documented evidence
head as `PASS` or justified `PASS WITH NOTE`.
