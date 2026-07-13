# WB-001 — Quota-safe W&B preserves local training evidence

- Roadmap ticket: `WB-001`
- Branch: `codex/wb-001-evidence-safe-wandb`
- Draft PR: unavailable during implementation because the delegated runtime has
  no GitHub publication command; complete body prepared at
  `/tmp/WB-001-pr-body.md`
- Experiment owner: implementation agent
- Status: R1 repair validation passed; DGX Attempts 2 and 5 failed, Attempts 3,
  4, and 6 were aborted, and depth-based compute-bound Attempt 7 is predeclared
- Started (UTC): 2026-07-13
- Last updated (UTC): 2026-07-13
- Model-run provenance: `docs/model-runs/WB-001-evidence-complete-wandb.md`

## Predeclared question and decision rule

- Hypothesis: one failure-isolated W&B boundary can expose compact scalar and
  lineage evidence at the existing trainer cadence while disabled/offline mode,
  missing auth/quota, and upload failure preserve all local work.
- Expected result: strict config, policy/auth/usage/size test matrices, and a
  socket-blocked offline smoke pass; scalar calls equal the declared trainer
  log boundaries; no smoke/CI/memorization artifact can upload.
- Success condition: all WB-001 acceptance tests pass; no raw corpus or
  recovery checkpoint enters W&B; an artifact uploads only after matching
  policy/reason, verified viewer/entity, fresh visible usage, projected
  headroom, and SHA deduplication; local evidence survives every external
  failure.
- Failure condition / stop condition: implicit online-to-offline fallback,
  raw-corpus upload, hard-coded service quota, missing/stale usage permitting an
  upload, W&B failure aborting training, or a second scalar cadence.
- Relevant baseline commit and run: stacked VAL-001 documentation head
  `74d9e24c251b62b23892b11ba0c1c9c723cd8a12`; no W&B performance run.
- Baseline metrics and evidence link: AS-IS unconditional watch and upload path
  in `ROADMAP.md`; no trustworthy disabled/offline/watch comparison existed.
- Smallest run capable of answering the question: CPU policy/config/trainer
  tests plus the canonical socket-blocked disabled/offline smoke. Performance
  conclusions require the separately predeclared R2 below.

## Planned budget

| Resource | Limit | Derivation / measurement source |
| --- | --- | --- |
| CPU correctness | Focused tests plus one disabled and one offline canonical smoke | Ticket validation asks for test doubles, matrix, missing login, schema, offline smoke |
| Elapsed time on target hardware | Retained Attempt 2: 3×3×100; retained Attempt 5: 3×3×300; predeclared Attempt 7: 3×3×260 steps | CHECK §§6.3 and 9.2 repeated disabled/offline/watch comparison; 260 steps execute 1,040 backward batches and one default-frequency watch event |
| Training tokens | Attempt 7: 133,120 targets/arm from a 133,248-token streaming cap; identical in every arm | 260 steps × batch 2 × sequence 64 × accumulation 4; the extra 128 stream tokens provide full windows |
| Optimizer steps | 2 CPU smoke invocations; retained Attempt 2: 900; retained Attempt 5: 2,700; Attempt 7: 2,340 | Attempt 7 uses 260 steps × 3 arms × 3 repetitions |
| Evaluation work and cadence | Identical across all R2 arms; no extra W&B cadence | Isolates W&B/watch overhead |
| Checkpoint count and bytes | Existing local checkpoint policy per arm, inventoried with exact bytes; no W&B artifact in required R2 | Artifact behavior is proven with test doubles, not service quota |
| Local / external / W&B storage | Temp directories for CPU smoke; future R2 records W&B directory bytes; zero cloud bytes required | W&B is not backup or bulk storage |

## Attempt 1 — network-free correctness and offline smoke

### Launch identity

- Started / ended (UTC): 2026-07-13 / 2026-07-13
- Outcome: succeeded
- Exact commands:
  ```text
  uv run pytest -q tests/test_wandb_tracking.py tests/test_config_profiles.py tests/test_trainer.py tests/test_reproducibility.py tests/test_evaluation.py tests/test_ci_quality_gate.py
  uv run --no-sync python scripts/offline_smoke.py
  ```
- Fully resolved Hydra configuration: base `config/train.yaml` composed with
  `profile=smoke_overfit`, tiny CPU model overrides in
  `scripts/offline_smoke.py`, first `wandb.mode=disabled`, then
  `wandb.mode=offline`. Each temporary run retained its own resolved config and
  manifest until the temporary directory closed.
- Git commit SHA: baseline `74d9e24`; implementation working tree, not yet
  committed by the delegated implementation agent.
- Worktree state: dirty with the WB-001 implementation diff; no unrelated
  pre-existing dirty files were present at start.
- Dependency lock identity: `uv.lock` unchanged; lock check pending final QA.
- Container/image identity: `N/A` — CPU R1 only; no DGX claim.

### Scientific identity

- Model architecture/config identity and parameter count: canonical tiny CPU
  smoke override, 1,672,090 random-initialized parameters.
- Initialization / pretrained-weight check: repository random initialization;
  no external weights or targets.
- Tokenizer: canonical pinned tokenizer fingerprint
  `12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b`.
- Training/validation manifest: explicit memorization smoke fingerprint
  `00c3797a7d0eda13950fd699a60c45fcd388829f016479caaeb369438767bd31`;
  this is deliberately not held-out validation.
- Random seeds: canonical Hydra reproducibility seed 42.
- Hardware/software: local CPU, Python/Torch identities captured by each
  temporary run manifest; W&B SDK 0.25.1 from the unchanged lock.
- Precision and numerical controls: FP32 CPU, deterministic mode.

### Counters, evidence, and integrity

- Automated result after integration repairs: focused integration `115 passed
  in 11.26s` plus `10 passed` for the R2 verifier; full repository suite `361
  passed, 1 skipped in 68.20s`.
- Offline smoke: both disabled and offline invocations completed while
  credentials were removed; name resolution, non-Unix `connect`, `connect_ex`,
  and `sendto` operations were blocked. Offline W&B 0.25.1 initialized, logged
  only at trainer boundaries, removed watch state, and finished without
  network.
- W&B run/project/artifact IDs: disabled arm `N/A`; offline arm local-only and
  temporary; artifact policy `none`, so no artifact candidate or cloud upload.
- Checkpoints: local verified final checkpoints completed in both arms and were
  removed with the temporary smoke directory.
- Integrity: no raw corpus content enters the sanitized W&B config; source
  manifest/reference fields remain. Smoke/CI/memorization purposes are denied
  from artifacts in code. Operational W&B changes do not alter checkpoint
  compatibility identity.
- Final static QA: repository Ruff, changed-file format check, `uv lock
  --check`, and `git diff --check` all passed.

### Attempt interpretation

- Result against success/failure conditions: R1 supported the functional and
  failure-isolation hypothesis; no stop condition occurred.
- Failure or anomaly: the first offline run wrote its W&B directory beneath
  the repository because `WANDB_DIR` was not explicit. The generated directory
  was removed, and the smoke now directs W&B into the temporary guard root.
- Most likely cause and supporting evidence: W&B defaults its offline run
  directory under the process working directory.
- Alternatives ruled out: no online fallback occurred; the socket guard stayed
  active and the run explicitly reported offline mode.
- What remains uncertain: hot-path cost and watch overhead on DGX Spark require
  the R2 protocol below.

## Integration QA audit and repair

A read-only precommit audit returned `FAIL` before the mandatory heavy review.
It reproduced a missing artifact-summary helper that contradicted successful
upload evidence, the W&B 0.25.1 string-team authentication shape, unverified
checkpoint reason/identity/step, missing run/artifact/runtime and non-finite
evidence, and incomplete socket interception. The repair now:

- binds every candidate to a verified full-state checkpoint kind, run identity,
  and exact optimizer step before quota or upload work;
- requires a callable artifact completion wait and exact `COMMITTED` state;
- records W&B run URL/ID and committed artifact ID/name/version/digest locally
  and in compact summaries;
- accepts the public SDK's string-valued team list;
- bounds training and standalone-evaluation finish without making W&B local
  completion authority;
- emits validation results at their exact trainer boundary and compact
  non-finite stability events; and
- expands the offline guard and narrows the research claim to the operations
  actually intercepted.

The repaired focused suite, full suite, Ruff, and offline smoke pass. This QA
audit is not substituted for the required independent heavy review.

## Attempt 2 protocol — DGX disabled/offline/watch comparison (completed FAIL)

Use one exact clean commit and the same pinned CUDA container, GPU, cache,
resolved `profile=stability_smoke` config, seed, data order, BF16 recipe,
validation/checkpoint settings, and 100-step horizon. Exclude steps 1–10 as
warm-up. Run three alternating repetitions of disabled, offline/watch-off, and
offline/watch-on; rotate the order across repetitions. W&B artifact policy is
`none` in every arm.

Retain per-step phase timing, target tokens/s, median/p95/max step time, data
wait, RSS/peak RSS, PyTorch peak allocated/reserved memory, loss, LR, gradient
norm/clipping/non-finites, W&B local bytes, full resolved configs, manifests,
and environment identity. Require exact target counters and training trajectory.
Report paired medians and spread. Investigate >=5% median tokens/s regression;
>=10% normally fails. Any objective, counter, data-order, or finite-value
divergence fails. Watch remains off by default regardless of result unless a
later experiment demonstrates a concrete need.

An optional online scalar run and one selected artifact are allowed only after
verified credentials and a fresh operator Billing UI/CSV usage snapshot show
headroom. They are not required to complete the network-free implementation
proof and must not precede human/operator credential and quota evidence.

The committed runner primes the shared cache outside the measured matrix, then
enforces the fixed Latin-square order, clean exact head and image, `network=none`,
read-only source, run-local evidence/W&B directories, idle GPU conditions, and
host/GPU/container sampling:

```bash
HEAD=$(git rev-parse HEAD)
IMAGE_ID=$(docker image inspect llm-scratch:env-001 --format '{{.Id}}')
docs/experiments/evidence/run_wb001_dgx.sh \
  "$HEAD" "$IMAGE_ID" "/tmp/wb001-r2-$HEAD" "/tmp/wb001-cache-$HEAD"
uv run python docs/experiments/evidence/verify_wb001_dgx.py \
  "/tmp/wb001-r2-$HEAD" --expected-commit "$HEAD" \
  --expected-image-id "$IMAGE_ID"
```

### Attempt 2 result — FAIL, evidence retained

- Measured commit: `049acf7fdfc37272a9c55c41d79aef5086e36e03`.
- Image: `llm-scratch:env-001`, exact ID
  `sha256:23a1bee69fe189e77105cdddeee9aeff6ef0763d58a691625fbfcab64efd1887`.
- Raw evidence root:
  `/tmp/wb001-r2-049acf7fdfc37272a9c55c41d79aef5086e36e03`;
  durable verifier projection:
  `docs/experiments/evidence/WB-001-dgx-r2-failed.json`.
- All nine containers exited 0. Every arm completed exactly 100 steps and 6,400
  target tokens; normalized configs, experiment/data/tokenizer IDs, scalar
  trajectories, model tensors, resume state, and stream cursors were exact.
  W&B lifecycle/storage gates, GPU/host coverage, finite values, checkpoint
  survival, swap, and allocator-stability gates passed.
- The run still failed honestly. Six arms exceeded the declared 10% data-wait
  ceiling (observed 8.16–12.68%), so the sequence-8 workload was too small for a
  trustworthy hot-path comparison. Docker's `stats --no-stream` produced only
  about 0.5 samples/s, below the declared 1 Hz/90% coverage. The hardware
  equality projection also included Docker's random container hostname rather
  than stable hardware identity.
- Offline/watch-off had a noisy median throughput change of -5.41% versus
  disabled (individual pairs -31.26%, -3.91%, and -5.41%). Watch-on incurred a
  consistent 26.26% median regression versus offline/watch-off and wrote about
  1.00 MB/run instead of about 37 KB/run because the 100-batch interval emitted
  repeated histograms. This is treated as a real unsafe default for opt-in
  watch, not discarded as mere noise.
- Each run retained three verified checkpoints totaling about 1.784 GB, W&B
  uploaded zero cloud bytes, allocator peaks stabilized near 1.07 GB allocated
  and 1.095 GB reserved, and no sustained swap I/O occurred.
- One initial command used an unobserved full-SHA expansion. The runner rejected
  it before container launch; the corrected command used `git rev-parse HEAD`.

### Attempt 3 — aborted before measured arms

Attempt 3 started at exact commit `ee4d41d0afe403f50031ffa53d907ff13cf5ba91`
but was interrupted after cache prime, before any comparison arm, when an
independent protocol audit returned `FAIL`. Its partial root is retained at
`/tmp/wb001-r3-ee4d41d0afe403f50031ffa53d907ff13cf5ba91` and is not treated as
measurement evidence. The audit found that 100 optimizer steps at accumulation
4 would execute only 400 backward batches, so a 1,000-batch watch interval
would register hooks but never emit a histogram. It also found that raw
streaming Docker stats lacked timestamps and that the predeclaration
incorrectly described sequence length 64 as the same model/work as Attempt 2.

### Predeclared Attempt 4 — audited adaptive retry

Attempt 4 keeps one identical configuration across all arms: exact image/GPU,
seed, sequence-64 model, data order, three-arm Latin square, artifact policy,
and decision gates. Relative to Attempt 2, sequence length 64 changes the input
shape, positional-encoding buffer, and total work. No cross-attempt performance
comparison will be made; only matched arms within Attempt 4 support its result.

The corrected protocol sets `data.streaming.train.max_tokens=153728`, which the
real loader resolves to 1,200 complete microbatches and 153,600 targets. It runs
300 optimizer steps with steps 1–30 as warm-up. At accumulation 4 this executes
1,200 backward batches, so the official/default
`wandb.watch.log_freq=1000` must emit at least one gradient-histogram history
record. A network-isolated inspector decodes a temporary copy of the local
`.wandb` record, hashes the retained original, and requires histogram content
only in watch-on arms. GPU, host, and Docker samples are filtered to the exact
training start/end window and require at least 90% count coverage. GPU remains
5 Hz with start/end/inter-sample gaps at most 0.75 s; vmstat remains 1 Hz with
2 s endpoint and 2.5 s inter-sample limits. Docker's measured `--no-stream`
poll is approximately 0.5 Hz, with 3.5 s endpoint and 4.5 s inter-sample
limits. A redirected streaming Docker smoke was rejected before launch because
its ANSI screen-refresh rows and burst duplicates were not independent samples.
Stable hardware equality excludes only the randomized container hostname;
exact image/mount/network identity remains separately checked.

Attempt 4 remains failed if data wait exceeds 10%, any sampler coverage gate
fails, local W&B record content is wrong, exact arm-to-arm trajectory/checkpoint
identity diverges, or any paired median regression is at least 10%. The 5%
investigation threshold and all memory/swap/storage/lifecycle gates are
unchanged.

### Attempt 4 result — stopped on thermal precondition

Attempt 4 ran at exact commit `e9dd9e37319a8b6ba631e038c189d82d6483e187`
and the same pinned image. Disabled arm `r1-p1` completed 300 steps, 153,600
targets, checkpoint/W&B evidence, and clean GPU/host/container samples in about
43 s. Before `r1-p2` launched, the fixed 30-second idle check observed a 3 °C
temperature spread against its 2 °C stop limit. The runner stopped with no
second measured arm. No throughput comparison is made. Raw evidence is retained
at `/tmp/wb001-r4-e9dd9e37319a8b6ba631e038c189d82d6483e187`; the structured
failure projection is
`docs/experiments/evidence/WB-001-dgx-r4-aborted.json`.

### Predeclared Attempt 5 — bounded adaptive cooldown

Attempt 5 keeps every Attempt 4 model, data, work, W&B, order, exactness,
performance, storage, and resource decision gate. Only the failed precondition
changes: before every arm, sample idle GPU state for at least 30 and at most 90
seconds using elapsed nanoseconds, not sample count; pass only when the trailing
30-second window contains at least 27 valid numeric temperature readings and
its spread is at most 2 °C, otherwise stop. Invalid/`N/A` temperatures cannot
coerce to a passing zero spread. This preserves the original 2 °C stability
threshold while allowing post-arm cooling samples to age out of the decision
window. The matrix restarts from repetition 1 in a fresh root and cache on a
new exact clean commit; Attempt 4's single arm is not reused.

### Attempt 5 result — FAIL, full evidence retained

Attempt 5 ran all nine arms at exact commit
`0dd2fd56cd6695be75ac37e3f563c4ca4916264d` and the pinned image. Raw evidence
is retained at `/tmp/wb001-r5-0dd2fd56cd6695be75ac37e3f563c4ca4916264d`;
the durable verifier projection is
`docs/experiments/evidence/WB-001-dgx-r5-failed.json`.

Every process exited 0. All arms completed exactly 300 updates and 153,600
targets. Normalized configs, experiment/data/tokenizer/hardware identity,
scalar trajectories, model tensors, resume state, and stream cursors were
exact. W&B lifecycle/storage/record-content gates passed: watch-on arms each
contained one structurally valid gradient-histogram history record, watch-off
and disabled arms contained none, and no cloud bytes or artifacts were used.
GPU/host/container temporal coverage, finite values, checkpoint survival,
memory stability, and swap gates passed.

The result remains `FAIL`. Every arm exceeded the 10% data-wait limit (11.89%
to 18.02%), so sequence 64 was still not compute-bound enough. Paired
offline/watch-off versus disabled changes were 10.99%, -6.93%, and 19.86%, for
a failing 10.99% median. Offline/watch-on versus disabled had a 0.73% median;
watch-on versus offline/watch-off had a -11.53% median, but neither is used to
override the data-wait failure or claim performance safety.

### Predeclared Attempt 6 — sequence-256 compute-bound retry

Attempt 6 changes only the within-matrix work horizon needed to resolve the
failed measurement: sequence length 256, 260 optimizer steps, steps 1–26 as
warm-up, and a 532,992-token stream cap. The real loader must yield 1,040 full
microbatches, 260 accumulation groups, and 532,480 targets. This still crosses
the default 1,000-batch watch interval and leaves 40 backward batches after the
required histogram event. Sequence 256 changes the model's positional buffer,
input shape, and work relative to Attempt 5; no cross-attempt throughput
comparison is allowed.

All arms within Attempt 6 retain the same seed, model, data order, Latin-square
order, W&B modes, artifact policy, cooldown, samplers, and decision gates.
Data wait above 10%, any exactness/lifecycle/storage/resource failure, or any
paired median regression at least 10% remains a failure; at least 5% remains an
investigation note. The matrix uses a fresh cache/root and exact clean commit;
no Attempt 5 arm is reused.

### Attempt 6 result — aborted during cache prime

Attempt 6 started at exact commit `54fb32b68787330b6e893b99d30e18842b821196`
and the pinned image, but sequence 256 left the deliberately tiny validation
fixture with zero complete target windows. Cache prime stopped at epoch-end
validation before any measured arm. Raw evidence is retained at
`/tmp/wb001-r6-54fb32b68787330b6e893b99d30e18842b821196`; the structured
failure projection is
`docs/experiments/evidence/WB-001-dgx-r6-aborted.json`. Validation is not
disabled or weakened to accommodate the performance protocol.

### Predeclared Attempt 7 — 18-layer compute-bound retry

Attempt 7 returns to sequence 64 so the actual validation fixture remains
non-empty, and raises only `model.num_layers` from 6 to 18. It keeps 260
optimizer steps, steps 1–26 as warm-up, and uses a 133,248-token stream cap for
exactly 1,040 microbatches, 260 full accumulation groups, and 133,120 targets.
The default 1,000-batch watch event is still required. The 18-layer model has
70,828,682 parameters on the pinned runtime and passed a network-isolated CUDA
one-step train/validation/checkpoint smoke before commit.

Depth changes the model and compute work relative to prior attempts, so no
cross-attempt throughput comparison is allowed. Within Attempt 7 every arm has
the same seed/model/data/order/cadence/cooldown/samplers and artifact policy.
All exactness, data-wait, 5% investigation, 10% failure, storage, memory, swap,
and temporal gates remain unchanged. The full matrix uses a fresh cache/root
and clean exact commit; no prior arm is reused.

## Conclusion

- Hypothesis result: supported at R1; DGX Attempt 2 failed its measurement and
  watch-cost gates, Attempts 3 and 4 made no comparison claim, and Attempt 5
  failed data-wait/performance gates. Attempt 6 stopped before measurement, so
  the overhead conclusion remains pending Attempt 7.
- Evidence-backed conclusion: the implementation can preserve local metrics and
  checkpoints across disabled/offline W&B and tested external failure paths,
  while fail-closing every artifact safety gate.
- Uncertainty and limitations: no online service call, real quota consumption,
  or artifact upload was performed; the failed DGX evidence is retained rather
  than used for a positive performance claim.
- Exactly one next step: run and verify the predeclared adaptive Attempt 7 on a
  clean repair commit before the mandatory independent review.
