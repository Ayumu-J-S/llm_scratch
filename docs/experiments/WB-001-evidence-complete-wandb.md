# WB-001 — Quota-safe W&B preserves local training evidence

- Roadmap ticket: `WB-001`
- Branch: `codex/wb-001-main-integration`
- Draft PR: [#46](https://github.com/Ayumu-J-S/llm_scratch/pull/46)
- Status: current-main integration repair is in progress after exact-head review
  cycle 20 returned `FAIL`; the complete cycle trail is in live PR #46. R1
  functional evidence remains supported. Attempt 9's historical
  `PASS WITH NOTE` and 168-gate result are retained, but its paired-overhead
  figures are withdrawn from current acceptance because scheduled scalar-log
  pauses were absent from the throughput denominator.
- Started (UTC): 2026-07-13
- Last updated (UTC): 2026-07-19

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
| Elapsed time on target hardware | Retained Attempt 2: 3×3×100; retained Attempt 5: 3×3×300; retained Attempts 7, 8, and 9: 3×3×260 steps each | CHECK §§6.3 and 9.2 repeated disabled/offline/watch comparison; 260 steps execute 1,040 backward batches and one default-frequency watch event |
| Training tokens | Attempts 7, 8, and 9: 133,120 targets/arm from a 133,248-token streaming cap; identical in every arm within an attempt | 260 steps × batch 2 × sequence 64 × accumulation 4; the extra 128 stream tokens provide full windows |
| Optimizer steps | 2 CPU smoke invocations; retained Attempt 2: 900; retained Attempt 5: 2,700; retained Attempts 7, 8, and 9: 2,340 each | Each current matrix uses 260 steps × 3 arms × 3 repetitions |
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
- Git commit SHA: baseline `74d9e24`; the first implementation attempt was not
  yet committed.
- Worktree state: dirty with the WB-001 implementation diff; no unrelated
  pre-existing dirty files were present at start.
- Dependency lock identity: `uv.lock` unchanged; lock check passed in final
  current-main validation.
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
  in 11.26s` plus `16 passed` for the R2 verifier and `2 passed` for the offline
  inspector; current full repository suite `379 passed, 1 skipped in 69.10s`.
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
  --check`, and `git diff --check` all passed. A whole-tree format check reports
  four unchanged baseline files; none is in the WB-001 diff.

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

## Online authentication and visibility smoke — PASS

On 2026-07-19, the integrated WB-001 implementation completed a deliberately
small three-step CPU run against the live W&B service:

- run URL: <https://wandb.ai/sunday-research/llm-scratch/runs/fcblar36>;
- run ID: `fcblar36`;
- run name: `wb001-online-smoke-20260719`;
- final state: `finished`;
- recorded history rows: 3;
- model artifacts: 0;
- model watch: disabled; and
- artifact policy: `none`.

The run made compact training/memorization metrics and lineage visible in the
configured `sunday-research/llm-scratch` project, then finished cleanly. This is
evidence that current authentication, online initialization, scalar logging,
visibility, and no-artifact completion work together. It is only a WB-001
authentication/visibility smoke on a tiny CPU workload. It is not the
`RUN-001` bounded real baseline, does not establish model quality, and does not
exercise artifact quota, retention, or upload behavior.

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

### Attempt 7 result — failed only the per-arm data-wait gate

Attempt 7 ran all nine arms at exact commit
`a4117cec2e86e3b0079f3a867ee76059a93dc988` in the pinned image. Raw evidence
is retained at
`/tmp/wb001-r7-a4117cec2e86e3b0079f3a867ee76059a93dc988`; the byte-identical
structured summary is
`docs/experiments/evidence/WB-001-dgx-r7-failed.json`.

The verifier passed 163 of 166 gates. Every arm completed 260 steps and
133,120 targets; exact trajectories/checkpoints, validation, W&B lifecycle and
decoded watch histograms, storage, temporal sampler coverage, memory, and swap
passed. Paired median changes also passed: offline/watch-off versus disabled
was 2.02%, offline/watch-on versus disabled was 3.37%, and watch-on versus
offline/watch-off was 2.04%.

The result remains `FAIL` because data wait was 10.65% in
`r1-p2-offline-off`, 10.49% in `r3-p1-offline-off`, and 11.90% in
`r3-p3-disabled`, above the unchanged 10% per-arm limit. The failures span
offline-off and disabled modes and therefore do not indicate a W&B-specific
regression. No Attempt 7 arm is eligible for reuse.

### Predeclared Attempt 8 — depth-26 compute-bound retry

Attempt 8 changes only `model.num_layers` from 18 to 26. It retains sequence
64, 260 steps, 26 warm-up steps, batch 2, accumulation 4, the 133,248-token
stream cap, 1,040 backward batches, nonempty epoch-end validation, and the
default 1,000-batch watch event. Every decision threshold, Latin-square order,
cooldown, sampler, fresh-root/cache requirement, and fail-closed verifier gate
is unchanged.

The worst Attempt 7 arm averaged 19.62 ms data wait and 145.31 ms non-wait
work per measured step. Passing requires more than 176.6 ms non-wait work.
Retained sequence-64 depth 6→18 evidence estimates 4.79–5.28 ms additional
non-wait work per layer: depth 24 projects 9.97–10.15% and is too marginal,
while depth 26 projects approximately 9.47–9.65%. This cross-attempt evidence
is used only to size the repair, never as an Attempt 8 performance result.

A network-isolated CUDA one-step train/validation/checkpoint feasibility run
passed with 85,024,394 parameters and three approximately 1.02 GB checkpoints.
Measured Attempt 7 headroom was ample: 1.61 GB maximum allocator reservation,
at least 121.48 GB host free/buffer/cache, zero swap I/O, and more than 431 GB
disk free. Attempt 8 must run as a wholly fresh matrix from its own clean exact
commit before any conclusion.

### Attempt 8 result — PASS WITH NOTE

Attempt 8 ran all nine fresh arms at exact commit
`b59f84483d1f85a5cd42005d48e8b99d60ab2695` in the pinned image. Raw evidence
is retained at
`/tmp/wb001-r8-b59f84483d1f85a5cd42005d48e8b99d60ab2695`; the byte-identical
structured summary is
`docs/experiments/evidence/WB-001-dgx-r8-pass-with-note.json`.

All 170 verifier gates passed. Every arm completed 260 optimizer steps and
133,120 targets with identical model, normalized resume, cursor, and trajectory
digests. Physical checkpoint file SHA-256 values may differ because each arm
retains arm-local operational metadata. Data wait ranged from 6.53% to 8.54%,
below the 10% failure limit but above the 5% investigation threshold in every
arm. Offline/watch-off versus disabled had a 1.75% median change.
Offline/watch-on versus disabled had a 6.56% median, and watch-on versus
offline/watch-off had a 5.46% median; both are investigation notes below the
10% failure threshold.

Each watch-on arm contained exactly one structurally valid decoded histogram
record with 315 gradient series; watch-off and disabled arms contained none.
The largest watch-on W&B directory was 1,079,003 bytes, watch-off stayed below
65,143 bytes, and disabled produced zero W&B storage. Validation, local
checkpoints, sampler coverage, memory stability, and zero swap-I/O gates all
passed. No online auth, quota, retention, upload, or cloud behavior is inferred
from this network-isolated artifact-policy-none matrix.

## Mandatory review and cycle-13 repair

The mandatory independent review of exact clean head
`23b6d2120f1a3738d4a3baf92e50c9b8f3c227f9` returned `FAIL` even though it
independently reproduced Attempt 8 byte-for-byte and confirmed all 170 gates.
It found that multiple milestone candidates could reuse the same visible quota
headroom, synchronous `run.log()` could stall training, and a partial
`run.watch()` failure could leave hooks installed. It also required the
protocol, checkpoint-identity wording, and PR provenance corrections retained
in this record and the prepared handoff.

Cycle 13 implements the three runtime repairs without changing the model,
objective, data, trainer cadence, Attempt 8 evidence, or default artifact/watch
policy:

- one persistent tracker-owned scalar worker bounds each SDK log call by Hydra
  `wandb.log_timeout_seconds=5`; timeout, exception, or queue saturation opens
  a circuit breaker and preserves local training evidence;
- a tracker-lifetime quota ledger reserves each accepted candidate before cloud
  submission, includes all earlier reservations, and retains the maximum
  observed usage plus minimum observed limit across refreshed snapshots; and
- watch state is registered before SDK installation so partial failure triggers
  immediate global cleanup, while normal teardown falls back to the same
  all-hook cleanup.

The final focused W&B/config suite passes `55 passed in 2.70s`, the relevant
WB/config/trainer/reproducibility/verifier selection passes `102 passed in
3.68s`, and lead validation passes the full repository suite at `379 passed, 1
skipped in 69.10s`. Repository Ruff, four-file changed-Python format, lock,
runner shell syntax, JSON parsing, and diff checks pass. A separate focused
repair audit returned `PASS` with `55 passed in 2.71s` plus clean static checks;
it is not the mandatory heavy exact-head re-review. The subsequent mandatory
re-review of exact clean implementation/evidence head
`5a0a7437e9f94fe56f0ed2dd4cad622cd9d9e25e` returned `PASS WITH NOTE` and
closed all six prior findings without adding an online/cloud claim.

## Attempt 9 — historical repaired-head DGX result, PASS WITH NOTE at the time

An independent prelaunch audit returned `PASS` at exact clean repair commit
`e507a3447ab0895960530cdb207ca0702ec41f85`. It confirmed that the runner,
verifier, inspector, trainer, and `stability_smoke` profile retained the
Attempt 8 protocol byte-for-byte: the 3×3 Latin square, depth 26, sequence 64,
260 optimizer steps with 26 warm-up steps, 1,040 backward batches, 133,120
targets, one epoch-end validation at step 260, 1,000-batch watch frequency,
and 10-step scalar cadence. The repaired arms add only the declared 5-second
scalar SDK timeout. The audit passed 73 focused tests and an actual local W&B
worker/watch smoke with decoded histogram evidence and clean hook teardown.

Attempt 9 then ran all nine fresh arms at that exact commit in image
`sha256:23a1bee69fe189e77105cdddeee9aeff6ef0763d58a691625fbfcab64efd1887`.
Raw evidence is retained at
`/tmp/wb001-r9-e507a3447ab0895960530cdb207ca0702ec41f85`; the durable summary is
`docs/experiments/evidence/WB-001-dgx-r9-pass-with-note.json`, SHA-256
`d8a5b4683b192df2b5f5876819dcdb628ed88de5c118fae3f306293660d6a598`.
Independent regeneration at `/tmp/wb001-r9-independent-summary.json` is
byte-identical to that summary. A second independent policy/quality validator
also returned `PASS WITH NOTE` with no actionable finding, classifying the
result as 159 required gates plus nine passing per-arm data-wait note gates.

The historical verifier reported all 168 applicable dynamically emitted gates
passing with zero failures. The
nine warnings are exactly one predeclared `data_wait_investigate` note per arm:
data wait ranged from 6.3273% to 8.1949%, below the unchanged 10% failure
threshold. Paired median throughput changes were 1.913997% for
offline/watch-off versus disabled, 2.224572% for offline/watch-on versus
disabled, and 2.632986% for watch-on versus offline/watch-off. Every paired
value was below 10%, and all three medians were below 5%. Attempt 9 therefore
has 168 gates rather than Attempt 8's 170: the verifier emits the two aggregate
investigation gates only when a relevant paired median is at least 5%; neither
conditional gate applied at the time. Cycle-20 review later found that the
throughput denominator ended before scheduled W&B scalar logging. These exact
figures and the original verdict remain immutable history, but they are
withdrawn as current overhead-acceptance evidence rather than silently
reinterpreted with a changed verifier.

Every arm completed 260 total, 26 warm-up, and 234 measured optimizer steps;
the corresponding target counts were 133,120 total and 119,808 measured.
Validation occurred once at step 260. Each watch-on arm contained exactly one
decoded histogram record with 315 series; watch-off and disabled arms contained
none. No scalar failure occurred. Across the matrix, minimum host
free/buffer/cache was 121,024,811,008 bytes; maxima were 3,263,101,403 bytes
for container memory, 1,798,200,320 bytes CUDA allocated, 2,000,683,008 bytes
CUDA reserved, and 1,078,947 local W&B bytes. The minimum post-run disk free
space was 367,225,155,584 bytes, and the maximum sustained swap-I/O run was
zero.

Attempt 9 previously superseded Attempt 8 as performance evidence; neither is
now cited for current overhead acceptance. Both remain retained for audit
history. This network-isolated,
artifact-policy-none matrix does not exercise or support claims about online
authentication, authoritative quota, retention, upload, or cloud behavior.
DGX Spark unified-memory headroom is supported by host, container, and allocator
evidence, and decoded W&B binary records prove only local history/watch content.
The historical mandatory heavy re-review accepted this evidence and the then
repaired implementation with the nonblocking limitations below; cycle 20
supersedes that acceptance for the current integration head.

## Independent implementation re-review — PASS WITH NOTE

An independent review covered exact clean implementation/evidence head
`5a0a7437e9f94fe56f0ed2dd4cad622cd9d9e25e` against baseline
`74d9e24c251b62b23892b11ba0c1c9c723cd8a12`, repaired implementation
`e507a3447ab0895960530cdb207ca0702ec41f85`, and the retained mandatory `FAIL`
at `23b6d2120f1a3738d4a3baf92e50c9b8f3c227f9`.

The verdict is `PASS WITH NOTE` with no actionable blocker. All six prior
findings are closed. The reviewer regenerated Attempt 9 byte-identically at
SHA-256
`d8a5b4683b192df2b5f5876819dcdb628ed88de5c118fae3f306293660d6a598`
and confirmed 168/168 applicable gates: 159 required gates plus nine declared
data-wait-note gates, with zero failures. The review output is
`/tmp/WB-001-heavy-rereview.txt`, SHA-256
`586bfe867db9373c82b693c1083339d8a8b5b6e1f1e887928daf8d3f2d879605`.

The notes are bounded: per-arm data wait remains 6.3273–8.1949%; network
isolation plus artifact policy `none` support no online/cloud claim; quota
reservation is conservative within one tracker lifetime rather than an
account-global multi-process reservation; and a permanently stuck daemon SDK
worker may remain until process exit while local training and tracker shutdown
stay bounded. Draft PR
[#46](https://github.com/Ayumu-J-S/llm_scratch/pull/46) is the current handoff;
guarded self-merge is authorized for the bounded roadmap goal series only after
the exact-head review and every merge gate pass.

## Current-main exact-head cycle 20 — FAIL; cycle 21 repair

Independent review of exact clean head
`31947afb540ad6044cd0b86074799a22d98e9555` returned `FAIL` after reproducing
the full CPU suite at 423 passed and one skipped. It found four actionable
defects: direct summary updates could hang optional tracking; epoch-only W&B
history used the final update rather than the token-weighted epoch aggregate;
cross-directory resume could silently lose the retained best artifact; and the
DGX verifier omitted scheduled scalar-log pauses from throughput.

The subsequent WB-only call-site audit also found that complete training and
standalone-evaluation initialization lacked caller-owned wall-clock bounds,
watch/unwatch and local artifact staging were direct SDK calls, and the DGX
verifier still read the pre-VAL top-level measurement layout rather than
schema-v3 segments. Cycle 21 therefore:

- bounds login, complete initialization, summaries, watch/unwatch, artifact
  staging/submission, entity verification, scalar logs, and finish; late
  initialization/watch results are cleaned up without blocking local work;
- reports token-weighted epoch NLL/perplexity at epoch-only history boundaries;
- verifies a prior `best.pt` across resume directories and emits an explicit
  blocked artifact decision when the expected retained best is unavailable;
- consumes one complete fresh schema-v3 measurement segment; and
- includes post-warm-up scheduled scalar-log pauses in the primary throughput
  denominator while keeping the data-wait fraction compute-only.

No exact repaired-head DGX rerun is available in this cycle. The Attempt 9
overhead numbers are therefore withdrawn/qualified, not reused as acceptance.
The functional roadmap acceptance matrix remains covered by test doubles,
offline smoke, policy/size/auth/quota cases, metric schema, and the online
no-artifact visibility smoke. A fresh independent exact-head review must return
at least `PASS WITH NOTE` before handoff. Opt-in watch histogram hooks still
perform SDK-owned local-backend publication in the model hot path outside the
tracker scalar worker; watch remains default-off and no universal timeout claim
is made for hook-generated records.

## Conclusion

- Hypothesis result: functional/failure-isolation behavior is supported at R1.
  Attempt 9's historical R2 exactness, resource, watch-content, and data-wait
  observations remain retained, but its paired-overhead result is withdrawn
  because the old verifier excluded scheduled scalar-log pauses.
- Evidence-backed conclusion: the implementation can preserve local metrics and
  checkpoints across disabled/offline W&B and tested external failure paths,
  while cycle-21 local validation is pending final exact-head review. The live
  three-step smoke also confirms authenticated online scalar visibility and
  clean no-artifact completion. Cycle-13
  quota, scalar-boundary, and watch-cleanup history remains useful, while the
  current repair no longer cites the flawed overhead denominator. The ticket is
  not ready until cycle-21 validation and independent exact-head review finish.
- Uncertainty and limitations: the online smoke did not consume artifact quota
  or exercise artifact retention/upload; failed DGX evidence is retained, and
  no current overhead or cross-attempt performance claim is made. Quota
  reservation is tracker-lifetime, and a stuck daemon SDK worker is
  process-lifetime bounded.
- Exactly one next step: validate cycle 21 and obtain an independent exact-head
  `PASS` or justified `PASS WITH NOTE`, then use the live PR handoff for the
  guarded merge audit.
