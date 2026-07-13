# WB-001 — Quota-safe W&B preserves local training evidence

- Roadmap ticket: `WB-001`
- Branch: `codex/wb-001-evidence-safe-wandb`
- Draft PR: unavailable during implementation because the delegated runtime has
  no GitHub publication command; complete body prepared at
  `/tmp/WB-001-pr-body.md`
- Experiment owner: implementation agent
- Status: R1 repair validation passed; predeclared DGX R2 comparison pending
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
| Elapsed time on target hardware | 3 repetitions of 3 arms, 100 steps/arm; not run in implementation pass | CHECK §§6.3 and 9.2 repeated disabled/offline/watch comparison |
| Training tokens | Fixed by the selected `stability_smoke` resolved config and identical in every arm | No target-token or objective difference permitted |
| Optimizer steps | 2 CPU smoke invocations; future R2 900 total steps | 100 steps × 3 arms × 3 repetitions |
| Evaluation work and cadence | Identical across all R2 arms; no extra W&B cadence | Isolates W&B/watch overhead |
| Checkpoint count and bytes | Existing local final checkpoint per arm; no W&B artifact in required R2 | Artifact behavior is proven with test doubles, not service quota |
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

## Predeclared Attempt 2 — DGX disabled/offline/watch comparison (not run)

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

## Conclusion

- Hypothesis result: supported at R1; DGX overhead conclusion pending.
- Evidence-backed conclusion: the implementation can preserve local metrics and
  checkpoints across disabled/offline W&B and tested external failure paths,
  while fail-closing every artifact safety gate.
- Uncertainty and limitations: no online service call, real quota consumption,
  artifact upload, or DGX measurement was performed.
- Exactly one next step: independently review the implementation against
  WB-001, `PHILOSOPHY.md`, and selected `CHECK.md`; if code review passes, run
  the predeclared R2 only when the target runtime is intentionally scheduled.
