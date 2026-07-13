# Evidence-safe Weights & Biases

W&B is an optional experiment view, not the owner of training state. The local
run manifest, resolved Hydra config, `metrics.jsonl`, checkpoints, and
`wandb_events.jsonl` remain available when W&B is disabled, offline, out of
quota, unreachable, or rejects an operation.

## Modes and scalar logging

The Hydra default is `wandb.mode=disabled`. `offline` writes a local W&B run
without cloud synchronization, and `online` requires a verified login before
initialization. These are the SDK's supported run modes; an online login is
forced and verified so a failed authentication cannot silently turn a claimed
online run into an offline one. See the official [mode
behavior](https://docs.wandb.ai/support/models/articles/what-is-the-difference-between-wandbinit),
[`wandb.init`](https://docs.wandb.ai/models/ref/python/functions/init), and
[`wandb.login`](https://docs.wandb.ai/models/ref/python/functions/login)
references.

```bash
# Local repository evidence only
uv run python src/train.py profile=pretrain_streaming wandb.mode=disabled

# Network-free W&B files plus the same local repository evidence
uv run python src/train.py profile=pretrain_streaming wandb.mode=offline

# Online compact metrics; artifact uploads still default to none
uv run python src/train.py profile=pretrain_streaming wandb.mode=online
```

Scalar logs occur only at the trainer's configured training-log and validation
boundaries; W&B adds no independent cadence. A training boundary emits one
compact dictionary containing step, target tokens, elapsed time, training
NLL/perplexity, target throughput, RSS/peak RSS, PyTorch peak
allocated/reserved memory on CUDA, gradient norm, clipping, non-finite count,
learning rate, and any validation result produced at that same step. When a
validation boundary does not coincide with a scheduled training log, its
compact aggregate and per-corpus scalars are emitted at that exact validation
step instead of being attributed to a later step. W&B documents that each
`Run.log` call normally advances the run step ([logging
reference](https://docs.wandb.ai/models/track/log)).

Scalar SDK calls run on one persistent tracker-owned worker. The training
thread waits at most `wandb.log_timeout_seconds` (default 5 seconds) for each
call. A timeout, SDK exception, or full one-item queue opens a circuit breaker,
records local failure evidence, and suppresses later W&B scalar calls without
stopping local training, metrics, or checkpoints. This bounds the external SDK
boundary without creating a thread per log event.

`wandb.watch.enabled=false` is the default. When explicitly enabled, its hook
type and frequency are configured under `wandb.watch`, and the hooks are
removed before finish. The tracker records the watched model before asking the
SDK to install hooks; if installation partially fails, it immediately requests
global unwatch cleanup and removes residual W&B hook bookkeeping. Normal
model-specific unwatch also falls back to global cleanup on failure. The default
watch interval is 1,000 batches, matching the SDK's documented default and
avoiding the failed R2's excessive 100-batch histogram volume. Model watching
can add gradient/parameter collection work independent of ordinary scalar
logging; see the official [PyTorch integration](https://docs.wandb.ai/models/integrations/pytorch).

The W&B run config contains the resolved training configuration except that
dataset values are reduced to manifest and external-reference metadata. Inline
documents or iterables are never passed to W&B. Compact summary lineage records
the experiment, Git, config, lock, tokenizer, and ordered data fingerprints.

## Artifact policy and quota preflight

Model artifact policy is one of `none|best|final|milestone` and defaults to
`none`. Recovery checkpoints are never upload candidates. Smoke, CI, stability
smoke, and memorization profiles are denied by code even if an override selects
an artifact policy. A candidate must pass all of these gates:

1. The configured policy matches the checkpoint reason.
2. The selected file is a verified repository checkpoint whose internal kind,
   run identity, and optimizer step match the requested artifact reason.
3. The run is online after `wandb.login(force=True, verify=True)` succeeded.
4. The public `wandb.Api().viewer` identity can verify the configured user or
   team entity.
5. A fresh operator-supplied usage snapshot names that same entity and records
   visible plan, storage usage/limit, and retention behavior.
6. Across the tracker lifetime, the maximum observed `used_bytes`, bytes
   reserved for every earlier accepted candidate, current checkpoint bytes,
   and configured safety reserve fit beneath the minimum observed
   `limit_bytes`.
7. The checkpoint SHA-256 has not already uploaded or been reserved in this
   run.
8. The asynchronous artifact reaches `COMMITTED` within the configured timeout.

The tracker reserves candidate bytes under one lock before cloud submission,
so serial or concurrent milestones cannot each spend the same visible
headroom. A reservation is retained when submission or completion is ambiguous;
operator review is required rather than assuming the service stored nothing.
Snapshot refreshes can only tighten the tracker ledger: lower observed usage or
a higher observed limit never relaxes a prior stricter observation.

The public Python API documents run/artifact access but no supported current
storage-usage/quota endpoint. Current usage is instead visible in the Billing
page and downloadable CSV, so the repository deliberately does not use private
GraphQL, infer a quota from a plan name, or hard-code changing service limits.
See the [public API overview](https://docs.wandb.ai/models/ref/python/public-api),
[Billing usage UI/CSV](https://docs.wandb.ai/platform/app/settings-page/billing-settings),
and current [pricing/storage](https://wandb.ai/site/pricing/).

Capture the visible values immediately before an intended upload in an
operator-managed JSON file:

```json
{
  "schema_version": 1,
  "captured_at_utc": "2026-07-13T16:00:00+00:00",
  "entity": "sunday-research",
  "plan": "value displayed in W&B Billing",
  "used_bytes": 123456789,
  "limit_bytes": 1000000000,
  "retention": "value displayed in W&B Billing",
  "source": "W&B Billing UI or downloaded usage CSV"
}
```

Then select one reason and safety reserve:

```bash
uv run python src/train.py profile=pretrain_streaming \
  wandb.mode=online \
  wandb.log_timeout_seconds=5 \
  wandb.artifact.policy=final \
  wandb.artifact.usage_snapshot_path=/operator/wandb-usage.json \
  wandb.artifact.max_usage_age_seconds=900 \
  wandb.artifact.reserve_bytes=250000000 \
  wandb.artifact.upload_timeout_seconds=600
```

`Artifact.add_file` uploads the selected local checkpoint; reference artifacts
are metadata lineage, not checkpoint backup. See the official [artifact
reference](https://docs.wandb.ai/models/ref/python/experiments/artifact) and
[run artifact methods](https://docs.wandb.ai/models/ref/python/experiments/run).

Every decision records policy, reason, checkpoint SHA-256 and bytes when the
candidate reaches identity preflight, auth result, usage/limit/retention,
effective maximum observed usage, tracker-reserved bytes, effective minimum
limit, configured safety reserve, projected bytes, upload outcome, and retry
disposition in `checkpoints/wandb_events.jsonl`. Missing login, unknown or stale
usage, wrong entity, insufficient headroom, network failure, or a non-committed
upload blocks only W&B. The local checkpoint and metrics remain intact. W&B's
own network-loss behavior is described in its [connectivity
documentation](https://docs.wandb.ai/support/models/articles/what-happens-if-internet-connection-is-l),
but local completion never depends on that external behavior.

## WB-001 target-hardware protocol

The retained DGX R2 comparison is Attempt 8 at exact commit
`b59f84483d1f85a5cd42005d48e8b99d60ab2695` in pinned image
`sha256:23a1bee69fe189e77105cdddeee9aeff6ef0763d58a691625fbfcab64efd1887`.
All arms use the same resolved config, seed, pinned data/cache, depth-26 model,
sequence length 64, CUDA/BF16 recipe, validation/checkpoint settings, 260
optimizer steps, 26 warm-up steps, batch size 2, accumulation 4, 1,040
backward batches, and 133,120 target tokens. Three Latin-square repetitions
compare:

- W&B disabled;
- W&B offline with watch off; and
- W&B offline with watch on.

The protocol retains per-step phase timing, target tokens/s, median/p95/max
step time, data wait, RSS/peak RSS, PyTorch peak allocated/reserved memory,
loss, gradient norm, W&B directory bytes, sampler coverage, validation identity,
and normalized checkpoint identities. It compares medians and spreads, not a
best run. Any model, normalized resume, cursor, or trajectory digest difference
is a failure; physical checkpoint file SHA-256 values may differ because each
arm records arm-local operational metadata. A median throughput regression of
5% or more is an investigation note; 10% or more fails. Watch remains off by
default.

`docs/experiments/evidence/run_wb001_dgx.sh` executes the fixed network-isolated
matrix after priming the shared cache, and
`docs/experiments/evidence/verify_wb001_dgx.py` reconstructs the declared
trajectory, checkpoint, lifecycle, resource, storage, and paired-performance
gates from the retained raw evidence.

Attempt 8 passed all 170 verifier gates. Per-arm data wait was 6.53–8.54%; the
paired median changes were 1.75% for offline/watch-off versus disabled, 6.56%
for offline/watch-on versus disabled, and 5.46% for watch-on versus
offline/watch-off. The latter two values and every per-arm data-wait value are
retained as predeclared 5–10% investigation notes, so the R2 result is `PASS
WITH NOTE`. The durable summary is
`docs/experiments/evidence/WB-001-dgx-r8-pass-with-note.json`; failed and
aborted earlier attempts remain in the experiment record.

An optional online scalar arm and one selected artifact may run only when a
human/operator supplies credentials and a fresh usage snapshot with headroom.
It is not required for the network-free implementation proof and must not
upload raw data or recovery checkpoints. Attempt 8 used network isolation and
artifact policy `none`; it does not validate online authentication, live quota
accounting, retention, artifact upload, or other cloud behavior. Its performance
conclusion is limited to the pinned depth-26 workload/runtime and within-attempt
comparisons; it makes no cross-attempt or long-run R3 claim.
