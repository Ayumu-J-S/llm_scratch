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

`wandb.watch.enabled=false` is the default. When explicitly enabled, its hook
type and frequency are configured under `wandb.watch`, and the hooks are
removed before finish. Model watching can add gradient/parameter collection
work independent of ordinary scalar logging; see the official [PyTorch
integration](https://docs.wandb.ai/models/integrations/pytorch).

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
6. `used_bytes + checkpoint_size_bytes + reserve_bytes <= limit_bytes`.
7. The checkpoint SHA-256 has not already uploaded in this run.
8. The asynchronous artifact reaches `COMMITTED` within the configured timeout.

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
reserve and projected bytes, upload outcome, and retry disposition in
`checkpoints/wandb_events.jsonl`. Missing login, unknown or stale usage,
wrong entity, insufficient headroom, network failure, or a non-committed upload
blocks only W&B. The local checkpoint and metrics remain intact. W&B's own
network-loss behavior is described in its [connectivity
documentation](https://docs.wandb.ai/support/models/articles/what-happens-if-internet-connection-is-l),
but local completion never depends on that external behavior.

## WB-001 target-hardware protocol

No DGX result is claimed by the implementation pass. The predeclared R2
comparison uses the same exact commit, resolved config, seed, pinned data/cache,
model, CUDA/BF16 recipe, validation/checkpoint settings, and 100 optimizer steps
for all arms. After 10 warm-up steps, run three alternating repetitions of:

- W&B disabled;
- W&B offline with watch off; and
- W&B offline with watch on.

Retain per-step phase timing, target tokens/s, median/p95/max step time, data
wait, RSS/peak RSS, PyTorch peak allocated/reserved memory, loss, gradient norm,
and W&B directory bytes. Compare medians and spreads, not a best run. Any
trajectory difference is a failure. Investigate a median throughput regression
of 5% or more; 10% or more normally fails. Watch remains off for the baseline
unless the measurement demonstrates a bounded need and cost.

`docs/experiments/evidence/run_wb001_dgx.sh` executes the fixed network-isolated
matrix after priming the shared cache, and
`docs/experiments/evidence/verify_wb001_dgx.py` reconstructs the declared
trajectory, checkpoint, lifecycle, resource, storage, and paired-performance
gates from the retained raw evidence.

An optional online scalar arm and one selected artifact may run only when a
human/operator supplies credentials and a fresh usage snapshot with headroom.
It is not required for the network-free implementation proof and must not
upload raw data or recovery checkpoints.
