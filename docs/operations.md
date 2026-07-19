# Agent-native experiment operations

`llm-scratch-ops` is the single operational surface for bounded repository
experiments. It composes the existing Hydra profiles and launches the existing
training, evaluation, and benchmark entrypoints; it does not implement a second
trainer or a scheduler.

Every Hydra-capable command requires an explicit executor, device, run root,
run ID, attempt ID, and a literal `--` before Hydra overrides. There is no
executor or device fallback.

```bash
# Network-free CPU smoke
uv run llm-scratch-ops smoke \
  --run-root /operator/llm-runs \
  --run-id OPS-001-smoke \
  --attempt-id attempt-0001 \
  --executor host \
  --device cpu \
  -- \
  training.max_steps=1

# Non-destructive real-profile readiness check
uv run llm-scratch-ops preflight \
  --run-root /operator/llm-runs \
  --run-id RUN-001 \
  --attempt-id attempt-0001 \
  --executor container \
  --image llm-scratch:env-001 \
  --device cuda \
  --profile pretrain_baseline \
  -- \
  wandb.mode=online

# Bounded canonical training
uv run llm-scratch-ops train \
  --run-root /operator/llm-runs \
  --run-id RUN-001 \
  --attempt-id attempt-0002 \
  --executor container \
  --image llm-scratch:env-001 \
  --device cuda \
  --profile pretrain_baseline \
  -- \
  training.max_time=3600 \
  wandb.mode=online
```

The remaining canonical actions use the same prefix:

- `config-check` composes and validates a profile without launching a child;
- `resume --checkpoint <verified.pt> --retry-from <attempt>` requires a failed
  or stopped sibling, derives the canonical profile from the verified
  checkpoint, and verifies checkpoint-owned config/cursor identity;
- `eval --checkpoint <verified.pt>` runs the canonical held-out evaluator;
- `benchmark --checkpoint <verified.pt>` runs only the BENCH-001 development
  subsets;
- `status` validates attempt, retry, PID, and container ownership without
  signaling anything; and
- `handoff` writes a validator-checked `handoff.json` and EXP-001-shaped
  `handoff.md` for a terminal attempt.

## Durable attempt layout

Each attempt is a non-overwriting sibling under
`<run-root>/<run-id>/attempts/<attempt-id>/`. Atomic `state.json` transitions and
one atomic JSON file per event avoid a partially appended event log. The attempt
retains the exact command and resolved config, complete stdout/stderr, preflight
declaration and storage forecast, PID or container ownership, checkpoint
verification, result, diagnosis, and handoff evidence. The scientific question,
conditions, and budget are written before preflight. A retry binds the failed sibling's
terminal state and result hashes; it never rewrites the failed attempt.

Real train, resume, evaluation, and benchmark actions require a clean Git
worktree. Evaluation and benchmark storage/manifest identity comes from the
verified checkpoint's resolved configuration rather than the thin operational
profile. Missing W&B credentials produces a `wandb login` prompt only after all
local checks finish. It blocks online actions but does not block an offline
smoke or a local config check. A host executor may use `WANDB_API_KEY` or the
operator's netrc. A container executor requires `WANDB_API_KEY`; Docker receives
only the environment-variable name, and neither commands nor attempt evidence
persist the credential value.

## Storage safety

Preflight groups run and cache forecasts by filesystem. It accounts for cache
remaining growth, rotating and selected checkpoints, an atomic checkpoint
write, and log headroom. A launch requires both:

- at least a 120 GB operational/live floor, dynamically raised when 100 GB plus
  the maximum in-flight atomic write is larger; and
- at least 100 GB projected free after the complete plan.

The live watchdog samples every projected filesystem, not only the run
directory. Crossing a floor, or losing filesystem visibility, stops only the
verified owned process/container and preserves the failure evidence. If a live
child no longer matches its captured process-group identity, the command does
not signal the old group: it terminates only the exact still-child PID and
records descendant ownership uncertainty. Container operations remain bound to
the immutable full container ID created for the attempt.
