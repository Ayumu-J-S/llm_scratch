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
  --profile pretrain_streaming \
  --experiment-record /operator/RUN-001.declaration.json \
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
  --profile pretrain_streaming \
  --experiment-record /operator/RUN-001.declaration.json \
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

For an actual experiment, pass an immutable `--experiment-record` JSON object.
Generated OPS-001 declarations are fixtures for command-surface checks, not a
substitute for the experiment's predeclared question. The explicit record has
exactly this schema:

```json
{
  "schema_version": 1,
  "ticket": "RUN-001",
  "predeclared_question": {
    "hypothesis": "...",
    "expected_result": "...",
    "success_condition": "...",
    "failure_condition": "...",
    "stop_condition": "...",
    "baseline": {"run_id": "...", "attempt_id": "..."}
  },
  "planned_budget": {"max_time_seconds": 3600, "max_steps": 1000}
}
```

The command records the source path and SHA-256 before preflight, then copies
the exact declared ticket, question, baseline, decision conditions, and budget
into the final handoff.

Real train, resume, evaluation, and benchmark actions require a clean Git
worktree. Evaluation and benchmark storage/manifest identity comes from the
verified checkpoint's resolved configuration rather than the thin operational
profile. Missing W&B credentials produces a `wandb login` prompt only after all
local checks finish. It blocks online actions but does not block an offline
smoke or a local config check. A host executor may use `WANDB_API_KEY` or the
operator's netrc. A container executor requires `WANDB_API_KEY`; Docker receives
only the environment-variable name, and neither commands nor attempt evidence
persist the credential value.

Container preflight also commits an exact bind-mount plan. It includes the
repository and run root as writable, linked-worktree/common Git metadata as
read-only, configured caches as writable, and tokenizer/data manifests, W&B
usage snapshots, and input checkpoints as read-only. Every external absolute
path must already exist; otherwise preflight rejects the launch. The same mount
manifest is consumed by the Git probe and the created training container.

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

`SIGTERM` and `SIGHUP` are captured by the operator and routed through the same
exact-owned process/container cleanup path. The terminal result and event log
record the signal name rather than bypassing checkpoint/result finalization.
