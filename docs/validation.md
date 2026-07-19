# Fixed held-out validation

`llm-scratch-evaluate` scores a verified repository checkpoint on the exact
held-out windows recorded by that checkpoint's resolved training configuration.
It is intentionally separate from generative benchmarks and refuses
same-corpus memorization profiles.

```bash
uv run llm-scratch-evaluate \
  profile=evaluation \
  evaluation.checkpoint_path=/absolute/path/to/milestone-step-000000001000.pt \
  evaluation.output_path=evaluation.json
```

The canonical profile defaults to `evaluation.device=cuda` because real
pretraining checkpoints own `training.precision=bf16`. The evaluator rejects a
CPU/BF16 pairing during preflight; it never changes checkpoint precision or
silently falls back to FP32.

For a bounded fixture checkpoint that itself records
`training.precision=fp32`, CPU scoring is explicit:

```bash
uv run llm-scratch-evaluate \
  profile=evaluation \
  evaluation.checkpoint_path=/absolute/path/to/fp32-fixture-milestone.pt \
  evaluation.device=cpu \
  evaluation.output_path=evaluation.json
```

The checkpoint owns model dimensions, weights, tokenizer identity, precision,
and train/validation manifests. Hydra's `evaluation` profile controls only the
checkpoint path, scoring device, local result path, and optional compact W&B
summary. The training entrypoint rejects this profile.

Each run atomically writes standards-compliant JSON containing checkpoint kind,
path, SHA-256, counters, tokenizer and resolved-config identities, scorer
revision, manifest/dataset/split identities, and per-corpus plus token-weighted
aggregate NLL/perplexity. Exact window and target-stream SHA-256 values identify
the contexts, labels, masks, order, and corpus attribution without storing raw
held-out text or token arrays.

Training-time validation calls the same scorer with a fresh deterministic
loader at every configured step/token cadence. Its logical checkpoint identity
contains the run identity and current optimizer/token counters; standalone
evaluation additionally records the physical checkpoint file. Same-corpus
smoke and memorization-gate profiles use only the `memorization/*` namespace and
do not create a best-validation checkpoint.

Both local training metrics and W&B records include the scorer revision so a
training-time result can be compared with standalone evidence only when their
scoring implementations match. Validation phase timing is opt-in through
`measurement.enabled`. CPU phase values are host wall time. On CUDA,
`forward_seconds` and `loss_seconds` are reported only when
`measurement.cuda_events=true`; they use CUDA events plus one explicit
synchronization at the end of the scoring pass, not host enqueue latency. With
measurement disabled (or CUDA events disabled on CUDA), the scorer returns no
phase-timing map and adds no CUDA events or explicit synchronization.
`pause_seconds` remains the full observed validation pause in every mode.

CUDA scoring hashes input/label/source identity from the loader's host tensors
before transfer. Loss sums and finite flags remain device tensors throughout
the bounded validation pass; there is no per-batch `.cpu()`, `.item()`, or
device-valued Python condition. When CUDA-event timing is enabled, the scorer
records every phase end, performs its one explicit end-of-pass synchronization,
and only then materializes the bounded aggregate/per-corpus sums and finite
flags. Without event timing, the same bounded final materialization is the only
required completion boundary.

To write an optional compact W&B summary after the local JSON succeeds:

```bash
uv run llm-scratch-evaluate \
  profile=evaluation \
  evaluation.checkpoint_path=/absolute/path/to/milestone.pt \
  evaluation.device=cuda \
  evaluation.wandb.mode=online
```

Use the same device and checkpoint-owned precision conditions when exact score
parity matters. `training.precision` is not an evaluator override.
Changing an upstream manifest, split, tokenizer, scorer revision, or checkpoint
invalidates prior result identities and requires a fresh evaluation.
