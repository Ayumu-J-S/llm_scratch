# Fixed held-out validation

`llm-scratch-evaluate` scores a verified repository checkpoint on the exact
held-out windows recorded by that checkpoint's resolved training configuration.
It is intentionally separate from generative benchmarks and refuses
same-corpus memorization profiles.

```bash
uv run llm-scratch-evaluate \
  profile=evaluation \
  evaluation.checkpoint_path=/absolute/path/to/milestone-step-000000001000.pt \
  evaluation.device=cuda \
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

To write an optional compact W&B summary after the local JSON succeeds:

```bash
uv run llm-scratch-evaluate \
  profile=evaluation \
  evaluation.checkpoint_path=/absolute/path/to/milestone.pt \
  evaluation.wandb.enabled=true \
  evaluation.wandb.mode=online
```

Use the same device and precision conditions when exact score parity matters.
Changing an upstream manifest, split, tokenizer, scorer revision, or checkpoint
invalidates prior result identities and requires a fresh evaluation.
