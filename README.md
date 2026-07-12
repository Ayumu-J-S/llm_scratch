# llm_scratch

A small scratch project for experimenting with a decoder-only autoregressive Transformer and a pinned Japanese/English tokenizer.

## Setup

### Prerequisites
- [uv](https://docs.astral.sh/uv/) installed
- Python 3.10+

### Create or update the environment
```bash
make sync
```

This uses `uv sync` to create or update the local `.venv` from `pyproject.toml`, and `uv` will generate or refresh `uv.lock` as needed.

### Activate the environment
```bash
make activate
```

`make activate` prints the command you should run in your current shell:

```bash
source .venv/bin/activate
```

A Make target cannot directly modify the parent shell, so the activation command must be run manually.

## Common commands

### Canonical tokenizer

Training, streaming, debugging, model construction, and future generation use
the vendored LLM-jp v1 tokenizer selected by TOK-001. Hydra selects it through
`config/tokenizer/canonical.yaml`. Startup validates its manifest fingerprint,
upstream revisions, Apache-2.0 license, artifact bytes, vocabulary, special
tokens, and fixed probes entirely offline. There is no project tokenizer
training command or runtime tokenizer download.

### Canonical Hydra workflows

Profiles are explicit and compose into the same `src/train.py` entrypoint:

```bash
# Fast, CPU-only memorization smoke (the only same-corpus profile)
make smoke

# Compose and preflight the manifest-backed real stream without training
make config-check PROFILE=pretrain_streaming

# Run the real manifest-backed stream profile on the selected CUDA device
make pretrain-streaming
```

The equivalent importable console commands are available after `make sync`:

```bash
uv run llm-scratch-config-check profile=pretrain_streaming
uv run llm-scratch-train profile=smoke_overfit runtime.device=cpu training.epochs=1 training.batch_size=2 wandb.enabled=false
```

`config/profile/smoke_overfit.yaml`, `pretrain_streaming.yaml`, and
`evaluation.yaml` are the canonical Hydra profiles. `evaluation` is a
composition placeholder until the evaluation ticket lands; it is not accepted
by the training loop as a hidden fallback. Every training invocation writes
the fully resolved configuration to `runs/<profile>/<timestamp>/resolved_config.yaml`
(the installed console wrapper uses `runs/<profile>/manual/`).

### Run the model training script
```bash
make train
```

This launches the Hydra-based training entrypoint:

```bash
uv run python src/train.py
```

Training uses a decoder-only autoregressive setup selected by the profile:
- each sample is a left-to-right language modeling window served lazily from a `Dataset`/`DataLoader` pipeline
- the tokenized corpus is treated as one continuous stream
- each training input is a contiguous slice of that stream with fixed length
- labels are the next-token-shifted slice for standard causal language modeling

The default `smoke_overfit` profile uses the committed `memorization_smoke` manifest
for both train and validation:
- a mutable path or a user-provided purpose label cannot authorize same-corpus training
- the manifest fingerprint, source checksum, document identities, and explicit smoke purpose are verified before tokenization
- optimizer class selection lives under `training.optimizer._target_`
- learning-rate scheduler selection lives under `training.scheduler._target_`
- training logs per-step `train/loss_step` plus epoch-aggregated train/validation loss and perplexity to Weights & Biases
- when W&B is enabled, training automatically enables W&B model watching with default settings so gradient panels can be collected
- this does not add a custom scalar grad-norm line, and short runs may still show little or no gradient data with W&B's default watch behavior
- when W&B is enabled, training also logs the final `model_last.pth` checkpoint as a model artifact
- you can additionally log model artifacts during training with `wandb.log_model_every_n_epoch=<n>`

At inference time, the model predicts one tokenizer token at a time, not one
whole word at a time. The canonical vocabulary has 50,570 IDs; BOS is 1, PAD is
4, and EOS/EOD is 7. The manifest fingerprint is logged so checkpoints and
future generation can enforce the same tokenizer identity.

You can override runtime values with Hydra arguments, for example:

```bash
uv run python src/train.py profile=pretrain_streaming training.epochs=10 training.batch_size=2
```

W&B is enabled by default. After syncing dependencies and authenticating with W&B, you can run:

```bash
uv run python src/train.py
```

Useful overrides:

```bash
uv run python src/train.py wandb.mode=offline
uv run python src/train.py wandb.enabled=false
uv run python src/train.py training.optimizer._target_=torch.optim.SGD training.optimizer.lr=0.1 training.optimizer.momentum=0.9
uv run python src/train.py training.scheduler.enabled=true training.scheduler._target_=torch.optim.lr_scheduler.CosineAnnealingLR training.scheduler.T_max=10
```

### Build an immutable data manifest

Real streaming configs set `require_manifests: true`. Their local or downloaded
JSONL sources are checked for byte size and SHA-256 before tokenization, then
document IDs, normalized-content hashes, split assignments, and dataset
fingerprints are checked against the committed index. Build a local manifest
and index with:

```bash
PYTHONPATH=src uv run python scripts/build_data_manifest.py --help
```

The committed bilingual manifest is only a smoke fixture. The real Japanese
source remains disabled until its full immutable shard inventory is recorded.
See `data/manifests/README.md`.

## Project files
- `ROADMAP.md`: dependency-ordered engineering and research tickets
- `PHILOSOPHY.md`: project decision policy and research principles
- `CHECK.md`: selective post-implementation ML-system review catalog
- `docs/agent-model-workflow.md`: required implementation, heavy-review, repair, and handoff flow
- `docs/experiments/`: predeclared experiment plans, per-attempt evidence, and conclusions
- `docs/model-runs/`: per-PR model execution records and aggregate model outcomes
- `src/train.py`: Hydra-based decoder-only training script using the canonical tokenizer
- `src/training/trainer.py`: decoder-only trainer loop, validation, checkpointing, and W&B logging
- `src/models/embedding.py`: token embedding and sinusoidal positional encoding
- `src/models/simple_decoder_transformer.py`: GPT-style decoder-only Transformer blocks with causal self-attention
- `src/tokenizer/canonical.py`: strict offline tokenizer manifest/wrapper validation
- `assets/tokenizers/llm-jp-v1/`: pinned tokenizer JSON, manifest, source notice, and license
- `src/data/text_dataset.py`: decoder-only autoregressive dataset and dataloader helpers
- `src/data/identity.py`, `src/data/manifests.py`, and `src/data/splits.py`: immutable data identity and deterministic split contract
- `src/data/stream_loader/`: streaming text loaders for large local or Hugging Face datasets
- `src/data/stream_loader/README.md`: stream loader usage notes, config shape, and large-dataset cautions
- `scripts/debug_stream_loader.py`: preview streaming batches before training
- `config/train.yaml`: model training configuration
- `config/tokenizer/canonical.yaml`: canonical tokenizer manifest path and expected fingerprint
- `config/stream_loader.yaml`: standalone streaming loader configuration for preview/debug runs
- `data/inputLearnText.txt`: training corpus
