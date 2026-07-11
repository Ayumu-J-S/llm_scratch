# llm_scratch

A small scratch project for experimenting with a decoder-only autoregressive Transformer and a character-level BPE tokenizer.

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

### Train the tokenizer
```bash
uv run python src/train_tokenizer.py
```

This produces the tokenizer artifact configured by `artifacts.tokenizers_dir` and `artifacts.tokenizer_filename`.

### Run the model training script
```bash
make train
```

This launches the Hydra-based training entrypoint, which expects an already-trained tokenizer artifact:

```bash
uv run python src/train.py
```

Training uses a decoder-only autoregressive setup built from one corpus:
- each sample is a left-to-right language modeling window served lazily from a `Dataset`/`DataLoader` pipeline
- the tokenized corpus is treated as one continuous stream
- each training input is a contiguous slice of that stream with fixed length
- labels are the next-token-shifted slice for standard causal language modeling

The default training config now defines `data.train` and `data.val`, with validation pointing to the same `data/inputLearnText.txt` file by default:
- this is deliberate for short-run memorization checks and explicit overfitting experiments
- optimizer class selection lives under `training.optimizer._target_`
- learning-rate scheduler selection lives under `training.scheduler._target_`
- training logs per-step `train/loss_step` plus epoch-aggregated train/validation loss and perplexity to Weights & Biases
- when W&B is enabled, training automatically enables W&B model watching with default settings so gradient panels can be collected
- this does not add a custom scalar grad-norm line, and short runs may still show little or no gradient data with W&B's default watch behavior
- when W&B is enabled, training also logs the final `model_last.pth` checkpoint as a model artifact
- you can additionally log model artifacts during training with `wandb.log_model_every_n_epoch=<n>`

At inference time, the model predicts one tokenizer token at a time, not one whole word at a time. Because the tokenizer is character-level BPE, a word like `give` may be produced over multiple decoding steps such as `g`, `iv`, then `e `.

You can override runtime values with Hydra arguments, for example:

```bash
uv run python src/train.py training.epochs=10 training.batch_size=64
```

W&B is enabled by default. After syncing dependencies and authenticating with W&B, you can run:

```bash
uv run python src/train.py
```

Useful overrides:

```bash
uv run python src/train.py wandb.mode=offline
uv run python src/train.py wandb.enabled=false
uv run python src/train.py data.val=data/inputLearnText.txt
uv run python src/train.py training.optimizer._target_=torch.optim.SGD training.optimizer.lr=0.1 training.optimizer.momentum=0.9
uv run python src/train.py training.scheduler.enabled=true training.scheduler._target_=torch.optim.lr_scheduler.CosineAnnealingLR training.scheduler.T_max=10
```

## Project files
- `ROADMAP.md`: dependency-ordered engineering and research tickets
- `PHILOSOPHY.md`: project decision policy and research principles
- `CHECK.md`: selective post-implementation ML-system review catalog
- `docs/agent-model-workflow.md`: required implementation, heavy-review, repair, and handoff flow
- `docs/experiments/`: predeclared experiment plans, per-attempt evidence, and conclusions
- `docs/model-runs/`: per-PR model execution records and aggregate model outcomes
- `src/train.py`: Hydra-based decoder-only training script that loads a saved tokenizer
- `src/training/trainer.py`: decoder-only trainer loop, validation, checkpointing, and W&B logging
- `src/train_tokenizer.py`: Hydra-based tokenizer training script
- `src/models/embedding.py`: token embedding and sinusoidal positional encoding
- `src/models/simple_decoder_transformer.py`: GPT-style decoder-only Transformer blocks with causal self-attention
- `src/tokenizer/bpe.py`: character-level BPE tokenizer implementation
- `src/data/text_dataset.py`: decoder-only autoregressive dataset and dataloader helpers
- `src/data/stream_loader/`: streaming text loaders for large local or Hugging Face datasets
- `src/data/stream_loader/README.md`: stream loader usage notes, config shape, and large-dataset cautions
- `scripts/debug_stream_loader.py`: preview streaming batches before training
- `config/train.yaml`: model training configuration
- `config/stream_loader.yaml`: standalone streaming loader configuration for preview/debug runs
- `config/train_tokenizer.yaml`: tokenizer training configuration
- `data/inputLearnText.txt`: training corpus
