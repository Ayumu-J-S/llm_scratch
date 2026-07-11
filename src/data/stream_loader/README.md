# Stream Loader

`StreamLoader` mixes one or more text sources by ratio, tokenizes the text, enforces an optional token budget, and yields samples in the requested output format. It is configured with plain mappings, so Hydra/OmegaConf configs can be loaded directly and converted before construction.

## Quick Preview

Preview the default config through the training dataset/collator path, without enabling background prefetch:

```bash
uv run python scripts/debug_stream_loader.py --config config/stream_loader.yaml --limit 1
```

The debug script caps each document to 50,000 characters before tokenization for faster inspection. Use `--max-doc-chars 0` to preview full documents, and `--prefetch` to use the config's prefetch settings.
It prints the trainer batch contract: `inputs`, `labels`, their shapes, and a shift check showing that labels are derived from the same packed token window.

## Training Usage

The default training config still uses the small local text path for quick memorization checks:

```yaml
data:
  mode: local_text
```

Switch to the streaming path when you want to train from larger sources without materializing the full corpus locally:

```bash
uv run python src/train.py data.mode=streaming
```

`src/train.py` builds streaming train and validation loaders from `data.streaming.train` and `data.streaming.validation`, adds the project tokenizer artifact, packs token windows, and returns the standard trainer batch contract:

- `inputs`: left-to-right causal-LM input tokens
- `labels`: the same packed stream shifted by one token

For a first smoke test, keep `max_tokens` small and use the debug script before a full training run.

## Python Usage

```python
from omegaconf import OmegaConf

from data.stream_loader import StreamLoader

cfg = OmegaConf.to_container(OmegaConf.load("config/stream_loader.yaml"), resolve=True)

with StreamLoader(cfg) as loader:
    for sample in loader:
        print(sample["source"], sample["token_count"])
        print(sample["text"][:200])
        break
```

For programmatic tests or small local experiments, pass an in-memory config:

```python
from data.stream_loader import StreamLoader

config = {
    "tokenizer": {
        "kind": "tokenizers",
        "path": "artifacts/tokenizers/tokenizer.json",
        "eos_token": "<eos>",
    },
    "output_mode": "tokenized_docs",
    "max_tokens": 10_000,
    "add_eos": True,
    "seed": 42,
    "datasets": [
        {
            "name": "local",
            "type": "jsonl",
            "path": "tests/fixtures/tiny_corpus.jsonl",
            "text_field": "text",
            "ratio": 1.0,
        }
    ],
}

for sample in StreamLoader(config):
    print(sample["source"], sample["input_ids"])
```

For the training path, wrap the stream with the PyTorch iterable dataset and collator:

```python
from data.streaming_dataset import create_streaming_token_dataloader

loader = create_streaming_token_dataloader(
    config=config,
    sequence_length=64,
    batch_size=8,
)
batch = next(iter(loader))
print(batch["inputs"].shape, batch["labels"].shape)
```

## Config Shape

```yaml
tokenizer:
  kind: qwen
  name: Qwen/Qwen3-0.6B
  use_fast: true

output_mode: raw_text
max_tokens: 5000000000
sequence_length: 4096
add_eos: true
preserve_metadata: true
seed: 42

prefetch:
  enabled: true
  buffer_size: 16

cache:
  dir: data/stream_loader_cache
  max_size_bytes: 750000000000

datasets:
  - name: japanese_educational_text
    type: hf
    path: hotchpotch/fineweb-2-edu-japanese
    revision: 180ca004c6a89b590daaad86cb062a07a5353c69
    config_name: sample_10BT
    split: train
    text_field: text
    ratio: 1.0
```

Use `config/stream_loader.yaml` for dataloader/tokenizer/debug work on the `sample_10BT` subset.

Required top-level fields:

- `tokenizer`: tokenizer mapping, or a tokenizer object passed to `StreamLoader`.
- `datasets`: one or more dataset mappings. `name` values must be unique.
- `ratio`: each dataset needs a positive ratio, and all ratios must sum to `1.0`.

Common optional fields:

- `output_mode`: one of `raw_text`, `bytes`, `tokenized_docs`, or `packed_sequences`.
- `max_tokens`: total token budget, or `"max"` to stream until sources are exhausted.
- `sequence_length`: packed sequence length for `packed_sequences`.
- `add_eos`: append the tokenizer EOS token to each document.
- `preserve_metadata`: include `metadata` for document modes or `source_spans` for packed sequences.
- `seed`: deterministic source sampling seed.
- `prefetch.enabled`: load samples in a background worker.

## Tokenizers

Supported tokenizer config kinds:

- `tokenizers`: load a local Hugging Face `tokenizers` JSON file.
- `hf`, `huggingface`, or `qwen`: load a tokenizer with `transformers.AutoTokenizer`.
- `mistral_tekken` or `tekken`: load a Mistral Tekken tokenizer.

You can pass an already constructed tokenizer object instead of a tokenizer mapping for synchronous or thread-prefetched loading. Process prefetch requires a tokenizer mapping because the worker process needs to construct its own tokenizer.

## Source Types

`memory`

```yaml
- name: smoke
  type: memory
  ratio: 1.0
  documents:
    - text: hello world
```

`iterable`

Use this from Python by passing an iterable or callable under `iterable`.

`jsonl`

```yaml
- name: local_jsonl
  type: jsonl
  path: data/corpus.jsonl
  text_field: text
  ratio: 1.0
```

`url_jsonl`

```yaml
cache:
  dir: data/stream_loader_cache
  max_size_bytes: 10000000000

datasets:
  - name: remote_jsonl
    type: url_jsonl
    url: https://example.com/shard.jsonl
    text_field: text
    ratio: 1.0
```

`url_jsonl` requires `cache.dir` because shards are downloaded into the bounded cache.
Use this for reasonably sized shards, not for one multi-terabyte JSONL object. A single `url_jsonl` source is downloaded to cache before it is read.

`hf`

```yaml
- name: fineweb
  type: hf
  path: HuggingFaceFW/fineweb-edu
  revision: 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
  config_name: sample-10BT
  split: train
  text_field: text
  ratio: 1.0
```

Hugging Face sources require `revision` to be a 40-character commit hash so runs are reproducible.

## Large Dataset Notes

- Prefer `hf` sources with `streaming=True` for public Hugging Face datasets that are too large to store locally.
- Prefer local `jsonl` or `url_jsonl` only when the files are already sharded into manageable pieces.
- Use `max_tokens` for small experimental runs before increasing the budget.
- Keep `num_workers=0` for the PyTorch `DataLoader` unless worker-aware stream partitioning is added.
- Keep `prefetch.enabled` off while debugging; enable it only after the single-process path is working.

## Output Modes

- `raw_text`: yields `source`, `token_count`, and `text`.
- `bytes`: yields `source`, `token_count`, and UTF-8 `bytes`.
- `tokenized_docs`: yields `source`, `token_count`, and `input_ids` as `np.uint32`.
- `packed_sequences`: concatenates tokenized documents and yields fixed-length `input_ids` arrays plus `token_count`.

When `preserve_metadata` is enabled, document modes include `metadata`. `packed_sequences` includes `source_spans` that identify which source contributed each token span.

## Prefetching

```yaml
prefetch:
  enabled: true
  buffer_size: 16
  mode: process
```

If `mode` is omitted, Hugging Face sources default to process prefetch and other sources default to thread prefetch. Thread prefetch is rejected for Hugging Face sources because those streaming iterators are safer in a separate process.

Always close a prefetched loader when stopping early. The context manager does that automatically:

```python
with StreamLoader(config) as loader:
    for sample in loader:
        break
```

## Quotas And Exhaustion

When `max_tokens` is an integer, the loader splits the token budget by dataset ratio. For example, `max_tokens: 1000` with ratios `0.7` and `0.3` emits 700 and 300 tokens respectively. If a source exhausts before its quota is met, `StreamLoaderError` is raised.
