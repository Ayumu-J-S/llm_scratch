# Checkpoint-based base-model continuations

`llm-scratch-generate` is a small local sampler for observing a model saved by
this repository's full-state checkpoint path. Its output is always labeled
`base-model-continuation`: it is prompt completion from the pretrained decoder,
not a chat response or instruction-following interface.

Use a checkpoint produced by `Trainer` (for example `final.pt`, `best.pt`, or a
milestone) and provide text plus a positive continuation budget:

```bash
uv run llm-scratch-generate \
  --checkpoint runs/smoke_overfit/.../checkpoints/final.pt \
  --prompt "Small language models" \
  --max-new-tokens 32 \
  --json
```

The command reconstructs the canonical tokenizer, decoder dimensions, context
length, and weights from the checkpoint's verified full-state payload. There
are intentionally no architecture or tokenizer override flags. Treat a
checkpoint as a local trusted artifact: full-state PyTorch checkpoints include
the repository's recovery state and should not be loaded from an untrusted
source.

Omit `--temperature` for greedy decoding. For reproducible stochastic sampling,
provide a positive temperature and a seed; `--top-k` is optional and applies
only in that mode:

```bash
uv run llm-scratch-generate \
  --checkpoint runs/smoke_overfit/.../checkpoints/final.pt \
  --prompt "小さな言語モデル" \
  --max-new-tokens 32 \
  --temperature 0.8 \
  --top-k 40 \
  --seed 42
```

Generation stops at EOS, after the positive `max_new_tokens` request, or when
the checkpoint-owned context limit is full. The JSON metadata records which
bound stopped the continuation, the checkpoint kind/optimizer step, tokenizer
fingerprint, prompt token count, sampling settings, and generated token IDs.

This command does not add chat templates, SFT behavior, serving, batching, KV
cache optimization, quantization, or a model-quality claim.
