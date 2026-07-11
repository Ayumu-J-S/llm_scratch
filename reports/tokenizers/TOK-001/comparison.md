# TOK-001 frozen tokenizer comparison

Generated from corpus `16d2596017853928346bbf6270fc723d67799b8387080aaa172c42a19804a45a` (160 documents) and candidate lock `56cec8a5e01dac1eeda829057f615d618ce66f729f5d667ea1f7e801b5d7a8f7`.
No model weights, model configuration, generation configuration, or chat template was downloaded or loaded.

## Decision

**Selected: `llm-jp-v1`.** llm-jp-v1 is the smallest eligible vocabulary; no larger candidate met the predeclared bilingual compression and throughput promotion thresholds.

The hard gates precede the Pareto comparison. A larger vocabulary can replace the smallest eligible candidate only under the promotion rule frozen in `candidates.lock.json`; individual timing passes remain in `comparison.json`.

## Candidate disposition

| Candidate | Revision/source | License evidence | Status | Hard gates |
| --- | --- | --- | --- | --- |
| `llm-jp-v1` | [`c3134b3a958b56d443c1484a3d640502637cfbd2`](https://huggingface.co/llm-jp/llm-jp-13b-v1.0/tree/c3134b3a958b56d443c1484a3d640502637cfbd2) | [official evidence](https://raw.githubusercontent.com/llm-jp/llm-jp-tokenizer/132f21625417ed0f3dc6484bf0bc1fb6a433acdd/LICENSE) | measured | PASS |
| `rinna-bilingual` | [`803fb7671ac30766ffc6d21139d809b549ee26a3`](https://huggingface.co/rinna/bilingual-gpt-neox-4b/tree/803fb7671ac30766ffc6d21139d809b549ee26a3) | [official evidence](https://huggingface.co/rinna/bilingual-gpt-neox-4b/blob/803fb7671ac30766ffc6d21139d809b549ee26a3/README.md#license) | excluded | N/A |
| `llm-jp-v3` | [`cd3823f4c1fcbb0ad2e2af46036ab1b0ca13192a`](https://huggingface.co/llm-jp/llm-jp-3-13b/tree/cd3823f4c1fcbb0ad2e2af46036ab1b0ca13192a) | [official evidence](https://raw.githubusercontent.com/llm-jp/llm-jp-tokenizer/c693f9d89cfd25060c872096809c83756ba58eb0/LICENSE) | measured | PASS |
| `qwen3-control` | [`c1899de289a04d12100db370d81485cdf75e47ca`](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) | [official evidence](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/LICENSE) | measured | FAIL |

`rinna-bilingual` exclusion: The exact revision declares MIT in README metadata and links the generic MIT text, but contains no repository-owned LICENSE file or tokenizer-specific redistribution notice. The predeclared redistribution/license-file gate therefore excludes it; no SentencePiece runtime or tokenizer artifact was added solely for this candidate.

## Compression and sequence-length tail

| Candidate | Stratum | tok/codepoint | tok/UTF-8 byte | doc tokens p50 | p95 | p99 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `llm-jp-v1` | japanese | 0.6576 | 0.2211 | 48.5 | 88.4 | 94.5 |
| `llm-jp-v1` | english | 0.2104 | 0.2104 | 44.5 | 72.2 | 75.2 |
| `llm-jp-v1` | mixed | 0.4046 | 0.2841 | 41.0 | 75.3 | 79.9 |
| `llm-jp-v1` | code_symbols | 0.5751 | 0.5603 | 72.5 | 108.4 | 114.5 |
| `llm-jp-v1` | emoji_unicode | 0.8879 | 0.5755 | 54.0 | 142.9 | 172.6 |
| `llm-jp-v1` | whitespace_normalization | 0.3778 | 0.3469 | 5.5 | 10.0 | 10.0 |
| `llm-jp-v1` | short | 1.4828 | 0.9556 | 2.0 | 4.0 | 4.8 |
| `llm-jp-v1` | long | 0.4023 | 0.2337 | 865.5 | 1266.3 | 1332.5 |
| `llm-jp-v3` | japanese | 0.5019 | 0.1687 | 40.0 | 64.2 | 67.2 |
| `llm-jp-v3` | english | 0.1951 | 0.1951 | 36.5 | 64.0 | 64.0 |
| `llm-jp-v3` | mixed | 0.2958 | 0.2077 | 30.0 | 51.5 | 58.3 |
| `llm-jp-v3` | code_symbols | 0.4005 | 0.3902 | 47.0 | 74.5 | 81.3 |
| `llm-jp-v3` | emoji_unicode | 0.7192 | 0.4661 | 48.0 | 112.7 | 137.7 |
| `llm-jp-v3` | whitespace_normalization | 0.3206 | 0.2945 | 4.5 | 10.0 | 10.0 |
| `llm-jp-v3` | short | 0.8966 | 0.5778 | 1.0 | 2.0 | 2.8 |
| `llm-jp-v3` | long | 0.3511 | 0.2040 | 753.0 | 1109.2 | 1158.7 |
| `qwen3-control` | japanese | 0.7175 | 0.2412 | 55.0 | 93.4 | 99.5 |
| `qwen3-control` | english | 0.1786 | 0.1786 | 33.0 | 57.0 | 72.2 |
| `qwen3-control` | mixed | 0.3435 | 0.2412 | 31.5 | 63.9 | 76.8 |
| `qwen3-control` | code_symbols | 0.3694 | 0.3599 | 44.5 | 71.9 | 85.6 |
| `qwen3-control` | emoji_unicode | 0.5492 | 0.3559 | 36.5 | 74.1 | 106.0 |
| `qwen3-control` | whitespace_normalization | 0.2794 | 0.2566 | 4.0 | 9.1 | 9.8 |
| `qwen3-control` | short | 0.7586 | 0.4889 | 1.0 | 1.1 | 2.6 |
| `qwen3-control` | long | 0.3861 | 0.2243 | 837.0 | 1209.3 | 1274.7 |

## Runtime (one warmup, five warm passes)

| Candidate | docs/s median [min,max] | codepoints/s median | bytes/s median | tokens/s median | latency µs p50/p95/p99 | load s | fresh VmHWM after corpus |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llm-jp-v1` | 11,254.0 [11,194.7,11,324.0] | 3,851,116 | 6,366,173 | 1,608,758 | 18.6/590.0/761.8 | 0.0419 | 80.6 MiB |
| `llm-jp-v3` | 10,127.0 [10,100.8,10,157.8] | 3,465,469 | 5,728,670 | 1,229,421 | 19.9/651.5/839.2 | 0.0732 | 123.1 MiB |
| `qwen3-control` | 12,445.4 [12,304.5,12,494.8] | 4,258,802 | 7,040,106 | 1,627,853 | 27.6/491.8/635.9 | 0.1937 | 154.0 MiB |

## Vocabulary-driven cost (embed 384, batch 64 × sequence 64)

The conventional model has untied token embeddings and a biased LM head. Training state is estimated as FP32 parameter + gradient + two Adam moments (16 bytes/parameter); checkpoint is model + two Adam moments (12 bytes/parameter). Activation and non-vocabulary model costs are intentionally excluded.

| Candidate | Vocab | vocab params | FP32 training state | model+Adam checkpoint | BF16/FP32 logits | LM-head MACs/batch | tokenizer files |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `llm-jp-v1` | 50,570 | 38,888,330 | 593.39 MiB | 445.04 MiB | 395.08 MiB / 790.16 MiB | 79,539,732,480 | 3.10 MiB |
| `llm-jp-v3` | 99,574 | 76,572,406 | 1,168.40 MiB | 876.30 MiB | 777.92 MiB / 1,555.84 MiB | 156,616,359,936 | 6.12 MiB |
| `qwen3-control` | 151,669 | 116,633,461 | 1,779.69 MiB | 1,334.76 MiB | 1,184.91 MiB / 2,369.83 MiB | 238,554,710,016 | 10.89 MiB |

## Correctness gates

### `llm-jp-v1` — PASS

- Token IDs SHA-256: `3c1078f72957170fd3c7ac94c9d3313b367f3bf243562a693588810f07dfe907`; vocab/max encoded/max vocab ID: 50,570/50,567/50,569.
- PAD/EOS/BOS/UNK: `{'token': '<pad|LLM-jp>', 'id': 4}`, `{'token': '<EOD|LLM-jp>', 'id': 7}`, `{'token': '<s|LLM-jp>', 'id': 1}`, `{'token': '<unk|LLM-jp>', 'id': 0}`.
- Pipeline: model `Unigram` (byte_fallback=True), normalizers `['Sequence', 'Replace']`, pre-tokenizers `[]`, decoders `['Sequence', 'ByteFallback', 'Replace', 'Fuse']`.
- Exact round trips: 160/160; unknown tokens: 0; explicit `<0xHH>` fallback tokens: 1618; corpus exceptions: 0.
- Malformed recipes: 10; all rejected deterministically: True.
- Failed gates: none.

### `llm-jp-v3` — PASS

- Token IDs SHA-256: `dc264e41423e0aacc3095a59be127ed9ff3a1676a8fbb77efc7756b1fbc65072`; vocab/max encoded/max vocab ID: 99,574/99,275/99,573.
- PAD/EOS/BOS/UNK: `{'token': '<PAD|LLM-jp>', 'id': 4}`, `{'token': '</s>', 'id': 2}`, `{'token': '<s>', 'id': 1}`, `{'token': '<unk>', 'id': 0}`.
- Pipeline: model `Unigram` (byte_fallback=True), normalizers `['Sequence', 'Replace']`, pre-tokenizers `[]`, decoders `['Sequence', 'ByteFallback', 'Replace', 'Fuse']`.
- Exact round trips: 160/160; unknown tokens: 0; explicit `<0xHH>` fallback tokens: 1488; corpus exceptions: 0.
- Malformed recipes: 10; all rejected deterministically: True.
- Failed gates: none.

### `qwen3-control` — FAIL

- Token IDs SHA-256: `1b963e8bf1bc25dc44b50a1f53df5f9b34ab4855288d31f5e82538a777ba22fb`; vocab/max encoded/max vocab ID: 151,669/151,476/151,668.
- PAD/EOS/BOS/UNK: `{'token': None, 'id': None}`, `{'token': '<|endoftext|>', 'id': 151643}`, `{'token': None, 'id': None}`, `{'token': '<unk>', 'id': 128244}`.
- Pipeline: model `BPE` (byte_fallback=False), normalizers `['NFC']`, pre-tokenizers `['Sequence', 'Split', 'ByteLevel']`, decoders `['ByteLevel']`.
- Exact round trips: 154/160; unknown tokens: 0; explicit `<0xHH>` fallback tokens: 0; corpus exceptions: 0.
- Malformed recipes: 10; all rejected deterministically: True.
- Failed gates: exact_unicode_round_trip.

## Reproduction

```bash
uv run python src/tokenizer/comparison.py --fetch --fetch-only
uv run python src/tokenizer/comparison.py
uv run pytest tests/test_tokenizer_comparison.py -q
```

Environment: `Linux-6.17.0-1021-nvidia-aarch64-with-glibc2.39`, Python `3.11.15`, tokenizers `0.22.2`, machine `aarch64`. Cache root: `/home/ayumu/.cache/llm-scratch/tokenizers/TOK-001`.

The comparison is CPU R1 evidence. It does not claim DGX R2 model-step throughput or a final integration verdict; winner packaging and the real streamed-batch/model check remain in TOK-001 phase 2.
