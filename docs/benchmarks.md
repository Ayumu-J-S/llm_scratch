# Versioned base-model benchmarks

`BENCH-001-suite-v1` is the first deliberately small checkpoint benchmark. It
tracks Japanese commonsense and general mathematical reasoning without adding
a leaderboard framework or any chat/SFT behavior.

## Pinned suite

The immutable registry is
[`data/benchmarks/suite-v1.json`](../data/benchmarks/suite-v1.json). Its compiled
fingerprint is
`af22cfe7ad1db1ed8dc30969177cf0b8fae8061da66a5ae9d69af45077593231`.
The evaluator refuses a registry whose canonical fingerprint differs.

| Task | Development source | Reserved final source | Revision | Scoring |
| --- | --- | --- | --- | --- |
| JCommonsenseQA v1.3 | 128 examples selected by `sha256-example-id-v1` from the official validation set | Complete official test set | JGLUE `6f071c09316baae89c3d083a90985b4b1cb9968c` | Zero-shot conditional choice log probability; length-normalized accuracy is primary |
| GSM8K | 128 examples selected by `sha256-example-id-v1` from the official train set | Complete official test set | grade-school-math `3101c7d5072418e28b9008a6636bde82a006892c` | Zero-shot greedy continuation, at most 128 new tokens; exact match uses the published `####` regex and comma normalization |

The registry records each source URL, byte size, SHA-256, record count, split,
repository, commit URL, and license. The development subset identities are:

- JCommonsenseQA:
  `fa5ce35310f98b171da7db6afeff222381161f1987a99d70e7ede9b77a283b0e`
- GSM8K:
  `3f5c12085bfcb2e4dc94d91d1d3630b476d06343c6c2fc6c6df2d5d87f90daba`

There are zero few-shot examples. Each task's exact prompt and scorer
specification has its own SHA-256 in addition to its readable revision.
Decoding configuration, source identity, selected-example identity, checkpoint
logical and physical identity, tokenizer fingerprint, device, and
checkpoint-owned precision are all attached to the stable evaluation identity.

## Commands and reserved-test boundary

Routine evaluation can only request the development sources:

```bash
make benchmark CHECKPOINT=/absolute/path/to/milestone.pt
```

The Hydra profile intentionally contains no source registry, partition,
reserved-access, or acknowledgement key. Adding such a key is rejected as an
unknown critical control. The separate final entrypoint checks an exact
environment acknowledgement before Hydra composition begins:

```bash
BENCHMARK_FINAL_ACK=BENCH-001-suite-v1 \
  make benchmark-final CHECKPOINT=/absolute/path/to/selected-milestone.pt
```

Final evaluation is intended for infrequent, predeclared decision-grade use.
Do not tune prompts, decoding, checkpoint selection, or model configuration
against its results.

Both commands take model dimensions, tokenizer, precision, training manifests,
and counters only from the verified full-state checkpoint. A CPU request for a
BF16 checkpoint fails; there is no implicit precision or device fallback.

## Complete contamination gate

Before scoring, the evaluator ignores the training horizon and scans every
document in every checkpoint-owned manifest selected as `train`. It records:

- exact whole-document hashes;
- normalized whole-document hashes using the repository text-identity policy;
- SHA-256 matches over normalized 48-codepoint shingles;
- source name, training document ID, upstream ID, manifest/dataset identity,
  scan counts, byte counts, and a complete scan-order digest.

The report contains no benchmark or training text. Any match writes an atomic
`blocked_contamination` result and exits without scoring, so the evidence is
retained while a contaminated score cannot be mistaken for a valid result.
The complete scan may read all pinned training shards and is intentionally a
milestone operation rather than a training hot-path check.

## Evidence and W&B policy

The local result is written atomically. Per-example traces retain only example
IDs, correctness/prediction metadata, counts, stop reasons, and hashes; prompts,
reference text, token IDs, and generated completions are never written.

When `benchmark.wandb.enabled=true`, W&B receives summary metrics and one
two-row table with task, access level, metric, score, correct/total counts, and
the protocol hash. Raw datasets, prompts, completions, per-example traces, and
model artifacts are not uploaded.

Optional external baselines use the separate
`llm-scratch-benchmark-external` aggregate recorder. It requires parameter
count, training-compute, tokenizer, context-length, and data-access disclosure;
rejects raw text, outputs, logits, token IDs, and weights; and marks the record
ineligible for training data or targets. It does not load external weights into
the repository checkpoint runner.
