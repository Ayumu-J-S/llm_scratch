# Versioned base-model benchmarks

`BENCH-001-suite-v1` is the first deliberately small checkpoint benchmark. It
tracks Japanese commonsense and general mathematical reasoning without adding
a leaderboard framework or any chat/SFT behavior.

## Pinned suite

The immutable registry is
[`data/benchmarks/suite-v1.json`](../data/benchmarks/suite-v1.json). Its compiled
fingerprint is
`adf433c320252b4d2cbdd6f9817f6f9e34a846bf81b39915a697def1ab477042`.
The evaluator refuses a registry whose canonical fingerprint differs.

| Task | Development source | Reserved final source | Revision | Scoring |
| --- | --- | --- | --- | --- |
| JCommonsenseQA v1.3 | 128 examples selected by `sha256-example-id-v1` from the official validation set | Complete official test set | JGLUE `6f071c09316baae89c3d083a90985b4b1cb9968c` | Zero-shot conditional choice log probability; length-normalized accuracy is primary |
| GSM8K | 128 examples selected by `sha256-example-id-v1` from the official train set | Complete official test set | grade-school-math `3101c7d5072418e28b9008a6636bde82a006892c` | Zero-shot greedy continuation, at most 128 new tokens; exact match uses the published `####` regex and comma normalization |

The registry records each source URL, byte size, SHA-256, record count, split,
repository, commit URL, and license. The development subset identities are:

- JCommonsenseQA:
  `37e39dca6ce5108fe720dda6e0246f7c8ef858e22961229540d2023faeabe0bd`
- GSM8K:
  `03fa95e872665b2be1633781879fe10cc5d04f5b2d1d3add055d60deecf6d9c6`

There are zero few-shot examples. Each task's exact prompt and scorer
specification has its own SHA-256 in addition to its readable revision.
Decoding configuration, source identity, selected-example identity, checkpoint
logical and physical identity, tokenizer fingerprint, device, and
checkpoint-owned precision are all attached to the stable evaluation identity.
The identity also binds the executable evaluator's Git commit, status, and a
content digest over its tracked diff plus every non-ignored untracked file,
along with the dependency-lock hash, OS/Python/PyTorch/CUDA stack, visible
device identity, and container-image metadata. Different dirty evaluator bytes
therefore cannot masquerade as an identical comparison.

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
BF16 checkpoint fails, as does a CUDA request when the selected runtime/device
does not support BF16; there is no implicit precision or device fallback.
Before opening any training shard, the evaluator tokenizes every fixed prompt
and choice and requires enough checkpoint context for JCommonsenseQA scoring
and the complete 128-token GSM8K generation allowance. Both task scorers run
under the checkpoint-owned FP32 or BF16 precision recorded in the result.

## Complete contamination gate

Before scoring, the evaluator ignores the training horizon and scans every
document in every checkpoint-owned manifest selected as `train`. It records:

- exact whole-document hashes;
- normalized whole-document hashes using the repository text-identity policy;
- source-faithful full-record hashes plus text-normalized,
  key-order/whitespace-independent canonical JSON-object identities;
- exact matches over normalized 48-codepoint shingles;
- source name, training document ID, upstream ID, manifest/dataset identity,
  scan counts, byte counts, and a complete scan-order digest.

The report contains no benchmark or training text. Any match writes an atomic
`blocked_contamination` result and exits without scoring, so the evidence is
retained while a contaminated score cannot be mistaken for a valid result.
The first complete scan may read all pinned training shards and is intentionally
an index-building operation rather than a training hot-path check. Its atomic
artifact is keyed by the full suite identity, ordered training manifest
content/fingerprints, normalization/matcher revision, relevant evaluator source
bytes, `uv.lock`, installed PyArrow, and the producing Python/platform runtime;
it also carries its own checksum. The first scan uses a collision-verified
rolling matcher with one constant-work update per corpus codepoint. It does not
slice, UTF-8 encode, or SHA-256 every corpus window, and its retained matcher
state is bounded by unique benchmark shingles. Later milestones with the same
corpus, suite, and producer reuse that verified report without opening or
materializing the corpus again. A corrupt or mismatched index fails closed.

Generated benchmark source shards use the ignored `outputs/benchmark_cache`
directory, so populating the cache does not change evaluator Git identity or
block a later clean pretraining run.

## Evidence and W&B policy

The local result is published atomically beneath the dedicated
`outputs/benchmark-results` root. Overrides cannot escape that root or redirect
it into repository input, cache, checkpoint, or artifact namespaces. Existing
files, symlinks, and hardlinks are never replaced; the default result name is
bound to the complete evaluation identity. Per-example traces retain only
example IDs, correctness/prediction metadata, counts, stop reasons, and hashes.
GSM8K generation evidence includes a versioned SHA-256 over the canonical-JSON
token ID sequence, while the raw IDs remain excluded. Prompts, reference text,
token IDs, and generated completions are never written.

When `benchmark.wandb.mode=online` (or `offline` for local capture), W&B
receives summary metrics and one two-row table with task, access level, metric,
score, correct/total counts, and the protocol hash. Raw datasets, prompts,
completions, per-example traces, and model artifacts are not uploaded.

Optional external baselines use the separate
`llm-scratch-benchmark-external` aggregate recorder. It requires parameter
count, training-compute, tokenizer, context-length, and data-access disclosure.
It also requires a protocol-, prompt-, and scorer-hash-bound no-truncation
preflight with per-task required lengths, enforces a fixed minimum of 129 tokens
for the GSM8K generation contract, and rejects subjects shorter than their
disclosed requirement. It rejects raw text, outputs, logits, token IDs, and
weights; and marks the record ineligible for training data or targets. The
recorder itself attaches the compiled development-suite fingerprint, protocol,
prompt, and scorer hashes, source hashes, selected-example hashes, selector,
access level, and fixed 128-example totals; callers cannot supply an alternate
protocol or partition identity. It does not load external weights into the
repository checkpoint runner. Records can be written only beneath the generated
`outputs/external-comparisons` tree with a `.json` suffix. Paths outside that
tree, existing files, symlinks, hardlinks, and nested artifact/checkpoint
namespaces are rejected before the exclusive atomic publish.
