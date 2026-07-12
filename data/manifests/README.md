# Data manifests

Runtime data is admitted through a strict, fingerprinted manifest and document
index. `pretraining` manifests expose disjoint `train` and `validation`
selections. Same-corpus use is limited to `memorization_smoke`. Benchmark
manifests are rejected by the training access path; `benchmark_reserved`
additionally requires an explicit evaluation-only grant.

Build a local JSONL manifest with:

```bash
PYTHONPATH=src uv run python scripts/build_data_manifest.py --help
```

The small committed examples under `tests/fixtures/data_manifests/` are CC0
test fixtures, not a real pretraining mixture.

## DATA-004 real-source inventories

The two schema-v2 manifests are the immutable source authority for the first
real bilingual baseline:

- `fineweb-en-sample-10bt.manifest.json`: plain English FineWeb
  `sample-10BT`, upstream `train` only.
- `fineweb2-ja-jpn-jpan.manifest.json`: direct Japanese FineWeb2
  `jpn_Jpan`, upstream `train` only. The upstream `test` artifact is excluded.

Both use the shared `normalized_content_sha256_v1` split policy, salt
`llm-scratch-data-004-split-v1`, and validation fraction `0.01`. Split
membership is derived from normalized content after the deterministic document
policy, so changing source order or prefetch cannot move a document between
project train and validation.

Regenerate the complete artifact inventories from the exact official Hugging
Face revision APIs without downloading corpus shards:

```bash
python3 scripts/capture_hf_source_manifests.py
python3 scripts/capture_hf_source_manifests.py --check
```

The capture fails closed if a repository/revision, dataset-card license,
config-to-split path, artifact count, aggregate byte count, LFS byte size, or
LFS SHA-256 identity differs. Each dataset fingerprint covers the source name,
full shard inventory, field mapping, and document policy. The manifest
fingerprint additionally covers usage terms and split construction.

Primary records:

- FineWeb card: <https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/9bb295ddab0e05d785b879661af7260fed5140fc/README.md>
- FineWeb revision API: <https://huggingface.co/api/datasets/HuggingFaceFW/fineweb/revision/9bb295ddab0e05d785b879661af7260fed5140fc?blobs=true>
- FineWeb2 card: <https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/blob/af9c13333eb981300149d5ca60a8e9d659b276b9/README.md>
- FineWeb2 revision API: <https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-2/revision/af9c13333eb981300149d5ca60a8e9d659b276b9?blobs=true>
- ODC-By 1.0: <https://opendatacommons.org/licenses/by/1-0/index.html>
- Common Crawl Terms of Use: <https://commoncrawl.org/terms-of-use>

ODC-By licenses the database and requires attribution; it does not license all
independent rights in every crawled page. Common Crawl content may also carry
source-site terms, copyright, privacy, or personality rights. These limitations
must remain in experiment and release documentation.
