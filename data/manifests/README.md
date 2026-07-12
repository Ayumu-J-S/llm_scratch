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
