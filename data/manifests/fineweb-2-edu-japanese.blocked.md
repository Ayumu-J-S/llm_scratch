# fineweb-2-edu-japanese inventory status

This source is deliberately not runnable from the training configuration in
DATA-002.

- Repository: `hotchpotch/fineweb-2-edu-japanese`
- Exact revision: `180ca004c6a89b590daaad86cb062a07a5353c69`
- Intended config/split: `sample_10BT` / `train`
- Dataset-card license: ODC-By 1.0
- Upstream usage terms: Common Crawl Terms of Use
- Dataset card: <https://huggingface.co/datasets/hotchpotch/fineweb-2-edu-japanese>
- Terms: <https://commoncrawl.org/terms-of-use>

The Hugging Face tree API exposes an LFS SHA-256 OID and byte size per shard,
but the exact paginated inventory of all 139 training shards has not been
committed and checked yet. The one test shard is excluded from the intended
training inventory. A repository commit alone is not treated as a content
checksum, so the real source remains fail-closed until DATA-004 records every
training artifact and wires a bounded source adapter.
