# DATA-004 bounded data preflight

- Mode: `warm`
- Code commit: `fee0f1a231e24957cee86568d9ef89f04eb4e27d`
- Dirty worktree: `false`
- Config SHA-256: `a3211b84b3b35e9e3dee473f18487f89a691f4dcecfe20d20ed8f38dc6c2c133`
- Tokenizer SHA-256: `12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b`
- Measurement scope: `loader_only`
- Whole-run elapsed seconds: `1100.518650554`
- Observed document-ID overlap: `0`
- Observed normalized-content overlap: `0`

## Stream audit

| Split | Source | Rows read | Documents | Bytes | Canonical tokens | Docs/s | Tokens/s | Duplicates | Rejected | Missing |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | fineweb2_ja_train | 2041 | 2008 | 7433823 | 1644568 | 123.607831 | 101235.79873 | 0 | 27 | 0 |
| train | fineweb_en_train | 2111 | 2088 | 5913121 | 1532968 | 128.532446 | 94365.961096 | 0 | 21 | 0 |
| validation | fineweb2_ja_validation | 202872 | 2015 | 8679080 | 1943924 | 11.794657 | 11378.619053 | 0 | 200856 | 0 |
| validation | fineweb_en_validation | 206040 | 2081 | 6266452 | 1635723 | 12.180984 | 9574.586709 | 0 | 203959 | 0 |

## Target accounting

- Packed windows: `524288`
- Trained targets: `4194304`
- Accounting reconciled: `true`
- Ratios within tolerance: `true`
- Loader-only target tokens/s: `4594.705725`
- Quota-truncated fragments: `2`
- Quota-removed tokens: `1898`

| Source | Expected | Realized | Deviation |
| --- | ---: | ---: | ---: |
| fineweb2_ja_train | 0.500000 | 0.500000 | +0.000000 |
| fineweb_en_train | 0.500000 | 0.500000 | +0.000000 |

## Cache and disk

- Cache telemetry: `{"active_leases": 0, "corruptions": 0, "downloaded_bytes": 0, "downloaded_bytes_per_second": 0.0, "downloads": 0, "evictions": 0, "free_bytes": 523737088000, "hits": 7, "misses": 0, "retries": 0, "size_bytes": 1269008673, "wait_timeouts": 0}`
- Cache cap within limit: `true`
- Headroom admission passed: `true`
- Reserved OS/checkpoint bytes: `256000000000`
- Largest-shard temporary bytes: `4844733228`
- Downloaded bytes/s: `0.0`

## Process resources and reproduction

- Current RSS bytes: `810139648`
- Peak RSS bytes: `819585024`
- Swap bytes: `0`
- Minor page faults/s: `151.554905`
- Major page faults/s: `0.0`
- Exact process argv: `["scripts/preflight_data.py", "profile=pretrain_streaming", "data.streaming.cache.dir=/tmp/llm-scratch-data004-repair-cold-fee0f1a", "data.streaming.train.max_target_tokens=4194304", "+preflight.max_documents_per_split=4096", "+preflight.output_dir=/tmp/data004-repair-evidence-fee0f1a", "+preflight.report_stem=representative-r3", "hydra.run.dir=/tmp/data004-repair-r3-fee0f1a"]`
- The JSON report embeds the complete safe resolved Hydra configuration needed to recompute its config SHA-256.
- Scope boundary: loader-only; no model, GPU, end-to-end consumption, or data-supply-sufficiency claim is made.

No raw corpus text is retained in either report.
