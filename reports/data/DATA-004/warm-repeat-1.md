# DATA-004 bounded data preflight

- Mode: `warm`
- Code commit: `fee0f1a231e24957cee86568d9ef89f04eb4e27d`
- Dirty worktree: `false`
- Config SHA-256: `bbd4253c4fc904467aa4d68f38c015126b6bdbcd311ae9b21bc0a1b899220d83`
- Tokenizer SHA-256: `12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b`
- Measurement scope: `loader_only`
- Whole-run elapsed seconds: `56.98054249`
- Observed document-ID overlap: `0`
- Observed normalized-content overlap: `0`

## Stream audit

| Split | Source | Rows read | Documents | Bytes | Canonical tokens | Docs/s | Tokens/s | Duplicates | Rejected | Missing |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | fineweb2_ja_train | 505 | 493 | 1665214 | 368017 | 120.137315 | 89680.67763 | 0 | 8 | 0 |
| train | fineweb_en_train | 537 | 531 | 1545669 | 393197 | 129.397391 | 95816.697061 | 0 | 4 | 0 |
| validation | fineweb2_ja_validation | 52085 | 522 | 1996754 | 450012 | 13.280376 | 11448.905276 | 0 | 51563 | 0 |
| validation | fineweb_en_validation | 48190 | 502 | 1509056 | 390521 | 12.771549 | 9935.374917 | 0 | 47687 | 0 |

## Target accounting

- Packed windows: `8192`
- Trained targets: `65536`
- Accounting reconciled: `true`
- Ratios within tolerance: `true`
- Loader-only target tokens/s: `4997.481529`
- Quota-truncated fragments: `2`
- Quota-removed tokens: `945`

| Source | Expected | Realized | Deviation |
| --- | ---: | ---: | ---: |
| fineweb2_ja_train | 0.500000 | 0.500000 | +0.000000 |
| fineweb_en_train | 0.500000 | 0.500000 | +0.000000 |

## Cache and disk

- Cache telemetry: `{"active_leases": 0, "corruptions": 0, "downloaded_bytes": 0, "downloaded_bytes_per_second": 0.0, "downloads": 0, "evictions": 0, "free_bytes": 523738824704, "hits": 6, "misses": 0, "retries": 0, "size_bytes": 1269008673, "wait_timeouts": 0}`
- Cache cap within limit: `true`
- Headroom admission passed: `true`
- Reserved OS/checkpoint bytes: `256000000000`
- Largest-shard temporary bytes: `4844733228`
- Downloaded bytes/s: `0.0`

## Process resources and reproduction

- Current RSS bytes: `646524928`
- Peak RSS bytes: `723161088`
- Swap bytes: `0`
- Minor page faults/s: `2775.912497`
- Major page faults/s: `0.0`
- Exact process argv: `["scripts/preflight_data.py", "profile=pretrain_streaming", "data.streaming.cache.dir=/tmp/llm-scratch-data004-repair-cold-fee0f1a", "data.streaming.train.max_target_tokens=65536", "+preflight.max_documents_per_split=1024", "+preflight.output_dir=/tmp/data004-repair-evidence-fee0f1a", "+preflight.report_stem=warm-repeat-1", "hydra.run.dir=/tmp/data004-repair-repeat-1-fee0f1a"]`
- The JSON report embeds the complete safe resolved Hydra configuration needed to recompute its config SHA-256.
- Scope boundary: loader-only; no model, GPU, end-to-end consumption, or data-supply-sufficiency claim is made.

No raw corpus text is retained in either report.
