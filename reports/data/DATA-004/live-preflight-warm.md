# DATA-004 bounded data preflight

- Mode: `warm`
- Code commit: `8548c4e73985640d77b009ac8414e9e936dffc75`
- Dirty worktree: `false`
- Config SHA-256: `c70b557b1d07fbb52871d8b43b60c872d7fedd88f34ede2a5f2bf2032840ea33`
- Tokenizer SHA-256: `12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b`

## Stream audit

| Split | Source | Documents | Bytes | Canonical tokens | EOS | Duplicates | Rejected |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | fineweb2_ja_train | 2008 | 7433823 | 1644568 | 2008 | 0 | 27 |
| train | fineweb_en_train | 2088 | 5913121 | 1532968 | 2088 | 0 | 21 |
| validation | fineweb2_ja_validation | 2015 | 8679080 | 1943924 | 2015 | 0 | 200856 |
| validation | fineweb_en_validation | 2081 | 6266452 | 1635723 | 2081 | 0 | 203959 |

## Target accounting

- Packed windows: `32768`
- Trained targets: `262144`
- Accounting reconciled: `true`
- Ratios within tolerance: `true`

| Source | Expected | Realized | Deviation |
| --- | ---: | ---: | ---: |
| fineweb2_ja_train | 0.500000 | 0.500000 | +0.000000 |
| fineweb_en_train | 0.500000 | 0.500000 | +0.000000 |

## Cache and disk

- Cache telemetry: `{"active_leases": 0, "corruptions": 0, "downloaded_bytes": 0, "downloads": 0, "evictions": 0, "free_bytes": 525022482432, "hits": 7, "misses": 0, "retries": 0, "size_bytes": 1269008673, "wait_timeouts": 0}`
- Cache cap within limit: `true`
- Headroom admission passed: `true`
- Reserved OS/checkpoint bytes: `256000000000`
- Largest-shard temporary bytes: `4844733228`

No raw corpus text is retained in either report.
