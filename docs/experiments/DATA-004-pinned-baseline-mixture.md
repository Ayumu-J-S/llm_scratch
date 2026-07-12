# DATA-004 - Pinned Japanese/English Baseline Mixture

- Roadmap ticket: `DATA-004`
- Branch: `codex/data-004-pinned-baseline-mixture`
- Draft PR: [#41](https://github.com/Ayumu-J-S/llm_scratch/pull/41)
- Experiment owner: implementation agent; exact runtime model/reasoning are not exposed
- Status: review `4680931313` technical repairs and rerun evidence complete;
  independent re-review pending; agent self-merge remains rights-policy blocked
- Started (UTC): 2026-07-12T21:43:54Z
- Model-run provenance: `docs/model-runs/DATA-004-pinned-baseline-mixture.md`

## Question and hypothesis

Can two exact-revision, license-recorded Japanese and English web corpora be
streamed through the canonical tokenizer with immutable shard identities,
content-disjoint train/validation membership, a 50/50 loss-target mixture, and
bounded QA/cache behavior suitable for the first real baseline?

Hypothesis: a shard-manifest path plus normalized-content split authority,
target-token debt scheduling, and a bounded Hydra QA preflight will produce a
reproducible bilingual stream whose realized loss-target ratio is within one
percentage point of 50/50 over 262,144 targets, while rejecting or counting all
declared data defects and preserving at least 200,000,000,000 bytes of free
filesystem headroom.

This ticket establishes data supply and QA. It does not claim model quality,
held-out NLL, benchmark ability, or the final model/time budget.

## Source audit and candidate decision

| Language | Repository / config | Audited revision | Decision |
| --- | --- | --- | --- |
| Japanese | `HuggingFaceFW/fineweb-2` / `jpn_Jpan` | `af9c13333eb981300149d5ca60a8e9d659b276b9` | selected: 175 upstream-train shards, 400,138,563 rows, 716,653,211,753 compressed bytes |
| English | `HuggingFaceFW/fineweb` / `sample-10BT` | `9bb295ddab0e05d785b879661af7260fed5140fc` | selected: 15 upstream-train shards, 14,868,862 rows, 30,639,384,917 compressed bytes |

The predeclared educational candidates were rejected before live data access.
`HuggingFaceFW/fineweb-edu` uses a classifier trained from Llama 3 70B outputs,
whose upstream model license documents a restriction on using outputs to
improve another language model. `hotchpotch/fineweb-2-edu-japanese` uses
DeepSeek-generated annotations; current DeepSeek terms permit training on
outputs, but they postdate dataset creation and do not establish the historical
terms. This ticket therefore prefers the direct heuristic/LID/MinHash FineWeb
and FineWeb-2 variants and does not train on model-generated quality labels.

The official revision APIs reproduced those aggregate inventories and expose an
LFS SHA-256 for every selected shard. FineWeb has no upstream test artifact.
FineWeb-2's single Japanese test shard (25,889 rows; 48,225,826 bytes) is
excluded because the card does not define its intended semantics. The smallest
declared live probes are 574,962,194 bytes for English and 329,375,758 bytes for
Japanese; the largest selected shards are 2,148,338,231 and 4,844,733,228 bytes,
respectively. Derive both project splits from upstream train with the shared
normalized-content rule.

Retain the pinned dataset cards, ODC-By 1.0, Common Crawl terms, and filtering
tool licenses alongside the machine-readable manifests. ODC-By covers database
rights, not every underlying page's copyright, privacy obligations, or site
terms; this remains a documented web-corpus limitation rather than a claim of
unrestricted content rights.

The committed immutable identities are:

| Source | Dataset fingerprint | Manifest fingerprint | First bounded artifact |
| --- | --- | --- | --- |
| Japanese | `522b11e5e13d82180e6d47021b66d5262bc1b6fd60a26977444ebd7a8be740db` | `2fc3eb60986c96fcb752b14d740dd5a3f7cea8b52bb5a13cb5834a1f805d6bba` | `data/jpn_Jpan/train/004_00034.parquet`, 329,375,758 bytes |
| English | `928e21caa4b9647c682c65998f2706dd71dedf583f3a91b832ffb3602fa2af6f` | `626a1eb095e9089e5c62ee2df9c058ab7c6dfc54064eca5c13e4d84e65a8d60a` | `sample/10BT/014_00000.parquet`, 574,962,194 bytes |

Artifact order is deterministic by `(size_bytes, path)`, so bounded evidence
starts with the audited small shards while the manifest still commits the full
inventory. The largest selected shard remains 4,844,733,228 bytes and is used
for disk admission.

## Predeclared implementation boundary

- Extend immutable manifests only as needed for full pinned shard inventories;
  keep existing small local manifests operational.
- Stream verified Parquet shards through the project cache without materializing
  the corpus or using an unbounded nested cache.
- Derive train/validation membership from normalized content with one shared,
  versioned salt so cross-source duplicates cannot cross the boundary.
- Schedule sources by loss-contributing target-token debt, not document count.
- Report source tokens, EOS, packed tokens, and loss targets separately.
- Add one Hydra-driven aggregate QA/preflight command; do not implement VAL-001
  scoring, DGX-001 model selection, or OPS-001's full command framework.

## Success conditions

1. Exact source revisions, license/terms, shard paths, byte sizes, and SHA-256
   identities reproduce from primary APIs.
2. Non-empty Japanese and English train/validation profiles compose and a
   bounded streamed train/validation run completes through the canonical path.
3. Train and validation have zero normalized-content/document overlap.
4. Over 262,144 loss targets, each source contributes 50% ± 1 percentage point;
   target accounting reconciles exactly with packed windows and trainer counts.
5. Injected empty, duplicate, wrong-script/language, invalid Unicode/control,
   truncation, fallback, and checksum cases fail or increment declared counters.
6. Cold and warm reports include documents, bytes, scripts/languages, canonical
   tokens, length/tokenization tails, rejection counts, realized ratios, and
   code/data/tokenizer/config fingerprints.
7. Cache cap is at most 200,000,000,000 bytes, runtime reserves at least
   200,000,000,000 free bytes, and admission includes largest-shard temp space.

## Failure and stop conditions

- Stop before parsing on revision, size, checksum, license/terms, or inventory mismatch.
- Stop on any train/validation identity or normalized-content overlap.
- Fail when the 262,144-target ratio exceeds tolerance or accounting does not reconcile.
- Stop live probes on undeclared artifacts, over 2 GB new downloads, or reserve violation.
- Stop training on CPU fallback, non-finite state, exhausted validation supply,
  or cursor/resume mismatch.
- Preserve every failed command/config/report/cause; do not relax gates after results.

## Budget and planned evidence

- Fixture/unit/static validation: up to 20 minutes per cycle.
- Cold plus warm live preflight: up to 45 minutes, one probe shard per source.
- DGX train/validation proof: 50-200 optimizer steps, up to 30 minutes.
- Representative observation: 15-30 minutes if required by CHECK R3, with
  three short repeats where practical.
- Total initial evidence budget: 90 minutes, excluding implementation time.

Evidence includes injected QA fixtures, corrupt/interrupted cache cases,
multi-shard cursor/resume and prefetch fingerprints, cross-source split
disjointness, cold/warm live reports, a small W&B-disabled real stream run, and
CHECK all of 4, 5.3, 5.4, 8.2 plus applicable 3/R2/R3.

## Implementation and validation results

The implementation adds lazy schema-v2 manifests, direct verified Parquet
streaming, artifact/row-group/row cursors, deterministic document policy,
trained-target debt accounting, process-safe bounded cache admission, and one
Hydra-driven aggregate preflight. Existing finite schema-v1 fixtures remain
unchanged. The production profile selects both sources at 0.5 for project train
and validation, with 262,144 and 65,536 trained-target horizons.

Network-free validation after repair head `fee0f1a` passed:

- `uv run pytest -q`: 287 passed, 1 skipped.
- `uv lock --check`, Ruff lint, changed-file format checks, and `git diff --check`.
- metadata-only `profile=pretrain_streaming` config preflight with no shard access.
- injected empty, duplicate, control, bad-Unicode, wrong-script, truncation,
  checksum, cache-floor, lease, exact cursor suffix, and target-accounting tests.

The formal reports are
`reports/data/DATA-004/live-preflight-cold.{json,md}` and
`reports/data/DATA-004/live-preflight-warm.{json,md}`.

The schema-v2 reports at code head `fee0f1a` replace the original evidence:

| Measure | Cold | Warm |
| --- | ---: | ---: |
| Accepted train documents | 4,096 | 4,096 |
| Accepted validation documents | 4,096 | 4,096 |
| Observed document-ID / normalized-content overlap | 0 / 0 | 0 / 0 |
| Japanese / English trained targets | 131,072 / 131,072 | 131,072 / 131,072 |
| Ratio deviation | 0.0 pp | 0.0 pp |
| Cache downloads / bytes | 3 / 1,269,008,673 | 0 / 0 |
| Cache hits / active leases at exit | 4 / 0 | 7 / 0 |
| Whole-run elapsed | 359.483 s | 227.570 s |
| Loader-only packed target rate | 5,492.52/s | 5,613.36/s |
| Peak RSS / swap | 814,043,136 / 0 bytes | 837,894,144 / 0 bytes |
| Quota-truncated fragments / removed tokens | 2 / 895 | 2 / 895 |
| Projected free at full cache plus largest temp | 440,163,898,869 bytes | 440,163,358,197 bytes |
| Required OS/checkpoint reserve | 256,000,000,000 bytes | 256,000,000,000 bytes |

Cold and warm membership, counts, token totals, fallback/rejection counts, and
target accounting are identical. Only measured tokenization latencies and
filesystem observations vary. The reports retain no raw corpus text.

Three comparable warm observations at 1,024 documents per split and 65,536
targets are retained in `reports/data/DATA-004/warm-repeat-{1,2,3}.{json,md}`
and `reports/data/DATA-004/warm-repeat-summary.json`. Whole-run elapsed was
56.981-58.185 seconds (median 57.359; spread 1.205); packed target supply was
4,961.34-5,040.06/s (median 4,997.48; spread 78.72); peak RSS was
695,148,544-723,161,088 bytes; swap and major faults were zero.

The representative R3 loader observation is retained in
`reports/data/DATA-004/representative-r3.{json,md}`. It ran for 1,100.519
seconds (18.34 minutes) with 4,194,304 exact targets, 50/50 attribution,
4,594.71 packed targets/s, peak RSS 819,585,024 bytes, zero swap/major faults/
downloads/retries/leases/missing-data events, and the same three-shard cache.
It is explicitly loader-only and does not claim GPU supply headroom.

The pinned-container R2 target smoke used the real BF16 CUDA/data path for 50
optimizer steps. It produced 3,200 targets in 5.921 training seconds (540.44
targets/s), 93.10 ms median and 95.92 ms p95 step time, finite loss/gradients,
five clipped updates, held-out validation, and verified best/recovery/final
checkpoints. GPU samples ranged 36-41 C, 4.45-19.45 W, 208-2,411 MHz, and
0-68% utilization. These are wiring/stability observations, not a model-quality
or DGX-001 sizing result. The exact digest-pinned command, environment,
identities, metrics, and checkpoint measurements are retained in
`reports/data/DATA-004/r2-summary.{json,md}`, with the complete Hydra config,
run manifest, and metrics in `r2-resolved-config.yaml`,
`r2-run-manifest.json`, and `r2-metrics.jsonl` in the same directory.

Every schema-v2 loader report embeds its exact original/effective argv and the
complete secret-checked resolved Hydra config. Reproduce a committed config
identity with:

```bash
uv run python -c 'import json,sys; from data.preflight import config_fingerprint; p=json.load(open(sys.argv[1])); actual=config_fingerprint(p["reproduction"]["resolved_hydra_config"]); assert actual==p["fingerprints"]["config"]==p["reproduction"]["config_sha256"]; print(actual)' reports/data/DATA-004/live-preflight-cold.json
```

The cold and warm hashes are respectively
`2399ae1bf1fbd1e8d3bac180280787fda10e43c9506d1585cfba45650cb9efc9`
and `f4dd39e18620142e78e9809db0ea0223955727ad519ed1ce53a44fae6ebc39e3`.

## Failed attempts retained

1. The first 4-document/32-target live integration report at
   `/tmp/tmp.VVORwKA74Y/data_preflight.{json,md}` exposed two active cache
   leases after bounded early termination. `ParquetManifestIterator.close()`
   was added, and a network-free regression proves early close releases the
   lease and allows cross-instance eviction. Formal reports show zero leases.
2. The first otherwise-passing warm report marked the tree dirty because the
   cold report was still untracked. It is retained at
   `/tmp/data004-preformal-warm-dirty.{json,md}` and was replaced by the formal
   clean-fingerprint warm run without changing data-path code.
3. The first R2 command used the host uv environment and failed before data
   loading because its PyTorch build has no CUDA. The failure is retained at
   `/tmp/data004-r2-fee0f1a`; the exact profile was then run in the repository's
   digest-pinned `llm-scratch:env-001` image without weakening it to CPU.

## Current conclusion

The technical findings from review `4680931313` are repaired: schema-v2 IDs are
content-bound while v1 fixtures retain their contract; both overlap types fail
and report; production source/tokenizer/row/missing-data and process metrics are
retained; quota truncation is explicit through resume/prefetch accounting; and
each report embeds a secret-checked complete resolved config plus exact argv
whose hash reproduces. Cold/warm, repeated warm, R2, and 18-minute R3 evidence
all pass their scoped gates. Independent re-review is still mandatory. The
reviewer's underlying web-page rights finding remains a separate human policy/
rights disposition gate that prohibits agent self-merge. No model-quality or
perfect-data claim is made.
