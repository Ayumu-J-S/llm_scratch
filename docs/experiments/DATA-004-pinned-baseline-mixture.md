# DATA-004 - Pinned Japanese/English Baseline Mixture

- Roadmap ticket: `DATA-004`
- Branch: `codex/data-004-pinned-baseline-mixture`
- Draft PR: pending
- Experiment owner: implementation agent; exact runtime model/reasoning are not exposed
- Status: predeclared; implementation not started
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

## Candidate sources requiring primary-source verification

| Language | Repository / config | Candidate revision | Intended use |
| --- | --- | --- | --- |
| Japanese | `hotchpotch/fineweb-2-edu-japanese` / `sample_10BT` | `180ca004c6a89b590daaad86cb062a07a5353c69` | project train/validation membership from upstream train shards |
| English | `HuggingFaceFW/fineweb-edu` / `sample-10BT` | `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` | project train/validation membership from upstream train shards |

Before committing inventories, re-fetch exact revisions through the official
Hugging Face API, record every selected shard path/size/LFS SHA-256, and retain
dataset cards, ODC-By 1.0, and Common Crawl terms. Exclude upstream test
artifacts. Record classifier/model-annotation provenance as a limitation;
annotations or model outputs must not become training targets.

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

## Current conclusion

No result yet. Source candidates, thresholds, budgets, and stop conditions were
written before implementation or live data access.

