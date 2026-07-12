# DATA-004 - Pinned Japanese/English Baseline Mixture

- Roadmap ticket: `DATA-004`
- Branch: `codex/data-004-pinned-baseline-mixture`
- Draft PR: [#41](https://github.com/Ayumu-J-S/llm_scratch/pull/41)
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

## Source audit and candidate decision

| Language | Repository / config | Audited revision | Decision |
| --- | --- | --- | --- |
| Japanese | `HuggingFaceFW/fineweb-2` / `jpn_Jpan` | `af9c13333eb981300149d5ca60a8e9d659b276b9` | selected pending final immutable inventory capture |
| English | `HuggingFaceFW/fineweb` / `sample-10BT` | `9bb295ddab0e05d785b879661af7260fed5140fc` | selected pending final immutable inventory capture |

The predeclared educational candidates were rejected before live data access.
`HuggingFaceFW/fineweb-edu` uses a classifier trained from Llama 3 70B outputs,
whose upstream model license documents a restriction on using outputs to
improve another language model. `hotchpotch/fineweb-2-edu-japanese` uses
DeepSeek-generated annotations; current DeepSeek terms permit training on
outputs, but they postdate dataset creation and do not establish the historical
terms. This ticket therefore prefers the direct heuristic/LID/MinHash FineWeb
and FineWeb-2 variants and does not train on model-generated quality labels.

Before committing manifests or downloading probe shards, re-fetch the selected
exact revisions through the official Hugging Face API, record every selected
train shard path/size/LFS SHA-256, and retain the dataset cards, ODC-By 1.0,
Common Crawl terms, and relevant filtering-tool licenses. Derive both project
splits from upstream train with the shared normalized-content rule.

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

No measurement result yet. The initial source candidates, thresholds, budgets,
and stop conditions were written before implementation or live data access; the
source audit then rejected both educational candidates on provenance/licensing
risk and selected direct non-generative variants, pending final inventory capture.
