# BENCH-001 — versioned base-model benchmark suite

## Predeclared question and conditions

- Question: can fixed checkpoint evaluation track Japanese commonsense and
  general mathematical reasoning without exposing reserved tests or allowing
  benchmark data into training?
- Expected result: deterministic scores and identities from the same fixture
  checkpoint; routine commands read only deterministic development subsets;
  injected contamination blocks scoring and identifies its training document.
- Failure conditions: a Hydra override grants final access; checkpoint,
  tokenizer, prompt, scorer, data, or decoding identity is absent; a complete
  training scan is not proven; raw benchmark/model text reaches local evidence
  or W&B; or an external result can enter the repository checkpoint path.
- Scope: JCommonsenseQA v1.3 and GSM8K only, zero-shot, with a narrow in-repo
  adapter.
- Out of scope: leaderboard breadth, chat/SFT tasks, inference optimization,
  automated judges, and using external outputs for training.

## Implementation and evidence trail

| Cycle | Phase | Outcome | Important evidence |
| ---: | --- | --- | --- |
| 1 | Implementation | Complete | Pinned registry, deterministic development subsets, guarded final entrypoint, checkpoint scoring, complete contamination scan, atomic local JSON, compact W&B table, isolated external aggregate recorder |
| 2 | Focused validation | PASS | Benchmark, generation, and config-profile tests pass; canonical online sources verify to 256 selected development examples and the documented subset hashes |
| 3 | Independent `/review` | Pending | Must cover `PHILOSOPHY.md`, ticket acceptance criteria, and `CHECK.md` 8.2, 8.3, and 9.2 on the exact proposed head |

## Resolved protocol

- Suite: `BENCH-001-suite-v1`
- Development selection: the first 128 examples by SHA-256 rank of canonical
  task/example ID, separately for each task.
- JCommonsenseQA: a fixed Japanese question/options/answer prompt; each choice
  is tokenized at an explicit continuation boundary and scored by conditional
  log probability; length normalization is primary and raw log-probability
  accuracy is retained as a secondary metric.
- GSM8K: fixed `Question`/`Answer` prompt, greedy continuation, 128-token cap,
  and the dataset repository's `####` answer regex.
- Final acknowledgement: `BENCHMARK_FINAL_ACK=BENCH-001-suite-v1`; checked
  outside Hydra.
- Contamination: complete checkpoint-owned train selections, exact/normalized
  whole-document identity, and normalized 48-codepoint shingles.

## Review selection

- `PHILOSOPHY.md`: train/test separation, intermediate checkpoint evaluation,
  fixed development/final boundaries, same-protocol external comparisons, and
  quota-safe W&B evidence.
- `CHECK.md` 8.2: development/final separation, contamination, external output
  exclusion, and complete evaluation identity.
- `CHECK.md` 8.3: no benchmark claim is made from the fixture proof.
- `CHECK.md` 9.2: W&B contains compact aggregates only; no artifact upload or
  raw task content.
- R1 is sufficient because this ticket changes evaluation behavior without
  changing training objective, optimizer, model, data order, or hot-path
  performance.

## Current conclusion

Implementation evidence satisfies the ticket's acceptance tests locally. This
record remains pending until the exact-head independent review and full CI
results are added to the pull request. No benchmark score from the zero-weight
fixture is a model-quality result.
