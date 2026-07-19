# HUMAN-001 — Blinded milestone continuations preserve a fair qualitative comparison

- Roadmap ticket: `HUMAN-001`
- Branch: `codex/human-001-blinded-eval`
- Draft PR: [#50](https://github.com/Ayumu-J-S/llm_scratch/pull/50)
- Experiment owner: repository evaluation workflow; actual human reviewers pending
- Status: planned — blocked on RUN-001 checkpoints and actual human ratings
- Started (UTC): 2026-07-19
- Last updated (UTC): 2026-07-19
- Review evidence: draft PR body/comments

## Predeclared question and decision rule

- Hypothesis (one falsifiable claim): two reconstructable checkpoints from the
  same RUN-001 pretraining run, separated by at least 25% of the later
  checkpoint's target-token counter, can be compared by humans without model
  identity leakage or evaluation-to-training reuse.
- Expected result: the later checkpoint has higher mean fluency, coherence, and
  naturalness and is preferred more often under a balanced blinded assignment;
  agreement metrics make uncertainty visible rather than hiding disagreement.
- Success condition: the real bundle uses the committed 4-Japanese/4-English
  prompt set and fixed 64-token, temperature-0.8, top-k-40 seeded contract; each
  checkpoint is A twice and B twice per language; at least two distinct humans
  rate all eight items; bundle/mapping integrity passes; every score maps to the
  exact checkpoint after unblinding; no evaluation material enters training.
- Failure condition / stop condition: stop before evaluation if checkpoint
  identity, reconstruction, separation, fixed 64-token context capacity, key
  permissions, namespace isolation, balance, or leakage checks fail. Treat
  incomplete/duplicate reviewer files, bundle/mapping tampering, or fewer than
  two real reviewers as incomplete, not evidence. Do not make a quality claim
  when agreement or effect evidence is insufficient.
- Relevant baseline commit and run: BENCH-001 exact head
  `dcab48eee00eb82a195cc6d2cd9006bb62a8517f`; RUN-001 run/checkpoints pending.
- Baseline metrics and evidence link: pending RUN-001; no human baseline result
  exists yet.
- Smallest run capable of answering the question: eight prompts, two
  checkpoints, one same-seed pair per prompt, and at least two human reviewers.

Do not edit these predeclared conditions after the first real preparation
attempt. Add a dated note and a new attempt if the protocol must change.

## Implementation review constraints added before any real attempt

Formal review requires the first real attempt to prove, before generation, a
complete exact/normalized scan of every checkpoint-owned training manifest for
all HUMAN prompts. A match blocks and retains authenticated identifier/count/
digest evidence without prompt text. The two checkpoints must share a unique
run lineage rather than only a deterministic experiment recipe ID. Private
mapping and result evidence must bind the fixed deterministic CUDA policy, Git
HEAD and dirty-content digest, `uv.lock`, complete resolved HUMAN Hydra config,
and concrete OS/PyTorch/CUDA/container identity. These are implementation
integrity gates, not observed model results or changes to the hypothesis.

## Planned budget

| Resource | Limit | Derivation / measurement source |
| --- | --- | --- |
| Elapsed time on target hardware | bounded generation only; stop after 16 continuations | 8 prompts × 2 checkpoints |
| Training tokens | 0 | Evaluation does not train |
| Optimizer steps | 0 | Evaluation does not optimize |
| Evaluation work and cadence | at most 1,024 generated tokens | 16 continuations × at most 64 new tokens |
| Checkpoint count and bytes | 2 existing RUN-001 checkpoints; 0 new checkpoints | Meaningful milestone pair; loaded sequentially |
| Local / external / W&B storage | compact JSON/index only; no W&B artifact; at least 100 GB remains free | Public bundle, private mapping/reviews/result, reusable prompt-scan index; checkpoints stay local |

## Attempt status — no decision-relevant run yet

No real attempt has launched. Synthetic identity fixtures validate only the
workflow invariants; they are not model outputs, human scores, agreement
evidence, or a quality result. The first real attempt must append the complete
launch identity, immutable checkpoints, resolved Hydra configuration, actual
reviewer count, integrity evidence, and interpretation below without deleting
this blocked state.

## Conclusion

- Hypothesis result: inconclusive
- Evidence-backed conclusion: the implementation can be validated without
  generating or rating fake RUN-001 continuations, but HUMAN-001 has no result
  until its dependency and real human inputs exist.
- Uncertainty and limitations: actual continuation quality, effect magnitude,
  reviewer agreement, and trade-offs are unknown.
- Exactly one next step or next question: after RUN-001 produces a same-run
  checkpoint pair meeting the 25% token-separation rule, generate one blinded
  bundle and send only its public JSON to at least two human reviewers.
