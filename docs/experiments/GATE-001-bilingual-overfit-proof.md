# GATE-001 — Fixed bilingual fixture can be memorized and resumed end to end

- Roadmap ticket: `GATE-001`
- Branch: `codex/gate-001-bilingual-overfit-proof`
- Draft PR: pending — initial provenance commit will open it
- Experiment owner: implementation agent; actual exact runtime model and reasoning are not exposed
- Status: planned
- Started (UTC): 2026-07-12
- Last updated (UTC): 2026-07-12
- Model-run provenance: `docs/model-runs/GATE-001-bilingual-overfit-proof.md`

## Predeclared question and decision rule

- Hypothesis: a small random-initialized decoder can memorize a fixed, versioned Japanese/English fixture within a bounded run, and a verified recovery checkpoint resumes its exact suffix and yields recognizable base-model continuations in both languages.
- Expected result: two independent same-seed executions produce equal optimizer/target-token counters, equal loss traces, equal full-state checkpoint identities, equal greedy samples, and a split/resume trajectory equal to the uninterrupted trajectory.
- Success condition: each full run reaches final token-weighted training NLL at or below `0.20` by optimizer step `200`; a split at step `100` resumes to the same step-`200 loss trace/checkpoint model digest as its uninterrupted counterpart; and greedy checkpoint continuations for the two fixed prompts include the fixture's Japanese and English suffixes. These are memorization checks only, never held-out validation or generalization.
- Failure condition / stop condition: stop and retain the attempt if loss or gradients become non-finite, if the run exceeds 200 optimizer steps or 20 minutes per execution, if the loss threshold is missed, if resume diverges in any compared counter/trace/identity/sample, or if a sampled continuation does not contain the predeclared fixture suffix.
- Relevant baseline commit and run: `origin/main` at `7f9c1728098f5e0dc18653b1660e07e5b36788ce`; no prior GATE-001 run exists.
- Baseline metrics and evidence link: `N/A — this is the first end-to-end memorization gate, not a baseline-quality comparison.`
- Smallest run capable of answering the question: two 200-step fixed-fixture CPU/GPU runs plus one 100+100-step interrupted/resumed trajectory, each using the checkpoint-backed `src/generate.py` implementation.

## Planned budget

| Resource | Limit | Derivation / measurement source |
| --- | --- | --- |
| Elapsed time on target hardware | 20 minutes per execution; stop at 60 minutes total | Bounded R2 gate; not a performance/thermal pilot |
| Training tokens | At most 200 updates per full execution; exact target count recorded | Explicit optimizer-step cap and observed target counters |
| Optimizer steps | 200 full; 100 checkpoint + 100 resume | Predeclared reproducibility comparison |
| Evaluation work and cadence | No held-out evaluation; greedy samples only after final checkpoint | Same-corpus memorization must not masquerade as validation |
| Checkpoint count and bytes | Recovery every 100 updates, final checkpoint per execution | Resume proof and checkpoint-backed generation |
| Local / external / W&B storage | Local run directories only; W&B disabled | Gate does not need an external artifact upload |

## Attempt 1 — pending implementation

No training process has been launched. This section will retain the complete
resolved Hydra configuration, immutable input identities, counters, checkpoint
digests, samples, and comparison result after the canonical command exists.

## Conclusion

- Hypothesis result: pending
- Evidence-backed conclusion: pending implementation and bounded runs
- Uncertainty and limitations: this gate will demonstrate memorization only; it cannot measure held-out validation, generalization, benchmark ability, or production-data readiness.
- Exactly one next step: implement the smallest canonical fixture runner and execute the predeclared comparison.
