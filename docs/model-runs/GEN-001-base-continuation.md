# GEN-001 - Minimal base-model continuation sampling

- PR: [#38](https://github.com/Ayumu-J-S/llm_scratch/pull/38) (draft)
- Branch: `codex/gen-001-base-continuation`
- Ticket: `GEN-001`
- Hypothesis: a compact, checkpoint-owned sampler can reconstruct the canonical
  tokenizer and decoder configuration from a verified full-state checkpoint,
  then expose reproducible base-model continuations without adding chat or
  serving behavior.
- Experiment record: `N/A` — this ticket exposes checkpoint behavior; its
  bounded memorization invocation is correctness evidence, not a model-quality
  or generalization experiment.
- Started: 2026-07-12
- Final verdict: in progress
- Record owner: implementation sub-agent `/root/gen001_implementation`

## Scope and decision context

- Goal: satisfy `ROADMAP.md` ticket `GEN-001`.
- In scope: importable checkpoint sampler, direct CLI, checkpoint-owned model
  and canonical-tokenizer reconstruction, greedy and seeded temperature/top-k
  sampling, EOS/context/`max_new_tokens` enforcement, result metadata, and
  tiny memorization-path evidence.
- Out of scope: chat/SFT templates or behavior, API serving, KV cache,
  batching, quantization, architecture choices, tokenizer changes, and
  generalization claims.
- Relevant `PHILOSOPHY.md` principles: pretraining output is visibly a
  base-model continuation, inference exists only for correctness/sampling/
  evaluation, and the one-machine path should stay direct and inspectable.
- Selected `CHECK.md` sections: 7.2 (generation/evaluation identity), 8.2
  (reproducible CLI execution), and 9.1 (checkpoint identity/reconstruction).
  GPU-performance, data-supply, long-run and W&B sections are N/A because this
  reads an existing local checkpoint and makes no throughput claim.
- Baseline commit: `924a77a12a3ba8c38f50679b8fbefa8ea757363b`.
- Intended evidence: valid full-state checkpoint round-trip; deterministic
  greedy and seeded top-k sampling; EOS and context-limit fixtures; and a
  bounded canonical tiny-memorization run followed by a labeled invocation.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input / requested work | Outcome | Observable findings and evidence |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | implementation | not exposed by runtime | not exposed by runtime | Baseline `924a77a`; requested Luna / Extra High; GEN-001, PHILOSOPHY, CHECK 7.2/8.2/9.1 | implemented; independent exact-head review pending | Added the importable `CheckpointSampler`, `llm-scratch-generate` CLI, verified full-state generation loader, result metadata, and the user-facing generation contract. Reconstruction consumes the checkpoint's `resolved_config`, canonical tokenizer config, matching tokenizer fingerprint, and strict model state only; the CLI has no architecture/tokenizer override. A pre-review hardening pass also rejects an envelope `identity.model_config` that differs from `resolved_config.model`, including same-shape dropout disagreement. |

## Runtime provenance block

Requested/default values are distinct from the runtime display. The active
runtime exposes Codex / GPT-5 but not an exact deployment identifier or
reasoning mode.

| Namespace | Product | Displayed family | Exact model identifier | Reasoning mode | Source / unavailable reason |
| --- | --- | --- | --- | --- | --- |
| requested | Codex | Luna | not exposed by runtime | Extra High | explicit implementation request |
| actual | Codex | GPT-5 | not exposed by runtime | not exposed by runtime | runtime did not display an exact deployment ID or reasoning mode |

- Machine-readable capture: `docs/model-runs/evidence/GEN-001-implementation-provenance.json`.
- Capture command: `uv run python scripts/capture_model_provenance.py --repo . --phase implementation --role implementation --task-path /root/gen001_implementation --requested-model Luna --requested-reasoning-mode 'Extra High' --actual-product Codex --actual-model-family GPT-5 --output docs/model-runs/evidence/GEN-001-implementation-provenance.json`.
- Privacy: the capture contains no prompts, hidden chain-of-thought, token
  counts, secrets, or raw thread identifier.

## Check selection and verdicts

### Review cycle 1

- Review model / mode: pending independent heavier review (requested Extra Thinking).
- Commit reviewed: pending candidate.
- Selected `CHECK.md` sections: 7.2, 8.2, 9.1.
- Major sections marked N/A and why: no model/training/data/runtime mechanism
  changes and no performance, DGX, or W&B claim.
- Ticket acceptance result: pending.
- Philosophy alignment: pending.
- Complexity / change-surface result: pending.
- ML-system result: pending.
- Verdict: pending.

## Failed-review handoff

N/A — no independent review has run.

## Final evidence

- Resolved command/config: `uv run --group dev llm-scratch-generate --checkpoint <full-state-final.pt> --prompt 'Small' --max-new-tokens 7 --json`; the generation command is intentionally direct rather than a second training/runtime config surface.
- Data/tokenizer/model identity: canonical tokenizer fingerprint `12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b`; sampler reconstructs `SimpleDecoderTransformer` strictly from the saved resolved config and rejects a mismatched tokenizer fingerprint or weight shape.
- Validation and measurements: `uv run --group dev pytest -q tests/test_generation.py tests/test_checkpoint.py tests/test_trainer.py tests/test_config_profiles.py` — 43 passed; an earlier independent candidate rerun reported `uv run --group dev pytest -q` — 256 passed, 1 skipped, and will be repeated for the successor head; changed-file Ruff, format, `git diff --check`, and `uv lock --check` passed. The fixtures cover a final-checkpoint round-trip, repeatable greedy decoding, repeatable seeded temperature/top-k sampling, EOS, a truncated context budget, metadata/CLI output, invalid stochastic options, and rejection of a saved-config/identity model mismatch.
- Bounded canonical memorization-path evidence: an explicit CPU-only
  `profile=smoke_overfit` run with the pinned memorization manifest, canonical
  tokenizer, 16-wide/one-layer decoder, batch 2, and `max_steps=1500` writes a
  CKPT-001 full-state `final.pt`; the CLI then samples it with prompt `Small`.
  This is a same-manifest memorization path, not held-out validation or a
  generalization claim. It finished 1,500 optimizer steps / 21,600 target
  tokens in 66.57 CPU seconds; final train loss `0.03937`, same-manifest
  memorization loss `0.07385` (perplexity `1.07665`), and the greedy sample
  stopped at EOS with completion `data.`. The `final.pt` was 20,103,413 bytes,
  wrote in 0.01178 seconds, verified in 0.00261 seconds, and recorded a
  0.01503-second checkpoint pause. These numbers demonstrate the bounded
  fixture path only, not held-out quality or throughput.
- Performance/resource result if applicable: N/A — no performance claim.
- Failed attempts retained at: this record.
- Known trade-offs: `max_new_tokens` is deliberately positive; a zero-token
  request is rejected because it is not a continuation. The sampler is direct
  one-request-at-a-time decoding without a KV cache, batching, or a serving
  layer by ticket scope.
- Unresolved risks: candidate needs independent exact-head review; bounded CPU
  memorization evidence cannot establish model quality, real-data performance,
  GPU behavior, or generation safety.
- Human decision requested: none.

## Merge authority and final audit

- Merge path: guarded agent self-merge, pending independent review and every
  exact-head gate.
- Human authorization: user instruction on 2026-07-12, “これからはとりあえず
  全部セルフマージしていいよ / とりあえずロードマップ完成させよう”.
- Authorization covers this named PR or bounded ticket/goal series: yes — the
  bounded roadmap implementation series includes GEN-001.
- Exact independently reviewed head SHA: pending.
- Latest independent verdict / model / mode: pending.
- All other final audit gates: pending; no final audit may change the reviewed head.

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | implementation | Began with the ticket, checkpoint interface, canonical tokenizer, and provenance boundary rather than adding an inference stack | Candidate implementation pending | exact checkpoint full-state shape and scope exclusions | in progress |

## Ledger update

- [x] Added the in-progress ticket row to `docs/model-runs/README.md`.
- [ ] Update aggregate counts when implementation and review conclude.
- [ ] Confirm PR execution-trail parity after PR creation.
- [ ] Record guarded exact-head audit and merge evidence.
- [x] Confirmed this is not the bootstrap self-merge-policy PR.
