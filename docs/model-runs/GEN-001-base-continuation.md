# GEN-001 - Minimal base-model continuation sampling

- PR: draft — branch pushed; PR URL pending connector creation
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
| 1 | implementation | not exposed by runtime | not exposed by runtime | Baseline `924a77a`; requested Luna / Extra High; GEN-001, PHILOSOPHY, CHECK 7.2/8.2/9.1 | in progress | Opened branch and captured provenance before code changes. Planned reconstruction is from the full-state checkpoint's resolved config and pinned canonical tokenizer only; the CLI will accept no manually supplied architecture values. |

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

- Resolved command/config: pending.
- Data/tokenizer/model identity: pending.
- Validation and measurements: pending.
- Performance/resource result if applicable: N/A — no performance claim.
- Failed attempts retained at: this record.
- Known trade-offs: pending.
- Unresolved risks: candidate not yet implemented or independently reviewed.
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
