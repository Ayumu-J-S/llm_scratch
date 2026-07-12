# STAB-001 - Stable conventional single-GPU training

- PR: draft — branch publishing in progress
- Branch: `codex/stab-001-stable-training`
- Ticket: `STAB-001`
- Hypothesis: A direct, conventional trainer can expose precision, accumulation,
  clipping, learning-rate schedule, and numerical-failure behavior through the
  canonical Hydra configuration without changing the model or data semantics.
- Experiment record: `N/A` — this ticket adds training-system correctness and
  smoke evidence; it does not make a scientific-result claim.
- Started: 2026-07-12
- Final verdict: in progress
- Final record owner: implementation sub-agent `/root/stab001_implementation`

## Scope and decision context

- Goal: establish the stable conventional single-GPU training recipe in
  `ROADMAP.md` STAB-001.
- In scope: BF16 autocast where CUDA supports it, explicit FP32 CPU behavior,
  accumulation, global-norm clipping, explicit AdamW parameters, optimizer-step
  warmup/decay, non-finite guards, effective-token metrics, Hydra validation,
  and focused tests.
- Out of scope: `torch.compile`, custom kernels, distributed training, exotic
  optimizers, architecture changes, checkpoint/resume payloads, real-data
  pretraining, and performance optimization.
- Relevant `PHILOSOPHY.md` principles: one DGX Spark is the real-machine
  boundary; establish a direct conventional baseline; make training behavior
  and failure visible; use Hydra as the runtime authority; do not overclaim
  performance from a short smoke.
- Baseline commit/run: `f73626ce2fee87d0f4dac839ee1b8ea93af59b2f`.
- Intended evidence: CPU accumulation equivalence with dropout disabled,
  clipping/scheduler/non-finite fixtures, resolved Hydra profiles, full local
  test/lint checks, and a 100-step GB10 CUDA smoke if the required runtime is
  available. A missing GPU runtime is recorded as an environment limitation,
  never passed off as a CUDA result.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `f73626c`; STAB-001, `PHILOSOPHY.md`, `CHECK.md` 5–7 and AGENTS | Requested Luna / Extra High implementation pass | in progress | Draft record and branch established before code; examining canonical trainer/config/test paths. | provenance capture below; live draft PR to follow this initial record commit |

## Runtime provenance block

Requested/default values are distinct from actual runtime display. The runtime
does not expose an exact deployment model identifier or reasoning mode.

| Namespace | Product | Displayed family | Exact model identifier | Reasoning mode | Source / unavailable reason |
| --- | --- | --- | --- | --- | --- |
| requested | Codex | Luna | not exposed by runtime | Extra High | explicit implementation request |
| actual | Codex | GPT-5 | not exposed by runtime | not exposed by runtime | exact ID and mode are not exposed by the runtime |

- Capture command: `uv run python scripts/capture_model_provenance.py --repo . --phase implementation --role implementation --task-path /root/stab001_implementation --requested-model Luna --requested-reasoning-mode 'Extra High' --actual-product Codex --actual-model-family GPT-5`
- Capture time: 2026-07-12T13:07:47Z.
- Codex CLI version: `codex-cli 0.144.1`.
- Branch/commit: `codex/stab-001-stable-training` at `f73626c` when captured.
- Phase/role/task path: implementation / implementation /
  `/root/stab001_implementation`.
- Privacy confirmation: no prompts, hidden chain-of-thought, token counts,
  secrets, or raw thread ID recorded.

## Check selection and verdicts

### Review cycle 1

- Review model / mode: pending independent review; actual model ID and mode
  will be recorded only if shown by the review runtime.
- Commit reviewed: pending.
- Selected `CHECK.md` sections: 5.1 (CUDA/runtime gate), 5.3–5.4 (DGX smoke
  boundaries and resource honesty), 6.1–6.3 (objective/accumulation,
  numerical health, cadence), and 7.1–7.4 (direct configuration and scope).
- Major sections marked N/A: data/tokenizer/evaluation/checkpoint semantics do
  not change in this ticket; R2–R4 steady-state or performance claims are not
  made by a 100-step stability smoke.
- Ticket acceptance result: pending.
- Philosophy alignment: pending.
- Complexity / change-surface result: pending.
- ML-system result: pending.
- Verdict: pending.

## Failed-review handoff

N/A — first independent review has not run.

## Repair result

N/A — no repair cycle yet.

## Final evidence

- Resolved Hydra command/config: pending.
- Data/tokenizer/model identity: pending; no data/model change is intended.
- Validation and measurements: pending.
- Performance/resource result if applicable: pending GB10 availability check.
- Failed attempts retained at: this record, if any occur.
- Known trade-offs: pending.
- Unresolved risks: the current host may expose only CPU PyTorch; CUDA evidence
  will remain explicitly blocked if that remains true.
- Human decision requested: none while implementation/review are in progress.

## Merge authority and final audit

- Merge path: guarded agent self-merge, pending all exact-head gates.
- Human authorization: user instruction, 2026-07-12 conversation: “これからは
  とりあえず全部セルフマージしていいよ / とりあえずロードマップ完成させよう”.
- Authorization evidence location: parent task conversation and final PR body.
- Authorization covers this named PR or bounded ticket/goal series: yes — the
  roadmap implementation series, including STAB-001.
- Exact independently reviewed head SHA: pending.
- Latest independent verdict / model / mode: pending.
- All actionable findings repaired and independently re-reviewed: pending.
- Blocking review decision / outstanding `CHANGES_REQUESTED` evidence: pending.
- Newer human objections since authorization/review: none observed.
- Human review dismissed by an agent: no.
- Unresolved review threads at final audit: pending.
- Branch-protection required-context inventory: pending.
- Applicable configured workflow/check inventory: pending.
- Observed exact-head check statuses: pending.
- Expected checks absent, pending, skipped, cancelled, or non-successful: pending.
- No-check evidence when both inventories are empty: pending.
- Target branch and base SHA at final audit: `main` / pending refresh.
- Up-to-date, conflict-free, and mergeable evidence: pending.
- Record, ledger, PR trail, validation, and risks parity: pending.
- Prohibited self-merge categories: pending final audit; no such scope is
  intended.
- Admin/bypass/force/disabled-check requirement: pending; must be no.
- Final audit PR body/comment location: pending.
- Final audit changed reviewed head: no — required.
- Immediate pre-merge re-fetch/compare observation location: pending.
- Immediate refresh compared authorization, head, base, review decision/objections,
  threads, expected checks/statuses, and mergeability: pending.
- Drift found: pending.
- Merge outcome: pending.

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | implementation | pending | pending | STAB-001, `PHILOSOPHY.md`, `CHECK.md`, current canonical trainer/config/tests | in progress |

## Ledger update

- [x] Added an in-progress PR/ticket row to `docs/model-runs/README.md`.
- [ ] Updated per-model attempt, pass, repair, and review counts at completion.
- [ ] Confirmed that the PR execution trail matches this record.
- [ ] Recorded final exact-head guarded self-merge audit evidence.
- [x] Confirmed that this is not the bootstrap self-merge-policy PR.
