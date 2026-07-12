# STAB-001 - Stable conventional single-GPU training

- PR: [#34](https://github.com/Ayumu-J-S/llm_scratch/pull/34) (draft)
- Branch: `codex/stab-001-stable-training`
- Ticket: `STAB-001`
- Hypothesis: a direct conventional trainer can expose BF16, accumulation,
  clipping, optimizer-step warmup/decay, and numerical-stop behavior through
  Hydra without changing model or data semantics.
- Experiment record: `N/A` — this is training-system correctness/stability
  evidence, not a quality or generalization result.
- Started: 2026-07-12
- Final verdict: implementation complete; independent review pending
- Final record owner: implementation sub-agent `/root/stab001_implementation`

## Scope and decision context

- Goal: establish the conventional single-GPU recipe in `ROADMAP.md` STAB-001.
- In scope: CUDA BF16 autocast, explicit FP32 CPU mode, token-weighted gradient
  accumulation, global-norm clipping, explicit AdamW settings, optimizer-step
  warmup/cosine decay, non-finite guards, effective-token metrics, a bounded
  real-path GB10 smoke, Hydra validation, and invariant tests.
- Out of scope: `torch.compile`, custom kernels, distributed training, exotic
  optimizers, architecture changes, checkpoint/resume payloads, real-data
  pretraining, and throughput/thermal optimization.
- Relevant `PHILOSOPHY.md` principles: DGX Spark is the one-machine boundary;
  a conventional inspectable baseline comes first; failures and limits remain
  visible; Hydra is the runtime authority; a short smoke must not imply a
  long-run performance claim.
- Baseline commit: `f73626ce2fee87d0f4dac839ee1b8ea93af59b2f`.
- Intended evidence: dropout-disabled accumulation equivalence, clipping,
  schedule, BF16-selection, and non-finite fixtures; local full suite; Hydra
  preflight; and 100 finite GB10 BF16 updates. CUDA absence would have been an
  environment blocker, not a passing smoke.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `f73626c`; STAB-001, `PHILOSOPHY.md`, `CHECK.md` 5–7, AGENTS | Requested Luna / Extra High implementation pass | implemented | Added direct BF16/FP32 autocast selection, token-weighted accumulation, norm metrics/clipping, explicit AdamW, optimizer-step warmup/cosine schedule, stability profile, config validation, and focused invariants. | rebased implementation `c390cac`; focused 30 passed; pre-rebase full suite and static checks passed |
| 2 | repair | not exposed by runtime | not exposed by runtime | first GB10 smoke at pre-rebase `afea743` | Use measured gradient norms to prevent clipping from silently defining every update. | implemented | The first 100-update GB10 run was finite but clipping at `1.0` occurred 95/100 updates (median norm 2.180, p95 8.859). Raised the conventional configured ceiling to `10.0`; the clipping test retains an explicit small threshold. | rebased repair `9af0eb2`; focused 23 passed, Ruff/diff/lock passed |
| 3 | repair | not exposed by runtime | not exposed by runtime | GB10/CPU runs at pre-rebase `45f61c0` | Stop bounded stream runs without leaving the non-daemon prefetch producer alive. | implemented | Both 2-update CPU and 100-update GB10 commands wrote metrics but stayed alive after a bounded stop. Added explicit closure of the active single-process DataLoader iterable generator; regression fixture proves budget-stop closure. | rebased repair `a65cac3`; exact-tree parity with `39da132`; CPU and GB10 runs then exited cleanly |
| 4 | handoff | not exposed by runtime | not exposed by runtime | `a65cac3` rebased on `b0f07c0` | Independent review against the ticket, philosophy, and selected CHECK sections. | pending | Rebase changed only inherited documentation relative to the exercised implementation tree; `git diff --exit-code 39da132 a65cac3 -- config src tests` passed. | final local suite/static/config rerun below; independent reviewer pending |

## Runtime provenance block

Requested/default values are separate from the actual runtime display. The
runtime did not expose an exact deployment identifier or reasoning mode.

| Namespace | Product | Displayed family | Exact model identifier | Reasoning mode | Source / unavailable reason |
| --- | --- | --- | --- | --- | --- |
| requested | Codex | Luna | not exposed by runtime | Extra High | explicit implementation request |
| actual | Codex | GPT-5 | not exposed by runtime | not exposed by runtime | exact deployment ID and reasoning mode are not displayed by this runtime |

- Capture command: `uv run python scripts/capture_model_provenance.py --repo . --phase implementation --role implementation --task-path /root/stab001_implementation --requested-model Luna --requested-reasoning-mode 'Extra High' --actual-product Codex --actual-model-family GPT-5`.
- Capture time / branch / commit: 2026-07-12T13:07:47Z /
  `codex/stab-001-stable-training` / `f73626c`.
- Codex CLI version: `codex-cli 0.144.1`.
- Privacy confirmation: no prompts, hidden chain-of-thought, token counts,
  secrets, or raw thread ID recorded.

## Check selection and verdicts

### Review cycle 1

- Review model / mode: pending independent reviewer; requested heavier / Extra
  Thinking. Actual fields will be recorded only if shown by that runtime.
- Commit reviewed: pending exact final documentation head.
- Selected `CHECK.md` sections: 5.1, 5.2, 5.3–5.4, 6.1–6.3, and 7.1–7.4.
- Major sections marked N/A: data/tokenizer/evaluation/checkpoint semantics are
  unchanged; R3/R4 thermal, storage, or sustained-throughput claims require a
  later pilot and are not implied by the 100-step smoke.
- Ticket acceptance result: implementation evidence supports all criteria;
  independent verdict pending.
- Philosophy alignment / complexity / ML-system result: pending independent
  review. The implementation remains direct trainer/config/test work and does
  not add a second execution path.
- Verdict: pending.

## Failed-review handoff

N/A — the independent review has not run. The two implementation-time repairs
above remain retained as observable negative evidence rather than being erased.

## Repair result

- Repair 1: 95% clipping at the initial `1.0` ceiling was an observed
  stability-risk condition. The exact 100-step rerun at `10.0` clipped only
  2/100 updates while retaining finite guards and test coverage.
- Repair 2: bounded stream runs had completed their metrics but not their
  process shutdown. The trainer now closes the active iterable generator in a
  `finally`; both CPU and GB10 confirmation runs exited without leftover
  training processes or Docker containers.

## Final evidence

- Resolved Hydra configuration: `profile=stability_smoke` uses CUDA BF16,
  batch 2, sequence 8, four microbatches/update, 64 configured effective
  targets/update, max grad norm 10, AdamW (`lr=3e-4`, betas `0.9,0.95`,
  `eps=1e-8`, weight decay `0.1`), 10-update warmup and 100-update cosine
  decay to 10% LR. It repeats only the pinned tiny fixture to fill the bounded
  smoke; it is explicitly not a quality claim.
- GB10 final command (source mounted from the exact exercised tree):
  `docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /tmp/llm_scratch-stab001:/tmp/llm_scratch-stab001 -v /home/ayumu/Documents/Proj/llm_scratch/.git:/home/ayumu/Documents/Proj/llm_scratch/.git:ro -w /tmp/llm_scratch-stab001 -e PYTHONPATH=/tmp/llm_scratch-stab001/src -e GIT_CONFIG_COUNT=1 -e GIT_CONFIG_KEY_0=safe.directory -e GIT_CONFIG_VALUE_0=/tmp/llm_scratch-stab001 llm-scratch:env-001 python src/train.py profile=stability_smoke`.
- GB10 environment: NVIDIA GB10 (compute 12.1), driver `580.159.03`, ARM64,
  pinned ENV-001 image `llm-scratch:env-001` (`23a1bee...`), PyTorch
  `2.13.0a0+8145d630e8.nv26.06`, CUDA build/runtime 13.3, BF16 supported.
- GB10 final result: 100 optimizer updates / 6,400 targets, all updates at 64
  effective targets and four microbatches; all training loss and gradient norms
  finite; gradient norm min/median/p95/max `0.846/1.877/8.182/12.914`; clipping
  2/100 updates; first/last train loss `11.1318 -> 0.2491`; LR used
  `3e-5 -> 3.008e-5`, with next LR `3e-5` at the decay floor; finite validation
  loss `8.1168`. The run emitted PyTorch's known memory-efficient-attention
  nondeterminism warning under the repository's `warn_only` policy; it did not
  claim bitwise CUDA determinism.
- CPU real-path confirmation at `a65cac3`: two updates, 64 effective targets
  and four microbatches/update; finite losses `11.1222`, `11.0536`; gradient
  norms `8.8093`, `5.4676`; no clipping at the 10 ceiling; process exited.
- Focused checks: `uv run pytest -q tests/test_trainer.py tests/test_config_profiles.py tests/test_train_streaming.py` — 30 passed.
- Full suite: `uv run pytest -q` — passed before and after the implementation
  work; rerun after the rebase completed successfully.
- Static/reproducibility: `uv run ruff check src tests scripts`, focused
  `ruff format --check`, `git diff --check`, and `uv lock --check` passed.
- Hydra checks: CPU smoke and CPU-resolved stability profile both passed via
  `scripts/config_check.py`; the CUDA profile was exercised above.
- Artifacts: local `metrics.jsonl` and `model_last.pth` were inspected for
  evidence then removed after every smoke. Ignored `runs/` retains local
  Hydra/run-manifest evidence. No W&B run or artifact was created.
- Data/tokenizer/model identity: pinned canonical tokenizer fingerprint
  `12ccbc...f484b`; separate pinned manifest train/validation selections;
  existing 49,535,114-parameter `SimpleDecoderTransformer`; no source,
  tokenizer, target, or architecture semantic change.
- Known trade-offs: the required DataLoader generator close reaches PyTorch's
  retained iterable fetcher because its single-process iterator exposes no
  public close method. It is narrowly scoped, guarded with `getattr`, and has a
  regression test.
- Unresolved risks: 100 steps is a stability smoke, not a 15–60 minute thermal
  pilot or a data/throughput baseline; checkpoints/resume remain CKPT-001.
- Human decision requested: none while independent review is pending.

## Merge authority and final audit

- Merge path: guarded agent self-merge, pending every exact-head gate.
- Human authorization: user instruction on 2026-07-12, “これからはとりあえず
  全部セルフマージしていいよ / とりあえずロードマップ完成させよう”.
- Authorization covers this named PR or bounded ticket/goal series: yes — the
  bounded roadmap implementation series includes STAB-001.
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
- Target branch and base SHA at final audit: `main` / `b0f07c0` before final refresh.
- Up-to-date, conflict-free, and mergeable evidence: pending.
- Record, ledger, PR trail, validation, and risks parity: pending.
- Prohibited self-merge categories: clear for intended scope; final audit still
  required. No secrets, security controls, private data, paid resource,
  destructive action, license question, release, deployment, or account change.
- Admin/bypass/force/disabled-check requirement: must be no.
- Final audit / immediate refresh / merge outcome: pending exact-head review.

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | implementation and repairs | Kept the recipe in direct trainer/Hydra/test surfaces; reproduced GB10 behavior; retained and repaired clipping and shutdown evidence | Initial configuration clipped too frequently; bounded iterable shutdown was not caught before live smoke | ticket scope, CHECK 5–7, exact metrics, GPU process inspection | independent review pending |

## Ledger update

- [x] Updated the in-progress PR/ticket row in `docs/model-runs/README.md`.
- [ ] Update aggregate model counts after independent review/merge outcome.
- [ ] Confirm final PR execution trail parity after review.
- [ ] Record guarded exact-head audit and merge evidence.
- [x] Confirmed that this is not the bootstrap self-merge-policy PR.
