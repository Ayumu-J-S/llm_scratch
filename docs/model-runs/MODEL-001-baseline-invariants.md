# MODEL-001 - Conventional Baseline Invariants

- PR: [#14](https://github.com/Ayumu-J-S/llm_scratch/pull/14) (merged at
  `5644d4fcc5a7ef5f08520580698a1fd86554f0e6`); docs-only finalization PR [#19](https://github.com/Ayumu-J-S/llm_scratch/pull/19)
- Branch: `codex/model-001-baseline-invariants`
- Ticket: `MODEL-001`
- Hypothesis: a compact invariant suite and bounded overfit proof can make the
  current conventional decoder-only Transformer a trustworthy reference without
  redesigning the architecture.
- Started: 2026-07-11T18:54:00Z
- Final verdict: PASS WITH NOTE
- Final record owner: primary task
- Runtime-visible product/family: `Codex / GPT-5`
- Exact runtime model identifier: `not exposed by runtime`
- Actual reasoning mode: `not exposed by runtime`
- Requested implementation model/reasoning: `Luna / Extra High`
- Requested independent-review model/reasoning: heavier reviewer / `Extra Thinking`

## Scope and decision context

- Goal: protect the current simple architecture before later model changes.
- In scope: output shape, maximum context, causal prefix invariance, finite
  loss/gradients, padding semantics, parameter count, deterministic tiny-batch
  overfit, and applicable R1/R2 measurements.
- Out of scope: RoPE, RMSNorm, SwiGLU, weight tying, new attention mechanisms,
  architecture redesign, trainer redesign, or optimization work.
- Relevant `PHILOSOPHY.md` principles: begin with a conventional, inspectable
  model; make the learning causal chain visible; prove tiny overfit and stable
  behavior before expanding complexity; preserve the smallest coherent change.
- Original baseline commit/run: `a05eb1de5656643757a1c3d98047c98dedea8bfa`.
- Integration baseline: `origin/main` at
  `5644d4fcc5a7ef5f08520580698a1fd86554f0e6`, after the MODEL-001 merge and
  merges of EXP-001,
  DATA-001, TOK-001, DATA-002, POLICY-001, and PROV-001.
- Intended evidence: exact invariant tests, predeclared overfit threshold and
  seed, CPU timing/memory reference, and CUDA forward/backward smoke only when
  the ENV-001 runtime is available.

## Predeclared implementation contract

- The architecture math stays unchanged: conventional MHA, LayerNorm, GELU,
  sinusoidal positions, residual paths, and untied LM head.
- Token inputs must be rank-2, non-empty, `torch.long`, and no longer than the
  configured context. `pad_token_id` must be a non-boolean integer inside the
  vocabulary.
- Padding has one authority: the model derives it from `pad_token_id`; the
  public external-mask override is removed because no repository caller uses it.
  Padded query logits are zero and all-pad rows remain finite using tensor-only
  operations, without `.item()`, host boolean conversion, or CUDA synchronization.
  A zero-valid-target objective remains LOOP-001 scope.
- Tiny test model: vocab 8, width 16, 4 heads, one layer, feed-forward width 32,
  context 6, dropout 0, PAD 0. Independent exact parameter oracle: 2,488.
- Overfit gate fixed before implementation: seed 17, a fixed 4x6 cyclic batch
  over IDs 1-4, AdamW lr 0.02 and weight decay 0, exactly 30 updates, all losses
  finite, final below initial, and final cross-entropy at most 0.02. Planner
  calibration observed 0.002615; the acceptance threshold remains 0.02.
- R1: current canonical FP32 CPU model, batch 4/context 64, one thread, two warmups
  and seven forward/loss/backward samples with every time plus median/p95/max and
  process RSS recorded.
- Current canonical oracle after TOK-001: vocabulary 50,570, PAD 4, width 384,
  six heads/layers, and 49,535,114 trainable parameters. The independent formula
  is `10,646,784 + 50,570 * 769`; the `vocab=512` / `11,040,512` oracle below is
  historical pre-TOK evidence only.
- R2: use the digest-pinned ENV-001 image to run the exact mounted candidate on
  GB10 with BF16. Record a 10-step correctness smoke and a 50-step, post-warmup,
  model-only CUDA-event reference without claiming efficiency or speedup.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | handoff | not exposed by runtime | not exposed by runtime | baseline plus ticket/philosophy/CHECK/model context | Requested Sol / Ultra plan for a bounded implementation and evidence contract | completed | Froze architecture; defined metadata-only input validation, single-authority padding semantics, exact invariant suite, fixed overfit gate, parameter oracle, and R1/R2 evidence | Planner handoff retained in primary task; no repository mutation |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `a186aa6ffa954c38bc88ca891bae34394f494737`, accepted predeclared contract | Requested model: Luna; requested reasoning: Extra High. Implement the smallest invariant suite and required direct model fixes; run CPU validation and R1 | completed | Added metadata-only input validation, constructor PAD validation, inferred padding with an all-pad-safe tensor sentinel, zero padded-query hidden states/logits, and one bounded invariant test module; architecture math is unchanged | 20 focused tests; 59 passed/3 opt-in network skips full suite; fixed overfit CE 3.028204918 -> 0.002615080 in 30 updates; R1 retained below |
| 1 | repair | not exposed by runtime | not exposed by runtime | uncommitted implementation plus primary precommit audit | Requested model: Luna; requested reasoning: Extra High. Complete exact accepted-plan assertions omitted by the first test pass without changing model code | completed | Switched shape evidence to the fixed 4x6 batch; strengthened causal suffix, nonzero-gradient, padded-objective/PAD-gradient, parameter utility, and external-mask-authority assertions | 20 focused tests; 59 passed/3 opt-in network skips; model implementation unchanged |
| 1 | review | not exposed by runtime | not exposed by runtime | `83315a6f10134f5745dc51a738a1e7a93d1b4a2d` | Requested reviewer: heavier reviewer; requested reasoning: Extra Thinking. Independent `/review` against acceptance, philosophy, and applicable CHECK | PASS WITH NOTE | No acceptance or integrity blockers; independently reproduced all invariants, exact overfit trajectory, counts, quality checks, and R1. CUDA R2 remains blocked by ENV-001; per-parameter nonzero-gradient assertion is a non-blocking strengthening note | 20 focused; 59 passed/3 skips; independent R1 median 0.191837724 s and 450,328 KiB RSS |
| 2 | integration planning | not exposed by runtime | not exposed by runtime | original PR plus merged EXP-001/DATA-001/TOK-001/DATA-002 and guarded merge policy | Requested model: Sol; requested reasoning: Ultra. Reconcile MODEL-001 with the canonical tokenizer/manifests, close the gradient and CUDA notes, and preserve the conventional architecture | completed | Required a normal merge, canonical parameter oracle, per-tensor gradient gates, real streamed backward, canonical CPU R1, and exact-candidate GB10 BF16 R2; excluded model/objective/trainer/data redesign | Accepted plan retained in the parent task and implemented here |
| 2 | integration implementation | not exposed by runtime | not exposed by runtime | `d71b00645393461689429039e0a3984f4c624661` plus `origin/main` `8a6f94bcf1c88e65f8c7cda03946ee7d469b9cb6` | Requested model: Luna; requested reasoning: Extra High. Merge normally, preserve ledger union, reconcile canonical semantics, close review notes, and run CPU/GPU evidence | implemented; review pending | Normal merge produced `ff1b24df83555fd86c1098b321eb48f227a1789b`; model source and architecture math stayed unchanged. Tests now require a gradient for every trainable tensor, finite values, and nonzero activity per tensor; the canonical oracle is 49,535,114; the streamed canonical batch now runs backward. | 23 focused and 162 full passed/1 opt-in skip; seed-17 CE 3.028204918 -> 0.002615080; canonical CPU R1 and dirty-candidate GB10 R2 retained below; exact stable-head review remains pending |
| 2 | integration refresh | not exposed by runtime | not exposed by runtime | `bf3fc1f` plus `origin/main` `7da2c03c8adddb7e5e9c02839e5079b7f33584af` | Requested model: Luna; requested reasoning: Extra High. Merge the latest provenance-audited main normally, reconcile the ledger union, and rerun the exact offline quality gates against the resulting head | implemented; review pending | Normal merge produced `db7f541`; MODEL-001 source and architecture math stayed unchanged. The ledger now includes PROV-001 and the roadmap keeps MODEL-001 Ready until its fresh independent review. Ruff format identified pre-existing drift but no unrelated files were changed in this integration refresh. | 29 focused model/stream tests; 166 full passed/1 opt-in skip; `uv lock --check` and Hydra resolution pass; Ruff check passes, while format check reports four pre-existing files; exact stable-head independent review remains pending |
| 3 | independent integration review | not exposed by runtime | not exposed by runtime | normative implementation head `a362ea1cc513581b7fc4e3b7d24ae9bdd527dc21`; PR #14 merged at `5644d4fcc5a7ef5f08520580698a1fd86554f0e6` before this docs-only finalization | Requested reviewer: heavier reviewer; requested reasoning: Extra Thinking. Review the exact normative head against MODEL-001, `PHILOSOPHY.md`, and selected `CHECK.md` sections | PASS WITH NOTE | CPU reviewer independently reproduced the model invariants and seed-17 overfit gate; the runtime-visible product/family was Codex / GPT-5, while exact deployment ID and actual reasoning mode were not exposed. GB10 R2 claims were retained as implementation evidence and were not rerun in this CPU-only review. | 76 focused; 166 full passed/1 opt-in skip; CPU shape/context/causal/finite-gradient/PAD/parameter/overfit checks passed; no actionable findings |

Rows marked `review pending` in earlier integration phases are historical state
snapshots. They are superseded by cycle 3's exact-head PASS WITH NOTE; the
final ledger row and merge-audit section are authoritative for completion.

## Check selection and verdicts

### Review cycle 1

- Review model / mode: not exposed by runtime / not exposed by runtime
- Commit reviewed: `83315a6f10134f5745dc51a738a1e7a93d1b4a2d`
- Selected `CHECK.md` sections: 5.2, 6.1, 6.2, 7, and 11 MODEL-001
- Major sections marked N/A and why: CUDA/BF16 efficiency and long GPU pilots are
  blocked by ENV-001 because this runtime is PyTorch 2.10.0+cpu; data, trainer,
  checkpoint, and W&B sections are unchanged by this focused model ticket.
- Ticket acceptance result: PASS; shape/context, causal prefix, finite loss and
  gradients, clear invalid input/PAD failures, exact counts, padding semantics,
  and the predeclared overfit gate all pass.
- Philosophy alignment: PASS; the change protects the conventional model without
  architecture novelty and preserves visible, bounded evidence.
- Complexity / change-surface result: PASS; one model file, one test module, and
  provenance docs changed; no sibling-ticket behavior was imported.
- ML-system result: PASS WITH NOTE; CPU invariants and R1 pass. CUDA/BF16 R2 is
  honestly deferred to ENV-001, and the committed test asserts collective
  nonzero gradient activity rather than the stronger per-parameter property
  independently observed during review.
- Verdict: PASS WITH NOTE

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |

### Review cycle 2 - integrated head

- Review model / mode: not exposed by runtime / not exposed by runtime
  (requested heavier reviewer / Extra Thinking)
- Commit reviewed: `a362ea1cc513581b7fc4e3b7d24ae9bdd527dc21`
- Selected `CHECK.md` sections: 5.2, 6.1, 6.2, 7, 8.1, and 11 MODEL-001
- Required review: re-run the canonical parameter/PAD oracle, streamed
  forward/backward, exact overfit gate, full offline suite, CPU R1, and GB10 BF16
  smoke/R2 against the exact committed candidate. Review current manifest and
  tokenizer authority and confirm no objective, data, tokenizer, trainer, or
  architecture redesign entered the diff.
- Ticket acceptance result: PASS WITH NOTE — CPU invariants, canonical count,
  PAD/causal behavior, all-parameter gradients, and seed-17 overfit passed; the
  reviewer could not independently rerun GB10 R2 in the CPU-only review runtime.
- Philosophy alignment: PASS — conventional architecture and bounded evidence
  remain intact; no hidden model/runtime claim was added.
- Complexity / change-surface result: PASS — implementation head contains no
  architecture redesign; finalization changes are docs-only.
- ML-system result: PASS WITH NOTE — R1/CPU evidence is independently reproduced;
  R2 remains implementation evidence, not a new performance conclusion.
- Verdict: PASS WITH NOTE

## Failed-review handoff

N/A - the first independent review passed with notes.

## Repair result

- Primary precommit audit found six accepted-plan assertions missing from the
  first test draft. The implementation agent strengthened those assertions
  without changing model code; the independent review then passed.

## Historical pre-integration evidence

- Cycle 1's `vocab=512`, PAD 0, and 11,040,512-parameter result describes only
  the model before TOK-001. Its CPU timing and `PASS WITH NOTE` remain valid
  historical evidence but are not the oracle or verdict for the current branch.
- The historical first standalone R1 command omitted `PYTHONPATH=src` and failed
  with `ModuleNotFoundError: No module named 'models'`; the corrected command and
  all original values remain in Git history.

## Integration evidence - final reviewed record

- Merge and dependency identity: PR #14 merged at
  `5644d4fcc5a7ef5f08520580698a1fd86554f0e6` after a normal, non-rebase merge
  of `origin/main`. The integrated head contains the merged roadmap dependencies,
  including PROV-001; no force push or history rewrite was used. The prior
  integration merge `ff1b24df83555fd86c1098b321eb48f227a1789b` remains the
  historical evidence for the pre-PROV-001 dependency set.
- Resolved Hydra command/config: `uv run python src/train.py --cfg job --resolve`
  passes. It resolves tokenizer fingerprint
  `12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b`,
  memorization-smoke manifest
  `00c3797a7d0eda13950fd699a60c45fcd388829f016479caaeb369438767bd31`,
  and the distinct streaming train/validation selections in bilingual manifest
  `47cca88c4a5595e27eb5d60d99918fb77c30b23f7c0ae98024153f25e14ffc19`.
- Canonical model identity: vocab 50,570, PAD 4, width 384, six heads, six
  layers, dropout 0.1, and 49,535,114 parameters. The independent count is
  `10,646,784 + 50,570 * (384-token-embedding + 384-head-weight + 1-head-bias)`.
  The token embedding and LM head are explicitly untied.
- Architecture/integrity: source still uses conventional
  `nn.MultiheadAttention`, two LayerNorms per block, residual paths, GELU FFN,
  sinusoidal positions, and an untied linear LM head. This integration changes
  tests and provenance, not architecture, objective, manifest, tokenizer,
  trainer, optimizer, or data code.
- CPU validation (prior integration): `uv run --group dev pytest -q` reports 162
  passed and one opt-in test skipped. The focused model plus real streaming
  canonical batch reports 23 passed. Ruff format/check, `uv lock --check`,
  `git diff --check`, Hydra resolution, canonical PAD/model integration, and
  immutable manifest guards pass. The streamed fixture now executes forward,
  CE, and backward and requires every trainable tensor's gradient to be
  present, finite, and contain nonzero activity.
- Refresh validation at `db7f541`: `uv run --group dev pytest -q` reports 166
  passed and one opt-in test skipped; the focused model/stream selection
  reports 29 passed. `uv run ruff format --check .`, `uv run ruff check .`,
  `uv lock --check`, `uv run python src/train.py --cfg job --resolve`, and
  `git diff --check` pass. `uv run ruff format --check .` reports four files
  that would be reformatted (the existing provenance capture/test plus
  pre-existing embedding/trainer formatting drift); these unrelated changes
  were intentionally left out of this integration refresh. The resolved Hydra
  output retains the canonical tokenizer fingerprint, immutable memorization-
  smoke fingerprint, and disjoint bilingual train/validation selections.
- Tiny overfit: seed 17, fixed B4/T6 cyclic batch, AdamW lr 0.02 and weight
  decay 0, exactly 30 updates; CE 3.028204917907715 ->
  0.002615080215036869, below the predeclared 0.02 threshold.
- Canonical CPU R1: Linux 6.17.0-1021-nvidia aarch64, Python 3.11.15,
  PyTorch 2.10.0+cpu, FP32, one Torch thread, B4/T64, 49,535,114 parameters,
  seed 17, two warmups and seven forward/CE/backward samples. Warmups were
  0.549019250 and 0.449708563 seconds. All samples were retained:
  0.414357343, 0.419365020, 0.415640549, 0.423561233, 0.430795222,
  0.429438020, and 0.430727222 seconds; median 0.423561233, nearest-rank p95
  and maximum 0.430795222 seconds, and `ru_maxrss` 1,123,500 KiB. All 75
  trainable parameter tensors had present, finite, nonzero gradients; the
  smallest nonzero count in any tensor was 384. This is a reference only, with
  no speed or efficiency claim.
- GB10 correctness smoke and R2: exact local candidate files were mounted
  read-only over pinned ENV-001 image
  `sha256:25a02a5357d3f22339ddea8de78e2b0725a47dc6bbe15f336fa74242889a648b`,
  based on NGC digest
  `sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1`.
  GB10 CC 12.1, PyTorch `2.13.0a0+8145d630e8.nv26.06`, compiled CUDA 13.3,
  and BF16 were observed. The 10-step B4/T64 candidate smoke produced BF16
  logits, finite losses, and present/finite/nonzero gradients in every
  trainable tensor on every step; PID 1 was visible in `nvidia-smi`.
- R2 measurement: after five warmups, 50 model-only forward/CE/backward steps
  used CUDA events and exactly one host synchronization after the measured
  window. Every step time is retained in the implementation output; median
  8.323584 ms, nearest-rank p95 8.989120 ms, max 9.275392 ms, and aggregate
  30,264.723 target tokens/s for 12,800 targets. This is a target-hardware
  reference, not a comparison or efficiency claim; optimizer, data, logging,
  validation, and checkpoint work were excluded.
- R2 CUDA-event samples in milliseconds, in execution order: 8.086784,
  8.052640, 8.076384, 8.790144, 8.049824, 8.019040, 8.040704, 8.026144,
  8.052960, 8.261760, 8.875168, 8.835232, 8.901856, 8.864416, 8.940896,
  8.989120, 8.796256, 8.688352, 8.954720, 9.046272, 8.275072, 8.745184,
  9.275392, 8.372096, 8.984064, 8.885568, 8.647648, 8.811744, 8.833184,
  8.909984, 8.901920, 8.924352, 8.073216, 8.807584, 8.052864, 8.037664,
  8.047680, 8.030336, 8.026304, 8.696032, 8.043840, 8.030016, 8.057024,
  8.820800, 8.052864, 8.043616, 8.057952, 8.046784, 8.057024, and
  8.038176.
- R2 memory: CUDA allocator peak 692,486,656 bytes allocated and 933,232,640
  bytes reserved; process `VmRSS` 1,754,608 -> 1,755,000 kB; system
  `MemAvailable` 120,184,004 -> 120,185,484 kB; swap remained completely free
  at 16,777,212 kB. Allocator values are not total DGX Spark unified-memory
  pressure, so process, system, and swap observations are reported together.
- Failed GPU evidence attempt retained: the first complete mounted-candidate R2
  reached report generation but failed because the mounted Git worktree's
  `.git` pointer names a host-only administrative path unavailable inside the
  container. The rerun passed the host-observed candidate SHA/image identity as
  explicit environment inputs and completed. No image, cache, or volume was
  pruned.
- GitHub state inspected before handoff: PR #14 is open and draft; there were no
  submitted GitHub reviews and no review threads. No review was dismissed, and
  the PR was not marked ready during that historical implementation phase; it
  later merged at the SHA recorded above.
- Merge authority: the user explicitly authorized self-merge for this bounded
  roadmap goal on 2026-07-12; PR #14 merged at `5644d4f` with the merge actor
  not exposed. This record makes no unobserved actor claim.
- Known trade-off: padded-query work is computed and then zeroed, favoring an
  inspectable invariant over a new attention path. CPU and GPU R1/R2 values are
  bounded single-run references, not conclusions about production training.
- Final review status: the independent review of normative head `a362ea1` is
  PASS WITH NOTE; this docs-only finalization records the merged outcome and
  separates requested values from actual runtime provenance.

## Merge authority and final audit

- PR #14 merged at `5644d4fcc5a7ef5f08520580698a1fd86554f0e6` before this
  docs-only finalization; the merge actor is not exposed by the available
  runtime evidence, so no self-merge claim is made for PR #14.
- User authorization for this bounded roadmap goal was explicit on 2026-07-12:
  “これからはとりあえず全部セルフマージしていいよ”.
- Normative independently reviewed head: `a362ea1cc513581b7fc4e3b7d24ae9bdd527dc21`.
- Finalization descendant: this record-only PR; descendants are accepted only
  when docs/ledger-only and independently re-reviewed. The live PR body records
  the current final head because a commit cannot contain its own SHA.
- Latest independent verdict/model/mode: PASS WITH NOTE / not exposed by runtime /
  not exposed by runtime (requested Extra Thinking).
- Blocking objections and unresolved threads: none observed on PR #14 before
  merge; finalization PR #19 remains subject to its own exact-head review.
- Merge outcome: merged at `5644d4fcc5a7ef5f08520580698a1fd86554f0e6`; actor not
  exposed by runtime evidence.

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | implementation and precommit repair | Kept the change to one model and one invariant test module; satisfied the predeclared causal, numerical, padding, parameter, and overfit checks | The first test pass omitted several exact accepted-plan assertions and the initial standalone R1 probe omitted `PYTHONPATH=src`; both were corrected before the stable implementation commit | Requested model/reasoning Luna / Extra High; frozen parameter oracle, exact overfit gate, primary precommit audit, single padding authority, and explicit architecture exclusions | completed; independent review PASS WITH NOTE |
| not exposed by runtime / not exposed by runtime | independent review | Reproduced all ticket invariants, exact overfit trajectory, parameter identities, quality checks, and an independent R1; checked the hot path for host synchronization | CUDA/BF16 could not be reviewed on the CPU-only runtime; identified a non-blocking stronger per-parameter gradient assertion | Stable commit, predeclared contract, failed-attempt record, and adversarial padding cases | PASS WITH NOTE |
| not exposed by runtime / not exposed by runtime | integration planning | Correctly treated the merged tokenizer, manifests, ENV runtime, and guarded merge policy as current authority; predeclared exact canonical CPU/GPU evidence without redesigning the model | Exact model/mode attribution remains unavailable | Requested model/reasoning Sol / Ultra; original failure note, current main, canonical identities, and CHECK MODEL-001 priorities | completed |
| not exposed by runtime / not exposed by runtime | integration implementation | Preserved the source architecture, reconciled the canonical oracle, closed the durable per-tensor gradient note, added real streamed backward, and completed bounded GB10 evidence | First GPU report attempted `git rev-parse` through a host-only worktree pointer inside the container; rerun used explicit observed identities | Requested model/reasoning Luna / Extra High; accepted integration plan, exact parameter formula, mounted-candidate requirement, and pinned ENV image | implemented; fresh independent review pending |
| not exposed by runtime / not exposed by runtime | integration refresh | Merged provenance-audited main without rewriting history; kept MODEL-001 Ready until the exact integrated head receives an independent review; reran the offline gates | Ruff 0.15.18 exposed pre-existing formatting drift in embedding/trainer files and the merged provenance files; no unrelated formatting churn was added in this refresh | Requested model/reasoning Luna / Extra High; latest `origin/main`, merged ledger, exact-head discipline, and the model-run acceptance contract | implemented; fresh independent review pending |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated per-model attempt, pass, repair, and review counts.
- [x] Finalized the integration-review counts and verdict after the independent
  review of normative head `a362ea1`.
- [x] Confirmed that the PR execution trail matches the stable reviewed head and
  merged outcome.
- [x] Refreshed the execution trail for integrated head `a362ea1`; final review
  is PASS WITH NOTE and PR #14 merged at `5644d4f`.
