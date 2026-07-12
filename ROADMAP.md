# Roadmap

This document turns `PHILOSOPHY.md` into a dependency-ordered engineering and
research backlog. It is based on the repository state at commit `f31dfb4` on
2026-07-11.

It is not a promise to implement every idea. It is the current best account of
what must be true before this repository can run trustworthy pretraining
experiments on one DGX Spark.

## How to use this roadmap

- One ticket should map to one focused branch and one pull request.
- A ticket's **In scope** and **Out of scope** sections are hard boundaries.
- Do not bundle opportunistic architecture changes into foundation tickets.
- A ticket is complete only when every acceptance criterion is demonstrated.
- Create GitHub issues only for the current Ready wave. Keep later blocked work
  here until its dependencies make the scope concrete; do not flood the issue
  tracker with speculative tickets.
- Negative results and failed attempts are recorded; they are not erased from
  the handoff.
- When evidence invalidates this ordering, update this roadmap before starting
  a materially different direction.
- External technical facts should be checked against current official or primary
  sources when a ticket begins. Do not turn changing service or hardware limits
  into permanent constants.

Ticket states:

- **Ready**: no unmet dependency; an agent may start it.
- **Blocked**: one or more listed dependencies are incomplete.
- **Deferred**: deliberately outside the current pretraining baseline.
- **Done**: merged with acceptance evidence.

Priorities:

- **P0**: required to make training correct, reproducible, or safe.
- **P1**: required before the first credible real pretraining baseline.
- **P2**: valuable after the first baseline exists.
- **P3**: optional optimization or research work, opened only after a measured
  baseline identifies a concrete bottleneck.

The numeric order below is the recommended default execution sequence, not just
a severity ranking. A later ticket may start early when it is in the same wave
and has no unmet dependency, but it must not displace an unfinished ticket on
the critical path. In particular, prepare and validate data before training on
it, prove the complete pipeline on a tiny fixture before a real-data run, and
defer C, CUDA, compilation, and other low-level optimization until after
`RUN-001` establishes the first trustworthy baseline.

## AS-IS snapshot

### Verified health

- `uv lock --check` passes.
- `uv run ruff check .` passes.
- `uv run pytest -q` reports 39 passed and 3 opt-in network integration tests
  skipped.
- A temporary CPU-only smoke run trained a 58,688-parameter fixture for one
  epoch, reduced train/validation loss, and wrote a checkpoint outside the
  repository. The local-text path therefore works as a small memorization path.
- The default configured model is a conventional decoder-only Transformer with
  6 layers, width 384, 6 attention heads, sequence length 64, and about 11.0M
  parameters with the current 512-token vocabulary.
- The stream loader already has useful tests for ratio validation, deterministic
  source selection, pinned Hugging Face dataset revisions, cache behavior,
  prefetch shutdown, tokenization, packing, and the trainer batch contract.

### What the repository is today

| Area | Working AS-IS | Gap relative to `PHILOSOPHY.md` |
| --- | --- | --- |
| Model | Conventional causal attention, embeddings, residual blocks, normalization, and LM head | No model invariant tests and no checkpoint-backed generation |
| Local data | Small end-to-end memorization path | 52 KB English-only corpus; train and validation are the same file |
| Streaming data | Multi-source loader, revision pinning, quotas, cache, and prefetch | Default training sources are empty; causal targets are lost at packed-window boundaries |
| Tokenizer | Custom 512-token character BPE works on its training corpus | Canonical training is fixed to that BPE; unseen Japanese characters raise errors; external tokenizer path is not used by training |
| Trainer | AdamW, optional scheduler, train loss, epoch validation, W&B logging | Epoch-only control; no token/step/time budget, global seed, BF16, accumulation, clipping, or failure guards |
| Checkpoint | Writes `model_last.pth` | Raw model weights only; no atomic save, rotation, compatibility check, or resume |
| Evaluation | Same-corpus loss sanity check | No held-out validation profile, benchmark harness, human-eval workflow, or intermediate-checkpoint evaluator |
| W&B | Resolved config and basic loss/perplexity logging | Unconditional model watch, final artifact upload, no quota preflight, and insufficient run identity/system metrics |
| Operations | Hydra entrypoints, uv lockfile, pytest, Ruff | No CI, run manifest, canonical preflight/resume/eval/status/handoff commands, or PR experiment template |

### Confirmed blockers

1. **The DGX Spark is currently unused by PyTorch.** The host is `aarch64` and
   exposes an NVIDIA GB10, but the environment contains `torch 2.10.0+cpu`,
   `torch.cuda.is_available()` is false, and `src/train.py` silently selects CPU.
   The lockfile's CUDA dependencies are guarded for `x86_64`.
2. **Packed streaming drops causal targets.** For context length `L`, the loader
   consumes `L+1` tokens per window and the collator shifts inside each window.
   The transition from one window to the next is never trained. At the current
   `L=64`, about 1/65 of continuous-stream transitions are lost.
3. **The training tokenizer cannot support the stated data.** The project BPE is
   trained on the English-only local file and raises on unseen characters.
   Although the standalone stream config names an external tokenizer, the real
   training entrypoint overwrites it with the project BPE.
4. **There is no credible train/validation separation.** Local train and
   validation are identical. Streaming train and validation source lists are
   empty; using different source-selection seeds would not create disjoint data.
5. **A long run cannot be resumed.** Checkpoints contain only model weights and
   overwrite one path every epoch.
6. **The current production-looking streaming command is not runnable.** The
   documented `data.mode=streaming` switch fails because `config/train.yaml`
   contains no sources, while the real sample source lives in a separate config.
7. **Storage defaults are unsafe for this machine.** The root filesystem is
   approximately 916 GB with about 525 GB free, while the stream-cache cap is
   configured as 750 GB. A cache policy must preserve operating and checkpoint
   headroom.

### Evidence map

- Device fallback: [`src/train.py`](src/train.py#L17-L18) and the ARM64 Torch
  selection in [`uv.lock`](uv.lock#L3861-L3908).
- Packed-window consumption and shift:
  [`src/data/streaming_dataset.py`](src/data/streaming_dataset.py#L24-L49) and
  [`src/data/stream_loader/loader.py`](src/data/stream_loader/loader.py#L451-L482).
- Canonical BPE override and unknown-character failure:
  [`src/train.py`](src/train.py#L46-L48),
  [`src/train.py`](src/train.py#L101-L106), and
  [`src/tokenizer/bpe.py`](src/tokenizer/bpe.py#L99-L113).
- Train/validation identity and empty stream profiles:
  [`config/train.yaml`](config/train.yaml#L1-L21).
- Epoch coupling and batch-mean validation:
  [`src/training/trainer.py`](src/training/trainer.py#L42-L90) and
  [`src/training/trainer.py`](src/training/trainer.py#L135-L153).
- Model-only overwrite checkpoint and W&B artifact path:
  [`src/training/trainer.py`](src/training/trainer.py#L186-L215).
- Incomplete command surface: [`Makefile`](Makefile#L1-L22) and the mismatched
  streaming instruction in
  [`src/data/stream_loader/README.md`](src/data/stream_loader/README.md#L16-L36).

### Readiness decision

Do not launch a long or real-data pretraining run yet. The local path may be used
for bounded debugging, and stream-loader tests may continue, but a run should not
be called a baseline until the following gates exist:

- CUDA training actually runs on the DGX Spark;
- every intended causal transition is trained exactly once in the packed stream;
- one pinned Japanese-capable tokenizer is used end to end;
- train and held-out validation manifests are disjoint;
- training is controlled by steps/tokens and reports token-weighted metrics;
- checkpoints are atomic and resume the full run state;
- a checkpoint can produce reproducible base-model continuations;
- W&B logging follows a quota-safe policy; and
- the end-to-end bilingual overfit gate passes.

## Backlog overview

| Order | Ticket | Priority | State | Depends on | Outcome |
| ---: | --- | --- | --- | --- | --- |
| 0 | PROV-001 | P0 | Done | — | Observable Codex model provenance for every implementation/review phase |
| 1 | DATA-001 | P0 | Done | — | Correct packed causal transitions |
| 2 | TOK-001 | P0 | Done | — | One selected, pinned tokenizer used end to end |
| 3 | DATA-002 | P0 | Done | — | Immutable manifests and disjoint split contract |
| 4 | MODEL-001 | P0 | Done | — | Protected conventional model invariants |
| 5 | ENV-001 | P0 | Done | — | CUDA-capable, reproducible DGX Spark runtime |
| 6 | EXP-001 | P0 | Done | — | Lightweight experiment and PR handoff contract |
| 7 | CFG-001 | P0 | Done | DATA-001, TOK-001, DATA-002, EXP-001 | Canonical Hydra profiles and commands |
| 8 | REP-001 | P0 | Done | CFG-001, TOK-001, DATA-002 | Reproducible run identity and global seed |
| 9 | LOOP-001 | P0 | Done | DATA-001, CFG-001, REP-001 | Step/token trainer and correct scalar metrics |
| 10 | DATA-003 | P0 | Done | DATA-001, DATA-002, REP-001, LOOP-001 | Deterministic stream horizon, shuffle, and cursor |
| 11 | STAB-001 | P0 | Blocked | ENV-001, LOOP-001 | Stable conventional single-GPU BF16 training recipe |
| 12 | CKPT-001 | P0 | Blocked | DATA-003, LOOP-001, STAB-001 | Atomic rotating full-state resume |
| 13 | CI-001 | P0 | Blocked | CFG-001, MODEL-001 | Network-free CPU quality gate |
| 14 | GEN-001 | P1 | Blocked | MODEL-001, TOK-001, CKPT-001 | Minimal base-model continuation CLI |
| 15 | GATE-001 | P1 | Blocked | ENV-001, MODEL-001, TOK-001, LOOP-001, STAB-001, CKPT-001, GEN-001 | Reproducible bilingual overfit proof |
| 16 | DATA-004 | P1 | Blocked | TOK-001, DATA-002, DATA-003, GATE-001 | Pinned Japanese/English mixture with QA |
| 17 | VAL-001 | P1 | Blocked | DATA-004, LOOP-001, CKPT-001 | Trustworthy lightweight held-out validation |
| 18 | WB-001 | P1 | Blocked | REP-001, LOOP-001, CKPT-001 | Evidence-complete, quota-safe W&B runs |
| 19 | BENCH-001 | P1 | Blocked | GEN-001, VAL-001, WB-001 | Versioned Japanese/general benchmark suite |
| 20 | DGX-001 | P1 | Blocked | STAB-001, GATE-001, DATA-004, WB-001 | Measured model profile and time/token budget |
| 21 | OPS-001 | P1 | Blocked | CI-001, CKPT-001, WB-001, VAL-001, BENCH-001 | Agent-native run and handoff loop |
| 22 | RUN-001 | P1 | Blocked | DATA-004, BENCH-001, DGX-001, OPS-001 | First bounded real pretraining baseline |
| 23 | HUMAN-001 | P2 | Blocked | BENCH-001, RUN-001 | Blinded base-model human evaluation |

### Execution waves

Work through these gates in order. Tickets within a wave may proceed in parallel
only when their explicit dependencies are satisfied.

1. **Inputs and correctness:** `DATA-001`, `TOK-001`, `DATA-002`, and
   `MODEL-001`. Fix token transitions, select the tokenizer, establish immutable
   splits, and protect the reference model before integrating the training path.
2. **Reproducible training foundation:** `ENV-001`, `EXP-001`, `CFG-001`,
   `REP-001`, `LOOP-001`, `DATA-003`, `STAB-001`, `CKPT-001`, and `CI-001`.
   This wave makes the small pipeline runnable, bounded, stable, resumable, and
   continuously checked. `STAB-001` covers required BF16 and gradient-safety
   behavior; it is not a performance-optimization ticket.
3. **Tiny end-to-end proof:** `GEN-001` and `GATE-001`. Do not select or process
   the full baseline corpus until the bilingual fixture can learn, resume, and
   generate correctly.
4. **Real-baseline preparation:** `DATA-004`, `VAL-001`, `WB-001`, `BENCH-001`,
   `DGX-001`, and `OPS-001`. Prepare and QA the pinned real-data mixture before
   any real pretraining; then validate, benchmark, measure, and automate it.
5. **First credible run:** `RUN-001`, followed by `HUMAN-001` when human
   comparison is useful.
6. **Measured optimization:** only after `RUN-001`, consider C/CUDA, compilation,
   custom kernels, data-path tuning, inference optimization, or architecture
   experiments. Open one narrowly scoped ticket for a bottleneck demonstrated by
   the baseline; otherwise do not do this work.

Do not start a real pretraining run before wave 4 is complete. After each ticket
merges, update the states and dependencies in the table before selecting more
work.

## Ticket details

### PROV-001 — Make Codex model provenance visible

- **Goal:** Make the Codex product/model family, exact model identifier, and
  reasoning mode auditable without inferring hidden runtime values.
- **In scope:** A stdlib capture command, a versioned requested-vs-actual JSON
  schema, redaction-safe provenance documentation, model-run template/workflow
  guidance, and focused tests.
- **Out of scope:** Runtime model selection or discovery, prompts, hidden
  chain-of-thought, token counts, secrets, historical record rewrites, or ML
  training behavior.
- **Acceptance criteria:**
  - Requested/config-default values are separate from values explicitly shown by
    the active runtime.
  - Exact model ID and reasoning mode are `not exposed by runtime` with an
    unavailable reason when the runtime does not display them; no family-to-ID
    or marketing-name inference is allowed.
  - Capture records safe UTC/Git/CLI context and never emits prompts, hidden
    chain-of-thought, token counts, secrets, or raw thread IDs.
  - Focused and repository tests plus lint pass, and the PR links the complete
    model-run record and execution trail.
- **Validation:** Focused provenance tests, full repository tests, Ruff, and a
  CLI JSON smoke capture. No training run is required.

### ENV-001 — Make the runtime CUDA-capable on DGX Spark

- **Goal:** A clean, documented setup uses the GB10 GPU instead of silently
  training on the CPU.
- **In scope:** Select the current NVIDIA-supported PyTorch delivery method for
  ARM64/GB10; pin it reproducibly; update dependency/lock or container setup;
  add an environment diagnostic; make device selection explicit in Hydra.
- **Out of scope:** Model changes, `torch.compile`, custom kernels, or performance
  tuning.
- **Acceptance criteria:**
  - A clean setup reports the GB10, CUDA runtime, PyTorch version, and BF16
    capability.
  - A 10-step forward/backward smoke test runs on CUDA.
  - A real training profile fails before loading data when CUDA is requested but
    unavailable; CPU remains an explicit test option.
  - The setup is reproducible from committed instructions and pinned inputs.
- **Validation:** Fresh-environment install, diagnostic command, CUDA smoke, and
  the existing CPU test suite.

### DATA-001 — Fix packed causal-stream semantics

- **Goal:** Train every intended next-token transition exactly once.
- **In scope:** Use stride `L` over `L+1` token windows by carrying the boundary
  token; preserve EOS/document semantics when quotas truncate; correct source
  spans and token accounting.
- **Out of scope:** Multi-node loading, performance tuning, production data
  selection, or shuffle policy.
- **Acceptance criteria:**
  - For token IDs `[2,3,4,5,6,7,8]` and `L=3`, emit windows
    `[2,3,4,5]` and `[5,6,7,8]`.
  - Collation produces `[2,3,4] -> [3,4,5]` and
    `[5,6,7] -> [6,7,8]`.
  - The learned transition multiset is exactly
    `{(2,3),(3,4),(4,5),(5,6),(6,7),(7,8)}`.
  - A truncated fragment is never silently concatenated with a later document or
    source without a defined boundary token.
- **Validation:** Property-style transition test, quota/EOS boundary tests, and a
  tiny forward/backward stream smoke.

### MODEL-001 — Protect the conventional baseline with invariant tests

- **Goal:** Make the current simple architecture a trustworthy reference before
  changing it.
- **In scope:** Tests for output shape, maximum context, causal prefix
  invariance, finite loss/gradients, padding semantics, parameter count, and a
  deterministic tiny-batch overfit.
- **Out of scope:** RoPE, RMSNorm, SwiGLU, weight tying, new attention mechanisms,
  or other architecture redesign.
- **Acceptance criteria:**
  - Changing only future tokens cannot change earlier logits.
  - A supported batch produces finite loss and gradients.
  - Invalid context length and padding usage fail clearly.
  - A fixed tiny batch reaches a predeclared loss threshold.
- **Validation:** CPU unit tests plus the CUDA forward/backward smoke from
  ENV-001 when available.

### EXP-001 — Define the experiment and PR handoff contract

- **Goal:** Let an agent hand a complete experiment to a human in a short review
  session.
- **In scope:** One ticket/branch/hypothesis convention; branch naming; compact
  experiment record; PR template; required hypothesis, expected result, time
  budget, config, commit, manifests, W&B/checkpoint IDs, failed attempts,
  integrity checks, conclusion, and next step.
- **Out of scope:** Merge automation, GitHub bots, or a generic workflow engine.
- **Acceptance criteria:**
  - A new agent can identify the current baseline and next unanswered question.
  - Negative results cannot omit the attempted config and evidence.
  - A sample record represents every consequential-run field required by
    `PHILOSOPHY.md`.
- **Validation:** Dry-run a fixture experiment from branch creation through PR
  handoff.

### TOK-001 — Select, pin, and integrate one canonical tokenizer

- **Goal:** Use one established Japanese-capable tokenizer consistently without
  importing pretrained model capability.
- **In scope:** Compare a small candidate set on a frozen Japanese/English sample;
  record license, immutable revision/files, tokens per character/byte, p50/p95
  sequence length, fallback/round-trip behavior, throughput, and memory; select
  one; integrate it into local, streaming, debug, model, and generation paths;
  remove the project BPE training path and unused tokenizer backends instead of
  retaining compatibility branches.
- **Out of scope:** Loading any pretrained model weights, custom tokenizer
  research, chat templates, or keeping compatibility aliases.
- **Acceptance criteria:**
  - Japanese and English real-stream batches encode and decode successfully.
  - The same pinned artifact produces identical token IDs offline.
  - Token IDs, vocab size, PAD/EOS behavior, and model vocabulary agree.
  - Moving or missing revisions fail before training.
  - An offline integration test is committed and not skipped.
- **Validation:** Frozen-corpus comparison report, round-trip/fallback tests,
  revision/hash mutation tests, and a streamed batch through the model.

### DATA-002 — Add immutable manifests and disjoint split construction

- **Goal:** Make data identity, provenance, and train/validation separation
  auditable.
- **In scope:** Source/revision/config/split/text-ID/license/terms metadata;
  SHA-256 for local and downloaded content; stable document IDs/content hashes;
  deterministic split assignment and fingerprints; a separate explicit
  memorization fixture; benchmark test-data boundary.
- **Out of scope:** Final legal judgment, full semantic deduplication, benchmark
  scoring, or uploading raw datasets to W&B.
- **Acceptance criteria:**
  - Every real source has an immutable identity and recorded usage terms.
  - Mutation or checksum mismatch fails before training.
  - Train and validation have zero overlap in document IDs and normalized
    content hashes.
  - Reordering or prefetch does not change split membership.
  - Same-corpus memorization is possible only through an explicitly named smoke
    profile.
- **Validation:** Manifest mutation, split determinism, overlap, and benchmark
  access-guard tests.

### CFG-001 — Establish canonical Hydra profiles and commands

- **Goal:** Replace ambiguous commands with small, explicit workflows.
- **In scope:** Hydra profiles for `smoke_overfit`, `pretrain_streaming`, and
  later evaluation; compose the real stream config into training; restore a
  durable resolved-config snapshot; add importable/console commands; make
  `README.md` and `Makefile` agree; reject empty or unsafe real profiles.
- **Out of scope:** Benchmark implementation, trainer redesign, or backward
  compatibility shims.
- **Acceptance criteria:**
  - A fresh checkout can compose configs and run the CPU smoke from documented
    commands only.
  - The real profile builds at least one train and one distinct validation batch.
  - Empty sources, identical real train/validation, and unknown critical keys
    fail in preflight.
  - The run directory retains the resolved configuration.
- **Validation:** Hydra composition tests, README command smoke, and local/stream
  fixture tests.

### REP-001 — Add run identity, manifests, and global seeding

- **Goal:** Make a run locally reproducible even when W&B is disabled.
- **In scope:** Seed Python, NumPy, Torch, CUDA, DataLoader, and model
  initialization from Hydra; write experiment ID, Git SHA/dirty state, config and
  lock hashes, hardware/software identity, tokenizer manifest, and data manifest
  to the run directory.
- **Out of scope:** Claiming bitwise determinism for every GPU kernel, raw dataset
  uploads, or W&B policy.
- **Acceptance criteria:**
  - Two CPU fixture runs with the same seed reproduce initial batches and loss
    sequence.
  - The run directory alone identifies code, config, environment, data, and
    tokenizer.
  - Dirty worktrees and manifest mismatches are explicit.
  - Remote tokenizer/dataset inputs without immutable revisions are rejected for
    real runs.
- **Validation:** Determinism, manifest mutation, dirty-tree, and hash tests.

### DATA-003 — Define stream horizon, shuffle, and exact cursor state

- **Goal:** Prevent repeated-prefix epochs and make streaming resume exact.
- **In scope:** A step/target-token horizon; deterministic bounded shuffle or an
  explicitly documented sequential policy; source RNG/cursor/buffer state;
  prefetch-equivalent ordering; repeat accounting.
- **Out of scope:** Multi-node sharding, adaptive data mixtures, or throughput
  optimization.
- **Acceptance criteria:**
  - The same seed and manifest produce the same document/token sequence.
  - An interrupted and resumed stream yields the exact uninterrupted suffix.
  - A new pass does not repeat the same prefix unless `repeat=true` is explicit.
  - Prefetch on/off does not change sample order or membership.
- **Validation:** Sequence fingerprint, interruption/resume, repeat-policy, and
  prefetch-equivalence tests.
- **Post-merge dependency status (2026-07-12):** DATA-001, DATA-002, REP-001,
  and LOOP-001 were all `Done` when DATA-003 merged. PR [#29](https://github.com/Ayumu-J-S/llm_scratch/pull/29)
  merged to `main` as `57266e1e843be2d08e10ef5f387da8466b0c590f` after the
  recorded guarded audit; CKPT-001 and DATA-004 still have other unmet
  prerequisites and remain `Blocked`.

### LOOP-001 — Introduce step/token budgets and correct metrics

- **Goal:** Control training by measurable work rather than ambiguous epochs.
- **In scope:** Authoritative optimizer-step, target-token, and elapsed-time
  counters; `max_steps`/`max_tokens`; independent logging, validation,
  checkpoint, and milestone cadences; token-weighted NLL/perplexity; scheduler
  ordering; empty-loader and non-finite checks.
- **Out of scope:** Benchmark tasks, checkpoint payload/resume, mixed precision,
  or distributed training.
- **Acceptance criteria:**
  - Events fire exactly at configured step/token boundaries.
  - Save cadence can change without changing validation cadence.
  - Partial-batch NLL matches a manual token-level calculation.
  - Local and streaming paths use the same counters and stopping semantics.
  - Counters and local metrics persist when W&B is disabled.
- **Validation:** Boundary/off-by-one tests, local/stream parity, partial-batch
  metric fixture, and max-token stop test.

### CKPT-001 — Build atomic, rotating, full-state checkpoints

- **Goal:** Resume an interrupted run without silently changing the experiment.
- **In scope:** Model, optimizer, scheduler, precision state, counters, RNG,
  stream cursor, resolved config, run/data/tokenizer IDs; temporary write,
  read-back verification, atomic rename; `keep_last_n`; best/final/milestone
  separation; explicit resume command and compatibility checks.
- **Out of scope:** W&B upload policy, model conversion, optimized inference
  formats, or cross-architecture compatibility.
- **Acceptance criteria:**
  - Interrupted-plus-resumed fixture training matches uninterrupted training.
  - A corrupt newest checkpoint falls back to the previous verified recovery
    checkpoint.
  - Rotation never removes best/final checkpoints and never exceeds its stated
    recovery count.
  - Model, tokenizer, data, or config incompatibility fails before training.
  - Older recovery checkpoints are removed only after the replacement verifies.
- **Validation:** Resume equivalence, corruption/fallback, atomic-write failure,
  rotation, and compatibility tests.

### CI-001 — Add the initial CPU quality gate

- **Goal:** Make every PR prove the network-free foundation still works.
- **In scope:** GitHub Actions and a matching local command for frozen uv sync,
  Ruff, pytest, Hydra composition, and an offline tiny smoke; manual/scheduled
  separation for network integration tests.
- **Out of scope:** DGX performance validation, W&B secrets, full training, or
  benchmarks.
- **Acceptance criteria:**
  - Pull requests show lint, unit, config, lock, and offline smoke checks.
  - CI needs no network model/data download and no credentials after dependency
    installation.
  - Lockfile drift fails.
- **Validation:** Local CI-equivalent command and a pull-request workflow run.

### STAB-001 — Establish a stable conventional single-GPU training recipe

- **Goal:** Add only the standard training features needed for a stable DGX Spark
  baseline.
- **Sequencing note:** This is a correctness and stability ticket, not a
  performance-optimization ticket. It remains before the first baseline because
  BF16 behavior, accumulation, clipping, and non-finite guards are part of a safe
  training loop.
- **In scope:** BF16 autocast where supported, FP32 CPU smoke, gradient
  accumulation, global-norm clipping, explicit AdamW parameters, warmup/decay in
  optimizer steps, and non-finite guards.
- **Out of scope:** `torch.compile`, custom C/CUDA, distributed training, exotic
  optimizers, or architecture changes.
- **Acceptance criteria:**
  - Effective tokens per optimizer update are explicit and logged.
  - BF16, clipping, scheduler, and non-finite behavior are configurable and
    tested.
  - A 100-step GB10 smoke completes without NaN/Inf.
- **Validation:** Accumulation equivalence with dropout disabled, scheduler and
  clipping tests, failure injection, and CUDA smoke.

### GEN-001 — Add minimal checkpoint-based continuation sampling

- **Goal:** Let a person observe what pretraining has taught the base model.
- **In scope:** Importable sampler and CLI; model/tokenizer reconstruction from a
  checkpoint; greedy and seeded temperature/top-k generation; EOS, context, and
  `max_new_tokens`; result metadata.
- **Out of scope:** Chat templates, SFT behavior, API serving, KV-cache
  optimization, batching systems, or quantization.
- **Acceptance criteria:**
  - The CLI loads a checkpoint without manually re-entering architecture values.
  - Greedy and seeded sampling are reproducible.
  - EOS and context limits are enforced.
  - Output is labeled as base-model continuation, not chat response.
- **Validation:** Checkpoint round-trip, deterministic generation, EOS/context,
  and tiny-overfit continuation tests.

### WB-001 — Make W&B evidence-complete and Free-plan safe

- **Goal:** Keep useful experiment evidence without treating W&B as bulk storage.
- **In scope:** Configurable scalar cadence; model watching off by default;
  tokens, throughput, elapsed time, peak memory, gradient norm, LR, validation,
  stability, Git/hardware/manifest IDs; `none|best|final|milestone` artifact
  policy; projected-size and authenticated-usage preflight; compact summaries;
  offline/disabled mode.
- **Out of scope:** Raw dataset upload, hard-coded plan quota, W&B as checkpoint
  backup, or automatic mass deletion.
- **Acceptance criteria:**
  - Smoke/CI uploads no artifacts.
  - A model uploads only when policy, reason, projected size, and visible quota
    allow it.
  - Unknown quota or missing login blocks bulk upload but preserves local work.
  - Dataset lineage uses manifests or reference metadata, never raw corpus files.
  - Logged scalar volume respects configured cadence.
- **Validation:** W&B test doubles, offline smoke, artifact policy/size matrix,
  missing-login path, and metric-schema test.

### GATE-001 — Pass the end-to-end tiny bilingual overfit gate

- **Goal:** Demonstrate the complete learning chain before real pretraining.
- **In scope:** A fixed, versioned tiny Japanese/English fixture; one bounded
  random-initialization run; resume; a complete local experiment record;
  optional offline W&B logging; checkpoint-backed continuations.
- **Out of scope:** Generalization claims, production data, benchmark scores, or
  architecture experiments.
- **Acceptance criteria:**
  - The model reaches a predeclared low loss within a bounded update count.
  - Resume preserves the trajectory.
  - Sampling shows recognizable learned Japanese and English continuations.
  - The record explicitly calls this memorization, not held-out validation.
- **Validation:** Run twice from the same seed and compare counters, loss trace,
  checkpoint identity, and samples.

### DATA-004 — Establish the pinned Japanese/English baseline mixture

- **Goal:** Provide a credible, measured corpus for the first real baseline.
- **In scope:** Research and select at least one English source alongside the
  Japanese source; pin revisions and licenses; set token-based mixture ratios;
  populate non-empty train/validation profiles; add compact QA for documents,
  bytes, scripts/languages, tokens, empty text, duplicates, Unicode/control
  characters, length, fallback, truncation, rejected counts, realized ratios,
  cache capacity, and disk headroom.
- **Out of scope:** Adaptive mixtures, model-based quality filtering, raw W&B
  artifacts, or claiming perfect data quality.
- **Acceptance criteria:**
  - A small streamed train and held-out validation run completes.
  - Realized target-token ratios stay within a declared tolerance.
  - Injected empty, duplicate, wrong-script, bad-Unicode, and checksum fixtures
    fail or are counted according to policy.
  - The report includes code/data/tokenizer fingerprints and rejection counts.
  - Cache limits cannot consume checkpoint/OS safety headroom.
- **Validation:** Fixture QA tests and a bounded live-source preflight.

### VAL-001 — Implement trustworthy lightweight held-out validation

- **Goal:** Measure pretraining progress on fixed Japanese and English text
  without conflating memorization with generalization.
- **In scope:** Versioned held-out corpora and fixed token windows; one shared
  scoring implementation for training-time and standalone checkpoint evaluation;
  per-corpus and aggregate token-weighted NLL/perplexity; step/token cadence;
  local JSON plus compact optional W&B summary.
- **Out of scope:** GSM8K or other generative benchmarks, human evaluation, and
  reserved final benchmark tests.
- **Acceptance criteria:**
  - Training-time and standalone scores match for the same checkpoint/fixture.
  - Real train/validation overlap fails preflight.
  - Results include checkpoint, manifest, and evaluated-token identities.
  - Memorization metrics use a distinct name and cannot masquerade as validation.
- **Validation:** Known-logit NLL, scoring parity, cadence, overlap, and checkpoint
  milestone tests.

### BENCH-001 — Add versioned base-model benchmark evaluation

- **Goal:** Track Japanese ability and general reasoning as checkpoints improve.
- **In scope:** Select a deliberately small first suite containing Japanese tasks
  and general/math reasoning such as GSM8K; integrate through a maintained
  harness or narrow adapter; pin task data, prompts, few-shot examples, decoding,
  and scoring; separate routine development subsets from reserved final tests;
  screen exact/normalized contamination; permit same-protocol external baselines
  only as isolated comparison runs.
- **Out of scope:** Broad leaderboard coverage, chat/SFT evaluation, inference
  optimization, LLM-as-judge scoring, or feeding external-model outputs into
  training.
- **Acceptance criteria:**
  - A fixture checkpoint has deterministic generation/scoring tests.
  - Result identity includes checkpoint, tokenizer, task revision, prompt/scorer
    hash, and decoding config.
  - Routine commands cannot access reserved final tests.
  - Injected benchmark contamination is detected and reported by source/doc ID.
  - W&B receives compact result tables, not raw benchmark datasets.
- **Validation:** Golden generation/scoring fixtures, dev/test access guard,
  contamination injection, rerun identity, and external-baseline isolation test.

### DGX-001 — Measure the reference implementation and choose a model profile

- **Goal:** Choose model size, context, batch, and run duration from GB10
  measurements rather than guesses.
- **In scope:** Smoke and candidate Hydra profiles; warmup and repeated timing;
  parameter count, context, micro/effective batch, precision, tokens/s, step time,
  memory, validation/checkpoint overhead; translate throughput into 1-hour,
  24-hour, and proposed full-run token budgets; select one profile with headroom.
- **Out of scope:** Compilation, native extensions, custom kernels, or changing
  the architecture to win a benchmark.
- **Acceptance criteria:**
  - Measurements are repeatable with disclosed conditions and variance.
  - One baseline profile is selected with storage and evaluation headroom.
  - The time/token/checkpoint plan follows directly from measured throughput.
  - A concrete bottleneck is named before any optimization ticket is opened.
- **Validation:** Repeated timed runs and a checkpoint/sample from the selected
  profile.

### OPS-001 — Automate the agent-native run and handoff loop

- **Goal:** Let an agent execute a bounded experiment and prepare human review
  without inventing the workflow each time.
- **In scope:** Canonical `preflight`, `config-check`, `smoke`, `train`, `resume`,
  `eval`, `benchmark`, `status`, and `handoff` commands; checks for Git/config,
  manifests, disk/cache/checkpoint estimate, GPU, W&B login/quota visibility;
  explicit PID/run directory; invalid-run stop and evidence preservation; retry
  linkage; generation of the EXP-001 handoff.
- **Out of scope:** Automatic merge, paid-resource provisioning, machine-wide
  cleanup, or a generic scheduler/orchestration framework.
- **Acceptance criteria:**
  - A fresh agent can run smoke through handoff in offline mode without human
    decomposition.
  - Missing W&B credentials prompt for login while local checks continue.
  - Failed runs retain config, logs, checkpoint status, diagnosis, and retry ID.
  - The handoff validator rejects missing integrity or result fields.
- **Validation:** Offline end-to-end fixture plus missing credential, disk/quota,
  and invalid-loss simulations.

### RUN-001 — Run the first bounded real pretraining baseline

- **Goal:** Produce the first credible random-initialization model and evidence
  for the next experiment.
- **In scope:** The selected conventional architecture, tokenizer, pinned data
  mixture, measured DGX profile, predeclared time/token budget, held-out
  validation, milestone benchmarks/continuations, monitored resume, compact W&B
  record, and best/final artifacts allowed by quota.
- **Out of scope:** SFT, continued pretraining, distillation, synthetic training
  data, architecture experiments, inference optimization, or custom kernels.
- **Acceptance criteria:**
  - Smoke and integrity gates pass before launch.
  - The run has explicit success, stop, and failure conditions.
  - Held-out validation and selected milestone evidence are produced.
  - The best/final checkpoint is reconstructable and the run is resumable.
  - The PR/handoff records failures, comparisons, integrity checks, conclusion,
    and one next hypothesis.
- **Validation:** Reproduce a bounded segment from config/checkpoint and audit
  train/validation separation, benchmark isolation, and artifact policy.

### HUMAN-001 — Add blinded base-model human evaluation

- **Goal:** Use limited human time to compare meaningful milestone
  continuations.
- **In scope:** Versioned Japanese/English continuation prompts, small rubric,
  fixed sampling config, blinded/randomized bundles, score import, agreement, and
  traceable checkpoint IDs after unblinding.
- **Out of scope:** Preference-data collection, SFT/RLHF, automated model judges,
  or evaluating chat behavior.
- **Acceptance criteria:** Model identity is hidden during review; randomization
  is reproducible; scores map back to checkpoints; prompts/outputs/scores never
  enter training data.
- **Validation:** Blinding-leakage and export/import round-trip tests.

## Deferred work

Do not create architecture or low-level performance tickets merely because they
are interesting. `DGX-001` may measure and name a bottleneck, but optimization
work remains deferred until `RUN-001` has produced the first trustworthy
baseline. After that baseline confirms that a bottleneck matters, open one P3
ticket for it with:

- a reference implementation/result;
- a correctness tolerance;
- a benchmark method and measurement conditions;
- the smallest C, CUDA, compilation, or data-path change that tests the idea; and
- an explicit rollback condition.

SFT, chat behavior, inference serving/optimization, C/CUDA or native-extension
work, compilation, multi-node training, and novel model architectures remain
deferred until `RUN-001` produces a trustworthy pretraining baseline.

## Research inputs

These are starting points, not decisions:

- [NVIDIA DGX Spark system and ARM64/unified-memory overview](https://docs.nvidia.com/dgx/dgx-spark/dgx-spark-porting-guide/overview.html)
- [NVIDIA DGX Spark software stack and CUDA guidance](https://docs.nvidia.com/dgx/dgx-spark/dgx-spark-porting-guide/porting/software-requirements.html)
- [NVIDIA NGC workflow for a pinned PyTorch container](https://docs.nvidia.com/dgx/dgx-spark/ngc.html)
- [LLM-jp model card describing a Japanese/English tokenizer lineage](https://huggingface.co/llm-jp/llm-jp-13b-v1.0)
- [LM Evaluation Harness task configuration and reproducibility guidance](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)
- [W&B pricing and storage accounting](https://wandb.ai/site/pricing/)
- [W&B reference artifacts for externally stored data](https://docs.wandb.ai/models/artifacts/track-external-files)
- [W&B artifact deletion behavior](https://docs.wandb.ai/models/artifacts/delete-artifacts)
