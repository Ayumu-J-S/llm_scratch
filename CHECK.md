# Improvement Ticket Review and Measurement Checklist

This document is a reminder catalog for reviewing work completed under a
`ROADMAP.md` ticket. A ticket is not complete merely because the code runs; the
result must remain healthy as a real training system.

This is not a checklist that must be executed in full for every change. It also
does not require new unit tests or generic software tests for every ticket.
Select only the areas affected by the change and judge them using real data,
the real training path, bounded DGX Spark measurements, logs, and profiles.
Ticket-specific acceptance criteria and validation in `ROADMAP.md` still apply.

## 1. How to use this document

Review work in this order:

1. State the ticket goal, the original problem, and the intended behavior in
   one sentence each.
2. Use the change router to select only the relevant review areas.
3. Run a comparable bounded measurement only when the change can affect
   performance or resource use.
4. Record a `PASS`, `PASS WITH NOTE`, or `FAIL` verdict.
5. Give a short reason when an important area is marked `N/A`.
6. Record findings, the repair handoff, and the re-review result in the pull
   request using `.github/pull_request_template.md`.

Verdicts mean:

- `PASS`: The intended improvement is demonstrated, with no unexplained
  regression in correctness, training health, performance, or operations.
- `PASS WITH NOTE`: The improvement is demonstrated, and a known trade-off or
  follow-up risk is documented with evidence.
- `FAIL`: The change alters the objective unintentionally, starves the GPU,
  fails to use the intended device, risks resource exhaustion, is not
  reproducible, or makes the next experiment unnecessarily difficult.

### Minimum review for every ticket

- [ ] Evidence directly demonstrates the ticket goal; code size or appearance
  is not used as evidence.
- [ ] The objective, target tokens, data boundaries, and evaluation boundaries
  have not changed unintentionally.
- [ ] Validation used the real command and resolved Hydra configuration, not
  only a separate debug path.
- [ ] No implicit fallback disguises failure, especially CUDA to CPU, real data
  to fixtures, or an online run to an untracked run.
- [ ] The ticket does not introduce unrelated mechanisms, config branches,
  compatibility layers, or execution paths without a documented need.

## 2. Change router

Select only the rows that apply. Combine rows for cross-cutting tickets.

| Changed area | Primary review areas | Typical review size |
| --- | --- | --- |
| Documentation, PR template, or record format only | Experiment handoff and changeability | Static review and dry run |
| Hydra config, CLI, or entrypoint | Execution path, reproducibility, implicit fallback, changeability | Config composition and short smoke |
| Dependencies, container, PyTorch, or CUDA | DGX environment, real GPU, precision, reproducible setup | 10-100 CUDA steps |
| Tokenizer | Data supply, compression, vocabulary-driven model cost, offline reproduction | Frozen corpus measurement and real batch |
| Dataset, manifest, split, or filter | Data identity, throughput, long-document tail, cache, split leakage | Loader-only and short end-to-end run |
| Packing, collation, shuffle, or cursor | Target transitions, token accounting, order, resume, throughput | Fixture validation and short end-to-end run |
| Model, attention, embedding, or loss | Mathematical behavior, training health, step time, kernels, memory | CPU smoke and 50-200 DGX steps |
| Optimizer, scheduler, AMP, or accumulation | Update semantics, numerical stability, effective tokens, GPU efficiency | About 100 DGX steps |
| Training loop or logging | Synchronization, step variance, stopping, local evidence, long-run stability | Short profiled DGX run |
| Checkpoint or resume | Pause time, write volume, atomicity, trajectory, disk forecast | Interrupt/resume fixture and real-size estimate |
| Validation, benchmark, or generation | Training isolation, checkpoint identity, evaluation time, reproducibility | Fixed-checkpoint comparison |
| W&B or artifacts | Hot-path overhead, failure isolation, quota, offline behavior | Disabled/offline/online comparison |
| Performance optimization | Objective equivalence, quality tolerance, reproducible speedup, rollback | At least three A/B runs against the reference |
| Production baseline run | All applicable areas in this document | Preflight, bounded pilot, planned run |

### Review sizes

- `R0 - review`: Inspect documentation, configuration, dependency direction,
  and the diff. Make no performance claim.
- `R1 - smoke`: Run 1-10 steps on CPU or a small fixture. Check wiring and
  obvious failures only.
- `R2 - target smoke`: Run 50-200 optimizer steps on DGX Spark with the real
  precision and real data path. Check steady-state performance and short-term
  stability.
- `R3 - pilot`: Run a representative 15-60 minute job. Check thermals, clocks,
  network tails, memory creep, checkpoints, and evaluation overhead.
- `R4 - consequential run`: Run the predeclared time or token budget used for a
  model-quality or research conclusion.

Do not force an R2 review on a documentation-only ticket. Conversely, do not
claim that a data, model, loop, or precision change is fast or GPU-efficient
after only an R1 smoke.

## 3. Shared comparison rules

### Comparison conditions

- [ ] Record the baseline commit or run ID.
- [ ] Keep hardware, OS, driver, CUDA, PyTorch, and precision fixed.
- [ ] Keep model config, tokenizer, sequence length, micro batch, gradient
  accumulation, and target-token count fixed.
- [ ] Keep the data manifest, source ratios, cache state, and network conditions
  fixed.
- [ ] Keep W&B, profiler, model watch, checkpoint, and validation settings fixed.
- [ ] Exclude compile or kernel-autotuning warmup from the measurement window.
- [ ] Use CUDA events or synchronization at benchmark boundaries when measuring
  CUDA time. Do not add a per-step synchronization to the normal hot path.
- [ ] Run short measurements at least three times and report the median and
  spread, not a single best result.
- [ ] Separate cold-cache and warm-cache loader results when relevant; do not
  average them together.

### Minimum metrics to retain

For performance-sensitive tickets, retain the applicable metrics:

- optimizer step-time median, p95, and maximum
- target tokens/s, defined as non-padding next-token targets that contribute to
  loss rather than raw input tokens
- data-wait time or time blocked in `next(loader)`, and its fraction of step time
- forward, backward, optimizer, logging, checkpoint, and validation time
- GPU active/utilization time series, clock, power, and temperature
- CPU use, load, available memory, swap in/out, disk I/O, and network receive
- PyTorch peak allocated/reserved memory and total DGX Spark available memory
- loss, token-weighted NLL, learning rate, gradient norm, and non-finite count
- parameter count, checkpoint size, and full-run disk forecast

### Default investigation triggers

These are default investigation triggers, not universal pass thresholds. A
ticket-specific predeclared budget takes precedence.

- A median tokens/s regression of 5% or more requires investigation and an
  explanation.
- A median tokens/s regression of 10% or more is normally a `FAIL` unless it is
  an intentional trade for demonstrated quality or safety.
- Sustained p95 step time above 1.5 times the median requires decomposition into
  data, logging, checkpoint, thermal, and network tails.
- Steady-state data wait above 5% of step time requires a loader-headroom check.
- Steady-state data wait above 10%, or recurring GPU timeline gaps, blocks a
  long run until the cause is understood.
- Loader-only target-token supply below 1.2 times end-to-end consumption means
  the data path has insufficient headroom.
- Monotonic memory growth after warmup or sustained swap in/out blocks a long run.
- NaN/Inf, token-count mismatch, split overlap, checksum mismatch, or an
  unintended resume-order difference is a zero-tolerance `FAIL`.

Do not use a single average GPU-utilization value as the pass criterion. A
short kernel or memcpy can count as active, while a small model may be unable to
saturate the GPU. Interpret utilization together with tokens/s, the GPU
timeline, kernels, data wait, and clocks.

## 4. Data supply and tokenizer

### 4.1 Is data starving the GPU?

- [ ] Measure `model-only`, `loader-only`, and `end-to-end` paths separately.
- [ ] For `model-only`, reuse a device-resident synthetic batch with the same
  shape to estimate the model-side ceiling.
- [ ] For `loader-only`, consume a fixed target-token count through the real
  source, tokenizer, and packing path.
- [ ] For `end-to-end`, measure data wait and GPU idle gaps on the real path.
- [ ] If loader-only is slow, separate source read, network, JSON decode,
  tokenization, packing, IPC, collation, and host-to-device transfer.
- [ ] Exercise documents near the p50, p95, and p99 lengths so one long document
  cannot silently stall the producer.
- [ ] Observe when the prefetch queue is empty and full. A persistently empty
  queue indicates producer starvation; a persistently full queue means a larger
  buffer will not help.
- [ ] Compare prefetch on and off with the same order and token count, including
  CPU and memory cost.
- [ ] Separate startup, first batch, steady-state batches, source switches, and
  reconnection latency.
- [ ] Check tolerable stall time for validation, post-resume startup, and cache misses.

### 4.2 Tokenization

- [ ] Count fallbacks and failures on Japanese, English, mixed-language text,
  ASCII symbols, emoji, and malformed-Unicode candidates.
- [ ] Compare tokens/character, tokens/UTF-8 byte, and sequence-length p50/p95/p99
  by language.
- [ ] Measure documents/s, characters/s, tokens/s, per-document p95/p99 latency,
  and peak RSS.
- [ ] Tokenizer throughput comfortably exceeds required training target tokens/s.
- [ ] If whole documents are tokenized at once, measure latency and temporary
  memory on the largest expected document.
- [ ] Artifact, revision, special-token IDs, and vocabulary size reproduce
  identically offline.
- [ ] Record round-trip behavior and the exact PAD/EOS/BOS semantics used by loss.
- [ ] Import no pretrained weights or chat template alongside an external tokenizer.
- [ ] Calculate how a vocabulary change affects embeddings, the LM head,
  optimizer state, checkpoint size, and step time.
- [ ] Compare characters/s or bytes/s as well as tokens/s so improved compression
  is weighed against a larger output layer.

### 4.3 Packing, boundaries, and token accounting

- [ ] Intended next-token transitions in the continuous stream are neither
  dropped nor duplicated.
- [ ] Document boundaries, source boundaries, and quota-truncated fragments have
  an explicit EOS or boundary policy.
- [ ] Emitted tokens, input tokens, target tokens, dropped remainders, and padding
  are counted separately.
- [ ] `max_tokens` unambiguously means tokenizer output, target tokens, or an
  optimizer budget.
- [ ] Realized source ratios use the declared token basis rather than document count.
- [ ] Partial batches and short final windows do not distort metrics.
- [ ] Buffer operations do not become quadratic in long-document size.
- [ ] Metadata and source spans remain correct, with acceptable overhead when enabled.

### 4.4 Sources, cache, and network

- [ ] Train and validation have zero overlap in document IDs and normalized
  content hashes.
- [ ] Source, revision, config, split, license/terms, and checksum are fixed
  before the run.
- [ ] Reordering sources or changing prefetch does not alter split membership.
- [ ] Remote timeout, retry, and backoff cannot hang forever or silently skip data.
- [ ] Record read throughput, retries, rejections, and missing-data rate per source.
- [ ] Observe cache hits, misses, eviction, interrupted downloads, corruption,
  and full-cache behavior.
- [ ] Cache limit, checkpoint forecast, logs, temporary files, and OS headroom
  fit within available disk.
- [ ] If a cache key uses only the URL, guarantee immutable contents or verify a
  content checksum.
- [ ] A remote-source failure preserves existing checkpoints and manifests and
  records the reason.

## 5. GPU, DGX Spark, and system resources

### 5.1 Real GPU path

- [ ] Hydra explicitly selects the device, and a requested unavailable CUDA
  device fails before data loading.
- [ ] The run records GPU name, compute capability, driver, CUDA runtime,
  PyTorch build, and BF16 capability.
- [ ] The process appears as a compute process in `nvidia-smi` or an equivalent monitor.
- [ ] Model parameters, inputs, labels, and loss are on the expected device, with
  no major computation falling back to CPU.
- [ ] Representative forward, backward, and optimizer steps contain CUDA kernels.
- [ ] CPU is allowed only through an explicit smoke profile; a real profile
  never silently falls back.
- [ ] The environment uses aarch64 packages or containers, not x86_64-only
  wheels or emulation.

### 5.2 Is the GPU being used effectively?

- [ ] Measure a continuous post-warmup window and exclude initialization.
- [ ] Match GPU idle gaps to CPU thread state, data wait, and logging at the same time.
- [ ] Attribute low utilization to small batches, short sequences, CPU
  synchronization, data starvation, excessive launches, or memory limits.
- [ ] Measure several micro-batch sizes up to the memory ceiling and inspect the
  tokens/s scaling curve.
- [ ] Verify that attention cost and memory change as expected with sequence length.
- [ ] Compare BF16 and FP32 at equal work, including speed, memory, loss,
  non-finites, and actual kernel dtype.
- [ ] When Tensor Cores are expected, confirm that shape, dtype, and kernels match.
- [ ] Ensure `.item()`, frequent synchronization, blocking copies, and per-step
  logging or artifacts do not stop the GPU queue.
- [ ] Do not mix profiler overhead into normal-run performance numbers.

### 5.3 DGX Spark-specific review

DGX Spark uses Grace Blackwell, ARM64, and 128 GB of unified CPU/GPU memory.
Do not reason about it like a discrete-VRAM system.

- [ ] Observe total available memory, process RSS, the PyTorch allocator, page
  faults, and swap together with GPU memory.
- [ ] Do not infer spare capacity from unsupported or unexpectedly small
  `nvidia-smi` Memory-Usage output.
- [ ] Swap does not grow continuously after warmup. If it does, correlate it
  with step-time tails and UMA page migration.
- [ ] Set headroom assuming CPU tokenization, filesystem cache, GPU tensors, and
  optimizer state compete for the same 128 GB.
- [ ] Measure worst-case overlap among normal training, checkpoint saving,
  validation, and prefetch buffers.
- [ ] Use the supplied 240 W power supply and inspect power/thermal capping,
  clock drops, and temperature during a pilot.
- [ ] Use a 15-60 minute pilot to expose throttling and clock drift that the
  first 100 steps cannot show.
- [ ] Keep desktop, browser, Jupyter, and other CUDA-process load comparable
  across runs.
- [ ] Check contention among dataset, cache, checkpoints, and NVMe I/O on the
  root filesystem.
- [ ] Re-measure the same baseline after OS, driver, or firmware updates and
  include environment identity in the run record.

### 5.4 Memory and storage verdict

- [ ] Estimate parameters, gradients, optimizer state, activations, batches,
  prefetch, tokenizer memory, and filesystem cache separately.
- [ ] Do not treat `torch.cuda.max_memory_allocated()` as total system memory use.
- [ ] The post-warmup peak stabilizes rather than growing with step count.
- [ ] Do not silently reduce the batch after OOM; retain the failed configuration.
- [ ] Include one checkpoint's real size, rotation count, atomic-write temporary
  copy, and best/final/milestone copies.
- [ ] Forecast full-run cache, checkpoints, logs, and profiles with safety headroom.
- [ ] Reject cache settings that consume OS or checkpoint headroom.

## 6. Training loop and numerical health

### 6.1 Is the system training the same objective?

- [ ] `inputs`, `labels`, masks, ignored padding, and EOS semantics are correct
  before and after the change.
- [ ] Loss reduction and token weighting are explicit and remain correct on
  partial batches.
- [ ] Gradient-accumulation loss scaling and optimizer-step semantics are correct.
- [ ] Effective target tokens per update match configuration and observation.
- [ ] The scheduler advances on the intended optimizer-step or token unit, not
  a micro step.
- [ ] Skipped updates, overflow, and clipping do not desynchronize counters and
  scheduler state.
- [ ] Dropout, train/eval mode, and seeds remain correct after validation and resume.

### 6.2 Numerical stability

- [ ] Loss, perplexity, gradient norm, parameter/update norm, and learning rate
  are visible at a useful cadence.
- [ ] A NaN/Inf records the step, batch identity, and preceding checkpoint and
  then stops safely.
- [ ] Initial loss is plausible for the vocabulary size and data distribution.
- [ ] Falling loss does not hide zero gradients, explosions, or one anomalous layer.
- [ ] Clipping is not active so often that it silently defines every update.
- [ ] BF16 follows the short FP32 loss trajectory within a declared tolerance,
  with necessary reductions and state kept in FP32.
- [ ] `exp(loss)` overflow does not break monitoring.
- [ ] Raw loss from different tokenizers, data, or models is not treated as
  automatically comparable.

### 6.3 Step breakdown and synchronization

- [ ] Separate time in `next(loader)`, host/device preparation, forward, loss,
  backward, clipping, optimizer, scheduler, and metrics.
- [ ] Understand device synchronization caused by `loss.item()` and gradient-norm reads.
- [ ] A/B test a lower W&B and console-log cadence to measure hot-path overhead.
- [ ] tqdm, JSON writes, manifest hashing, and checkpoint verification do not
  enter every steady-state step.
- [ ] Validation and checkpoint cadence are independent and each pause is budgeted.
- [ ] Event off-by-one errors do not add evaluation, saves, or target-token overshoot.

### 6.4 What to inspect in a 15-60 minute pilot

- [ ] Tokens/s, step time, clock, temperature, and memory reach a steady state.
- [ ] Loss and gradient norms remain finite, and spikes are explainable.
- [ ] Source ratios and rejection rates do not drift materially over time.
- [ ] Network retries, queue starvation, and disk I/O wait are not periodic.
- [ ] Throughput returns to baseline after checkpointing and validation.
- [ ] ETA agrees with measured throughput plus validation and checkpoint overhead.
- [ ] Stopping preserves evidence needed to resume or diagnose the run.

## 7. Model changeability and software health

This section does not reward abstraction count. It asks whether the next
experiment can remain small and safe.

### 7.1 Change surface

- [ ] A model change does not require rewriting the data loader, checkpoint
  policy, or W&B integration.
- [ ] A tokenizer change does not require synchronizing several independent
  tokenizer-selection points.
- [ ] A source change does not require new trainer or model branches.
- [ ] Scientifically meaningful optimizer, scheduler, and precision choices live
  in Hydra rather than source constants.
- [ ] Only scientifically meaningful choices are configurable; implementation
  details are not exposed indiscriminately.
- [ ] Local/streaming and train/eval/generate paths do not duplicate the same responsibility.
- [ ] The entrypoint assembles components rather than mixing data, model,
  training, evaluation, and checkpoint responsibilities.

### 7.2 Dependency direction and replaceability

- [ ] The trainer depends on an explicit forward/loss contract, not the internals
  of one model class.
- [ ] The model imports no W&B, Hydra, dataset, or filesystem code.
- [ ] The data layer imports no model or optimizer code.
- [ ] Checkpoints store reconstruction identity without pickling unnecessary
  runtime objects.
- [ ] A missing optional integration does not break an unused offline or CPU path.
- [ ] Performance-specific code stays behind a small boundary and remains
  comparable with a reference behavior.

### 7.3 Repository policy

- [ ] Runtime and training configuration stays in Hydra; no separate `config.py`
  is added.
- [ ] Direct imports are used without unnecessary re-exports or service locators.
- [ ] No unused compatibility alias, deprecated path, or shim is retained.
- [ ] A focused ticket does not bundle unrelated architecture changes.
- [ ] Canonical behavior does not live only in a notebook.
- [ ] No custom kernel, compile path, or distribution layer is added without a
  measured bottleneck.

### 7.4 Next-change thought experiment

Answer these questions without adding code:

- Which files change when replacing LayerNorm with another normalization?
- Can positional representation change without touching trainer or data code?
- Can one tokenizer identity keep train, stream, generation, and model vocabulary aligned?
- Can a dataset mixture change through Hydra and manifests alone?
- Can a one-hour run become a 24-hour run by changing token/save/eval budgets
  without source edits?
- Does resume reject a mismatched config before doing work?

If answers spread across many unrelated files or manual steps, inspect whether
the ticket introduced new coupling. Do not build a framework or plugin system
solely for hypothetical future flexibility.

## 8. Reproducibility, research integrity, and evaluation

### 8.1 Run identity

- [ ] Git SHA, dirty state, resolved Hydra config, and lock/container identity exist.
- [ ] Hardware, OS, driver, CUDA, PyTorch, precision, and seed are recorded.
- [ ] Model, tokenizer, and train/validation manifest fingerprints are recorded.
- [ ] Starting and ending target-token, optimizer-step, and elapsed-time counters exist.
- [ ] The run directory identifies every input even when W&B is disabled.
- [ ] Required reproducibility is stated without promising bitwise equality
  across PyTorch versions or platforms.
- [ ] Any deterministic-mode performance trade-off is measured separately.

### 8.2 Training and evaluation data

- [ ] Train and validation have zero identity or content overlap.
- [ ] Benchmark development and reserved-test access paths are separated.
- [ ] Exact and normalized benchmark-contamination checks are recorded.
- [ ] External-model weights, logits, and outputs never enter training data or targets.
- [ ] No pretrained capability other than the tokenizer enters the
  random-initialization model.
- [ ] Same-corpus memorization is not labeled held-out validation.
- [ ] Checkpoint, prompt, few-shot examples, decoding, and scorer revision are
  attached to evaluation results.

### 8.3 Sound conclusions

- [ ] Hypothesis and success/failure conditions were written before observing results.
- [ ] The baseline difference is one interpretable change.
- [ ] A speedup is not caused by changing the tokenizer or reducing target count.
- [ ] A lower loss is not caused by leakage, masks, reduction, or a changed scale.
- [ ] The conclusion is not based only on the best seed or checkpoint.
- [ ] Negative results retain config, run, cause, and what was ruled out.
- [ ] Claims include conditions, magnitude, spread, and trade-offs rather than
  only saying faster or better.

## 9. Checkpointing, W&B, and long-running operations

### 9.1 Checkpoint and resume

- [ ] Save model, optimizer, scheduler, precision state, counters, RNG, and stream cursor.
- [ ] Verify resolved config, data, tokenizer, and run identity before resume.
- [ ] Write to a temporary path, read back, then atomically rename.
- [ ] Measure pause, size, write throughput, and verification time at real scale.
- [ ] Measure additional memory during saving, including UMA/system impact.
- [ ] Rotate only after the replacement checkpoint verifies successfully.
- [ ] A corrupt newest checkpoint can fall back to the previous verified one.
- [ ] The first resumed batch, counters, and LR match the expected uninterrupted suffix.
- [ ] Signals, exceptions, disk full, and network failure cannot corrupt the last
  verified checkpoint.

### 9.2 W&B and logging

- [ ] Training, local metrics, and checkpoints work with W&B disabled or offline.
- [ ] Missing login, unknown quota, and network loss cannot destroy local work or
  unpredictably stall the hot path.
- [ ] Scalar-cadence A/B measurements quantify logging overhead.
- [ ] Model watch is enabled only for runs that need it and is off by default.
- [ ] Raw corpora and every checkpoint are not uploaded as artifacts.
- [ ] Artifact policy is explicit, such as `none|best|final|milestone`.
- [ ] Projected size and current plan/usage/retention are checked before upload.
- [ ] Failed uploads remain distinct from training failure and retain retry evidence.
- [ ] Metric keys and step/token axes remain comparable across runs.

### 9.3 Long-run preflight

- [ ] A short real-data pilot has completed.
- [ ] Success, stop, and failure conditions are declared.
- [ ] Measured tokens/s determines the run, validation, checkpoint, and benchmark forecast.
- [ ] Worst-case memory, disk, cache, and W&B artifact use are forecast.
- [ ] GPU, temperature, clocks, data wait, loss, gradients, and disk are observable.
- [ ] PID, run directory, command, start time, and expected end time are known.
- [ ] Stopping an invalid run preserves config, logs, and the last verified checkpoint.
- [ ] A retry links to the original failure rather than deleting and rewriting history.

## 10. Current implementation watchlist

This table records review targets visible in the current code. It is not a list
of items that must all be fixed immediately.

| Current location | Easy-to-miss risk | What to inspect when related code changes |
| --- | --- | --- |
| Module-level `DEVICE` in `src/train.py` | CUDA failure silently falls back to CPU, and device is fixed at import time | Explicit Hydra device, pre-data failure, real CUDA process |
| Batch `.to(device)` in `src/train.py` | Copy remains blocking even when pinned memory is enabled | H2D time, `non_blocking` A/B, UMA measurement |
| Preview batch in `src/train.py` | Starts and closes the real stream, then training restarts from the beginning | Duplicate startup, worker shutdown, first-batch reload time |
| `StreamingTokenDataset` | `num_workers=0`; the custom prefetch process is the only producer | Loader headroom, single-core saturation, order-safe worker design |
| `StreamLoader._sample_iter` | Tokenizes a whole document synchronously, creating long-document latency and memory spikes | Document-length p99, tokenization p99, peak RSS |
| `StreamLoader._packed_iter` | Repeated deletion from the front of a large list can be costly | CPU profile, buffer size, tokens/s on the largest document |
| Process prefetch | Serializes each NumPy window through a process queue | IPC share, buffer-size A/B, queue-empty rate |
| Stream re-iteration by epoch | Seed and source iterators restart the same prefix | Horizon, repeat policy, cursor, resume suffix |
| Local text dataset | Shuffles overlapping map-style windows over one full token tensor | Keep it smoke-only; do not extend it to real data |
| `Trainer._train_epoch` | Per-step `loss.item()` and W&B logging can synchronize CUDA and perform I/O | Logging-cadence A/B, timeline gaps between steps |
| `run.watch(self.model)` | Gradient/parameter watching can add hot-path cost | Watch off/on difference and W&B-disabled baseline |
| Epoch validation | A streaming epoch may trigger a large full validation pass | Validation budget, token cadence, pause time |
| Batch-mean loss | A partial final batch has the same weight as a full batch | Evaluated target-token count and manual NLL |
| Raw model-only checkpoint | No optimizer/counter/cursor and synchronous overwrite of one file | Resume, pause, atomicity, rotation, disk |
| External tokenizer | Larger vocabulary can greatly expand embeddings, LM head, optimizer, and checkpoints | Parameter delta, memory, step time, bytes/s |
| 750 GB cache limit | Exceeds the currently observed roughly 525 GB root free space | Dynamic preflight, safe limit, full-run forecast |
| DGX Spark UMA | Dedicated-VRAM reporting does not show total pressure | Report `free`, RSS, swap, and PyTorch peaks together |

## 11. ROADMAP ticket selection guide

These are the primary areas for each ticket. Unlisted areas do not need to be
executed automatically.

| Ticket | Priority review | Ticket-specific reminder | Suggested size |
| --- | --- | --- | --- |
| ENV-001 | 5.1, 5.3, 3 | Clean install, aarch64, explicit CUDA, BF16, no silent fallback | R2 |
| DATA-001 | 4.3, 4.1, 6.1 | No missing/duplicate transition and no supply regression after carry | R1 + R2 |
| MODEL-001 | 6.1, 6.2, 5.2, 7 | Preserve invariants and record reference step time/memory | R1 + R2 |
| EXP-001 | 8.1, 8.3, 7 | Negative results cannot omit config or evidence | R0 |
| TOK-001 | 4.2, 4.1, 5.4, 8.2 | Japanese fallback, offline pin, vocabulary-driven model cost | R1 + R2 |
| DATA-002 | 4.4, 8.2, 4.1 | Do not put checksum/split validation on every hot-path sample | R1 |
| CFG-001 | 1, 2, 7.1, 8.1 | Real profile composes real sources without duplicating debug config | R1 |
| REP-001 | 8.1, 3, 6.1 | Seed coverage, run identity, deterministic-performance trade-off | R1; R2 if needed |
| DATA-003 | 4.1, 4.3, 9.1 | Shuffle/cursor memory, state size, supply speed, exact suffix | R2 |
| LOOP-001 | All of 6, 3, 9.2 | Counters, token-weighted metrics, event cadence, per-step sync | R2 |
| CKPT-001 | 9.1, 5.4, 6.3 | Real-size atomic-save pause and post-resume trajectory/cursor | R2 |
| CI-001 | 1, 7.3, 8.1 | No network/credentials and parity with canonical local command | R0 + R1 |
| STAB-001 | 5.2, 6.1, 6.2, 3 | Stability first: BF16 kernels, accumulation, clipping, non-finite stop, throughput | R2 + R3 |
| GEN-001 | 7.2, 8.2, 9.1 | Reconstruct from checkpoint without chat/SFT branches in training | R1 |
| WB-001 | 9.2, 6.3, 5.4 | Logging/watch/upload off/on overhead and quota-failure isolation | R2 |
| GATE-001 | 6, 8, 9.1 | Label memorization honestly; verify resume and sampling end to end | R2 |
| DATA-004 | All of 4, 5.3, 5.4, 8.2 | Cold/warm live-source throughput, long-document tail, disk headroom | R2 + R3 |
| VAL-001 | 6.1, 8.2, 6.3 | Standalone/training-time parity, token budget, training pause | R2 |
| BENCH-001 | 8.2, 8.3, 9.2 | Reserved-test guard, contamination, decoding/scorer identity | R1 |
| DGX-001 | 3, 4.1, all of 5, 6.4 | Model/loader/end-to-end decomposition and thermal repeated measurement | R3 |
| OPS-001 | 9.3, 8.1, 5.4 | Non-destructive preflight and retained failure/retry evidence | R2 end to end |
| RUN-001 | 9.3 and applicable parts of 4-9 | Time/token/storage plan derived directly from pilot and stop conditions | R4 |
| HUMAN-001 | 8.2, 8.3 | Preserve blinding; never return prompts, outputs, or scores to training | R1 |

### Adding a later performance ticket

In addition to the deferred-work policy in `ROADMAP.md`, include:

- the measured bottleneck and the trace or breakdown that demonstrates it
- reference commit, config, and result
- correctness tolerance for the objective, transitions, and numerical output
- target metric and minimum meaningful improvement
- measurement window, warmup, repetitions, cache, profiler, and logging conditions
- speed, memory, quality, and complexity trade-offs
- rollback condition that removes the change when it does not improve the result

## 12. Bottleneck diagnosis order

### Training is slow

1. Compare target tokens/s and step time with baseline.
2. Measure model-only with a synthetic device batch.
3. Measure loader-only with the real path.
4. Inspect end-to-end data wait and GPU gaps.
5. If model-only is slow, inspect kernels, dtype, shape, synchronization, and optimizer.
6. If loader-only is slow, inspect source, tokenizer, packing, IPC, cache, and network.
7. If both are fast alone, inspect copy, synchronization, logging, queueing, and contention.
8. If only the pilot slows down, inspect thermals, clocks, swap, cache eviction,
   and network tails.

### GPU utilization is low

1. Confirm a real CUDA process and CUDA kernels.
2. If tokens/s is sufficient, consider whether a small model or short kernels
   explain the low average.
3. For GPU gaps, inspect simultaneous loader blocking, CPU state, `.item()`, and logging.
4. Without gaps, inspect launch overhead, memory bounds, and non-Tensor-Core dtype/shape.
5. Use a batch/sequence scaling curve to test insufficient parallelism.

### Memory is growing

1. Separate PyTorch allocated/reserved memory, RSS, system available memory, and swap.
2. Separate one-time warmup/autotune growth from per-step monotonic growth.
3. Inspect retained graphs, logging/watch, metric lists, prefetch queues,
   long-document tokenization, and filesystem cache.
4. Check whether the peak occurs only during checkpoint/validation and whether it recovers.
5. On DGX Spark, inspect total UMA rather than GPU-memory output alone.

### Loss or quality regressed

1. Confirm tokenizer, data, target tokens, masks, and reduction are unchanged.
2. Confirm effective batch, LR schedule, optimizer steps, precision, and seed.
3. Separate train-loss, held-out NLL, and generation/benchmark regressions.
4. Check whether a throughput change dropped targets or changed padding accounting.
5. Decide first whether this is a numerical regression or an incomparable experiment definition.

## 13. Review record template

Keep only applicable fields and paste the result into the ticket or PR.

```markdown
## ML system check

- Ticket / hypothesis:
- Verdict: PASS / PASS WITH NOTE / FAIL
- Baseline commit or run:
- Candidate commit or run:
- Selected sections: for example 4.1, 5.2, 6.3, 7.1
- Why other major sections are N/A:

### Conditions
- Hydra command and resolved config:
- Data/tokenizer/model fingerprints:
- Hardware / OS / driver / CUDA / PyTorch:
- Precision / sequence / micro batch / accumulation:
- Warmup / measured steps / repetitions / cache state:
- W&B / profiler / checkpoint / validation settings:

### Result
- Intended behavior:
- target tokens/s, baseline -> candidate:
- step median / p95, baseline -> candidate:
- data wait and GPU gap:
- GPU clock / power / temperature:
- system available memory / swap / PyTorch peak:
- loss / gradient / LR health:
- checkpoint/eval/logging overhead:

### Engineering judgment
- Objective or data semantics changed?:
- New coupling, duplicate path, or config source?:
- Next model/data experiment remains localized?:
- Known trade-off and why acceptable:
- Unresolved risk and next action:
- Evidence paths / run URLs / trace paths:
```

## 14. Choosing observation tools

Use only the tools required by the ticket. Do not leave all instrumentation on
or profile every run.

- `nvidia-smi` / `nvidia-smi dmon`: Low-frequency time series for compute
  processes, GPU activity, power, clocks, and temperature. Account for DGX
  Spark memory-reporting limitations.
- `free -h`, `vmstat 1`: Total UMA available memory, swap, run queue, and I/O wait.
- `pidstat`, `iostat`: When available, separate process CPU/RSS/I/O and NVMe wait.
- PyTorch Profiler: Capture a few dozen CPU/CUDA steps to inspect operators,
  copies, synchronization, shapes, and memory, using a warmup/active schedule.
- Nsight Systems: Correlate GPU gaps with CPU blocking, CUDA APIs, kernels,
  memcpy, and NVTX when PyTorch Profiler is insufficient.
- W&B/local JSON: Observe low-frequency trends over a pilot or long run; do not
  use these as a replacement for detailed traces.
- Plain wall-clock timer: Repeat clearly bounded operations such as loader-only,
  model-only, checkpoint, and validation.

Shape recording, stack traces, CUDA-memory traces, and fine GPU metrics can have
material overhead. Separate profile runs from normal runs and never report a
profiled result as normal throughput.

## 15. Primary and official references

- [DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html): ARM64, 128 GB unified memory, power, and hardware specifications.
- [DGX Spark User Guide / Known Issues](https://docs.nvidia.com/dgx/dgx-spark/index.html): UMA memory reporting and `nvidia-smi` limitations.
- [DGX Spark Porting Guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/index.html): ARM/UMA, profiling, and DGX Spark optimization.
- [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html): Utilization, power, clocks, temperature, and `dmon` definitions.
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/): CPU/CUDA timelines, GPU metrics, and UMA/page-fault observation.
- [Nsight Systems Analysis Guide](https://docs.nvidia.com/nsight-systems/2025.5/AnalysisGuide/index.html): GPU starvation, low utilization, and CPU-blocking diagnosis.
- [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler): CPU/CUDA activity, warmup/active schedules, and traces.
- [PyTorch DataLoader](https://docs.pytorch.org/docs/stable/data.html): Iterable datasets, workers, prefetch, and pinned memory.
- [PyTorch Data Loading Optimization](https://docs.pytorch.org/tutorials/intermediate/intermediate_data_loading_tutorial.html): Pinning, non-blocking transfer, and data-pipeline measurement.
- [PyTorch Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html): Seeds, platform differences, and deterministic-performance trade-offs.
- [PyTorch Automatic Mixed Precision](https://docs.pytorch.org/docs/stable/accelerator/amp.html): Autocast, reduced precision, and gradient scaling.
- [NVIDIA Deep Learning Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/index.html): Math/latency/memory bounds, Tensor Cores, shapes, and data movement.

External versions, quotas, and hardware/software constraints change. Recheck
current official documentation when a ticket starts; do not treat values or
commands in this document as permanent constants.
