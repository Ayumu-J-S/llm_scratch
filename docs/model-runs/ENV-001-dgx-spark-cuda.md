# ENV-001 - Reproducible DGX Spark CUDA Runtime

- PR: [#15](https://github.com/Ayumu-J-S/llm_scratch/pull/15) (ready for human review)
- Branch: `codex/env-001-dgx-spark-cuda`
- Ticket: `ENV-001`
- Hypothesis: a digest-pinned NVIDIA NGC PyTorch ARM64 container plus explicit
  Hydra device selection can make the GB10 CUDA path reproducible and fail
  closed without changing the model or optimizing performance.
- Started: 2026-07-11T19:20:00Z
- Final verdict: PASS WITH NOTE
- Final record owner: primary task; exact runtime identity not exposed

## Scope and decision context

- Goal: make a clean documented setup use the GB10 instead of silently falling
  back to CPU.
- In scope: current NVIDIA-supported ARM64/GB10 PyTorch delivery; immutable
  container pin; environment diagnostic; 10-step CUDA forward/backward smoke;
  explicit Hydra device selection; fail-before-data behavior; explicit CPU test
  path; clean-environment and existing-suite evidence.
- Out of scope: model changes, `torch.compile`, custom kernels, performance
  tuning, real pretraining, long thermal pilots, and trainer redesign.
- Relevant `PHILOSOPHY.md` principles: one DGX Spark is the real boundary;
  environment identity and failures are visible; smoke before substantial
  compute; prefer a reproducible conventional path over hidden fallback.
- Baseline commit/run: `a05eb1de5656643757a1c3d98047c98dedea8bfa`.
- Intended evidence: host/container identity, clean pinned setup, diagnostic,
  real GB10 BF16 capability, 10 finite CUDA optimizer steps, CUDA process
  visibility, early unavailable-device failure, explicit CPU suite, and no
  unsupported UMA memory claim.

## Current verified facts and primary sources

- Host: Linux aarch64, NVIDIA GB10, driver 580.159.03. Host uv environment:
  PyTorch 2.10.0+cpu, `torch.version.cuda is None`, CUDA unavailable.
- NVIDIA documents that DGX Spark ships with the NVIDIA Container Toolkit and
  recommends GPU containers via `docker run --gpus=all`:
  <https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html>.
- NVIDIA says the latest DGX Spark CUDA/framework features are delivered through
  NGC containers:
  <https://docs.nvidia.com/dgx/dgx-spark/known-issues.html>.
- NVIDIA NGC PyTorch `26.06-py3` is CUDA 13.3.0, Ubuntu 24.04, and PyTorch
  2.13.0a0+8145d630e8:
  <https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-06.html>.
- NVIDIA's framework matrix describes multi-architecture x86/ARM SBSA support:
  <https://docs.nvidia.com/deeplearning/frameworks/support-matrix/>.
- CUDA 13.x minor-version compatibility requires an R580-or-newer driver:
  <https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html>.
- Registry resolution on 2026-07-11: tag `nvcr.io/nvidia/pytorch:26.06-py3`
  resolves to multi-arch digest
  `sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1`;
  its linux/arm64 manifest is
  `sha256:dcae8df08ef61b019b8eb109113428cba4ef0e37484c6e722406150dd5ada759`.

## Predeclared implementation contract

- Host uv remains the explicit CPU development/test environment. The CUDA path
  is the NGC image pinned by the multi-architecture digest above and must resolve
  to the recorded linux/arm64 manifest.
- Generate a committed runtime-only requirements export with `uv export
  --locked --no-default-groups --no-dev --no-emit-project --prune torch`.
  The overlay must reject Torch, Triton, NVIDIA, or CUDA providers, install the
  complete hash-locked closure with `--no-deps` into a system-site-packages
  venv, and prove the NGC Torch version/CUDA build/module identity did not move.
- Hydra defaults to exact `runtime.device: cuda`; exact `cpu` is the only test
  override. Unknown values and unavailable CUDA fail before tokenizer, data,
  model, trainer, or W&B construction. No `auto` or implicit fallback exists.
- Diagnostic JSON/text reports host/OS/Python/Torch/module path, compiled and
  actual runtime CUDA, driver, devices/compute capability, BF16 support, image
  digests, RSS, system memory/swap, and allocator values with an explicit DGX
  Spark unified-memory caveat.
- CUDA smoke uses `SimpleDecoderTransformer`, fixed synthetic inputs, AdamW,
  BF16 autocast, and exactly ten finite forward/CE/backward/optimizer steps. It
  verifies CUDA placement, finite/nonzero gradients, CUDA Adam moment tensors,
  expected CPU step-counter bookkeeping, synchronization, and current-PID
  visibility through `nvidia-smi`.
- Record pull/build time and disk state without pruning shared Docker data.
  Ten steps are correctness evidence, not an R2 throughput/thermal/stability
  claim; longer pilots remain later-ticket scope.

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | handoff | not exposed by runtime | not exposed by runtime | ticket, philosophy, CHECK, host facts, official NVIDIA sources | Requested model: Sol; requested reasoning: Ultra. Plan for the smallest reproducible container/device/diagnostic/smoke design | completed | Selected the digest-pinned NGC image, lock-derived non-Torch overlay with provider guards, explicit Hydra device authority, JSON diagnostic, exact ten-step BF16 repository-model smoke, CPU validation, and adversarial matrix | Planner handoff retained in primary task; no repository mutation |
| 1 | implementation | not exposed by runtime | not exposed by runtime | `100a11129b97edf50967d41d75ac4c99f18f9bc9`, accepted plan | Requested model: Luna; requested reasoning: Extra High. Implement and exercise the clean GB10 runtime | completed | Added the digest-pinned ARM64 NGC container, byte-stable non-Torch overlay and provider/Torch-identity guards, explicit initialized-device Hydra gate, environment diagnostic, exact ten-step BF16 repository-model CUDA smoke, CPU tests, commands, and operating guide | 58 passed / 3 external skips; clean pull/build; strict GB10 diagnostic; ten finite BF16 CUDA updates; negative no-GPU exits |
| 1 | review | not exposed by runtime | not exposed by runtime | `7d929f05fe2dd4bc626b11f0c2b4ea701b47c177` | Requested model: heavier reviewer; requested reasoning: Extra Thinking. Independent `/review` against the ticket, philosophy, and applicable CHECK sections | FAIL | Found a version-suffixed CUDA-provider guard gap, an optimizer-state reporting overclaim, and a final image that predated the exact candidate tree | 58 passed / 3 skips and real GB10 behavior passed, but the committed reproducibility contract did not |
| 1 | repair | not exposed by runtime | not exposed by runtime | failed cycle-1 review plus `7d929f05fe2dd4bc626b11f0c2b4ea701b47c177` | Requested model: Luna; requested reasoning: Extra High. Close every review finding without broadening ticket scope | completed | Reject version-suffixed CUDA providers, distinguish CUDA Adam moments from CPU step bookkeeping, and prepare an exact clean-commit rebuild | 59 passed / 3 skips; poison/false-positive, lock-parity, Ruff, Hydra, and diff gates passed |
| 2 | review | not exposed by runtime | not exposed by runtime | `95dee560076258247776ead9940bfbe30e619699` and clean no-cache image `sha256:25a02a5357d3f22339ddea8de78e2b0725a47dc6bbe15f336fa74242889a648b` | Requested model: heavier reviewer; requested reasoning: Extra Thinking. Fresh independent `/review` of the repaired implementation and real GB10 evidence | PASS WITH NOTE | All blocking findings repaired; exact ten-step wiring proof is sufficient for ENV-001 but is not R2 real-data/steady-state evidence | Independent 59 passed / 3 skips, provider adversarial matrix, exact source parity, strict diagnostic/smoke, and no-GPU failures passed |

## Check selection and verdicts

### Review cycle 1

- Review model / mode: `not exposed by runtime / not exposed by runtime`
  Requested reviewer: heavier / Extra Thinking.
- Commit reviewed: `7d929f05fe2dd4bc626b11f0c2b4ea701b47c177`
- Selected `CHECK.md` sections: 1, 2, 3, 5.1, 5.3, 7, and 11 ENV-001
- Major sections marked N/A and why: section 4 because data/tokenizer/packing
  did not change; 5.2 because no efficiency claim exists; 5.4 because no
  training-sized capacity/checkpoint claim exists; section 6 beyond smoke
  health because objective/optimizer/scheduler/loop behavior did not change;
  8.2 and 9 because evaluation data, checkpoints, resume, and W&B did not
  change; R3/R4 because no pilot or consequential-run claim exists.
- Ticket acceptance result: runtime behavior passed, but the committed
  provider-rejection contract and exact final-image provenance failed.
- Philosophy alignment: bounded compute, fail-closed device authority, and UMA
  honesty passed; inaccurate provider and optimizer claims required repair.
- Complexity / change-surface result: pass; localized runtime work with no
  model, objective, data, tokenizer, compile, kernel, or performance change.
- ML-system result: the real GB10 path passed, but reproducibility evidence was
  incomplete.
- Verdict: FAIL

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |
| P1 | runtime overlay | The provider regex required a boundary immediately after `cuda`, so version-suffixed CUDA providers could enter a future regenerated overlay | `jax-cuda12-plugin==0.8` and `cupy-cuda13x==13.0` returned no forbidden project | Recognize version-suffixed CUDA namespaces and add poison plus false-positive tests |
| P2 | smoke evidence | The report excluded AdamW's CPU `step` tensor but labeled optimizer state simply `cuda` | Independent probe reported `exp_avg:cuda`, `exp_avg_sq:cuda`, `step:cpu` | Assert/report CUDA Adam moments and CPU step bookkeeping separately |
| P2 | image provenance | Image `sha256:894745a9...` contained the Makefile before the final `--quiet` edit | Candidate commit time was later than the image and the file differed | Rebuild without cache from the exact clean repair commit and verify source parity |

### Review cycle 2

- Review model / mode: `not exposed by runtime / not exposed by runtime`
  Requested reviewer: heavier / Extra Thinking.
- Commit reviewed: `95dee560076258247776ead9940bfbe30e619699`
- Selected `CHECK.md` sections: 1, 2, 3, 5.1, 5.3, 7, and 11 ENV-001
- Major sections marked N/A and why: same scoped exclusions as cycle 1.
- Ticket acceptance result: pass; the pinned clean ARM64 setup, diagnostic,
  exact ten-step CUDA/BF16 smoke, explicit CPU option, and fail-before-data
  behavior are demonstrated.
- Philosophy alignment: pass; one-machine boundary, visible failure, bounded
  validation, pinned inputs, honest UMA limits, and no hidden fallback.
- Complexity / change-surface result: pass; the repair remained localized.
- ML-system result: pass within the declared R1 wiring boundary.
- Verdict: PASS WITH NOTE. ENV-001's exact ten-step proof is complete, but it
  is not CHECK R2 real-data/steady-state evidence; later stability/training
  tickets retain that responsibility.

## Failed-review handoff

- Implementation model/mode: `not exposed by runtime / not exposed by runtime`.
  Requested implementation model/mode: Luna / Extra High.
- Failed review: cycle 1 at `7d929f05fe2dd4bc626b11f0c2b4ea701b47c177`.
- Repair model/mode: `not exposed by runtime / not exposed by runtime`.
  Requested repair model/mode: Luna / Extra High. Selected because the original implementation
  agent retained the exact container/provider/smoke context.
- Context handed off: the complete independent FAIL, exact poison package
  reproductions, expected AdamW state placement, requirement for a clean
  repair-commit rebuild, ticket boundaries, and prohibition on PR/Docker-data
  mutation.

## Repair result

- `cuda`, `cuda12`, and `cuda13x` delimiter-bounded provider namespaces are
  rejected; `jax-cuda12-plugin`, `cupy-cuda13x`, and `cuda-python` fail while
  unrelated `cudatext`, `education-tools`, and `torchmetrics` remain allowed.
- The smoke now requires/reports CUDA Adam moment tensors and separately
  requires/reports CPU step-counter bookkeeping.
- Repair commit `95dee560076258247776ead9940bfbe30e619699` was clean before a
  no-cache rebuild. Image `sha256:25a02a5357d3f22339ddea8de78e2b0725a47dc6bbe15f336fa74242889a648b`
  was created afterward, and the five critical runtime files reproduced the
  same host/container aggregate SHA-256
  `09158ffaa92430cb6c1713c71f543e282b37ab5f21a58dfdddafea60860679b9`.
- Fresh independent cycle 2 returned PASS WITH NOTE.

## Final evidence

- Resolved Hydra command/config: `uv run python src/train.py --cfg job
  --resolve` reports exact `runtime.device: cuda`. The default no-GPU container
  training command exits 1 at `select_device` before any tokenizer/data/model
  log; `runtime.device=cpu` remains the explicit tested override.
- Data/tokenizer/model identity: synthetic fixed token IDs only; repository
  `SimpleDecoderTransformer` with vocabulary 257, width 64, 4 heads, 2 layers,
  context 32, batch 4, no padding, and dropout 0.0. No tokenizer artifact,
  source corpus, target boundary, or evaluation path changed.
- Validation and measurements:
  - Host before: aarch64, GB10, driver 580.159.03; uv Torch 2.10.0+cpu,
    `torch.version.cuda=None`, CUDA unavailable; 525 GiB root free; Docker
    images 44.66 GB, volumes 135.7 GB, build cache 306.9 MB.
  - Exact digest pull completed in 1326.82 seconds. Base image identity:
    `sha256:9629b436aef8bd90147fd657137047aee94e7b81ada54c3f7209cbce1d24b490`,
    linux/arm64, with the declared repository digest.
  - First no-cache overlay build completed in 31.43 seconds. The pre-review
    audit image was
    `sha256:894745a95da330c063f06a2a2257428bd02a3a68ffdd10c890626c42bf748d27`;
    cycle 1 correctly rejected it as final provenance because a later
    non-semantic Makefile edit was not present. The clean repair commit's
    no-cache build took 25 seconds. Final local image identity:
    `sha256:25a02a5357d3f22339ddea8de78e2b0725a47dc6bbe15f336fa74242889a648b`,
    linux/arm64, with exact base and arm64-manifest labels.
  - Pre/post overlay Torch identity remained
    `2.13.0a0+8145d630e8.nv26.06`, CUDA build 13.3, module
    `/usr/local/lib/python3.12/dist-packages/torch/__init__.py`, with installed
    `METADATA` + `RECORD` SHA-256
    `c85803b0af1091de7f318c53570de7484270535cf8fcce3aa9030cf137697519`.
  - Strict JSON diagnostic passed on NVIDIA GB10, compute capability 12.1,
    CUDA runtime API value 13030, driver 580.159.03, BF16 true, one CUDA
    device, 130596048896 bytes unified memory. The report carries the required
    allocator/UMA caveat.
  - The final smoke completed 10 AdamW updates under BF16 autocast. Losses were
    all finite and decreased from 5.595581 to 4.068237; every trainable
    parameter had a finite gradient and at least one was nonzero; model
    parameters, inputs, labels, logits, loss, and Adam moment tensors were CUDA;
    expected AdamW CPU step-counter bookkeeping was asserted and separately
    reported; current PID was visible through `nvidia-smi`; peak PyTorch
    allocation was 70,545,408 bytes (not interpreted as total memory).
  - Without `--gpus all`, required diagnostic exited 2 with valid JSON, CUDA
    smoke exited 1, and the default training command exited 1 before data. No
    CUDA-to-CPU fallback occurred.
  - Host validation: `uv run pytest -q` = 59 passed, 3 explicit external skips;
    Ruff, format, `uv lock --check`, `git diff --check`, provider-poison test,
    runtime-provider scan, Hydra resolution, exact byte-for-byte temporary lock
    regeneration, JSON CLI parsing, and CUDA count/init/name failures all
    passed. Canonical `make sync`, `runtime-lock`, `diagnose`, `dgx-diagnose`,
    `dgx-smoke`, and repaired `test-cpu` targets passed.
  - Host after: 499 GiB root free; Docker images 71.29 GB, volumes unchanged at
    135.7 GB, build cache 3.172 GB. Nothing was pruned or deleted.
- Performance/resource result if applicable: R1 wiring proof only. No
  throughput, GPU-efficiency, thermal, stability, or available-memory claim.
- Failed attempts retained at: `/tmp/env001-evidence` during implementation
  (not committed). The first local test collection imported a script as a
  package and was repaired to exercise it as a subprocess. The first strict
  JSON capture contained the NGC entrypoint banner and returned no runtime API
  value; machine-readable commands now bypass the banner and the diagnostic
  queries `cudaRuntimeGetVersion` directly. Initial raw requirement exports
  differed only in uv's generated output-path header; the repaired export omits
  that header and now reproduces entirely byte-for-byte. `pip check` reports
  NGC's pre-existing
  `triton-kernels -> pytest` metadata gap in both the untouched base and the
  overlay; the runtime export intentionally excludes dev dependencies, and the
  real CUDA smoke passed. The first canonical `make sync --locked` check honored this project's
  notebook-only default group, removed pytest/Ruff, and caused `make test-cpu`
  to fail with a missing executable. `make sync` now explicitly installs the
  locked `dev` group; the target then restored four packages and the complete
  CPU suite passed.
- Known trade-offs: the supported CUDA environment is containerized while the
  host uv lock remains the explicit CPU development/test environment.
- Unresolved risks: ENV-001 is only a ten-step wiring proof; STAB-001 and later
  tickets must establish the real BF16 recipe and longer
  real-data/stability/resource evidence.
- Human decision requested: review the PASS WITH NOTE evidence and decide
  whether to merge; a human remains the sole merge authority.

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |
| not exposed by runtime / not exposed by runtime | implementation | Preserved NGC framework identity, kept device authority small, exercised real GB10 and negative paths, retained failed evidence | Initial strict JSON capture was contaminated by the inherited NGC banner; the first committed provider regex and optimizer summary were too broad | Predeclared digests, exact smoke contract, provider guard, CHECK 5.1/5.3 boundaries, and failed-review reproductions | repaired successfully |
| not exposed by runtime / not exposed by runtime | independent review | Re-ran real GB10 and negative paths, distinguished ticket wiring proof from R2 claims, and found contract/provenance gaps despite passing runtime behavior | Exact runtime did not expose the requested heavier model or Extra Thinking setting | Stable commits, retained failed evidence, official NVIDIA sources, and exact machine commands | cycle 1 FAIL; cycle 2 PASS WITH NOTE |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [x] Updated per-model implementation, repair, and review counts.
- [x] Confirmed that the implementation trail matches this record and the PR
  handoff.
