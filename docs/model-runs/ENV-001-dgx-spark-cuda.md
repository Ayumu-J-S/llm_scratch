# ENV-001 - Reproducible DGX Spark CUDA Runtime

- PR: draft
- Branch: `codex/env-001-dgx-spark-cuda`
- Ticket: `ENV-001`
- Hypothesis: a digest-pinned NVIDIA NGC PyTorch ARM64 container plus explicit
  Hydra device selection can make the GB10 CUDA path reproducible and fail
  closed without changing the model or optimizing performance.
- Started: 2026-07-11T19:20:00Z
- Final verdict: in progress
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

## Execution timeline

| Cycle | Phase | Exact model identifier | Reasoning mode | Input commit/context | Requested work | Outcome | Main findings / changes | Evidence |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | handoff | pending | requested Sol / Ultra | ticket, philosophy, CHECK, host facts, official NVIDIA sources | Select the smallest reproducible container/device/diagnostic/smoke design | pending | No plan claimed yet | pending |
| 1 | implementation | pending | requested Luna / Extra High | pending accepted plan | Implement and exercise the clean GB10 runtime | pending | No implementation claimed yet | pending |
| 1 | review | pending | requested heavier / Extra Thinking | pending stable implementation commit | Independent `/review` | pending | No verdict claimed yet | pending |

## Check selection and verdicts

### Review cycle 1

- Review model / mode: pending
- Commit reviewed: pending
- Selected `CHECK.md` sections: 1, 2, 3, 5.1, 5.3, 7, and 11 ENV-001
- Major sections marked N/A and why: pending review
- Ticket acceptance result: pending
- Philosophy alignment: pending
- Complexity / change-surface result: pending
- ML-system result: pending
- Verdict: pending

#### Findings

| Severity | Area | What was wrong or good | Evidence | Required action |
| --- | --- | --- | --- | --- |

## Failed-review handoff

N/A - review pending.

## Repair result

N/A - review pending.

## Final evidence

- Resolved Hydra command/config: pending
- Data/tokenizer/model identity: pending
- Validation and measurements: pending
- Performance/resource result if applicable: no performance claim; pending
  environment/smoke evidence only
- Failed attempts retained at: this record
- Known trade-offs: the supported CUDA environment is containerized while the
  host uv lock remains the explicit CPU development/test environment.
- Unresolved risks: container pull/run, dependency overlay compatibility, and
  real GB10 smoke remain to be proved.
- Human decision requested: review only after independent verdict; a human
  remains the sole merge authority.

## Model assessment from this ticket

| Model / mode | Role | What it handled well | What it missed or made worse | Context that helped | Outcome |
| --- | --- | --- | --- | --- | --- |

## Ledger update

- [x] Added the PR/ticket row to `docs/model-runs/README.md`.
- [ ] Updated per-model attempt, pass, repair, and review counts.
- [ ] Confirmed that the PR execution trail matches this record.
