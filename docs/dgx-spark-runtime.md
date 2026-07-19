# DGX Spark CUDA runtime

ENV-001 keeps two explicit environments:

- the host uv lock is the CPU development/test environment;
- the CUDA environment is NVIDIA's ARM64 NGC PyTorch 26.06 image, pinned to
  multi-architecture digest
  `sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1`.

The recorded linux/arm64 child manifest is
`sha256:dcae8df08ef61b019b8eb109113428cba4ef0e37484c6e722406150dd5ada759`.
NVIDIA documents Docker GPU access on DGX Spark and delivers current framework
support through NGC containers. The selected release contains PyTorch
2.13.0a0+8145d630e8 and CUDA 13.3.0; [PR #15](https://github.com/Ayumu-J-S/llm_scratch/pull/15)
and the commands below preserve the source links and exact machine evidence.

## Build from pinned inputs

Install Docker with NVIDIA Container Toolkit as provided by the DGX Spark setup.
Regenerate and verify the hash-locked non-Torch overlay when `uv.lock` changes:

```bash
make runtime-lock
git diff --exit-code requirements/runtime.txt
```

The export omits uv's command header so regeneration is byte-for-byte stable
regardless of the destination filename. The CPU suite checks exact parity by
exporting to a temporary path.

Build the ARM64 image without a stale layer:

```bash
docker pull nvcr.io/nvidia/pytorch@sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1
make dgx-build
```

The Dockerfile does not run `uv sync`. It creates a system-site-packages venv,
rejects Torch, Triton, NVIDIA, and CUDA provider packages in the overlay, then
installs the complete export with `pip --require-hashes --no-deps`. Build-time
guards compare the NGC Torch version, CUDA build, and resolved module path before
and after installation, along with a SHA-256 over Torch's installed `METADATA`
and `RECORD` files.

## Diagnose the actual runtime

Require the GPU and BF16 capability, with human-readable output:

```bash
make dgx-diagnose
```

Machine-readable evidence is available with:

```bash
docker run --rm --gpus all --entrypoint python llm-scratch:env-001 \
  scripts/diagnose_environment.py --json --require-cuda --require-bf16
```

For training-sized commands, retain the Make targets' NGC-recommended
`--ipc=host --ulimit memlock=-1 --ulimit stack=67108864` runtime flags.

The diagnostic reports the host/OS/architecture, Python and Torch identities,
compiled and actual CUDA runtime, driver, device and compute capability, BF16,
image digests, process RSS, system memory/swap, and CUDA allocator counters. DGX
Spark has unified CPU/GPU memory: allocator values are not total available
memory, and unsupported `nvidia-smi` Memory-Usage must not be interpreted as
spare capacity.

Both negative checks must exit nonzero when no GPU is passed:

```bash
docker run --rm llm-scratch:env-001 \
  python scripts/diagnose_environment.py --require-cuda --require-bf16
docker run --rm llm-scratch:env-001 python scripts/cuda_smoke.py
```

## Exact ten-step CUDA proof

```bash
make dgx-smoke
```

This executes exactly ten AdamW updates of the repository's
`SimpleDecoderTransformer` on fixed synthetic CUDA inputs under BF16 autocast.
It requires finite loss, finite/nonzero gradients, CUDA model/input/logit/loss
placement, CUDA Adam moment tensors, expected CPU step-counter bookkeeping,
synchronization, and visibility of its own PID as an `nvidia-smi` compute
process. It is a wiring/correctness smoke, not throughput, available memory,
thermal, or long-run stability evidence.

## Training device authority

`config/train.yaml` defaults to exactly `runtime.device=cuda`. If CUDA is
unavailable, the entrypoint raises before tokenizer or data loading. CPU is
available only as an explicit development override:

```bash
uv run python src/train.py runtime.device=cpu wandb.mode=disabled
make test-cpu
```

There is no `auto` setting and no silent CUDA-to-CPU fallback.

## DGX-001 profile measurement and selection

`config/dgx.yaml` predeclares a nine-arm matrix: the conventional width-384
decoder at 18, 26, and 34 layers, crossed with 1,024, 2,048, and 4,096-token
contexts. Micro-batch size scales 8, 4, and 2 respectively, so every arm trains
exactly 32,768 targets per optimizer update at accumulation 4. This compares
model size, useful context, throughput, and UMA pressure without changing the
objective or architecture.

Inspect the plan without starting a container:

```bash
make dgx-plan
```

Run all arms three times at one exact clean commit and the pinned ENV-001 image:

```bash
HEAD=$(git rev-parse HEAD)
make dgx-measurements \
  EXPECTED_COMMIT="$HEAD" \
  OUTPUT_ROOT="/tmp/dgx-001-$HEAD"
make dgx-summarize OUTPUT_ROOT="/tmp/dgx-001-$HEAD"
```

Every arm excludes ten warm-up updates, retains twenty measured updates, uses
CUDA events only at measurement boundaries, exercises the pinned bilingual
train/validation path, writes a verified final checkpoint, and samples host/GPU
state out of band. The summarizer fails closed on incomplete repetitions,
commit/image drift, non-finite training, unavailable CUDA events, sampler gaps,
UMA or disk floors, swap growth, temperature above 80 C, allocator growth, or a
missing verified checkpoint. It reports median and spread, step median/p95/max,
trained-target tokens/s, phase/data-wait decomposition, memory, validation and
checkpoint overhead, and conservative 1-hour/24-hour/7-day budgets.

Selection is deterministic: a candidate must pass every gate and project at
least one billion targets in seven days from its slowest repetition. Among
candidates no more than 20% slower than the fastest, choose the deepest model;
then choose the longest context retaining at least 85% of that model's fastest
throughput. Within a 3% tie, lower measured allocator use wins. This leaves
quality/storage headroom instead of selecting the largest arm that merely
avoids OOM.

After reviewing `dgx-summary.json`, run the selected arm for the required
30-minute thermal/storage pilot and retain its verified checkpoint plus two
labeled base-model continuations:

```bash
make dgx-pilot \
  EXPECTED_COMMIT="$HEAD" \
  OUTPUT_ROOT="/tmp/dgx-001-pilot-$HEAD" \
  SELECTED="p85-ctx2048"
```

The committed `profile=pretrain_baseline` is a separate one-hour cap. It uses
online W&B scalar logging with watch disabled and artifact policy `none`, while
validation, rotating recovery checkpoints, and milestones run every 5M, 2.5M,
and 100M trained targets. A final DGX-001 record must show that its model/context
shape agrees with the exact-head summary before the profile is treated as
selected.
