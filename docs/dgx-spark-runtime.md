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
