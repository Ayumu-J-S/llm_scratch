FROM nvcr.io/nvidia/pytorch@sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1

ARG BASE_IMAGE="nvcr.io/nvidia/pytorch@sha256:43c018d6a12963f1a1bad85ef8574b5c2a978eec2be0ebcacfb87f69e0d210e1"
ARG BASE_ARM64_MANIFEST="sha256:dcae8df08ef61b019b8eb109113428cba4ef0e37484c6e722406150dd5ada759"
ARG RUNTIME_SPEC_SHA256
RUN test -n "${RUNTIME_SPEC_SHA256}"
LABEL org.opencontainers.image.title="llm-scratch DGX Spark runtime" \
      org.opencontainers.image.base.name="${BASE_IMAGE}" \
      io.llm-scratch.base.arm64-manifest="${BASE_ARM64_MANIFEST}" \
      io.llm-scratch.runtime-spec-sha256="${RUNTIME_SPEC_SHA256}"
ENV LLM_SCRATCH_BASE_IMAGE="${BASE_IMAGE}" \
    LLM_SCRATCH_BASE_ARM64_MANIFEST="${BASE_ARM64_MANIFEST}" \
    PATH="/opt/llm-scratch-venv/bin:${PATH}" \
    PYTHONPATH="/workspace/src"

WORKDIR /workspace
COPY requirements/runtime.txt /tmp/requirements-runtime.txt
COPY scripts/check_runtime_requirements.py /tmp/check_runtime_requirements.py

# Preserve NVIDIA's framework stack. The overlay is lock-derived, hash checked,
# and forbidden from supplying any Torch/CUDA provider.
RUN python /tmp/check_runtime_requirements.py /tmp/requirements-runtime.txt && \
    python - <<'PY'
import hashlib
import importlib.metadata
import json
from pathlib import Path
import torch

distribution = importlib.metadata.distribution("torch")
metadata_files = sorted(
    file for file in (distribution.files or []) if file.name in {"METADATA", "RECORD"}
)
if {file.name for file in metadata_files} != {"METADATA", "RECORD"}:
    raise SystemExit(f"Torch distribution metadata is incomplete: {metadata_files!r}")
metadata_hash = hashlib.sha256()
for file in metadata_files:
    metadata_hash.update(str(file).encode())
    metadata_hash.update(b"\0")
    metadata_hash.update(Path(distribution.locate_file(file)).read_bytes())
identity = {
    "version": torch.__version__,
    "cuda": torch.version.cuda,
    "module": str(Path(torch.__file__).resolve()),
    "distribution_metadata_sha256": metadata_hash.hexdigest(),
}
Path("/tmp/ngc-torch-identity.json").write_text(json.dumps(identity, sort_keys=True))
PY
RUN python -m venv --system-site-packages /opt/llm-scratch-venv && \
    /opt/llm-scratch-venv/bin/python -m pip install \
      --disable-pip-version-check --no-deps --require-hashes \
      -r /tmp/requirements-runtime.txt && \
    /opt/llm-scratch-venv/bin/python - <<'PY'
import hashlib
import importlib.metadata
import json
from pathlib import Path
import torch

before = json.loads(Path("/tmp/ngc-torch-identity.json").read_text())
distribution = importlib.metadata.distribution("torch")
metadata_files = sorted(
    file for file in (distribution.files or []) if file.name in {"METADATA", "RECORD"}
)
if {file.name for file in metadata_files} != {"METADATA", "RECORD"}:
    raise SystemExit(f"Torch distribution metadata is incomplete: {metadata_files!r}")
metadata_hash = hashlib.sha256()
for file in metadata_files:
    metadata_hash.update(str(file).encode())
    metadata_hash.update(b"\0")
    metadata_hash.update(Path(distribution.locate_file(file)).read_bytes())
after = {
    "version": torch.__version__,
    "cuda": torch.version.cuda,
    "module": str(Path(torch.__file__).resolve()),
    "distribution_metadata_sha256": metadata_hash.hexdigest(),
}
if after != before:
    raise SystemExit(f"NGC Torch identity changed during overlay install: {before!r} -> {after!r}")
if after["cuda"] is None:
    raise SystemExit(f"NGC Torch is not a CUDA build: {after!r}")
print(json.dumps(after, sort_keys=True))
PY

CMD ["python", "-c", "import torch; print(torch.__version__, torch.version.cuda)"]
