from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_runtime_requirements_reject_framework_providers(tmp_path):
    requirements = tmp_path / "poison.txt"
    requirements.write_text(
        "hydra-core==1.3.2\ntorch==2.10.0\ntorchvision==0.25.0\n"
        "nvidia-cublas-cu12==12.9\npytorch-triton==3.5\n"
        "jax-cuda12-plugin==0.8\ncupy-cuda13x==13.0\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "scripts/check_runtime_requirements.py", str(requirements)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert (
        "torch, torchvision, nvidia-cublas-cu12, pytorch-triton, "
        "jax-cuda12-plugin, cupy-cuda13x" in result.stderr
    )


def test_runtime_requirements_allow_unrelated_cuda_substrings(tmp_path):
    requirements = tmp_path / "non-providers.txt"
    requirements.write_text(
        "cudatext==1.0\neducation-tools==2.0\ntorchmetrics==1.9\n",
        encoding="utf-8",
    )
    subprocess.run(
        [sys.executable, "scripts/check_runtime_requirements.py", str(requirements)],
        check=True,
    )


def test_exported_runtime_requirements_do_not_supply_framework_providers():
    subprocess.run(
        [
            sys.executable,
            "scripts/check_runtime_requirements.py",
            "requirements/runtime.txt",
        ],
        check=True,
    )


def test_exported_runtime_requirements_reproduce_byte_for_byte(tmp_path):
    regenerated = tmp_path / "runtime.txt"
    subprocess.run(
        [
            "uv",
            "export",
            "--quiet",
            "--locked",
            "--no-default-groups",
            "--no-dev",
            "--no-emit-project",
            "--prune",
            "torch",
            "--no-header",
            "--output-file",
            str(regenerated),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert regenerated.read_bytes() == Path("requirements/runtime.txt").read_bytes()
