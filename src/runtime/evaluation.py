"""Shared fail-closed runtime identity for checkpoint evaluation."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from data.identity import canonical_json_bytes
from runtime.environment import collect_environment
from runtime.reproducibility import collect_git_identity, seed_everything, sha256_bytes, sha256_file


EVALUATION_DETERMINISM_POLICY = {
    "revision": "strict-math-sdpa-v1",
    "seed": 0,
    "deterministic_algorithms": {"enabled": True, "warn_only": False},
    "cublas_workspace_config": ":4096:8",
    "scaled_dot_product_attention": {
        "math": True,
        "flash": False,
        "memory_efficient": False,
        "cudnn": False,
    },
    "cudnn": {"deterministic": True, "benchmark": False},
    "float32": {
        "matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    },
}


def apply_evaluation_determinism_policy() -> dict[str, Any]:
    """Apply and verify the fixed evaluation execution policy."""

    required_workspace = str(EVALUATION_DETERMINISM_POLICY["cublas_workspace_config"])
    configured_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if torch.cuda.is_initialized() and configured_workspace != required_workspace:
        raise RuntimeError(
            "evaluation determinism requires CUBLAS_WORKSPACE_CONFIG="
            f"{required_workspace} before CUDA initialization"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = required_workspace

    seed_everything(int(EVALUATION_DETERMINISM_POLICY["seed"]), deterministic=True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    observed = {
        "deterministic_algorithms": {
            "enabled": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        },
        "scaled_dot_product_attention": {
            "math": torch.backends.cuda.math_sdp_enabled(),
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        },
        "cudnn": {
            "deterministic": torch.backends.cudnn.deterministic,
            "benchmark": torch.backends.cudnn.benchmark,
        },
        "float32": {
            "matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        },
    }
    for key in ("deterministic_algorithms", "scaled_dot_product_attention", "cudnn", "float32"):
        if observed[key] != EVALUATION_DETERMINISM_POLICY[key]:
            raise RuntimeError(
                f"evaluation determinism policy was not applied for {key}: {observed[key]}"
            )
    return copy.deepcopy(EVALUATION_DETERMINISM_POLICY)


def collect_evaluator_identity(
    root_dir: str | Path,
    *,
    resolved_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture executable source, dependencies, runtime, and complete configuration."""

    root = Path(root_dir).resolve()
    config = copy.deepcopy(dict(resolved_config))
    environment = collect_environment()
    cuda = environment["cuda"]
    torch_identity = environment["torch"]
    return {
        "git": collect_git_identity(root),
        "resolved_config": config,
        "resolved_config_sha256": sha256_bytes(canonical_json_bytes(config)),
        "lock": {
            "path": str(root / "uv.lock"),
            "sha256": sha256_file(root / "uv.lock"),
        },
        "environment": {
            "os": environment["os"],
            "os_release": environment["os_release"],
            "architecture": environment["architecture"],
            "python": environment["python"],
            "torch": {
                "version": torch_identity["version"],
                "module": torch_identity["module"],
                "compiled_cuda": torch_identity["compiled_cuda"],
            },
            "cuda": {
                "available": cuda["available"],
                "runtime_version": cuda["runtime_version"],
                "driver_version": cuda["driver_version"],
                "device_count": cuda["device_count"],
                "devices": cuda["devices"],
                "bf16_supported": cuda["bf16_supported"],
            },
            "container_image": environment["container_image"],
        },
    }
