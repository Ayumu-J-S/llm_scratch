from __future__ import annotations

import os
import platform
import resource
import subprocess
import sys
from ctypes import CDLL, byref, c_int
from pathlib import Path
from typing import Any

import torch


UMA_CAVEAT = (
    "DGX Spark uses unified CPU/GPU memory; CUDA allocator values are not total available "
    "system memory and nvidia-smi Memory-Usage may be unsupported."
)


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def _meminfo() -> dict[str, int | None]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            name, raw = line.split(":", 1)
            values[name] = int(raw.strip().split()[0]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        return {
            "total_bytes": None,
            "available_bytes": None,
            "swap_total_bytes": None,
            "swap_free_bytes": None,
        }
    return {
        "total_bytes": values.get("MemTotal"),
        "available_bytes": values.get("MemAvailable"),
        "swap_total_bytes": values.get("SwapTotal"),
        "swap_free_bytes": values.get("SwapFree"),
    }


def _cuda_runtime_version() -> int | str | None:
    if not torch.cuda.is_available():
        return None
    try:
        cudart = CDLL("libcudart.so")
        version = c_int()
        error = cudart.cudaRuntimeGetVersion(byref(version))
    except OSError:
        return None
    if error != 0:
        return f"cudaRuntimeGetVersion error {error}"
    return version.value


def _os_release() -> dict[str, str | None]:
    try:
        release = platform.freedesktop_os_release()
    except OSError:
        release = {}
    return {
        "name": release.get("NAME"),
        "version": release.get("VERSION"),
        "version_id": release.get("VERSION_ID"),
        "pretty_name": release.get("PRETTY_NAME"),
    }


def collect_environment() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    devices: list[dict[str, Any]] = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "compute_capability": [properties.major, properties.minor],
                    "total_memory_bytes": properties.total_memory,
                }
            )

    allocator: dict[str, int] | None = None
    if cuda_available:
        allocator = {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        }

    driver = _command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    image = {
        "base": os.environ.get("LLM_SCRATCH_BASE_IMAGE"),
        "base_arm64_manifest": os.environ.get("LLM_SCRATCH_BASE_ARM64_MANIFEST"),
    }
    return {
        "host": platform.node(),
        "os": platform.platform(),
        "os_release": _os_release(),
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "torch": {
            "version": torch.__version__,
            "module": str(Path(torch.__file__).resolve()),
            "compiled_cuda": torch.version.cuda,
        },
        "cuda": {
            "available": cuda_available,
            "runtime_version": _cuda_runtime_version(),
            "driver_version": driver,
            "device_count": len(devices),
            "devices": devices,
            "bf16_supported": bool(cuda_available and torch.cuda.is_bf16_supported()),
            "allocator": allocator,
        },
        "process": {
            "pid": os.getpid(),
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "system_memory": _meminfo(),
        "container_image": image,
        "memory_interpretation": UMA_CAVEAT,
    }


def unmet_requirements(
    report: dict[str, Any], *, require_cuda: bool, require_bf16: bool
) -> list[str]:
    failures: list[str] = []
    if require_cuda and not report["cuda"]["available"]:
        failures.append("CUDA is required but unavailable")
    if require_bf16 and not report["cuda"]["bf16_supported"]:
        failures.append("CUDA BF16 support is required but unavailable")
    return failures
