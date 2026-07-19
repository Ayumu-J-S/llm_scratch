"""Fail-closed identity checks for the DGX Spark measurement target."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def validate_dgx_spark_environment(
    environment: Mapping[str, Any],
    preflight: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the exact GB10 target and its host/device unified-memory alias."""

    architecture = environment.get("architecture")
    if architecture != expected.get("architecture"):
        raise ValueError("DGX measurement host architecture differs from the target")

    cuda = environment.get("cuda")
    if not isinstance(cuda, Mapping) or cuda.get("available") is not True:
        raise ValueError("DGX measurement target lacks CUDA")
    if cuda.get("bf16_supported") is not True:
        raise ValueError("DGX measurement target lacks CUDA BF16")
    device_count = _integer(cuda.get("device_count"), "CUDA device_count")
    if device_count != expected.get("device_count"):
        raise ValueError("DGX measurement target must expose exactly one expected GPU")
    devices = cuda.get("devices")
    if not isinstance(devices, Sequence) or isinstance(devices, (str, bytes)):
        raise ValueError("DGX measurement target lacks CUDA device identity")
    if len(devices) != device_count or len(devices) != 1 or not isinstance(devices[0], Mapping):
        raise ValueError("DGX measurement target has an invalid CUDA device inventory")
    device = devices[0]
    if device.get("index") != 0 or device.get("name") != expected.get("gpu_name"):
        raise ValueError("DGX measurement GPU identity differs from NVIDIA GB10")
    if device.get("compute_capability") != expected.get("compute_capability"):
        raise ValueError("DGX measurement compute capability differs from the GB10 target")

    system_memory = environment.get("system_memory")
    if not isinstance(system_memory, Mapping):
        raise ValueError("DGX measurement target lacks host memory identity")
    host_total = _integer(system_memory.get("total_bytes"), "host total memory")
    device_total = _integer(device.get("total_memory_bytes"), "CUDA total memory")
    minimum = _integer(expected.get("min_unified_memory_bytes"), "minimum unified memory")
    maximum = _integer(expected.get("max_unified_memory_bytes"), "maximum unified memory")
    if not minimum <= host_total <= maximum or not minimum <= device_total <= maximum:
        raise ValueError("DGX measurement memory capacity differs from the 128 GB target")
    if expected.get("require_equal_host_device_memory") is not True:
        raise ValueError("DGX protocol must require equal host/device unified-memory totals")
    if host_total != device_total:
        raise ValueError("DGX host and CUDA totals do not identify one unified-memory pool")

    preflight_host = preflight.get("host")
    preflight_gpu = preflight.get("gpu")
    if not isinstance(preflight_host, Mapping) or not isinstance(preflight_gpu, Mapping):
        raise ValueError("DGX preflight lacks target hardware identity")
    if _integer(preflight_host.get("memory_total_bytes"), "preflight total memory") != host_total:
        raise ValueError("DGX preflight and runtime disagree on unified-memory capacity")
    if preflight_gpu.get("name") != expected.get("gpu_name"):
        raise ValueError("DGX preflight and runtime disagree on GPU identity")

    gpu_uuid = _nonempty_string(preflight_gpu.get("uuid"), "preflight GPU UUID")
    if not gpu_uuid.startswith("GPU-"):
        raise ValueError("DGX preflight GPU UUID is not a stable physical GPU identity")
    driver_version = _nonempty_string(cuda.get("driver_version"), "CUDA driver version")
    runtime_version = _integer(cuda.get("runtime_version"), "CUDA runtime version")
    if runtime_version <= 0:
        raise ValueError("CUDA runtime version must be positive")
    os_identity = _nonempty_string(environment.get("os"), "host OS/kernel identity")
    torch_identity = environment.get("torch")
    if not isinstance(torch_identity, Mapping):
        raise ValueError("DGX measurement target lacks PyTorch runtime identity")
    torch_version = _nonempty_string(torch_identity.get("version"), "PyTorch version")
    compiled_cuda = _nonempty_string(torch_identity.get("compiled_cuda"), "compiled CUDA version")

    return {
        "architecture": architecture,
        "gpu_name": device["name"],
        "gpu_uuid": gpu_uuid,
        "device_count": device_count,
        "compute_capability": list(device["compute_capability"]),
        "unified_memory_bytes": host_total,
        "host_device_memory_equal": True,
        "driver_version": driver_version,
        "cuda_runtime_version": runtime_version,
        "torch_version": torch_version,
        "torch_compiled_cuda": compiled_cuda,
        "os_kernel": os_identity,
    }
