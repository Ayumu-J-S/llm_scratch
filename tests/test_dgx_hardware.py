from __future__ import annotations

from copy import deepcopy

import pytest

from dgx.hardware import validate_dgx_spark_environment


EXPECTED = {
    "architecture": "aarch64",
    "gpu_name": "NVIDIA GB10",
    "device_count": 1,
    "compute_capability": [12, 1],
    "min_unified_memory_bytes": 120_000_000_000,
    "max_unified_memory_bytes": 140_000_000_000,
    "require_equal_host_device_memory": True,
}
TOTAL = 130_596_048_896


def _evidence() -> tuple[dict, dict]:
    return (
        {
            "architecture": "aarch64",
            "os": "Linux-6.17.0-1021-nvidia-aarch64-with-glibc2.39",
            "torch": {"version": "2.13.0", "compiled_cuda": "13.3"},
            "cuda": {
                "available": True,
                "bf16_supported": True,
                "runtime_version": 13030,
                "driver_version": "580.159.03",
                "device_count": 1,
                "devices": [
                    {
                        "index": 0,
                        "name": "NVIDIA GB10",
                        "compute_capability": [12, 1],
                        "total_memory_bytes": TOTAL,
                    }
                ],
            },
            "system_memory": {"total_bytes": TOTAL},
        },
        {
            "host": {"memory_total_bytes": TOTAL},
            "gpu": {"name": "NVIDIA GB10", "uuid": "GPU-fixture-uuid"},
        },
    )


def test_target_hardware_identity_is_explicit_and_unified():
    environment, preflight = _evidence()
    assert validate_dgx_spark_environment(environment, preflight, EXPECTED) == {
        "architecture": "aarch64",
        "gpu_name": "NVIDIA GB10",
        "gpu_uuid": "GPU-fixture-uuid",
        "device_count": 1,
        "compute_capability": [12, 1],
        "unified_memory_bytes": TOTAL,
        "host_device_memory_equal": True,
        "driver_version": "580.159.03",
        "cuda_runtime_version": 13030,
        "torch_version": "2.13.0",
        "torch_compiled_cuda": "13.3",
        "os_kernel": "Linux-6.17.0-1021-nvidia-aarch64-with-glibc2.39",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda environment, _preflight: environment.update(architecture="x86_64"),
        lambda environment, _preflight: environment["cuda"].update(device_count=2),
        lambda environment, _preflight: environment["cuda"]["devices"][0].update(
            name="NVIDIA H100"
        ),
        lambda environment, _preflight: environment["cuda"]["devices"][0].update(
            compute_capability=[9, 0]
        ),
        lambda environment, _preflight: environment["cuda"]["devices"][0].update(
            total_memory_bytes=80_000_000_000
        ),
        lambda environment, _preflight: environment["system_memory"].update(
            total_bytes=129_000_000_000
        ),
        lambda _environment, preflight: preflight["gpu"].update(name="NVIDIA H100"),
        lambda _environment, preflight: preflight["gpu"].update(uuid=""),
        lambda environment, _preflight: environment["cuda"].update(driver_version=None),
        lambda environment, _preflight: environment.update(os=""),
    ],
)
def test_non_target_hardware_fails_closed(mutation):
    environment, preflight = _evidence()
    environment = deepcopy(environment)
    preflight = deepcopy(preflight)
    mutation(environment, preflight)
    with pytest.raises(ValueError):
        validate_dgx_spark_environment(environment, preflight, EXPECTED)
