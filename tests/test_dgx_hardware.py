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
            "cuda": {
                "available": True,
                "bf16_supported": True,
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
            "gpu": {"name": "NVIDIA GB10"},
        },
    )


def test_target_hardware_identity_is_explicit_and_unified():
    environment, preflight = _evidence()
    assert validate_dgx_spark_environment(environment, preflight, EXPECTED) == {
        "architecture": "aarch64",
        "gpu_name": "NVIDIA GB10",
        "device_count": 1,
        "compute_capability": [12, 1],
        "unified_memory_bytes": TOTAL,
        "host_device_memory_equal": True,
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
    ],
)
def test_non_target_hardware_fails_closed(mutation):
    environment, preflight = _evidence()
    environment = deepcopy(environment)
    preflight = deepcopy(preflight)
    mutation(environment, preflight)
    with pytest.raises(ValueError):
        validate_dgx_spark_environment(environment, preflight, EXPECTED)
