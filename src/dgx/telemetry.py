"""Low-overhead DGX Spark host/GPU telemetry for bounded measurements."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def _integer_file(path: Path, keys: set[str]) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields and fields[0].rstrip(":") in keys:
            multiplier = 1024 if fields[-1:] == ["kB"] else 1
            values[fields[0].rstrip(":")] = int(fields[1]) * multiplier
    return values


def _optional_number(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    return float(value)


def gpu_sample() -> dict[str, Any]:
    query = (
        "name,uuid,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu,utilization.memory"
    )
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"DGX-001 requires exactly one visible GPU, observed {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 7:
        raise RuntimeError(f"unexpected nvidia-smi sample: {rows[0]!r}")
    return {
        "name": fields[0],
        "uuid": fields[1],
        "temperature_c": _optional_number(fields[2]),
        "sm_clock_mhz": _optional_number(fields[3]),
        "power_watts": _optional_number(fields[4]),
        "gpu_utilization_percent": _optional_number(fields[5]),
        "memory_utilization_percent": _optional_number(fields[6]),
    }


def system_sample(path: Path) -> dict[str, Any]:
    memory = _integer_file(
        Path("/proc/meminfo"),
        {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree", "Cached"},
    )
    vmstat = _integer_file(Path("/proc/vmstat"), {"pswpin", "pswpout", "pgfault"})
    process = _integer_file(Path("/proc/self/status"), {"VmRSS", "VmHWM"})
    disk = shutil.disk_usage(path)
    return {
        "wall_time_unix_ns": time.time_ns(),
        "monotonic_seconds": time.monotonic(),
        "host": {
            "memory_total_bytes": memory["MemTotal"],
            "memory_available_bytes": memory["MemAvailable"],
            "swap_total_bytes": memory["SwapTotal"],
            "swap_used_bytes": memory["SwapTotal"] - memory["SwapFree"],
            "cached_bytes": memory["Cached"],
            "swap_in_pages": vmstat["pswpin"],
            "swap_out_pages": vmstat["pswpout"],
            "page_faults": vmstat["pgfault"],
            "process_rss_bytes": process["VmRSS"],
            "process_peak_rss_bytes": process["VmHWM"],
            "disk_free_bytes": disk.free,
        },
        "gpu": gpu_sample(),
    }


class TelemetrySampler:
    """Write timestamped JSONL samples without synchronizing the training loop."""

    def __init__(self, output_path: Path, *, interval_seconds: float = 1.0):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.output_path = Path(output_path)
        self.interval_seconds = float(interval_seconds)
        self.errors: list[str] = []
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("telemetry sampler is already started")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="dgx-telemetry", daemon=False)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(5.0, self.interval_seconds * 2))
            if self._thread.is_alive():
                raise RuntimeError("telemetry sampler did not stop")

    def _run(self) -> None:
        deadline = time.monotonic()
        with self.output_path.open("a", encoding="utf-8") as handle:
            while not self._stop.is_set():
                try:
                    sample = system_sample(self.output_path.parent)
                except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
                    self.errors.append(f"{type(error).__name__}: {error}")
                else:
                    handle.write(json.dumps(sample, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    self.samples += 1
                deadline += self.interval_seconds
                self._stop.wait(max(0.0, deadline - time.monotonic()))
