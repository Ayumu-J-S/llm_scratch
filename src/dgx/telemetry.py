"""Low-overhead DGX Spark host/GPU telemetry for bounded measurements."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import _thread
from collections.abc import Mapping
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


def system_sample(path: Path, additional_disk_paths: tuple[Path, ...] = ()) -> dict[str, Any]:
    memory = _integer_file(
        Path("/proc/meminfo"),
        {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree", "Cached"},
    )
    vmstat = _integer_file(Path("/proc/vmstat"), {"pswpin", "pswpout", "pgfault"})
    process = _integer_file(Path("/proc/self/status"), {"VmRSS", "VmHWM"})
    process_io = _integer_file(
        Path("/proc/self/io"), {"read_bytes", "write_bytes", "rchar", "wchar"}
    )
    load_fields = Path("/proc/loadavg").read_text(encoding="utf-8").split()
    process_stat = Path("/proc/self/stat").read_text(encoding="utf-8")
    process_tail = process_stat[process_stat.rfind(")") + 2 :].split()
    disk_read_sectors = 0
    disk_written_sectors = 0
    disk_io_milliseconds = 0
    for line in Path("/proc/diskstats").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 14 and not fields[2].startswith(("loop", "ram")):
            disk_read_sectors += int(fields[5])
            disk_written_sectors += int(fields[9])
            disk_io_milliseconds += int(fields[12])
    network_rx_bytes = 0
    network_tx_bytes = 0
    for line in Path("/proc/net/dev").read_text(encoding="utf-8").splitlines()[2:]:
        interface, values = line.split(":", 1)
        fields = values.split()
        if interface.strip() != "lo":
            network_rx_bytes += int(fields[0])
            network_tx_bytes += int(fields[8])
    disk_paths = (Path(path), *additional_disk_paths)
    disk_by_device = {}
    for disk_path in disk_paths:
        device = disk_path.stat().st_dev
        if device not in disk_by_device:
            disk_by_device[device] = {
                "device": device,
                "path": str(disk_path),
                "free_bytes": shutil.disk_usage(disk_path).free,
            }
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
            "process_user_ticks": int(process_tail[11]),
            "process_system_ticks": int(process_tail[12]),
            "process_read_bytes": process_io["read_bytes"],
            "process_write_bytes": process_io["write_bytes"],
            "process_read_characters": process_io["rchar"],
            "process_write_characters": process_io["wchar"],
            "load_1m": float(load_fields[0]),
            "load_5m": float(load_fields[1]),
            "load_15m": float(load_fields[2]),
            "disk_read_sectors": disk_read_sectors,
            "disk_written_sectors": disk_written_sectors,
            "disk_io_milliseconds": disk_io_milliseconds,
            "network_rx_bytes": network_rx_bytes,
            "network_tx_bytes": network_tx_bytes,
            "disk_free_bytes": min(item["free_bytes"] for item in disk_by_device.values()),
            "disk_free_by_device": sorted(
                disk_by_device.values(), key=lambda item: (item["device"], item["path"])
            ),
        },
        "gpu": gpu_sample(),
    }


class TelemetrySampler:
    """Write timestamped JSONL samples without synchronizing the training loop."""

    def __init__(
        self,
        output_path: Path,
        *,
        interval_seconds: float = 1.0,
        hard_limits: Mapping[str, int | float] | None = None,
        interrupt_on_violation: bool = False,
        additional_disk_paths: tuple[Path, ...] = (),
    ):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.output_path = Path(output_path)
        self.interval_seconds = float(interval_seconds)
        self.errors: list[str] = []
        self.violations: list[str] = []
        self.samples = 0
        self.hard_limits = dict(hard_limits or {})
        self.interrupt_on_violation = bool(interrupt_on_violation)
        self.additional_disk_paths = tuple(Path(path) for path in additional_disk_paths)
        self._initial_swap_in_pages: int | None = None
        self._initial_swap_out_pages: int | None = None
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
                    sample = system_sample(self.output_path.parent, self.additional_disk_paths)
                except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
                    self.errors.append(f"{type(error).__name__}: {error}")
                else:
                    handle.write(json.dumps(sample, sort_keys=True) + "\n")
                    handle.flush()
                    self.samples += 1
                    violations = self._hard_limit_violations(sample)
                    if violations:
                        self.violations.extend(violations)
                        if self.interrupt_on_violation:
                            _thread.interrupt_main()
                            self._stop.set()
                # Schedule from collection completion so an overrun never causes
                # immediate catch-up samples that masquerade as steady cadence.
                deadline = time.monotonic() + self.interval_seconds
                self._stop.wait(max(0.0, deadline - time.monotonic()))
            handle.flush()
            os.fsync(handle.fileno())

    def _hard_limit_violations(self, sample: Mapping[str, Any]) -> list[str]:
        host = sample["host"]
        gpu = sample["gpu"]
        if self._initial_swap_in_pages is None:
            self._initial_swap_in_pages = int(host["swap_in_pages"])
            self._initial_swap_out_pages = int(host["swap_out_pages"])
        violations: list[str] = []
        if int(host["memory_available_bytes"]) < int(
            self.hard_limits.get("min_available_memory_bytes", 0)
        ):
            violations.append("available UMA fell below the hard floor")
        if int(host["disk_free_bytes"]) < int(self.hard_limits.get("min_free_disk_bytes", 0)):
            violations.append("free disk fell below the hard floor")
        temperature = gpu.get("temperature_c")
        if temperature is None or float(temperature) > float(
            self.hard_limits.get("max_temperature_c", float("inf"))
        ):
            violations.append("GPU temperature was unavailable or above the hard ceiling")
        if int(host["swap_in_pages"]) - int(self._initial_swap_in_pages) > int(
            self.hard_limits.get("max_swap_in_pages", 0)
        ):
            violations.append("swap-in exceeded the hard ceiling")
        if int(host["swap_out_pages"]) - int(self._initial_swap_out_pages or 0) > int(
            self.hard_limits.get("max_swap_out_pages", 0)
        ):
            violations.append("swap-out exceeded the hard ceiling")
        return violations
