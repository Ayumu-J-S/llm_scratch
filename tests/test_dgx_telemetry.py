import time
from pathlib import Path
from threading import Event

import pytest

from dgx.telemetry import TelemetrySampler


def test_sampler_schedules_from_collection_completion_without_catchup(monkeypatch, tmp_path):
    starts = []

    def slow_sample(_path, _additional_disk_paths=()):
        starts.append(time.monotonic())
        time.sleep(0.02)
        return {
            "monotonic_seconds": time.monotonic(),
            "host": {
                "memory_available_bytes": 1_000_000,
                "disk_free_bytes": 1_000_000,
                "swap_in_pages": 0,
                "swap_out_pages": 0,
            },
            "gpu": {"temperature_c": 40.0},
        }

    monkeypatch.setattr("dgx.telemetry.system_sample", slow_sample)
    sampler = TelemetrySampler(tmp_path / "system.jsonl", interval_seconds=0.01)
    sampler.start()
    deadline = time.monotonic() + 1.0
    while sampler.samples < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    sampler.stop()

    assert sampler.samples >= 3
    gaps = [right - left for left, right in zip(starts, starts[1:])]
    assert min(gaps) >= 0.025


def test_hard_disk_violation_arms_main_thread_interrupt(monkeypatch, tmp_path):
    interrupted = Event()

    def low_disk_sample(_path, _additional_disk_paths=()):
        return {
            "monotonic_seconds": time.monotonic(),
            "host": {
                "memory_available_bytes": 1_000_000,
                "disk_free_bytes": 119_000_000_000,
                "swap_in_pages": 0,
                "swap_out_pages": 0,
            },
            "gpu": {"temperature_c": 40.0},
        }

    monkeypatch.setattr("dgx.telemetry.system_sample", low_disk_sample)
    monkeypatch.setattr("dgx.telemetry._thread.interrupt_main", interrupted.set)
    sampler = TelemetrySampler(
        tmp_path / "system.jsonl",
        interval_seconds=0.01,
        hard_limits={"min_free_disk_bytes": 120_000_000_000},
        interrupt_on_violation=True,
    )
    sampler.start()
    assert interrupted.wait(timeout=1.0)
    sampler.stop()
    assert "free disk fell below the hard floor" in sampler.violations


def test_fail_closed_sampler_interrupts_when_resource_sampling_fails(monkeypatch, tmp_path):
    interrupted = Event()

    def unavailable(_path, _additional_disk_paths=()):
        raise OSError("disk telemetry unavailable")

    monkeypatch.setattr("dgx.telemetry.system_sample", unavailable)
    monkeypatch.setattr("dgx.telemetry._thread.interrupt_main", interrupted.set)
    sampler = TelemetrySampler(
        tmp_path / "system.jsonl",
        interval_seconds=0.01,
        interrupt_on_violation=True,
    )
    sampler.start()
    assert interrupted.wait(timeout=1.0)
    with pytest.raises(RuntimeError, match="sample: OSError: disk telemetry unavailable"):
        sampler.stop()
    assert sampler.errors == ["sample: OSError: disk telemetry unavailable"]


def _assert_background_failure(sampler, interrupted: Event, pattern: str) -> None:
    sampler.start()
    assert interrupted.wait(timeout=1.0)
    with pytest.raises(RuntimeError, match=pattern):
        sampler.stop()
    assert sampler._thread is not None and not sampler._thread.is_alive()
    assert sampler.errors


def test_output_open_failure_is_retained_and_fail_closed(monkeypatch, tmp_path):
    interrupted = Event()
    sampler = TelemetrySampler(
        tmp_path / "system.jsonl", interval_seconds=0.01, interrupt_on_violation=True
    )

    def fail_open():
        raise OSError("open evidence failed")

    monkeypatch.setattr(sampler, "_open_output", fail_open)
    monkeypatch.setattr("dgx.telemetry._thread.interrupt_main", interrupted.set)
    _assert_background_failure(sampler, interrupted, "open-or-run: OSError: open evidence failed")


@pytest.mark.parametrize("phase", ["write", "flush"])
def test_mid_record_io_failure_is_retained_and_fail_closed(monkeypatch, tmp_path, phase):
    interrupted = Event()

    class BrokenEvidence:
        def write(self, _value):
            if phase == "write":
                raise OSError("evidence write failed")

        def flush(self):
            if phase == "flush":
                raise OSError("evidence flush failed")

        def fileno(self):
            return 1

        def close(self):
            return None

    sampler = TelemetrySampler(
        tmp_path / "system.jsonl", interval_seconds=0.01, interrupt_on_violation=True
    )
    monkeypatch.setattr(sampler, "_open_output", BrokenEvidence)
    monkeypatch.setattr("dgx.telemetry.system_sample", lambda *_args: {"host": {}, "gpu": {}})
    monkeypatch.setattr("dgx.telemetry._thread.interrupt_main", interrupted.set)
    _assert_background_failure(sampler, interrupted, f"write: OSError: evidence {phase} failed")


def test_malformed_hard_limit_sample_is_retained_and_fail_closed(monkeypatch, tmp_path):
    interrupted = Event()
    malformed = {
        "monotonic_seconds": time.monotonic(),
        "host": {"memory_available_bytes": 1},
        "gpu": {"temperature_c": 40.0},
    }
    monkeypatch.setattr("dgx.telemetry.system_sample", lambda *_args: malformed)
    monkeypatch.setattr("dgx.telemetry._thread.interrupt_main", interrupted.set)
    sampler = TelemetrySampler(
        tmp_path / "system.jsonl", interval_seconds=0.01, interrupt_on_violation=True
    )
    _assert_background_failure(sampler, interrupted, "open-or-run: KeyError")


def test_final_fsync_failure_is_retained_and_fail_closed(monkeypatch, tmp_path):
    interrupted = Event()
    monkeypatch.setattr("dgx.telemetry._thread.interrupt_main", interrupted.set)
    monkeypatch.setattr(
        "dgx.telemetry.os.fsync", lambda _fd: (_ for _ in ()).throw(OSError("fsync failed"))
    )
    sampler = TelemetrySampler(
        tmp_path / "system.jsonl", interval_seconds=0.01, interrupt_on_violation=True
    )
    sampler.start()
    deadline = time.monotonic() + 1.0
    while sampler.samples < 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    with pytest.raises(RuntimeError, match="finalize: OSError: fsync failed"):
        sampler.stop()
    assert interrupted.is_set()


def test_output_close_failure_is_retained_and_fail_closed(monkeypatch, tmp_path):
    interrupted = Event()

    class CloseBrokenEvidence:
        def write(self, _value):
            return None

        def flush(self):
            return None

        def fileno(self):
            return 1

        def close(self):
            raise OSError("close failed")

    sampler = TelemetrySampler(
        tmp_path / "system.jsonl", interval_seconds=0.01, interrupt_on_violation=True
    )
    monkeypatch.setattr(sampler, "_open_output", CloseBrokenEvidence)
    monkeypatch.setattr("dgx.telemetry.os.fsync", lambda _fd: None)
    monkeypatch.setattr(
        "dgx.telemetry.system_sample",
        lambda *_args: {
            "host": {
                "memory_available_bytes": 1,
                "disk_free_bytes": 1,
                "swap_in_pages": 0,
                "swap_out_pages": 0,
            },
            "gpu": {"temperature_c": 40.0},
        },
    )
    monkeypatch.setattr("dgx.telemetry._thread.interrupt_main", interrupted.set)
    sampler.start()
    deadline = time.monotonic() + 1.0
    while sampler.samples < 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    with pytest.raises(RuntimeError, match="close: OSError: close failed"):
        sampler.stop()
    assert interrupted.is_set()


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="requires /dev/full")
def test_dev_full_reproduction_fails_closed_without_silent_thread_death(monkeypatch):
    interrupted = Event()
    monkeypatch.setattr("dgx.telemetry._thread.interrupt_main", interrupted.set)
    monkeypatch.setattr(
        "dgx.telemetry.system_sample",
        lambda *_args: {
            "host": {
                "memory_available_bytes": 1,
                "disk_free_bytes": 1,
                "swap_in_pages": 0,
                "swap_out_pages": 0,
            },
            "gpu": {"temperature_c": 40.0},
        },
    )
    sampler = TelemetrySampler(
        Path("/dev/full"),
        interval_seconds=0.01,
        interrupt_on_violation=True,
    )
    _assert_background_failure(sampler, interrupted, "OSError")
