import time
from threading import Event

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
    sampler.stop()
    assert sampler.errors == ["OSError: disk telemetry unavailable"]
