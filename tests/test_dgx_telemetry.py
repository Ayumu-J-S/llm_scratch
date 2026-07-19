import time

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
