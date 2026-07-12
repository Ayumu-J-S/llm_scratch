from __future__ import annotations

import json
import time

from data.stream_loader import StreamLoader


TOKENIZER_CONFIG = {
    "manifest_path": "assets/tokenizers/llm-jp-v1/manifest.json",
    "expected_fingerprint": "12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b",
}


def _config(**overrides):
    config = {
        "tokenizer": TOKENIZER_CONFIG,
        "output_mode": "raw_text",
        "max_tokens": 12,
        "add_eos": False,
        "seed": 17,
        "horizon": {
            "repeat": False,
            "shuffle": True,
            "shuffle_buffer_size": 3,
        },
        "datasets": [
            {
                "name": "fixture",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": f"document-{index}"} for index in range(20)],
            }
        ],
    }
    config.update(overrides)
    return config


def _sequence(config):
    return [(sample["source"], sample["text"]) for sample in StreamLoader(config)]


def test_same_seed_and_manifest_reproduce_bounded_shuffle_sequence():
    assert _sequence(_config()) == _sequence(_config())


def test_cursor_resume_reproduces_exact_uninterrupted_suffix():
    uninterrupted = _sequence(_config())
    loader = StreamLoader(_config())
    iterator = iter(loader)
    prefix = [next(iterator) for _ in range(4)]
    cursor = loader.state_dict()
    # The cursor is intentionally JSON-safe so a checkpoint can persist it.
    json.loads(json.dumps(cursor))
    iterator.close()

    resumed = list(StreamLoader({**_config(), "cursor": cursor}))
    assert [(item["source"], item["text"]) for item in prefix] + [
        (item["source"], item["text"]) for item in resumed
    ] == uninterrupted


def test_second_pass_continues_without_repeating_prefix_unless_repeat_enabled():
    loader = StreamLoader(_config())
    first = [(item["source"], item["text"]) for item in loader]
    second = [(item["source"], item["text"]) for item in loader]
    assert second
    assert not set(first[:3]).intersection(second[:3])

    repeated = StreamLoader(
        _config(horizon={"repeat": True, "shuffle": True, "shuffle_buffer_size": 3})
    )
    first_repeated = [(item["source"], item["text"]) for item in repeated]
    second_repeated = [(item["source"], item["text"]) for item in repeated]
    assert first_repeated
    assert second_repeated


def test_prefetch_on_and_off_preserve_sequence():
    baseline = _sequence(_config())
    threaded = _sequence(
        _config(prefetch={"enabled": True, "mode": "thread", "buffer_size": 2})
    )
    assert threaded == baseline


def test_process_prefetch_preserves_sequence_and_cursor():
    baseline = _sequence(_config())
    process = _sequence(
        _config(prefetch={"enabled": True, "mode": "process", "buffer_size": 2})
    )
    assert process == baseline


def test_process_prefetch_interruption_resumes_exact_suffix():
    config = _config(prefetch={"enabled": True, "mode": "process", "buffer_size": 2})
    full = _sequence(config)
    loader = StreamLoader(config)
    iterator = iter(loader)
    prefix = [next(iterator) for _ in range(3)]
    cursor = loader.state_dict()
    iterator.close()
    resumed = list(StreamLoader({**config, "cursor": cursor}))
    assert [(item["source"], item["text"]) for item in prefix] + [
        (item["source"], item["text"]) for item in resumed
    ] == full


def test_thread_prefetch_interruption_does_not_capture_ahead_of_consumer():
    config = _config(prefetch={"enabled": True, "mode": "thread", "buffer_size": 2})
    full = _sequence(config)
    loader = StreamLoader(config)
    iterator = iter(loader)
    prefix = [next(iterator)]
    # Give the producer enough time to fill the queue; the parent's cursor
    # should still refer to the one acknowledged sample above.
    time.sleep(0.05)
    cursor = loader.state_dict()
    iterator.close()
    resumed = list(StreamLoader({**config, "cursor": cursor}))
    assert [(item["source"], item["text"]) for item in prefix] + [
        (item["source"], item["text"]) for item in resumed
    ] == full


def test_thread_prefetch_reuse_retains_completed_pass_cursor():
    config = _config(prefetch={"enabled": True, "mode": "thread", "buffer_size": 2})
    loader = StreamLoader(config)
    first = [(item["source"], item["text"]) for item in loader]
    second = [(item["source"], item["text"]) for item in loader]
    assert first
    assert second
    assert not set(first[:3]).intersection(second[:3])


def test_process_prefetch_load_state_dict_propagates_cursor_to_worker():
    config = _config(prefetch={"enabled": True, "mode": "process", "buffer_size": 2})
    full = _sequence(config)
    sync_loader = StreamLoader({**config, "prefetch": {"enabled": False}})
    iterator = iter(sync_loader)
    prefix = [next(iterator) for _ in range(3)]
    cursor = sync_loader.state_dict()
    iterator.close()

    resumed_loader = StreamLoader(config)
    resumed_loader.load_state_dict(cursor)
    resumed = list(resumed_loader)
    assert [(item["source"], item["text"]) for item in prefix] + [
        (item["source"], item["text"]) for item in resumed
    ] == full


def test_process_prefetch_reuse_continues_completed_pass_cursor():
    config = _config(prefetch={"enabled": True, "mode": "process", "buffer_size": 2})
    loader = StreamLoader(config)
    first = [(item["source"], item["text"]) for item in loader]
    second = [(item["source"], item["text"]) for item in loader]
    assert first
    assert second
    assert not set(first[:3]).intersection(second[:3])


def test_packed_window_cursor_keeps_unemitted_residual_tokens():
    config = _config(
        output_mode="packed_sequences",
        max_tokens=24,
        sequence_length=5,
        add_eos=False,
        horizon={"repeat": False, "shuffle": False},
    )
    uninterrupted = [sample["input_ids"].tolist() for sample in StreamLoader(config)]
    loader = StreamLoader(config)
    iterator = iter(loader)
    prefix = [next(iterator)["input_ids"].tolist()]
    cursor = loader.state_dict()
    iterator.close()
    resumed = [sample["input_ids"].tolist() for sample in StreamLoader({**config, "cursor": cursor})]
    assert prefix + resumed == uninterrupted
