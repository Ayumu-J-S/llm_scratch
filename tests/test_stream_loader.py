from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from omegaconf import OmegaConf

from data.stream_loader import (
    BoundedShardCache,
    RetryPolicy,
    StreamLoader,
)
from tokenizer.canonical import CanonicalTokenizer


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_CONFIG = {
    "manifest_path": "assets/tokenizers/llm-jp-v1/manifest.json",
    "expected_fingerprint": "12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b",
}
TOKENIZER = CanonicalTokenizer.from_config(TOKENIZER_CONFIG)


def configured(config):
    config["tokenizer"] = dict(TOKENIZER_CONFIG)
    return config


def make_loader(config):
    return StreamLoader(configured(config))


def base_config(**overrides):
    config = {
        "output_mode": "tokenized_docs",
        "max_tokens": 12,
        "add_eos": False,
        "seed": 7,
        "datasets": [
            {
                "name": "a",
                "type": "memory",
                "ratio": 0.5,
                "documents": [{"text": "aaaaaa", "id": "a1"}],
            },
            {
                "name": "b",
                "type": "memory",
                "ratio": 0.5,
                "documents": [{"text": "bbbbbb", "id": "b1"}],
            },
        ],
    }
    config.update(overrides)
    return configured(config)


def collect(config):
    return list(make_loader(config))


def test_ratio_validation_rejects_invalid_sum():
    config = base_config()
    config["datasets"][0]["ratio"] = 0.2

    with pytest.raises(ValueError, match="sum to 1.0"):
        make_loader(config)


def test_token_quota_enforced_by_token_count():
    config = base_config(max_tokens=10)
    config["datasets"][0]["ratio"] = 0.7
    config["datasets"][1]["ratio"] = 0.3
    config["datasets"][0]["documents"] = [{"text": "b" * 50}]
    config["datasets"][1]["documents"] = [{"text": "c" * 50}]

    loader = make_loader(config)
    samples = list(loader)

    counts = {}
    for sample in samples:
        counts[sample["source"]] = counts.get(sample["source"], 0) + sample["token_count"]

    assert counts == {"a": 7, "b": 3}
    assert sum(sample["token_count"] for sample in samples) == 10


def test_max_text_chars_truncates_before_tokenization():
    config = {
        "output_mode": "raw_text",
        "max_tokens": "max",
        "add_eos": False,
        "datasets": [
            {
                "name": "truncated",
                "type": "memory",
                "ratio": 1.0,
                "max_text_chars": 3,
                "documents": [{"text": "abcdef"}],
            }
        ],
    }

    sample = collect(config)[0]

    assert sample["text"] == "abc"
    assert sample["token_count"] == len(TOKENIZER.encode("abc"))


@pytest.mark.parametrize("output_mode", ["raw_text", "bytes", "tokenized_docs", "packed_sequences"])
def test_output_modes(output_mode):
    config = {
        "output_mode": output_mode,
        "max_tokens": "max",
        "sequence_length": 3,
        "add_eos": False,
        "datasets": [
            {
                "name": "fixture",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "abcd", "id": "1"}],
            }
        ],
    }

    sample = collect(config)[0]

    if output_mode == "raw_text":
        assert sample["text"] == "abcd"
    elif output_mode == "bytes":
        assert sample["bytes"] == b"abcd"
    else:
        assert sample["input_ids"].dtype == np.uint32
        assert sample["input_ids"].tolist() == TOKENIZER.encode("abcd")


def test_hydra_dictconfig_tokenizer_config_works_with_default_eos():
    config = OmegaConf.create(
        {
            "tokenizer": TOKENIZER_CONFIG,
            "output_mode": "tokenized_docs",
            "max_tokens": len(TOKENIZER.encode("abc")) + 1,
            "datasets": [
                {
                    "name": "hydra",
                    "type": "memory",
                    "ratio": 1.0,
                    "documents": [{"text": "abc"}],
                }
            ],
        }
    )

    sample = collect(config)[0]

    assert sample["input_ids"].tolist() == TOKENIZER.encode("abc") + [TOKENIZER.eos_token_id]


def test_debug_cli_runs_from_repo_root(tmp_path):
    config_path = tmp_path / "debug_loader.yaml"
    config_path.write_text(
        """
tokenizer:
  manifest_path: {manifest_path}
  expected_fingerprint: {fingerprint}
output_mode: raw_text
max_tokens: 5
add_eos: false
prefetch:
  enabled: true
  buffer_size: 2
datasets:
  - name: debug
    type: memory
    ratio: 1.0
    documents:
      - text: "hello hello hello hello hello"
""".format(
            manifest_path=ROOT / TOKENIZER_CONFIG["manifest_path"],
            fingerprint=TOKENIZER_CONFIG["expected_fingerprint"],
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/debug_stream_loader.py",
            "--config",
            str(config_path),
            "--limit",
            "1",
            "--sequence-length",
            "4",
            "--batch-size",
            "1",
        ],
        cwd=Path.cwd(),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "batch=0 batch_size=1 sequence_length=4" in result.stdout
    assert "inputs.shape=(1, 4) labels.shape=(1, 4)" in result.stdout
    assert "shift_check=True" in result.stdout


def test_deterministic_seeded_sampling():
    config = {
        "output_mode": "raw_text",
        "max_tokens": 8,
        "add_eos": False,
        "seed": 123,
        "datasets": [
            {
                "name": "a",
                "type": "memory",
                "ratio": 0.5,
                "documents": [{"text": "aa"}, {"text": "aa"}],
            },
            {
                "name": "b",
                "type": "memory",
                "ratio": 0.5,
                "documents": [{"text": "bb"}, {"text": "bb"}],
            },
        ],
    }

    first = [sample["source"] for sample in collect(config)]
    second = [sample["source"] for sample in collect(config)]
    config["seed"] = 987
    third = [sample["source"] for sample in collect(config)]

    assert first == second
    assert first != third


def test_async_startup_does_not_start_worker_until_iteration():
    config = base_config(
        max_tokens=3,
        prefetch={"enabled": True, "buffer_size": 2},
        datasets=[
            {
                "name": "a",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "aaaaaa"}],
            }
        ],
    )
    loader = make_loader(config)

    assert not loader.is_prefetching
    iterator = iter(loader)
    assert next(iterator)["source"] == "a"
    assert loader.is_prefetching or loader._thread is not None
    list(iterator)
    assert not loader.is_prefetching


def test_hf_prefetch_defaults_to_process_mode():
    config = {
        "tokenizer": TOKENIZER_CONFIG,
        "output_mode": "raw_text",
        "max_tokens": 2,
        "prefetch": {"enabled": True},
        "datasets": [
            {
                "name": "hf",
                "type": "hf",
                "path": "unused/offline",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "ratio": 1.0,
            }
        ],
    }

    loader = StreamLoader(config)

    assert loader.prefetch_mode == "process"


def test_hf_thread_prefetch_is_rejected():
    config = {
        "tokenizer": TOKENIZER_CONFIG,
        "output_mode": "raw_text",
        "max_tokens": 1,
        "prefetch": {"enabled": True, "mode": "thread"},
        "datasets": [
            {
                "name": "hf",
                "type": "hf",
                "path": "unused/offline",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "ratio": 1.0,
            }
        ],
    }

    with pytest.raises(ValueError, match="thread prefetch is unsafe"):
        StreamLoader(config)


def test_process_prefetch_emits_samples():
    config = {
        "tokenizer": TOKENIZER_CONFIG,
        "output_mode": "raw_text",
        "max_tokens": 2,
        "add_eos": False,
        "prefetch": {"enabled": True, "mode": "process", "buffer_size": 2},
        "datasets": [
            {
                "name": "process",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "abc"}],
            }
        ],
    }

    sample = list(StreamLoader(config))[0]

    assert sample["source"] == "process"
    assert sample["text"] == "abc"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX fifo support")
def test_process_prefetch_shutdown_terminates_blocked_worker(tmp_path):
    fifo_path = tmp_path / "blocked.jsonl"
    os.mkfifo(fifo_path)
    config = {
        "tokenizer": TOKENIZER_CONFIG,
        "output_mode": "raw_text",
        "max_tokens": "max",
        "add_eos": False,
        "shutdown_timeout_seconds": 0.2,
        "prefetch": {"enabled": True, "mode": "process", "buffer_size": 1},
        "datasets": [
            {
                "name": "blocked",
                "type": "jsonl",
                "path": str(fifo_path),
                "ratio": 1.0,
            }
        ],
    }
    loader = StreamLoader(config)
    loader._start_prefetch_worker()

    try:
        for _ in range(100):
            if loader.is_prefetching:
                break
            time.sleep(0.01)
        assert loader.is_prefetching
        loader.close()
        assert not loader.is_prefetching
    finally:
        loader.close()


def test_overlapping_prefetch_iterators_are_rejected_after_worker_exit():
    config = {
        "output_mode": "raw_text",
        "max_tokens": 2,
        "add_eos": False,
        "prefetch": {"enabled": True, "buffer_size": 10},
        "datasets": [
            {
                "name": "small",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "x"}],
            }
        ],
    }
    loader = make_loader(config)
    first_iterator = iter(loader)

    try:
        assert next(first_iterator)["text"] == "x"
        for _ in range(100):
            if not loader.is_prefetching:
                break
            time.sleep(0.01)
        assert not loader.is_prefetching

        second_iterator = iter(loader)
        with pytest.raises(RuntimeError, match="already being iterated"):
            next(second_iterator)
    finally:
        loader.close()


def test_constructor_does_not_touch_dataset_iterable():
    calls = {"count": 0}

    def records():
        calls["count"] += 1
        yield {"text": "abc"}

    make_loader(
        {
            "output_mode": "raw_text",
            "max_tokens": 3,
            "add_eos": False,
            "datasets": [
                {
                    "name": "lazy",
                    "type": "iterable",
                    "ratio": 1.0,
                    "iterable": records,
                }
            ],
        }
    )

    assert calls["count"] == 0


def test_hugging_face_dataset_requires_revision():
    config = {
        "output_mode": "raw_text",
        "max_tokens": 1,
        "add_eos": False,
        "datasets": [
            {
                "name": "demo",
                "type": "hf",
                "path": "owner/dataset",
                "split": "train",
                "text_field": "text",
                "ratio": 1.0,
            }
        ],
    }

    with pytest.raises(ValueError, match="hf dataset demo requires revision"):
        make_loader(config)


def test_hugging_face_dataset_rejects_moving_revision():
    config = {
        "output_mode": "raw_text",
        "max_tokens": 1,
        "add_eos": False,
        "datasets": [
            {
                "name": "demo",
                "type": "hf",
                "path": "owner/dataset",
                "revision": "main",
                "split": "train",
                "text_field": "text",
                "ratio": 1.0,
            }
        ],
    }

    with pytest.raises(ValueError, match="40-character commit hash"):
        make_loader(config)


def test_hugging_face_source_passes_revision_to_load_dataset(monkeypatch):
    calls = {}

    def fake_load_dataset(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return [{"text": "abc"}]

    monkeypatch.setitem(sys.modules, "datasets", Mock(load_dataset=fake_load_dataset))
    config = {
        "output_mode": "raw_text",
        "max_tokens": 2,
        "add_eos": False,
        "datasets": [
            {
                "name": "demo",
                "type": "hf",
                "path": "owner/dataset",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "split": "train",
                "text_field": "text",
                "ratio": 1.0,
            }
        ],
    }

    sample = collect(config)[0]

    assert sample["text"] == "abc"
    assert calls["args"] == ("owner/dataset",)
    assert calls["kwargs"]["revision"] == "0123456789abcdef0123456789abcdef01234567"
    assert calls["kwargs"]["streaming"] is True


def test_source_iterators_close_on_early_stop():
    closed = threading.Event()

    def records():
        try:
            yield {"text": "abc"}
            yield {"text": "abc"}
        finally:
            closed.set()

    config = {
        "output_mode": "raw_text",
        "max_tokens": "max",
        "add_eos": False,
        "datasets": [
            {
                "name": "closable",
                "type": "iterable",
                "ratio": 1.0,
                "iterable": records,
            }
        ],
    }

    iterator = iter(make_loader(config))

    assert next(iterator)["text"] == "abc"
    iterator.close()
    assert closed.is_set()


def test_finite_quota_exhaustion_raises():
    config = {
        "output_mode": "raw_text",
        "max_tokens": 10,
        "add_eos": False,
        "datasets": [
            {
                "name": "too_small",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "abc"}],
            }
        ],
    }

    with pytest.raises(RuntimeError, match="quota"):
        collect(config)


def test_sampling_does_not_reencode_untruncated_documents(monkeypatch):
    original_encode = CanonicalTokenizer.encode
    calls = {"count": 0}

    def encode(tokenizer, text):
        calls["count"] += 1
        return original_encode(tokenizer, text)

    monkeypatch.setattr(CanonicalTokenizer, "encode", encode)
    config = {
        "output_mode": "raw_text",
        "max_tokens": 3,
        "add_eos": True,
        "datasets": [
            {
                "name": "docs",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "abc"}],
            }
        ],
    }

    samples = collect(config)

    assert samples[0]["text"] == "abc"
    assert calls["count"] == 1


def test_jsonl_fixture_dataset_preserves_metadata():
    fixture_path = Path("tests/fixtures/tiny_corpus.jsonl")
    config = {
        "output_mode": "raw_text",
        "max_tokens": 5,
        "add_eos": False,
        "preserve_metadata": True,
        "datasets": [
            {
                "name": "fixture",
                "type": "jsonl",
                "path": str(fixture_path),
                "ratio": 1.0,
            }
        ],
    }

    sample = collect(config)[0]

    assert sample["text"] == "alpha"
    assert sample["metadata"]["id"] == "a"


def test_cache_size_enforcement_evicts_released_shards(tmp_path):
    cache = BoundedShardCache(tmp_path, max_size_bytes=6, retry_policy=RetryPolicy(max_attempts=1))

    with cache.acquire("one", lambda path: path.write_bytes(b"1111")) as first_path:
        assert first_path.exists()
    with cache.acquire("two", lambda path: path.write_bytes(b"2222")) as second_path:
        assert second_path.exists()

    assert cache.size_bytes <= 6
    assert not first_path.exists()
    assert second_path.exists()


def test_safe_eviction_keeps_active_shards(tmp_path):
    cache = BoundedShardCache(tmp_path, max_size_bytes=8, retry_policy=RetryPolicy(max_attempts=1))

    with cache.acquire("one", lambda path: path.write_bytes(b"1111")) as first_path:
        with cache.acquire("two", lambda path: path.write_bytes(b"2222")) as second_path:
            assert first_path.exists()
            assert second_path.exists()

    assert cache.size_bytes == 8


def test_cache_blocks_when_no_safe_space_is_available(tmp_path):
    cache = BoundedShardCache(tmp_path, max_size_bytes=5, retry_policy=RetryPolicy(max_attempts=1))
    completed = threading.Event()
    errors = []

    with cache.acquire("one", lambda path: path.write_bytes(b"11111")):
        thread = threading.Thread(
            target=lambda: _fetch_cache_key(cache, "two", b"22", completed, errors)
        )
        thread.start()
        time.sleep(0.2)
        assert not completed.is_set()

    thread.join(timeout=2)
    assert completed.is_set()
    assert not errors


def test_retry_behavior(tmp_path):
    attempts = {"count": 0}
    cache = BoundedShardCache(
        tmp_path,
        max_size_bytes=10,
        retry_policy=RetryPolicy(max_attempts=3, initial_delay_seconds=0.0),
    )

    def flaky_downloader(path):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise OSError("temporary network failure")
        path.write_bytes(b"ok")

    with cache.acquire("flaky", flaky_downloader) as path:
        assert path.read_bytes() == b"ok"

    assert attempts["count"] == 3


def test_worker_shutdown_is_clean_and_deterministic():
    config = {
        "output_mode": "raw_text",
        "max_tokens": "max",
        "add_eos": False,
        "prefetch": {"enabled": True, "buffer_size": 1},
        "datasets": [
            {
                "name": "slow",
                "type": "iterable",
                "ratio": 1.0,
                "iterable": lambda: ({"text": "x"} for _ in range(1000)),
            }
        ],
    }

    loader = make_loader(config)
    iterator = iter(loader)
    assert next(iterator)["text"] == "x"
    loader.close()

    assert not loader.is_prefetching


def test_packed_sequence_boundaries_and_eod_handling():
    config = {
        "output_mode": "packed_sequences",
        "max_tokens": 8,
        "sequence_length": 4,
        "add_eos": True,
        "preserve_metadata": True,
        "datasets": [
            {
                "name": "docs",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "ab"}, {"text": "cd"}, {"text": "ef"}],
            }
        ],
    }

    sequence = collect(config)[0]

    expected = TOKENIZER.encode("ab") + [TOKENIZER.eos_token_id]
    expected += TOKENIZER.encode("cd") + [TOKENIZER.eos_token_id]
    assert sequence["input_ids"].tolist() == expected[:4]
    assert sequence["source_spans"] == [
        {"source": "docs", "start": 0, "end": 3},
        {"source": "docs", "start": 3, "end": 4},
    ]


def test_tokenized_docs_end_of_document_token():
    config = {
        "output_mode": "tokenized_docs",
        "max_tokens": 3,
        "add_eos": True,
        "datasets": [
            {
                "name": "docs",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "ab"}],
            }
        ],
    }

    sample = collect(config)[0]

    assert sample["input_ids"].tolist() == TOKENIZER.encode("ab") + [TOKENIZER.eos_token_id]


def test_tokenizer_round_trip_behavior():
    text = "hello 日本語"

    token_ids = TOKENIZER.encode(text)
    assert TOKENIZER.decode(token_ids) == text


@pytest.mark.skipif(
    os.getenv("RUN_HF_DATASET_INTEGRATION") != "1",
    reason="set RUN_HF_DATASET_INTEGRATION=1 to stream a public HF dataset",
)
def test_hugging_face_streaming_dataset_integration(tmp_path):
    config = {
        "output_mode": "raw_text",
        "max_tokens": 20,
        "add_eos": False,
        "cache": {"dir": str(tmp_path / "hf-cache"), "max_size_bytes": 10_000_000},
        "datasets": [
            {
                "name": "demo",
                "type": "hf",
                "path": "lhoestq/demo1",
                "revision": "87ecf163bedca9d80598b528940a9c4f99e14c11",
                "split": "train",
                "text_field": "review",
                "ratio": 1.0,
            }
        ],
    }

    sample = collect(config)[0]

    assert sample["source"] == "demo"
    assert sample["text"]


def _fetch_cache_key(cache, key, payload, completed, errors):
    try:
        with cache.acquire(key, lambda path: path.write_bytes(payload)):
            completed.set()
    except Exception as error:  # noqa: BLE001 - surfaced in assertion.
        errors.append(error)
        completed.set()
