from __future__ import annotations

from collections import Counter
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
from tokenizers import Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split

from data.stream_loader import (
    BoundedShardCache,
    RetryPolicy,
    StreamLoader,
    StreamLoaderError,
)


CHAR_VOCAB = {
    "<unk>": 0,
    "<eos>": 1,
    "a": 2,
    "b": 3,
    "c": 4,
    "d": 5,
    "e": 6,
    "f": 7,
    "g": 8,
    "h": 9,
    "l": 10,
    "m": 11,
    "o": 12,
    "p": 13,
    "s": 14,
    "t": 15,
    "x": 16,
    " ": 17,
    "日": 18,
    "本": 19,
    "語": 20,
}


def make_tokenizers_tokenizer():
    tokenizer = Tokenizer(WordLevel(CHAR_VOCAB, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")
    tokenizer.decoder = Fuse()
    return tokenizer


def tokenizer_config(tmp_path):
    tokenizer_path = tmp_path / "char_tokenizer.json"
    make_tokenizers_tokenizer().save(str(tokenizer_path))
    return {"kind": "tokenizers", "path": str(tokenizer_path), "eos_token": "<eos>"}


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
    return config


def collect(config, tokenizer=None):
    if tokenizer is None and "tokenizer" not in config:
        tokenizer = make_tokenizers_tokenizer()
    return list(StreamLoader(config, tokenizer=tokenizer))


def test_ratio_validation_rejects_invalid_sum():
    config = base_config()
    config["datasets"][0]["ratio"] = 0.2

    with pytest.raises(ValueError, match="sum to 1.0"):
        StreamLoader(config, tokenizer=make_tokenizers_tokenizer())


def test_token_quota_enforced_by_token_count():
    config = base_config(max_tokens=10)
    config["datasets"][0]["ratio"] = 0.7
    config["datasets"][1]["ratio"] = 0.3
    config["datasets"][0]["documents"] = [{"text": "aaaaaaaaaa"}]
    config["datasets"][1]["documents"] = [{"text": "bbbbbbbbbb"}]

    loader = StreamLoader(config, tokenizer=make_tokenizers_tokenizer())
    samples = list(loader)

    counts = {}
    for sample in samples:
        counts[sample["source"]] = counts.get(sample["source"], 0) + sample["token_count"]

    assert counts == {"a": 7, "b": 3}
    assert sum(sample["token_count"] for sample in samples) == 10


def test_max_text_chars_truncates_before_tokenization():
    config = {
        "output_mode": "raw_text",
        "max_tokens": 3,
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
    assert sample["token_count"] == 3


@pytest.mark.parametrize("output_mode", ["raw_text", "bytes", "tokenized_docs", "packed_sequences"])
def test_output_modes(output_mode):
    config = {
        "output_mode": output_mode,
        "max_tokens": 4,
        "sequence_length": 4,
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
        assert sample["input_ids"].tolist() == [2, 3, 4, 5]
        if output_mode == "packed_sequences":
            assert "token_count" not in sample
            assert sample["window_token_count"] == 4
            assert sample["target_token_count"] == 3


def test_hydra_dictconfig_tokenizer_config_works_with_default_eos(tmp_path):
    config = OmegaConf.create(
        {
            "tokenizer": tokenizer_config(tmp_path),
            "output_mode": "tokenized_docs",
            "max_tokens": 4,
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

    assert sample["input_ids"].tolist() == [2, 3, 4, 1]


def test_debug_cli_runs_from_repo_root(tmp_path):
    config_path = tmp_path / "debug_loader.yaml"
    config_path.write_text(
        """
tokenizer:
  kind: tokenizers
  path: {tokenizer_path}
  eos_token: <eos>
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
      - text: hello
""".format(tokenizer_path=tokenizer_config(tmp_path)["path"]),
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
        max_tokens=6,
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
    loader = StreamLoader(config, tokenizer=make_tokenizers_tokenizer())

    assert not loader.is_prefetching
    iterator = iter(loader)
    assert next(iterator)["source"] == "a"
    assert loader.is_prefetching or loader._thread is not None
    list(iterator)
    assert not loader.is_prefetching


def test_hf_prefetch_defaults_to_process_mode(tmp_path):
    config = {
        "tokenizer": tokenizer_config(tmp_path),
        "output_mode": "raw_text",
        "max_tokens": 1,
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


def test_hf_thread_prefetch_is_rejected(tmp_path):
    config = {
        "tokenizer": tokenizer_config(tmp_path),
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


def test_process_prefetch_emits_samples(tmp_path):
    config = {
        "tokenizer": tokenizer_config(tmp_path),
        "output_mode": "raw_text",
        "max_tokens": 3,
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


def test_process_prefetch_returns_unique_and_packed_accounting(tmp_path):
    config = {
        "tokenizer": tokenizer_config(tmp_path),
        "output_mode": "packed_sequences",
        "max_tokens": 7,
        "sequence_length": 4,
        "add_eos": False,
        "prefetch": {"enabled": True, "mode": "process", "buffer_size": 2},
        "datasets": [
            {
                "name": "process",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "abcdefg"}],
            }
        ],
    }
    loader = StreamLoader(config)

    windows = list(loader)

    assert len(windows) == 2
    assert loader.token_counts == {"process": 7}
    assert loader.packed_token_counts == {
        "window_token_count": 8,
        "target_token_count": 6,
        "dropped_target_count": 0,
    }


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX fifo support")
def test_process_prefetch_shutdown_terminates_blocked_worker(tmp_path):
    fifo_path = tmp_path / "blocked.jsonl"
    os.mkfifo(fifo_path)
    config = {
        "tokenizer": tokenizer_config(tmp_path),
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
        "max_tokens": 1,
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
    loader = StreamLoader(config, tokenizer=make_tokenizers_tokenizer())
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

    StreamLoader(
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
        },
        tokenizer=make_tokenizers_tokenizer(),
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
        StreamLoader(config, tokenizer=make_tokenizers_tokenizer())


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
        StreamLoader(config, tokenizer=make_tokenizers_tokenizer())


def test_hugging_face_source_passes_revision_to_load_dataset(monkeypatch):
    calls = {}

    def fake_load_dataset(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return [{"text": "abc"}]

    monkeypatch.setitem(sys.modules, "datasets", Mock(load_dataset=fake_load_dataset))
    config = {
        "output_mode": "raw_text",
        "max_tokens": 3,
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

    iterator = iter(StreamLoader(config, tokenizer=make_tokenizers_tokenizer()))

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


def test_sampling_does_not_reencode_untruncated_documents():
    tokenizer = make_tokenizers_tokenizer()
    tokenizer.encode = Mock(wraps=tokenizer.encode)
    config = {
        "output_mode": "raw_text",
        "max_tokens": 4,
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

    samples = list(StreamLoader(config, tokenizer=tokenizer))

    assert samples[0]["text"] == "abc"
    assert tokenizer.encode.call_count == 1


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

    loader = StreamLoader(config, tokenizer=make_tokenizers_tokenizer())
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

    assert sequence["input_ids"].tolist() == [2, 3, 1, 4]
    assert sequence["source_spans"] == [
        {"source": "docs", "start": 0, "end": 3},
        {"source": "docs", "start": 3, "end": 4},
    ]


def test_packed_windows_carry_boundary_token_and_preserve_all_transitions():
    config = {
        "output_mode": "packed_sequences",
        "max_tokens": 7,
        "sequence_length": 4,
        "add_eos": False,
        "datasets": [
            {
                "name": "docs",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "abcdefg"}],
            }
        ],
    }

    windows = collect(config)

    assert [window["input_ids"].tolist() for window in windows] == [
        [2, 3, 4, 5],
        [5, 6, 7, 8],
    ]
    assert Counter(
        (left, right)
        for window in windows
        for left, right in zip(window["input_ids"][:-1], window["input_ids"][1:])
    ) == Counter({(2, 3): 1, (3, 4): 1, (4, 5): 1, (5, 6): 1, (6, 7): 1, (7, 8): 1})


@pytest.mark.parametrize("target_length", [2, 3, 4, 5])
def test_packed_transition_multiset_matches_stream_with_repeated_ids(target_length):
    text = ("abacaba" * target_length)[: 1 + 3 * target_length]
    expected_ids = make_tokenizers_tokenizer().encode(text).ids
    config = {
        "output_mode": "packed_sequences",
        "max_tokens": len(expected_ids),
        "sequence_length": target_length + 1,
        "add_eos": False,
        "datasets": [
            {
                "name": "docs",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": text}],
            }
        ],
    }

    actual = Counter(
        (left, right)
        for window in collect(config)
        for left, right in zip(window["input_ids"][:-1], window["input_ids"][1:])
    )

    assert actual == Counter(zip(expected_ids[:-1], expected_ids[1:]))


@pytest.mark.parametrize(
    ("quota", "text", "expected"),
    [
        (3, "ab", [2, 3, 1]),
        (2, "ab", [2, 1]),
        (3, "abcdef", [2, 3, 1]),
        (1, "abcdef", [1]),
    ],
)
def test_finite_quota_reserves_final_slot_for_eos(quota, text, expected):
    config = {
        "output_mode": "tokenized_docs",
        "max_tokens": quota,
        "add_eos": True,
        "datasets": [
            {
                "name": "docs",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": text}],
            }
        ],
    }

    loader = StreamLoader(config, tokenizer=make_tokenizers_tokenizer())
    samples = list(loader)

    assert samples[0]["input_ids"].tolist() == expected
    assert loader.token_counts == {"docs": quota}


def test_packed_quota_truncation_without_eos_fails_explicitly():
    config = {
        "output_mode": "packed_sequences",
        "max_tokens": 3,
        "sequence_length": 4,
        "add_eos": False,
        "datasets": [
            {
                "name": "docs",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "abcdef"}],
            }
        ],
    }

    with pytest.raises(StreamLoaderError, match="no boundary token"):
        collect(config)


def test_packed_source_quotas_remain_exact_when_truncation_adds_eos():
    config = {
        "output_mode": "packed_sequences",
        "max_tokens": 10,
        "sequence_length": 4,
        "add_eos": True,
        "datasets": [
            {
                "name": "a",
                "type": "memory",
                "ratio": 0.7,
                "documents": [{"text": "aaaaaaaaaa"}],
            },
            {
                "name": "b",
                "type": "memory",
                "ratio": 0.3,
                "documents": [{"text": "bbbbbbbbbb"}],
            },
        ],
    }
    loader = StreamLoader(config, tokenizer=make_tokenizers_tokenizer())

    list(loader)

    assert loader.token_counts == {"a": 7, "b": 3}


def test_packed_source_spans_shift_with_stride_and_retain_carried_source():
    config = {
        "output_mode": "packed_sequences",
        "max_tokens": 8,
        "sequence_length": 4,
        "add_eos": True,
        "preserve_metadata": True,
        "seed": 3,
        "datasets": [
            {
                "name": "a",
                "type": "memory",
                "ratio": 0.5,
                "documents": [{"text": "abc"}],
            },
            {
                "name": "b",
                "type": "memory",
                "ratio": 0.5,
                "documents": [{"text": "def"}],
            },
        ],
    }

    windows = collect(config)
    first_source = windows[0]["source_spans"][0]["source"]
    second_source = "b" if first_source == "a" else "a"

    assert windows[0]["source_spans"] == [{"source": first_source, "start": 0, "end": 4}]
    assert windows[1]["source_spans"] == [
        {"source": first_source, "start": 0, "end": 1},
        {"source": second_source, "start": 1, "end": 4},
    ]


@pytest.mark.parametrize(
    ("token_total", "drop_remainder", "expected"),
    [
        (7, True, {"window_token_count": 8, "target_token_count": 6, "dropped_target_count": 0}),
        (8, True, {"window_token_count": 8, "target_token_count": 6, "dropped_target_count": 1}),
        (8, False, {"window_token_count": 10, "target_token_count": 7, "dropped_target_count": 0}),
        (7, False, {"window_token_count": 8, "target_token_count": 6, "dropped_target_count": 0}),
    ],
)
def test_packed_accounting_separates_windows_targets_sources_and_dropped_tail(
    token_total, drop_remainder, expected
):
    config = {
        "output_mode": "packed_sequences",
        "max_tokens": token_total,
        "sequence_length": 4,
        "drop_remainder": drop_remainder,
        "add_eos": False,
        "datasets": [
            {
                "name": "docs",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "abcdefgh"[:token_total]}],
            }
        ],
    }
    loader = StreamLoader(config, tokenizer=make_tokenizers_tokenizer())

    windows = list(loader)

    assert loader.token_counts == {"docs": token_total}
    assert loader.packed_token_counts == expected
    assert sum(window["window_token_count"] for window in windows) == expected["window_token_count"]
    assert sum(window["target_token_count"] for window in windows) == expected["target_token_count"]

    list(loader)
    assert loader.packed_token_counts == expected


def test_bounded_long_document_packing_sanity():
    token_total = 8_193
    config = {
        "output_mode": "packed_sequences",
        "max_tokens": token_total,
        "sequence_length": 65,
        "add_eos": False,
        "datasets": [
            {
                "name": "docs",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "a" * token_total}],
            }
        ],
    }
    loader = StreamLoader(config, tokenizer=make_tokenizers_tokenizer())

    windows = list(loader)

    assert len(windows) == 128
    assert loader.token_counts == {"docs": token_total}
    assert loader.packed_token_counts == {
        "window_token_count": 8_320,
        "target_token_count": 8_192,
        "dropped_target_count": 0,
    }


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

    assert sample["input_ids"].tolist() == [2, 3, 1]


def test_tokenizer_round_trip_behavior():
    tokenizer = make_tokenizers_tokenizer()
    text = "hello 日本語"

    token_ids = tokenizer.encode(text).ids
    assert tokenizer.decode(token_ids, skip_special_tokens=False) == text


def test_builtin_tekken_tokenizer_integration_round_trip():
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tokenizer = MistralTokenizer.v3(is_tekken=True).instruct_tokenizer.tokenizer
    text = "Tokenizer smoke test."

    assert tokenizer.decode(tokenizer.encode(text, bos=False, eos=False)) == text


@pytest.mark.skipif(
    os.getenv("RUN_TOKENIZER_INTEGRATION") != "1",
    reason="set RUN_TOKENIZER_INTEGRATION=1 to download tokenizer artifacts",
)
def test_qwen_tokenizer_integration_round_trip():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", use_fast=True)
    text = "Qwen tokenizer smoke test."

    assert tokenizer.is_fast
    assert tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)) == text


@pytest.mark.skipif(
    os.getenv("RUN_TOKENIZER_INTEGRATION") != "1",
    reason="set RUN_TOKENIZER_INTEGRATION=1 to download tokenizer artifacts",
)
def test_mistral_nemo_tekken_tokenizer_integration_round_trip():
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    tokenizer = MistralTokenizer.from_hf_hub(
        "mistralai/Mistral-Nemo-Base-2407"
    ).instruct_tokenizer.tokenizer
    text = "Tekken tokenizer smoke test."

    assert tokenizer.decode(tokenizer.encode(text, bos=False, eos=False)) == text


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
