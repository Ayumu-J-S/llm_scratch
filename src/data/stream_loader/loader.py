from __future__ import annotations

import copy
import json
import math
import multiprocessing as mp
import random
import queue
import re
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import numpy as np

from data.manifests import ResolvedManifest, preflight_manifest
from data.splits import DataPurpose
from data.stream_loader.cache import BoundedShardCache, RetryPolicy, download_url_to_path
from tokenizer.canonical import CanonicalTokenizer


OUTPUT_MODES = {"raw_text", "bytes", "tokenized_docs", "packed_sequences"}
_SENTINEL = "__stream_loader_prefetch_done__"
_ERROR_MARKER = "__stream_loader_prefetch_error__"
_ACCOUNTING_MARKER = "__stream_loader_prefetch_accounting__"
_CURSOR_MARKER = "__stream_loader_prefetch_cursor__"
_HF_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)


class StreamLoaderError(RuntimeError):
    pass


@dataclass
class RawDocument:
    text: str
    metadata: dict[str, Any]


@dataclass
class TokenizedSample:
    source: str
    text: str
    token_ids: list[int]
    metadata: dict[str, Any]

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


@dataclass
class SourceState:
    name: str
    ratio: float
    iterator: Iterator[RawDocument]
    quota: int | None
    dataset: Mapping[str, Any] | None = None
    emitted_tokens: int = 0
    exhausted: bool = False
    # Number of raw documents consumed from the source.  This is distinct from
    # emitted token accounting because a bounded shuffle buffer may contain
    # documents which have already been read but not yielded.
    source_position: int = 0
    epoch: int = 0
    shuffle_buffer: list[RawDocument] | None = None
    shuffle_rng: random.Random | None = None

    @property
    def remaining_quota(self) -> int | None:
        if self.quota is None:
            return None
        return self.quota - self.emitted_tokens


class MemoryTextSource:
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.documents = cfg.get("documents", [])
        self.text_field = cfg.get("text_field", "text")
        self.metadata_fields = cfg.get("metadata_fields")
        self.max_text_chars = _optional_positive_int(cfg.get("max_text_chars"))

    def __iter__(self) -> Iterator[RawDocument]:
        for index, item in enumerate(self.documents):
            yield _record_to_document(
                item,
                text_field=self.text_field,
                metadata_fields=self.metadata_fields,
                max_text_chars=self.max_text_chars,
                fallback_index=index,
            )


class IterableTextSource:
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        iterable = cfg.get("iterable")
        if iterable is None:
            raise ValueError("iterable source requires iterable")
        self.iterable = iterable
        self.text_field = cfg.get("text_field", "text")
        self.metadata_fields = cfg.get("metadata_fields")
        self.max_text_chars = _optional_positive_int(cfg.get("max_text_chars"))

    def __iter__(self) -> Iterator[RawDocument]:
        records = self.iterable() if callable(self.iterable) else self.iterable
        for index, item in enumerate(records):
            yield _record_to_document(
                item,
                text_field=self.text_field,
                metadata_fields=self.metadata_fields,
                max_text_chars=self.max_text_chars,
                fallback_index=index,
            )


class ManifestTextSource:
    def __init__(self, manifest: ResolvedManifest) -> None:
        self.manifest = manifest

    def __iter__(self) -> Iterator[RawDocument]:
        for document in self.manifest.documents:
            yield RawDocument(
                text=document.text,
                metadata=document.metadata(self.manifest),
            )


class JsonlTextSource:
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        path = cfg.get("path")
        if path is None:
            raise ValueError("jsonl source requires path")
        self.path = Path(path)
        self.text_field = cfg.get("text_field", "text")
        self.metadata_fields = cfg.get("metadata_fields")
        self.max_text_chars = _optional_positive_int(cfg.get("max_text_chars"))

    def __iter__(self) -> Iterator[RawDocument]:
        with self.path.open("r", encoding="utf-8") as file:
            for index, line in enumerate(file):
                if not line.strip():
                    continue
                record = json.loads(line)
                yield _record_to_document(
                    record,
                    text_field=self.text_field,
                    metadata_fields=self.metadata_fields,
                    max_text_chars=self.max_text_chars,
                    fallback_index=index,
                )


class UrlJsonlTextSource:
    def __init__(
        self,
        cfg: Mapping[str, Any],
        cache: BoundedShardCache,
        retry_policy: RetryPolicy,
    ) -> None:
        url = cfg.get("url")
        if url is None:
            raise ValueError("url_jsonl source requires url")
        self.url = str(url)
        self.cache = cache
        self.retry_policy = retry_policy
        self.text_field = cfg.get("text_field", "text")
        self.metadata_fields = cfg.get("metadata_fields")
        self.max_text_chars = _optional_positive_int(cfg.get("max_text_chars"))
        self.timeout_seconds = float(cfg.get("timeout_seconds", 30.0))

    def __iter__(self) -> Iterator[RawDocument]:
        def downloader(tmp_path: Path) -> None:
            download_url_to_path(
                self.url,
                tmp_path,
                timeout_seconds=self.timeout_seconds,
            )

        with self.cache.acquire(self.url, downloader) as path:
            with path.open("r", encoding="utf-8") as file:
                for index, line in enumerate(file):
                    if not line.strip():
                        continue
                    yield _record_to_document(
                        json.loads(line),
                        text_field=self.text_field,
                        metadata_fields=self.metadata_fields,
                        max_text_chars=self.max_text_chars,
                        fallback_index=index,
                    )


class HuggingFaceTextSource:
    def __init__(self, cfg: Mapping[str, Any], cache_dir: str | None) -> None:
        path = cfg.get("path")
        if path is None:
            raise ValueError("hf source requires path")
        self.path = str(path)
        revision = cfg.get("revision")
        if revision is None:
            raise ValueError("hf source requires revision")
        self.revision = str(revision)
        if _HF_COMMIT_PATTERN.fullmatch(self.revision) is None:
            raise ValueError("hf source revision must be a 40-character commit hash")
        self.config_name = cfg.get("config_name")
        self.split = cfg.get("split", "train")
        self.data_files = cfg.get("data_files")
        self.text_field = cfg.get("text_field", "text")
        self.metadata_fields = cfg.get("metadata_fields")
        self.max_text_chars = _optional_positive_int(cfg.get("max_text_chars"))
        self.cache_dir = cache_dir
        self.trust_remote_code = bool(cfg.get("trust_remote_code", False))

    def __iter__(self) -> Iterator[RawDocument]:
        from datasets import load_dataset

        dataset = load_dataset(
            self.path,
            name=self.config_name,
            split=self.split,
            data_files=self.data_files,
            revision=self.revision,
            streaming=True,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
        )
        iterator = iter(dataset)
        try:
            for index, record in enumerate(iterator):
                yield _record_to_document(
                    record,
                    text_field=self.text_field,
                    metadata_fields=self.metadata_fields,
                    max_text_chars=self.max_text_chars,
                    fallback_index=index,
                )
        finally:
            _close_iterator(iterator)


class StreamLoader:
    def __init__(
        self,
        config: Mapping[str, Any],
        resolved_manifests: Mapping[str, ResolvedManifest] | None = None,
        cursor: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = dict(config)
        horizon = self.config.get("horizon", {})
        if horizon is None:
            horizon = {}
        if not isinstance(horizon, Mapping):
            raise ValueError("horizon must be a mapping when set")
        self.horizon_config = dict(horizon)
        self.output_mode = self.config.get("output_mode", "raw_text")
        if self.output_mode not in OUTPUT_MODES:
            raise ValueError(f"Unsupported output_mode: {self.output_mode}")

        self.seed = int(self.config.get("seed", 0))
        self.max_tokens = self.config.get("max_tokens", "max")
        if self.max_tokens == "max" and "max_tokens" in self.horizon_config:
            self.max_tokens = self.horizon_config["max_tokens"]
        if self.max_tokens != "max":
            self.max_tokens = int(self.max_tokens)
            if self.max_tokens < 1:
                raise ValueError("max_tokens must be positive or 'max'")

        self.cursor_enabled = bool(
            cursor is not None
            or self.config.get("cursor") is not None
            or self.horizon_config
            or "repeat" in self.config
            or "shuffle" in self.config
            or "shuffle_buffer_size" in self.config
        )
        self._explicit_repeat = "repeat" in self.config or "repeat" in self.horizon_config
        self.repeat = bool(
            self.config.get(
                "repeat",
                self.horizon_config.get("repeat", False if self.cursor_enabled else True),
            )
        )
        shuffle_config = self.config.get("shuffle", self.horizon_config.get("shuffle", False))
        if isinstance(shuffle_config, Mapping):
            self.shuffle = bool(shuffle_config.get("enabled", True))
            configured_buffer = shuffle_config.get("buffer_size")
        else:
            self.shuffle = bool(shuffle_config)
            configured_buffer = None
        configured_buffer = self.config.get(
            "shuffle_buffer_size",
            self.horizon_config.get("shuffle_buffer_size", configured_buffer),
        )
        self.shuffle_buffer_size = int(configured_buffer if configured_buffer is not None else 1)
        if self.shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be positive")
        if self.shuffle and self.shuffle_buffer_size < 2:
            # A buffer of one is an explicitly sequential policy, not shuffle.
            self.shuffle = False

        supplied_cursor = cursor if cursor is not None else self.config.get("cursor")
        self._cursor: dict[str, Any] | None = (
            _copy_cursor(supplied_cursor) if supplied_cursor is not None else None
        )
        # A supplied cursor resumes the suffix on this object's first iterator.
        # Later iterators on the same loader begin an explicit next pass.
        self._resume_cursor_pending = bool(
            self.config.get("_stream_loader_resume_cursor_pending", supplied_cursor is not None)
        )
        self._pass_index = int(self._cursor.get("pass_index", 0)) if self._cursor else 0
        if supplied_cursor is not None:
            self.config["cursor"] = _copy_cursor(supplied_cursor)
        self._active_source_states: list[SourceState] | None = None
        self._active_rng: random.Random | None = None
        self._consumer_cursor: dict[str, Any] | None = None

        self.sequence_length = int(self.config.get("sequence_length", 4096))
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")

        self.drop_remainder = bool(self.config.get("drop_remainder", True))
        self.add_eos = bool(self.config.get("add_eos", True))
        self.preserve_metadata = bool(self.config.get("preserve_metadata", False))
        self.shutdown_timeout_seconds = float(self.config.get("shutdown_timeout_seconds", 5.0))

        self.tokenizer = CanonicalTokenizer.from_config(self.config.get("tokenizer"))
        retry_config = self.config.get("retry", {})
        self.retry_policy = RetryPolicy(
            max_attempts=int(retry_config.get("max_attempts", 3)),
            initial_delay_seconds=float(retry_config.get("initial_delay_seconds", 0.25)),
            max_delay_seconds=float(retry_config.get("max_delay_seconds", 5.0)),
            multiplier=float(retry_config.get("multiplier", 2.0)),
        )

        cache_config = self.config.get("cache", {})
        self.cache_dir = cache_config.get("dir")
        self.cache: BoundedShardCache | None = None
        if self.cache_dir is not None:
            self.cache = BoundedShardCache(
                self.cache_dir,
                max_size_bytes=int(cache_config.get("max_size_bytes", 1 << 30)),
                retry_policy=self.retry_policy,
            )

        self.dataset_configs = [dict(item) for item in self.config.get("datasets", [])]
        self._validate_dataset_configs()
        if self._cursor is not None:
            self._validate_cursor(self._cursor)
        if resolved_manifests is None:
            self.resolved_manifests = _preflight_manifest_datasets(
                self.dataset_configs,
                cache=self.cache,
            )
        else:
            self.resolved_manifests = dict(resolved_manifests)
            _validate_resolved_manifest_inputs(self.dataset_configs, self.resolved_manifests)

        prefetch_config = self.config.get("prefetch", {})
        self.prefetch_enabled = bool(prefetch_config.get("enabled", False))
        self.prefetch_buffer_size = int(prefetch_config.get("buffer_size", 8))
        if self.prefetch_enabled and self.prefetch_buffer_size < 1:
            raise ValueError("prefetch.buffer_size must be positive")
        default_prefetch_mode = "process" if _has_hf_datasets(self.dataset_configs) else "thread"
        self.prefetch_mode = str(prefetch_config.get("mode", default_prefetch_mode))
        if self.prefetch_mode not in {"thread", "process"}:
            raise ValueError("prefetch.mode must be 'thread' or 'process'")
        if self.prefetch_mode == "process" and self.resolved_manifests:
            raise ValueError("manifest sources are preflighted once and require thread prefetch")
        if (
            self.prefetch_enabled
            and self.prefetch_mode == "thread"
            and _has_hf_datasets(self.dataset_configs)
        ):
            raise ValueError("thread prefetch is unsafe for hf datasets; use process prefetch")
        self._thread: threading.Thread | None = None
        self._process: mp.Process | None = None
        self._queue: queue.Queue[Any] | None = None
        self._stop_event: Any | None = None
        self._reset_accounting()

    def _reset_accounting(self) -> None:
        self.token_counts: dict[str, int] = {dataset["name"]: 0 for dataset in self.dataset_configs}
        self.packed_token_counts: dict[str, int] = {
            "window_token_count": 0,
            "target_token_count": 0,
            "dropped_target_count": 0,
        }

    def _source_rng(self, name: str, epoch: int) -> random.Random:
        # Stable arithmetic avoids Python's process-randomized hash() and keeps
        # process/thread prefetch order identical.
        source_hash = 0
        for byte in name.encode("utf-8"):
            source_hash = (source_hash * 257 + byte) & 0xFFFFFFFF
        rng = random.Random((self.seed * 1_000_003 + source_hash + epoch * 65_537) & 0xFFFFFFFF)
        return rng

    def _start_next_pass(self) -> None:
        self._pass_index = int(self._cursor.get("pass_index", 0)) + 1
        next_rng = random.Random(self.seed + self._pass_index)
        source_states = {}
        for dataset in self.dataset_configs:
            name = dataset["name"]
            previous = dict(self._cursor.get("source_states", {}).get(name, {}))
            if self.repeat:
                previous.update(
                    {
                        "source_position": 0,
                        "epoch": self._pass_index,
                        "emitted_tokens": 0,
                        "exhausted": False,
                        "shuffle_buffer": [],
                        "shuffle_rng_state": _jsonable_rng_state(
                            self._source_rng(name, self._pass_index).getstate()
                        ),
                    }
                )
            else:
                previous["emitted_tokens"] = 0
            source_states[name] = previous
        self._cursor = {
            "version": 1,
            "seed": self.seed,
            "pass_index": self._pass_index,
            "output_mode": self.output_mode,
            "dataset_names": [dataset["name"] for dataset in self.dataset_configs],
            "rng_state": _jsonable_rng_state(next_rng.getstate()),
            "source_states": source_states,
            "packed_buffer": [],
            "packed_spans": [],
            "pass_complete": False,
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.prefetch_enabled:
            return self._iter_async()
        return self._iter_sync()

    def state_dict(self) -> dict[str, Any]:
        """Return a serializable cursor for exact stream continuation.

        The cursor is updated before every yielded sample, so callers may
        interrupt an iterator and persist this state without consuming an
        additional document.  It intentionally contains only source position,
        shuffle buffers, and RNG state; model/checkpoint state belongs to the
        trainer checkpoint contract.
        """

        if self.prefetch_enabled and self._consumer_cursor is not None:
            return _copy_cursor(self._consumer_cursor)
        if (
            not self.prefetch_enabled
            and self._active_source_states is not None
            and self._active_rng is not None
        ):
            self._cursor = self._capture_cursor(
                self._active_source_states,
                self._active_rng,
            )
        return _copy_cursor(self._cursor or self._initial_cursor())

    @property
    def cursor_state(self) -> dict[str, Any]:
        return self.state_dict()

    def load_state_dict(
        self,
        cursor: Mapping[str, Any],
        *,
        resume_completed: bool = True,
    ) -> None:
        """Install a cursor before iteration.

        A normal interrupted cursor always resumes its exact unconsumed suffix.
        ``resume_completed=False`` instead treats a terminal cursor as an
        epoch boundary and starts the deterministic next pass on first
        iteration. This is used by full-state checkpoints captured after a
        natural trainer epoch; it never changes mid-pass cursor semantics.
        """

        if self.is_prefetching:
            raise StreamLoaderError("cannot load a cursor while prefetch is active")
        self._validate_cursor(cursor)
        self._cursor = _copy_cursor(cursor)
        self._pass_index = int(self._cursor.get("pass_index", 0))
        self.cursor_enabled = True
        self._resume_cursor_pending = not (
            bool(self._cursor.get("pass_complete")) and not resume_completed
        )
        # Process prefetch receives a plain serialized config.  Keep the
        # explicit cursor in that config as well as on the parent so a spawned
        # worker resumes from the requested state rather than its initial
        # position.
        self.config["cursor"] = _copy_cursor(cursor)
        self.config["_stream_loader_resume_cursor_pending"] = self._resume_cursor_pending

    def _initial_cursor(self) -> dict[str, Any]:
        return {
            "version": 1,
            "seed": self.seed,
            "pass_index": 0,
            "output_mode": self.output_mode,
            "dataset_names": [dataset["name"] for dataset in self.dataset_configs],
            "rng_state": _jsonable_rng_state(random.Random(self.seed).getstate()),
            "source_states": {
                dataset["name"]: {
                    "source_position": 0,
                    "epoch": 0,
                    "emitted_tokens": 0,
                    "exhausted": False,
                    "shuffle_buffer": [],
                    "shuffle_rng_state": _jsonable_rng_state(
                        self._source_rng(dataset["name"], 0).getstate()
                    ),
                }
                for dataset in self.dataset_configs
            },
            "packed_buffer": [],
            "packed_spans": [],
        }

    def _capture_cursor(
        self,
        source_states: list[SourceState],
        rng: random.Random,
    ) -> dict[str, Any]:
        cursor = {
            "version": 1,
            "seed": self.seed,
            "pass_index": int(getattr(self, "_pass_index", 0)),
            "output_mode": self.output_mode,
            "dataset_names": [dataset["name"] for dataset in self.dataset_configs],
            "rng_state": _jsonable_rng_state(rng.getstate()),
            "source_states": {
                state.name: {
                    "source_position": state.source_position,
                    "epoch": state.epoch,
                    "emitted_tokens": state.emitted_tokens,
                    "exhausted": state.exhausted,
                    "shuffle_buffer": [
                        _serialize_document(document) for document in (state.shuffle_buffer or [])
                    ],
                    "shuffle_rng_state": _jsonable_rng_state(
                        (state.shuffle_rng or self._source_rng(state.name, state.epoch)).getstate()
                    ),
                }
                for state in source_states
            },
        }
        if hasattr(self, "_packed_cursor_buffer"):
            cursor["packed_buffer"] = list(self._packed_cursor_buffer)
            cursor["packed_spans"] = copy.deepcopy(getattr(self, "_packed_cursor_spans", []))
        return cursor

    def _producer_cursor_state(self) -> dict[str, Any]:
        """Capture the worker's cursor without reading consumer acknowledgements."""

        if self._active_source_states is not None and self._active_rng is not None:
            return self._capture_cursor(self._active_source_states, self._active_rng)
        return _copy_cursor(self._cursor or self._initial_cursor())

    def _validate_cursor(self, cursor: Mapping[str, Any]) -> None:
        if int(cursor.get("version", -1)) != 1:
            raise ValueError("unsupported stream cursor version")
        if int(cursor.get("seed", self.seed)) != self.seed:
            raise ValueError("stream cursor seed does not match loader seed")
        names = list(cursor.get("dataset_names", []))
        expected = [dataset["name"] for dataset in self.dataset_configs]
        if names != expected:
            raise ValueError("stream cursor dataset names do not match loader datasets")
        if not isinstance(cursor.get("source_states"), Mapping):
            raise ValueError("stream cursor source_states must be a mapping")

    def close(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.shutdown_timeout_seconds)
            if self._thread.is_alive():
                raise StreamLoaderError("prefetch worker did not shut down")
        if self._process is not None:
            self._process.join(timeout=self.shutdown_timeout_seconds)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=self.shutdown_timeout_seconds)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=self.shutdown_timeout_seconds)
            if self._process.is_alive():
                raise StreamLoaderError("prefetch process did not shut down")
        if self._queue is not None and hasattr(self._queue, "close"):
            self._queue.close()
        self._thread = None
        self._process = None
        self._queue = None
        self._stop_event = None

    def __enter__(self) -> StreamLoader:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @property
    def is_prefetching(self) -> bool:
        return (self._thread is not None and self._thread.is_alive()) or (
            self._process is not None and self._process.is_alive()
        )

    def _iter_sync(self) -> Iterator[dict[str, Any]]:
        yield from self._output_iter(stop_event=None)

    def _iter_async(self) -> Iterator[dict[str, Any]]:
        self._start_prefetch_worker()
        assert self._queue is not None
        normal_completion = False
        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if self._process is not None and not self._process.is_alive():
                        if self._process.exitcode not in {0, None}:
                            raise StreamLoaderError(
                                f"prefetch process exited with code {self._process.exitcode}"
                            )
                        break
                    continue
                if item == _SENTINEL:
                    normal_completion = True
                    break
                if _is_prefetch_error(item):
                    raise StreamLoaderError(item["message"])
                if _is_prefetch_accounting(item):
                    self.token_counts = dict(item["token_counts"])
                    self.packed_token_counts = dict(item["packed_token_counts"])
                    continue
                if _is_prefetch_cursor(item):
                    self._cursor = _copy_cursor(item["cursor"])
                    self._consumer_cursor = _copy_cursor(item["cursor"])
                    # A subsequent process-prefetch pass serializes
                    # ``self.config`` into a spawned worker.  Keep that plain
                    # config synchronized with the acknowledged cursor so
                    # loader reuse continues after the completed pass.
                    if self.cursor_enabled:
                        self.config["cursor"] = _copy_cursor(item["cursor"])
                    continue
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            if not normal_completion and self._consumer_cursor is not None:
                self._cursor = _copy_cursor(self._consumer_cursor)
            self.close()

    def _start_prefetch_worker(self) -> None:
        if (
            self._thread is not None
            or self._process is not None
            or self._queue is not None
            or self._stop_event is not None
        ):
            raise StreamLoaderError("loader is already being iterated")

        self._reset_accounting()
        self._consumer_cursor = _copy_cursor(self._cursor) if self._cursor else None
        if self.prefetch_mode == "process":
            self._start_process_prefetch_worker()
            return

        self._start_thread_prefetch_worker()

    def _start_thread_prefetch_worker(self) -> None:
        self._queue = queue.Queue(maxsize=self.prefetch_buffer_size)
        self._stop_event = threading.Event()

        def worker() -> None:
            assert self._queue is not None
            assert self._stop_event is not None
            try:
                for item in self._output_iter(stop_event=self._stop_event):
                    if self._stop_event.is_set():
                        break
                    # Publish an acknowledgement cursor immediately before
                    # the corresponding sample.  The worker may prefetch ahead
                    # of the consumer, but the parent's cursor must never do
                    # so: an interrupted consumer resumes after its last
                    # yielded item, not after the queue head.
                    self._put_prefetch_item(
                        {
                            _CURSOR_MARKER: True,
                            "cursor": self._producer_cursor_state(),
                        }
                    )
                    self._put_prefetch_item(item)
            except BaseException as error:  # noqa: BLE001 - propagate to consumer.
                self._put_prefetch_item(error)
            finally:
                if not self._stop_event.is_set() and self._cursor is not None:
                    # The final worker cursor carries pass_complete=True.  A
                    # final marker prevents _iter_async's consumer cursor
                    # (which represents the last sample) from overwriting the
                    # completed-pass state when the loader is reused.
                    self._put_prefetch_item(
                        {_CURSOR_MARKER: True, "cursor": _copy_cursor(self._cursor)}
                    )
                self._put_prefetch_item(_SENTINEL)

        self._thread = threading.Thread(
            target=worker,
            name="stream-loader-prefetch",
            daemon=False,
        )
        self._thread.start()

    def _start_process_prefetch_worker(self) -> None:
        context = mp.get_context("spawn")
        self._queue = context.Queue(maxsize=self.prefetch_buffer_size)
        self._stop_event = context.Event()
        process_config = _process_prefetch_config(self.config)
        # The spawned loader is a new object, so preserve whether its cursor is
        # a checkpoint resume or this parent's next-pass iteration.
        process_config["_stream_loader_resume_cursor_pending"] = self._resume_cursor_pending
        self._resume_cursor_pending = False
        self._process = context.Process(
            target=_process_prefetch_worker,
            args=(process_config, self._queue, self._stop_event),
            name="stream-loader-prefetch",
            daemon=False,
        )
        self._process.start()

    def _put_prefetch_item(self, item: Any) -> None:
        assert self._queue is not None
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _output_iter(
        self,
        stop_event: threading.Event | None,
    ) -> Iterator[dict[str, Any]]:
        self._reset_accounting()
        if self.output_mode == "packed_sequences":
            yield from self._packed_iter(stop_event=stop_event)
            return

        for sample in self._sample_iter(stop_event=stop_event):
            if stop_event is not None and stop_event.is_set():
                return
            yield self._format_document_sample(sample)

    def _packed_iter(
        self,
        stop_event: threading.Event | None,
    ) -> Iterator[dict[str, Any]]:
        saved_cursor = self._cursor if self.cursor_enabled else None
        buffer: list[int] = list(saved_cursor.get("packed_buffer", [])) if saved_cursor else []
        spans: list[dict[str, Any]] = (
            copy.deepcopy(saved_cursor.get("packed_spans", [])) if saved_cursor else []
        )
        self._packed_cursor_buffer = buffer
        self._packed_cursor_spans = spans
        stride = self.sequence_length - 1
        if stride < 1:
            raise ValueError("packed_sequences requires sequence_length of at least 2")

        for sample in self._sample_iter(stop_event=stop_event):
            if stop_event is not None and stop_event.is_set():
                return

            sample_start = len(buffer)
            buffer.extend(sample.token_ids)
            spans.append(
                {
                    "source": sample.source,
                    "start": sample_start,
                    "end": sample_start + sample.token_count,
                }
            )

            while len(buffer) >= self.sequence_length:
                token_ids = buffer[: self.sequence_length]
                output = {
                    "input_ids": np.asarray(token_ids, dtype=np.uint32),
                    "window_token_count": self.sequence_length,
                    "target_token_count": stride,
                }
                if self.preserve_metadata:
                    output["source_spans"] = _slice_spans(spans, self.sequence_length)
                self.packed_token_counts["window_token_count"] += self.sequence_length
                self.packed_token_counts["target_token_count"] += stride
                if self.cursor_enabled:
                    self._packed_cursor_buffer = list(buffer[stride:])
                    self._packed_cursor_spans = _shift_spans(spans, stride)
                    if self._active_source_states is not None and self._active_rng is not None:
                        self._cursor = self._capture_cursor(
                            self._active_source_states,
                            self._active_rng,
                        )
                yield output
                del buffer[:stride]
                spans = _shift_spans(spans, stride)

        if len(buffer) > 1 and not self.drop_remainder:
            output = {
                "input_ids": np.asarray(buffer, dtype=np.uint32),
                "window_token_count": len(buffer),
                "target_token_count": max(len(buffer) - 1, 0),
            }
            if self.preserve_metadata:
                output["source_spans"] = _slice_spans(spans, len(buffer))
            self.packed_token_counts["window_token_count"] += len(buffer)
            self.packed_token_counts["target_token_count"] += max(len(buffer) - 1, 0)
            # The final short window consumes the whole packed residual.  Update
            # the cursor before yielding so a checkpoint taken immediately by
            # the consumer cannot re-emit these tokens on resume.
            self._clear_packed_cursor_residual()
            yield output
        elif buffer:
            self.packed_token_counts["dropped_target_count"] += max(len(buffer) - 1, 0)
        self._clear_packed_cursor_residual()
        if (
            self.cursor_enabled
            and self._active_source_states is not None
            and self._active_rng is not None
        ):
            self._cursor = self._capture_cursor(self._active_source_states, self._active_rng)

    def _clear_packed_cursor_residual(self) -> None:
        self._packed_cursor_buffer = []
        self._packed_cursor_spans = []
        if self.cursor_enabled and self._cursor is not None:
            self._cursor["packed_buffer"] = []
            self._cursor["packed_spans"] = []

    def _format_document_sample(self, sample: TokenizedSample) -> dict[str, Any]:
        base = {
            "source": sample.source,
            "token_count": sample.token_count,
        }
        if self.preserve_metadata:
            base["metadata"] = sample.metadata

        if self.output_mode == "raw_text":
            return {**base, "text": sample.text}
        if self.output_mode == "bytes":
            return {**base, "bytes": sample.text.encode("utf-8")}
        if self.output_mode == "tokenized_docs":
            return {
                **base,
                "input_ids": np.asarray(sample.token_ids, dtype=np.uint32),
            }

        raise ValueError(f"Unsupported output_mode: {self.output_mode}")

    def _sample_iter(
        self,
        stop_event: threading.Event | None,
    ) -> Iterator[TokenizedSample]:
        resume_cursor = self._resume_cursor_pending
        self._resume_cursor_pending = False
        if self._cursor is not None and self._cursor.get("pass_complete"):
            if resume_cursor:
                return
            self._start_next_pass()
        cursor = self._cursor if self.cursor_enabled else None
        rng = random.Random(self.seed)
        if cursor is not None and cursor.get("rng_state") is not None:
            rng.setstate(_restore_rng_state(cursor["rng_state"]))
        all_source_states: list[SourceState] = []
        self._active_rng = rng
        completed = False

        try:
            all_source_states = self._build_source_states()
            self._active_source_states = all_source_states
            source_states = list(all_source_states)

            while source_states:
                if stop_event is not None and stop_event.is_set():
                    return

                state = self._choose_source(source_states, rng)
                if state is None:
                    break

                document = self._next_source_document(state)
                if document is None:
                    source_states = [item for item in source_states if not item.exhausted]
                    continue

                token_ids = self.tokenizer.encode(document.text)
                if self.add_eos:
                    token_ids.append(int(self.tokenizer.eos_token_id))
                original_token_count = len(token_ids)

                remaining_quota = state.remaining_quota
                if remaining_quota is not None:
                    if remaining_quota <= 0:
                        source_states = [
                            item for item in source_states if item.remaining_quota != 0
                        ]
                        continue
                    if len(token_ids) > remaining_quota:
                        if self.add_eos:
                            assert self.tokenizer.eos_token_id is not None
                            token_ids = token_ids[: max(remaining_quota - 1, 0)]
                            token_ids.append(int(self.tokenizer.eos_token_id))
                        elif self.output_mode == "packed_sequences":
                            raise StreamLoaderError(
                                "packed_sequences cannot quota-truncate a document without "
                                "add_eos=true because the fragment would have no boundary token"
                            )
                        else:
                            token_ids = token_ids[:remaining_quota]

                if not token_ids:
                    continue

                text = document.text
                if len(token_ids) != original_token_count:
                    decode_ids = list(token_ids)
                    if self.add_eos and decode_ids[-1] == self.tokenizer.eos_token_id:
                        decode_ids = decode_ids[:-1]
                    text = self.tokenizer.decode(decode_ids)

                state.emitted_tokens += len(token_ids)
                self.token_counts[state.name] = state.emitted_tokens

                if self.cursor_enabled:
                    self._cursor = self._capture_cursor(all_source_states, rng)

                yield TokenizedSample(
                    source=state.name,
                    text=text,
                    token_ids=token_ids,
                    metadata=document.metadata,
                )

                source_states = [
                    item
                    for item in source_states
                    if item.remaining_quota is None or item.remaining_quota > 0
                ]

            if self.max_tokens != "max" and not self._can_repeat_source():
                missing = {
                    state.name: state.remaining_quota
                    for state in all_source_states
                    if state.remaining_quota is not None and state.remaining_quota > 0
                }
                if missing:
                    raise StreamLoaderError(
                        f"Datasets exhausted before max_tokens quota was met: {missing}"
                    )
            completed = True
        finally:
            for state in all_source_states:
                _close_iterator(state.iterator)
            if self.cursor_enabled and all_source_states:
                self._cursor = self._capture_cursor(all_source_states, rng)
                self._cursor["pass_complete"] = completed
            self._active_source_states = None
            self._active_rng = None

    def _build_source_states(self) -> list[SourceState]:
        quotas = self._token_quotas()
        cursor_states = {}
        if self._cursor is not None:
            cursor_states = dict(self._cursor.get("source_states", {}))
        states = []
        for dataset in self.dataset_configs:
            name = dataset["name"]
            saved = cursor_states.get(name, {})
            epoch = int(saved.get("epoch", 0))
            state = SourceState(
                name=name,
                ratio=float(dataset["ratio"]),
                iterator=iter(self._build_source(dataset)),
                quota=quotas.get(name),
                dataset=dataset,
                emitted_tokens=int(saved.get("emitted_tokens", 0)),
                exhausted=bool(saved.get("exhausted", False)),
                source_position=int(saved.get("source_position", 0)),
                epoch=epoch,
                shuffle_buffer=[
                    _deserialize_document(document) for document in saved.get("shuffle_buffer", [])
                ],
                shuffle_rng=self._source_rng(name, epoch),
            )
            if saved.get("shuffle_rng_state") is not None:
                state.shuffle_rng.setstate(_restore_rng_state(saved["shuffle_rng_state"]))
            if state.source_position:
                self._skip_source_documents(state)
            states.append(state)
        return states

    def _skip_source_documents(self, state: SourceState) -> None:
        """Replay the immutable source up to the saved raw-document cursor."""

        skipped = 0
        try:
            while skipped < state.source_position:
                next(state.iterator)
                skipped += 1
        except StopIteration:
            if self._can_repeat_source() and state.dataset is not None:
                # A cursor can point beyond one finite pass only when repeat
                # was explicitly requested.  Re-open at the saved epoch.
                _close_iterator(state.iterator)
                state.iterator = iter(self._build_source(state.dataset))
                state.source_position = 0
                state.epoch += 1
                state.exhausted = False
            else:
                state.exhausted = True

    def _next_source_document(self, state: SourceState) -> RawDocument | None:
        if state.exhausted and not state.shuffle_buffer:
            return None
        if not self.shuffle:
            try:
                document = next(state.iterator)
                state.source_position += 1
                return document
            except StopIteration:
                if not self._can_repeat_source() or state.dataset is None:
                    state.exhausted = True
                    return None
                _close_iterator(state.iterator)
                state.iterator = iter(self._build_source(state.dataset))
                state.source_position = 0
                state.epoch += 1
                state.shuffle_rng = self._source_rng(state.name, state.epoch)
                return self._next_source_document(state)

        if state.shuffle_buffer is None:
            state.shuffle_buffer = []
        while len(state.shuffle_buffer) < self.shuffle_buffer_size and not state.exhausted:
            try:
                state.shuffle_buffer.append(next(state.iterator))
                state.source_position += 1
            except StopIteration:
                state.exhausted = True
        if not state.shuffle_buffer:
            if self._can_repeat_source() and state.dataset is not None:
                _close_iterator(state.iterator)
                state.iterator = iter(self._build_source(state.dataset))
                state.source_position = 0
                state.epoch += 1
                state.exhausted = False
                state.shuffle_rng = self._source_rng(state.name, state.epoch)
                return self._next_source_document(state)
            return None

        rng = state.shuffle_rng or self._source_rng(state.name, state.epoch)
        index = rng.randrange(len(state.shuffle_buffer))
        document = state.shuffle_buffer.pop(index)
        if not state.exhausted:
            try:
                state.shuffle_buffer.append(next(state.iterator))
                state.source_position += 1
            except StopIteration:
                state.exhausted = True
        state.shuffle_rng = rng
        return document

    def _can_repeat_source(self) -> bool:
        # A max='max' pass is source-bounded even when a caller opts into
        # repeat for subsequent explicit passes; otherwise list(loader) would
        # never terminate on a finite fixture.
        return self._explicit_repeat and self.repeat and self.max_tokens != "max"

    def _build_source(self, dataset: Mapping[str, Any]) -> Iterable[RawDocument]:
        source_type = dataset.get("type", dataset.get("source", "hf"))
        if source_type == "memory":
            return MemoryTextSource(dataset)
        if source_type == "iterable":
            return IterableTextSource(dataset)
        if source_type == "jsonl":
            return JsonlTextSource(dataset)
        if source_type == "url_jsonl":
            if self.cache is None:
                raise ValueError("url_jsonl source requires cache.dir")
            return UrlJsonlTextSource(dataset, cache=self.cache, retry_policy=self.retry_policy)
        if source_type == "hf":
            return HuggingFaceTextSource(dataset, cache_dir=self.cache_dir)
        if source_type == "manifest":
            return ManifestTextSource(self.resolved_manifests[dataset["name"]])
        raise ValueError(f"Unsupported dataset source type: {source_type}")

    def _choose_source(
        self,
        source_states: list[SourceState],
        rng: random.Random,
    ) -> SourceState | None:
        candidates = [
            state
            for state in source_states
            if (not state.exhausted or state.shuffle_buffer)
            and (state.remaining_quota is None or state.remaining_quota > 0)
        ]
        if not candidates:
            return None

        if self.max_tokens != "max":
            return rng.choices(
                candidates,
                weights=[state.ratio for state in candidates],
                k=1,
            )[0]

        return rng.choices(
            candidates,
            weights=[state.ratio for state in candidates],
            k=1,
        )[0]

    def _token_quotas(self) -> dict[str, int | None]:
        if self.max_tokens == "max":
            return {dataset["name"]: None for dataset in self.dataset_configs}

        raw_quotas = [
            (dataset["name"], self.max_tokens * float(dataset["ratio"]))
            for dataset in self.dataset_configs
        ]
        quotas = {name: int(math.floor(quota)) for name, quota in raw_quotas}
        residual = self.max_tokens - sum(quotas.values())
        remainders = sorted(
            raw_quotas,
            key=lambda item: (item[1] - math.floor(item[1]), item[0]),
            reverse=True,
        )
        for index in range(residual):
            quotas[remainders[index][0]] += 1
        return quotas

    def _validate_dataset_configs(self) -> None:
        if not self.dataset_configs:
            raise ValueError("config.datasets must contain at least one dataset")

        names = [dataset.get("name") for dataset in self.dataset_configs]
        if any(name is None for name in names):
            raise ValueError("each dataset requires name")
        if len(set(names)) != len(names):
            raise ValueError("dataset names must be unique")

        ratios = []
        for dataset in self.dataset_configs:
            ratio = dataset.get("ratio")
            if ratio is None:
                raise ValueError(f"dataset {dataset['name']} requires ratio")
            ratio = float(ratio)
            if ratio <= 0:
                raise ValueError("dataset ratios must be positive")
            dataset["ratio"] = ratio
            ratios.append(ratio)
            source_type = dataset.get("type", dataset.get("source", "hf"))
            if self.config.get("require_manifests", False) and source_type != "manifest":
                raise ValueError("real data configs require manifest-backed datasets")
            if source_type == "manifest":
                forbidden = sorted(
                    field for field in ("access", "allow_reserved_benchmark") if field in dataset
                )
                if forbidden:
                    raise ValueError(
                        "training manifest datasets cannot configure evaluation authority: "
                        f"{forbidden}"
                    )
                required = {"manifest_path", "expected_fingerprint", "selection"}
                missing = sorted(field for field in required if field not in dataset)
                if missing:
                    raise ValueError(
                        f"manifest dataset {dataset['name']} is missing fields: {missing}"
                    )
            if source_type == "hf":
                revision = dataset.get("revision")
                if revision is None:
                    raise ValueError("hf dataset {} requires revision".format(dataset["name"]))
                if _HF_COMMIT_PATTERN.fullmatch(str(revision)) is None:
                    raise ValueError(
                        "hf dataset {} revision must be a 40-character commit hash".format(
                            dataset["name"]
                        )
                    )

        if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"dataset ratios must sum to 1.0, got {sum(ratios)}")


def preflight_stream_manifests(config: Mapping[str, Any]) -> dict[str, ResolvedManifest]:
    config_dict = copy.deepcopy(dict(config))
    retry_config = config_dict.get("retry", {})
    retry_policy = RetryPolicy(
        max_attempts=int(retry_config.get("max_attempts", 3)),
        initial_delay_seconds=float(retry_config.get("initial_delay_seconds", 0.25)),
        max_delay_seconds=float(retry_config.get("max_delay_seconds", 5.0)),
        multiplier=float(retry_config.get("multiplier", 2.0)),
    )
    cache_config = config_dict.get("cache", {})
    cache = None
    if cache_config.get("dir") is not None:
        cache = BoundedShardCache(
            cache_config["dir"],
            max_size_bytes=int(cache_config.get("max_size_bytes", 1 << 30)),
            retry_policy=retry_policy,
        )
    datasets = [dict(item) for item in config_dict.get("datasets", [])]
    return _preflight_manifest_datasets(datasets, cache=cache)


def _preflight_manifest_datasets(
    datasets: list[dict[str, Any]],
    *,
    cache: BoundedShardCache | None,
) -> dict[str, ResolvedManifest]:
    resolved = {}
    for dataset in datasets:
        if dataset.get("type", dataset.get("source", "hf")) != "manifest":
            continue
        forbidden = sorted(
            field for field in ("access", "allow_reserved_benchmark") if field in dataset
        )
        if forbidden:
            raise ValueError(
                f"training manifest datasets cannot configure evaluation authority: {forbidden}"
            )
        resolved[dataset["name"]] = preflight_manifest(
            dataset["manifest_path"],
            expected_fingerprint=dataset["expected_fingerprint"],
            selection=dataset["selection"],
            access="training",
            allow_reserved_benchmark=False,
            cache=cache,
        )
    return resolved


def _validate_resolved_manifest_inputs(
    datasets: list[dict[str, Any]],
    resolved: Mapping[str, ResolvedManifest],
) -> None:
    expected = {
        dataset["name"]: dataset
        for dataset in datasets
        if dataset.get("type", dataset.get("source", "hf")) == "manifest"
    }
    if set(resolved) != set(expected):
        raise ValueError("resolved manifest names do not match configured manifest datasets")
    for name, manifest in resolved.items():
        dataset = expected[name]
        if manifest.purpose in {
            DataPurpose.BENCHMARK_DEV,
            DataPurpose.BENCHMARK_RESERVED,
        }:
            raise ValueError("benchmark manifests cannot enter the training loader")
        if (
            manifest.manifest_fingerprint != dataset["expected_fingerprint"]
            or manifest.selection != dataset["selection"]
        ):
            raise ValueError(f"resolved manifest identity does not match dataset {name!r}")


def _record_to_document(
    record: Any,
    *,
    text_field: str,
    metadata_fields: Iterable[str] | None,
    max_text_chars: int | None,
    fallback_index: int,
) -> RawDocument:
    if isinstance(record, str):
        if max_text_chars is not None:
            record = record[:max_text_chars]
        return RawDocument(text=record, metadata={"index": fallback_index})
    if not isinstance(record, Mapping):
        raise TypeError(f"Dataset sample must be str or mapping, got {type(record)}")
    if text_field not in record:
        raise KeyError(f"Dataset sample is missing text field {text_field!r}")
    text = record[text_field]
    if not isinstance(text, str):
        raise TypeError(f"Dataset text field {text_field!r} must be str")
    if max_text_chars is not None:
        text = text[:max_text_chars]

    if metadata_fields is None:
        metadata = {key: value for key, value in record.items() if key != text_field}
    else:
        metadata = {key: record[key] for key in metadata_fields if key in record}
    metadata.setdefault("index", fallback_index)
    return RawDocument(text=text, metadata=metadata)


def _process_prefetch_config(config: Mapping[str, Any]) -> dict[str, Any]:
    process_config = copy.deepcopy(dict(config))
    process_config.setdefault("prefetch", {})["enabled"] = False
    return process_config


def _process_prefetch_worker(
    config: Mapping[str, Any],
    output_queue: Any,
    stop_event: Any,
) -> None:
    loader: StreamLoader | None = None
    try:
        loader = StreamLoader(config)
        for item in loader._output_iter(stop_event=stop_event):
            if stop_event.is_set():
                break
            # Publish the cursor before the corresponding sample so an
            # interrupting consumer can persist the state of the last yielded
            # item even though the worker runs in another process.
            _put_prefetch_queue_item(
                output_queue,
                stop_event,
                {_CURSOR_MARKER: True, "cursor": loader.state_dict()},
            )
            _put_prefetch_queue_item(output_queue, stop_event, item)
        if not stop_event.is_set():
            _put_prefetch_queue_item(
                output_queue,
                stop_event,
                {
                    _ACCOUNTING_MARKER: True,
                    "token_counts": loader.token_counts,
                    "packed_token_counts": loader.packed_token_counts,
                },
            )
            _put_prefetch_queue_item(
                output_queue,
                stop_event,
                {_CURSOR_MARKER: True, "cursor": loader.state_dict()},
            )
    except BaseException as error:  # noqa: BLE001 - propagate to consumer.
        _put_prefetch_queue_item(
            output_queue,
            stop_event,
            {
                _ERROR_MARKER: True,
                "message": f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
            },
        )
    finally:
        if loader is not None:
            loader.close()
        _put_prefetch_queue_item(output_queue, stop_event, _SENTINEL, force=True)


def _put_prefetch_queue_item(
    output_queue: Any,
    stop_event: Any,
    item: Any,
    *,
    force: bool = False,
) -> None:
    while force or not stop_event.is_set():
        try:
            output_queue.put(item, timeout=0.1)
            return
        except queue.Full:
            if force and stop_event.is_set():
                return
            continue


def _is_prefetch_error(item: Any) -> bool:
    return isinstance(item, Mapping) and bool(item.get(_ERROR_MARKER))


def _is_prefetch_accounting(item: Any) -> bool:
    return isinstance(item, Mapping) and bool(item.get(_ACCOUNTING_MARKER))


def _is_prefetch_cursor(item: Any) -> bool:
    return isinstance(item, Mapping) and bool(item.get(_CURSOR_MARKER))


def _has_hf_datasets(dataset_configs: Iterable[Mapping[str, Any]]) -> bool:
    return any(
        dataset.get("type", dataset.get("source", "hf")) == "hf" for dataset in dataset_configs
    )


def _serialize_document(document: RawDocument) -> dict[str, Any]:
    return {"text": document.text, "metadata": copy.deepcopy(document.metadata)}


def _deserialize_document(value: Mapping[str, Any]) -> RawDocument:
    if not isinstance(value, Mapping) or not isinstance(value.get("text"), str):
        raise ValueError("stream cursor shuffle_buffer contains an invalid document")
    metadata = value.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("stream cursor document metadata must be a mapping")
    return RawDocument(text=value["text"], metadata=dict(metadata))


def _jsonable_rng_state(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable_rng_state(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_rng_state(item) for item in value]
    return value


def _restore_rng_state(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_restore_rng_state(item) for item in value)
    return value


def _copy_cursor(cursor: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(cursor, Mapping):
        raise ValueError("stream cursor must be a mapping")
    return copy.deepcopy(dict(cursor))


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value < 1:
        raise ValueError("max_text_chars must be positive when set")
    return value


def _close_iterator(iterator: Any) -> None:
    close = getattr(iterator, "close", None)
    if close is not None:
        close()


def _slice_spans(spans: list[dict[str, Any]], length: int) -> list[dict[str, Any]]:
    sliced = []
    for span in spans:
        start = max(span["start"], 0)
        end = min(span["end"], length)
        if start < end:
            sliced.append({"source": span["source"], "start": start, "end": end})
    return sliced


def _shift_spans(
    spans: list[dict[str, Any]],
    consumed_tokens: int,
) -> list[dict[str, Any]]:
    shifted = []
    for span in spans:
        start = span["start"] - consumed_tokens
        end = span["end"] - consumed_tokens
        if end > 0:
            shifted.append(
                {
                    "source": span["source"],
                    "start": max(start, 0),
                    "end": end,
                }
            )
    return shifted
