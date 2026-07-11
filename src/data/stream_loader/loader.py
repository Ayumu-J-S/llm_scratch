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

from data.stream_loader.cache import BoundedShardCache, RetryPolicy, download_url_to_path


OUTPUT_MODES = {"raw_text", "bytes", "tokenized_docs", "packed_sequences"}
_SENTINEL = "__stream_loader_prefetch_done__"
_ERROR_MARKER = "__stream_loader_prefetch_error__"
_ACCOUNTING_MARKER = "__stream_loader_prefetch_accounting__"
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
    emitted_tokens: int = 0
    exhausted: bool = False

    @property
    def remaining_quota(self) -> int | None:
        if self.quota is None:
            return None
        return self.quota - self.emitted_tokens


@dataclass(frozen=True)
class TokenizerHandle:
    tokenizer: Any
    kind: str
    eos_token_id: int | None


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
        tokenizer: Any | Mapping[str, Any] | None = None,
    ) -> None:
        self.config = dict(config)
        self.output_mode = self.config.get("output_mode", "raw_text")
        if self.output_mode not in OUTPUT_MODES:
            raise ValueError(f"Unsupported output_mode: {self.output_mode}")

        self.seed = int(self.config.get("seed", 0))
        self.max_tokens = self.config.get("max_tokens", "max")
        if self.max_tokens != "max":
            self.max_tokens = int(self.max_tokens)
            if self.max_tokens < 1:
                raise ValueError("max_tokens must be positive or 'max'")

        self.sequence_length = int(self.config.get("sequence_length", 4096))
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")

        self.drop_remainder = bool(self.config.get("drop_remainder", True))
        self.add_eos = bool(self.config.get("add_eos", True))
        self.preserve_metadata = bool(self.config.get("preserve_metadata", False))
        self.shutdown_timeout_seconds = float(self.config.get("shutdown_timeout_seconds", 5.0))

        tokenizer_config = tokenizer if tokenizer is not None else self.config.get("tokenizer")
        self.tokenizer = _load_tokenizer(tokenizer_config)
        if self.add_eos and self.tokenizer.eos_token_id is None:
            raise ValueError("add_eos requires tokenizer.eos_token_id")

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

        prefetch_config = self.config.get("prefetch", {})
        self.prefetch_enabled = bool(prefetch_config.get("enabled", False))
        self.prefetch_buffer_size = int(prefetch_config.get("buffer_size", 8))
        if self.prefetch_enabled and self.prefetch_buffer_size < 1:
            raise ValueError("prefetch.buffer_size must be positive")
        default_prefetch_mode = "process" if _has_hf_datasets(self.dataset_configs) else "thread"
        self.prefetch_mode = str(prefetch_config.get("mode", default_prefetch_mode))
        if self.prefetch_mode not in {"thread", "process"}:
            raise ValueError("prefetch.mode must be 'thread' or 'process'")
        if (
            self.prefetch_enabled
            and self.prefetch_mode == "thread"
            and _has_hf_datasets(self.dataset_configs)
        ):
            raise ValueError("thread prefetch is unsafe for hf datasets; use process prefetch")
        if (
            self.prefetch_enabled
            and self.prefetch_mode == "process"
            and tokenizer is not None
            and _to_plain_mapping(tokenizer) is None
        ):
            raise ValueError("process prefetch requires tokenizer to be configured by mapping")

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

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self.prefetch_enabled:
            return self._iter_async()
        return self._iter_sync()

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
                    break
                if _is_prefetch_error(item):
                    raise StreamLoaderError(item["message"])
                if _is_prefetch_accounting(item):
                    self.token_counts = dict(item["token_counts"])
                    self.packed_token_counts = dict(item["packed_token_counts"])
                    continue
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
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
                    self._put_prefetch_item(item)
            except BaseException as error:  # noqa: BLE001 - propagate to consumer.
                self._put_prefetch_item(error)
            finally:
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
        buffer: list[int] = []
        spans: list[dict[str, Any]] = []
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
            yield output
        elif buffer:
            self.packed_token_counts["dropped_target_count"] += max(len(buffer) - 1, 0)

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
        rng = random.Random(self.seed)
        all_source_states: list[SourceState] = []

        try:
            all_source_states = self._build_source_states()
            source_states = list(all_source_states)

            while source_states:
                if stop_event is not None and stop_event.is_set():
                    return

                state = self._choose_source(source_states, rng)
                if state is None:
                    break

                try:
                    document = next(state.iterator)
                except StopIteration:
                    state.exhausted = True
                    source_states = [item for item in source_states if not item.exhausted]
                    continue

                token_ids = _encode_text(self.tokenizer, document.text)
                if self.add_eos:
                    assert self.tokenizer.eos_token_id is not None
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
                    text = _decode_tokens(self.tokenizer, decode_ids)

                state.emitted_tokens += len(token_ids)
                self.token_counts[state.name] = state.emitted_tokens

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

            if self.max_tokens != "max":
                missing = {
                    state.name: state.remaining_quota
                    for state in all_source_states
                    if state.remaining_quota is not None and state.remaining_quota > 0
                }
                if missing:
                    raise StreamLoaderError(
                        f"Datasets exhausted before max_tokens quota was met: {missing}"
                    )
        finally:
            for state in all_source_states:
                _close_iterator(state.iterator)

    def _build_source_states(self) -> list[SourceState]:
        quotas = self._token_quotas()
        return [
            SourceState(
                name=dataset["name"],
                ratio=float(dataset["ratio"]),
                iterator=iter(self._build_source(dataset)),
                quota=quotas.get(dataset["name"]),
            )
            for dataset in self.dataset_configs
        ]

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
        raise ValueError(f"Unsupported dataset source type: {source_type}")

    def _choose_source(
        self,
        source_states: list[SourceState],
        rng: random.Random,
    ) -> SourceState | None:
        candidates = [
            state
            for state in source_states
            if not state.exhausted and (state.remaining_quota is None or state.remaining_quota > 0)
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


def _has_hf_datasets(dataset_configs: Iterable[Mapping[str, Any]]) -> bool:
    return any(
        dataset.get("type", dataset.get("source", "hf")) == "hf" for dataset in dataset_configs
    )


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


def _load_tokenizer(config: Any) -> TokenizerHandle:
    if config is None:
        raise ValueError("tokenizer config is required")

    config_mapping = _to_plain_mapping(config)
    if config_mapping is not None:
        kind = config_mapping.get("kind")
        if kind is None:
            if config_mapping.get("path") is not None:
                kind = "tokenizers"
            else:
                raise ValueError("tokenizer config requires kind")

        if kind in {"tokenizers", "hf_tokenizers"}:
            from tokenizers import Tokenizer

            path = config_mapping.get("path")
            if path is None:
                raise ValueError("tokenizers tokenizer config requires path")
            tokenizer = Tokenizer.from_file(str(path))
            eos_token = config_mapping.get("eos_token", "<eos>")
            eos_token_id = None if eos_token is None else tokenizer.token_to_id(eos_token)
            return TokenizerHandle(
                tokenizer=tokenizer, kind="tokenizers", eos_token_id=eos_token_id
            )

        if kind in {"hf", "huggingface", "qwen"}:
            from transformers import AutoTokenizer

            model_name = config_mapping.get("name")
            if kind == "qwen" and model_name is None:
                model_name = "Qwen/Qwen3-0.6B"
            if model_name is None:
                raise ValueError("Hugging Face tokenizer config requires name")
            use_fast = bool(config_mapping.get("use_fast", True))
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=use_fast,
                trust_remote_code=bool(config_mapping.get("trust_remote_code", False)),
                local_files_only=bool(config_mapping.get("local_files_only", False)),
            )
            if use_fast and not getattr(tokenizer, "is_fast", False):
                raise ValueError(f"{model_name} did not load as a fast tokenizer")
            return TokenizerHandle(
                tokenizer=tokenizer,
                kind="transformers",
                eos_token_id=tokenizer.eos_token_id,
            )

        if kind in {"mistral_tekken", "tekken"}:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

            if bool(config_mapping.get("use_builtin_tekken", False)):
                tokenizer = MistralTokenizer.v3(is_tekken=True)
            else:
                tokenizer = MistralTokenizer.from_hf_hub(
                    config_mapping.get("name", "mistralai/Mistral-Nemo-Base-2407"),
                    revision=config_mapping.get("revision"),
                    local_files_only=bool(config_mapping.get("local_files_only", False)),
                )
            base_tokenizer = tokenizer.instruct_tokenizer.tokenizer
            if base_tokenizer.__class__.__name__ != "Tekkenizer":
                raise ValueError("configured Mistral tokenizer is not Tekken")
            return TokenizerHandle(
                tokenizer=tokenizer,
                kind="mistral_tekken",
                eos_token_id=int(base_tokenizer.eos_id),
            )

        if kind in {"bpe", "project_bpe"}:
            from tokenizer.bpe import BPETokenizer

            path = config_mapping.get("path")
            if path is None:
                raise ValueError("bpe tokenizer config requires path")
            tokenizer = BPETokenizer.load(str(path))
            return TokenizerHandle(
                tokenizer=tokenizer,
                kind="project_bpe",
                eos_token_id=tokenizer.eos_token_id,
            )

        raise ValueError(f"Unsupported tokenizer kind: {kind}")

    kind = _infer_tokenizer_kind(config)
    return TokenizerHandle(
        tokenizer=config,
        kind=kind,
        eos_token_id=_infer_eos_token_id(config, kind),
    )


def _encode_text(handle: TokenizerHandle, text: str) -> list[int]:
    if handle.kind == "tokenizers":
        return [int(token_id) for token_id in handle.tokenizer.encode(text).ids]
    if handle.kind == "mistral_tekken":
        return [
            int(token_id)
            for token_id in handle.tokenizer.instruct_tokenizer.tokenizer.encode(
                text,
                bos=False,
                eos=False,
            )
        ]
    if handle.kind == "transformers":
        return [
            int(token_id) for token_id in handle.tokenizer.encode(text, add_special_tokens=False)
        ]
    if handle.kind == "project_bpe":
        return [int(token_id) for token_id in handle.tokenizer.encode(text)]
    raise TypeError(f"Unsupported tokenizer object: {type(handle.tokenizer)}")


def _decode_tokens(handle: TokenizerHandle, token_ids: list[int]) -> str:
    if handle.kind == "tokenizers":
        return handle.tokenizer.decode(token_ids, skip_special_tokens=False)
    if handle.kind == "mistral_tekken":
        return handle.tokenizer.instruct_tokenizer.tokenizer.decode(token_ids)
    if handle.kind == "transformers":
        return handle.tokenizer.decode(token_ids, skip_special_tokens=False)
    if handle.kind == "project_bpe":
        return handle.tokenizer.decode(token_ids, skip_special_tokens=False)
    raise TypeError(f"Unsupported tokenizer object: {type(handle.tokenizer)}")


def _infer_tokenizer_kind(tokenizer: Any) -> str:
    try:
        from tokenizers import Tokenizer
    except ImportError:
        Tokenizer = None

    if Tokenizer is not None and isinstance(tokenizer, Tokenizer):
        return "tokenizers"

    try:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    except ImportError:
        MistralTokenizer = None

    if MistralTokenizer is not None and isinstance(tokenizer, MistralTokenizer):
        return "mistral_tekken"

    try:
        from transformers import PreTrainedTokenizerBase
    except ImportError:
        PreTrainedTokenizerBase = None

    if PreTrainedTokenizerBase is not None and isinstance(tokenizer, PreTrainedTokenizerBase):
        return "transformers"

    try:
        from tokenizer.bpe import BPETokenizer
    except ImportError:
        BPETokenizer = None

    if BPETokenizer is not None and isinstance(tokenizer, BPETokenizer):
        return "project_bpe"

    raise TypeError(
        "tokenizer must come from transformers, tokenizers, mistral-common, "
        "or tokenizer.bpe.BPETokenizer"
    )


def _infer_eos_token_id(tokenizer: Any, kind: str) -> int | None:
    if kind == "tokenizers":
        return tokenizer.token_to_id("<eos>")
    if kind == "mistral_tekken":
        return int(tokenizer.instruct_tokenizer.tokenizer.eos_id)
    if kind == "transformers":
        return getattr(tokenizer, "eos_token_id", None)
    if kind == "project_bpe":
        return getattr(tokenizer, "eos_token_id", None)
    return None


def _to_plain_mapping(config: Any) -> dict[str, Any] | None:
    try:
        from omegaconf import DictConfig, OmegaConf
    except ImportError:
        pass
    else:
        if isinstance(config, DictConfig):
            container = OmegaConf.to_container(config, resolve=True)
            if not isinstance(container, dict):
                raise TypeError("tokenizer config must be a mapping")
            return container

    if isinstance(config, Mapping):
        return dict(config)

    return None
