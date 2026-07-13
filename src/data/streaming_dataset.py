from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, IterableDataset

from data.stream_loader import StreamLoader
from data.stream_loader.loader import preflight_stream_manifests
from tokenizer.canonical import CanonicalTokenizer


class StreamingTokenDataset(IterableDataset):
    def __init__(
        self,
        config: Mapping[str, Any] | DictConfig,
        sequence_length: int,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")

        self.sequence_length = int(sequence_length)
        self.window_length = self.sequence_length + 1
        self.config = _stream_loader_config(config, window_length=self.window_length)
        CanonicalTokenizer.from_config(self.config.get("tokenizer"))
        self.resolved_manifests = preflight_stream_manifests(self.config)
        self._cursor: dict[str, Any] | None = _optional_cursor(self.config.get("cursor"))
        self._resume_cursor_pending = bool(
            self._cursor is not None and not self._cursor.get("pass_complete")
        )
        self._active_loader: StreamLoader | None = None

    def __iter__(self):
        with StreamLoader(
            self.config,
            resolved_manifests=self.resolved_manifests,
        ) as loader:
            if self._cursor is not None:
                loader.load_state_dict(
                    self._cursor,
                    resume_completed=self._resume_cursor_pending,
                )
            self._resume_cursor_pending = False
            self._active_loader = loader
            completed = False
            try:
                for sample in loader:
                    # StreamLoader advances this acknowledgement cursor before
                    # yielding the sample.  Store it before our own yield so a
                    # DataLoader batch boundary can be checkpointed without
                    # replaying its final item on resume.
                    self._cursor = loader.state_dict()
                    input_ids = torch.as_tensor(sample["input_ids"], dtype=torch.long)
                    if input_ids.numel() != self.window_length:
                        raise ValueError(
                            "StreamingTokenDataset expected packed windows of "
                            f"{self.window_length} tokens, got {input_ids.numel()}"
                        )
                    output: dict[str, Any] = {"input_ids": input_ids}
                    if "source_spans" in sample:
                        source_spans = copy.deepcopy(sample["source_spans"])
                        output["source_spans"] = source_spans
                        output["target_sources"] = target_sources_from_spans(
                            source_spans, target_count=input_ids.numel() - 1
                        )
                    yield output
                completed = True
            finally:
                if self._active_loader is loader:
                    # Retain every cursor for checkpoint observability. A
                    # later terminal restore explicitly asks StreamLoader for
                    # its next pass; an interrupted cursor resumes its suffix.
                    if loader.cursor_enabled:
                        self._cursor = loader.state_dict()
                        self._resume_cursor_pending = not completed
                    else:
                        self._cursor = None
                        self._resume_cursor_pending = False
                    self._active_loader = None

    def state_dict(self) -> dict[str, Any] | None:
        """Return the next-sample stream cursor for a trainer checkpoint."""

        if self._active_loader is not None:
            if not self._active_loader.cursor_enabled:
                return None
            self._cursor = self._active_loader.state_dict()
        if self._cursor is None:
            # Validate and materialize the initial JSON-safe cursor through the
            # existing StreamLoader contract instead of duplicating it here.
            probe = StreamLoader(self.config, resolved_manifests=self.resolved_manifests)
            if not probe.cursor_enabled:
                return None
            self._cursor = probe.state_dict()
        return copy.deepcopy(self._cursor)

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Install a previously captured stream cursor before iteration starts."""

        if self._active_loader is not None:
            raise RuntimeError("cannot replace a streaming cursor while iteration is active")
        cursor = _optional_cursor(state)
        assert cursor is not None
        config = copy.deepcopy(self.config)
        # A completed pass has no unconsumed suffix. Its next invocation must
        # take the existing StreamLoader next-pass path, not treat completion
        # as a zero-length interrupted suffix.
        self._resume_cursor_pending = not bool(cursor.get("pass_complete"))
        probe = StreamLoader(config, resolved_manifests=self.resolved_manifests)
        probe.load_state_dict(cursor, resume_completed=self._resume_cursor_pending)
        self._cursor = probe.state_dict()


def causal_lm_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    input_ids = torch.stack([sample["input_ids"] for sample in samples]).long()
    if input_ids.size(1) < 2:
        raise ValueError("causal LM batches require at least two tokens per sample")

    batch: dict[str, Any] = {
        "inputs": input_ids[:, :-1].contiguous(),
        "labels": input_ids[:, 1:].contiguous(),
    }
    if any("target_sources" in sample for sample in samples):
        if not all("target_sources" in sample for sample in samples):
            raise ValueError("all samples in a metadata-preserving batch need target_sources")
        batch["target_sources"] = [list(sample["target_sources"]) for sample in samples]
        batch["source_spans"] = [
            copy.deepcopy(sample.get("source_spans", [])) for sample in samples
        ]
    return batch


def create_streaming_token_dataloader(
    *,
    config: Mapping[str, Any] | DictConfig,
    sequence_length: int,
    batch_size: int,
    drop_last: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    generator: torch.Generator | None = None,
    worker_init_fn=None,
) -> DataLoader:
    dataset = StreamingTokenDataset(
        config=config,
        sequence_length=sequence_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        worker_init_fn=worker_init_fn,
        collate_fn=causal_lm_collate_fn,
    )


def _stream_loader_config(
    config: Mapping[str, Any] | DictConfig,
    *,
    window_length: int,
) -> dict[str, Any]:
    config_dict = _to_plain_dict(config)
    if "sources" in config_dict and "datasets" not in config_dict:
        config_dict["datasets"] = config_dict.pop("sources")

    config_dict["output_mode"] = "packed_sequences"
    config_dict["sequence_length"] = window_length
    config_dict["drop_remainder"] = True
    return config_dict


def target_sources_from_spans(
    source_spans: list[Mapping[str, Any]], *, target_count: int
) -> list[str]:
    """Convert packed token spans to sources aligned with causal labels.

    A packed window's source spans address positions in ``input_ids``.  Labels
    address positions 1..W-1, so the source of label ``i`` is the span
    containing packed position ``i + 1``.
    """

    if target_count < 0:
        raise ValueError("target_count must be non-negative")
    result: list[str] = []
    for label_index in range(target_count):
        packed_position = label_index + 1
        matches = [
            str(span["source"])
            for span in source_spans
            if int(span["start"]) <= packed_position < int(span["end"])
        ]
        if len(matches) != 1:
            raise ValueError(
                "source spans must cover each causal target exactly once; "
                f"position={packed_position}, matches={matches}"
            )
        result.append(matches[0])
    return result


def _to_plain_dict(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        container = OmegaConf.to_container(config, resolve=True)
        if not isinstance(container, dict):
            raise TypeError("streaming config must be a mapping")
        return copy.deepcopy(container)
    if isinstance(config, Mapping):
        return copy.deepcopy(dict(config))
    raise TypeError("streaming config must be a mapping")


def _optional_cursor(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("streaming cursor must be a mapping when configured")
    return copy.deepcopy(dict(value))
