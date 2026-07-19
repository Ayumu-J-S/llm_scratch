"""One token-weighted scorer for training-time and standalone evaluation.

The scorer consumes already-tokenized fixed windows.  It never writes the
held-out token IDs to a result: the evaluated windows and valid target stream
are represented by batching-independent SHA-256 digests instead.
"""

from __future__ import annotations

import hashlib
import math
import struct
import sys
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from training.optimization import autocast_context


SCORER_REVISION = "VAL-001-causal-lm-scorer-v2"
_DEFAULT_SOURCE = "aggregate"


@dataclass(frozen=True)
class CorpusScore:
    """Token-weighted metrics for one source/corpus."""

    nll_sum: float
    nll: float
    perplexity: float | None
    perplexity_overflow: bool
    target_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "nll_sum": self.nll_sum,
            "nll": self.nll,
            "perplexity": self.perplexity,
            "perplexity_overflow": self.perplexity_overflow,
            "target_tokens": self.target_tokens,
        }


@dataclass(frozen=True)
class EvaluationResult:
    """Auditable aggregate and per-corpus evaluation output."""

    namespace: str
    aggregate: CorpusScore
    by_corpus: dict[str, CorpusScore]
    evaluated_windows: int
    evaluated_window_sha256: str
    evaluated_token_sha256: str
    manifest_identity: dict[str, Any]
    logical_checkpoint_identity: dict[str, Any] | None
    physical_checkpoint_identity: dict[str, Any] | None
    scorer_revision: str
    pause_seconds: float
    timing: dict[str, Any] = field(default_factory=dict)

    @property
    def nll(self) -> float:
        return self.aggregate.nll

    @property
    def perplexity(self) -> float | None:
        return self.aggregate.perplexity

    @property
    def target_tokens(self) -> int:
        return self.aggregate.target_tokens

    @property
    def evaluated_targets_per_second(self) -> float | None:
        if self.pause_seconds <= 0.0:
            return None
        return self.target_tokens / self.pause_seconds

    def as_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "scorer_revision": self.scorer_revision,
            "aggregate": self.aggregate.as_dict(),
            "by_corpus": {name: score.as_dict() for name, score in sorted(self.by_corpus.items())},
            "evaluated_windows": self.evaluated_windows,
            "evaluated_window_sha256": self.evaluated_window_sha256,
            "evaluated_token_sha256": self.evaluated_token_sha256,
            "manifest_identity": self.manifest_identity,
            "logical_checkpoint_identity": self.logical_checkpoint_identity,
            "physical_checkpoint_identity": self.physical_checkpoint_identity,
            "pause_seconds": self.pause_seconds,
            "evaluated_targets_per_second": self.evaluated_targets_per_second,
            "timing": self.timing,
        }


class CausalLMScorer:
    """Score a model over a fresh, deterministic loader for every call."""

    def __init__(
        self,
        *,
        device: torch.device,
        precision: str = "fp32",
        ignore_index: int = -100,
        measure_phase_timing: bool = False,
        cuda_events: bool = False,
    ) -> None:
        self.device = device
        self.precision = precision
        self.ignore_index = int(ignore_index)
        self.measure_phase_timing = bool(measure_phase_timing)
        self.cuda_events = bool(
            self.measure_phase_timing and cuda_events and self.device.type == "cuda"
        )

    def score(
        self,
        model: torch.nn.Module,
        loader_factory: Callable[[], Iterable[Mapping[str, Any]]],
        *,
        namespace: str = "validation",
        logical_checkpoint_identity: Mapping[str, Any] | None = None,
        physical_checkpoint_identity: Mapping[str, Any] | None = None,
        manifest_identity: Mapping[str, Any] | None = None,
        configured_data_fingerprints: Iterable[str] | None = None,
    ) -> EvaluationResult:
        """Return exact token-weighted metrics and identities for one score pass.

        The factory is invoked for each pass so validation never reuses an
        exhausted iterator or a stateful streaming cursor.
        """

        if namespace not in {"validation", "memorization"}:
            raise ValueError("evaluation namespace must be validation or memorization")
        started = time.perf_counter()
        phase_timing_enabled = self.measure_phase_timing and (
            self.device.type != "cuda" or self.cuda_events
        )
        loader_started = time.perf_counter() if phase_timing_enabled else None
        loader = loader_factory()
        loader_construction_seconds = (
            time.perf_counter() - loader_started if loader_started is not None else 0.0
        )
        actual_manifest_identity, loader_configured_fingerprints = _manifest_identity_from_loader(
            loader, namespace=namespace
        )
        if manifest_identity is not None and dict(manifest_identity) != actual_manifest_identity:
            raise ValueError(
                "declared manifest identity does not match the actual validation loader"
            )
        resolved_manifest_identity = actual_manifest_identity
        if configured_data_fingerprints is not None:
            configured_fingerprints = [str(value) for value in configured_data_fingerprints]
            if configured_fingerprints and (
                not actual_manifest_identity or not loader_configured_fingerprints
            ):
                raise ValueError("validation loader is missing required manifest identity metadata")
            checkpoint_identity = _checkpoint_identity_from_logical(logical_checkpoint_identity)
            if (
                checkpoint_identity is not None
                and checkpoint_identity.get("data_fingerprints") != configured_fingerprints
            ):
                raise ValueError(
                    "checkpoint identity.data_fingerprints do not match ordered configured manifests"
                )
            if loader_configured_fingerprints and not _configured_split_matches(
                configured_fingerprints, loader_configured_fingerprints
            ):
                raise ValueError(
                    "actual validation loader manifests do not match ordered configured manifests"
                )
        iterator_started = time.perf_counter() if phase_timing_enabled else None
        iterator = iter(loader)
        iterator_creation_seconds = (
            time.perf_counter() - iterator_started if iterator_started is not None else 0.0
        )
        was_training = model.training
        model.eval()
        aggregate_loss = 0.0
        aggregate_tokens = 0
        corpus_loss: dict[str, float] = {}
        corpus_tokens: dict[str, int] = {}
        window_digest = hashlib.sha256()
        token_digest = hashlib.sha256()
        evaluated_windows = 0
        timing: dict[str, Any] = {}
        if phase_timing_enabled:
            timing = {
                "phase_timing_method": "cuda_events" if self.cuda_events else "host_wall",
                "loader_construction_seconds": loader_construction_seconds,
                "iterator_creation_seconds": iterator_creation_seconds,
                "data_wait_seconds": 0.0,
                "host_device_preparation_seconds": 0.0,
                "forward_seconds": 0.0,
                "loss_seconds": 0.0,
                "validation_seconds": 0.0,
                "iterator_close_seconds": 0.0,
            }
        cuda_phase_events: dict[str, list[tuple[Any, Any]]] = {
            "forward_seconds": [],
            "loss_seconds": [],
        }
        try:
            with torch.no_grad():
                batch_index = 0
                while True:
                    wait_started = time.perf_counter() if phase_timing_enabled else None
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        break
                    if wait_started is not None:
                        timing["data_wait_seconds"] += time.perf_counter() - wait_started
                    batch_index += 1
                    preparation_started = time.perf_counter() if phase_timing_enabled else None
                    inputs = batch["inputs"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    if preparation_started is not None:
                        timing["host_device_preparation_seconds"] += (
                            time.perf_counter() - preparation_started
                        )
                    forward_phase = self._start_compute_phase(phase_timing_enabled)
                    with autocast_context(self.device, self.precision):
                        logits = model(inputs)
                    self._end_compute_phase(
                        timing,
                        cuda_phase_events,
                        "forward_seconds",
                        forward_phase,
                    )
                    loss_phase = self._start_compute_phase(phase_timing_enabled)
                    with autocast_context(self.device, self.precision):
                        flat_labels = labels.reshape(-1)
                        flat_losses = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            flat_labels,
                            reduction="none",
                            ignore_index=self.ignore_index,
                        )
                    self._end_compute_phase(
                        timing,
                        cuda_phase_events,
                        "loss_seconds",
                        loss_phase,
                    )
                    if not torch.isfinite(flat_losses).all():
                        error = FloatingPointError(
                            f"non-finite {namespace} loss at validation batch {batch_index}"
                        )
                        error.batch_index = batch_index
                        raise error

                    source_rows = _target_source_rows(batch.get("target_sources"), labels)
                    batch_inputs = inputs.detach().cpu().tolist()
                    batch_labels = labels.detach().cpu().tolist()
                    batch_losses = flat_losses.detach().cpu().reshape(labels.shape).tolist()
                    for row_index, (input_row, label_row) in enumerate(
                        zip(batch_inputs, batch_labels)
                    ):
                        sources = source_rows[row_index]
                        if len(sources) != len(label_row):
                            raise ValueError(
                                "target_sources must be label-aligned with every validation window"
                            )
                        for token_index, (label, loss) in enumerate(
                            zip(label_row, batch_losses[row_index])
                        ):
                            if int(label) == self.ignore_index:
                                continue
                            if not math.isfinite(float(loss)):
                                error = FloatingPointError(
                                    f"non-finite {namespace} loss at validation batch "
                                    f"{batch_index}, row {row_index}"
                                )
                                error.batch_index = batch_index
                                raise error
                            source = str(sources[token_index])
                            if not source:
                                raise ValueError("every valid validation target needs a source")
                            value = float(loss)
                            token_id = int(label)
                            aggregate_loss += value
                            aggregate_tokens += 1
                            corpus_loss[source] = corpus_loss.get(source, 0.0) + value
                            corpus_tokens[source] = corpus_tokens.get(source, 0) + 1
                            _update_int(token_digest, token_id)
                            _update_text(token_digest, source)
                        _update_window_digest(
                            window_digest,
                            input_row,
                            label_row,
                            sources,
                            self.ignore_index,
                        )
                        evaluated_windows += 1
        finally:
            close_started = time.perf_counter() if phase_timing_enabled else None
            try:
                _close_evaluation_iterator(iterator)
            finally:
                if close_started is not None:
                    timing["iterator_close_seconds"] += time.perf_counter() - close_started
                if was_training:
                    model.train()
                else:
                    model.eval()

        if aggregate_tokens == 0:
            raise ValueError("validation loader is empty or contains zero target tokens")
        if namespace == "validation" and resolved_manifest_identity:
            expected_sources = set(resolved_manifest_identity)
            observed_sources = set(corpus_tokens)
            if observed_sources != expected_sources:
                raise ValueError(
                    "validation target attribution does not match declared manifests; "
                    f"expected={sorted(expected_sources)}, observed={sorted(observed_sources)}"
                )
        if sum(corpus_tokens.values()) != aggregate_tokens:
            raise RuntimeError("per-corpus validation target counts do not reconcile")
        if self.cuda_events:
            # One score-boundary synchronization makes all recorded event
            # durations completion-aware without imposing a barrier per batch.
            torch.cuda.synchronize(self.device)
            for phase_name, pairs in cuda_phase_events.items():
                timing[phase_name] = sum(
                    start_event.elapsed_time(end_event) / 1000.0 for start_event, end_event in pairs
                )
        aggregate = _make_score(aggregate_loss, aggregate_tokens)
        by_corpus = {
            source: _make_score(corpus_loss[source], corpus_tokens[source])
            for source in corpus_tokens
        }
        total_seconds = max(0.0, time.perf_counter() - started)
        if phase_timing_enabled:
            timing["validation_seconds"] = total_seconds
            timing["total_seconds"] = total_seconds
        return EvaluationResult(
            namespace=namespace,
            aggregate=aggregate,
            by_corpus=by_corpus,
            evaluated_windows=evaluated_windows,
            evaluated_window_sha256=window_digest.hexdigest(),
            evaluated_token_sha256=token_digest.hexdigest(),
            manifest_identity=resolved_manifest_identity,
            logical_checkpoint_identity=(
                dict(logical_checkpoint_identity)
                if logical_checkpoint_identity is not None
                else None
            ),
            physical_checkpoint_identity=(
                dict(physical_checkpoint_identity)
                if physical_checkpoint_identity is not None
                else None
            ),
            scorer_revision=SCORER_REVISION,
            pause_seconds=total_seconds,
            timing=timing,
        )

    def _start_compute_phase(self, phase_timing_enabled: bool) -> float | tuple[Any, Any] | None:
        if not phase_timing_enabled:
            return None
        if not self.cuda_events:
            return time.perf_counter()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return start_event, end_event

    def _end_compute_phase(
        self,
        timing: dict[str, Any],
        cuda_phase_events: dict[str, list[tuple[Any, Any]]],
        phase_name: str,
        phase: float | tuple[Any, Any] | None,
    ) -> None:
        if phase is None:
            return
        if isinstance(phase, float):
            timing[phase_name] += time.perf_counter() - phase
            return
        start_event, end_event = phase
        end_event.record()
        cuda_phase_events[phase_name].append((start_event, end_event))


def manifest_identities(manifests: Mapping[str, Any] | None) -> dict[str, Any]:
    """Reduce resolved manifests to non-content identity fields for results."""

    if not manifests:
        return {}
    result: dict[str, Any] = {}
    for name, manifest in manifests.items():
        purpose = getattr(manifest.purpose, "value", manifest.purpose)
        result[str(name)] = {
            "manifest_fingerprint": str(manifest.manifest_fingerprint),
            "dataset_fingerprint": str(manifest.dataset_fingerprint),
            "purpose": str(purpose),
            "selection": str(manifest.selection),
            "split_fingerprints": dict(manifest.split_fingerprints),
            "split_policy_fingerprint": manifest.split_policy_fingerprint,
        }
    return result


def _manifest_identity_from_loader(
    loader: Any, *, namespace: str
) -> tuple[dict[str, Any], list[str]]:
    dataset = getattr(loader, "dataset", None)
    actual = manifest_identities(getattr(dataset, "resolved_manifests", None))
    required_selection = "validation" if namespace == "validation" else "all"
    for name, observed in actual.items():
        if observed["selection"] != required_selection:
            raise ValueError(
                f"{namespace} loader manifest {name!r} uses selection "
                f"{observed['selection']!r}, expected {required_selection!r}"
            )
    configured = _configured_manifest_inputs_from_loader(loader)
    if configured:
        configured_names = [name for name, _, _ in configured]
        if configured_names != list(actual):
            raise ValueError(
                "validation loader configured manifests do not match resolved manifest names"
            )
        for name, fingerprint, selection in configured:
            observed = actual[name]
            if observed["manifest_fingerprint"] != fingerprint:
                raise ValueError(
                    f"validation loader manifest fingerprint changed for configured source {name!r}"
                )
            if observed["selection"] != selection:
                raise ValueError(
                    f"validation loader manifest selection changed for configured source {name!r}"
                )
    return actual, [fingerprint for _, fingerprint, _ in configured]


def _configured_manifest_inputs_from_loader(loader: Any) -> list[tuple[str, str, str]]:
    dataset = getattr(loader, "dataset", None)
    config = getattr(dataset, "config", None)
    if not isinstance(config, Mapping):
        return []
    datasets = config.get("datasets", config.get("sources", []))
    if not isinstance(datasets, list):
        return []
    result: list[tuple[str, str, str]] = []
    for source in datasets:
        if not isinstance(source, Mapping):
            continue
        source_type = source.get("type", source.get("source", "hf"))
        fingerprint = source.get("expected_fingerprint")
        selection = source.get("selection")
        if source_type == "manifest" and fingerprint and selection:
            result.append((str(source["name"]), str(fingerprint), str(selection)))
    return result


def _configured_split_matches(all_fingerprints: list[str], split_fingerprints: list[str]) -> bool:
    if len(all_fingerprints) < len(split_fingerprints):
        return False
    # The configuration's validation sources follow the train sources. Keep
    # the check order-sensitive while allowing non-manifest sources to be
    # absent from the identity list.
    return all_fingerprints[-len(split_fingerprints) :] == split_fingerprints


def _checkpoint_identity_from_logical(
    logical_checkpoint_identity: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if logical_checkpoint_identity is None:
        return None
    value = logical_checkpoint_identity.get("checkpoint_identity")
    return value if isinstance(value, Mapping) else None


def _target_source_rows(value: Any, labels: torch.Tensor) -> list[list[str]]:
    rows = labels.shape[0]
    columns = labels.shape[1]
    if value is None:
        return [[_DEFAULT_SOURCE] * columns for _ in range(rows)]
    if not isinstance(value, (list, tuple)) or len(value) != rows:
        raise ValueError("target_sources must contain one row per validation sample")
    result: list[list[str]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != columns:
            raise ValueError("target_sources must be label-aligned with validation labels")
        result.append([str(source) for source in row])
    return result


def _make_score(loss_sum: float, target_tokens: int) -> CorpusScore:
    nll = loss_sum / target_tokens
    overflow = nll > math.log(sys.float_info.max)
    if overflow:
        perplexity = None
    else:
        perplexity = math.exp(nll)
    return CorpusScore(
        nll_sum=loss_sum,
        nll=nll,
        perplexity=perplexity,
        perplexity_overflow=overflow,
        target_tokens=target_tokens,
    )


def _update_int(digest: "hashlib._Hash", value: int) -> None:
    digest.update(struct.pack("<q", int(value)))


def _update_text(digest: "hashlib._Hash", value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(struct.pack("<Q", len(encoded)))
    digest.update(encoded)


def _update_window_digest(
    digest: "hashlib._Hash",
    inputs: list[int],
    labels: list[int],
    sources: list[str],
    ignore_index: int,
) -> None:
    """Hash one window as a framed record, independent of batch grouping."""

    digest.update(b"window-v1")
    digest.update(struct.pack("<Q", len(inputs)))
    for value in inputs:
        _update_int(digest, int(value))
    digest.update(struct.pack("<Q", len(labels)))
    for value in labels:
        _update_int(digest, int(value))
    digest.update(struct.pack("<Q", len(sources)))
    for source in sources:
        _update_text(digest, source)
    digest.update(struct.pack("<Q", sum(int(label) != ignore_index for label in labels)))
    # Valid target IDs are included in the separate target digest.  The window
    # digest only needs the mask and window boundaries, so no floating losses
    # enter the identity.
    for label in labels:
        digest.update(b"1" if int(label) != ignore_index else b"0")


def _close_evaluation_iterator(iterator: Any) -> None:
    close = getattr(iterator, "close", None)
    if callable(close):
        close()
    fetcher = getattr(iterator, "_dataset_fetcher", None)
    dataset_iterator = getattr(fetcher, "dataset_iter", None)
    close = getattr(dataset_iterator, "close", None)
    if callable(close):
        close()
