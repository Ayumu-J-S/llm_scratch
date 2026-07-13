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
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from training.optimization import autocast_context


SCORER_REVISION = "VAL-001-causal-lm-scorer-v1"
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
        }


class CausalLMScorer:
    """Score a model over a fresh, deterministic loader for every call."""

    def __init__(
        self,
        *,
        device: torch.device,
        precision: str = "fp32",
        ignore_index: int = -100,
    ) -> None:
        self.device = device
        self.precision = precision
        self.ignore_index = int(ignore_index)

    def score(
        self,
        model: torch.nn.Module,
        loader_or_factory: Iterable[Mapping[str, Any]] | Callable[[], Iterable[Mapping[str, Any]]],
        *,
        namespace: str = "validation",
        logical_checkpoint_identity: Mapping[str, Any] | None = None,
        physical_checkpoint_identity: Mapping[str, Any] | None = None,
        manifest_identity: Mapping[str, Any] | None = None,
    ) -> EvaluationResult:
        """Return exact token-weighted metrics and identities for one score pass.

        A callable is preferred and is invoked for each pass.  Passing an
        iterable remains useful for small map-style fixtures and tests.
        """

        if namespace not in {"validation", "memorization"}:
            raise ValueError("evaluation namespace must be validation or memorization")
        loader = loader_or_factory() if callable(loader_or_factory) else loader_or_factory
        resolved_manifest_identity = dict(
            manifest_identity
            if manifest_identity is not None
            else _manifest_identity_from_loader(loader)
        )
        iterator = iter(loader)
        was_training = model.training
        model.eval()
        started = time.perf_counter()
        aggregate_loss = 0.0
        aggregate_tokens = 0
        corpus_loss: dict[str, float] = {}
        corpus_tokens: dict[str, int] = {}
        window_digest = hashlib.sha256()
        token_digest = hashlib.sha256()
        evaluated_windows = 0
        try:
            with torch.no_grad():
                for batch_index, batch in enumerate(iterator, start=1):
                    inputs = batch["inputs"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    with autocast_context(self.device, self.precision):
                        logits = model(inputs)
                        flat_labels = labels.reshape(-1)
                        flat_losses = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)),
                            flat_labels,
                            reduction="none",
                            ignore_index=self.ignore_index,
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
            _close_evaluation_iterator(iterator)
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
        aggregate = _make_score(aggregate_loss, aggregate_tokens)
        by_corpus = {
            source: _make_score(corpus_loss[source], corpus_tokens[source])
            for source in corpus_tokens
        }
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
            pause_seconds=max(0.0, time.perf_counter() - started),
        )


def manifest_identities(manifests: Mapping[str, Any] | None) -> dict[str, Any]:
    """Reduce resolved manifests to non-content identity fields for results."""

    if not manifests:
        return {}
    result: dict[str, Any] = {}
    for name, manifest in sorted(manifests.items()):
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


def _manifest_identity_from_loader(loader: Any) -> dict[str, Any]:
    dataset = getattr(loader, "dataset", None)
    return manifest_identities(getattr(dataset, "resolved_manifests", None))


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
