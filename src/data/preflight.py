from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import shlex
import shutil
import subprocess
import sys
import time
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data.identity import normalized_content_sha256


REPORT_SCHEMA_VERSION = 2
DEFAULT_CACHE_CAP_LIMIT_BYTES = 200_000_000_000
DECLARED_REJECTIONS = (
    "control_character",
    "empty",
    "empty_after_truncation",
    "invalid_unicode",
    "non_string",
    "wrong_script",
)


class DataPreflightError(RuntimeError):
    """Raised when bounded data evidence is incomplete or unsafe."""


@dataclass
class _SourceAudit:
    documents: int = 0
    utf8_bytes: int = 0
    canonical_tokens: int = 0
    eos_tokens: int = 0
    tokenizer_byte_fallback_tokens: int = 0
    tokenization_failures: int = 0
    duplicate_documents: int = 0
    fallback_document_ids: int = 0
    truncated_documents: int = 0
    scripts: dict[str, int] = field(
        default_factory=lambda: {"japanese": 0, "latin": 0, "other_letters": 0}
    )
    languages: dict[str, int] = field(
        default_factory=lambda: {"en": 0, "ja": 0, "mixed": 0, "other": 0}
    )
    rejections: dict[str, int] = field(
        default_factory=lambda: {reason: 0 for reason in DECLARED_REJECTIONS}
    )
    document_utf8_bytes: list[int] = field(default_factory=list)
    document_tokens: list[int] = field(default_factory=list)
    tokenization_latency_ms: list[float] = field(default_factory=list)
    loader_next_seconds: float = 0.0
    qa_retokenization_seconds: float = 0.0

    def report(self, *, phase_elapsed_seconds: float) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "utf8_bytes": self.utf8_bytes,
            "canonical_tokens": self.canonical_tokens,
            "eos_tokens": self.eos_tokens,
            "tokenizer_byte_fallback_tokens": self.tokenizer_byte_fallback_tokens,
            "tokenization_failures": self.tokenization_failures,
            "scripts": dict(sorted(self.scripts.items())),
            "languages": dict(sorted(self.languages.items())),
            "quality": {
                "duplicates": self.duplicate_documents,
                "fallback_document_ids": self.fallback_document_ids,
                "truncated": self.truncated_documents,
                "rejections": dict(sorted(self.rejections.items())),
            },
            "tails": {
                "document_utf8_bytes": distribution(self.document_utf8_bytes),
                "document_tokens": distribution(self.document_tokens),
                "tokenization_latency_ms": distribution(self.tokenization_latency_ms),
            },
            "timing": {
                "loader_next_seconds": round(self.loader_next_seconds, 9),
                "qa_retokenization_seconds": round(self.qa_retokenization_seconds, 9),
            },
            "rates": {
                "documents_per_second": _rate(self.documents, phase_elapsed_seconds),
                "utf8_bytes_per_second": _rate(self.utf8_bytes, phase_elapsed_seconds),
                "canonical_tokens_per_second": _rate(self.canonical_tokens, phase_elapsed_seconds),
                "qa_retokenizer_documents_per_second": _rate(
                    self.documents, self.qa_retokenization_seconds
                ),
                "qa_retokenizer_utf8_bytes_per_second": _rate(
                    self.utf8_bytes, self.qa_retokenization_seconds
                ),
                "qa_retokenizer_canonical_tokens_per_second": _rate(
                    self.canonical_tokens, self.qa_retokenization_seconds
                ),
            },
        }


@dataclass(frozen=True)
class AuditResult:
    report: dict[str, Any]
    content_hashes: frozenset[str]
    document_ids: frozenset[str]


def audit_raw_stream(
    samples: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_documents: int,
    add_eos: bool,
) -> AuditResult:
    """Aggregate a bounded raw-text stream without retaining corpus text."""

    if max_documents < 1:
        raise ValueError("max_documents must be positive")
    sources: dict[str, _SourceAudit] = {}
    seen_hashes: set[str] = set()
    seen_document_ids: set[str] = set()
    loader_next_latencies_ms: list[float] = []
    processed = 0
    iterator = iter(samples)
    phase_started = time.monotonic_ns()
    try:
        while processed < max_documents:
            wait_started = time.monotonic_ns()
            try:
                sample = next(iterator)
            except StopIteration:
                break
            next_seconds = (time.monotonic_ns() - wait_started) / 1_000_000_000.0
            source = _nonempty_string(sample.get("source"), "sample.source")
            text = sample.get("text")
            if not isinstance(text, str):
                raise DataPreflightError("raw preflight samples must contain string text")
            metadata = sample.get("metadata", {})
            if not isinstance(metadata, Mapping):
                raise DataPreflightError("sample.metadata must be a mapping")

            started = time.perf_counter_ns()
            try:
                token_ids = tokenizer.encode(text)
            except Exception as error:
                raise DataPreflightError(
                    f"canonical tokenization failed for source {source!r}"
                ) from error
            latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            utf8_bytes = len(text.encode("utf-8"))
            japanese, latin, other = script_counts(text)
            content_hash = metadata.get("content_sha256")
            if not isinstance(content_hash, str) or len(content_hash) != 64:
                content_hash = normalized_content_sha256(text)
            document_id = metadata.get("document_id")
            if not _is_sha256(document_id):
                raise DataPreflightError(
                    "raw preflight samples must contain a SHA-256 metadata.document_id"
                )

            audit = sources.setdefault(source, _SourceAudit())
            audit.loader_next_seconds += next_seconds
            audit.qa_retokenization_seconds += latency_ms / 1_000.0
            audit.documents += 1
            audit.utf8_bytes += utf8_bytes
            audit.canonical_tokens += len(token_ids)
            audit.eos_tokens += int(add_eos)
            count_byte_fallback = getattr(tokenizer, "count_byte_fallback_tokens", None)
            if not callable(count_byte_fallback):
                raise DataPreflightError(
                    "canonical tokenizer must expose count_byte_fallback_tokens"
                )
            audit.tokenizer_byte_fallback_tokens += int(count_byte_fallback(token_ids))
            audit.scripts["japanese"] += japanese
            audit.scripts["latin"] += latin
            audit.scripts["other_letters"] += other
            audit.languages[classify_language(japanese, latin, other)] += 1
            audit.document_utf8_bytes.append(utf8_bytes)
            audit.document_tokens.append(len(token_ids))
            audit.tokenization_latency_ms.append(latency_ms)
            if content_hash in seen_hashes:
                audit.duplicate_documents += 1
            else:
                seen_hashes.add(content_hash)
            seen_document_ids.add(str(document_id))
            if bool(metadata.get("fallback_id", False)):
                audit.fallback_document_ids += 1
            if bool(metadata.get("truncated", False)):
                audit.truncated_documents += 1
            loader_next_latencies_ms.append(next_seconds * 1_000.0)
            processed += 1
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()

    phase_elapsed_seconds = (time.monotonic_ns() - phase_started) / 1_000_000_000.0
    first_wait_ms = loader_next_latencies_ms[0] if loader_next_latencies_ms else None
    steady_waits = loader_next_latencies_ms[1:]
    loader_next_seconds = sum(loader_next_latencies_ms) / 1_000.0
    qa_retokenization_seconds = sum(audit.qa_retokenization_seconds for audit in sources.values())
    return AuditResult(
        report={
            "bounded": processed >= max_documents,
            "max_documents": max_documents,
            "documents": processed,
            "timing": {
                "elapsed_seconds": round(phase_elapsed_seconds, 9),
                "first_sample_loader_next_ms": (
                    round(first_wait_ms, 6) if first_wait_ms is not None else None
                ),
                "steady_sample_loader_next_ms": distribution(steady_waits),
                "loader_next_seconds": round(loader_next_seconds, 9),
                "loader_next_fraction": _rate(loader_next_seconds, phase_elapsed_seconds),
                "qa_retokenization_seconds": round(qa_retokenization_seconds, 9),
            },
            "rates": {
                "documents_per_second": _rate(processed, phase_elapsed_seconds),
                "source_rows_per_second": None,
            },
            "sources": {
                name: audit.report(phase_elapsed_seconds=phase_elapsed_seconds)
                for name, audit in sorted(sources.items())
            },
        },
        content_hashes=frozenset(seen_hashes),
        document_ids=frozenset(seen_document_ids),
    )


def merge_loader_quality(
    audit: dict[str, Any],
    *,
    rejection_counts: Mapping[str, Mapping[str, int]] | None = None,
    fallback_counts: Mapping[str, int] | None = None,
    truncated_counts: Mapping[str, int] | None = None,
    raw_row_counts: Mapping[str, int] | None = None,
    missing_data_counts: Mapping[str, Mapping[str, int]] | None = None,
    source_read_seconds: Mapping[str, float] | None = None,
    tokenization_seconds: Mapping[str, float] | None = None,
) -> None:
    """Merge production-adapter counters into an accepted-document audit."""

    sources = audit.get("sources")
    if not isinstance(sources, dict):
        raise DataPreflightError("audit.sources must be a dictionary")
    names = set(sources)
    names.update((rejection_counts or {}).keys())
    names.update((fallback_counts or {}).keys())
    names.update((truncated_counts or {}).keys())
    names.update((raw_row_counts or {}).keys())
    names.update((missing_data_counts or {}).keys())
    names.update((source_read_seconds or {}).keys())
    names.update((tokenization_seconds or {}).keys())
    phase_elapsed_seconds = float(audit.get("timing", {}).get("elapsed_seconds", 0.0))
    for name in sorted(names):
        source = sources.setdefault(
            name, _SourceAudit().report(phase_elapsed_seconds=phase_elapsed_seconds)
        )
        quality = source["quality"]
        observed = (rejection_counts or {}).get(name, {})
        quality["rejections"] = {
            key: int(observed.get(key, 0))
            for key in sorted(set(DECLARED_REJECTIONS) | set(observed))
        }
        if name in (fallback_counts or {}):
            quality["fallback_document_ids"] = int((fallback_counts or {})[name])
        if name in (truncated_counts or {}):
            quality["truncated"] = int((truncated_counts or {})[name])
        rows_read = int((raw_row_counts or {}).get(name, source["documents"]))
        if rows_read < 0:
            raise DataPreflightError("loader raw-row counters must be non-negative")
        missing = {
            str(reason): _nonnegative_int(count, f"missing_data_counts.{name}.{reason}")
            for reason, count in sorted((missing_data_counts or {}).get(name, {}).items())
        }
        rejected = sum(quality["rejections"].values())
        quality["source_rows_read"] = rows_read
        quality["missing_data"] = missing
        quality["missing_data_events"] = sum(missing.values())
        quality["rejection_rate_per_row"] = _rate(rejected, rows_read)
        quality["missing_data_event_rate_per_row"] = _rate(sum(missing.values()), rows_read)
        source["rates"]["source_rows_per_second"] = _rate(rows_read, phase_elapsed_seconds)
        source["rates"]["rejections_per_second"] = _rate(rejected, phase_elapsed_seconds)
        source["rates"]["missing_data_events_per_second"] = _rate(
            sum(missing.values()), phase_elapsed_seconds
        )
        read_seconds = float((source_read_seconds or {}).get(name, 0.0))
        production_tokenization_seconds = float((tokenization_seconds or {}).get(name, 0.0))
        if read_seconds < 0 or production_tokenization_seconds < 0:
            raise DataPreflightError("loader performance counters must be non-negative")
        source["timing"]["production_source_read_seconds"] = round(read_seconds, 9)
        source["timing"]["production_tokenization_seconds"] = round(
            production_tokenization_seconds, 9
        )
        source["rates"]["production_source_rows_per_second"] = _rate(rows_read, read_seconds)
        source["rates"]["production_tokenizer_documents_per_second"] = _rate(
            int(source["documents"]), production_tokenization_seconds
        )
        source["rates"]["production_tokenizer_utf8_bytes_per_second"] = _rate(
            int(source["utf8_bytes"]), production_tokenization_seconds
        )
        source["rates"]["production_tokenizer_canonical_tokens_per_second"] = _rate(
            int(source["canonical_tokens"]), production_tokenization_seconds
        )
    audit["rates"]["source_rows_per_second"] = _rate(
        sum(
            int((raw_row_counts or {}).get(name, values["documents"]))
            for name, values in sources.items()
        ),
        phase_elapsed_seconds,
    )


def assert_observed_disjointness(train: AuditResult, validation: AuditResult) -> dict[str, int]:
    """Fail with both bounded overlap counts; retain only hashes, never raw text."""

    document_id_overlap = train.document_ids & validation.document_ids
    normalized_content_overlap = train.content_hashes & validation.content_hashes
    result = {
        "observed_document_id_overlap": len(document_id_overlap),
        "observed_normalized_content_overlap": len(normalized_content_overlap),
    }
    if document_id_overlap or normalized_content_overlap:
        raise DataPreflightError(
            "observed train/validation overlap: "
            f"document_ids={len(document_id_overlap)}, "
            f"normalized_content={len(normalized_content_overlap)}"
        )
    return result


def merge_quota_truncation(loader: Any, packing: dict[str, Any]) -> None:
    """Report scheduler quota cuts separately from byte-policy truncation."""

    fragments = {
        str(name): _nonnegative_int(value, f"quota_truncated_fragment_counts.{name}")
        for name, value in sorted(getattr(loader, "quota_truncated_fragment_counts", {}).items())
    }
    removed = {
        str(name): _nonnegative_int(value, f"quota_removed_token_counts.{name}")
        for name, value in sorted(getattr(loader, "quota_removed_token_counts", {}).items())
    }
    names = sorted(set(fragments) | set(removed) | set(packing["expected_ratios"]))
    packing["quota_truncation"] = {
        "fragments_by_source": {name: fragments.get(name, 0) for name in names},
        "removed_tokens_by_source": {name: removed.get(name, 0) for name in names},
        "total_fragments": sum(fragments.values()),
        "total_removed_tokens": sum(removed.values()),
        "policy": "trained-target source quota; distinct from byte-policy document truncation",
    }
    source_read = {
        str(name): round(float(value), 9)
        for name, value in sorted(getattr(loader, "source_read_seconds", {}).items())
    }
    tokenization = {
        str(name): round(float(value), 9)
        for name, value in sorted(getattr(loader, "tokenization_seconds", {}).items())
    }
    packing["production_timing_by_source"] = {
        name: {
            "source_read_seconds": source_read.get(name, 0.0),
            "tokenization_seconds": tokenization.get(name, 0.0),
        }
        for name in names
    }


def summarize_packing(
    samples: Iterable[Mapping[str, Any]],
    *,
    expected_ratios: Mapping[str, float],
    tolerance: float,
) -> dict[str, Any]:
    """Reconcile emitted packed windows and target-token source accounting."""

    if not 0.0 <= tolerance <= 1.0:
        raise ValueError("ratio tolerance must be between zero and one")
    expected = {str(name): float(ratio) for name, ratio in expected_ratios.items()}
    if not expected or not math.isclose(sum(expected.values()), 1.0, abs_tol=1e-6):
        raise ValueError("expected source ratios must sum to one")
    source_targets = {name: 0 for name in expected}
    window_tokens = 0
    target_tokens = 0
    windows = 0
    loader_next_latencies_ms: list[float] = []
    phase_started = time.monotonic_ns()
    iterator = iter(samples)
    try:
        while True:
            wait_started = time.monotonic_ns()
            try:
                sample = next(iterator)
            except StopIteration:
                break
            loader_next_latencies_ms.append((time.monotonic_ns() - wait_started) / 1_000_000.0)
            window = _nonnegative_int(sample.get("window_token_count"), "window_token_count")
            targets = _nonnegative_int(sample.get("target_token_count"), "target_token_count")
            if targets != max(window - 1, 0):
                raise DataPreflightError(
                    "packed window target count does not equal window length - 1"
                )
            counts = sample.get("source_target_counts")
            if not isinstance(counts, Mapping):
                raise DataPreflightError("packed sample source_target_counts must be a mapping")
            unexpected = sorted(set(counts) - set(expected))
            if unexpected:
                raise DataPreflightError(f"packed sample contains undeclared sources: {unexpected}")
            counted = 0
            for name, value in counts.items():
                count = _nonnegative_int(value, f"source_target_counts.{name}")
                source_targets[str(name)] += count
                counted += count
            if counted != targets:
                raise DataPreflightError("packed sample source targets do not reconcile")
            windows += 1
            window_tokens += window
            target_tokens += targets
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    phase_elapsed_seconds = (time.monotonic_ns() - phase_started) / 1_000_000_000.0
    if target_tokens < 1:
        raise DataPreflightError("preflight produced no trained targets")

    realized = {name: source_targets[name] / target_tokens for name in sorted(expected)}
    deviations = {name: realized[name] - expected[name] for name in sorted(expected)}
    within = all(abs(value) <= tolerance for value in deviations.values())
    return {
        "windows": windows,
        "window_tokens": window_tokens,
        "trained_targets": target_tokens,
        "source_trained_targets": dict(sorted(source_targets.items())),
        "expected_ratios": dict(sorted(expected.items())),
        "realized_ratios": realized,
        "ratio_deviations": deviations,
        "ratio_tolerance": tolerance,
        "ratios_within_tolerance": within,
        "accounting_reconciled": sum(source_targets.values()) == target_tokens,
        "timing": {
            "elapsed_seconds": round(phase_elapsed_seconds, 9),
            "first_sample_loader_next_ms": (
                round(loader_next_latencies_ms[0], 6) if loader_next_latencies_ms else None
            ),
            "steady_sample_loader_next_ms": distribution(loader_next_latencies_ms[1:]),
            "loader_next_seconds": round(sum(loader_next_latencies_ms) / 1_000.0, 9),
        },
        "rates": {
            "windows_per_second": _rate(windows, phase_elapsed_seconds),
            "target_tokens_per_second": _rate(target_tokens, phase_elapsed_seconds),
        },
    }


def assert_loader_accounting(loader: Any, packing: Mapping[str, Any]) -> None:
    """Cross-check the report against StreamLoader's independent counters."""

    expected_targets = dict(packing["source_trained_targets"])
    if dict(getattr(loader, "trained_target_counts", {})) != expected_targets:
        raise DataPreflightError("loader trained-target counters do not match packed samples")
    packed = dict(getattr(loader, "packed_token_counts", {}))
    if int(packed.get("target_token_count", -1)) != int(packing["trained_targets"]):
        raise DataPreflightError("loader packed target total does not match packed samples")
    if int(packed.get("window_token_count", -1)) != int(packing["window_tokens"]):
        raise DataPreflightError("loader packed window total does not match packed samples")


def cache_telemetry(loaders: Iterable[Any]) -> dict[str, int]:
    counters = {
        "hits": 0,
        "misses": 0,
        "downloads": 0,
        "downloaded_bytes": 0,
        "retries": 0,
        "evictions": 0,
        "corruptions": 0,
        "wait_timeouts": 0,
        "active_leases": 0,
        "size_bytes": 0,
        "free_bytes": 0,
    }
    for loader in loaders:
        cache = getattr(loader, "cache", None)
        if cache is None:
            continue
        telemetry = cache.telemetry
        for key in (
            "hits",
            "misses",
            "downloads",
            "downloaded_bytes",
            "retries",
            "evictions",
            "corruptions",
            "wait_timeouts",
        ):
            counters[key] += int(telemetry.get(key, 0))
        counters["active_leases"] += int(telemetry.get("active_leases", 0))
        counters["size_bytes"] = max(counters["size_bytes"], int(telemetry.get("size_bytes", 0)))
        counters["free_bytes"] = int(telemetry.get("free_bytes", counters["free_bytes"]))
    return counters


def assert_cache_released(telemetry: Mapping[str, int]) -> None:
    active_leases = int(telemetry.get("active_leases", -1))
    if active_leases != 0:
        raise DataPreflightError(f"preflight ended with unreleased cache leases: {active_leases}")


def disk_forecast(
    cache_dir: str | Path,
    *,
    cache_cap_bytes: int,
    current_managed_cache_bytes: int,
    reserved_os_checkpoint_bytes: int,
    largest_shard_bytes: int,
    cache_cap_limit_bytes: int = DEFAULT_CACHE_CAP_LIMIT_BYTES,
) -> dict[str, Any]:
    if (
        min(
            cache_cap_bytes,
            current_managed_cache_bytes,
            reserved_os_checkpoint_bytes,
            largest_shard_bytes,
            cache_cap_limit_bytes,
        )
        < 0
    ):
        raise ValueError("disk forecast byte values must be non-negative")
    usage = shutil.disk_usage(Path(cache_dir))
    future_full_cache_free = usage.free + current_managed_cache_bytes - cache_cap_bytes
    future_full_cache_with_temp_free = future_full_cache_free - largest_shard_bytes
    return {
        "filesystem_total_bytes": usage.total,
        "filesystem_used_bytes": usage.used,
        "filesystem_free_bytes": usage.free,
        "cache_cap_bytes": cache_cap_bytes,
        "current_managed_cache_bytes": current_managed_cache_bytes,
        "cache_cap_limit_bytes": cache_cap_limit_bytes,
        "reserved_os_checkpoint_bytes": reserved_os_checkpoint_bytes,
        "largest_shard_temp_bytes": largest_shard_bytes,
        "projected_free_at_full_cache_bytes": future_full_cache_free,
        "projected_free_at_full_cache_with_largest_temp_bytes": (future_full_cache_with_temp_free),
        "cache_cap_within_limit": cache_cap_bytes <= cache_cap_limit_bytes,
        "headroom_admission_passed": (
            cache_cap_bytes <= cache_cap_limit_bytes
            and future_full_cache_with_temp_free >= reserved_os_checkpoint_bytes
        ),
    }


def assert_cold_cache_is_empty(cache_dir: str | Path) -> None:
    path = Path(cache_dir)
    if not path.is_dir():
        raise DataPreflightError(
            f"--cold requires an existing empty dedicated cache directory: {path}"
        )
    entries = list(path.iterdir())
    if entries:
        raise DataPreflightError(
            "--cold refuses to delete or reuse cache contents; provide an already-empty "
            f"dedicated directory: {path}"
        )


def data_fingerprints(loaders: Iterable[Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for loader in loaders:
        for name, manifest in sorted(getattr(loader, "resolved_manifests", {}).items()):
            key = f"{name}:{manifest.selection}"
            identity = {
                "dataset": str(manifest.dataset_fingerprint),
                "manifest": str(manifest.manifest_fingerprint),
                "selection": str(manifest.selection),
            }
            previous = result.get(key)
            if previous is not None and previous != identity:
                raise DataPreflightError(f"conflicting manifest identity for source {key!r}")
            result[key] = identity
    return result


def code_fingerprint(root: str | Path) -> dict[str, Any]:
    root_path = Path(root)
    commit = _git(root_path, "rev-parse", "HEAD")
    status = _git(root_path, "status", "--porcelain", "--untracked-files=normal")
    return {"git_commit": commit, "dirty": bool(status)}


def config_fingerprint(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def reproduction_payload(
    config: Mapping[str, Any],
    argv: Iterable[str],
    *,
    effective_argv: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Embed complete secret-free inputs so the report's config hash is reproducible."""

    resolved = json.loads(json.dumps(config, ensure_ascii=False))
    forbidden = _find_forbidden_config_keys(resolved)
    if forbidden:
        raise DataPreflightError(
            f"resolved config contains secret-bearing key(s), refusing report: {forbidden}"
        )
    arguments = [str(value) for value in argv]
    effective_arguments = (
        [str(value) for value in effective_argv] if effective_argv is not None else arguments
    )
    payload = {
        "resolved_hydra_config": resolved,
        "argv": arguments,
        "effective_argv": effective_arguments,
        "python_command": shlex.join([sys.executable, *arguments]),
        "config_sha256": config_fingerprint(resolved),
        "note": (
            "argv/python_command retain operator inputs; effective_argv records internal "
            "--cold translation for Hydra. An outer uv launcher is not observable inside Python."
        ),
    }
    if payload["config_sha256"] != config_fingerprint(payload["resolved_hydra_config"]):
        raise AssertionError("embedded config fingerprint is not reproducible")
    return payload


def process_resource_snapshot() -> dict[str, int]:
    """Capture monotonic process counters without a sampler thread or artificial work."""

    usage = resource.getrusage(resource.RUSAGE_SELF)
    status = _proc_status()
    peak_rss_bytes = int(usage.ru_maxrss) * (1024 if sys.platform != "darwin" else 1)
    return {
        "monotonic_ns": time.monotonic_ns(),
        "current_rss_bytes": int(status.get("VmRSS", 0)) * 1024,
        "peak_rss_bytes": peak_rss_bytes,
        "swap_bytes": int(status.get("VmSwap", 0)) * 1024,
        "minor_page_faults": int(usage.ru_minflt),
        "major_page_faults": int(usage.ru_majflt),
    }


def process_resource_report(
    start: Mapping[str, int], end: Mapping[str, int], *, elapsed_seconds: float
) -> dict[str, Any]:
    if elapsed_seconds <= 0:
        raise ValueError("resource-report elapsed_seconds must be positive")
    minor = max(0, int(end["minor_page_faults"]) - int(start["minor_page_faults"]))
    major = max(0, int(end["major_page_faults"]) - int(start["major_page_faults"]))
    return {
        "start": dict(start),
        "end": dict(end),
        "current_rss_bytes": int(end["current_rss_bytes"]),
        "peak_rss_bytes": int(end["peak_rss_bytes"]),
        "peak_rss_growth_bytes": max(0, int(end["peak_rss_bytes"]) - int(start["peak_rss_bytes"])),
        "swap_bytes": int(end["swap_bytes"]),
        "swap_growth_bytes": int(end["swap_bytes"]) - int(start["swap_bytes"]),
        "minor_page_faults": minor,
        "major_page_faults": major,
        "minor_page_faults_per_second": _rate(minor, elapsed_seconds),
        "major_page_faults_per_second": _rate(major, elapsed_seconds),
    }


def measurement_conditions(*, mode: str) -> dict[str, Any]:
    return {
        "scope": "loader_only",
        "claim_boundary": (
            "Measures source iteration, filtering, canonical tokenization, packing, cache, "
            "and process resources only; it does not measure a model, GPU utilization, "
            "host-to-device transfer, end-to-end training consumption, or supply headroom."
        ),
        "cache_state": mode,
        "clock": "time.monotonic_ns",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pid": os.getpid(),
        "loader_next_definition": (
            "External next(loader) wall time includes source read/filter, production "
            "tokenization, packing, prefetch synchronization, and Python overhead; separate "
            "instrumented source-read and production-tokenization counters are also reported."
        ),
        "first_sample_definition": "latency of the first external next(loader) call in a phase",
        "steady_sample_definition": "latencies of all later external next(loader) calls in a phase",
        "rate_denominator": "monotonic elapsed wall time for the named loader-only phase",
        "process_memory_definition": (
            "current RSS and swap from /proc/self/status; peak RSS and page faults from "
            "getrusage(RUSAGE_SELF)"
        ),
    }


def summarize_repeated_reports(reports: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Retain every observation and deterministic median/range for comparable runs."""

    observations = list(reports)
    if len(observations) < 2:
        raise DataPreflightError("repeated summary requires at least two observations")
    signatures = [_comparison_signature(report) for report in observations]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise DataPreflightError("reports are not comparable under DATA-004 conditions")
    metric_sets = [set(_repeat_metrics(report)) for report in observations]
    metric_names = sorted(set.intersection(*metric_sets))
    metrics: dict[str, Any] = {}
    for name in metric_names:
        values = [float(_repeat_metrics(report)[name]) for report in observations]
        ordered = sorted(values)
        middle = len(ordered) // 2
        median = (
            ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0
        )
        metrics[name] = {
            "observations": values,
            "median": round(median, 9),
            "min": round(ordered[0], 9),
            "max": round(ordered[-1], 9),
            "spread": round(ordered[-1] - ordered[0], 9),
        }
    return {
        "schema_version": 1,
        "observation_count": len(observations),
        "comparison_signature": signatures[0],
        "metrics": metrics,
        "source_reports": [
            {
                "mode": report["mode"],
                "code_commit": report["fingerprints"]["code"]["git_commit"],
                "config_sha256": report["fingerprints"]["config"],
            }
            for report in observations
        ],
    }


def write_reports(
    report: Mapping[str, Any], json_path: str | Path, markdown_path: str | Path
) -> None:
    json_target = Path(json_path)
    markdown_target = Path(markdown_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_target.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: Mapping[str, Any]) -> str:
    fingerprints = report["fingerprints"]
    packing = report["packing"]
    disk = report["disk"]
    measurement = report.get("measurement", {})
    conditions = measurement.get("conditions", {})
    resources = measurement.get("process_resources", {})
    integrity = report.get("integrity", {})
    lines = [
        "# DATA-004 bounded data preflight",
        "",
        f"- Mode: `{report['mode']}`",
        f"- Code commit: `{fingerprints['code']['git_commit']}`",
        f"- Dirty worktree: `{str(fingerprints['code']['dirty']).lower()}`",
        f"- Config SHA-256: `{fingerprints['config']}`",
        f"- Tokenizer SHA-256: `{fingerprints['tokenizer']}`",
        f"- Measurement scope: `{conditions.get('scope', 'not recorded')}`",
        f"- Whole-run elapsed seconds: `{measurement.get('whole_run_elapsed_seconds', 'not recorded')}`",
        f"- Observed document-ID overlap: `{integrity.get('observed_document_id_overlap', 'not recorded')}`",
        f"- Observed normalized-content overlap: `{integrity.get('observed_normalized_content_overlap', 'not recorded')}`",
        "",
        "## Stream audit",
        "",
        "| Split | Source | Rows read | Documents | Bytes | Canonical tokens | Docs/s | Tokens/s | Duplicates | Rejected | Missing |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, split_report in sorted(report["splits"].items()):
        for source, values in sorted(split_report["sources"].items()):
            quality = values["quality"]
            rates = values.get("rates", {})
            lines.append(
                f"| {split} | {source} | {quality.get('source_rows_read', 'n/a')} | "
                f"{values['documents']} | {values['utf8_bytes']} | "
                f"{values['canonical_tokens']} | {rates.get('documents_per_second', 'n/a')} | "
                f"{rates.get('canonical_tokens_per_second', 'n/a')} | "
                f"{quality['duplicates']} | {sum(quality['rejections'].values())} | "
                f"{quality.get('missing_data_events', 'n/a')} |"
            )
    lines.extend(
        [
            "",
            "## Target accounting",
            "",
            f"- Packed windows: `{packing['windows']}`",
            f"- Trained targets: `{packing['trained_targets']}`",
            f"- Accounting reconciled: `{str(packing['accounting_reconciled']).lower()}`",
            f"- Ratios within tolerance: `{str(packing['ratios_within_tolerance']).lower()}`",
            f"- Loader-only target tokens/s: `{packing.get('rates', {}).get('target_tokens_per_second', 'not recorded')}`",
            f"- Quota-truncated fragments: `{packing.get('quota_truncation', {}).get('total_fragments', 'not recorded')}`",
            f"- Quota-removed tokens: `{packing.get('quota_truncation', {}).get('total_removed_tokens', 'not recorded')}`",
            "",
            "| Source | Expected | Realized | Deviation |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for source, expected in sorted(packing["expected_ratios"].items()):
        lines.append(
            f"| {source} | {expected:.6f} | {packing['realized_ratios'][source]:.6f} | "
            f"{packing['ratio_deviations'][source]:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Cache and disk",
            "",
            f"- Cache telemetry: `{json.dumps(report['cache'], sort_keys=True)}`",
            f"- Cache cap within limit: `{str(disk['cache_cap_within_limit']).lower()}`",
            f"- Headroom admission passed: `{str(disk['headroom_admission_passed']).lower()}`",
            f"- Reserved OS/checkpoint bytes: `{disk['reserved_os_checkpoint_bytes']}`",
            f"- Largest-shard temporary bytes: `{disk['largest_shard_temp_bytes']}`",
            f"- Downloaded bytes/s: `{report['cache'].get('downloaded_bytes_per_second', 'not recorded')}`",
            "",
            "## Process resources and reproduction",
            "",
            f"- Current RSS bytes: `{resources.get('current_rss_bytes', 'not recorded')}`",
            f"- Peak RSS bytes: `{resources.get('peak_rss_bytes', 'not recorded')}`",
            f"- Swap bytes: `{resources.get('swap_bytes', 'not recorded')}`",
            f"- Minor page faults/s: `{resources.get('minor_page_faults_per_second', 'not recorded')}`",
            f"- Major page faults/s: `{resources.get('major_page_faults_per_second', 'not recorded')}`",
            f"- Exact process argv: `{json.dumps(report.get('reproduction', {}).get('argv', []), ensure_ascii=False)}`",
            "- The JSON report embeds the complete safe resolved Hydra configuration needed to recompute its config SHA-256.",
            "- Scope boundary: loader-only; no model, GPU, end-to-end consumption, or data-supply-sufficiency claim is made.",
            "",
            "No raw corpus text is retained in either report.",
            "",
        ]
    )
    return "\n".join(lines)


def distribution(values: Iterable[int | float]) -> dict[str, int | float | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}

    def percentile(fraction: float) -> float:
        index = max(0, math.ceil(fraction * len(ordered)) - 1)
        return round(ordered[index], 6)

    return {
        "count": len(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": round(ordered[-1], 6),
    }


def script_counts(text: str) -> tuple[int, int, int]:
    japanese = latin = other = 0
    for character in text:
        codepoint = ord(character)
        if (
            0x3040 <= codepoint <= 0x30FF
            or 0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        ):
            japanese += 1
        elif "LATIN" in unicodedata.name(character, ""):
            latin += 1
        elif unicodedata.category(character).startswith("L"):
            other += 1
    return japanese, latin, other


def classify_language(japanese: int, latin: int, other: int) -> str:
    if japanese and latin:
        return "mixed"
    if japanese:
        return "ja"
    if latin:
        return "en"
    return "other"


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise DataPreflightError(f"{label} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataPreflightError(f"{label} must be a non-negative integer")
    return value


def _rate(numerator: int | float, elapsed_seconds: int | float) -> float | None:
    denominator = float(elapsed_seconds)
    if denominator <= 0:
        return None
    return round(float(numerator) / denominator, 6)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _proc_status() -> dict[str, int]:
    path = Path("/proc/self/status")
    if not path.exists():
        return {}
    result: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("VmRSS:", "VmSwap:")):
            continue
        key, raw = line.split(":", 1)
        parts = raw.split()
        if parts:
            result[key] = int(parts[0])
    return result


def _find_forbidden_config_keys(value: Any, prefix: str = "") -> list[str]:
    forbidden_names = {
        "access_token",
        "api_key",
        "auth_token",
        "credential",
        "credentials",
        "password",
        "secret",
    }
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).lower() in forbidden_names and child not in (None, "", False):
                found.append(path)
            found.extend(_find_forbidden_config_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forbidden_config_keys(child, f"{prefix}[{index}]"))
    return found


def _comparison_signature(report: Mapping[str, Any]) -> dict[str, Any]:
    reproduction = report.get("reproduction", {})
    config = json.loads(json.dumps(reproduction.get("resolved_hydra_config", {})))
    preflight = config.get("preflight")
    if isinstance(preflight, dict):
        preflight.pop("output_dir", None)
        preflight.pop("report_stem", None)
    conditions = report.get("measurement", {}).get("conditions", {})
    return {
        "mode": report.get("mode"),
        "limits": report.get("limits"),
        "code_commit": report.get("fingerprints", {}).get("code", {}).get("git_commit"),
        "data": report.get("fingerprints", {}).get("data"),
        "tokenizer": report.get("fingerprints", {}).get("tokenizer"),
        "semantic_config_sha256": config_fingerprint(config),
        "platform": conditions.get("platform"),
        "machine": conditions.get("machine"),
        "python": conditions.get("python"),
        "scope": conditions.get("scope"),
    }


def _repeat_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    measurement = report["measurement"]
    resources = measurement["process_resources"]
    result = {
        "whole_run_elapsed_seconds": float(measurement["whole_run_elapsed_seconds"]),
        "packing.target_tokens_per_second": float(
            report["packing"]["rates"]["target_tokens_per_second"]
        ),
        "process.peak_rss_bytes": float(resources["peak_rss_bytes"]),
        "process.current_rss_bytes": float(resources["current_rss_bytes"]),
        "process.minor_page_faults_per_second": float(resources["minor_page_faults_per_second"]),
        "process.major_page_faults_per_second": float(resources["major_page_faults_per_second"]),
        "cache.downloaded_bytes_per_second": float(report["cache"]["downloaded_bytes_per_second"]),
    }
    for split, split_report in sorted(report["splits"].items()):
        for source, source_report in sorted(split_report["sources"].items()):
            for metric in (
                "documents_per_second",
                "utf8_bytes_per_second",
                "canonical_tokens_per_second",
                "source_rows_per_second",
            ):
                value = source_report["rates"].get(metric)
                if value is not None:
                    result[f"{split}.{source}.{metric}"] = float(value)
    return result
