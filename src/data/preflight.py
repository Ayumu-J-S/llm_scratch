from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data.identity import normalized_content_sha256


REPORT_SCHEMA_VERSION = 1
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

    def report(self) -> dict[str, Any]:
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
        }


@dataclass(frozen=True)
class AuditResult:
    report: dict[str, Any]
    content_hashes: frozenset[str]


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
    processed = 0
    iterator = iter(samples)
    try:
        while processed < max_documents:
            try:
                sample = next(iterator)
            except StopIteration:
                break
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

            audit = sources.setdefault(source, _SourceAudit())
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
            if bool(metadata.get("fallback_id", False)):
                audit.fallback_document_ids += 1
            if bool(metadata.get("truncated", False)):
                audit.truncated_documents += 1
            processed += 1
    finally:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()

    return AuditResult(
        report={
            "bounded": processed >= max_documents,
            "max_documents": max_documents,
            "documents": processed,
            "sources": {name: audit.report() for name, audit in sorted(sources.items())},
        },
        content_hashes=frozenset(seen_hashes),
    )


def merge_loader_quality(
    audit: dict[str, Any],
    *,
    rejection_counts: Mapping[str, Mapping[str, int]] | None = None,
    fallback_counts: Mapping[str, int] | None = None,
    truncated_counts: Mapping[str, int] | None = None,
) -> None:
    """Merge production-adapter counters into an accepted-document audit."""

    sources = audit.get("sources")
    if not isinstance(sources, dict):
        raise DataPreflightError("audit.sources must be a dictionary")
    names = set(sources)
    names.update((rejection_counts or {}).keys())
    names.update((fallback_counts or {}).keys())
    names.update((truncated_counts or {}).keys())
    for name in sorted(names):
        source = sources.setdefault(name, _SourceAudit().report())
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
    for sample in samples:
        window = _nonnegative_int(sample.get("window_token_count"), "window_token_count")
        targets = _nonnegative_int(sample.get("target_token_count"), "target_token_count")
        if targets != max(window - 1, 0):
            raise DataPreflightError("packed window target count does not equal window length - 1")
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
    lines = [
        "# DATA-004 bounded data preflight",
        "",
        f"- Mode: `{report['mode']}`",
        f"- Code commit: `{fingerprints['code']['git_commit']}`",
        f"- Dirty worktree: `{str(fingerprints['code']['dirty']).lower()}`",
        f"- Config SHA-256: `{fingerprints['config']}`",
        f"- Tokenizer SHA-256: `{fingerprints['tokenizer']}`",
        "",
        "## Stream audit",
        "",
        "| Split | Source | Documents | Bytes | Canonical tokens | EOS | Duplicates | Rejected |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, split_report in sorted(report["splits"].items()):
        for source, values in sorted(split_report["sources"].items()):
            quality = values["quality"]
            lines.append(
                f"| {split} | {source} | {values['documents']} | {values['utf8_bytes']} | "
                f"{values['canonical_tokens']} | {values['eos_tokens']} | "
                f"{quality['duplicates']} | {sum(quality['rejections'].values())} |"
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
