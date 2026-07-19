"""Complete checkpoint-owned training split contamination scan."""

from __future__ import annotations

import hashlib
import fcntl
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from benchmarks.suite import LoadedSuite, contamination_probes
from data.identity import canonical_json_bytes, normalize_text_identity
from data.manifests import preflight_manifest
from data.stream_loader.cache import BoundedShardCache, RetryPolicy
from data.stream_loader.loader import ManifestTextSource
from runtime.reproducibility import sha256_file


ROOT_DIR = Path(__file__).resolve().parents[2]
SHINGLE_CODEPOINTS = 48
SCAN_REVISION = "BENCH-001-contamination-v22"
NORMALIZATION_REVISION = "normalize-text-identity-nfc-strip-plus-json-object-v20"
SCAN_INDEX_SCHEMA_VERSION = 2
MATCHER_REVISION = "collision-verified-rolling-hash-codepoint-v1"
JSON_OBJECT_NORMALIZATION_REVISION = (
    "bounded-all-object-string-input-projection-json-nfc-sha256-v19"
)
PRODUCER_IDENTITY_REVISION = "contamination-producer-v1"
PRODUCER_SOURCE_SCOPE_REVISION = "src-python-pyproject-lock-v1"
_PRODUCER_PACKAGES = ("pyarrow",)
_INPUT_ONLY_OMITTED_FIELDS = {
    # q_id selects and identifies the pinned source example, but the evaluator
    # prompt consumes only the question and five choices. Treat that exact
    # prompt-bearing mapping as contamination even when source metadata and the
    # answer label were not copied into training data.
    "jcommonsenseqa": frozenset({"label", "q_id"}),
    "gsm8k": frozenset({"answer"}),
}
_ROLLING_BASE = 1_000_000_007
_ROLLING_MASK = (1 << 64) - 1
JSON_TRAVERSAL_MAX_TOTAL_BYTES = 8 * 1024 * 1024
JSON_TRAVERSAL_MAX_NODES = 32_768
JSON_TRAVERSAL_MAX_DEPTH = 32
JSON_TRAVERSAL_MAX_DECODED_STRINGS = 8_192
_SCAN_IMPLEMENTATION_FILES = {
    "benchmarks.contamination": Path(__file__).resolve(),
    "benchmarks.suite": ROOT_DIR / "src/benchmarks/suite.py",
    "data.identity": ROOT_DIR / "src/data/identity.py",
    "data.manifests": ROOT_DIR / "src/data/manifests.py",
    "data.parquet_source": ROOT_DIR / "src/data/parquet_source.py",
    "data.quality": ROOT_DIR / "src/data/quality.py",
    "data.splits": ROOT_DIR / "src/data/splits.py",
    "data.stream_loader.loader": ROOT_DIR / "src/data/stream_loader/loader.py",
}


Reference = dict[str, str]


@dataclass(frozen=True)
class _ProbeIndex:
    exact: dict[str, list[Reference]]
    normalized: dict[str, list[Reference]]
    structured_json: dict[str, list[Reference]]
    structured_json_projections: dict[tuple[str, ...], dict[str, list[Reference]]]
    shingles: _ShingleMatcher


class _ShingleMatcher:
    """Collision-verified rolling matcher over benchmark-owned codepoint shingles."""

    def __init__(self, patterns: Mapping[str, list[Reference]]) -> None:
        self._patterns: dict[int, dict[str, tuple[Reference, ...]]] = {}
        for pattern, references in sorted(patterns.items()):
            if len(pattern) != SHINGLE_CODEPOINTS:
                raise ContaminationScanError("contamination shingle has the wrong length")
            self._patterns.setdefault(_rolling_key(pattern), {})[pattern] = tuple(references)
        self._power = pow(_ROLLING_BASE, SHINGLE_CODEPOINTS - 1, 1 << 64)

    @property
    def pattern_count(self) -> int:
        return sum(len(bucket) for bucket in self._patterns.values())

    @property
    def stored_codepoints(self) -> int:
        return self.pattern_count * SHINGLE_CODEPOINTS

    def references_in(
        self,
        text: str,
        *,
        operation_counter: list[int] | None = None,
        verification_counter: list[int] | None = None,
    ) -> list[Reference]:
        """Return matched references once using one constant-work update per codepoint."""

        if len(text) < SHINGLE_CODEPOINTS:
            if operation_counter is not None:
                operation_counter.append(len(text))
            if verification_counter is not None:
                verification_counter.append(0)
            return []
        rolling_hash = _rolling_key(text[:SHINGLE_CODEPOINTS])
        operations = SHINGLE_CODEPOINTS
        verifications = 0
        matched_references: set[tuple[str, str, str]] = set()
        references: list[Reference] = []
        final_start = len(text) - SHINGLE_CODEPOINTS
        for start in range(final_start + 1):
            bucket = self._patterns.get(rolling_hash)
            if bucket:
                verifications += 1
                candidate = text[start : start + SHINGLE_CODEPOINTS]
                matched = bucket.get(candidate)
                if matched is not None:
                    for reference in matched:
                        identity = (
                            reference["task"],
                            reference["benchmark_example_id"],
                            reference["benchmark_field"],
                        )
                        if identity not in matched_references:
                            matched_references.add(identity)
                            references.append(reference)
            if start == final_start:
                break
            outgoing = ord(text[start]) + 1
            incoming = ord(text[start + SHINGLE_CODEPOINTS]) + 1
            rolling_hash = (
                (rolling_hash - outgoing * self._power) * _ROLLING_BASE + incoming
            ) & _ROLLING_MASK
            operations += 1
        if operation_counter is not None:
            operation_counter.append(operations)
        if verification_counter is not None:
            verification_counter.append(verifications)
        return references


class ContaminationScanError(ValueError):
    """The checkpoint cannot prove a complete benchmark contamination scan."""


@dataclass(frozen=True)
class _JsonTraversalLimits:
    """Hard per-document bounds for recursively decoded JSON contamination probes."""

    total_bytes: int = JSON_TRAVERSAL_MAX_TOTAL_BYTES
    nodes: int = JSON_TRAVERSAL_MAX_NODES
    depth: int = JSON_TRAVERSAL_MAX_DEPTH
    decoded_strings: int = JSON_TRAVERSAL_MAX_DECODED_STRINGS

    def __post_init__(self) -> None:
        if min(self.total_bytes, self.nodes, self.depth, self.decoded_strings) < 1:
            raise ValueError("JSON traversal limits must be positive")


@dataclass
class _JsonTraversalBudget:
    limits: _JsonTraversalLimits
    total_bytes: int = 0
    nodes: int = 0
    decoded_strings: int = 0

    def claim_parse(self, text: str, *, depth: int) -> None:
        if depth > self.limits.depth:
            raise _JsonTraversalLimitExceeded("depth")
        byte_count = len(text.encode("utf-8", errors="strict"))
        if self.total_bytes + byte_count > self.limits.total_bytes:
            raise _JsonTraversalLimitExceeded("total_bytes")
        self.total_bytes += byte_count

    def claim_node(self, *, depth: int, decoded_string: bool = False) -> None:
        if depth > self.limits.depth:
            raise _JsonTraversalLimitExceeded("depth")
        if self.nodes >= self.limits.nodes:
            raise _JsonTraversalLimitExceeded("nodes")
        if decoded_string and self.decoded_strings >= self.limits.decoded_strings:
            raise _JsonTraversalLimitExceeded("decoded_strings")
        self.nodes += 1
        if decoded_string:
            self.decoded_strings += 1


class _JsonTraversalLimitExceeded(RuntimeError):
    """A hostile or oversized decoded JSON candidate exceeded a fixed scan bound."""

    def __init__(self, limit_name: str) -> None:
        self.limit_name = limit_name
        super().__init__(f"JSON traversal exhausted the {limit_name} limit")


class _JsonObjectPairs(list[tuple[str, Any]]):
    """A decoded JSON object that retains repeated keys in source order."""


def scan_checkpoint_training_data(
    checkpoint_config: Mapping[str, Any],
    suite: LoadedSuite,
    *,
    fallback_cache: BoundedShardCache,
) -> dict[str, Any]:
    """Scan every document in the checkpoint's selected training manifests.

    The configured training horizon, source ratios, shuffle, and repeat policy
    are deliberately ignored.  Contamination is a property of the complete
    checkpoint-owned train split, not only of the prefix consumed by one run.
    """

    sources, streaming = _training_sources(checkpoint_config)
    training_cache = _training_cache(streaming, fallback=fallback_cache)
    index_identity = _scan_index_identity(sources, suite)
    index_identity_sha256 = _sha256_json(index_identity)
    index_dir = training_cache.cache_dir / "contamination-scans"
    index_path = index_dir / f"{index_identity_sha256}.json"
    lock_path = training_cache.lock_dir / f"contamination-{index_identity_sha256}.lock"
    index_dir.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if index_path.is_file():
            report = _read_scan_index(
                index_path,
                expected_identity=index_identity,
                expected_identity_sha256=index_identity_sha256,
            )
            _assert_scan_identity_stable(index_identity, sources=sources, suite=suite)
            return report
        report = _scan_training_sources(
            sources,
            suite,
            training_cache=training_cache,
            index_identity_sha256=index_identity_sha256,
        )
        _assert_scan_identity_stable(index_identity, sources=sources, suite=suite)
        _write_scan_index(
            index_path,
            index_identity=index_identity,
            index_identity_sha256=index_identity_sha256,
            report=report,
        )
        return report


def _assert_scan_identity_stable(
    expected_identity: Mapping[str, Any],
    *,
    sources: list[dict[str, Any]],
    suite: LoadedSuite,
) -> None:
    """Reject evidence when producer or immutable inputs changed during use."""

    if _scan_index_identity(sources, suite) != expected_identity:
        raise ContaminationScanError(
            "contamination scan producer or immutable input identity changed during execution"
        )


def _scan_training_sources(
    sources: list[dict[str, Any]],
    suite: LoadedSuite,
    *,
    training_cache: BoundedShardCache,
    index_identity_sha256: str,
) -> dict[str, Any]:
    """Perform the one-time corpus scan used to build a reusable verified index."""

    probe_index = _build_probe_index(suite)
    matches: list[dict[str, Any]] = []
    scanned_documents = 0
    scanned_utf8_bytes = 0
    scan_digest = hashlib.sha256()
    source_evidence: list[dict[str, Any]] = []
    for source in sources:
        manifest_path = _root_path(str(source["manifest_path"]))
        manifest = preflight_manifest(
            manifest_path,
            expected_fingerprint=str(source["expected_fingerprint"]),
            selection="train",
            access="training",
            allow_reserved_benchmark=False,
            cache=training_cache,
        )
        iterator = iter(
            ManifestTextSource(
                manifest,
                cache=training_cache,
                timeout_seconds=float(source.get("timeout_seconds", 30.0)),
            )
        )
        source_documents = 0
        source_bytes = 0
        try:
            for document in iterator:
                source_documents += 1
                scanned_documents += 1
                encoded = document.text.encode("utf-8", errors="strict")
                source_bytes += len(encoded)
                scanned_utf8_bytes += len(encoded)
                metadata = document.metadata
                document_id = metadata.get("document_id")
                if not isinstance(document_id, str) or not document_id:
                    raise ContaminationScanError(
                        f"training source {source['name']!r} omitted document_id metadata"
                    )
                content_sha256 = metadata.get("content_sha256")
                if (
                    not isinstance(content_sha256, str)
                    or len(content_sha256) != 64
                    or any(character not in "0123456789abcdef" for character in content_sha256)
                ):
                    raise ContaminationScanError(
                        f"training source {source['name']!r} omitted content_sha256 metadata"
                    )
                scan_digest.update(
                    canonical_json_bytes(
                        {
                            "source": source["name"],
                            "document_id": document_id,
                            "content_sha256": content_sha256,
                        }
                    )
                )
                matches.extend(
                    _document_matches(
                        document.text,
                        source_name=str(source["name"]),
                        document_id=document_id,
                        upstream_id=metadata.get("upstream_id"),
                        probe_index=probe_index,
                    )
                )
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        source_evidence.append(
            {
                "name": str(source["name"]),
                "manifest_fingerprint": manifest.manifest_fingerprint,
                "dataset_fingerprint": manifest.dataset_fingerprint,
                "selection": manifest.selection,
                "documents": source_documents,
                "utf8_bytes": source_bytes,
            }
        )
    matches = sorted(
        _deduplicate_matches(matches),
        key=lambda item: (
            item["source"],
            item["training_document_id"],
            item["task"],
            item["benchmark_example_id"],
            item["benchmark_field"],
            item["match_type"],
        ),
    )
    counts = {
        match_type: sum(match["match_type"] == match_type for match in matches)
        for match_type in ("exact", "normalized", "structured_json", "shingle_48")
    }
    return {
        "scan_revision": SCAN_REVISION,
        "scan_index_identity_sha256": index_identity_sha256,
        "scan_complete": True,
        "benchmark_access": suite.access,
        "benchmark_examples": suite.example_count,
        "shingle_codepoints": SHINGLE_CODEPOINTS,
        "training_sources": source_evidence,
        "scanned_documents": scanned_documents,
        "scanned_utf8_bytes": scanned_utf8_bytes,
        "scanned_document_order_sha256": scan_digest.hexdigest(),
        "match_counts": counts,
        "matches": matches,
        "contaminated": bool(matches),
    }


def _scan_index_identity(sources: list[dict[str, Any]], suite: LoadedSuite) -> dict[str, Any]:
    """Bind reusable evidence to corpus, task content, producer, and scanner bytes."""

    return {
        "scan_revision": SCAN_REVISION,
        "normalization_revision": NORMALIZATION_REVISION,
        "unicode_database_version": unicodedata.unidata_version,
        "shingle_codepoints": SHINGLE_CODEPOINTS,
        "matcher_revision": MATCHER_REVISION,
        "json_object_normalization_revision": JSON_OBJECT_NORMALIZATION_REVISION,
        "implementation_sha256": {
            name: sha256_file(path) for name, path in sorted(_SCAN_IMPLEMENTATION_FILES.items())
        },
        "producer": _producer_identity(),
        "suite": suite.identity(),
        "training_sources": [
            {
                "name": str(source["name"]),
                "manifest_fingerprint": str(source["expected_fingerprint"]),
                "manifest_sha256": sha256_file(_root_path(str(source["manifest_path"]))),
                "selection": str(source["selection"]),
            }
            for source in sources
        ],
    }


def _producer_identity() -> dict[str, Any]:
    """Capture the actual source, lock, and runtime that produce cached evidence."""

    packages: dict[str, str] = {}
    for package in _PRODUCER_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError as error:
            raise ContaminationScanError(
                f"contamination producer dependency is not installed: {package}"
            ) from error
    return {
        "revision": PRODUCER_IDENTITY_REVISION,
        "source": _producer_source_identity(),
        "dependency_lock_sha256": sha256_file(ROOT_DIR / "uv.lock"),
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_cache_tag": sys.implementation.cache_tag,
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "packages": packages,
        },
    }


def _producer_source_identity() -> dict[str, Any]:
    """Hash relevant evaluator source without traversing generated model/data artifacts."""

    paths = [ROOT_DIR / "pyproject.toml", ROOT_DIR / "uv.lock"]
    paths.extend(sorted((ROOT_DIR / "src").rglob("*.py")))
    entries = [
        {
            "path": str(path.relative_to(ROOT_DIR)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    try:
        git_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ContaminationScanError(
            "cannot identify the contamination producer revision"
        ) from error
    return {
        "scope_revision": PRODUCER_SOURCE_SCOPE_REVISION,
        "git_head": git_head,
        "files": entries,
        "content_sha256": hashlib.sha256(canonical_json_bytes(entries)).hexdigest(),
    }


def _read_scan_index(
    path: Path,
    *,
    expected_identity: Mapping[str, Any],
    expected_identity_sha256: str,
) -> dict[str, Any]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContaminationScanError(f"contamination scan index is unreadable: {path}") from error
    if not isinstance(artifact, Mapping):
        raise ContaminationScanError("contamination scan index must be a JSON object")
    required = {
        "schema_version",
        "index_identity",
        "index_identity_sha256",
        "report",
        "artifact_sha256",
    }
    if set(artifact) != required or artifact.get("schema_version") != SCAN_INDEX_SCHEMA_VERSION:
        raise ContaminationScanError("contamination scan index schema is invalid")
    identity = artifact.get("index_identity")
    if (
        not isinstance(identity, Mapping)
        or dict(identity) != dict(expected_identity)
        or artifact.get("index_identity_sha256") != expected_identity_sha256
        or _sha256_json(identity) != expected_identity_sha256
    ):
        raise ContaminationScanError("contamination scan index identity does not match this run")
    unsigned = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if artifact.get("artifact_sha256") != _sha256_json(unsigned):
        raise ContaminationScanError("contamination scan index failed its artifact checksum")
    report = artifact.get("report")
    if (
        not isinstance(report, Mapping)
        or report.get("scan_complete") is not True
        or report.get("scan_index_identity_sha256") != expected_identity_sha256
    ):
        raise ContaminationScanError("contamination scan index contains an incomplete report")
    return dict(report)


def _write_scan_index(
    path: Path,
    *,
    index_identity: Mapping[str, Any],
    index_identity_sha256: str,
    report: Mapping[str, Any],
) -> None:
    unsigned = {
        "schema_version": SCAN_INDEX_SCHEMA_VERSION,
        "index_identity": dict(index_identity),
        "index_identity_sha256": index_identity_sha256,
        "report": dict(report),
    }
    artifact = {**unsigned, "artifact_sha256": _sha256_json(unsigned)}
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                artifact,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _training_sources(
    checkpoint_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    profile = checkpoint_config.get("profile")
    data = checkpoint_config.get("data")
    if not isinstance(profile, Mapping) or profile.get("purpose") != "pretraining":
        raise ContaminationScanError("benchmarking requires a pretraining checkpoint")
    if not isinstance(data, Mapping) or data.get("mode") != "streaming":
        raise ContaminationScanError("benchmarking requires checkpoint-owned streaming data")
    streaming = data.get("streaming")
    if not isinstance(streaming, Mapping):
        raise ContaminationScanError("checkpoint data.streaming must be a mapping")
    train = streaming.get("train")
    if not isinstance(train, Mapping):
        raise ContaminationScanError("checkpoint data.streaming.train must be a mapping")
    configured = train.get("sources", train.get("datasets"))
    if not isinstance(configured, list) or not configured:
        raise ContaminationScanError("checkpoint training split contains no sources")
    sources: list[dict[str, Any]] = []
    for index, source_value in enumerate(configured):
        if not isinstance(source_value, Mapping):
            raise ContaminationScanError(f"training source {index} must be a mapping")
        source = dict(source_value)
        if source.get("type", source.get("source", "hf")) != "manifest":
            raise ContaminationScanError("complete contamination scan requires manifest sources")
        required = {"name", "manifest_path", "expected_fingerprint", "selection"}
        if not required.issubset(source):
            raise ContaminationScanError(
                f"training manifest source {index} is missing {sorted(required - set(source))}"
            )
        if source["selection"] != "train":
            raise ContaminationScanError(
                "benchmarking requires every checkpoint training manifest to select train"
            )
        sources.append(source)
    return sources, streaming


def _training_cache(
    streaming: Mapping[str, Any], *, fallback: BoundedShardCache
) -> BoundedShardCache:
    cache = streaming.get("cache")
    if not isinstance(cache, Mapping) or cache.get("dir") is None:
        return fallback
    retry = streaming.get("retry", {})
    if not isinstance(retry, Mapping):
        raise ContaminationScanError("checkpoint data.streaming.retry must be a mapping")
    return BoundedShardCache(
        _root_path(str(cache["dir"])),
        max_size_bytes=int(cache.get("max_size_bytes", 1 << 30)),
        min_free_bytes=int(cache.get("min_free_bytes", 0)),
        wait_timeout_seconds=float(cache.get("wait_timeout_seconds", 30.0)),
        retry_policy=RetryPolicy(
            max_attempts=int(retry.get("max_attempts", 3)),
            initial_delay_seconds=float(retry.get("initial_delay_seconds", 0.25)),
            max_delay_seconds=float(retry.get("max_delay_seconds", 5.0)),
            multiplier=float(retry.get("multiplier", 2.0)),
        ),
    )


def _build_probe_index(suite: LoadedSuite) -> _ProbeIndex:
    exact: dict[str, list[Reference]] = {}
    normalized_index: dict[str, list[Reference]] = {}
    structured_json: dict[str, list[Reference]] = {}
    structured_json_projections: dict[tuple[str, ...], dict[str, list[Reference]]] = {}
    shingle_patterns: dict[str, list[Reference]] = {}
    for task in suite.tasks:
        for example in task.examples:
            for field, text in contamination_probes(example):
                reference = {
                    "task": task.name,
                    "benchmark_example_id": example.example_id,
                    "benchmark_field": field,
                }
                _append(exact, _sha256(text), reference)
                normalized = normalize_text_identity(text)
                _append(normalized_index, _sha256(normalized), reference)
                if field == "canonical_record":
                    _append(structured_json, _sha256(text), reference)
                    normalized_record = _normalize_decoded_json(example.record)
                    if not isinstance(normalized_record, Mapping):
                        raise ContaminationScanError("benchmark canonical record must be a mapping")
                    projections = [normalized_record]
                    omitted_fields = _INPUT_ONLY_OMITTED_FIELDS.get(task.name)
                    if omitted_fields is not None:
                        input_only = {
                            key: value
                            for key, value in normalized_record.items()
                            if key not in omitted_fields
                        }
                        if len(input_only) == len(normalized_record):
                            raise ContaminationScanError(
                                f"benchmark task {task.name} lacks its expected answer field"
                            )
                        projections.append(input_only)
                    for projection in projections:
                        required_keys = tuple(sorted(projection))
                        projection_index = structured_json_projections.setdefault(required_keys, {})
                        _append(
                            projection_index,
                            _sha256(canonical_json_bytes(projection).decode("utf-8")),
                            reference,
                        )
                for shingle in _iter_shingles(normalized):
                    _append_unique(shingle_patterns, shingle, reference)
    return _ProbeIndex(
        exact=exact,
        normalized=normalized_index,
        structured_json=structured_json,
        structured_json_projections=structured_json_projections,
        shingles=_ShingleMatcher(shingle_patterns),
    )


def _document_matches(
    text: str,
    *,
    source_name: str,
    document_id: str,
    upstream_id: Any,
    probe_index: _ProbeIndex,
) -> list[dict[str, Any]]:
    normalized = normalize_text_identity(text)
    matches: list[dict[str, Any]] = []
    observed: list[tuple[str, list[Reference]]] = [
        ("exact", probe_index.exact.get(_sha256(text), [])),
        ("normalized", probe_index.normalized.get(_sha256(normalized), [])),
        ("shingle_48", probe_index.shingles.references_in(normalized)),
    ]
    try:
        for structured_json, mapping in _canonical_json_candidates(
            normalized,
            projection_key_sets=probe_index.structured_json_projections,
        ):
            observed.append(
                (
                    "structured_json",
                    probe_index.structured_json.get(_sha256(structured_json), []),
                )
            )
            mapping_keys = set(mapping)
            for required_keys, projection_index in probe_index.structured_json_projections.items():
                if not set(required_keys).issubset(mapping_keys):
                    continue
                projected = {key: mapping[key] for key in required_keys}
                observed.append(
                    (
                        "structured_json",
                        projection_index.get(
                            _sha256(canonical_json_bytes(projected).decode("utf-8")),
                            [],
                        ),
                    )
                )
    except _JsonTraversalLimitExceeded as error:
        raise ContaminationScanError(
            "complete contamination scan aborted: "
            f"JSON traversal {error.limit_name} limit exhausted for "
            f"source={source_name!r}, document_id={document_id!r}"
        ) from error
    for match_type, references in observed:
        for reference in references:
            matches.append(
                {
                    "match_type": match_type,
                    **reference,
                    "source": source_name,
                    "training_document_id": document_id,
                    "training_upstream_id": None if upstream_id is None else str(upstream_id),
                }
            )
    return _deduplicate_matches(matches)


def _canonical_json_object(text: str) -> str | None:
    """Return a bounded key-order/whitespace-independent standalone object identity."""

    stripped = text.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return None
    decoded = _decode_json_candidate(
        stripped,
        budget=_JsonTraversalBudget(_JsonTraversalLimits()),
        depth=0,
    )
    if decoded is None or not isinstance(decoded[0], Mapping):
        return None
    try:
        return canonical_json_bytes(decoded[0]).decode("utf-8", errors="strict")
    except (RecursionError, TypeError, ValueError, UnicodeError, OverflowError):
        return None


def _normalize_decoded_json(
    value: Any,
    *,
    budget: _JsonTraversalBudget | None = None,
    depth: int = 0,
    decoded_values: list[tuple[str, int]] | None = None,
) -> Any:
    """Apply NFC under fixed bounds and reject normalized-key collisions."""

    active_budget = budget or _JsonTraversalBudget(_JsonTraversalLimits())
    active_budget.claim_node(depth=depth, decoded_string=isinstance(value, str))
    if isinstance(value, str):
        normalized_string = unicodedata.normalize("NFC", value)
        if decoded_values is not None:
            decoded_values.append((normalized_string, depth))
        return normalized_string
    if isinstance(value, list):
        return [
            _normalize_decoded_json(
                item,
                budget=active_budget,
                depth=depth + 1,
                decoded_values=decoded_values,
            )
            for item in value
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("decoded JSON object keys must be strings")
            active_budget.claim_node(depth=depth + 1, decoded_string=True)
            normalized_key = unicodedata.normalize("NFC", key)
            if normalized_key in normalized:
                raise ValueError("decoded JSON object keys collide after NFC normalization")
            if decoded_values is not None:
                decoded_values.append((normalized_key, depth + 1))
            normalized[normalized_key] = _normalize_decoded_json(
                item,
                budget=active_budget,
                depth=depth + 1,
                decoded_values=decoded_values,
            )
        return normalized
    return value


def _decode_json_candidate(
    text: str,
    *,
    budget: _JsonTraversalBudget,
    depth: int,
    projection_key_sets: Iterable[tuple[str, ...]] = (),
) -> tuple[Any | None, tuple[tuple[str, int], ...], tuple[Mapping[str, Any], ...]] | None:
    """Decode and normalize one JSON candidate without allowing parser failures to escape."""

    try:
        if depth > budget.limits.depth:
            raise _JsonTraversalLimitExceeded("depth")
        if not _json_structure_within_depth(
            text,
            max_depth=budget.limits.depth - depth,
        ):
            return None
        budget.claim_parse(text, depth=depth)
        pair_value = json.loads(text, object_pairs_hook=_JsonObjectPairs)
        value = _json_pairs_last_wins(pair_value)
        decoded_values: list[tuple[str, int]] = []
        projections = tuple(
            projection
            for required_keys in projection_key_sets
            for projection in _normalized_mapping_projections(
                pair_value,
                required_keys=required_keys,
                budget=budget,
                depth=depth,
                decoded_values=decoded_values,
            )
        )
        try:
            normalized = _normalize_decoded_json(
                value,
                budget=budget,
                depth=depth,
                decoded_values=decoded_values,
            )
        except (TypeError, ValueError, UnicodeError, OverflowError):
            normalized = None
        return normalized, tuple(decoded_values), projections
    except (
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
        UnicodeError,
        OverflowError,
    ):
        return None


def _normalized_mapping_projections(
    value: Any,
    *,
    required_keys: tuple[str, ...],
    budget: _JsonTraversalBudget,
    depth: int,
    decoded_values: list[tuple[str, int]],
) -> tuple[Mapping[str, Any], ...]:
    """Normalize every bounded schema projection without dropping duplicate keys."""

    if not isinstance(value, _JsonObjectPairs):
        return ()
    required = set(required_keys)
    options: dict[str, list[Any]] = {key: [] for key in required_keys}
    for key, item in value:
        if not isinstance(key, str):
            return ()
        normalized_key = unicodedata.normalize("NFC", key)
        if normalized_key not in required:
            continue
        options[normalized_key].append(item)
    if any(not values for values in options.values()):
        return ()

    combination_count = math.prod(len(options[key]) for key in required_keys)
    if combination_count > budget.limits.nodes - budget.nodes:
        raise _JsonTraversalLimitExceeded("nodes")
    projections: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for values in itertools.product(*(options[key] for key in required_keys)):
        selected = dict(zip(required_keys, values, strict=True))
        try:
            normalized = _normalize_decoded_json(
                selected,
                budget=budget,
                depth=depth,
                decoded_values=decoded_values,
            )
            if not isinstance(normalized, Mapping):
                continue
            identity = canonical_json_bytes(normalized).decode("utf-8", errors="strict")
        except (RecursionError, TypeError, ValueError, UnicodeError, OverflowError):
            continue
        if identity in seen:
            continue
        seen.add(identity)
        projections.append(normalized)
    return tuple(projections)


def _json_pairs_last_wins(value: Any) -> Any:
    """Reconstruct ordinary JSON semantics while retaining pairs for projections."""

    if isinstance(value, _JsonObjectPairs):
        return {key: _json_pairs_last_wins(item) for key, item in value}
    if isinstance(value, list):
        return [_json_pairs_last_wins(item) for item in value]
    return value


def _json_structure_within_depth(text: str, *, max_depth: int) -> bool:
    """Reject over-depth JSON containers before version-dependent parser recursion."""

    container_depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            elif character in "\r\n":
                return False
            continue
        if character == '"':
            in_string = True
        elif character in "{[":
            container_depth += 1
            if container_depth > max_depth:
                return False
        elif character in "}]":
            container_depth = max(0, container_depth - 1)
    return True


def _canonical_mappings(value: Any) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """Yield identities for every mapping in an already bounded normalized value."""

    pending = [value]
    while pending:
        candidate = pending.pop()
        if isinstance(candidate, Mapping):
            try:
                yield canonical_json_bytes(candidate).decode("utf-8", errors="strict"), candidate
            except (RecursionError, TypeError, ValueError, UnicodeError, OverflowError):
                pass
            pending.extend(reversed(tuple(candidate.values())))
        elif isinstance(candidate, list):
            pending.extend(reversed(candidate))


def _canonical_json_candidates(
    text: str,
    *,
    limits: _JsonTraversalLimits | None = None,
    projection_key_sets: Iterable[tuple[str, ...]] = (),
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    """Yield bounded canonical mappings, including recursively JSON-encoded strings.

    Direct candidate substrings remain disjoint and linear in each scanned text.
    Decoded JSON string values are traversed iteratively under one per-document
    byte/node/depth/string budget, so nested object, array, serialized-record,
    and quoted-prose wrappers cannot amplify work without bound.
    """

    budget = _JsonTraversalBudget(limits or _JsonTraversalLimits())
    pending: list[tuple[str, int]] = [(text, 0)]
    seen_depth: dict[str, int] = {_sha256(text): 0}
    cursor = 0
    while cursor < len(pending):
        candidate_text, decode_depth = pending[cursor]
        cursor += 1
        if decode_depth >= budget.limits.depth and _may_contain_json(candidate_text):
            raise _JsonTraversalLimitExceeded("depth")
        object_ranges: list[tuple[int, int]] = []
        decoded_object_ranges: list[tuple[int, int]] = []
        for start, end, structural_depth in _embedded_json_object_ranges(candidate_text):
            if len(object_ranges) >= budget.limits.nodes:
                raise _JsonTraversalLimitExceeded("nodes")
            object_ranges.append((start, end))
            decoded = _decode_json_candidate(
                candidate_text[start:end],
                budget=budget,
                depth=decode_depth + structural_depth,
                projection_key_sets=projection_key_sets,
            )
            if decoded is None:
                continue
            normalized, decoded_values, projections = decoded
            if normalized is not None:
                decoded_object_ranges.append((start, end))
                yield from _canonical_mappings(normalized)
            for projection in projections:
                yield canonical_json_bytes(projection).decode("utf-8", errors="strict"), projection
            _queue_decoded_json_strings(
                pending,
                seen_depth,
                decoded_values,
                limits=budget.limits,
            )
        decoded_object_ranges.sort()
        object_range_index = 0
        for start, end, structural_depth in _json_string_literal_ranges(candidate_text):
            while (
                object_range_index < len(decoded_object_ranges)
                and decoded_object_ranges[object_range_index][1] <= start
            ):
                object_range_index += 1
            if (
                object_range_index < len(decoded_object_ranges)
                and decoded_object_ranges[object_range_index][0] <= start
                and end <= decoded_object_ranges[object_range_index][1]
            ):
                continue
            decoded = _decode_json_candidate(
                candidate_text[start:end],
                budget=budget,
                depth=decode_depth + structural_depth,
            )
            if decoded is None or not isinstance(decoded[0], str):
                continue
            _queue_decoded_json_strings(
                pending,
                seen_depth,
                decoded[1],
                limits=budget.limits,
            )


def _canonical_json_objects(
    text: str,
    *,
    limits: _JsonTraversalLimits | None = None,
) -> Iterable[str]:
    """Yield only canonical object identities for diagnostics and boundary tests."""

    for canonical, _mapping in _canonical_json_candidates(text, limits=limits):
        yield canonical


def _queue_decoded_json_strings(
    pending: list[tuple[str, int]],
    seen_depth: dict[str, int],
    decoded_values: Iterable[tuple[str, int]],
    *,
    limits: _JsonTraversalLimits,
) -> None:
    for value, value_depth in decoded_values:
        next_depth = value_depth + 1
        if not _may_contain_json(value):
            continue
        if next_depth > limits.depth:
            raise _JsonTraversalLimitExceeded("depth")
        digest = _sha256(value)
        previous_depth = seen_depth.get(digest)
        if previous_depth is not None and previous_depth <= next_depth:
            continue
        seen_depth[digest] = next_depth
        pending.append((value, next_depth))


def _may_contain_json(text: str) -> bool:
    stripped = text.strip()
    return (
        "{" in text
        or "[" in text
        or (len(stripped) >= 2 and stripped.startswith('"') and stripped.endswith('"'))
    )


def _embedded_json_object_ranges(
    text: str,
) -> Iterable[tuple[int, int, int]]:
    """Yield every completed object under every malformed-prefix lexer state."""

    observed: set[tuple[int, int]] = set()
    # JSON string scanning has three states: outside a string, inside a string,
    # and inside a string immediately after an escape. Malformed prose has no
    # authoritative string grammar, so two additional prefix-only hypotheses
    # treat backslashes literally for the complete pass. Keeping those malformed
    # hypotheses active inside an incomplete outer object lets a later valid
    # nested object restart lexical recovery. Every yielded slice still has to
    # parse as exact JSON before it can match. The fixed five passes keep work
    # linear while preventing an escaped unmatched quote from hiding any object.
    for initial_in_string, initial_escaped, honors_escapes in (
        (False, False, True),
        (True, False, True),
        (True, True, True),
        (False, False, False),
        (True, False, False),
    ):
        for start, end in _json_object_candidate_ranges(
            text,
            initial_in_string=initial_in_string,
            initial_escaped=initial_escaped,
            honors_escapes=honors_escapes,
        ):
            identity = (start, end)
            if identity in observed:
                continue
            observed.add(identity)
            yield start, end, 0


def _json_object_candidate_ranges(
    text: str,
    *,
    initial_in_string: bool,
    initial_escaped: bool,
    honors_escapes: bool,
) -> Iterable[tuple[int, int]]:
    """Extract every completed object with one bounded lexical interpretation.

    The open-object stack is bounded by the document itself, whose source policy
    already caps UTF-8 size.  Yielding on close preserves intermediate mappings:
    a selected record may contain nested provenance and itself sit inside an
    over-depth envelope that the JSON parser must reject.
    """

    object_starts: list[int] = []
    in_string = initial_in_string
    escaped = initial_escaped
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\" and honors_escapes:
                escaped = True
            elif character == '"':
                in_string = False
            elif character in "\r\n":
                object_starts.clear()
                in_string = False
                escaped = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            object_starts.append(index)
        elif character == "}":
            if not object_starts:
                continue
            start = object_starts.pop()
            if not object_starts:
                in_string = False
                escaped = False
            yield start, index + 1


def _json_string_literal_ranges(
    text: str,
) -> Iterable[tuple[int, int, int]]:
    """Yield complete string literals with constant state, regardless of wrapper depth."""

    start: int | None = None
    escaped = False
    for index, character in enumerate(text):
        if start is None:
            if character == '"':
                start = index
                escaped = False
            continue
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"':
            yield start, index + 1, 0
            start = None
        elif character in "\r\n":
            start = None
            escaped = False


def _deduplicate_matches(matches: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[bytes, dict[str, Any]] = {}
    for match in matches:
        unique[canonical_json_bytes(match)] = match
    return list(unique.values())


def _iter_shingles(text: str) -> Iterable[str]:
    if len(text) < SHINGLE_CODEPOINTS:
        return
    for index in range(len(text) - SHINGLE_CODEPOINTS + 1):
        yield text[index : index + SHINGLE_CODEPOINTS]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()


def _rolling_key(text: str) -> int:
    rolling_hash = 0
    for character in text:
        value = ord(character) + 1
        rolling_hash = (rolling_hash * _ROLLING_BASE + value) & _ROLLING_MASK
    return rolling_hash


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _append(index: dict[str, list[dict[str, str]]], digest: str, reference: dict[str, str]) -> None:
    index.setdefault(digest, []).append(reference)


def _append_unique(index: dict[str, list[Reference]], pattern: str, reference: Reference) -> None:
    references = index.setdefault(pattern, [])
    if reference not in references:
        references.append(reference)


def _root_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT_DIR / path).resolve()
