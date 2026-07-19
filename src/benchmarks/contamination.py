"""Complete checkpoint-owned training split contamination scan."""

from __future__ import annotations

import hashlib
import fcntl
import json
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
SCAN_REVISION = "BENCH-001-contamination-v6"
NORMALIZATION_REVISION = "normalize-text-identity-nfc-strip-plus-json-object-v4"
SCAN_INDEX_SCHEMA_VERSION = 2
MATCHER_REVISION = "collision-verified-rolling-hash-codepoint-v1"
JSON_OBJECT_NORMALIZATION_REVISION = "normalized-embedded-canonical-json-object-sha256-v3"
PRODUCER_IDENTITY_REVISION = "contamination-producer-v1"
PRODUCER_SOURCE_SCOPE_REVISION = "src-python-pyproject-lock-v1"
_PRODUCER_PACKAGES = ("pyarrow",)
_ROLLING_BASE = 1_000_000_007
_ROLLING_MASK = (1 << 64) - 1
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
            return _read_scan_index(
                index_path,
                expected_identity=index_identity,
                expected_identity_sha256=index_identity_sha256,
            )
        report = _scan_training_sources(
            sources,
            suite,
            training_cache=training_cache,
            index_identity_sha256=index_identity_sha256,
        )
        _write_scan_index(
            index_path,
            index_identity=index_identity,
            index_identity_sha256=index_identity_sha256,
            report=report,
        )
        return report


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
                for shingle in _iter_shingles(normalized):
                    _append_unique(shingle_patterns, shingle, reference)
    return _ProbeIndex(
        exact=exact,
        normalized=normalized_index,
        structured_json=structured_json,
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
    for structured_json in _canonical_json_objects(normalized):
        observed.append(
            (
                "structured_json",
                probe_index.structured_json.get(_sha256(structured_json), []),
            )
        )
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
    """Return a key-order/whitespace-independent identity for standalone JSON objects."""

    stripped = text.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        return None
    try:
        value = json.loads(stripped)
        if not isinstance(value, Mapping):
            return None
        return canonical_json_bytes(value).decode("utf-8", errors="strict")
    except (json.JSONDecodeError, TypeError, ValueError, UnicodeError):
        return None


def _canonical_json_objects(text: str) -> Iterable[str]:
    """Yield canonical innermost JSON objects embedded in an arbitrary document.

    Benchmark source records are flat objects. Scanning only innermost balanced
    objects therefore detects a complete record both directly and inside JSON
    or prose wrappers while keeping candidate substrings disjoint: parsing and
    candidate materialization remain linear in the document size.
    """

    starts: list[int] = []
    has_nested_object: list[bool] = []
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if not starts:
            if character == "{":
                starts.append(index)
                has_nested_object.append(False)
                in_string = False
                escaped = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            elif character in "\r\n":
                # A literal newline cannot occur in a JSON string. Recover from
                # malformed prose so it cannot hide a later valid record.
                starts.clear()
                has_nested_object.clear()
                in_string = False
                escaped = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            has_nested_object[-1] = True
            starts.append(index)
            has_nested_object.append(False)
        elif character == "}":
            start = starts.pop()
            nested = has_nested_object.pop()
            if not nested:
                canonical = _canonical_json_object(text[start : index + 1])
                if canonical is not None:
                    yield canonical
            if not starts:
                in_string = False
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
