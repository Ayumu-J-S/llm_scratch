"""Complete checkpoint-owned training scan for HUMAN-001 prompt occurrences."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from data.identity import canonical_json_bytes, normalize_text_identity
from data.manifests import preflight_manifest
from data.stream_loader.cache import BoundedShardCache, RetryPolicy
from data.stream_loader.loader import ManifestTextSource
from human_evaluation.schema import Prompt
from runtime.reproducibility import sha256_file


SCAN_REVISION = "HUMAN-001-prompt-contamination-v1"
NORMALIZATION_REVISION = "normalize-text-identity-v1"
MINIMUM_FREE_BYTES = 100_000_000_000
_ROOT_DIR = Path(__file__).resolve().parents[2]
_IMPLEMENTATION_FILES = {
    "human_evaluation.contamination": Path(__file__).resolve(),
    "data.identity": _ROOT_DIR / "src/data/identity.py",
    "data.manifests": _ROOT_DIR / "src/data/manifests.py",
    "data.stream_loader.cache": _ROOT_DIR / "src/data/stream_loader/cache.py",
    "data.stream_loader.loader": _ROOT_DIR / "src/data/stream_loader/loader.py",
}


class HumanPromptContaminationError(ValueError):
    """The checkpoint cannot prove complete prompt isolation from training."""


def scan_checkpoint_training_prompts(
    checkpoint_config: Mapping[str, Any],
    checkpoint_identity: Mapping[str, Any],
    prompts: Sequence[Prompt],
    *,
    prompt_set_version: str,
    prompt_set_sha256: str,
    evaluated_checkpoints: Sequence[Mapping[str, Any]],
    fallback_cache: BoundedShardCache,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Scan every document from every checkpoint-owned training manifest."""

    root = Path(repository_root).resolve()
    sources, streaming = _training_sources(checkpoint_config)
    cache = _training_cache(streaming, fallback=fallback_cache, root=root)
    free_bytes_before = shutil.disk_usage(cache.cache_dir).free
    if free_bytes_before < MINIMUM_FREE_BYTES:
        raise HumanPromptContaminationError(
            "prompt scan cannot preserve the required 100 GB free-space reserve"
        )
    prompt_identities = [
        {
            "prompt_id": prompt.id,
            "language": prompt.language,
            "exact_sha256": _sha256(prompt.text),
            "normalized_sha256": _sha256(normalize_text_identity(prompt.text)),
        }
        for prompt in prompts
    ]
    identity = {
        "scan_revision": SCAN_REVISION,
        "normalization_revision": NORMALIZATION_REVISION,
        "implementation_sha256": {
            name: sha256_file(path) for name, path in sorted(_IMPLEMENTATION_FILES.items())
        },
        "prompt_set": {
            "version": prompt_set_version,
            "sha256": prompt_set_sha256,
            "prompts": prompt_identities,
        },
        "checkpoint": copy.deepcopy(dict(checkpoint_identity)),
        "evaluated_checkpoints": _evaluated_checkpoint_identity(evaluated_checkpoints),
        "training_sources": [
            {
                "name": str(source["name"]),
                "manifest_fingerprint": str(source["expected_fingerprint"]),
                "manifest_sha256": sha256_file(_root_path(root, str(source["manifest_path"]))),
                "selection": str(source["selection"]),
            }
            for source in sources
        ],
    }
    identity_sha256 = _sha256_json(identity)
    index_dir = cache.cache_dir / "human-contamination-scans"
    index_path = index_dir / f"{identity_sha256}.json"
    index_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache.lock_dir / f"human-contamination-{identity_sha256}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if index_path.is_file():
            return _read_index(index_path, identity=identity, identity_sha256=identity_sha256)
        report = _scan_sources(
            sources,
            prompts,
            cache=cache,
            root=root,
            identity=identity,
            identity_sha256=identity_sha256,
            free_bytes_before=free_bytes_before,
        )
        if shutil.disk_usage(cache.cache_dir).free < MINIMUM_FREE_BYTES:
            raise HumanPromptContaminationError(
                "prompt scan violated the required 100 GB free-space reserve"
            )
        _write_index(
            index_path,
            identity=identity,
            identity_sha256=identity_sha256,
            report=report,
        )
        return report


def _scan_sources(
    sources: Sequence[Mapping[str, Any]],
    prompts: Sequence[Prompt],
    *,
    cache: BoundedShardCache,
    root: Path,
    identity: Mapping[str, Any],
    identity_sha256: str,
    free_bytes_before: int,
) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    source_evidence: list[dict[str, Any]] = []
    scanned_documents = 0
    scanned_utf8_bytes = 0
    scan_digest = hashlib.sha256()
    normalized_prompts = {prompt.id: normalize_text_identity(prompt.text) for prompt in prompts}
    for source in sources:
        manifest = preflight_manifest(
            _root_path(root, str(source["manifest_path"])),
            expected_fingerprint=str(source["expected_fingerprint"]),
            selection="train",
            access="training",
            allow_reserved_benchmark=False,
            cache=cache,
        )
        iterator = iter(
            ManifestTextSource(
                manifest,
                cache=cache,
                timeout_seconds=float(source.get("timeout_seconds", 30.0)),
            )
        )
        source_documents = 0
        source_bytes = 0
        try:
            for document in iterator:
                metadata = document.metadata
                document_id = metadata.get("document_id")
                content_sha256 = metadata.get("content_sha256")
                if not isinstance(document_id, str) or not document_id:
                    raise HumanPromptContaminationError(
                        f"training source {source['name']!r} omitted document_id metadata"
                    )
                if not _is_sha256(content_sha256):
                    raise HumanPromptContaminationError(
                        f"training source {source['name']!r} omitted content_sha256 metadata"
                    )
                encoded = document.text.encode("utf-8", errors="strict")
                source_documents += 1
                source_bytes += len(encoded)
                scanned_documents += 1
                scanned_utf8_bytes += len(encoded)
                scan_digest.update(
                    canonical_json_bytes(
                        {
                            "source": source["name"],
                            "document_id": document_id,
                            "content_sha256": content_sha256,
                        }
                    )
                )
                normalized_document = normalize_text_identity(document.text)
                for prompt in prompts:
                    if prompt.text in document.text:
                        matches.append(
                            _match(
                                "exact",
                                prompt=prompt,
                                source=source,
                                document_id=document_id,
                                upstream_id=metadata.get("upstream_id"),
                            )
                        )
                    if normalized_prompts[prompt.id] in normalized_document:
                        matches.append(
                            _match(
                                "normalized",
                                prompt=prompt,
                                source=source,
                                document_id=document_id,
                                upstream_id=metadata.get("upstream_id"),
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
        {canonical_json_bytes(match): match for match in matches}.values(),
        key=lambda match: (
            match["source"],
            match["training_document_id"],
            match["prompt_id"],
            match["match_type"],
        ),
    )
    return {
        "scan_revision": SCAN_REVISION,
        "scan_complete": True,
        "identity": dict(identity),
        "identity_sha256": identity_sha256,
        "training_sources": source_evidence,
        "scanned_documents": scanned_documents,
        "scanned_utf8_bytes": scanned_utf8_bytes,
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "free_bytes_before": free_bytes_before,
        "free_bytes_after": shutil.disk_usage(cache.cache_dir).free,
        "scanned_document_order_sha256": scan_digest.hexdigest(),
        "match_counts": {
            kind: sum(match["match_type"] == kind for match in matches)
            for kind in ("exact", "normalized")
        },
        "matches": matches,
        "contaminated": bool(matches),
    }


def _training_sources(
    checkpoint_config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    profile = checkpoint_config.get("profile")
    data = checkpoint_config.get("data")
    if not isinstance(profile, Mapping) or profile.get("purpose") != "pretraining":
        raise HumanPromptContaminationError("human evaluation requires a pretraining checkpoint")
    if not isinstance(data, Mapping) or data.get("mode") != "streaming":
        raise HumanPromptContaminationError(
            "human evaluation requires checkpoint-owned streaming data"
        )
    streaming = data.get("streaming")
    if not isinstance(streaming, Mapping):
        raise HumanPromptContaminationError("checkpoint data.streaming must be a mapping")
    train = streaming.get("train")
    configured = train.get("sources", train.get("datasets")) if isinstance(train, Mapping) else None
    if not isinstance(configured, list) or not configured:
        raise HumanPromptContaminationError("checkpoint training split contains no sources")
    sources: list[dict[str, Any]] = []
    for index, value in enumerate(configured):
        if not isinstance(value, Mapping):
            raise HumanPromptContaminationError(f"training source {index} must be a mapping")
        source = dict(value)
        required = {"name", "manifest_path", "expected_fingerprint", "selection"}
        if source.get("type", source.get("source", "hf")) != "manifest":
            raise HumanPromptContaminationError("complete prompt scan requires manifest sources")
        if not required.issubset(source):
            raise HumanPromptContaminationError(
                f"training source {index} is missing {sorted(required - set(source))}"
            )
        if source["selection"] != "train":
            raise HumanPromptContaminationError("every training manifest must select train")
        sources.append(source)
    return sources, streaming


def _evaluated_checkpoint_identity(
    checkpoints: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(checkpoints) != 2:
        raise HumanPromptContaminationError("prompt scan requires exactly two checkpoints")
    values: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        slot = checkpoint.get("slot")
        sha256 = checkpoint.get("sha256")
        optimizer_step = checkpoint.get("optimizer_step")
        target_tokens = checkpoint.get("target_tokens")
        if slot not in {"earlier", "later"} or not _is_sha256(sha256):
            raise HumanPromptContaminationError("evaluated checkpoint identity is invalid")
        if (
            isinstance(optimizer_step, bool)
            or not isinstance(optimizer_step, int)
            or optimizer_step < 0
            or isinstance(target_tokens, bool)
            or not isinstance(target_tokens, int)
            or target_tokens <= 0
        ):
            raise HumanPromptContaminationError("evaluated checkpoint counters are invalid")
        values.append(
            {
                "slot": slot,
                "sha256": sha256,
                "optimizer_step": optimizer_step,
                "target_tokens": target_tokens,
            }
        )
    if {value["slot"] for value in values} != {"earlier", "later"}:
        raise HumanPromptContaminationError("prompt scan checkpoint slots must be unique")
    return sorted(values, key=lambda value: value["slot"])


def _training_cache(
    streaming: Mapping[str, Any], *, fallback: BoundedShardCache, root: Path
) -> BoundedShardCache:
    value = streaming.get("cache")
    if not isinstance(value, Mapping) or value.get("dir") is None:
        return fallback
    retry = streaming.get("retry", {})
    if not isinstance(retry, Mapping):
        raise HumanPromptContaminationError("checkpoint data.streaming.retry must be a mapping")
    return BoundedShardCache(
        _root_path(root, str(value["dir"])),
        max_size_bytes=int(value.get("max_size_bytes", 1 << 30)),
        min_free_bytes=max(int(value.get("min_free_bytes", 0)), MINIMUM_FREE_BYTES),
        wait_timeout_seconds=float(value.get("wait_timeout_seconds", 30.0)),
        retry_policy=RetryPolicy(
            max_attempts=int(retry.get("max_attempts", 3)),
            initial_delay_seconds=float(retry.get("initial_delay_seconds", 0.25)),
            max_delay_seconds=float(retry.get("max_delay_seconds", 5.0)),
            multiplier=float(retry.get("multiplier", 2.0)),
        ),
    )


def _match(
    match_type: str,
    *,
    prompt: Prompt,
    source: Mapping[str, Any],
    document_id: str,
    upstream_id: Any,
) -> dict[str, Any]:
    return {
        "match_type": match_type,
        "prompt_id": prompt.id,
        "language": prompt.language,
        "source": str(source["name"]),
        "training_document_id": document_id,
        "training_upstream_id": None if upstream_id is None else str(upstream_id),
    }


def _read_index(path: Path, *, identity: Mapping[str, Any], identity_sha256: str) -> dict[str, Any]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HumanPromptContaminationError(f"prompt scan index is unreadable: {path}") from error
    if not isinstance(artifact, Mapping):
        raise HumanPromptContaminationError("prompt scan index must be an object")
    required = {"schema_version", "identity", "identity_sha256", "report", "artifact_sha256"}
    unsigned = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if (
        set(artifact) != required
        or artifact.get("schema_version") != 1
        or artifact.get("identity") != dict(identity)
        or artifact.get("identity_sha256") != identity_sha256
        or artifact.get("artifact_sha256") != _sha256_json(unsigned)
    ):
        raise HumanPromptContaminationError("prompt scan index identity/checksum is invalid")
    report = artifact.get("report")
    if (
        not isinstance(report, Mapping)
        or report.get("scan_complete") is not True
        or report.get("identity") != dict(identity)
        or report.get("identity_sha256") != identity_sha256
    ):
        raise HumanPromptContaminationError("prompt scan index contains an incomplete report")
    return dict(report)


def _write_index(
    path: Path,
    *,
    identity: Mapping[str, Any],
    identity_sha256: str,
    report: Mapping[str, Any],
) -> None:
    unsigned = {
        "schema_version": 1,
        "identity": dict(identity),
        "identity_sha256": identity_sha256,
        "report": dict(report),
    }
    artifact = {**unsigned, "artifact_sha256": _sha256_json(unsigned)}
    descriptor, temporary_name = tempfile.mkstemp(prefix=".human-scan-", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(artifact, handle, ensure_ascii=False, indent=2, sort_keys=True)
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


def _root_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="strict")).hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
