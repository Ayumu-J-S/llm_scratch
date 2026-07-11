from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from data.identity import (
    canonical_fingerprint,
    normalize_text_identity,
    normalized_content_sha256,
    stable_document_id,
)
from data.splits import DataPurpose, assign_split, dataset_fingerprint, split_fingerprint

if TYPE_CHECKING:
    from data.stream_loader.cache import BoundedShardCache


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HF_REVISION = re.compile(r"^[0-9a-f]{40}$")
_MANIFEST_FIELDS = {
    "schema_version",
    "name",
    "purpose",
    "source",
    "usage",
    "split",
    "index_path",
    "index_sha256",
    "dataset_fingerprint",
    "split_fingerprints",
    "manifest_fingerprint",
}


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedDocument:
    text: str
    document_id: str
    content_sha256: str
    assigned_split: str
    source_index: int
    upstream_id: str | None

    def metadata(self, manifest: ResolvedManifest) -> dict[str, Any]:
        return {
            "assigned_split": self.assigned_split,
            "content_sha256": self.content_sha256,
            "dataset_fingerprint": manifest.dataset_fingerprint,
            "document_id": self.document_id,
            "manifest_fingerprint": manifest.manifest_fingerprint,
            "source_index": self.source_index,
            "upstream_id": self.upstream_id,
        }


@dataclass(frozen=True)
class ResolvedManifest:
    name: str
    purpose: DataPurpose
    selection: str
    manifest_fingerprint: str
    dataset_fingerprint: str
    split_fingerprints: tuple[tuple[str, str], ...]
    documents: tuple[ResolvedDocument, ...]


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    manifest = _load_json_object(manifest_path)
    _require_exact_fields(manifest, _MANIFEST_FIELDS, "manifest")
    if manifest["schema_version"] != 1:
        raise ManifestError("manifest.schema_version must be 1")
    _require_nonempty_string(manifest["name"], "manifest.name")
    try:
        DataPurpose(manifest["purpose"])
    except (TypeError, ValueError) as error:
        raise ManifestError("manifest.purpose is invalid") from error
    _validate_usage(manifest["usage"])
    _validate_source(manifest["source"])
    _validate_split_config(manifest["split"])
    _validate_package_path(manifest["index_path"], "manifest.index_path")
    _require_sha256(manifest["index_sha256"], "manifest.index_sha256")
    _require_sha256(manifest["dataset_fingerprint"], "manifest.dataset_fingerprint")
    _validate_split_fingerprints(manifest["split_fingerprints"])
    fingerprint = manifest.pop("manifest_fingerprint")
    _require_sha256(fingerprint, "manifest.manifest_fingerprint")
    actual = canonical_fingerprint(manifest)
    if actual != fingerprint:
        raise ManifestError(f"manifest fingerprint mismatch: expected {fingerprint}, got {actual}")
    manifest["manifest_fingerprint"] = fingerprint
    return manifest


def preflight_manifest(
    manifest_path: str | Path,
    *,
    expected_fingerprint: str,
    selection: str,
    access: str = "training",
    allow_reserved_benchmark: bool = False,
    cache: BoundedShardCache | None = None,
) -> ResolvedManifest:
    _require_sha256(expected_fingerprint, "expected_fingerprint")
    path = Path(manifest_path).resolve()
    manifest = load_manifest(path)
    if manifest["manifest_fingerprint"] != expected_fingerprint:
        raise ManifestError("configured manifest fingerprint does not match the manifest")
    purpose = DataPurpose(manifest["purpose"])
    _guard_purpose(purpose, selection, access, allow_reserved_benchmark)

    index_path = _resolve_relative(path.parent, manifest["index_path"])
    _verify_file(index_path, manifest["index_sha256"], None, "document index")
    index = _load_json_object(index_path)
    _require_exact_fields(index, {"schema_version", "documents"}, "document index")
    if index["schema_version"] != 1 or not isinstance(index["documents"], list):
        raise ManifestError("document index must use schema version 1 with a documents list")
    entries = [_validate_index_entry(item) for item in index["documents"]]
    _validate_index_invariants(entries)
    if purpose is DataPurpose.PRETRAINING:
        present_splits = {entry["split"] for entry in entries}
        if present_splits != {"train", "validation"}:
            raise ManifestError(
                "pretraining manifests require non-empty train and validation splits"
            )
    actual_dataset_fingerprint = dataset_fingerprint(entries)
    if actual_dataset_fingerprint != manifest["dataset_fingerprint"]:
        raise ManifestError("dataset fingerprint does not match the document index")
    for split_name, expected in manifest["split_fingerprints"].items():
        if split_fingerprint(entries, split_name) != expected:
            raise ManifestError(f"{split_name} fingerprint does not match the document index")

    raw_documents = _resolve_source_documents(path.parent, manifest["source"], cache)
    if len(raw_documents) != len(entries):
        raise ManifestError("source document count does not match the document index")
    resolved: list[ResolvedDocument] = []
    for source_index, ((text, upstream_id), entry) in enumerate(zip(raw_documents, entries)):
        if entry["source_index"] != source_index:
            raise ManifestError("document index source_index values must be contiguous and ordered")
        content_sha256 = normalized_content_sha256(text)
        if content_sha256 != entry["content_sha256"]:
            raise ManifestError(f"document content mismatch at source index {source_index}")
        expected_id = stable_document_id(
            source_identity=manifest["name"],
            content_sha256=content_sha256,
            upstream_id=upstream_id,
        )
        if expected_id != entry["document_id"] or upstream_id != entry["upstream_id"]:
            raise ManifestError(f"document identity mismatch at source index {source_index}")
        assigned = assign_split(
            content_sha256=content_sha256,
            salt=manifest["split"]["salt"],
            validation_fraction=manifest["split"]["validation_fraction"],
        )
        if assigned != entry["split"]:
            raise ManifestError(f"split assignment mismatch at source index {source_index}")
        if selection == "all" or selection == assigned:
            resolved.append(
                ResolvedDocument(
                    text=text,
                    document_id=expected_id,
                    content_sha256=content_sha256,
                    assigned_split=assigned,
                    source_index=source_index,
                    upstream_id=upstream_id,
                )
            )
    if not resolved:
        raise ManifestError(f"manifest selection {selection!r} contains no documents")
    return ResolvedManifest(
        name=manifest["name"],
        purpose=purpose,
        selection=selection,
        manifest_fingerprint=manifest["manifest_fingerprint"],
        dataset_fingerprint=actual_dataset_fingerprint,
        split_fingerprints=tuple(sorted(manifest["split_fingerprints"].items())),
        documents=tuple(resolved),
    )


def validate_disjoint_manifests(
    train_manifests: Mapping[str, ResolvedManifest],
    validation_manifests: Mapping[str, ResolvedManifest],
) -> None:
    train_ids = {
        document.document_id
        for manifest in train_manifests.values()
        for document in manifest.documents
    }
    validation_ids = {
        document.document_id
        for manifest in validation_manifests.values()
        for document in manifest.documents
    }
    overlapping_ids = train_ids & validation_ids
    if overlapping_ids:
        raise ManifestError(f"train/validation document_id overlap: {sorted(overlapping_ids)[:3]}")

    train_content = {
        document.content_sha256
        for manifest in train_manifests.values()
        for document in manifest.documents
    }
    validation_content = {
        document.content_sha256
        for manifest in validation_manifests.values()
        for document in manifest.documents
    }
    overlapping_content = train_content & validation_content
    if overlapping_content:
        raise ManifestError(
            f"train/validation normalized-content overlap: {sorted(overlapping_content)[:3]}"
        )


def build_local_jsonl_manifest(
    *,
    source_path: str | Path,
    manifest_path: str | Path,
    index_path: str | Path,
    name: str,
    purpose: DataPurpose,
    license_name: str,
    terms_url: str,
    salt: str,
    validation_fraction: str,
    text_field: str = "text",
    id_field: str | None = "id",
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(source_path).resolve()
    manifest_destination = Path(manifest_path).resolve()
    index_destination = Path(index_path).resolve()
    package = manifest_destination.parent
    _require_within_package(source, package, "source_path")
    _require_within_package(index_destination, package, "index_path")
    source_bytes = source.read_bytes()
    documents = _read_jsonl_bytes(source_bytes, text_field=text_field, id_field=id_field)
    entries = []
    for source_index, (text, upstream_id) in enumerate(documents):
        content_sha256 = normalized_content_sha256(text)
        entries.append(
            {
                "content_sha256": content_sha256,
                "document_id": stable_document_id(
                    source_identity=name,
                    content_sha256=content_sha256,
                    upstream_id=upstream_id,
                ),
                "source_index": source_index,
                "split": assign_split(
                    content_sha256=content_sha256,
                    salt=salt,
                    validation_fraction=validation_fraction,
                ),
                "upstream_id": upstream_id,
            }
        )
    _validate_index_invariants(entries)
    index = {"schema_version": 1, "documents": entries}
    index_bytes = _canonical_pretty_json(index)
    manifest = {
        "schema_version": 1,
        "name": name,
        "purpose": purpose.value,
        "source": {
            "kind": "local_jsonl",
            "path": _relative_path(manifest_destination.parent, source),
            "size_bytes": len(source_bytes),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "text_field": text_field,
            "id_field": id_field,
        },
        "usage": {"license": license_name, "terms_url": terms_url},
        "split": {"salt": salt, "validation_fraction": validation_fraction},
        "index_path": _relative_path(manifest_destination.parent, index_destination),
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "dataset_fingerprint": dataset_fingerprint(entries),
        "split_fingerprints": {
            split: split_fingerprint(entries, split) for split in ("train", "validation")
        },
    }
    manifest["manifest_fingerprint"] = canonical_fingerprint(manifest)
    return manifest, index


def write_manifest_pair(
    manifest: Mapping[str, Any],
    index: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    index_path: str | Path,
) -> None:
    Path(index_path).write_bytes(_canonical_pretty_json(index))
    Path(manifest_path).write_bytes(_canonical_pretty_json(manifest))


def _resolve_source_documents(
    base: Path,
    source: Mapping[str, Any],
    cache: BoundedShardCache | None,
) -> list[tuple[str, str | None]]:
    kind = source["kind"]
    if kind == "local_jsonl":
        path = _resolve_relative(base, source["path"])
        try:
            source_bytes = path.read_bytes()
        except OSError as error:
            raise ManifestError(f"local source is missing: {path}") from error
        _verify_bytes(
            source_bytes,
            source["sha256"],
            source["size_bytes"],
            "local source",
        )
        return _read_jsonl_bytes(
            source_bytes,
            text_field=source["text_field"],
            id_field=source["id_field"],
        )
    if kind == "url_jsonl":
        from data.stream_loader.cache import download_url_to_path

        if cache is None:
            raise ManifestError("url_jsonl preflight requires a bounded cache")

        def downloader(destination: Path) -> None:
            download_url_to_path(source["url"], destination, source["timeout_seconds"])

        with cache.acquire(
            source["url"],
            downloader,
            expected_sha256=source["sha256"],
            expected_size_bytes=source["size_bytes"],
        ) as path:
            return _read_jsonl(path, text_field=source["text_field"], id_field=source["id_field"])
    raise ManifestError(
        "hf manifests are identity-validatable but require a DATA-004 source adapter before runtime"
    )


def _validate_source(source: Any) -> None:
    if not isinstance(source, dict):
        raise ManifestError("manifest.source must be an object")
    kind = source.get("kind")
    common = {"kind", "text_field", "id_field"}
    if kind == "local_jsonl":
        _require_exact_fields(source, common | {"path", "size_bytes", "sha256"}, "source")
        _validate_artifact(source, "source")
    elif kind == "url_jsonl":
        _require_exact_fields(
            source,
            common | {"url", "size_bytes", "sha256", "timeout_seconds"},
            "source",
        )
        _require_nonempty_string(source["url"], "source.url")
        _require_sha256(source["sha256"], "source.sha256")
        if not isinstance(source["size_bytes"], int) or source["size_bytes"] < 0:
            raise ManifestError("source.size_bytes must be a non-negative integer")
        if (
            not isinstance(source["timeout_seconds"], (int, float))
            or source["timeout_seconds"] <= 0
        ):
            raise ManifestError("source.timeout_seconds must be positive")
    elif kind == "hf":
        _require_exact_fields(
            source,
            common | {"repo_id", "revision", "config_name", "split", "data_files"},
            "source",
        )
        for field in ("repo_id", "config_name", "split"):
            _require_nonempty_string(source[field], f"source.{field}")
        if not isinstance(source["revision"], str) or not _HF_REVISION.fullmatch(
            source["revision"]
        ):
            raise ManifestError("source.revision must be an exact lowercase 40-hex HF commit")
        if not isinstance(source["data_files"], list) or not source["data_files"]:
            raise ManifestError("source.data_files must contain immutable artifact metadata")
        for artifact in source["data_files"]:
            _require_exact_fields(artifact, {"path", "size_bytes", "sha256"}, "HF artifact")
            _validate_artifact(artifact, "HF artifact")
    else:
        raise ManifestError("source.kind must be local_jsonl, url_jsonl, or hf")
    _require_nonempty_string(source["text_field"], "source.text_field")
    if source["id_field"] is not None:
        _require_nonempty_string(source["id_field"], "source.id_field")


def _validate_usage(usage: Any) -> None:
    _require_exact_fields(usage, {"license", "terms_url"}, "usage")
    _require_nonempty_string(usage["license"], "usage.license")
    _require_nonempty_string(usage["terms_url"], "usage.terms_url")


def _validate_split_config(split: Any) -> None:
    _require_exact_fields(split, {"salt", "validation_fraction"}, "split")
    _require_nonempty_string(split["salt"], "split.salt")
    # Exercise the exact decimal-string and threshold validation.
    assign_split(
        content_sha256="0" * 64,
        salt=split["salt"],
        validation_fraction=split["validation_fraction"],
    )


def _validate_split_fingerprints(value: Any) -> None:
    _require_exact_fields(value, {"train", "validation"}, "split_fingerprints")
    for name, fingerprint in value.items():
        _require_sha256(fingerprint, f"split_fingerprints.{name}")


def _validate_index_entry(entry: Any) -> dict[str, Any]:
    _require_exact_fields(
        entry,
        {"document_id", "content_sha256", "split", "source_index", "upstream_id"},
        "document index entry",
    )
    _require_sha256(entry["document_id"], "document_id")
    _require_sha256(entry["content_sha256"], "content_sha256")
    if entry["split"] not in {"train", "validation"}:
        raise ManifestError("document split must be train or validation")
    if not isinstance(entry["source_index"], int) or entry["source_index"] < 0:
        raise ManifestError("source_index must be a non-negative integer")
    if entry["upstream_id"] is not None:
        _require_nonempty_string(entry["upstream_id"], "upstream_id")
    return entry


def _validate_index_invariants(entries: Iterable[Mapping[str, Any]]) -> None:
    ids: dict[str, Mapping[str, Any]] = {}
    contents: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        document_id = entry["document_id"]
        content = entry["content_sha256"]
        if document_id in ids:
            raise ManifestError(f"duplicate document_id: {document_id}")
        if content in contents:
            raise ManifestError(f"duplicate normalized content: {content}")
        ids[document_id] = entry
        contents[content] = entry


def _guard_purpose(
    purpose: DataPurpose,
    selection: str,
    access: str,
    allow_reserved_benchmark: bool,
) -> None:
    if access not in {"training", "evaluation"}:
        raise ManifestError("access must be training or evaluation")
    if purpose in {DataPurpose.BENCHMARK_DEV, DataPurpose.BENCHMARK_RESERVED}:
        if access == "training":
            raise ManifestError("benchmark manifests cannot be opened by the training path")
        if purpose is DataPurpose.BENCHMARK_RESERVED and not allow_reserved_benchmark:
            raise ManifestError("reserved benchmark access requires an explicit evaluation grant")
    if purpose is DataPurpose.PRETRAINING and selection not in {"train", "validation"}:
        raise ManifestError("pretraining manifests require train or validation selection")
    if purpose is DataPurpose.MEMORIZATION_SMOKE and selection != "all":
        raise ManifestError("memorization_smoke requires explicit all-document selection")
    if (
        purpose in {DataPurpose.BENCHMARK_DEV, DataPurpose.BENCHMARK_RESERVED}
        and selection != "all"
    ):
        raise ManifestError("benchmark manifests require all-document selection")


def _read_jsonl(
    path: Path, *, text_field: str, id_field: str | None
) -> list[tuple[str, str | None]]:
    documents = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, object_pairs_hook=_unique_object)
            except (json.JSONDecodeError, ManifestError) as error:
                raise ManifestError(
                    f"invalid JSONL record at line {line_number}: {error}"
                ) from error
            if not isinstance(record, dict) or text_field not in record:
                raise ManifestError(
                    f"JSONL line {line_number} is missing text field {text_field!r}"
                )
            text = record[text_field]
            if not isinstance(text, str):
                raise ManifestError(f"JSONL line {line_number} text must be a string")
            normalize_text_identity(text)
            upstream_id = None if id_field is None else record.get(id_field)
            if upstream_id is not None:
                upstream_id = str(upstream_id)
                if not upstream_id:
                    raise ManifestError(f"JSONL line {line_number} has an empty upstream ID")
            documents.append((text, upstream_id))
    if not documents:
        raise ManifestError("source contains no documents")
    return documents


def _read_jsonl_bytes(
    value: bytes,
    *,
    text_field: str,
    id_field: str | None,
) -> list[tuple[str, str | None]]:
    try:
        text = value.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise ManifestError(f"source is not valid UTF-8: {error}") from error
    documents = []
    for line_number, line in enumerate(io.StringIO(text, newline=""), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, ManifestError) as error:
            raise ManifestError(f"invalid JSONL record at line {line_number}: {error}") from error
        if not isinstance(record, dict) or text_field not in record:
            raise ManifestError(f"JSONL line {line_number} is missing text field {text_field!r}")
        document_text = record[text_field]
        if not isinstance(document_text, str):
            raise ManifestError(f"JSONL line {line_number} text must be a string")
        normalize_text_identity(document_text)
        upstream_id = None if id_field is None else record.get(id_field)
        if upstream_id is not None:
            upstream_id = str(upstream_id)
            if not upstream_id:
                raise ManifestError(f"JSONL line {line_number} has an empty upstream ID")
        documents.append((document_text, upstream_id))
    if not documents:
        raise ManifestError("source contains no documents")
    return documents


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file, object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise ManifestError(f"{path} must contain a JSON object")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _require_exact_fields(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ManifestError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ManifestError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _require_nonempty_string(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{field} must be a non-empty string")


def _require_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ManifestError(f"{field} must be a lowercase SHA-256 hex digest")


def _validate_artifact(value: Mapping[str, Any], label: str) -> None:
    _require_nonempty_string(value["path"], f"{label}.path")
    _validate_package_path(value["path"], f"{label}.path")
    _require_sha256(value["sha256"], f"{label}.sha256")
    if not isinstance(value["size_bytes"], int) or value["size_bytes"] < 0:
        raise ManifestError(f"{label}.size_bytes must be a non-negative integer")


def _verify_file(path: Path, expected_sha256: str, expected_size: int | None, label: str) -> None:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ManifestError(f"{label} is missing: {path}") from error
    if expected_size is not None and size != expected_size:
        raise ManifestError(f"{label} size mismatch: expected {expected_size}, got {size}")
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ManifestError(f"{label} checksum mismatch")


def _verify_bytes(
    value: bytes,
    expected_sha256: str,
    expected_size: int | None,
    label: str,
) -> None:
    if expected_size is not None and len(value) != expected_size:
        raise ManifestError(f"{label} size mismatch: expected {expected_size}, got {len(value)}")
    if hashlib.sha256(value).hexdigest() != expected_sha256:
        raise ManifestError(f"{label} checksum mismatch")


def _resolve_relative(base: Path, value: Any) -> Path:
    _require_nonempty_string(value, "path")
    _validate_package_path(value, "path")
    path = Path(value)
    resolved_base = base.resolve()
    resolved = (resolved_base / path).resolve()
    _require_within_package(resolved, resolved_base, "path")
    return resolved


def _relative_path(base: Path, target: Path) -> str:
    return str(target.relative_to(base))


def _validate_package_path(value: Any, field: str) -> None:
    _require_nonempty_string(value, field)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ManifestError(f"{field} must stay within the manifest package")


def _require_within_package(path: Path, package: Path, field: str) -> None:
    try:
        path.resolve().relative_to(package.resolve())
    except ValueError as error:
        raise ManifestError(f"{field} must stay within the manifest package") from error


def _canonical_pretty_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
