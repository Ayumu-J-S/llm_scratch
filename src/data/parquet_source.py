from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pyarrow.parquet as pq

from data.identity import stable_document_id
from data.manifests import ManifestError, ResolvedManifest
from data.quality import apply_document_policy
from data.splits import assign_split
from data.stream_loader.cache import BoundedShardCache, download_url_to_path


def huggingface_artifact_url(manifest: ResolvedManifest, artifact: Mapping[str, Any]) -> str:
    if manifest.source is None:
        raise ValueError("artifact URLs require a lazy manifest")
    repo = quote(str(manifest.source["repo_id"]), safe="/")
    revision = str(manifest.source["revision"])
    path = quote(str(artifact["path"]), safe="/")
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


class ParquetManifestSource:
    def __init__(
        self,
        manifest: ResolvedManifest,
        cache: BoundedShardCache,
        *,
        cursor: Mapping[str, Any] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not manifest.is_lazy or manifest.source is None or manifest.split_policy is None:
            raise ValueError("Parquet source requires a resolved lazy manifest")
        if manifest.document_policy is None:
            raise ValueError("Parquet source requires a document policy")
        self.manifest = manifest
        self.cache = cache
        self.cursor = cursor
        self.timeout_seconds = timeout_seconds

    def __iter__(self) -> ParquetManifestIterator:
        return ParquetManifestIterator(
            self.manifest,
            self.cache,
            cursor=self.cursor,
            timeout_seconds=self.timeout_seconds,
        )


class ParquetManifestIterator(Iterator[Any]):
    def __init__(
        self,
        manifest: ResolvedManifest,
        cache: BoundedShardCache,
        *,
        cursor: Mapping[str, Any] | None,
        timeout_seconds: float,
    ) -> None:
        # Import here to avoid a loader -> adapter -> loader module cycle at import time.
        from data.stream_loader.loader import RawDocument

        self._raw_document_type = RawDocument
        self.manifest = manifest
        self.cache = cache
        self.timeout_seconds = timeout_seconds
        self.cursor = _initial_cursor(manifest) if cursor is None else _validate_cursor(manifest, cursor)
        self.rejection_counts: dict[str, int] = {}
        self.fallback_count = 0
        self.document_count = 0
        self.text_bytes = 0
        self.truncated_count = 0
        self._closed = False
        self._rows = self._iter_rows()

    def __iter__(self) -> ParquetManifestIterator:
        return self

    def __next__(self) -> Any:
        if self._closed:
            raise StopIteration
        return next(self._rows)

    def close(self) -> None:
        """Release an open shard lease when a bounded consumer stops early."""

        if self._closed:
            return
        self._closed = True
        self._rows.close()

    def _iter_rows(self) -> Iterator[Any]:
        assert self.manifest.source is not None
        assert self.manifest.split_policy is not None
        assert self.manifest.document_policy is not None
        artifacts = self.manifest.source["data_files"]
        text_field = self.manifest.source["text_field"]
        id_field = self.manifest.source["id_field"]
        metadata_fields = list(self.manifest.source["metadata_fields"])
        columns = list(dict.fromkeys([text_field, *metadata_fields, *([id_field] if id_field else [])]))
        start_artifact = int(self.cursor["artifact_index"])
        for artifact_index in range(start_artifact, len(artifacts)):
            artifact = artifacts[artifact_index]
            url = huggingface_artifact_url(self.manifest, artifact)

            def downloader(path: Path) -> None:
                download_url_to_path(url, path, self.timeout_seconds)

            with self.cache.acquire(
                url,
                downloader,
                expected_sha256=artifact["sha256"],
                expected_size_bytes=artifact["size_bytes"],
            ) as path:
                parquet = pq.ParquetFile(path)
                missing = sorted(set(columns) - set(parquet.schema_arrow.names))
                if missing:
                    raise ManifestError(
                        f"Parquet artifact {artifact['path']} is missing columns: {missing}"
                    )
                first_group = int(self.cursor["row_group_index"]) if artifact_index == start_artifact else 0
                for row_group_index in range(first_group, parquet.num_row_groups):
                    table = parquet.read_row_group(row_group_index, columns=columns)
                    first_row = (
                        int(self.cursor["row_offset"])
                        if artifact_index == start_artifact and row_group_index == first_group
                        else 0
                    )
                    for row_offset in range(first_row, table.num_rows):
                        row = {name: table[name][row_offset].as_py() for name in columns}
                        self.cursor = _next_cursor(
                            self.manifest,
                            artifacts,
                            parquet.num_row_groups,
                            artifact_index,
                            row_group_index,
                            row_offset,
                            table.num_rows,
                        )
                        result = apply_document_policy(row[text_field], self.manifest.document_policy)
                        if not result.accepted:
                            assert result.reason is not None
                            self._count_rejection(result.reason)
                            continue
                        assert result.text is not None and result.content_sha256 is not None
                        assigned = assign_split(
                            content_sha256=result.content_sha256,
                            salt=self.manifest.split_policy["salt"],
                            validation_fraction=self.manifest.split_policy["validation_fraction"],
                        )
                        if assigned != self.manifest.selection:
                            self._count_rejection(f"other_split_{assigned}")
                            continue
                        upstream_id = row.get(id_field) if id_field else None
                        if upstream_id is None:
                            self.fallback_count += 1
                        else:
                            upstream_id = str(upstream_id)
                        document_id = stable_document_id(
                            source_identity=self.manifest.name,
                            content_sha256=result.content_sha256,
                            upstream_id=upstream_id,
                        )
                        metadata = {field: row.get(field) for field in metadata_fields}
                        metadata.update(
                            {
                                "artifact_path": artifact["path"],
                                "assigned_split": assigned,
                                "content_sha256": result.content_sha256,
                                "dataset_fingerprint": self.manifest.dataset_fingerprint,
                                "document_id": document_id,
                                "fallback_id": upstream_id is None,
                                "japanese_count": result.japanese_count,
                                "latin_count": result.latin_count,
                                "manifest_fingerprint": self.manifest.manifest_fingerprint,
                                "row_group_index": row_group_index,
                                "row_offset": row_offset,
                                "truncated": result.truncated,
                                "upstream_id": upstream_id,
                            }
                        )
                        self.document_count += 1
                        self.text_bytes += len(result.text.encode("utf-8"))
                        self.truncated_count += int(result.truncated)
                        yield self._raw_document_type(text=result.text, metadata=metadata)
        self.cursor = {
            "manifest_fingerprint": self.manifest.manifest_fingerprint,
            "artifact_index": len(artifacts),
            "row_group_index": 0,
            "row_offset": 0,
        }

    def _count_rejection(self, reason: str) -> None:
        self.rejection_counts[reason] = self.rejection_counts.get(reason, 0) + 1


def _initial_cursor(manifest: ResolvedManifest) -> dict[str, Any]:
    return {
        "manifest_fingerprint": manifest.manifest_fingerprint,
        "artifact_index": 0,
        "row_group_index": 0,
        "row_offset": 0,
    }


def _validate_cursor(
    manifest: ResolvedManifest, cursor: Mapping[str, Any]
) -> dict[str, Any]:
    expected = {"manifest_fingerprint", "artifact_index", "row_group_index", "row_offset"}
    if set(cursor) != expected:
        raise ValueError("Parquet cursor fields differ from the v1 contract")
    if cursor["manifest_fingerprint"] != manifest.manifest_fingerprint:
        raise ValueError("Parquet cursor manifest fingerprint does not match")
    result = dict(cursor)
    for field in ("artifact_index", "row_group_index", "row_offset"):
        if isinstance(result[field], bool) or not isinstance(result[field], int) or result[field] < 0:
            raise ValueError(f"Parquet cursor {field} must be a non-negative integer")
    assert manifest.source is not None
    if result["artifact_index"] > len(manifest.source["data_files"]):
        raise ValueError("Parquet cursor artifact_index is out of range")
    return result


def _next_cursor(
    manifest: ResolvedManifest,
    artifacts: list[Mapping[str, Any]],
    num_row_groups: int,
    artifact_index: int,
    row_group_index: int,
    row_offset: int,
    rows_in_group: int,
) -> dict[str, Any]:
    next_artifact = artifact_index
    next_group = row_group_index
    next_row = row_offset + 1
    if next_row >= rows_in_group:
        next_group += 1
        next_row = 0
    if next_group >= num_row_groups:
        next_artifact += 1
        next_group = 0
    if next_artifact > len(artifacts):
        next_artifact = len(artifacts)
    return {
        "manifest_fingerprint": manifest.manifest_fingerprint,
        "artifact_index": next_artifact,
        "row_group_index": next_group,
        "row_offset": next_row,
    }
