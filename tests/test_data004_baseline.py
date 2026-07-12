from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data.identity import (
    canonical_fingerprint,
    content_bound_document_id,
    normalized_content_sha256,
    stable_document_id,
)
from data.manifests import (
    ManifestError,
    load_manifest,
    preflight_manifest,
    validate_disjoint_manifests,
)
from data.parquet_source import huggingface_artifact_url
from data.quality import DocumentPolicy, QualityTracker, apply_document_policy
from data.splits import assign_split
from data.stream_loader import BoundedShardCache, CacheSpaceError, StreamLoader


TOKENIZER_CONFIG = {
    "manifest_path": "assets/tokenizers/llm-jp-v1/manifest.json",
    "expected_fingerprint": "12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b",
}


def _v2_bundle(
    tmp_path: Path,
    *,
    salt: str = "shared-data004-v1",
    texts: list[str] | None = None,
    ids: list[str | None] | None = None,
) -> tuple[Path, Path, dict]:
    parquet_path = tmp_path / "fixture.parquet"
    texts = texts or [f"English fixture document number {index}." for index in range(40)]
    ids = ids or [f"doc-{index}" for index in range(len(texts))]
    table = pa.table(
        {
            "text": texts,
            "id": ids,
            "domain": ["example.invalid"] * len(texts),
        }
    )
    pq.write_table(table, parquet_path, row_group_size=3)
    payload = parquet_path.read_bytes()
    source = {
        "kind": "hf_parquet",
        "repo_id": "owner/dataset",
        "revision": "1" * 40,
        "config_name": "sample",
        "split": "train",
        "data_files": [
            {
                "path": "data/fixture.parquet",
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
        "text_field": "text",
        "id_field": "id",
        "metadata_fields": ["domain"],
    }
    policy = {
        "version": 1,
        "language": "en",
        "max_utf8_bytes": 4096,
        "reject_controls": True,
        "reject_wrong_script": False,
    }
    manifest = {
        "schema_version": 2,
        "name": "fixture-en",
        "purpose": "pretraining",
        "source": source,
        "usage": {
            "license": "ODC-By-1.0",
            "terms_url": "https://example.invalid/terms",
        },
        "split": {
            "method": "normalized_content_sha256_v1",
            "salt": salt,
            "validation_fraction": "0.25",
        },
        "document_policy": policy,
        "dataset_fingerprint": canonical_fingerprint(
            {"name": "fixture-en", "source": source, "document_policy": policy}
        ),
    }
    manifest["manifest_fingerprint"] = canonical_fingerprint(manifest)
    manifest_path = tmp_path / f"manifest-{salt}.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path, parquet_path, manifest


def _prime_cache(
    cache: BoundedShardCache,
    manifest_path: Path,
    parquet_path: Path,
    selection: str = "train",
) -> None:
    manifest = load_manifest(manifest_path)
    resolved = preflight_manifest(
        manifest_path,
        expected_fingerprint=manifest["manifest_fingerprint"],
        selection=selection,
    )
    artifact = manifest["source"]["data_files"][0]
    url = huggingface_artifact_url(resolved, artifact)

    def copy(destination: Path) -> None:
        shutil.copyfile(parquet_path, destination)

    with cache.acquire(
        url,
        copy,
        expected_sha256=artifact["sha256"],
        expected_size_bytes=artifact["size_bytes"],
    ):
        pass


def _loader_config(tmp_path: Path, manifest_path: Path, manifest: dict, selection: str) -> dict:
    return {
        "tokenizer": TOKENIZER_CONFIG,
        "output_mode": "raw_text",
        "max_tokens": "max",
        "add_eos": False,
        "horizon": {"repeat": False, "shuffle": False},
        "cache": {
            "dir": str(tmp_path / "cache"),
            "max_size_bytes": 10_000_000,
            "min_free_bytes": 0,
            "wait_timeout_seconds": 1.0,
        },
        "datasets": [
            {
                "name": "fixture",
                "type": "manifest",
                "manifest_path": str(manifest_path),
                "expected_fingerprint": manifest["manifest_fingerprint"],
                "selection": selection,
                "ratio": 1.0,
            }
        ],
    }


def test_quality_policy_counts_injected_defects_and_utf8_truncation():
    strict_ja = DocumentPolicy(language="ja", max_utf8_bytes=7, reject_wrong_script=True)
    tracker = QualityTracker()
    assert tracker.observe(apply_document_policy("", strict_ja)) == "empty"
    assert tracker.observe(apply_document_policy("bad\u0000text", strict_ja)) == (
        "control_character"
    )
    assert tracker.observe(apply_document_policy("english only", strict_ja)) == "wrong_script"
    assert tracker.observe(apply_document_policy("bad\ud800", strict_ja)) == "invalid_unicode"
    accepted = apply_document_policy(" 日本語です。 ", strict_ja)
    assert accepted.accepted and accepted.truncated
    assert accepted.text is not None
    assert len(accepted.text.encode("utf-8")) <= 7
    assert tracker.observe(accepted) is None
    assert tracker.observe(accepted) == "duplicate"
    assert tracker.counts["truncated"] == 3


def test_v2_manifest_identity_is_strict_and_split_policy_is_shared(tmp_path):
    path, _, manifest = _v2_bundle(tmp_path)
    train = preflight_manifest(
        path,
        expected_fingerprint=manifest["manifest_fingerprint"],
        selection="train",
    )
    validation = preflight_manifest(
        path,
        expected_fingerprint=manifest["manifest_fingerprint"],
        selection="validation",
    )
    assert train.is_lazy and train.documents == ()
    validate_disjoint_manifests({"train": train}, {"validation": validation})

    other_path, _, other = _v2_bundle(tmp_path, salt="different")
    other_validation = preflight_manifest(
        other_path,
        expected_fingerprint=other["manifest_fingerprint"],
        selection="validation",
    )
    with pytest.raises(ManifestError, match="shared split policy"):
        validate_disjoint_manifests({"train": train}, {"validation": other_validation})

    mutated = dict(manifest)
    mutated["unknown"] = True
    path.write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(ManifestError, match="unknown"):
        load_manifest(path)


def test_v2_document_ids_bind_content_even_when_upstream_id_is_reused(tmp_path):
    salt = "shared-data004-v1"
    by_split: dict[str, str] = {}
    for index in range(100):
        text = f"Distinct content with reused upstream ID {index}."
        split = assign_split(
            content_sha256=normalized_content_sha256(text),
            salt=salt,
            validation_fraction="0.25",
        )
        by_split.setdefault(split, text)
        if set(by_split) == {"train", "validation"}:
            break
    assert set(by_split) == {"train", "validation"}

    texts = [by_split["train"], by_split["validation"]]
    path, parquet_path, manifest = _v2_bundle(
        tmp_path,
        salt=salt,
        texts=texts,
        ids=["reused", "reused"],
    )
    cache = BoundedShardCache(tmp_path / "cache", max_size_bytes=10_000_000)
    _prime_cache(cache, path, parquet_path)
    samples = []
    for selection in ("train", "validation"):
        config = _loader_config(tmp_path, path, manifest, selection)
        config["preserve_metadata"] = True
        samples.extend(StreamLoader(config))

    metadata = [sample["metadata"] for sample in samples]
    assert {item["upstream_id"] for item in metadata} == {"reused"}
    assert len({item["document_id"] for item in metadata}) == 2
    assert len({item["content_sha256"] for item in metadata}) == 2
    assert stable_document_id(
        source_identity="fixture-en",
        content_sha256=metadata[0]["content_sha256"],
        upstream_id="reused",
    ) == stable_document_id(
        source_identity="fixture-en",
        content_sha256=metadata[1]["content_sha256"],
        upstream_id="reused",
    )
    for item in metadata:
        assert item["document_id"] == content_bound_document_id(
            source_identity="fixture-en",
            content_sha256=item["content_sha256"],
            upstream_id="reused",
        )


def test_v2_checksum_fails_before_parquet_parsing(tmp_path, monkeypatch):
    path, parquet_path, manifest = _v2_bundle(tmp_path)
    manifest["source"]["data_files"][0]["sha256"] = "f" * 64
    manifest["dataset_fingerprint"] = canonical_fingerprint(
        {
            "name": manifest["name"],
            "source": manifest["source"],
            "document_policy": manifest["document_policy"],
        }
    )
    manifest["manifest_fingerprint"] = canonical_fingerprint(
        {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    )
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "data.parquet_source.download_url_to_path",
        lambda _url, destination, _timeout: shutil.copyfile(parquet_path, destination),
    )
    config = _loader_config(tmp_path, path, manifest, "train")
    with pytest.raises(ValueError, match="immutable identity"):
        list(StreamLoader(config))


def test_parquet_loader_reports_raw_rows_rejections_and_null_data(tmp_path):
    texts = [f"English row {index}." for index in range(8)]
    path, parquet_path, manifest = _v2_bundle(
        tmp_path,
        texts=texts,
        ids=[None, *[f"doc-{index}" for index in range(1, len(texts))]],
    )
    cache = BoundedShardCache(tmp_path / "cache", max_size_bytes=10_000_000)
    _prime_cache(cache, path, parquet_path)
    loader = StreamLoader(_loader_config(tmp_path, path, manifest, "train"))
    list(loader)

    assert loader.raw_row_counts == {"fixture": len(texts)}
    assert loader.missing_data_counts == {"fixture": {"null:id": 1}}
    assert sum(loader.rejection_counts["fixture"].values()) + loader.document_counts[
        "fixture"
    ] == len(texts)


def test_parquet_missing_required_column_is_fail_closed_and_reported(tmp_path):
    path, parquet_path, manifest = _v2_bundle(tmp_path)
    manifest["source"]["text_field"] = "missing_text"
    manifest["dataset_fingerprint"] = canonical_fingerprint(
        {
            "name": manifest["name"],
            "source": manifest["source"],
            "document_policy": manifest["document_policy"],
        }
    )
    manifest["manifest_fingerprint"] = canonical_fingerprint(
        {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    )
    path.write_text(json.dumps(manifest), encoding="utf-8")
    cache = BoundedShardCache(tmp_path / "cache", max_size_bytes=10_000_000)
    _prime_cache(cache, path, parquet_path)
    loader = StreamLoader(_loader_config(tmp_path, path, manifest, "train"))

    with pytest.raises(ManifestError, match="missing columns.*missing_text"):
        list(loader)
    assert loader.raw_row_counts == {"fixture": 0}
    assert loader.missing_data_counts == {"fixture": {"missing_column:missing_text": 1}}


def test_parquet_native_cursor_resumes_exact_suffix(tmp_path):
    path, parquet_path, manifest = _v2_bundle(tmp_path)
    cache = BoundedShardCache(tmp_path / "cache", max_size_bytes=10_000_000)
    _prime_cache(cache, path, parquet_path)
    config = _loader_config(tmp_path, path, manifest, "train")
    full = [item["text"] for item in StreamLoader(config)]
    assert len(full) > 3

    loader = StreamLoader(config)
    iterator = iter(loader)
    prefix = [next(iterator)["text"] for _ in range(2)]
    cursor = json.loads(json.dumps(loader.state_dict()))
    iterator.close()
    artifact_cursor = cursor["source_states"]["fixture"]["artifact_cursor"]
    assert set(artifact_cursor) == {
        "manifest_fingerprint",
        "artifact_index",
        "row_group_index",
        "row_offset",
    }
    resumed = [item["text"] for item in StreamLoader({**config, "cursor": cursor})]
    assert prefix + resumed == full


def test_early_parquet_stream_close_releases_lease_for_cross_instance_eviction(tmp_path):
    path, parquet_path, manifest = _v2_bundle(tmp_path)
    artifact = manifest["source"]["data_files"][0]
    cache_dir = tmp_path / "cache"
    cache = BoundedShardCache(cache_dir, max_size_bytes=artifact["size_bytes"])
    _prime_cache(cache, path, parquet_path)
    config = _loader_config(tmp_path, path, manifest, "train")
    config["cache"]["max_size_bytes"] = artifact["size_bytes"]

    loader = StreamLoader(config)
    iterator = iter(loader)
    assert next(iterator)["text"]
    assert loader.cache is not None
    assert loader.cache.telemetry["active_leases"] == 1
    iterator.close()
    assert loader.cache.telemetry["active_leases"] == 0

    replacement = b"x" * artifact["size_bytes"]
    replacement_digest = hashlib.sha256(replacement).hexdigest()
    other = BoundedShardCache(
        cache_dir,
        max_size_bytes=artifact["size_bytes"],
        wait_timeout_seconds=0.5,
    )
    with other.acquire(
        "replacement",
        lambda destination: destination.write_bytes(replacement),
        expected_sha256=replacement_digest,
        expected_size_bytes=len(replacement),
    ) as replacement_path:
        assert replacement_path.read_bytes() == replacement
    assert other.telemetry["evictions"] == 1


def test_trained_target_debt_ratio_and_packed_accounting_are_exact():
    config = {
        "tokenizer": TOKENIZER_CONFIG,
        "output_mode": "packed_sequences",
        "sequence_length": 9,
        "max_tokens": "max",
        "max_target_tokens": 64,
        "mixture_basis": "trained_targets",
        "add_eos": True,
        "horizon": {"repeat": False, "shuffle": False},
        "datasets": [
            {
                "name": "ja",
                "type": "memory",
                "ratio": 0.5,
                "documents": [{"text": "日本語の文です。"} for _ in range(20)],
            },
            {
                "name": "en",
                "type": "memory",
                "ratio": 0.5,
                "documents": [{"text": "An English sentence."} for _ in range(20)],
            },
        ],
    }
    loader = StreamLoader(config)
    windows = list(loader)
    assert loader.trained_target_counts == {"ja": 32, "en": 32}
    assert loader.packed_token_counts["target_token_count"] == 64
    assert sum(window["target_token_count"] for window in windows) == 64
    assert sum(sum(window["source_target_counts"].values()) for window in windows) == 64
    assert sum(loader.quota_truncated_fragment_counts.values()) > 0
    assert sum(loader.quota_removed_token_counts.values()) > 0

    prefix_loader = StreamLoader(config)
    iterator = iter(prefix_loader)
    prefix = [next(iterator)["input_ids"].tolist() for _ in range(3)]
    cursor = json.loads(json.dumps(prefix_loader.state_dict()))
    iterator.close()
    resumed_loader = StreamLoader({**config, "cursor": cursor})
    suffix = [window["input_ids"].tolist() for window in resumed_loader]
    assert prefix + suffix == [window["input_ids"].tolist() for window in windows]
    assert resumed_loader.trained_target_counts == {"ja": 32, "en": 32}
    assert resumed_loader.quota_truncated_fragment_counts == (
        loader.quota_truncated_fragment_counts
    )
    assert resumed_loader.quota_removed_token_counts == loader.quota_removed_token_counts


def test_cache_pre_reservation_floor_timeout_and_cross_instance_lease(tmp_path):
    payload_a = b"a" * 128
    payload_b = b"b" * 128
    digest_a = hashlib.sha256(payload_a).hexdigest()
    digest_b = hashlib.sha256(payload_b).hexdigest()
    free = shutil.disk_usage(tmp_path).free
    floor_cache = BoundedShardCache(
        tmp_path / "floor",
        max_size_bytes=1024,
        min_free_bytes=free,
        wait_timeout_seconds=0.1,
    )
    called = False

    def should_not_download(destination: Path) -> None:
        nonlocal called
        called = True
        destination.write_bytes(payload_a)

    with pytest.raises(CacheSpaceError, match="headroom"):
        with floor_cache.acquire(
            "floor",
            should_not_download,
            expected_sha256=digest_a,
            expected_size_bytes=len(payload_a),
        ):
            pass
    assert not called

    cache_dir = tmp_path / "leased"
    first = BoundedShardCache(cache_dir, len(payload_a), wait_timeout_seconds=0.2)
    second = BoundedShardCache(cache_dir, len(payload_b), wait_timeout_seconds=0.2)
    with first.acquire(
        "a",
        lambda path: path.write_bytes(payload_a),
        expected_sha256=digest_a,
        expected_size_bytes=len(payload_a),
    ):
        with pytest.raises(CacheSpaceError, match="timed out reserving"):
            with second.acquire(
                "b",
                lambda path: path.write_bytes(payload_b),
                expected_sha256=digest_b,
                expected_size_bytes=len(payload_b),
            ):
                pass
    assert first.telemetry["downloads"] == 1
