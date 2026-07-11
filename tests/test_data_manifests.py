from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split

import data.manifests as manifests_module
import data.stream_loader.cache as cache_module
from data.identity import (
    canonical_fingerprint,
    normalize_text_identity,
    normalized_content_sha256,
    stable_document_id,
)
from data.manifests import (
    ManifestError,
    build_local_jsonl_manifest,
    load_manifest,
    preflight_manifest,
)
from data.splits import DataPurpose, assign_split, dataset_fingerprint, split_fingerprint
from data.stream_loader import BoundedShardCache, StreamLoader
from data.streaming_dataset import StreamingTokenDataset


FIXTURES = Path(__file__).parent / "fixtures" / "data_manifests"
BILINGUAL_MANIFEST = FIXTURES / "bilingual.manifest.json"
BILINGUAL_FINGERPRINT = "47cca88c4a5595e27eb5d60d99918fb77c30b23f7c0ae98024153f25e14ffc19"
MEMORIZATION_FINGERPRINT = "00c3797a7d0eda13950fd699a60c45fcd388829f016479caaeb369438767bd31"
BENCHMARK_FINGERPRINT = "a12e307b8c3817efe956d8d37c55edcde56f30cf695e83a9e60155ff5949eb79"


def character_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")
    tokenizer.decoder = Fuse()
    return tokenizer


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _copy_manifest_bundle(tmp_path: Path, source: Path = BILINGUAL_MANIFEST) -> tuple[Path, dict]:
    manifest = json.loads(source.read_text(encoding="utf-8"))
    for key in ("source",):
        original = source.parent / manifest[key]["path"]
        copied = tmp_path / original.name
        copied.write_bytes(original.read_bytes())
        manifest[key]["path"] = copied.name
    original_index = source.parent / manifest["index_path"]
    copied_index = tmp_path / original_index.name
    copied_index.write_bytes(original_index.read_bytes())
    manifest["index_path"] = copied_index.name
    manifest["manifest_fingerprint"] = canonical_fingerprint(
        {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    )
    destination = tmp_path / source.name
    _write_json(destination, manifest)
    return destination, manifest


def _cache_process_worker(
    cache_dir: str,
    log_path: str,
    payload: bytes,
    digest: str,
) -> None:
    cache = BoundedShardCache(cache_dir, max_size_bytes=4096)

    def downloader(path: Path) -> None:
        with Path(log_path).open("a", encoding="utf-8") as log:
            log.write("download\n")
        time.sleep(0.15)
        path.write_bytes(payload)

    with cache.acquire(
        "https://example.invalid/concurrent",
        downloader,
        expected_sha256=digest,
        expected_size_bytes=len(payload),
    ) as path:
        assert path.read_bytes() == payload


def test_text_identity_normalization_is_bounded_and_utf8_strict():
    assert normalize_text_identity("\ufeff  e\u0301\r\nline\r  ") == "é\nline"
    assert normalized_content_sha256("\ufeff  e\u0301\r\nline\r  ") == (
        normalized_content_sha256("é\nline")
    )
    assert normalized_content_sha256("A  B") != normalized_content_sha256("a  B")
    assert normalized_content_sha256("A  B") != normalized_content_sha256("A B")
    with pytest.raises(UnicodeEncodeError):
        normalize_text_identity("bad\ud800")


def test_upstream_and_content_derived_ids_are_stable():
    content = normalized_content_sha256("document")
    assert stable_document_id(
        source_identity="source", content_sha256=content, upstream_id="7"
    ) == stable_document_id(source_identity="source", content_sha256=content, upstream_id=7)
    assert stable_document_id(
        source_identity="source", content_sha256=content, upstream_id=None
    ) == stable_document_id(source_identity="source", content_sha256=content, upstream_id=None)


def test_split_is_content_based_and_decimal_fraction_is_strict():
    content = normalized_content_sha256("stable")
    first = assign_split(content_sha256=content, salt="fixed", validation_fraction="0.2")
    assert first == assign_split(content_sha256=content, salt="fixed", validation_fraction="0.20")
    with pytest.raises(TypeError, match="decimal string"):
        assign_split(content_sha256=content, salt="fixed", validation_fraction=0.2)  # type: ignore[arg-type]


def test_committed_manifest_is_strict_and_disjoint():
    train = preflight_manifest(
        BILINGUAL_MANIFEST,
        expected_fingerprint=BILINGUAL_FINGERPRINT,
        selection="train",
    )
    validation = preflight_manifest(
        BILINGUAL_MANIFEST,
        expected_fingerprint=BILINGUAL_FINGERPRINT,
        selection="validation",
    )
    assert {document.document_id for document in train.documents}.isdisjoint(
        document.document_id for document in validation.documents
    )
    assert {document.content_sha256 for document in train.documents}.isdisjoint(
        document.content_sha256 for document in validation.documents
    )
    assert train.dataset_fingerprint == validation.dataset_fingerprint


def test_manifest_schema_and_fingerprint_mutations_fail(tmp_path):
    path, manifest = _copy_manifest_bundle(tmp_path)
    manifest["unknown"] = True
    _write_json(path, manifest)
    with pytest.raises(ManifestError, match="unknown"):
        load_manifest(path)

    path, manifest = _copy_manifest_bundle(tmp_path)
    manifest["name"] = "mutated"
    _write_json(path, manifest)
    with pytest.raises(ManifestError, match="fingerprint mismatch"):
        load_manifest(path)


def test_hf_source_requires_exact_revision_and_immutable_artifacts(tmp_path):
    path, manifest = _copy_manifest_bundle(tmp_path)
    manifest["source"] = {
        "kind": "hf",
        "repo_id": "owner/dataset",
        "revision": "1" * 40,
        "config_name": "default",
        "split": "train",
        "data_files": [{"path": "train-000.parquet", "size_bytes": 12, "sha256": "2" * 64}],
        "text_field": "text",
        "id_field": "id",
    }
    manifest["manifest_fingerprint"] = canonical_fingerprint(
        {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    )
    _write_json(path, manifest)
    assert load_manifest(path)["source"]["revision"] == "1" * 40

    manifest["source"]["revision"] = "main"
    manifest["manifest_fingerprint"] = canonical_fingerprint(
        {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    )
    _write_json(path, manifest)
    with pytest.raises(ManifestError, match="40-hex"):
        load_manifest(path)

    manifest["source"]["revision"] = "1" * 40
    manifest["source"]["data_files"][0]["sha256"] = "moving-tag"
    manifest["manifest_fingerprint"] = canonical_fingerprint(
        {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    )
    _write_json(path, manifest)
    with pytest.raises(ManifestError, match="SHA-256"):
        load_manifest(path)


def test_manifest_package_rejects_absolute_traversal_and_builder_escape(tmp_path):
    path, manifest = _copy_manifest_bundle(tmp_path)
    manifest["source"]["path"] = str((tmp_path / "bilingual.jsonl").resolve())
    manifest["manifest_fingerprint"] = canonical_fingerprint(
        {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    )
    _write_json(path, manifest)
    with pytest.raises(ManifestError, match="manifest package"):
        load_manifest(path)

    path, manifest = _copy_manifest_bundle(tmp_path)
    manifest["index_path"] = "../outside.index.json"
    manifest["manifest_fingerprint"] = canonical_fingerprint(
        {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    )
    _write_json(path, manifest)
    with pytest.raises(ManifestError, match="manifest package"):
        load_manifest(path)

    outside = tmp_path.parent / "outside.jsonl"
    outside.write_text('{"id":"1","text":"outside"}\n', encoding="utf-8")
    with pytest.raises(ManifestError, match="source_path"):
        build_local_jsonl_manifest(
            source_path=outside,
            manifest_path=tmp_path / "manifest.json",
            index_path=tmp_path / "index.json",
            name="escape",
            purpose=DataPurpose.PRETRAINING,
            license_name="CC0-1.0",
            terms_url="https://example.invalid/terms",
            salt="salt",
            validation_fraction="0.5",
        )


def test_source_path_size_and_checksum_mutations_fail(tmp_path):
    path, manifest = _copy_manifest_bundle(tmp_path)
    source = tmp_path / manifest["source"]["path"]
    source.write_text(source.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="size mismatch"):
        preflight_manifest(
            path,
            expected_fingerprint=manifest["manifest_fingerprint"],
            selection="train",
        )

    path, manifest = _copy_manifest_bundle(tmp_path)
    manifest["source"]["path"] = "missing.jsonl"
    manifest["manifest_fingerprint"] = canonical_fingerprint(
        {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    )
    _write_json(path, manifest)
    with pytest.raises(ManifestError, match="missing"):
        preflight_manifest(
            path,
            expected_fingerprint=manifest["manifest_fingerprint"],
            selection="train",
        )


def test_index_source_and_split_mutations_fail(tmp_path):
    path, manifest = _copy_manifest_bundle(tmp_path)
    index_path = tmp_path / manifest["index_path"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["documents"][0]["split"] = (
        "validation" if index["documents"][0]["split"] == "train" else "train"
    )
    _write_json(index_path, index)
    manifest["index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    manifest["manifest_fingerprint"] = canonical_fingerprint(
        {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    )
    _write_json(path, manifest)
    with pytest.raises(ManifestError, match="dataset fingerprint"):
        preflight_manifest(
            path,
            expected_fingerprint=manifest["manifest_fingerprint"],
            selection="train",
        )


def test_duplicate_id_and_content_are_rejected(tmp_path):
    source = tmp_path / "duplicate.jsonl"
    source.write_text(
        '{"id":"same","text":"one"}\n{"id":"same","text":"two"}\n',
        encoding="utf-8",
    )
    # Reusing an upstream ID creates the same stable document ID despite different content.
    with pytest.raises(ManifestError, match="duplicate document_id"):
        build_local_jsonl_manifest(
            source_path=source,
            manifest_path=tmp_path / "manifest.json",
            index_path=tmp_path / "index.json",
            name="duplicates",
            purpose=DataPurpose.PRETRAINING,
            license_name="CC0-1.0",
            terms_url="https://example.invalid/terms",
            salt="salt",
            validation_fraction="0.5",
        )


def test_reorder_preserves_ids_membership_and_fingerprints(tmp_path):
    records = [json.loads(line) for line in (FIXTURES / "bilingual.jsonl").read_text().splitlines()]
    reversed_source = tmp_path / "reversed.jsonl"
    reversed_source.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in reversed(records)) + "\n",
        encoding="utf-8",
    )
    original_source = tmp_path / "original.jsonl"
    original_source.write_bytes((FIXTURES / "bilingual.jsonl").read_bytes())
    _, original_index = build_local_jsonl_manifest(
        source_path=original_source,
        manifest_path=tmp_path / "a.json",
        index_path=tmp_path / "a.index.json",
        name="fixture-bilingual-v1",
        purpose=DataPurpose.PRETRAINING,
        license_name="CC0-1.0",
        terms_url="https://example.invalid/terms",
        salt="llm-scratch-data-002-v1",
        validation_fraction="0.25",
    )
    _, reversed_index = build_local_jsonl_manifest(
        source_path=reversed_source,
        manifest_path=tmp_path / "b.json",
        index_path=tmp_path / "b.index.json",
        name="fixture-bilingual-v1",
        purpose=DataPurpose.PRETRAINING,
        license_name="CC0-1.0",
        terms_url="https://example.invalid/terms",
        salt="llm-scratch-data-002-v1",
        validation_fraction="0.25",
    )
    assert dataset_fingerprint(original_index["documents"]) == dataset_fingerprint(
        reversed_index["documents"]
    )
    for split in ("train", "validation"):
        assert split_fingerprint(original_index["documents"], split) == split_fingerprint(
            reversed_index["documents"], split
        )


def test_same_corpus_requires_explicit_memorization_purpose():
    with pytest.raises(ManifestError, match="train or validation"):
        preflight_manifest(
            BILINGUAL_MANIFEST,
            expected_fingerprint=BILINGUAL_FINGERPRINT,
            selection="all",
        )
    smoke = preflight_manifest(
        FIXTURES / "memorization.manifest.json",
        expected_fingerprint=MEMORIZATION_FINGERPRINT,
        selection="all",
    )
    assert len(smoke.documents) == 2


def test_benchmark_guard_runs_before_source_open(tmp_path):
    path, manifest = _copy_manifest_bundle(tmp_path, FIXTURES / "benchmark_reserved.manifest.json")
    (tmp_path / manifest["source"]["path"]).unlink()
    with pytest.raises(ManifestError, match="benchmark manifests"):
        preflight_manifest(
            path,
            expected_fingerprint=manifest["manifest_fingerprint"],
            selection="all",
            access="training",
        )
    with pytest.raises(ManifestError, match="explicit evaluation grant"):
        preflight_manifest(
            path,
            expected_fingerprint=manifest["manifest_fingerprint"],
            selection="all",
            access="evaluation",
        )


def test_manifest_loader_propagates_metadata_and_preserves_text(monkeypatch):
    calls = 0
    original = manifests_module.normalized_content_sha256

    def counted(text: str) -> str:
        nonlocal calls
        calls += 1
        return original(text)

    monkeypatch.setattr(manifests_module, "normalized_content_sha256", counted)
    config = {
        "output_mode": "raw_text",
        "max_tokens": "max",
        "add_eos": False,
        "preserve_metadata": True,
        "require_manifests": True,
        "prefetch": {"enabled": True, "mode": "thread", "buffer_size": 2},
        "datasets": [
            {
                "name": "fixture",
                "type": "manifest",
                "manifest_path": str(BILINGUAL_MANIFEST),
                "expected_fingerprint": BILINGUAL_FINGERPRINT,
                "selection": "train",
                "ratio": 1.0,
            }
        ],
    }
    loader = StreamLoader(config, tokenizer=character_tokenizer())
    assert calls == 20
    samples = list(loader)
    assert calls == 20
    source_texts = {
        json.loads(line)["text"]
        for line in (FIXTURES / "bilingual.jsonl").read_text(encoding="utf-8").splitlines()
    }
    assert all(sample["text"] in source_texts for sample in samples)
    assert all("document_id" in sample["metadata"] for sample in samples)


def test_streaming_dataset_preflights_only_once_across_two_epochs(monkeypatch):
    calls = 0
    original = manifests_module.normalized_content_sha256

    def counted(text: str) -> str:
        nonlocal calls
        calls += 1
        return original(text)

    monkeypatch.setattr(manifests_module, "normalized_content_sha256", counted)
    dataset = StreamingTokenDataset(
        config={
            "max_tokens": "max",
            "add_eos": False,
            "require_manifests": True,
            "sources": [
                {
                    "name": "fixture",
                    "type": "manifest",
                    "manifest_path": str(BILINGUAL_MANIFEST),
                    "expected_fingerprint": BILINGUAL_FINGERPRINT,
                    "selection": "train",
                    "ratio": 1.0,
                }
            ],
        },
        sequence_length=8,
        tokenizer=character_tokenizer(),
    )
    assert calls == 20
    assert list(dataset)
    assert list(dataset)
    assert calls == 20


def test_stream_loader_rejects_hydra_benchmark_authority_bypass():
    with pytest.raises(ValueError, match="cannot configure evaluation authority"):
        StreamLoader(
            {
                "output_mode": "raw_text",
                "max_tokens": "max",
                "add_eos": False,
                "require_manifests": True,
                "datasets": [
                    {
                        "name": "reserved-bypass",
                        "type": "manifest",
                        "manifest_path": str(FIXTURES / "benchmark_reserved.manifest.json"),
                        "expected_fingerprint": BENCHMARK_FINGERPRINT,
                        "selection": "all",
                        "access": "evaluation",
                        "allow_reserved_benchmark": True,
                        "ratio": 1.0,
                    }
                ],
            },
            tokenizer=character_tokenizer(),
        )


def test_seed_and_prefetch_do_not_change_manifest_membership():
    base = {
        "output_mode": "raw_text",
        "max_tokens": "max",
        "add_eos": False,
        "preserve_metadata": True,
        "require_manifests": True,
        "datasets": [
            {
                "name": "fixture",
                "type": "manifest",
                "manifest_path": str(BILINGUAL_MANIFEST),
                "expected_fingerprint": BILINGUAL_FINGERPRINT,
                "selection": "validation",
                "ratio": 1.0,
            }
        ],
    }
    memberships = []
    for seed, enabled in ((1, False), (999, True)):
        config = {
            **base,
            "seed": seed,
            "prefetch": {"enabled": enabled, "mode": "thread", "buffer_size": 3},
        }
        memberships.append(
            {
                sample["metadata"]["document_id"]
                for sample in StreamLoader(config, tokenizer=character_tokenizer())
            }
        )
    assert memberships[0] == memberships[1]


def test_real_config_rejects_memory_and_iterable_sources():
    for source_type in ("memory", "iterable"):
        with pytest.raises(ValueError, match="manifest-backed"):
            StreamLoader(
                {
                    "require_manifests": True,
                    "output_mode": "raw_text",
                    "add_eos": False,
                    "datasets": [
                        {
                            "name": "invalid",
                            "type": source_type,
                            "ratio": 1.0,
                            "documents": [{"text": "x"}],
                            "iterable": ["x"],
                        }
                    ],
                },
                tokenizer=character_tokenizer(),
            )


def test_url_cache_rejects_corrupt_hit_and_download(tmp_path):
    expected = b'{"id":"1","text":"ok"}\n'
    digest = hashlib.sha256(expected).hexdigest()
    cache = BoundedShardCache(tmp_path / "cache", max_size_bytes=1024)
    downloads = 0

    def downloader(path: Path) -> None:
        nonlocal downloads
        downloads += 1
        path.write_bytes(expected)

    with cache.acquire(
        "https://example.invalid/data",
        downloader,
        expected_sha256=digest,
        expected_size_bytes=len(expected),
    ) as cached:
        assert cached.read_bytes() == expected
    cached.write_bytes(b"corrupt")
    with cache.acquire(
        "https://example.invalid/data",
        downloader,
        expected_sha256=digest,
        expected_size_bytes=len(expected),
    ) as repaired:
        assert repaired.read_bytes() == expected
    assert downloads == 2

    def corrupt_download(path: Path) -> None:
        path.write_bytes(b"wrong")

    with pytest.raises(ValueError, match="immutable identity"):
        with cache.acquire(
            "https://example.invalid/other",
            corrupt_download,
            expected_sha256=digest,
            expected_size_bytes=len(expected),
        ):
            pass


def test_url_manifest_verifies_fresh_and_corrupt_cached_content(tmp_path, monkeypatch):
    path, manifest = _copy_manifest_bundle(tmp_path)
    source_path = tmp_path / manifest["source"]["path"]
    source_bytes = source_path.read_bytes()
    url = "https://example.invalid/bilingual.jsonl"
    manifest["source"] = {
        "kind": "url_jsonl",
        "url": url,
        "size_bytes": len(source_bytes),
        "sha256": hashlib.sha256(source_bytes).hexdigest(),
        "timeout_seconds": 1.0,
        "text_field": "text",
        "id_field": "id",
    }
    manifest["manifest_fingerprint"] = canonical_fingerprint(
        {key: value for key, value in manifest.items() if key != "manifest_fingerprint"}
    )
    _write_json(path, manifest)
    downloads = 0

    def download(_url: str, destination: Path, _timeout: float) -> None:
        nonlocal downloads
        downloads += 1
        destination.write_bytes(source_bytes)

    monkeypatch.setattr(cache_module, "download_url_to_path", download)
    cache = BoundedShardCache(tmp_path / "url-cache", max_size_bytes=4096)
    resolved = preflight_manifest(
        path,
        expected_fingerprint=manifest["manifest_fingerprint"],
        selection="train",
        cache=cache,
    )
    assert resolved.documents
    cached_path = next((tmp_path / "url-cache").glob("*.shard"))
    cached_path.write_bytes(b"corrupt")
    preflight_manifest(
        path,
        expected_fingerprint=manifest["manifest_fingerprint"],
        selection="train",
        cache=cache,
    )
    assert downloads == 2


def test_cache_sha_key_and_cross_process_lock(tmp_path):
    cache_dir = tmp_path / "concurrent-cache"
    cache_dir.mkdir()
    active_temp = cache_dir / ".other-process.active.tmp"
    active_temp.write_bytes(b"in progress")
    payload = b"immutable payload"
    digest = hashlib.sha256(payload).hexdigest()
    log_path = tmp_path / "downloads.log"
    context = mp.get_context("fork")
    processes = [
        context.Process(
            target=_cache_process_worker,
            args=(str(cache_dir), str(log_path), payload, digest),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0

    assert active_temp.exists()
    assert log_path.read_text(encoding="utf-8").splitlines() == ["download"]
    assert len(list(cache_dir.glob("*.shard"))) == 1

    second_payload = b"different immutable payload"
    second_digest = hashlib.sha256(second_payload).hexdigest()
    cache = BoundedShardCache(cache_dir, max_size_bytes=4096)
    with cache.acquire(
        "https://example.invalid/concurrent",
        lambda path: path.write_bytes(second_payload),
        expected_sha256=second_digest,
        expected_size_bytes=len(second_payload),
    ):
        pass
    assert len(list(cache_dir.glob("*.shard"))) == 2
