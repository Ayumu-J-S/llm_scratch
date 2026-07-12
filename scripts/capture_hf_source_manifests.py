from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


HF_API = "https://huggingface.co/api/datasets"
LICENSE = "ODC-By-1.0"
TERMS_URL = "https://commoncrawl.org/terms-of-use"
SPLIT_POLICY = {
    "method": "normalized_content_sha256_v1",
    "salt": "llm-scratch-data-004-split-v1",
    "validation_fraction": "0.01",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    filename: str
    repo_id: str
    revision: str
    config_name: str
    split: str
    prefix: str
    config_glob: str
    expected_files: int
    expected_bytes: int
    language: str
    metadata_fields: tuple[str, ...]


SOURCES = (
    SourceSpec(
        name="fineweb-en-sample-10bt-v1",
        filename="fineweb-en-sample-10bt.manifest.json",
        repo_id="HuggingFaceFW/fineweb",
        revision="9bb295ddab0e05d785b879661af7260fed5140fc",
        config_name="sample-10BT",
        split="train",
        prefix="sample/10BT/",
        config_glob="sample/10BT/*",
        expected_files=15,
        expected_bytes=30_639_384_917,
        language="en",
        metadata_fields=(
            "dump",
            "url",
            "date",
            "file_path",
            "language",
            "language_score",
            "token_count",
        ),
    ),
    SourceSpec(
        name="fineweb2-ja-jpn-jpan-v1",
        filename="fineweb2-ja-jpn-jpan.manifest.json",
        repo_id="HuggingFaceFW/fineweb-2",
        revision="af9c13333eb981300149d5ca60a8e9d659b276b9",
        config_name="jpn_Jpan",
        split="train",
        prefix="data/jpn_Jpan/train/",
        config_glob="data/jpn_Jpan/train/*",
        expected_files=175,
        expected_bytes=716_653_211_753,
        language="ja",
        metadata_fields=(
            "dump",
            "url",
            "date",
            "file_path",
            "language",
            "language_score",
            "language_script",
            "minhash_cluster_size",
            "top_langs",
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture DATA-004 Hugging Face Parquet inventories at exact revisions."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "manifests",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if committed manifests differ; do not write files.",
    )
    args = parser.parse_args()

    for spec in SOURCES:
        metadata = _fetch_revision(spec)
        artifacts = _validated_inventory(metadata, spec)
        manifest = _build_manifest(spec, artifacts)
        output = args.output_dir / spec.filename
        rendered = _pretty_json(manifest)
        if args.check:
            try:
                committed = output.read_bytes()
            except OSError as error:
                raise SystemExit(f"missing committed manifest: {output}") from error
            if committed != rendered:
                raise SystemExit(f"manifest differs from official revision API: {output}")
        else:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            output.write_bytes(rendered)
        print(f"{spec.name} {manifest['manifest_fingerprint']} {len(artifacts)}")


def _fetch_revision(spec: SourceSpec) -> dict[str, Any]:
    url = f"{HF_API}/{spec.repo_id}/revision/{spec.revision}?blobs=true"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "llm-scratch-DATA-004-manifest-capture/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            value = json.load(response)
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot fetch official revision metadata: {url}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"official revision metadata is not an object: {url}")
    if value.get("id") != spec.repo_id or value.get("sha") != spec.revision:
        raise SystemExit(f"repository or exact revision mismatch: {spec.repo_id}")
    if value.get("private") is not False or value.get("gated") is not False:
        raise SystemExit(f"source is no longer public and ungated: {spec.repo_id}")
    card_data = value.get("cardData")
    if not isinstance(card_data, dict) or card_data.get("license") != "odc-by":
        raise SystemExit(f"dataset-card license mismatch: {spec.repo_id}")
    _validate_config(card_data, spec)
    return value


def _validate_config(card_data: dict[str, Any], spec: SourceSpec) -> None:
    configs = card_data.get("configs")
    if not isinstance(configs, list):
        raise SystemExit(f"dataset card has no config inventory: {spec.repo_id}")
    matches = [item for item in configs if item.get("config_name") == spec.config_name]
    if len(matches) != 1:
        raise SystemExit(f"config mismatch: {spec.repo_id}/{spec.config_name}")
    data_files = matches[0].get("data_files")
    expected = {"split": spec.split, "path": spec.config_glob}
    if not isinstance(data_files, list) or expected not in data_files:
        raise SystemExit(
            f"config no longer maps {spec.split} to {spec.config_glob}: {spec.repo_id}"
        )


def _validated_inventory(metadata: dict[str, Any], spec: SourceSpec) -> list[dict[str, Any]]:
    siblings = metadata.get("siblings")
    if not isinstance(siblings, list):
        raise SystemExit(f"revision has no sibling inventory: {spec.repo_id}")
    candidates = [
        item
        for item in siblings
        if isinstance(item, dict) and str(item.get("rfilename", "")).startswith(spec.prefix)
    ]
    if any(not str(item.get("rfilename", "")).endswith(".parquet") for item in candidates):
        raise SystemExit(f"unexpected non-Parquet artifact under selected prefix: {spec.repo_id}")

    artifacts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates:
        path = item.get("rfilename")
        size = item.get("size")
        lfs = item.get("lfs")
        if not isinstance(path, str) or not path.startswith(spec.prefix):
            raise SystemExit(f"invalid artifact path: {spec.repo_id}")
        if path in seen:
            raise SystemExit(f"duplicate artifact path: {spec.repo_id}/{path}")
        seen.add(path)
        if not isinstance(size, int) or size <= 0:
            raise SystemExit(f"invalid artifact size: {spec.repo_id}/{path}")
        if not isinstance(lfs, dict) or lfs.get("size") != size:
            raise SystemExit(f"missing or inconsistent LFS identity: {spec.repo_id}/{path}")
        sha256 = lfs.get("sha256")
        if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
            raise SystemExit(f"invalid LFS SHA-256: {spec.repo_id}/{path}")
        artifacts.append({"path": path, "size_bytes": size, "sha256": sha256})

    artifacts.sort(key=lambda item: item["path"])
    actual_bytes = sum(item["size_bytes"] for item in artifacts)
    if len(artifacts) != spec.expected_files or actual_bytes != spec.expected_bytes:
        raise SystemExit(
            f"inventory mismatch for {spec.repo_id}/{spec.config_name}/{spec.split}: "
            f"expected {spec.expected_files} files/{spec.expected_bytes} bytes, "
            f"got {len(artifacts)} files/{actual_bytes} bytes"
        )
    return artifacts


def _build_manifest(spec: SourceSpec, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    source = {
        "kind": "hf_parquet",
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "config_name": spec.config_name,
        "split": spec.split,
        "data_files": artifacts,
        "text_field": "text",
        "id_field": "id",
        "metadata_fields": list(spec.metadata_fields),
    }
    document_policy = {
        "version": 1,
        "language": spec.language,
        "max_utf8_bytes": 1_048_576,
        "reject_controls": True,
        "reject_wrong_script": True,
    }
    manifest = {
        "schema_version": 2,
        "name": spec.name,
        "purpose": "pretraining",
        "source": source,
        "usage": {"license": LICENSE, "terms_url": TERMS_URL},
        "split": dict(SPLIT_POLICY),
        "document_policy": document_policy,
        "dataset_fingerprint": _fingerprint(
            {"name": spec.name, "source": source, "document_policy": document_policy}
        ),
    }
    manifest["manifest_fingerprint"] = _fingerprint(manifest)
    return manifest


def _fingerprint(value: Any) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _pretty_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


if __name__ == "__main__":
    main()
