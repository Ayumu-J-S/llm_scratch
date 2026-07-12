from __future__ import annotations

import argparse
from pathlib import Path

from data.manifests import build_local_jsonl_manifest, write_manifest_pair
from data.splits import DataPurpose


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an immutable local JSONL data manifest")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--purpose", required=True, choices=[item.value for item in DataPurpose])
    parser.add_argument("--license", required=True, dest="license_name")
    parser.add_argument("--terms-url", required=True)
    parser.add_argument("--salt", required=True)
    parser.add_argument("--validation-fraction", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="id")
    args = parser.parse_args()

    manifest, index = build_local_jsonl_manifest(
        source_path=args.source,
        manifest_path=args.manifest,
        index_path=args.index,
        name=args.name,
        purpose=DataPurpose(args.purpose),
        license_name=args.license_name,
        terms_url=args.terms_url,
        salt=args.salt,
        validation_fraction=args.validation_fraction,
        text_field=args.text_field,
        id_field=args.id_field or None,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    write_manifest_pair(
        manifest,
        index,
        manifest_path=args.manifest,
        index_path=args.index,
    )
    print(manifest["manifest_fingerprint"])


if __name__ == "__main__":
    main()
