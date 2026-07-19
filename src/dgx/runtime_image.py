"""Content identity for the DGX runtime image build inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_BUILD_INPUTS = (
    Path("Dockerfile"),
    Path("requirements/runtime.txt"),
    Path("scripts/check_runtime_requirements.py"),
    Path("src/dgx/runtime_image.py"),
)
RUNTIME_SPEC_LABEL = "io.llm-scratch.runtime-spec-sha256"


def runtime_image_spec(root: Path = ROOT) -> dict[str, object]:
    files = []
    for relative_path in RUNTIME_BUILD_INPUTS:
        path = root / relative_path
        files.append(
            {
                "path": relative_path.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {
        "schema_version": 1,
        "platform": "linux/arm64",
        "files": files,
    }


def runtime_image_spec_sha256(root: Path = ROOT) -> str:
    encoded = json.dumps(
        runtime_image_spec(root),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha256", action="store_true")
    args = parser.parse_args(argv)
    payload: object = runtime_image_spec_sha256() if args.sha256 else runtime_image_spec()
    print(payload if isinstance(payload, str) else json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
