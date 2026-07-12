#!/usr/bin/env python3
"""Verify the committed uv lock and prove that a changed project is rejected."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def run_lock_check(project_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "lock", "--check"],
        cwd=project_dir,
        check=False,
        capture_output=True,
        text=True,
    )


def main() -> None:
    current = run_lock_check(ROOT_DIR)
    if current.returncode:
        raise SystemExit(current.stderr or current.stdout)

    with tempfile.TemporaryDirectory(prefix="llm-scratch-lock-drift-") as temporary:
        project_dir = Path(temporary)
        for name in ("pyproject.toml", "uv.lock"):
            shutil.copy2(ROOT_DIR / name, project_dir / name)
        pyproject = project_dir / "pyproject.toml"
        original = pyproject.read_text(encoding="utf-8")
        if 'version = "0.1.0"' not in original:
            raise SystemExit("lock-drift probe could not find the expected project version")
        pyproject.write_text(
            original.replace('version = "0.1.0"', 'version = "0.1.1"', 1),
            encoding="utf-8",
        )
        drifted = run_lock_check(project_dir)
        if drifted.returncode == 0:
            raise SystemExit("uv lock --check accepted an intentionally drifted pyproject.toml")

    print("PASS: committed uv.lock is current and intentional project drift is rejected")


if __name__ == "__main__":
    main()
