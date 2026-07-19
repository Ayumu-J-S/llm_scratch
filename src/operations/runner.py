"""Direct OPS-001 command dispatch; execution paths are implemented incrementally."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path


def dispatch(args: Namespace, overrides: list[str], *, root_dir: Path) -> int:
    del args, overrides, root_dir
    raise NotImplementedError("OPS-001 command execution is still under implementation")
