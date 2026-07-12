from __future__ import annotations

import re
import sys
from pathlib import Path


FORBIDDEN_PROVIDER = re.compile(
    r"(?:^|[-_.])"
    r"(torch|torchaudio|torchdata|torchtext|torchvision|torchao|pytorch|triton|nvidia|"
    r"cuda(?:\d+[a-z]?)?|"
    r"cudnn|cublas|cufft|curand|cusolver|cusparse|nccl)"
    r"(?:$|[-_.])"
)
REQUIREMENT_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==")


def forbidden_projects(requirements: str) -> list[str]:
    projects: list[str] = []
    for line in requirements.splitlines():
        match = REQUIREMENT_NAME.match(line)
        if match and FORBIDDEN_PROVIDER.search(match.group(1).lower()):
            projects.append(match.group(1))
    return projects


def main() -> None:
    path = Path(sys.argv[1])
    forbidden = forbidden_projects(path.read_text(encoding="utf-8"))
    if forbidden:
        raise SystemExit(
            "runtime overlay must not install Torch/CUDA providers: " + ", ".join(forbidden)
        )
    print(f"runtime overlay provider check passed: {path}")


if __name__ == "__main__":
    main()
