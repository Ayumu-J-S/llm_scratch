#!/usr/bin/env python3
"""Run the tiny canonical CPU smoke with Python network access blocked."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CREDENTIAL_ENVIRONMENT = (
    "WANDB_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "GITHUB_TOKEN",
    "GH_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
)
NETWORK_GUARD = '''\
import socket as _socket

_original_connect = _socket.socket.connect


def _deny_network(*_args, **_kwargs):
    raise OSError("CI-001 offline smoke blocks network access")


def _guarded_connect(self, address):
    if self.family == _socket.AF_UNIX:
        return _original_connect(self, address)
    return _deny_network(self, address)


_socket.socket.connect = _guarded_connect
_socket.create_connection = _deny_network
_socket.getaddrinfo = _deny_network
_socket.gethostbyname = _deny_network
_socket.gethostbyname_ex = _deny_network
'''


def offline_environment(guard_dir: Path) -> dict[str, str]:
    """Return child-only settings that make online fallback impossible."""

    environment = os.environ.copy()
    for variable in CREDENTIAL_ENVIRONMENT:
        environment.pop(variable, None)
    inherited_python_path = environment.get("PYTHONPATH")
    environment.update(
        {
            "CI": "true",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "UV_OFFLINE": "1",
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
            "PYTHONPATH": (
                str(guard_dir)
                if not inherited_python_path
                else f"{guard_dir}{os.pathsep}{inherited_python_path}"
            ),
        }
    )
    return environment


def verify_network_guard(environment: dict[str, str]) -> None:
    """Fail if the child interpreter did not load the socket guard."""

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import socket; socket.getaddrinfo('example.invalid', 443)",
        ],
        cwd=ROOT_DIR,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0 or "CI-001 offline smoke blocks network access" not in probe.stderr:
        raise RuntimeError("offline smoke socket guard was not active in the child interpreter")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="llm-scratch-offline-smoke-") as temporary:
        temporary_dir = Path(temporary)
        guard_dir = temporary_dir / "guard"
        guard_dir.mkdir()
        (guard_dir / "sitecustomize.py").write_text(NETWORK_GUARD, encoding="utf-8")
        run_dir = temporary_dir / "run"
        environment = offline_environment(guard_dir)
        verify_network_guard(environment)
        command = [
            sys.executable,
            "src/train.py",
            "profile=smoke_overfit",
            "runtime.device=cpu",
            "training.epochs=1",
            "training.batch_size=2",
            "model.embed_size=16",
            "model.num_heads=4",
            "model.num_layers=1",
            "model.dropout=0.0",
            "wandb.enabled=false",
            f"hydra.run.dir={run_dir}",
            "artifacts.checkpoints_dir=checkpoints",
        ]
        subprocess.run(
            command,
            cwd=ROOT_DIR,
            env=environment,
            check=True,
        )

    print("PASS: tiny canonical CPU smoke completed with credentials removed and sockets blocked")


if __name__ == "__main__":
    main()
