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
NETWORK_GUARD = """\
import socket as _socket

_original_connect = _socket.socket.connect
_original_connect_ex = _socket.socket.connect_ex
_original_sendto = _socket.socket.sendto


def _deny_network(*_args, **_kwargs):
    raise OSError("CI-001 offline smoke blocks network access")


def _guarded_connect(self, address):
    if self.family == _socket.AF_UNIX:
        return _original_connect(self, address)
    return _deny_network(self, address)


def _guarded_connect_ex(self, address):
    if self.family == _socket.AF_UNIX:
        return _original_connect_ex(self, address)
    return _deny_network(self, address)


def _guarded_sendto(self, *args, **kwargs):
    if self.family == _socket.AF_UNIX:
        return _original_sendto(self, *args, **kwargs)
    return _deny_network(self, *args, **kwargs)


_socket.socket.connect = _guarded_connect
_socket.socket.connect_ex = _guarded_connect_ex
_socket.socket.sendto = _guarded_sendto
_socket.create_connection = _deny_network
_socket.getaddrinfo = _deny_network
_socket.gethostbyname = _deny_network
_socket.gethostbyname_ex = _deny_network
"""


def offline_environment(guard_dir: Path, *, wandb_mode: str = "disabled") -> dict[str, str]:
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
            "WANDB_MODE": wandb_mode,
            "WANDB_DIR": str(guard_dir.parent / "wandb"),
            "PYTHONPATH": (
                str(guard_dir)
                if not inherited_python_path
                else f"{guard_dir}{os.pathsep}{inherited_python_path}"
            ),
        }
    )
    if wandb_mode == "disabled":
        environment["WANDB_DISABLED"] = "true"
    else:
        environment.pop("WANDB_DISABLED", None)
    return environment


def verify_network_guard(environment: dict[str, str]) -> None:
    """Fail if the child interpreter did not load the socket guard."""

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import socket

probes = (
    lambda: socket.getaddrinfo("example.invalid", 443),
    lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect_ex(("127.0.0.1", 9)),
    lambda: socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(b"probe", ("127.0.0.1", 9)),
)
for probe in probes:
    try:
        probe()
    except OSError as error:
        if "CI-001 offline smoke blocks network access" not in str(error):
            raise
    else:
        raise RuntimeError("offline smoke network operation escaped the socket guard")
""",
        ],
        cwd=ROOT_DIR,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError("offline smoke socket guard was not active in the child interpreter")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="llm-scratch-offline-smoke-") as temporary:
        temporary_dir = Path(temporary)
        guard_dir = temporary_dir / "guard"
        guard_dir.mkdir()
        (guard_dir / "sitecustomize.py").write_text(NETWORK_GUARD, encoding="utf-8")
        for wandb_mode in ("disabled", "offline"):
            run_dir = temporary_dir / f"run-{wandb_mode}"
            environment = offline_environment(guard_dir, wandb_mode=wandb_mode)
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
                f"wandb.mode={wandb_mode}",
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
