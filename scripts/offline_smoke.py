#!/usr/bin/env python3
"""Run the tiny canonical CPU smoke with process-tree network access blocked."""

from __future__ import annotations

import ctypes
import errno
import os
import socket
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
_SCMP_ACT_ALLOW = 0x7FFF0000
_SCMP_ACT_ERRNO = 0x00050000 | errno.EPERM
_SCMP_CMP_NE = 1
_PR_SET_NO_NEW_PRIVS = 38


class _ScmpArgCompare(ctypes.Structure):
    _fields_ = (
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    )


def install_process_tree_network_guard() -> None:
    """Allow only Unix sockets for this process and every descendant."""

    if not sys.platform.startswith("linux"):
        raise RuntimeError("offline smoke process-tree isolation requires Linux seccomp")
    try:
        seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    except OSError as error:
        raise RuntimeError("offline smoke requires libseccomp.so.2") from error
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = (
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    libc.prctl.restype = ctypes.c_int
    seccomp.seccomp_init.argtypes = (ctypes.c_uint32,)
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_rule_add_array.argtypes = (
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(_ScmpArgCompare),
    )
    seccomp.seccomp_rule_add_array.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = (ctypes.c_void_p,)
    seccomp.seccomp_load.restype = ctypes.c_int
    seccomp.seccomp_release.argtypes = (ctypes.c_void_p,)
    seccomp.seccomp_syscall_resolve_name.argtypes = (ctypes.c_char_p,)
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int

    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "unable to enable no-new-privileges for seccomp")
    context = seccomp.seccomp_init(_SCMP_ACT_ALLOW)
    if not context:
        raise RuntimeError("unable to initialize the offline smoke seccomp filter")
    try:
        socket_syscall = seccomp.seccomp_syscall_resolve_name(b"socket")
        if socket_syscall < 0:
            raise RuntimeError("libseccomp could not resolve the socket syscall")
        comparison = _ScmpArgCompare(0, _SCMP_CMP_NE, socket.AF_UNIX, 0)
        result = seccomp.seccomp_rule_add_array(
            context,
            _SCMP_ACT_ERRNO,
            socket_syscall,
            1,
            ctypes.byref(comparison),
        )
        if result != 0:
            raise OSError(-result, "unable to restrict sockets to AF_UNIX")
        result = seccomp.seccomp_load(context)
        if result != 0:
            raise OSError(-result, "unable to load the offline smoke seccomp filter")
    finally:
        seccomp.seccomp_release(context)


def run_process_tree_network_isolated(
    command: list[str],
    *,
    environment: dict[str, str],
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Execute a command through a seccomp launcher inherited by native children."""

    launcher = [sys.executable, str(Path(__file__).resolve()), "--isolated-exec", *command]
    return subprocess.run(
        launcher,
        cwd=ROOT_DIR,
        env=environment,
        check=False,
        capture_output=capture_output,
        close_fds=True,
        text=True,
    )


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
        close_fds=True,
        text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError("offline smoke socket guard was not active in the child interpreter")


def verify_process_tree_network_guard(environment: dict[str, str]) -> None:
    """Prove the OS filter blocks native sockets after exec and in a descendant."""

    descendant_probe = (
        "import ctypes,errno,socket;"
        "c=ctypes.CDLL(None,use_errno=True);"
        "fd=c.socket(socket.AF_INET6,socket.SOCK_STREAM,0);"
        "assert fd == -1 and ctypes.get_errno() == errno.EPERM"
    )
    probe = run_process_tree_network_isolated(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            """
import ctypes
import errno
import socket
import subprocess
import sys

libc = ctypes.CDLL(None, use_errno=True)
for family in (socket.AF_INET, socket.AF_INET6, getattr(socket, "AF_NETLINK", 16), 4242):
    descriptor = libc.socket(family, socket.SOCK_STREAM, 0)
    assert descriptor == -1 and ctypes.get_errno() == errno.EPERM
unix_descriptor = libc.socket(socket.AF_UNIX, socket.SOCK_STREAM, 0)
assert unix_descriptor >= 0
libc.close(unix_descriptor)
subprocess.run([sys.executable, "-I", "-S", "-c", sys.argv[1]], check=True, close_fds=True)
""",
            descendant_probe,
        ],
        environment=environment,
        capture_output=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            "offline smoke process-tree network isolation failed: "
            f"{probe.stderr.strip() or probe.stdout.strip()}"
        )


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
            verify_process_tree_network_guard(environment)
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
            result = run_process_tree_network_isolated(
                command,
                environment=environment,
            )
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, command)

    print(
        "PASS: tiny canonical CPU smoke completed with credentials removed and "
        "Python/native process-tree non-Unix socket creation blocked"
    )


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--isolated-exec":
        if len(sys.argv) < 3:
            raise SystemExit("--isolated-exec requires a command")
        install_process_tree_network_guard()
        os.execvpe(sys.argv[2], sys.argv[2:], os.environ)
    main()
