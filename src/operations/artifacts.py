"""Immutable attempt directories, atomic records, and child ownership identity."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_TERMINAL = {"succeeded", "failed", "stopped"}
_TRANSITIONS = {
    "created": {"preflighting"},
    "preflighting": {"ready", "failed"},
    "ready": {"running", "succeeded", "failed"},
    "running": _TERMINAL,
}


class AttemptError(RuntimeError):
    """An attempt record is invalid, ambiguous, or would be overwritten."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def atomic_write_bytes(path: str | Path, payload: bytes, *, mode: int = 0o600) -> None:
    """Commit one file by fsync and same-directory replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value))


def load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AttemptError(f"invalid JSON record: {path}") from error
    if not isinstance(value, dict):
        raise AttemptError(f"JSON record must be an object: {path}")
    return value


@dataclass(frozen=True)
class Attempt:
    run_root: Path
    run_id: str
    attempt_id: str

    @property
    def run_dir(self) -> Path:
        return self.run_root / self.run_id

    @property
    def path(self) -> Path:
        return self.run_dir / "attempts" / self.attempt_id

    @property
    def state_path(self) -> Path:
        return self.path / "state.json"

    @property
    def events_dir(self) -> Path:
        return self.path / "events"

    def state(self) -> dict[str, Any]:
        return load_json(self.state_path)

    def transition(self, status: str, **fields: Any) -> dict[str, Any]:
        state = self.state()
        previous = state.get("status")
        allowed = _TRANSITIONS.get(str(previous), set())
        if status not in allowed:
            raise AttemptError(f"invalid attempt transition: {previous!r} -> {status!r}")
        updated = {**state, **fields, "status": status, "updated_at": utc_now()}
        atomic_write_json(self.state_path, updated)
        self.event("state_transition", previous=previous, status=status, fields=fields)
        return updated

    def event(self, kind: str, **fields: Any) -> Path:
        if not isinstance(kind, str) or not kind:
            raise AttemptError("event kind must be a non-empty string")
        self.events_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.events_dir.glob("*.json"))
        sequence = len(existing) + 1
        destination = self.events_dir / f"{sequence:06d}.json"
        if destination.exists():
            raise AttemptError(f"event path already exists: {destination}")
        atomic_write_json(
            destination,
            {
                "schema_version": 1,
                "sequence": sequence,
                "timestamp": utc_now(),
                "kind": kind,
                **fields,
            },
        )
        return destination


def create_attempt(
    *,
    run_root: str | Path,
    run_id: str,
    attempt_id: str,
    action: str,
    executor: str,
    device: str | None,
    retry_from: str | None,
) -> Attempt:
    _validate_identifier(run_id, "run ID")
    _validate_identifier(attempt_id, "attempt ID")
    root = Path(run_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    attempt = Attempt(root, run_id, attempt_id)
    if attempt.path.exists():
        raise AttemptError(f"attempt already exists and will not be replaced: {attempt.path}")
    attempt.path.mkdir(parents=True, mode=0o700)
    attempt.path.chmod(0o700)

    retry_binding: dict[str, Any] | None = None
    if retry_from is not None:
        _validate_identifier(retry_from, "retry attempt ID")
        if retry_from == attempt_id:
            raise AttemptError("an attempt cannot retry itself")
        parent = Attempt(root, run_id, retry_from)
        parent_state = parent.state()
        if parent_state.get("status") not in _TERMINAL:
            raise AttemptError("a retry requires a terminal sibling attempt")
        parent_result = parent.path / "result.json"
        if not parent_result.is_file():
            raise AttemptError("a retry requires the sibling result.json")
        retry_binding = {
            "attempt_id": retry_from,
            "state_sha256": sha256_file(parent.state_path),
            "result_sha256": sha256_file(parent_result),
        }

    run_record = attempt.run_dir / "run.json"
    if run_record.exists():
        existing = load_json(run_record)
        if existing.get("run_id") != run_id:
            raise AttemptError("run directory identity does not match the requested run ID")
    else:
        attempt.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            run_record,
            {
                "schema_version": 1,
                "run_id": run_id,
                "created_at": utc_now(),
                "hostname": socket.gethostname(),
            },
        )

    state = {
        "schema_version": 1,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "action": action,
        "executor": executor,
        "device": device,
        "status": "created",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "retry_from": retry_binding,
    }
    atomic_write_json(attempt.state_path, state)
    attempt.event("attempt_created", action=action, retry_from=retry_binding)
    return attempt


def process_identity(pid: int) -> dict[str, Any]:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        raise AttemptError("PID must be a positive integer")
    stat_path = Path(f"/proc/{pid}/stat")
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    try:
        stat_fields = stat_path.read_text(encoding="utf-8").split()
        cmdline = cmdline_path.read_bytes()
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError as error:
        raise AttemptError(f"cannot capture ownership for PID {pid}") from error
    if len(stat_fields) < 22:
        raise AttemptError(f"invalid /proc stat for PID {pid}")
    return {
        "schema_version": 1,
        "pid": pid,
        "process_group_id": os.getpgid(pid),
        "proc_start_ticks": int(stat_fields[21]),
        "boot_id": boot_id,
        "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
    }


def process_identity_matches(identity: Mapping[str, Any]) -> bool:
    try:
        current = process_identity(int(identity["pid"]))
    except (AttemptError, KeyError, TypeError, ValueError, ProcessLookupError):
        return False
    return all(current.get(field) == identity.get(field) for field in current)


def _validate_identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise AttemptError(
            f"{label} must match {_IDENTIFIER.pattern} and cannot contain path separators"
        )


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
