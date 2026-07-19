"""Owned process/container lifecycle and storage watchdog operations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from operations.artifacts import (
    Attempt,
    AttemptError,
    atomic_write_json,
    load_json,
    process_identity,
    utc_now,
)
from operations.preflight import PreflightError


_PROCESS_GROUP_TERM_GRACE_SECONDS = 2.0
_PROCESS_GROUP_KILL_GRACE_SECONDS = 2.0


def wait_with_disk_watchdog(
    process: subprocess.Popen[bytes],
    *,
    attempt: Attempt,
    watchdog: dict[str, Any],
    container_record: Mapping[str, Any] | None,
    trusted_ownership: Mapping[str, Any] | None,
    stop_request: Mapping[str, Any] | None = None,
) -> int:
    while process.poll() is None:
        signal_number = stop_request.get("signal") if stop_request is not None else None
        if signal_number is not None:
            watchdog.update(
                {
                    "triggered": True,
                    "reason": "external_signal",
                    "signal": int(signal_number),
                    "signal_name": str(stop_request.get("signal_name", "unknown")),
                }
            )
            attempt.event(
                "external_signal_stop",
                signal=int(signal_number),
                signal_name=str(stop_request.get("signal_name", "unknown")),
            )
            stop_errors = stop_owned_process(
                process,
                attempt=attempt,
                container_record=container_record,
                trusted_ownership=trusted_ownership,
            )
            if stop_errors:
                raise AttemptError("; ".join(stop_errors))
            return process.wait()
        for target in watchdog["filesystems"]:
            try:
                free = shutil.disk_usage(target["path"]).free
            except OSError as error:
                watchdog.update(
                    {
                        "triggered": True,
                        "reason": "disk_sampling_failure",
                        "sampling_error": str(error),
                    }
                )
                attempt.event("disk_watchdog_sampling_failure", error=str(error))
                stop_errors = stop_owned_process(
                    process,
                    attempt=attempt,
                    container_record=container_record,
                    trusted_ownership=trusted_ownership,
                )
                if stop_errors:
                    raise AttemptError("; ".join(stop_errors))
                return process.wait()
            previous = target.get("minimum_observed_free_bytes")
            target["minimum_observed_free_bytes"] = (
                free if previous is None else min(int(previous), free)
            )
            floor = int(target["effective_live_floor_bytes"])
            if free < floor:
                watchdog.update(
                    {
                        "triggered": True,
                        "reason": "disk_reserve",
                        "device": target["device"],
                        "observed_free_bytes": free,
                    }
                )
                attempt.event(
                    "disk_watchdog_stop",
                    device=target["device"],
                    path=target["path"],
                    free_bytes=free,
                    floor_bytes=floor,
                )
                stop_errors = stop_owned_process(
                    process,
                    attempt=attempt,
                    container_record=container_record,
                    trusted_ownership=trusted_ownership,
                )
                if stop_errors:
                    raise AttemptError("; ".join(stop_errors))
                return process.wait()
        time.sleep(0.25)
    return process.wait()


def stop_owned_process(
    process: subprocess.Popen[bytes],
    *,
    attempt: Attempt,
    container_record: Mapping[str, Any] | None,
    trusted_ownership: Mapping[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    trusted_process_group = False
    if container_record is not None:
        observed = container_identity(container_record)
        if not observed.get("labels_match", False):
            attempt.event("container_ownership_anomaly", observed=observed)
            errors.append("container ownership labels changed after launch")
        stopped = subprocess.run(
            ["docker", "stop", "--time", "10", str(container_record["id"])],
            check=False,
            capture_output=True,
        )
        if stopped.returncode != 0:
            killed = subprocess.run(
                ["docker", "kill", str(container_record["id"])],
                check=False,
                capture_output=True,
            )
            if killed.returncode != 0:
                errors.append("immutable owned container ID could not be stopped or killed")
    else:
        if trusted_ownership is None or process.pid != trusted_ownership.get("pid"):
            attempt.event(
                "process_group_ownership_unavailable",
                child_pid=process.pid,
                trusted_pid=(
                    trusted_ownership.get("pid") if trusted_ownership is not None else None
                ),
            )
            errors.append(
                "trusted process-group ownership is unavailable; "
                "descendant process ownership is uncertain"
            )
            process.terminate()
        else:
            try:
                disk_ownership = load_json(attempt.path / "pid.json")
            except AttemptError as error:
                disk_ownership = None
                errors.append(f"PID evidence is unreadable: {error}")
            if disk_ownership != dict(trusted_ownership):
                attempt.event(
                    "ownership_evidence_mutated",
                    trusted_pid=trusted_ownership.get("pid"),
                )
                errors.append("on-disk PID ownership evidence changed after launch")
            trusted_process_group = _trusted_process_is_live(process, trusted_ownership)
            if trusted_process_group:
                try:
                    os.killpg(int(trusted_ownership["process_group_id"]), signal.SIGTERM)
                except ProcessLookupError:
                    return errors
            else:
                errors.append(
                    "live /proc identity differs from trusted child lifecycle; "
                    "descendant process ownership is uncertain"
                )
                attempt.event(
                    "process_group_ownership_uncertain",
                    trusted_pid=trusted_ownership.get("pid"),
                    trusted_process_group_id=trusted_ownership.get("process_group_id"),
                )
                process.terminate()
    if trusted_process_group:
        assert trusted_ownership is not None
        process_group_id = int(trusted_ownership["process_group_id"])
        if not _wait_for_process_group_exit(
            process,
            process_group_id,
            timeout_seconds=_PROCESS_GROUP_TERM_GRACE_SECONDS,
        ):
            try:
                os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as error:
                errors.append(f"owned process group could not be killed: {error}")
            if not _wait_for_process_group_exit(
                process,
                process_group_id,
                timeout_seconds=_PROCESS_GROUP_KILL_GRACE_SECONDS,
            ):
                members = _live_process_group_members(process_group_id)
                attempt.event(
                    "process_group_cleanup_incomplete",
                    process_group_id=process_group_id,
                    live_member_pids=members,
                )
                errors.append(
                    "owned process group still has live members after SIGKILL: "
                    + ", ".join(str(pid) for pid in members)
                )
        return errors

    try:
        process.wait(timeout=12)
    except subprocess.TimeoutExpired:
        if container_record is not None:
            subprocess.run(
                ["docker", "kill", str(container_record["id"])],
                check=False,
                capture_output=True,
            )
        else:
            process.kill()
    return errors


def _wait_for_process_group_exit(
    process: subprocess.Popen[bytes],
    process_group_id: int,
    *,
    timeout_seconds: float,
) -> bool:
    """Reap the leader and wait until no non-zombie member remains in its group."""

    deadline = time.monotonic() + timeout_seconds
    while True:
        process.poll()
        if not _live_process_group_members(process_group_id):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def _live_process_group_members(process_group_id: int) -> list[int]:
    """Return live Linux process IDs still owned by one process group."""

    members: list[int] = []
    try:
        entries = Path("/proc").iterdir()
    except OSError:
        return [process_group_id]
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            suffix = stat[stat.rfind(")") + 2 :].split()
            state = suffix[0]
            observed_group = int(suffix[2])
        except (OSError, ValueError, IndexError):
            continue
        if observed_group == process_group_id and state != "Z":
            members.append(int(entry.name))
    return sorted(members)


def _container_name(attempt: Attempt) -> str:
    """Map the exact case-sensitive attempt identity to one bounded Docker name."""

    identity = f"{attempt.run_id}\0{attempt.attempt_id}".encode("utf-8", errors="strict")
    digest = hashlib.sha256(identity).hexdigest()
    return f"llm-scratch-{attempt.run_id[:20]}-{attempt.attempt_id[:20]}-{digest}"


def create_container(
    attempt: Attempt,
    *,
    args: Namespace,
    root_dir: Path,
    inner_command: list[str],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    device_record = _mapping(preflight["checks"]["device"], "device preflight")
    image_id = str(device_record.get("image_id", ""))
    if not image_id.startswith("sha256:"):
        raise PreflightError("container launch requires the immutable preflight image ID")
    name = _container_name(attempt)
    labels = {
        "llm-scratch.ops.run_id": attempt.run_id,
        "llm-scratch.ops.attempt_id": attempt.attempt_id,
    }
    command = ["docker", "create", "--name", name]
    for key, value in labels.items():
        command.extend(["--label", f"{key}={value}"])
    command.extend(
        [
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--ipc=host",
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
            "--workdir",
            str(root_dir),
        ]
    )
    mount_record = _mapping(
        preflight["checks"].get("container_mounts"), "container mount preflight"
    )
    if mount_record.get("status") != "passed" or mount_record.get("required") is not True:
        raise PreflightError("container launch requires a passed validated mount plan")
    mounts = _validated_mount_args(mount_record.get("mounts"))
    command.extend(mounts["args"])
    if args.device == "cuda":
        command.extend(["--gpus", "all"])
    wandb_record = preflight.get("checks", {}).get("wandb", {})
    if isinstance(wandb_record, Mapping) and wandb_record.get("mode") == "online":
        for key in ("WANDB_API_KEY", "WANDB_BASE_URL"):
            if os.environ.get(key):
                command.extend(["--env", key])
    for key, value in child_environment(
        args.action,
        wandb_evidence_path=attempt.path / "work" / "wandb_events.jsonl",
    ).items():
        if key in {
            "UV_OFFLINE",
            "HF_HUB_OFFLINE",
            "HF_DATASETS_OFFLINE",
            "LLM_SCRATCH_WANDB_EVIDENCE_PATH",
        }:
            command.extend(["--env", f"{key}={value}"])
    command.extend(["--env", "GIT_OPTIONAL_LOCKS=0"])
    inner = ["python", inner_command[1], *inner_command[2:]]
    command.extend(["--entrypoint", inner[0], image_id, *inner[1:]])
    result = subprocess.run(command, cwd=root_dir, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("docker create failed: " + result.stderr.strip())
    container_id = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
        raise RuntimeError("docker create did not return an immutable full container ID")
    record = {
        "schema_version": 1,
        "id": container_id,
        "name": name,
        "image_id": image_id,
        "labels": labels,
        "mounts": mounts["records"],
        "created_at": utc_now(),
        "removed": False,
    }
    try:
        post_create_identity = container_identity(record)
    except BaseException as error:
        post_create_identity = {
            "available": False,
            "error": f"{type(error).__name__}: {error}",
        }
    record["post_create_identity"] = post_create_identity
    if not _identity_matches(post_create_identity):
        removal_error = _force_remove_container(record)
        atomic_write_json(attempt.path / "container.json", record)
        message = "created container does not retain the exact attempt ownership identity"
        if removal_error is not None:
            message += "; " + removal_error
        raise AttemptError(message)
    return record


def _validated_mount_args(value: Any) -> dict[str, Any]:
    if not isinstance(value, list) or not value:
        raise PreflightError("container launch requires a non-empty mount list")
    arguments: list[str] = []
    records: list[dict[str, Any]] = []
    for item in value:
        record = _mapping(item, "container mount")
        if set(record) != {"source", "destination", "read_only", "kind", "purposes"}:
            raise PreflightError("container mount record has unexpected fields")
        source = Path(str(record["source"]))
        destination = Path(str(record["destination"]))
        kind = str(record["kind"])
        if not source.is_absolute() or not destination.is_absolute():
            raise PreflightError("container mount paths must remain absolute")
        if kind == "directory" and not source.is_dir():
            raise PreflightError(f"validated container directory disappeared: {source}")
        if kind == "file" and not source.is_file():
            raise PreflightError(f"validated container file disappeared: {source}")
        if kind not in {"directory", "file"}:
            raise PreflightError(f"invalid container mount kind: {kind}")
        specification = f"type=bind,src={source},dst={destination}"
        if record["read_only"] is True:
            specification += ",readonly"
        elif record["read_only"] is not False:
            raise PreflightError("container mount read_only must be boolean")
        purposes = record["purposes"]
        if not isinstance(purposes, list) or not purposes:
            raise PreflightError("container mount purposes must be a non-empty list")
        arguments.extend(["--mount", specification])
        records.append(dict(record))
    return {"args": arguments, "records": records}


def container_owned(record: Mapping[str, Any]) -> bool:
    observed = container_identity(record)
    return _identity_matches(observed)


def _identity_matches(observed: Mapping[str, Any]) -> bool:
    return bool(
        observed.get("id_matches")
        and observed.get("image_matches")
        and observed.get("labels_match")
    )


def container_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "inspect", str(record.get("id", ""))],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {"available": False, "error": result.stderr.strip()}
    try:
        payload = json.loads(result.stdout)
        return _identity_from_inspect(record, payload)
    except (IndexError, KeyError, TypeError, json.JSONDecodeError):
        return {"available": False, "error": "invalid docker inspect payload"}


def capture_and_remove_container(attempt: Attempt, record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        inspect = subprocess.run(
            ["docker", "inspect", record["id"]], capture_output=True, text=True
        )
        if inspect.returncode != 0:
            errors.append("docker inspect failed during finalization: " + inspect.stderr.strip())
            record["inspect_failed"] = True
        else:
            payload = json.loads(inspect.stdout)
            identity = _identity_from_inspect(record, payload)
            if not _identity_matches(identity):
                record["ownership_anomaly"] = identity
                errors.append("container identity changed before final evidence capture")
            state = payload[0].get("State", {})
            record["terminal_state"] = {
                key: state.get(key)
                for key in (
                    "Status",
                    "ExitCode",
                    "OOMKilled",
                    "Error",
                    "StartedAt",
                    "FinishedAt",
                )
            }
    except BaseException as error:
        errors.append(f"container finalization: {type(error).__name__}: {error}")
    removal_error = _force_remove_container(record)
    if removal_error is not None:
        errors.append(removal_error)
    try:
        atomic_write_json(attempt.path / "container.json", record)
    except OSError as error:
        errors.append(f"container evidence write: {error}")
    return errors


def _identity_from_inspect(record: Mapping[str, Any], payload: Any) -> dict[str, Any]:
    observed = payload[0]
    labels = observed["Config"]["Labels"] or {}
    return {
        "available": True,
        "id_matches": observed["Id"] == record["id"],
        "image_matches": observed["Image"] == record["image_id"],
        "labels_match": all(labels.get(key) == value for key, value in record["labels"].items()),
    }


def _force_remove_container(record: dict[str, Any]) -> str | None:
    try:
        removed = subprocess.run(
            ["docker", "rm", "--force", str(record["id"])],
            capture_output=True,
            text=True,
        )
        record["removed"] = removed.returncode == 0
        record["remove_error"] = removed.stderr.strip() or None
        if removed.returncode != 0:
            return "owned container removal failed: " + removed.stderr.strip()
    except BaseException as error:
        message = f"owned container removal failed: {type(error).__name__}: {error}"
        record["removed"] = False
        record["remove_error"] = message
        return message
    return None


def watchdog_targets(preflight: Mapping[str, Any]) -> list[dict[str, Any]]:
    storage = _mapping(preflight["checks"]["storage"], "storage preflight")
    targets: dict[int, dict[str, Any]] = {}
    for record in storage.get("filesystems", []):
        if not isinstance(record, Mapping):
            continue
        device = int(record.get("device", -1))
        if device < 0:
            continue
        floor = int(record["effective_live_floor_bytes"])
        existing = targets.get(device)
        if existing is None or floor > int(existing["effective_live_floor_bytes"]):
            targets[device] = {
                "device": device,
                "path": str(record["path"]),
                "effective_live_floor_bytes": floor,
                "minimum_observed_free_bytes": None,
            }
    if not targets:
        raise PreflightError("storage preflight did not provide watchdog filesystems")
    return [targets[device] for device in sorted(targets)]


def child_environment(
    action: str,
    *,
    wandb_evidence_path: Path | None = None,
) -> dict[str, str]:
    environment = dict(os.environ)
    if wandb_evidence_path is not None:
        environment["LLM_SCRATCH_WANDB_EVIDENCE_PATH"] = str(
            wandb_evidence_path.resolve()
        )
    if action == "smoke":
        environment.pop("WANDB_API_KEY", None)
        environment.pop("WANDB_BASE_URL", None)
        environment.update(
            {
                "UV_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "WANDB_MODE": "disabled",
                "WANDB_DISABLED": "true",
            }
        )
    return environment


def _trusted_process_is_live(process: subprocess.Popen[bytes], identity: Mapping[str, Any]) -> bool:
    if process.poll() is not None or process.pid != identity.get("pid"):
        return False
    try:
        current = process_identity(process.pid)
    except AttemptError:
        return False
    fields = ("pid", "process_group_id", "proc_start_ticks", "boot_id")
    return all(current.get(field) == identity.get(field) for field in fields)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AttemptError(f"{label} must be a mapping")
    return value
