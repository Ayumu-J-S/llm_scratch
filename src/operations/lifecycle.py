"""Owned process/container lifecycle and storage watchdog operations."""

from __future__ import annotations

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


def wait_with_disk_watchdog(
    process: subprocess.Popen[bytes],
    *,
    attempt: Attempt,
    watchdog: dict[str, Any],
    container_record: Mapping[str, Any] | None,
    trusted_ownership: Mapping[str, Any] | None,
) -> int:
    while process.poll() is None:
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
            return ["trusted in-memory child ownership is unavailable"]
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
    try:
        process.wait(timeout=12)
    except subprocess.TimeoutExpired:
        if container_record is not None:
            subprocess.run(
                ["docker", "kill", str(container_record["id"])],
                check=False,
                capture_output=True,
            )
        elif trusted_process_group:
            assert trusted_ownership is not None
            os.killpg(int(trusted_ownership["process_group_id"]), signal.SIGKILL)
        else:
            process.kill()
    return errors


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
    name = f"llm-scratch-{attempt.run_id}-{attempt.attempt_id}".lower()[:120]
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
            "--mount",
            f"type=bind,src={root_dir},dst={root_dir}",
            "--workdir",
            str(root_dir),
        ]
    )
    if not attempt.run_root.is_relative_to(root_dir):
        command.extend(
            [
                "--mount",
                f"type=bind,src={attempt.run_root},dst={attempt.run_root}",
            ]
        )
    if args.device == "cuda":
        command.extend(["--gpus", "all"])
    wandb_record = preflight.get("checks", {}).get("wandb", {})
    if isinstance(wandb_record, Mapping) and wandb_record.get("mode") == "online":
        for key in ("WANDB_API_KEY", "WANDB_BASE_URL"):
            if os.environ.get(key):
                command.extend(["--env", key])
    for key, value in child_environment(args.action).items():
        if key in {"UV_OFFLINE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"}:
            command.extend(["--env", f"{key}={value}"])
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
        "created_at": utc_now(),
        "removed": False,
    }
    post_create_identity = container_identity(record)
    record["post_create_identity"] = post_create_identity
    if not _identity_matches(post_create_identity):
        removed = subprocess.run(
            ["docker", "rm", "--force", container_id],
            capture_output=True,
            text=True,
        )
        record["removed"] = removed.returncode == 0
        record["remove_error"] = removed.stderr.strip() or None
        atomic_write_json(attempt.path / "container.json", record)
        message = "created container does not retain the exact attempt ownership identity"
        if removed.returncode != 0:
            message += "; exact container ID force-removal failed: " + removed.stderr.strip()
        raise AttemptError(message)
    return record


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
        observed = payload[0]
        labels = observed["Config"]["Labels"] or {}
        return {
            "available": True,
            "id_matches": observed["Id"] == record["id"],
            "image_matches": observed["Image"] == record["image_id"],
            "labels_match": all(
                labels.get(key) == value for key, value in record["labels"].items()
            ),
        }
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
            removed = subprocess.run(
                ["docker", "rm", "--force", record["id"]],
                capture_output=True,
                text=True,
            )
        else:
            identity = container_identity(record)
            if not identity.get("labels_match", False):
                record["ownership_anomaly"] = identity
                errors.append("container labels changed before final evidence capture")
            payload = json.loads(inspect.stdout)[0]
            state = payload.get("State", {})
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
            removed = subprocess.run(
                ["docker", "rm", "--force", record["id"]],
                capture_output=True,
                text=True,
            )
        record["removed"] = removed.returncode == 0
        record["remove_error"] = removed.stderr.strip() or None
        if removed.returncode != 0:
            errors.append("owned container removal failed: " + removed.stderr.strip())
    except BaseException as error:
        errors.append(f"container finalization: {type(error).__name__}: {error}")
    try:
        atomic_write_json(attempt.path / "container.json", record)
    except OSError as error:
        errors.append(f"container evidence write: {error}")
    return errors


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


def child_environment(action: str) -> dict[str, str]:
    environment = dict(os.environ)
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
