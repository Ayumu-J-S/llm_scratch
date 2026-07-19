"""Direct OPS-001 dispatch around the canonical Hydra entrypoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from argparse import Namespace
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from operations.artifacts import (
    Attempt,
    AttemptError,
    atomic_write_json,
    create_attempt,
    load_json,
    process_identity,
    process_identity_matches,
    sha256_file,
    utc_now,
)
from operations.lifecycle import (
    capture_and_remove_container,
    child_environment,
    container_owned,
    create_container,
    stop_owned_process,
    wait_with_disk_watchdog,
    watchdog_targets,
)
from operations.preflight import (
    PreflightError,
    reject_writable_git_overlap,
    require_ready,
    run_preflight,
)
from training.checkpoint import load_checkpoint_for_generation


_PROFILE = {
    "smoke": "smoke_overfit",
    "eval": "evaluation",
    "benchmark": "benchmark",
}
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_QUESTION_FIELDS = {
    "hypothesis",
    "expected_result",
    "success_condition",
    "failure_condition",
    "stop_condition",
    "baseline",
}
_PROTECTED_ALWAYS = {
    "runtime.device",
    "hydra.run.dir",
    "hydra.output_subdir",
    "artifacts.checkpoints_dir",
}
_PROTECTED_BY_ACTION = {
    "resume": {"artifacts.resume_path"},
    "eval": {"evaluation.checkpoint_path", "evaluation.output_path", "evaluation.device"},
    "benchmark": {
        "benchmark.checkpoint_path",
        "benchmark.output_root",
        "benchmark.output_path",
        "benchmark.device",
    },
}
_ENTRYPOINT = {
    "smoke": "src/train.py",
    "train": "src/train.py",
    "resume": "src/train.py",
    "eval": "src/evaluate.py",
    "benchmark": "src/benchmark.py",
}


def dispatch(args: Namespace, overrides: list[str], *, root_dir: Path) -> int:
    if args.action == "status":
        return _print_status(_existing_attempt(args), root_dir=root_dir)
    if args.action == "handoff":
        from operations.handoff import generate_handoff

        attempt = _existing_attempt(args)
        handoff = generate_handoff(attempt, root_dir=root_dir)
        print(handoff)
        return 0
    if args.action == "resume" and not args.retry_from:
        raise AttemptError("resume requires --retry-from to preserve interruption/failure lineage")
    reject_writable_git_overlap(root_dir, args.run_root, purpose="run_root")

    attempt = create_attempt(
        run_root=args.run_root,
        run_id=args.run_id,
        attempt_id=args.attempt_id,
        action=args.action,
        executor=args.executor,
        device=args.device,
        retry_from=args.retry_from,
    )
    try:
        initial_command = {
            "schema_version": 1,
            "action": args.action,
            "executor": args.executor,
            "device": args.device,
            "requested_hydra_overrides": list(overrides),
            "checkpoint_path": (
                str(args.checkpoint.expanduser().resolve())
                if getattr(args, "checkpoint", None) is not None
                else None
            ),
            "experiment_record_path": (
                str(args.experiment_record.expanduser().resolve())
                if getattr(args, "experiment_record", None) is not None
                else None
            ),
            "created_at": utc_now(),
        }
        atomic_write_json(attempt.path / "command.json", initial_command)
        selected_profile = _selected_profile(args, root_dir=root_dir)
        composed_overrides = _operational_overrides(
            args,
            overrides,
            attempt=attempt,
            selected_profile=selected_profile,
        )
        cfg = _compose(root_dir, composed_overrides)
        resolved_path = attempt.path / "resolved_config.yaml"
        OmegaConf.save(cfg, resolved_path, resolve=True)
        command_record = {
            "schema_version": 1,
            "action": args.action,
            "executor": args.executor,
            "device": args.device,
            "hydra_overrides": composed_overrides,
            "resolved_config_path": str(resolved_path),
            "resolved_config_sha256": sha256_file(resolved_path),
            "created_at": utc_now(),
        }
        declaration = _attempt_declaration(
            cfg,
            attempt=attempt,
            experiment_record=getattr(args, "experiment_record", None),
        )
        atomic_write_json(attempt.path / "declaration.json", declaration)
        command_record["declaration_sha256"] = sha256_file(attempt.path / "declaration.json")
        command_record["experiment_record"] = declaration["source"]
        atomic_write_json(attempt.path / "command.json", command_record)
        attempt.transition("preflighting")
        checkpoint_path = getattr(args, "checkpoint", None)
        preflight = run_preflight(
            cfg,
            root_dir=root_dir,
            run_root=attempt.run_root,
            action=args.action,
            executor=args.executor,
            device=args.device,
            image=args.image,
            checkpoint_path=checkpoint_path,
        )
        atomic_write_json(attempt.path / "preflight.json", preflight)
        atomic_write_json(
            attempt.path / "plan.json",
            _attempt_plan(attempt=attempt, preflight=preflight),
        )
        if preflight.get("login_prompt"):
            print(str(preflight["login_prompt"]), file=sys.stderr)
        if not preflight["ready"]:
            raise PreflightError(
                "preflight blocked: "
                + "; ".join(preflight["local_errors"] + preflight["online_errors"])
            )
        attempt.transition("ready")
        if args.action in {"config-check", "preflight"}:
            result = _nonexecution_result(attempt, preflight)
            atomic_write_json(attempt.path / "result.json", result)
            attempt.transition("succeeded", outcome=result["outcome"])
            print(attempt.path)
            return 0
        require_ready(preflight)
        return _execute(
            attempt,
            args=args,
            root_dir=root_dir,
            command_record=command_record,
            preflight=preflight,
        )
    except BaseException as error:
        _record_prelaunch_failure(attempt, error)
        if isinstance(error, KeyboardInterrupt):
            return 130
        print(f"OPS attempt failed: {error}", file=sys.stderr)
        return 1


def _compose(root_dir: Path, overrides: list[str]) -> DictConfig:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base=None, config_dir=str(root_dir / "config")):
        return hydra.compose(config_name="train", overrides=overrides)


def _selected_profile(args: Namespace, *, root_dir: Path) -> str | None:
    if args.action in {"preflight", "train"}:
        selected = str(args.profile)
    elif args.action == "resume":
        try:
            checkpoint = load_checkpoint_for_generation(args.checkpoint.expanduser().resolve())
            state = _mapping(checkpoint.payload.get("state"), "checkpoint state")
            resolved = _mapping(
                state.get("resolved_config"),
                "checkpoint resolved config",
            )
            profile = _mapping(resolved.get("profile"), "checkpoint profile")
            selected = str(profile.get("name", ""))
        except Exception as error:
            raise AttemptError(f"cannot derive resume profile from checkpoint: {error}") from error
    else:
        selected = _PROFILE.get(args.action)
    if selected is None:
        return None
    if _PROFILE_NAME.fullmatch(selected) is None:
        raise AttemptError(f"invalid canonical profile name: {selected!r}")
    profile_path = root_dir / "config" / "profile" / f"{selected}.yaml"
    if not profile_path.is_file():
        raise AttemptError(f"canonical profile does not exist: {selected}")
    return selected


def _operational_overrides(
    args: Namespace,
    user_overrides: list[str],
    *,
    attempt: Attempt,
    selected_profile: str | None = None,
) -> list[str]:
    protected = set(_PROTECTED_ALWAYS) | set(_PROTECTED_BY_ACTION.get(args.action, set()))
    if selected_profile is not None:
        protected.add("profile")
    for override in user_overrides:
        key = _override_key(override)
        if key == "hydra" or key.startswith("hydra."):
            raise AttemptError("Hydra output authority belongs to llm-scratch-ops")
        if any(
            key == protected_key
            or key.startswith(protected_key + ".")
            or protected_key.startswith(key + ".")
            for protected_key in protected
        ):
            raise AttemptError(f"Hydra override {key!r} belongs to llm-scratch-ops")

    values: list[str] = []
    if selected_profile is not None:
        values.append(f"profile={selected_profile}")
    values.extend(user_overrides)
    values.append(f"runtime.device={args.device}")
    values.append(f"hydra.run.dir={_hydra_string(attempt.path / 'work')}")
    values.append("hydra.output_subdir=null")
    values.append("artifacts.checkpoints_dir=checkpoints")
    if args.action == "smoke":
        values.extend(
            [
                "wandb.mode=disabled",
                "wandb.artifact.policy=none",
            ]
        )
    elif args.action == "resume":
        values.append(f"artifacts.resume_path={_hydra_string(args.checkpoint.resolve())}")
    elif args.action == "eval":
        result = attempt.path / "results" / "evaluation.json"
        values.extend(
            [
                f"evaluation.checkpoint_path={_hydra_string(args.checkpoint.resolve())}",
                f"evaluation.output_path={_hydra_string(result)}",
                f"evaluation.device={args.device}",
            ]
        )
    elif args.action == "benchmark":
        result_root = attempt.path / "results" / "benchmark"
        values.extend(
            [
                f"benchmark.checkpoint_path={_hydra_string(args.checkpoint.resolve())}",
                f"benchmark.output_root={_hydra_string(result_root)}",
                "benchmark.output_path=null",
                f"benchmark.device={args.device}",
            ]
        )
    return values


def _execute(
    attempt: Attempt,
    *,
    args: Namespace,
    root_dir: Path,
    command_record: dict[str, Any],
    preflight: dict[str, Any],
) -> int:
    with _capture_stop_signals() as stop_request:
        return _execute_with_stop_request(
            attempt,
            args=args,
            root_dir=root_dir,
            command_record=command_record,
            preflight=preflight,
            stop_request=stop_request,
        )


def _execute_with_stop_request(
    attempt: Attempt,
    *,
    args: Namespace,
    root_dir: Path,
    command_record: dict[str, Any],
    preflight: dict[str, Any],
    stop_request: dict[str, Any],
) -> int:
    inner_command = [
        sys.executable,
        str(root_dir / _ENTRYPOINT[args.action]),
        *command_record["hydra_overrides"],
    ]
    command_record = {**command_record, "inner_command": inner_command}
    started = utc_now()
    monotonic_started = time.monotonic()
    stdout_path = attempt.path / "stdout.log"
    stderr_path = attempt.path / "stderr.log"
    attempt.transition("running", started_at=started)

    watchdog: dict[str, Any] = {
        "triggered": False,
        "filesystems": watchdog_targets(preflight),
    }
    container_record: dict[str, Any] | None = None
    process: subprocess.Popen[bytes] | None = None
    trusted_ownership: dict[str, Any] | None = None
    exit_code = 1
    lifecycle_errors: list[str] = []
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            if args.executor == "host":
                process = subprocess.Popen(
                    inner_command,
                    cwd=root_dir,
                    stdout=stdout,
                    stderr=stderr,
                    env=child_environment(
                        args.action,
                        wandb_evidence_path=attempt.path / "work" / "wandb_events.jsonl",
                    ),
                    start_new_session=True,
                )
                trusted_ownership = process_identity(process.pid)
                atomic_write_json(attempt.path / "pid.json", trusted_ownership)
                command_record["launch_command"] = inner_command
            else:
                container_record = create_container(
                    attempt,
                    args=args,
                    root_dir=root_dir,
                    inner_command=inner_command,
                    preflight=preflight,
                )
                atomic_write_json(attempt.path / "container.json", container_record)
                launch_command = ["docker", "start", "--attach", container_record["id"]]
                command_record["launch_command"] = launch_command
                process = subprocess.Popen(
                    launch_command,
                    cwd=root_dir,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                trusted_ownership = process_identity(process.pid)
                atomic_write_json(attempt.path / "pid.json", trusted_ownership)
            atomic_write_json(attempt.path / "command.json", command_record)
            exit_code = wait_with_disk_watchdog(
                process,
                attempt=attempt,
                watchdog=watchdog,
                container_record=container_record,
                trusted_ownership=trusted_ownership,
                stop_request=stop_request,
            )
        except KeyboardInterrupt:
            watchdog["triggered"] = True
            watchdog["reason"] = "operator_interrupt"
        except BaseException as error:
            lifecycle_errors.append(f"execution lifecycle: {type(error).__name__}: {error}")
        finally:
            if process is not None and process.poll() is None:
                try:
                    lifecycle_errors.extend(
                        stop_owned_process(
                            process,
                            attempt=attempt,
                            container_record=container_record,
                            trusted_ownership=trusted_ownership,
                        )
                    )
                except BaseException as error:
                    lifecycle_errors.append(f"owned child cleanup: {type(error).__name__}: {error}")
            if process is not None:
                try:
                    exit_code = process.wait(timeout=15)
                except BaseException as error:
                    lifecycle_errors.append(
                        f"child wait finalization: {type(error).__name__}: {error}"
                    )
            for label, handle in (("stdout", stdout), ("stderr", stderr)):
                try:
                    handle.flush()
                    os.fsync(handle.fileno())
                except OSError as error:
                    lifecycle_errors.append(f"{label} evidence fsync: {error}")
            if container_record is not None:
                lifecycle_errors.extend(capture_and_remove_container(attempt, container_record))

    _apply_pending_signal(attempt, watchdog=watchdog, stop_request=stop_request)
    try:
        result = _execution_result(
            attempt,
            action=args.action,
            exit_code=exit_code,
            started_at=started,
            ended_at=utc_now(),
            elapsed_seconds=time.monotonic() - monotonic_started,
            watchdog=watchdog,
            lifecycle_errors=lifecycle_errors,
        )
    except BaseException as error:
        lifecycle_errors.append(f"result collection: {type(error).__name__}: {error}")
        result = _fallback_execution_result(
            attempt,
            action=args.action,
            exit_code=exit_code,
            started_at=started,
            elapsed_seconds=time.monotonic() - monotonic_started,
            watchdog=watchdog,
            lifecycle_errors=lifecycle_errors,
        )
    _apply_pending_signal(attempt, watchdog=watchdog, stop_request=stop_request)
    if watchdog.get("reason") == "external_signal" and result["outcome"] == "succeeded":
        result["outcome"] = "stopped"
        result["watchdog"] = watchdog
        result["diagnosis"] = {
            "category": "external_signal",
            "summary": f"attempt stopped by {watchdog.get('signal_name')}",
            "retry_recommended": True,
        }
        atomic_write_json(attempt.path / "diagnosis.json", result["diagnosis"])
    atomic_write_json(attempt.path / "result.json", result)
    status = (
        "succeeded"
        if result["outcome"] == "succeeded"
        else "stopped"
        if result["outcome"] == "stopped"
        else "failed"
    )
    attempt.transition(status, outcome=result["outcome"], exit_code=exit_code)
    print(attempt.path)
    return 0 if status == "succeeded" else 1


@contextmanager
def _capture_stop_signals():
    request: dict[str, Any] = {}
    previous: dict[int, Any] = {}

    def capture(signum, _frame) -> None:
        if "signal" not in request:
            request["signal"] = signum
            request["signal_name"] = signal.Signals(signum).name

    try:
        for signum in (signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, capture)
        yield request
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _apply_pending_signal(
    attempt: Attempt,
    *,
    watchdog: dict[str, Any],
    stop_request: Mapping[str, Any],
) -> None:
    if stop_request.get("signal") is None or watchdog.get("reason") == "external_signal":
        return
    watchdog.update(
        {
            "triggered": True,
            "reason": "external_signal",
            "signal": int(stop_request["signal"]),
            "signal_name": str(stop_request.get("signal_name", "unknown")),
        }
    )
    attempt.event(
        "external_signal_recorded",
        signal=watchdog["signal"],
        signal_name=watchdog["signal_name"],
    )


def _execution_result(
    attempt: Attempt,
    *,
    action: str,
    exit_code: int,
    started_at: str,
    ended_at: str,
    elapsed_seconds: float,
    watchdog: dict[str, Any],
    lifecycle_errors: list[str],
) -> dict[str, Any]:
    metrics = _metrics_summary(attempt.path / "work" / "checkpoints" / "metrics.jsonl")
    checkpoints = _checkpoint_status(attempt.path / "work" / "checkpoints")
    output_files = _result_outputs(attempt, action)
    stderr_tail = _tail(attempt.path / "stderr.log")
    invalid_numeric = bool(metrics.get("invalid_numeric")) or "non-finite" in stderr_tail.lower()
    checkpoint_integrity_failed = action in {"smoke", "train", "resume"} and (
        not checkpoints["files"]
        or any(record.get("verified") is not True for record in checkpoints["files"])
    )
    if lifecycle_errors:
        outcome = "failed"
    elif watchdog.get("triggered"):
        outcome = "stopped"
    elif invalid_numeric:
        outcome = "invalid_numeric"
    elif checkpoint_integrity_failed:
        outcome = "failed"
    elif exit_code == 0 and output_files.get("complete", True):
        outcome = "succeeded"
    else:
        outcome = "failed"
    diagnosis = {
        "category": (
            "disk_reserve"
            if watchdog.get("reason") == "disk_reserve"
            else "operator_interrupt"
            if watchdog.get("reason") == "operator_interrupt"
            else "external_signal"
            if watchdog.get("reason") == "external_signal"
            else "invalid_numeric"
            if invalid_numeric
            else "checkpoint_integrity"
            if checkpoint_integrity_failed
            else "none"
            if outcome == "succeeded"
            else "lifecycle_failure"
            if lifecycle_errors
            else "child_exit"
        ),
        "summary": (
            "attempt completed without an operational failure"
            if outcome == "succeeded"
            else f"attempt stopped by {watchdog.get('signal_name')}"
            if watchdog.get("reason") == "external_signal"
            else "; ".join(lifecycle_errors)
            if lifecycle_errors
            else "required final/recovery checkpoint evidence is missing or invalid"
            if checkpoint_integrity_failed
            else stderr_tail[-2000:] or f"child exited with status {exit_code}"
        ),
        "retry_recommended": outcome != "succeeded",
    }
    atomic_write_json(attempt.path / "checkpoint_status.json", checkpoints)
    atomic_write_json(attempt.path / "diagnosis.json", diagnosis)
    return {
        "schema_version": 1,
        "run_id": attempt.run_id,
        "attempt_id": attempt.attempt_id,
        "action": action,
        "outcome": outcome,
        "exit_code": exit_code,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": elapsed_seconds,
        "counters": metrics.get("counters", {}),
        "metrics": metrics,
        "checkpoint_status": checkpoints,
        "outputs": output_files,
        "diagnosis": diagnosis,
        "watchdog": watchdog,
        "lifecycle_errors": lifecycle_errors,
    }


def _fallback_execution_result(
    attempt: Attempt,
    *,
    action: str,
    exit_code: int,
    started_at: str,
    elapsed_seconds: float,
    watchdog: dict[str, Any],
    lifecycle_errors: list[str],
) -> dict[str, Any]:
    diagnosis = {
        "category": "lifecycle_failure",
        "summary": "; ".join(lifecycle_errors),
        "retry_recommended": True,
    }
    checkpoints = {"directory": None, "files": [], "last_verified": None}
    for name, value in (("diagnosis.json", diagnosis), ("checkpoint_status.json", checkpoints)):
        try:
            atomic_write_json(attempt.path / name, value)
        except OSError:
            pass
    return {
        "schema_version": 1,
        "run_id": attempt.run_id,
        "attempt_id": attempt.attempt_id,
        "action": action,
        "outcome": "failed",
        "exit_code": exit_code,
        "started_at": started_at,
        "ended_at": utc_now(),
        "elapsed_seconds": elapsed_seconds,
        "counters": {},
        "metrics": {},
        "checkpoint_status": checkpoints,
        "outputs": {"complete": False, "files": []},
        "diagnosis": diagnosis,
        "watchdog": watchdog,
        "lifecycle_errors": lifecycle_errors,
    }


def _nonexecution_result(attempt: Attempt, preflight: Mapping[str, Any]) -> dict[str, Any]:
    timestamp = utc_now()
    diagnosis = {
        "category": "none",
        "summary": "non-destructive checks completed",
        "retry_recommended": False,
    }
    checkpoint_status = {"directory": None, "files": [], "last_verified": None}
    atomic_write_json(attempt.path / "checkpoint_status.json", checkpoint_status)
    atomic_write_json(attempt.path / "diagnosis.json", diagnosis)
    return {
        "schema_version": 1,
        "run_id": attempt.run_id,
        "attempt_id": attempt.attempt_id,
        "action": attempt.state()["action"],
        "outcome": "succeeded",
        "exit_code": 0,
        "started_at": timestamp,
        "ended_at": timestamp,
        "elapsed_seconds": 0.0,
        "counters": {},
        "metrics": {},
        "checkpoint_status": checkpoint_status,
        "outputs": {"complete": True, "files": []},
        "diagnosis": diagnosis,
        "watchdog": {
            "triggered": False,
            "effective_live_floor_bytes": preflight["checks"]["storage"].get(
                "effective_run_live_floor_bytes"
            ),
        },
        "lifecycle_errors": [],
    }


def _attempt_plan(
    *,
    attempt: Attempt,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    storage = _mapping(preflight["checks"]["storage"], "storage preflight")
    return {
        "schema_version": 1,
        "recorded_at": utc_now(),
        "declaration_sha256": sha256_file(attempt.path / "declaration.json"),
        "storage_forecast": {
            "maximum_atomic_write_bytes": storage.get("maximum_atomic_write_bytes"),
            "checkpoint_plan_bytes": storage.get("checkpoint_plan_bytes"),
            "effective_run_live_floor_bytes": storage.get("effective_run_live_floor_bytes"),
            "filesystems": storage.get("filesystems", []),
        },
    }


def _attempt_declaration(
    cfg: DictConfig,
    *,
    attempt: Attempt,
    experiment_record: Path | None,
) -> dict[str, Any]:
    if experiment_record is not None:
        return _load_experiment_declaration(experiment_record)
    action = str(attempt.state()["action"])
    purpose = str(cfg.profile.get("purpose", ""))
    if purpose == "pretraining" and action in {"preflight", "train", "resume"}:
        raise AttemptError(
            f"{action} with a pretraining profile requires --experiment-record"
        )
    profile = str(cfg.profile.name)
    training = cfg.get("training", {})
    retry = attempt.state().get("retry_from")
    return {
        "schema_version": 1,
        "recorded_at": utc_now(),
        "source": {"kind": "generated_ops_fixture", "path": None, "sha256": None},
        "ticket": "OPS-001",
        "predeclared_question": {
            "hypothesis": (
                f"The bounded {profile} {action} path completes under its declared "
                "limits and retains finite, integrity-bound evidence."
            ),
            "expected_result": (
                "non-destructive checks complete"
                if action in {"config-check", "preflight"}
                else "the canonical child exits successfully with finite local evidence"
            ),
            "success_condition": ("preflight ready, no reserve stop, exit zero, finite results"),
            "failure_condition": ("preflight, ownership, child, result, or integrity gate fails"),
            "stop_condition": (
                "stop before filesystem free space crosses the effective live floor"
            ),
            "baseline": retry or "none — no failed sibling baseline was declared",
        },
        "planned_budget": {
            "training": {
                "max_time": training.get("max_time"),
                "max_steps": training.get("max_steps"),
                "max_tokens": training.get("max_tokens"),
                "epochs": training.get("epochs"),
            },
            "storage_policy": {
                "live_floor_bytes": 120_000_000_000,
                "post_plan_reserve_bytes": 100_000_000_000,
                "maximum_inflight_write_rule": "128 bytes per parameter plus 4000000000 bytes",
            },
        },
    }


def _load_experiment_declaration(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    try:
        payload_bytes = source.read_bytes()
        value = json.loads(payload_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise AttemptError(f"invalid experiment declaration: {source}") from error
    declaration = _mapping(value, "experiment declaration")
    expected = {"schema_version", "ticket", "predeclared_question", "planned_budget"}
    if set(declaration) != expected or declaration.get("schema_version") != 1:
        raise AttemptError(
            "experiment declaration requires exactly schema_version, ticket, "
            "predeclared_question, and planned_budget"
        )
    ticket = declaration.get("ticket")
    if not isinstance(ticket, str) or not ticket.strip():
        raise AttemptError("experiment declaration ticket must be non-empty")
    question = _mapping(declaration.get("predeclared_question"), "predeclared question")
    if set(question) != _QUESTION_FIELDS:
        raise AttemptError(
            f"predeclared question fields must be exactly {sorted(_QUESTION_FIELDS)}"
        )
    for key in _QUESTION_FIELDS - {"baseline"}:
        value = question[key]
        if not isinstance(value, str) or not value.strip():
            raise AttemptError(f"predeclared question {key} must be non-empty")
    baseline = question["baseline"]
    if isinstance(baseline, Mapping):
        if not baseline:
            raise AttemptError("predeclared baseline mapping must be non-empty")
    elif not isinstance(baseline, str) or not baseline.strip():
        raise AttemptError("predeclared baseline must be a non-empty string or mapping")
    planned_budget = _mapping(declaration.get("planned_budget"), "planned budget")
    if not planned_budget:
        raise AttemptError("planned budget must be non-empty")
    return {
        "schema_version": 1,
        "recorded_at": utc_now(),
        "source": {
            "kind": "explicit_experiment_record",
            "path": str(source),
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        },
        "ticket": ticket,
        "predeclared_question": dict(question),
        "planned_budget": dict(planned_budget),
    }


def _record_prelaunch_failure(attempt: Attempt, error: BaseException) -> None:
    try:
        state = attempt.state()
    except AttemptError:
        return
    if state.get("status") in {"succeeded", "failed", "stopped"}:
        return
    diagnosis = {
        "category": "preflight_or_launch",
        "summary": str(error),
        "retry_recommended": True,
    }
    checkpoint_status = _checkpoint_status(attempt.path / "work" / "checkpoints")
    result = {
        "schema_version": 1,
        "run_id": attempt.run_id,
        "attempt_id": attempt.attempt_id,
        "action": state.get("action"),
        "outcome": "failed",
        "exit_code": None,
        "started_at": state.get("created_at"),
        "ended_at": utc_now(),
        "elapsed_seconds": 0.0,
        "counters": {},
        "metrics": {},
        "checkpoint_status": checkpoint_status,
        "outputs": {"complete": False, "files": []},
        "diagnosis": diagnosis,
        "watchdog": {"triggered": False},
        "lifecycle_errors": [],
    }
    for name, value in (
        ("diagnosis.json", diagnosis),
        ("checkpoint_status.json", checkpoint_status),
        ("result.json", result),
    ):
        path = attempt.path / name
        if not path.exists():
            atomic_write_json(path, value)
    try:
        if state["status"] == "created":
            attempt.transition("preflighting")
        attempt.transition("failed", outcome="failed")
    except AttemptError:
        pass


def status_payload(attempt: Attempt, *, root_dir: Path) -> dict[str, Any]:
    state = attempt.state()
    integrity: dict[str, Any] = {
        "state_sha256": sha256_file(attempt.state_path),
        "result_present": (attempt.path / "result.json").is_file(),
        "retry_binding_valid": True,
    }
    retry = state.get("retry_from")
    if isinstance(retry, Mapping):
        parent = Attempt(attempt.run_root, attempt.run_id, str(retry.get("attempt_id")))
        try:
            integrity["retry_binding_valid"] = sha256_file(parent.state_path) == retry.get(
                "state_sha256"
            ) and sha256_file(parent.path / "result.json") == retry.get("result_sha256")
        except OSError:
            integrity["retry_binding_valid"] = False
    ownership: dict[str, Any] = {"status": "not_running"}
    if state.get("status") == "running":
        pid_path = attempt.path / "pid.json"
        if not pid_path.is_file():
            ownership = {"status": "stale", "reason": "missing_pid_record"}
        else:
            pid = load_json(pid_path)
            ownership = {
                "status": "owned" if process_identity_matches(pid) else "stale",
                "pid": pid.get("pid"),
            }
            container_path = attempt.path / "container.json"
            if container_path.is_file():
                container = load_json(container_path)
                ownership["container_owned"] = container_owned(container)
                if not ownership["container_owned"]:
                    ownership["status"] = "stale"
    return {
        "schema_version": 1,
        "root": str(root_dir),
        "attempt_path": str(attempt.path),
        "state": state,
        "ownership": ownership,
        "integrity": integrity,
        "result": (
            load_json(attempt.path / "result.json")
            if (attempt.path / "result.json").is_file()
            else None
        ),
    }


def _print_status(attempt: Attempt, *, root_dir: Path) -> int:
    payload = status_payload(attempt, root_dir=root_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if payload["ownership"]["status"] == "stale":
        return 2
    if payload["integrity"]["retry_binding_valid"] is not True:
        return 3
    return 0


def _metrics_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"records": 0, "counters": {}, "invalid_numeric": False, "path": None}
    records = 0
    counters: dict[str, Any] = {}
    last_metrics: dict[str, Any] = {}
    invalid: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AttemptError(f"invalid metrics JSON at {path}:{line_number}") from error
            if not isinstance(row, Mapping):
                raise AttemptError(f"metrics row is not an object at {path}:{line_number}")
            records += 1
            for key in ("optimizer_step", "target_tokens", "elapsed_seconds"):
                if key in row:
                    counters[key] = row[key]
            for key, value in row.items():
                normalized = str(key).lower()
                scientific = any(
                    token in normalized
                    for token in ("loss", "nll", "perplexity", "gradient", "nonfinite")
                )
                if scientific and isinstance(value, (int, float)) and not isinstance(value, bool):
                    if not math.isfinite(float(value)):
                        invalid.append(f"line {line_number}: {key}")
                    if "nonfinite" in normalized and float(value) != 0:
                        invalid.append(f"line {line_number}: {key}={value}")
                if scientific:
                    last_metrics[str(key)] = value
    return {
        "records": records,
        "counters": counters,
        "last_scientific_metrics": last_metrics,
        "invalid_numeric": bool(invalid),
        "invalid_numeric_evidence": invalid,
        "path": str(path),
        "sha256": sha256_file(path),
    }


def _checkpoint_status(directory: Path) -> dict[str, Any]:
    files = []
    if directory.is_dir():
        for path in sorted(directory.glob("*.pt")):
            record = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "verified": False,
                "verification_error": None,
            }
            try:
                loaded = load_checkpoint_for_generation(path)
                physical = loaded.physical_identity
                if (
                    physical["sha256"] != record["sha256"]
                    or physical["size_bytes"] != record["size_bytes"]
                ):
                    raise AttemptError("checkpoint physical identity changed during verification")
                record["verified"] = True
                record["kind"] = loaded.payload["kind"]
            except Exception as error:
                record["verification_error"] = str(error)
            files.append(record)
    verified = [record for record in files if record["verified"]]
    return {
        "directory": str(directory),
        "files": files,
        "last_verified": verified[-1] if verified else None,
    }


def _result_outputs(attempt: Attempt, action: str) -> dict[str, Any]:
    paths: list[Path] = []
    if action == "eval":
        paths = [attempt.path / "results" / "evaluation.json"]
    elif action == "benchmark":
        root = attempt.path / "results" / "benchmark"
        paths = sorted(root.glob("*.json")) if root.is_dir() else []
    files = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
        if path.is_file()
    ]
    expected = action in {"eval", "benchmark"}
    return {"complete": bool(files) if expected else True, "files": files}


def _existing_attempt(args: Namespace) -> Attempt:
    identifier = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    if identifier.fullmatch(args.run_id) is None or identifier.fullmatch(args.attempt_id) is None:
        raise AttemptError("run ID and attempt ID cannot contain path separators")
    attempt = Attempt(Path(args.run_root).expanduser().resolve(), args.run_id, args.attempt_id)
    if not attempt.state_path.is_file():
        raise AttemptError(f"attempt does not exist: {attempt.path}")
    return attempt


def _override_key(override: str) -> str:
    value = override
    while value.startswith("+"):
        value = value[1:]
    if value.startswith("~"):
        value = value[1:]
    key = value.split("=", 1)[0]
    if not key:
        raise AttemptError(f"invalid Hydra override: {override!r}")
    return key


def _hydra_string(value: str | Path) -> str:
    return json.dumps(str(value))


def _tail(path: Path, limit: int = 16_384) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - limit))
        return handle.read().decode("utf-8", errors="replace")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AttemptError(f"{label} must be a mapping")
    return value
