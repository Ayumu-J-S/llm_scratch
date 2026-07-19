"""Generate and validate a machine-checkable EXP-001-shaped handoff."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from operations.artifacts import (
    Attempt,
    atomic_write_bytes,
    atomic_write_json,
    load_json,
    sha256_file,
    utc_now,
)
from operations.runner import status_payload
from runtime.reproducibility import verify_run_manifest
from training.checkpoint import load_checkpoint_for_generation


_TOP_LEVEL = {
    "schema_version",
    "ticket",
    "recorded_at",
    "predeclared_question",
    "planned_budget",
    "launch_identity",
    "scientific_identity",
    "results",
    "integrity",
    "conclusion",
}
_QUESTION = {
    "hypothesis",
    "expected_result",
    "success_condition",
    "failure_condition",
    "stop_condition",
    "baseline",
}
_RESULTS = {
    "outcome",
    "exit_code",
    "started_at",
    "ended_at",
    "elapsed_seconds",
    "counters",
    "metrics",
    "checkpoint_status",
    "outputs",
    "diagnosis",
    "watchdog",
    "lifecycle_errors",
}
_RESULT_FILE = {"schema_version", "run_id", "attempt_id", "action", *_RESULTS}
_INTEGRITY = {
    "git",
    "resolved_config_sha256",
    "declaration_sha256",
    "result_sha256",
    "state_sha256",
    "retry_binding",
    "evidence_files",
}
_WANDB_ACTION_OUTCOMES = {
    "init": {"disabled", "succeeded", "failed"},
    "watch": {"disabled", "succeeded", "failed"},
    "runtime_summary": {"failed"},
    "log": {"failed"},
    "logging": {"failed"},
    "summary": {"succeeded", "failed"},
    "artifact": {"uploaded", "blocked", "upload_failed", "failed"},
    "artifact_cleanup": {"failed"},
    "finish": {"succeeded", "failed"},
    "unwatch": {"succeeded", "failed"},
}


class HandoffValidationError(ValueError):
    """The handoff omits or misbinds required experiment evidence."""


def generate_handoff(attempt: Attempt, *, root_dir: Path) -> Path:
    status = status_payload(attempt, root_dir=root_dir)
    state = _mapping(status["state"], "attempt state")
    if state.get("status") not in {"succeeded", "failed", "stopped"}:
        raise HandoffValidationError("handoff requires a terminal attempt")
    if status["ownership"]["status"] == "stale":
        raise HandoffValidationError("handoff refuses a stale running-attempt ownership record")
    if status["integrity"]["retry_binding_valid"] is not True:
        raise HandoffValidationError("retry sibling hashes no longer match")

    result = _mapping(status.get("result"), "attempt result")
    command = load_json(attempt.path / "command.json")
    preflight = load_json(attempt.path / "preflight.json")
    plan = load_json(attempt.path / "plan.json")
    declaration = load_json(attempt.path / "declaration.json")
    _validate_result_identity(result, state=state, attempt=attempt)
    _validate_attempt_artifacts(result, preflight=preflight, attempt=attempt)
    run_manifest = _run_manifest_evidence(
        attempt,
        root_dir=root_dir,
        action=str(state["action"]),
        outcome=str(result["outcome"]),
    )
    wandb = _wandb_evidence(
        attempt,
        preflight=preflight,
        action=str(state["action"]),
        outcome=str(result["outcome"]),
    )
    runtime = _runtime_evidence(
        attempt,
        state=state,
        result=result,
        run_manifest=run_manifest,
    )
    evidence = _evidence_files(attempt)
    git = _mapping(_mapping(preflight["checks"], "preflight checks")["git"], "Git check")
    storage = _mapping(preflight["checks"]["storage"], "storage check")
    manifests = _mapping(preflight["checks"]["manifests"], "manifest check")
    result_fields = {field: result[field] for field in _RESULTS}
    action = str(state["action"])
    profile = _profile_from_config(attempt.path / "resolved_config.yaml")
    payload = {
        "schema_version": 1,
        "ticket": declaration["ticket"],
        "recorded_at": utc_now(),
        "predeclared_question": declaration["predeclared_question"],
        "planned_budget": declaration["planned_budget"],
        "launch_identity": {
            "run_id": attempt.run_id,
            "attempt_id": attempt.attempt_id,
            "action": action,
            "executor": state["executor"],
            "device": state["device"],
            "exact_command": command,
            "retry_from": state.get("retry_from"),
            "attempt_path": str(attempt.path),
        },
        "scientific_identity": {
            "profile": profile,
            "parameter_count": storage.get("parameter_count"),
            "tokenizer_fingerprint": manifests.get("tokenizer_fingerprint"),
            "data_manifests": manifests.get("data_manifests", []),
            "checkpoint": preflight["checks"].get("checkpoint"),
            "wandb": wandb,
            "run_manifest": run_manifest,
            "runtime": runtime,
            "storage_forecast": plan["storage_forecast"],
        },
        "results": result_fields,
        "integrity": {
            "git": {
                "sha": git["sha"],
                "dirty": git["dirty"],
                "worktree_status": git["worktree_status"],
                "lock_sha256": git["lock_sha256"],
            },
            "resolved_config_sha256": command["resolved_config_sha256"],
            "declaration_sha256": command["declaration_sha256"],
            "result_sha256": sha256_file(attempt.path / "result.json"),
            "state_sha256": sha256_file(attempt.state_path),
            "retry_binding": state.get("retry_from"),
            "evidence_files": evidence,
        },
        "conclusion": _conclusion(result, state=state, attempt=attempt),
    }
    validate_handoff(payload, attempt=attempt, root_dir=root_dir)
    json_path = attempt.path / "handoff.json"
    atomic_write_json(json_path, payload)
    markdown_path = attempt.path / "handoff.md"
    atomic_write_bytes(markdown_path, _markdown(payload, json_path).encode("utf-8"))
    return markdown_path


def validate_handoff(
    payload: Mapping[str, Any],
    *,
    attempt: Attempt | None = None,
    root_dir: Path | None = None,
) -> None:
    top = _exact(payload, _TOP_LEVEL, "handoff")
    if top["schema_version"] != 1:
        raise HandoffValidationError("handoff.schema_version must be 1")
    _nonempty(top["ticket"], "handoff.ticket")
    _nonempty(top["recorded_at"], "handoff.recorded_at")
    question = _exact(top["predeclared_question"], _QUESTION, "predeclared_question")
    for key in _QUESTION:
        if key == "baseline" and isinstance(question[key], Mapping):
            if not question[key]:
                raise HandoffValidationError("predeclared_question.baseline cannot be empty")
            continue
        _nonempty(question[key], f"predeclared_question.{key}")
    _mapping(top["planned_budget"], "planned_budget")
    launch = _mapping(top["launch_identity"], "launch_identity")
    for key in ("run_id", "attempt_id", "action", "executor", "device", "attempt_path"):
        _nonempty(launch.get(key), f"launch_identity.{key}")
    _mapping(launch.get("exact_command"), "launch_identity.exact_command")
    scientific = _mapping(top["scientific_identity"], "scientific_identity")
    for key in (
        "profile",
        "parameter_count",
        "tokenizer_fingerprint",
        "data_manifests",
        "wandb",
        "run_manifest",
        "runtime",
        "storage_forecast",
    ):
        if key not in scientific:
            raise HandoffValidationError(f"scientific_identity is missing {key}")
    results = _exact(top["results"], _RESULTS, "results")
    for key in _RESULTS:
        if results[key] is None and key != "exit_code":
            raise HandoffValidationError(f"results.{key} cannot be null")
    if results["outcome"] == "succeeded" and results["exit_code"] != 0:
        raise HandoffValidationError("successful handoff result requires exit_code 0")
    for key in ("counters", "metrics", "checkpoint_status", "outputs", "diagnosis", "watchdog"):
        _mapping(results[key], f"results.{key}")
    if not isinstance(results["lifecycle_errors"], list) or any(
        not isinstance(error, str) or not error for error in results["lifecycle_errors"]
    ):
        raise HandoffValidationError("results.lifecycle_errors must be a list of strings")
    if results["lifecycle_errors"] and results["outcome"] == "succeeded":
        raise HandoffValidationError("a result with lifecycle errors cannot be successful")
    integrity = _exact(top["integrity"], _INTEGRITY, "integrity")
    git = _mapping(integrity["git"], "integrity.git")
    for key in ("sha", "dirty", "worktree_status", "lock_sha256"):
        if key not in git:
            raise HandoffValidationError(f"integrity.git is missing {key}")
    for key in (
        "resolved_config_sha256",
        "declaration_sha256",
        "result_sha256",
        "state_sha256",
    ):
        _sha256(integrity[key], f"integrity.{key}")
    evidence = integrity["evidence_files"]
    if not isinstance(evidence, list) or not evidence:
        raise HandoffValidationError("integrity.evidence_files must be non-empty")
    for index, record in enumerate(evidence):
        item = _mapping(record, f"integrity.evidence_files[{index}]")
        if set(item) != {"path", "sha256", "size_bytes"}:
            raise HandoffValidationError("evidence file entries require path/sha256/size_bytes")
        _nonempty(item["path"], "evidence path")
        _sha256(item["sha256"], "evidence sha256")
        if not isinstance(item["size_bytes"], int) or item["size_bytes"] < 0:
            raise HandoffValidationError("evidence size_bytes must be non-negative")
    conclusion = _mapping(top["conclusion"], "conclusion")
    for key in ("condition_result", "evidence_backed_summary", "uncertainty", "next_step"):
        _nonempty(conclusion.get(key), f"conclusion.{key}")

    if attempt is not None:
        if root_dir is None:
            raise HandoffValidationError("root_dir is required for attempt-bound validation")
        if launch["run_id"] != attempt.run_id or launch["attempt_id"] != attempt.attempt_id:
            raise HandoffValidationError("handoff launch identity differs from attempt path")
        if Path(str(launch["attempt_path"])).resolve() != attempt.path.resolve():
            raise HandoffValidationError("handoff attempt path differs from launch identity")
        if integrity["result_sha256"] != sha256_file(attempt.path / "result.json"):
            raise HandoffValidationError("handoff result hash does not match result.json")
        if integrity["state_sha256"] != sha256_file(attempt.state_path):
            raise HandoffValidationError("handoff state hash does not match state.json")
        if integrity["resolved_config_sha256"] != sha256_file(
            attempt.path / "resolved_config.yaml"
        ):
            raise HandoffValidationError("handoff config hash does not match resolved config")
        if integrity["declaration_sha256"] != sha256_file(attempt.path / "declaration.json"):
            raise HandoffValidationError("handoff declaration hash does not match declaration")
        declaration = load_json(attempt.path / "declaration.json")
        if top["ticket"] != declaration.get("ticket"):
            raise HandoffValidationError("handoff ticket differs from declaration.json")
        if top["predeclared_question"] != declaration.get("predeclared_question"):
            raise HandoffValidationError("handoff question differs from declaration.json")
        if top["planned_budget"] != declaration.get("planned_budget"):
            raise HandoffValidationError("handoff budget differs from declaration.json")
        actual_result = load_json(attempt.path / "result.json")
        state = load_json(attempt.state_path)
        _validate_result_identity(actual_result, state=state, attempt=attempt)
        if {field: actual_result[field] for field in _RESULTS} != dict(results):
            raise HandoffValidationError("handoff results differ from result.json")
        preflight = load_json(attempt.path / "preflight.json")
        _validate_attempt_artifacts(actual_result, preflight=preflight, attempt=attempt)
        command = load_json(attempt.path / "command.json")
        expected_launch = {
            "run_id": attempt.run_id,
            "attempt_id": attempt.attempt_id,
            "action": state["action"],
            "executor": state["executor"],
            "device": state["device"],
            "exact_command": command,
            "retry_from": state.get("retry_from"),
            "attempt_path": str(attempt.path),
        }
        if dict(launch) != expected_launch:
            raise HandoffValidationError("handoff launch identity differs from attempt evidence")
        action = str(state["action"])
        outcome = str(actual_result["outcome"])
        actual_manifest = _run_manifest_evidence(
            attempt,
            root_dir=root_dir,
            action=action,
            outcome=outcome,
        )
        if scientific["run_manifest"] != actual_manifest:
            raise HandoffValidationError("handoff run-manifest evidence differs from child output")
        actual_wandb = _wandb_evidence(
            attempt,
            preflight=preflight,
            action=action,
            outcome=outcome,
        )
        if scientific["wandb"] != actual_wandb:
            raise HandoffValidationError("handoff W&B evidence differs from child output")
        actual_runtime = _runtime_evidence(
            attempt,
            state=state,
            result=actual_result,
            run_manifest=actual_manifest,
        )
        if scientific["runtime"] != actual_runtime:
            raise HandoffValidationError("handoff runtime evidence differs from child output")
        checks = _mapping(preflight.get("checks"), "preflight checks")
        storage = _mapping(checks.get("storage"), "storage check")
        manifests = _mapping(checks.get("manifests"), "manifest check")
        plan = load_json(attempt.path / "plan.json")
        expected_scientific = {
            "profile": _profile_from_config(attempt.path / "resolved_config.yaml"),
            "parameter_count": storage.get("parameter_count"),
            "tokenizer_fingerprint": manifests.get("tokenizer_fingerprint"),
            "data_manifests": manifests.get("data_manifests", []),
            "checkpoint": checks.get("checkpoint"),
            "wandb": actual_wandb,
            "run_manifest": actual_manifest,
            "runtime": actual_runtime,
            "storage_forecast": plan["storage_forecast"],
        }
        if dict(scientific) != expected_scientific:
            raise HandoffValidationError(
                "handoff scientific identity differs from retained attempt evidence"
            )
        git_check = _mapping(checks.get("git"), "Git check")
        expected_git = {
            "sha": git_check["sha"],
            "dirty": git_check["dirty"],
            "worktree_status": git_check["worktree_status"],
            "lock_sha256": git_check["lock_sha256"],
        }
        if dict(git) != expected_git:
            raise HandoffValidationError("handoff Git identity differs from preflight evidence")
        if integrity["retry_binding"] != state.get("retry_from"):
            raise HandoffValidationError("handoff retry binding differs from attempt state")
        if dict(conclusion) != _conclusion(actual_result, state=state, attempt=attempt):
            raise HandoffValidationError("handoff conclusion differs from terminal evidence")
        expected_evidence = _evidence_files(attempt)
        if evidence != expected_evidence:
            raise HandoffValidationError("handoff evidence inventory differs from attempt files")
        for record in evidence:
            path = Path(str(record["path"]))
            if not path.is_file() or path.stat().st_size != record["size_bytes"]:
                raise HandoffValidationError(f"handoff evidence file changed: {path}")
            if sha256_file(path) != record["sha256"]:
                raise HandoffValidationError(f"handoff evidence hash changed: {path}")
        current = status_payload(attempt, root_dir=root_dir)
        if current["integrity"]["retry_binding_valid"] is not True:
            raise HandoffValidationError("handoff retry binding no longer validates")


def _evidence_files(attempt: Attempt) -> list[dict[str, Any]]:
    paths = [
        attempt.path / "command.json",
        attempt.path / "resolved_config.yaml",
        attempt.path / "preflight.json",
        attempt.path / "declaration.json",
        attempt.path / "plan.json",
        attempt.path / "state.json",
        attempt.path / "result.json",
        attempt.path / "checkpoint_status.json",
        attempt.path / "diagnosis.json",
    ]
    state = attempt.state()
    if state.get("started_at") is not None:
        for required_log in ("stdout.log", "stderr.log"):
            path = attempt.path / required_log
            if not path.is_file():
                raise HandoffValidationError(f"execution log is missing: {path}")
            paths.append(path)
    for optional in ("pid.json", "container.json"):
        path = attempt.path / optional
        if path.is_file():
            paths.append(path)
    for optional in (
        attempt.path / "work" / "run_manifest.json",
        attempt.path / "work" / "wandb_events.jsonl",
        attempt.path / "work" / "checkpoints" / "wandb_events.jsonl",
    ):
        if optional.is_file():
            paths.append(optional)
    result = load_json(attempt.path / "result.json")
    metrics = _mapping(result.get("metrics", {}), "result metrics")
    metrics_path = metrics.get("path")
    if metrics_path is not None:
        path = _attempt_owned_path(metrics_path, attempt, "metrics evidence")
        if not path.is_file():
            raise HandoffValidationError(f"metrics evidence file is missing: {path}")
        paths.append(path)
    measurement_path = _measurement_evidence_path(
        attempt,
        outcome=str(result.get("outcome", "")),
    )
    if measurement_path is not None:
        paths.append(measurement_path)
    outputs = _mapping(result.get("outputs"), "result outputs")
    for record in outputs.get("files", []):
        item = _mapping(record, "result output file")
        paths.append(Path(str(item.get("path", ""))))
    paths.extend(sorted(attempt.events_dir.glob("*.json")))
    paths = list(dict.fromkeys(paths))
    records = []
    for path in paths:
        if not path.is_file():
            raise HandoffValidationError(f"required evidence file is missing: {path}")
        records.append(
            {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        )
    return records


def _validate_attempt_artifacts(
    result: Mapping[str, Any],
    *,
    preflight: Mapping[str, Any],
    attempt: Attempt,
) -> None:
    action = str(attempt.state().get("action", ""))
    outcome = str(result.get("outcome", ""))
    metrics = _mapping(result.get("metrics", {}), "result metrics")
    metrics_path = metrics.get("path")
    if metrics_path is not None:
        path = _attempt_owned_path(metrics_path, attempt, "metrics evidence")
        if not path.is_file():
            raise HandoffValidationError(f"metrics evidence file is missing: {path}")
        digest = metrics.get("sha256")
        _sha256(digest, "metrics evidence SHA-256")
        if sha256_file(path) != digest:
            raise HandoffValidationError(f"metrics evidence hash changed: {path}")
    _measurement_evidence_path(attempt, outcome=outcome)
    checkpoint_status = _mapping(result.get("checkpoint_status"), "checkpoint status")
    checkpoint_files = checkpoint_status.get("files", [])
    if not isinstance(checkpoint_files, list):
        raise HandoffValidationError("checkpoint status files must be a list")
    if outcome == "succeeded" and action in {"smoke", "train", "resume"}:
        if not checkpoint_files:
            raise HandoffValidationError("successful training requires checkpoint evidence")
        if any(
            not isinstance(record, Mapping) or record.get("verified") is not True
            for record in checkpoint_files
        ):
            raise HandoffValidationError("successful training has an unverified checkpoint")
    for index, value in enumerate(checkpoint_files):
        record = _mapping(value, f"checkpoint status files[{index}]")
        path = _attempt_owned_path(record.get("path"), attempt, "checkpoint")
        _verify_file_identity(path, record, f"checkpoint {path}")
        if record.get("verified") is True:
            try:
                loaded = load_checkpoint_for_generation(path)
            except Exception as error:
                raise HandoffValidationError(
                    f"checkpoint verification failed: {path}: {error}"
                ) from error
            physical = loaded.physical_identity
            if physical.get("sha256") != record.get("sha256") or physical.get(
                "size_bytes"
            ) != record.get("size_bytes"):
                raise HandoffValidationError(f"checkpoint changed during verification: {path}")
        elif outcome != "succeeded":
            error = record.get("verification_error")
            if not isinstance(error, str) or not error:
                raise HandoffValidationError(
                    "unverified failed-checkpoint evidence requires a verification error"
                )

    outputs = _mapping(result.get("outputs"), "result outputs")
    files = outputs.get("files", [])
    if not isinstance(files, list):
        raise HandoffValidationError("result output files must be a list")
    for index, value in enumerate(files):
        record = _mapping(value, f"result output files[{index}]")
        path = _attempt_owned_path(record.get("path"), attempt, "result output")
        _verify_file_identity(path, record, f"result output {path}")

    checks = _mapping(preflight.get("checks"), "preflight checks")
    checkpoint = checks.get("checkpoint")
    if checkpoint is None:
        return
    checkpoint_check = _mapping(checkpoint, "preflight checkpoint")
    physical = checkpoint_check.get("physical_identity")
    if not isinstance(physical, Mapping):
        return
    path = Path(str(physical.get("path", "")))
    _verify_file_identity(path, physical, f"input checkpoint {path}")
    if checkpoint_check.get("status") == "passed":
        try:
            loaded = load_checkpoint_for_generation(path)
        except Exception as error:
            raise HandoffValidationError(
                f"input checkpoint verification failed: {path}: {error}"
            ) from error
        if loaded.physical_identity.get("sha256") != physical.get(
            "sha256"
        ) or loaded.physical_identity.get("size_bytes") != physical.get("size_bytes"):
            raise HandoffValidationError(f"input checkpoint changed during verification: {path}")
    elif outcome == "succeeded":
        raise HandoffValidationError("successful attempt has an unverified input checkpoint")
    else:
        error = checkpoint_check.get("error")
        if not isinstance(error, str) or not error:
            raise HandoffValidationError(
                "unverified input-checkpoint evidence requires a verification error"
            )


def _validate_result_identity(
    result: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    attempt: Attempt,
) -> None:
    actual = _exact(result, _RESULT_FILE, "result.json")
    if actual["schema_version"] != 1:
        raise HandoffValidationError("result schema_version must be 1")
    if actual["run_id"] != attempt.run_id or actual["run_id"] != state.get("run_id"):
        raise HandoffValidationError("result run ID differs from attempt state")
    if actual["attempt_id"] != attempt.attempt_id or actual["attempt_id"] != state.get(
        "attempt_id"
    ):
        raise HandoffValidationError("result attempt ID differs from attempt state")
    if actual["action"] != state.get("action"):
        raise HandoffValidationError("result action differs from attempt state")
    if actual["outcome"] != state.get("outcome"):
        raise HandoffValidationError("result outcome differs from attempt state")


def _run_manifest_evidence(
    attempt: Attempt,
    *,
    root_dir: Path,
    action: str,
    outcome: str,
) -> dict[str, Any]:
    manifest_path = attempt.path / "work" / "run_manifest.json"
    required = outcome == "succeeded" and action in {"smoke", "train", "resume"}
    if not manifest_path.is_file():
        if required:
            raise HandoffValidationError("successful training is missing work/run_manifest.json")
        return {"status": "unavailable", "reason": "child did not commit a run manifest"}
    try:
        payload = verify_run_manifest(attempt.path / "work", root_dir=root_dir)
    except Exception as error:
        raise HandoffValidationError(f"run manifest verification failed: {error}") from error
    return {
        "status": "verified",
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "experiment_id": payload.get("experiment_id"),
        "git": payload.get("git"),
        "lock": payload.get("lock"),
        "hardware_software": payload.get("hardware_software"),
    }


def _wandb_evidence(
    attempt: Attempt,
    *,
    preflight: Mapping[str, Any],
    action: str,
    outcome: str,
) -> dict[str, Any]:
    candidates = (
        attempt.path / "work" / "checkpoints" / "wandb_events.jsonl",
        attempt.path / "work" / "wandb_events.jsonl",
    )
    records: list[dict[str, Any]] = []
    evidence_files: list[dict[str, Any]] = []
    for path in candidates:
        if not path.is_file():
            continue
        evidence_files.append(
            {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        )
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise HandoffValidationError(f"cannot read W&B evidence: {path}") from error
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise HandoffValidationError(
                    f"invalid W&B evidence at {path}:{line_number}"
                ) from error
            if not isinstance(value, dict) or value.get("schema_version") != 1:
                raise HandoffValidationError(f"invalid W&B evidence record at {path}:{line_number}")
            records.append(value)

    checks = _mapping(preflight.get("checks"), "preflight checks")
    configured = _mapping(checks.get("wandb"), "preflight W&B check")
    mode = str(configured.get("mode", "disabled"))
    executed = action not in {"config-check", "preflight"}
    if outcome == "succeeded" and executed and not records:
        raise HandoffValidationError("successful child is missing W&B outcome evidence")
    initializations = []
    artifacts = []
    outcomes = []
    for record in records:
        action_name = record.get("action")
        record_outcome = record.get("outcome")
        recorded_at = record.get("recorded_at_utc")
        if not isinstance(recorded_at, str) or not recorded_at:
            raise HandoffValidationError("W&B evidence record lacks recorded_at_utc")
        if (
            not isinstance(action_name, str)
            or action_name not in _WANDB_ACTION_OUTCOMES
            or not isinstance(record_outcome, str)
            or record_outcome not in _WANDB_ACTION_OUTCOMES[action_name]
        ):
            raise HandoffValidationError(
                f"invalid W&B lifecycle action/outcome: {action_name!r}/{record_outcome!r}"
            )
        record_mode = record.get("mode")
        if record_mode is not None and record_mode != mode:
            raise HandoffValidationError(
                f"W&B evidence mode {record_mode!r} differs from preflight mode {mode!r}"
            )
        outcomes.append({"action": action_name, "outcome": record_outcome})
        if action_name == "init":
            initializations.append(
                {
                    "mode": record.get("mode", mode),
                    "project": record.get("project", configured.get("project")),
                    "entity": record.get("entity", configured.get("entity")),
                    "run_id": record.get("run_id"),
                    "run_url": record.get("run_url"),
                    "outcome": record_outcome,
                }
            )
        if action_name == "artifact":
            artifacts.append(
                {
                    key: record.get(key)
                    for key in (
                        "outcome",
                        "block_reason",
                        "reason",
                        "checkpoint",
                        "model_artifact",
                        "artifact",
                    )
                }
            )
    if outcome == "succeeded" and executed:
        if not initializations:
            raise HandoffValidationError("successful attempt lacks a W&B initialization outcome")
        initialization_outcomes = {item.get("outcome") for item in initializations}
        if mode == "disabled":
            if initialization_outcomes != {"disabled"}:
                raise HandoffValidationError(
                    "disabled W&B mode requires a disabled initialization outcome"
                )
        elif mode in {"offline", "online"}:
            if "disabled" in initialization_outcomes or not initialization_outcomes.issubset(
                {"succeeded", "failed"}
            ):
                raise HandoffValidationError(
                    f"{mode} W&B mode has an inconsistent initialization outcome"
                )
        else:
            raise HandoffValidationError(f"unsupported preflight W&B mode: {mode}")
    if mode == "online" and outcome == "succeeded":
        succeeded = [item for item in initializations if item.get("outcome") == "succeeded"]
        if any(not item.get("run_id") or not item.get("run_url") for item in succeeded):
            raise HandoffValidationError("successful online W&B initialization lacks run ID/URL")
    return {
        "status": "recorded" if records else "not_executed",
        "configured": dict(configured),
        "evidence_files": evidence_files,
        "event_count": len(records),
        "action_outcomes": outcomes,
        "initializations": initializations,
        "artifacts": artifacts,
    }


def _runtime_evidence(
    attempt: Attempt,
    *,
    state: Mapping[str, Any],
    result: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    action = str(state.get("action", ""))
    outcome = str(result.get("outcome", ""))
    actual: Any = None
    source: str | None = None
    if run_manifest.get("status") == "verified":
        actual = run_manifest.get("hardware_software")
        source = str(run_manifest.get("path"))
    elif action in {"eval", "benchmark"}:
        files = _mapping(result.get("outputs"), "result outputs").get("files", [])
        if files:
            output = load_json(Path(str(_mapping(files[0], "result output")["path"])))
            if action == "eval":
                actual = _mapping(output.get("evaluator_run"), "evaluator run").get("environment")
            else:
                evaluation = _mapping(output.get("evaluation_identity"), "evaluation identity")
                evaluator = _mapping(evaluation.get("evaluator"), "benchmark evaluator")
                actual = evaluator.get("environment")
            source = str(files[0]["path"])
    if outcome == "succeeded" and action not in {"config-check", "preflight"}:
        if not isinstance(actual, Mapping):
            raise HandoffValidationError("successful execution lacks actual runtime identity")
    container: dict[str, Any] | None = None
    container_path = attempt.path / "container.json"
    if container_path.is_file():
        record = load_json(container_path)
        container = {
            key: record.get(key)
            for key in (
                "id",
                "name",
                "image_id",
                "labels",
                "terminal_state",
                "removed",
                "mounts",
            )
        }
    return {
        "status": "recorded" if isinstance(actual, Mapping) else "not_executed",
        "source": source,
        "hardware_software": actual,
        "container": container,
    }


def _attempt_owned_path(value: Any, attempt: Attempt, label: str) -> Path:
    path = Path(str(value)).resolve()
    if path != attempt.path.resolve() and attempt.path.resolve() not in path.parents:
        raise HandoffValidationError(f"{label} is outside the attempt: {path}")
    return path


def _measurement_evidence_path(attempt: Attempt, *, outcome: str) -> Path | None:
    """Resolve and require attempt-owned trainer timing evidence when enabled."""

    state = attempt.state()
    if state.get("action") not in {"smoke", "train", "resume"}:
        return None
    from omegaconf import OmegaConf

    config_path = attempt.path / "resolved_config.yaml"
    if not config_path.is_file():
        if outcome == "succeeded":
            raise HandoffValidationError("successful training is missing resolved_config.yaml")
        return None
    cfg = OmegaConf.load(config_path)
    measurement = cfg.get("measurement", {}) or {}
    if measurement.get("enabled") is not True:
        return None
    artifacts = cfg.get("artifacts", {}) or {}
    checkpoint_dir = Path(str(artifacts.get("checkpoints_dir", "checkpoints")))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = attempt.path / "work" / checkpoint_dir
    configured = measurement.get("output_path")
    if configured:
        path = Path(str(configured))
        if not path.is_absolute():
            path = checkpoint_dir.parent / path
    else:
        path = checkpoint_dir.parent / "measurement.json"
    path = _attempt_owned_path(path, attempt, "measurement evidence")
    if path.is_file():
        return path
    if outcome == "succeeded":
        raise HandoffValidationError(f"measurement evidence file is missing: {path}")
    return None


def _verify_file_identity(path: Path, record: Mapping[str, Any], label: str) -> None:
    if not path.is_file():
        raise HandoffValidationError(f"{label} is missing")
    size = record.get("size_bytes")
    digest = record.get("sha256")
    if not isinstance(size, int) or size < 0 or path.stat().st_size != size:
        raise HandoffValidationError(f"{label} size changed")
    _sha256(digest, f"{label} SHA-256")
    if sha256_file(path) != digest:
        raise HandoffValidationError(f"{label} hash changed")


def _profile_from_config(path: Path) -> str:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    return str(cfg.profile.name)


def _conclusion(
    result: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
    attempt: Attempt,
) -> dict[str, str]:
    diagnosis = _mapping(result.get("diagnosis"), "result diagnosis")
    return {
        "condition_result": (
            "pending_evidence_review" if result.get("outcome") == "succeeded" else "not_evaluated"
        ),
        "evidence_backed_summary": str(diagnosis["summary"]),
        "uncertainty": (
            "target-hardware behavior is not claimed by an offline/host fixture"
            if state.get("device") == "cpu"
            else "only the recorded bounded attempt is supported"
        ),
        "next_step": (
            "review this exact handoff"
            if result.get("outcome") == "succeeded"
            else f"retry in a new sibling attempt linked to {attempt.attempt_id}"
        ),
    }


def _markdown(payload: Mapping[str, Any], json_path: Path) -> str:
    question = payload["predeclared_question"]
    launch = payload["launch_identity"]
    results = payload["results"]
    conclusion = payload["conclusion"]
    return f"""# {payload["ticket"]} — {launch["run_id"]} / {launch["attempt_id"]}

Generated from the validated machine record `{json_path}` (SHA-256 `{sha256_file(json_path)}`).

## Predeclared question and decision rule

- Hypothesis: {question["hypothesis"]}
- Expected result: {question["expected_result"]}
- Success condition: {question["success_condition"]}
- Failure condition: {question["failure_condition"]}
- Stop condition: {question["stop_condition"]}
- Baseline: `{json.dumps(question["baseline"], ensure_ascii=False, sort_keys=True)}`

## Planned budget

```json
{json.dumps(payload["planned_budget"], ensure_ascii=False, indent=2, sort_keys=True)}
```

## Attempt — {results["outcome"]}

### Launch identity

```json
{json.dumps(launch, ensure_ascii=False, indent=2, sort_keys=True)}
```

### Scientific identity

```json
{json.dumps(payload["scientific_identity"], ensure_ascii=False, indent=2, sort_keys=True)}
```

### Counters, evidence, and integrity

```json
{json.dumps({"results": results, "integrity": payload["integrity"]}, ensure_ascii=False, indent=2, sort_keys=True)}
```

### Attempt interpretation

- Result against conditions: {conclusion["condition_result"]}
- Evidence-backed summary: {conclusion["evidence_backed_summary"]}
- Remaining uncertainty: {conclusion["uncertainty"]}

## Conclusion

- Exactly one next step: {conclusion["next_step"]}
"""


def _exact(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    mapping = _mapping(value, label)
    if set(mapping) != keys:
        raise HandoffValidationError(f"{label} keys must be exactly {sorted(keys)}")
    return mapping


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HandoffValidationError(f"{label} must be a mapping")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HandoffValidationError(f"{label} must be a non-empty string")
    return value


def _sha256(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise HandoffValidationError(f"{label} must be a lowercase SHA-256")
    return text
