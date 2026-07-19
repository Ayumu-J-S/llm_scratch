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
_INTEGRITY = {
    "git",
    "resolved_config_sha256",
    "declaration_sha256",
    "result_sha256",
    "state_sha256",
    "retry_binding",
    "evidence_files",
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
            "wandb": preflight["checks"]["wandb"],
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
        "conclusion": {
            "condition_result": (
                "supported" if result["outcome"] == "succeeded" else "not_supported"
            ),
            "evidence_backed_summary": result["diagnosis"]["summary"],
            "uncertainty": (
                "target-hardware behavior is not claimed by an offline/host fixture"
                if state["device"] == "cpu"
                else "only the recorded bounded attempt is supported"
            ),
            "next_step": (
                "review this exact handoff"
                if result["outcome"] == "succeeded"
                else f"retry in a new sibling attempt linked to {attempt.attempt_id}"
            ),
        },
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
            continue
        _nonempty(question[key], f"predeclared_question.{key}")
    _mapping(top["planned_budget"], "planned_budget")
    launch = _mapping(top["launch_identity"], "launch_identity")
    for key in ("run_id", "attempt_id", "action", "executor", "device", "attempt_path"):
        _nonempty(launch.get(key), f"launch_identity.{key}")
    _mapping(launch.get("exact_command"), "launch_identity.exact_command")
    scientific = _mapping(top["scientific_identity"], "scientific_identity")
    for key in ("profile", "parameter_count", "tokenizer_fingerprint", "data_manifests", "wandb"):
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
    for optional in ("stdout.log", "stderr.log", "pid.json", "container.json"):
        path = attempt.path / optional
        if path.is_file():
            paths.append(path)
    paths.extend(sorted(attempt.events_dir.glob("*.json")))
    records = []
    for path in paths:
        if not path.is_file():
            raise HandoffValidationError(f"required evidence file is missing: {path}")
        records.append(
            {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        )
    return records


def _profile_from_config(path: Path) -> str:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    return str(cfg.profile.name)


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
