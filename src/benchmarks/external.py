"""Isolated aggregate-only records for optional external comparisons.

External weights and outputs are never loaded by the repository checkpoint
runner.  A comparison record can only retain aggregate scores produced under
the same published protocol plus the disclosures needed to interpret them.
"""

from __future__ import annotations

import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmarks.runner import _write_json_atomic
from benchmarks.suite import PROTOCOL_MINIMUM_CONTEXT_LENGTH, canonical_external_dev_identity


class ExternalComparisonError(ValueError):
    """An external result is not aggregate-only or lacks required disclosure."""


ROOT_DIR = Path(__file__).resolve().parents[2]
EXTERNAL_COMPARISON_ROOT = ROOT_DIR / "outputs/external-comparisons"


_SUBJECT_FIELDS = {
    "name",
    "parameter_count",
    "training_compute",
    "tokenizer",
    "context_length",
    "data_access",
    "protocol_context_preflight",
}
_CONTEXT_PREFLIGHT_FIELDS = {
    "protocol_sha256",
    "passed",
    "no_truncation",
    "required_context_length",
    "task_required_context_lengths",
}
_TASK_FIELDS = {"primary_metric", "value", "correct", "total"}
_FORBIDDEN_KEYS = {
    "prompt",
    "prompts",
    "completion",
    "completions",
    "output",
    "outputs",
    "text",
    "token_ids",
    "logits",
    "weights",
}


def write_external_comparison(
    payload: Mapping[str, Any],
    *,
    output_path: str | Path,
) -> Path:
    """Write a separate external-comparison record after strict validation."""

    if set(payload) != {"subject", "tasks"}:
        raise ExternalComparisonError("external comparison fields differ from the v1 schema")
    suite_identity = canonical_external_dev_identity()
    subject = _mapping(payload["subject"], "subject")
    if set(subject) != _SUBJECT_FIELDS:
        raise ExternalComparisonError("external subject disclosure is incomplete")
    if not isinstance(subject["name"], str) or not subject["name"]:
        raise ExternalComparisonError("external subject name must be non-empty")
    if (
        isinstance(subject["parameter_count"], bool)
        or not isinstance(subject["parameter_count"], int)
        or subject["parameter_count"] < 1
    ):
        raise ExternalComparisonError("external parameter_count must be positive")
    if (
        isinstance(subject["context_length"], bool)
        or not isinstance(subject["context_length"], int)
        or subject["context_length"] < PROTOCOL_MINIMUM_CONTEXT_LENGTH
    ):
        raise ExternalComparisonError(
            "external context_length must satisfy the fixed protocol minimum of "
            f"{PROTOCOL_MINIMUM_CONTEXT_LENGTH} tokens"
        )
    for field in ("training_compute", "tokenizer", "data_access"):
        if not isinstance(subject[field], str) or not subject[field]:
            raise ExternalComparisonError(f"external {field} disclosure must be non-empty")
    _validate_context_preflight(subject, suite_identity=suite_identity)
    tasks = _mapping(payload["tasks"], "tasks")
    if set(tasks) != {"jcommonsenseqa", "gsm8k"}:
        raise ExternalComparisonError("external comparison must contain both suite tasks")
    for name, result_value in tasks.items():
        result = _mapping(result_value, f"tasks.{name}")
        if set(result) != _TASK_FIELDS:
            raise ExternalComparisonError(f"external task {name} must be aggregate-only")
        expected_metric = (
            "length_normalized_accuracy" if name == "jcommonsenseqa" else "exact_match"
        )
        if result["primary_metric"] != expected_metric:
            raise ExternalComparisonError(
                f"external task {name} primary_metric must be {expected_metric}"
            )
        total = result["total"]
        correct = result["correct"]
        value = result["value"]
        if (
            isinstance(total, bool)
            or not isinstance(total, int)
            or total < 1
            or isinstance(correct, bool)
            or not isinstance(correct, int)
            or not 0 <= correct <= total
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ExternalComparisonError(f"external task {name} metrics are invalid")
        expected_total = suite_identity["tasks"][name]["selected_examples"]
        if total != expected_total:
            raise ExternalComparisonError(
                f"external task {name} total must match the pinned development partition "
                f"({expected_total})"
            )
        if abs(float(value) - correct / total) > 1e-12:
            raise ExternalComparisonError(f"external task {name} value differs from correct/total")
    _reject_forbidden_keys(payload)
    output = _isolated_output_path(output_path)
    record = {
        "schema_version": 1,
        "kind": "external_baseline_comparison",
        "isolation": {
            "repository_checkpoint_runner": False,
            "eligible_for_training_data_or_targets": False,
            "contains_model_outputs": False,
        },
        "suite": suite_identity,
        **dict(payload),
    }
    _write_json_atomic(output, record)
    return output


def _validate_context_preflight(
    subject: Mapping[str, Any],
    *,
    suite_identity: Mapping[str, Any],
) -> None:
    preflight = _mapping(subject["protocol_context_preflight"], "protocol_context_preflight")
    if set(preflight) != _CONTEXT_PREFLIGHT_FIELDS:
        raise ExternalComparisonError("external protocol context preflight is incomplete")
    if preflight["protocol_sha256"] != suite_identity["protocol_sha256"]:
        raise ExternalComparisonError(
            "external context preflight must bind the compiled benchmark protocol"
        )
    if preflight["passed"] is not True or preflight["no_truncation"] is not True:
        raise ExternalComparisonError(
            "external context preflight must attest complete no-truncation execution"
        )
    task_requirements = _mapping(
        preflight["task_required_context_lengths"],
        "protocol_context_preflight.task_required_context_lengths",
    )
    if set(task_requirements) != {"jcommonsenseqa", "gsm8k"}:
        raise ExternalComparisonError(
            "external context preflight must disclose both task requirements"
        )
    for name, value in task_requirements.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ExternalComparisonError(
                f"external context preflight requirement for {name} must be positive"
            )
    if task_requirements["gsm8k"] < PROTOCOL_MINIMUM_CONTEXT_LENGTH:
        raise ExternalComparisonError(
            "external GSM8K context requirement cannot fit the fixed generation cap"
        )
    required = preflight["required_context_length"]
    if (
        isinstance(required, bool)
        or not isinstance(required, int)
        or required != max(task_requirements.values())
    ):
        raise ExternalComparisonError(
            "external required_context_length must equal the maximum task requirement"
        )
    if subject["context_length"] < required:
        raise ExternalComparisonError(
            "external subject context_length is below its protocol preflight requirement"
        )


def _isolated_output_path(output_path: str | Path) -> Path:
    """Resolve one JSON path inside the dedicated, non-checkpoint comparison tree."""

    configured_root = EXTERNAL_COMPARISON_ROOT.absolute()
    resolved_root = configured_root.resolve()
    if resolved_root != configured_root:
        raise ExternalComparisonError(
            "external comparison root must not traverse a symlink or checkpoint namespace"
        )
    requested = Path(output_path)
    candidate = requested if requested.is_absolute() else configured_root / requested
    output = candidate.resolve()
    if output == resolved_root or resolved_root not in output.parents:
        raise ExternalComparisonError(
            "external comparison output must be inside outputs/external-comparisons"
        )
    if output.suffix != ".json":
        raise ExternalComparisonError("external comparison output must use a .json suffix")
    protected_names = {"artifact", "artifacts", "checkpoint", "checkpoints"}
    relative_parts = output.relative_to(resolved_root).parts[:-1]
    if any(part.casefold() in protected_names for part in relative_parts):
        raise ExternalComparisonError(
            "external comparison output must not enter an artifact/checkpoint namespace"
        )
    try:
        existing = candidate.lstat()
    except FileNotFoundError:
        return output
    except OSError as error:
        raise ExternalComparisonError("external comparison output cannot be inspected") from error
    if stat.S_ISLNK(existing.st_mode):
        raise ExternalComparisonError("external comparison output must not be a symlink")
    if not stat.S_ISREG(existing.st_mode):
        raise ExternalComparisonError("external comparison output must be a regular file")
    if existing.st_nlink != 1:
        raise ExternalComparisonError(
            "external comparison output must not share an inode with a checkpoint or other file"
        )
    return output


def _reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = sorted(str(key) for key in value if str(key).lower() in _FORBIDDEN_KEYS)
        if forbidden:
            raise ExternalComparisonError(
                f"external comparison contains forbidden raw/model field(s): {forbidden}"
            )
        for item in value.values():
            _reject_forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            _reject_forbidden_keys(item)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExternalComparisonError(f"external {label} must be a mapping")
    return value
