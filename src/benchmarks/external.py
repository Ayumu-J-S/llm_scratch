"""Isolated aggregate-only records for optional external comparisons.

External weights and outputs are never loaded by the repository checkpoint
runner.  A comparison record can only retain aggregate scores produced under
the same published protocol plus the disclosures needed to interpret them.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmarks.runner import _write_json_atomic
from benchmarks.suite import canonical_external_dev_identity


class ExternalComparisonError(ValueError):
    """An external result is not aggregate-only or lacks required disclosure."""


_SUBJECT_FIELDS = {
    "name",
    "parameter_count",
    "training_compute",
    "tokenizer",
    "context_length",
    "data_access",
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
        or subject["context_length"] < 1
    ):
        raise ExternalComparisonError("external context_length must be positive")
    for field in ("training_compute", "tokenizer", "data_access"):
        if not isinstance(subject[field], str) or not subject[field]:
            raise ExternalComparisonError(f"external {field} disclosure must be non-empty")
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
    output = Path(output_path).resolve()
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
