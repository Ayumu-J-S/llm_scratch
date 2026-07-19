from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import operate
from operations import runner
from operations.artifacts import Attempt, sha256_file
from operations import preflight as preflight_module
from operations.handoff import HandoffValidationError, generate_handoff, validate_handoff


def _prefix(tmp_path, action, attempt_id):
    return [
        action,
        "--run-root",
        str(tmp_path),
        "--run-id",
        "OPS-001-fixture",
        "--attempt-id",
        attempt_id,
        "--executor",
        "host",
        "--device",
        "cpu",
    ]


def _small_smoke_overrides():
    return [
        "training.epochs=1",
        "training.batch_size=1",
        "training.max_steps=1",
        "training.validation_every_n_steps=null",
        "training.checkpoint_every_n_steps=null",
        "training.milestone_every_n_steps=null",
        "model.embed_size=16",
        "model.num_heads=4",
        "model.num_layers=1",
    ]


def _experiment_record(tmp_path: Path) -> tuple[Path, dict]:
    payload = {
        "schema_version": 1,
        "ticket": "RUN-001",
        "predeclared_question": {
            "hypothesis": "The one-step fixture preserves finite, verified evidence.",
            "expected_result": "One optimizer step and a verified final checkpoint.",
            "success_condition": "The attempt and its integrity-bound handoff succeed.",
            "failure_condition": "Any preflight, child, numeric, or integrity gate fails.",
            "stop_condition": "Stop before the declared filesystem reserve is crossed.",
            "baseline": {"kind": "untrained_fixture", "optimizer_step": 0},
        },
        "planned_budget": {
            "training": {
                "epochs": 1,
                "max_steps": 1,
                "max_tokens": None,
                "max_time": None,
            },
            "device": "cpu",
        },
    }
    path = tmp_path / "RUN-001.declaration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_offline_smoke_through_validated_handoff(tmp_path):
    experiment_record, declared = _experiment_record(tmp_path)
    command = [
        *_prefix(tmp_path, "smoke", "attempt-0001"),
        "--experiment-record",
        str(experiment_record),
        "--",
        *_small_smoke_overrides(),
    ]

    assert operate.main(command) == 0
    assert (
        operate.main(
            [
                "handoff",
                "--run-root",
                str(tmp_path),
                "--run-id",
                "OPS-001-fixture",
                "--attempt-id",
                "attempt-0001",
            ]
        )
        == 0
    )

    attempt = tmp_path / "OPS-001-fixture" / "attempts" / "attempt-0001"
    result = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
    handoff = json.loads((attempt / "handoff.json").read_text(encoding="utf-8"))
    declaration = json.loads((attempt / "declaration.json").read_text(encoding="utf-8"))
    preflight = json.loads((attempt / "preflight.json").read_text(encoding="utf-8"))

    assert result["outcome"] == "succeeded"
    assert result["counters"]["optimizer_step"] == 1
    assert result["metrics"]["invalid_numeric"] is False
    assert result["lifecycle_errors"] == []
    assert all(record["verified"] for record in result["checkpoint_status"]["files"])
    assert handoff["ticket"] == "RUN-001"
    assert handoff["predeclared_question"] == declared["predeclared_question"]
    assert declaration["source"] == {
        "kind": "explicit_experiment_record",
        "path": str(experiment_record.resolve()),
        "sha256": hashlib.sha256(experiment_record.read_bytes()).hexdigest(),
    }
    assert handoff["results"]["lifecycle_errors"] == []
    assert handoff["conclusion"]["condition_result"] == "pending_evidence_review"
    assert handoff["scientific_identity"]["data_manifests"][0]["split"] == "memorization"
    assert handoff["scientific_identity"]["run_manifest"]["status"] == "verified"
    assert handoff["scientific_identity"]["runtime"]["status"] == "recorded"
    assert handoff["scientific_identity"]["wandb"]["initializations"][0]["outcome"] == "disabled"
    assert preflight["checks"]["storage"]["effective_run_live_floor_bytes"] >= 120_000_000_000
    assert (attempt / "handoff.md").is_file()
    assert (attempt / "stdout.log").is_file()
    assert (attempt / "stderr.log").is_file()

    missing_result = copy.deepcopy(handoff)
    del missing_result["results"]["diagnosis"]
    with pytest.raises(HandoffValidationError, match="results keys"):
        validate_handoff(missing_result)

    missing_integrity = copy.deepcopy(handoff)
    del missing_integrity["integrity"]["result_sha256"]
    with pytest.raises(HandoffValidationError, match="integrity keys"):
        validate_handoff(missing_integrity)

    attempt_record = Attempt(tmp_path, "OPS-001-fixture", "attempt-0001")
    root_dir = Path(__file__).resolve().parents[1]
    tampered_claims = []
    tampered = copy.deepcopy(handoff)
    tampered["ticket"] = "OPS-001"
    tampered_claims.append(tampered)
    tampered = copy.deepcopy(handoff)
    tampered["launch_identity"]["device"] = "cuda"
    tampered_claims.append(tampered)
    for field, value in (
        ("profile", "different-profile"),
        ("parameter_count", 1),
        ("tokenizer_fingerprint", "0" * 64),
        ("data_manifests", []),
        ("checkpoint", {"status": "fabricated"}),
        ("storage_forecast", {}),
    ):
        tampered = copy.deepcopy(handoff)
        tampered["scientific_identity"][field] = value
        tampered_claims.append(tampered)
    tampered = copy.deepcopy(handoff)
    tampered["integrity"]["git"]["sha"] = "0" * 40
    tampered_claims.append(tampered)
    tampered = copy.deepcopy(handoff)
    tampered["conclusion"]["condition_result"] = "not_supported"
    tampered_claims.append(tampered)
    for tampered in tampered_claims:
        with pytest.raises(HandoffValidationError):
            validate_handoff(tampered, attempt=attempt_record, root_dir=root_dir)

    checkpoint = Path(result["checkpoint_status"]["files"][-1]["path"])
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corruption")
    with pytest.raises(HandoffValidationError, match="checkpoint .* (size|hash) changed"):
        validate_handoff(
            handoff,
            attempt=attempt_record,
            root_dir=root_dir,
        )


def test_config_check_prints_login_prompt_after_local_checks(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(preflight_module, "_wandb_credential_visible", lambda _executor: False)
    command = [
        *_prefix(tmp_path, "config-check", "attempt-0001"),
        "--",
        "profile=smoke_overfit",
        "wandb.mode=online",
    ]

    assert operate.main(command) == 0

    captured = capsys.readouterr()
    assert "wandb login" in captured.err
    attempt = tmp_path / "OPS-001-fixture" / "attempts" / "attempt-0001"
    report = json.loads((attempt / "preflight.json").read_text(encoding="utf-8"))
    assert report["local_checks_complete"] is True
    assert report["ready"] is True


def test_handoff_requires_state_bound_result_logs_and_metrics(tmp_path):
    command = [
        *_prefix(tmp_path, "smoke", "attempt-integrity"),
        "--",
        *_small_smoke_overrides(),
        "measurement.enabled=true",
        "measurement.warmup_optimizer_steps=0",
    ]
    assert operate.main(command) == 0

    attempt = Attempt(tmp_path, "OPS-001-fixture", "attempt-integrity")
    result_path = attempt.path / "result.json"
    result_bytes = result_path.read_bytes()
    result = json.loads(result_bytes)
    for field, value, match in (
        ("schema_version", 2, "result schema_version"),
        ("run_id", "different-run", "result run ID differs"),
        ("attempt_id", "different-attempt", "result attempt ID differs"),
        ("action", "config-check", "result action differs"),
        ("outcome", "failed", "result outcome differs"),
    ):
        contradictory = copy.deepcopy(result)
        contradictory[field] = value
        if field == "action":
            contradictory["checkpoint_status"] = {
                "directory": None,
                "files": [],
                "last_verified": None,
            }
        result_path.write_text(json.dumps(contradictory), encoding="utf-8")
        with pytest.raises(HandoffValidationError, match=match):
            generate_handoff(attempt, root_dir=Path(__file__).resolve().parents[1])
        result_path.write_bytes(result_bytes)

    for evidence_path, match in (
        (attempt.path / "stdout.log", "execution log is missing"),
        (attempt.path / "stderr.log", "execution log is missing"),
        (
            Path(str(result["metrics"]["path"])),
            "metrics evidence file is missing",
        ),
    ):
        evidence_bytes = evidence_path.read_bytes()
        evidence_path.unlink()
        with pytest.raises(HandoffValidationError, match=match):
            generate_handoff(attempt, root_dir=Path(__file__).resolve().parents[1])
        evidence_path.write_bytes(evidence_bytes)

    metrics_path = Path(str(result["metrics"]["path"]))
    metrics_bytes = metrics_path.read_bytes()
    metrics_path.write_bytes(metrics_bytes + b"{}\n")
    with pytest.raises(HandoffValidationError, match="metrics evidence hash changed"):
        generate_handoff(attempt, root_dir=Path(__file__).resolve().parents[1])
    metrics_path.write_bytes(metrics_bytes)

    measurement_path = attempt.path / "work" / "measurement.json"
    measurement_bytes = measurement_path.read_bytes()
    measurement_path.unlink()
    with pytest.raises(HandoffValidationError, match="measurement evidence file is missing"):
        generate_handoff(attempt, root_dir=Path(__file__).resolve().parents[1])
    measurement_path.write_bytes(measurement_bytes)

    generated = generate_handoff(attempt, root_dir=Path(__file__).resolve().parents[1])
    handoff = json.loads(generated.with_name("handoff.json").read_text(encoding="utf-8"))
    measurement_record = next(
        record
        for record in handoff["integrity"]["evidence_files"]
        if record["path"] == str(measurement_path)
    )
    assert measurement_record["sha256"] == sha256_file(measurement_path)

    measurement_path.write_bytes(measurement_bytes + b"\n")
    with pytest.raises(
        HandoffValidationError,
        match="handoff evidence (inventory differs|file changed|hash changed)",
    ):
        validate_handoff(
            handoff,
            attempt=attempt,
            root_dir=Path(__file__).resolve().parents[1],
        )
    measurement_path.write_bytes(measurement_bytes)

    generate_handoff(attempt, root_dir=Path(__file__).resolve().parents[1])


def test_failed_child_and_successful_retry_retain_hash_bound_evidence(tmp_path, monkeypatch):
    monkeypatch.setitem(runner._ENTRYPOINT, "smoke", "src/intentionally_missing_ops_child.py")
    failed_command = [
        *_prefix(tmp_path, "smoke", "attempt-failed"),
        "--",
        *_small_smoke_overrides(),
    ]

    assert operate.main(failed_command) == 1
    assert (
        operate.main(
            [
                "handoff",
                "--run-root",
                str(tmp_path),
                "--run-id",
                "OPS-001-fixture",
                "--attempt-id",
                "attempt-failed",
            ]
        )
        == 0
    )

    failed = tmp_path / "OPS-001-fixture" / "attempts" / "attempt-failed"
    failed_result = json.loads((failed / "result.json").read_text(encoding="utf-8"))
    failed_state_sha256 = sha256_file(failed / "state.json")
    failed_result_sha256 = sha256_file(failed / "result.json")
    assert failed_result["outcome"] == "failed"
    assert failed_result["exit_code"] != 0
    for name in (
        "command.json",
        "preflight.json",
        "result.json",
        "diagnosis.json",
        "stdout.log",
        "stderr.log",
        "handoff.json",
    ):
        assert (failed / name).is_file()

    monkeypatch.setitem(runner._ENTRYPOINT, "smoke", "src/train.py")
    retry_command = [
        *_prefix(tmp_path, "smoke", "attempt-retry"),
        "--retry-from",
        "attempt-failed",
        "--",
        *_small_smoke_overrides(),
    ]
    assert operate.main(retry_command) == 0
    assert (
        operate.main(
            [
                "handoff",
                "--run-root",
                str(tmp_path),
                "--run-id",
                "OPS-001-fixture",
                "--attempt-id",
                "attempt-retry",
            ]
        )
        == 0
    )

    retry = tmp_path / "OPS-001-fixture" / "attempts" / "attempt-retry"
    retry_state = json.loads((retry / "state.json").read_text(encoding="utf-8"))
    retry_result = json.loads((retry / "result.json").read_text(encoding="utf-8"))
    retry_handoff = json.loads((retry / "handoff.json").read_text(encoding="utf-8"))
    binding = retry_state["retry_from"]
    assert retry_result["outcome"] == "succeeded"
    assert binding == {
        "attempt_id": "attempt-failed",
        "state_sha256": failed_state_sha256,
        "result_sha256": failed_result_sha256,
    }
    assert retry_handoff["integrity"]["retry_binding"] == binding
    assert retry_handoff["predeclared_question"]["baseline"] == binding
    assert sha256_file(failed / "state.json") == failed_state_sha256
    assert sha256_file(failed / "result.json") == failed_result_sha256


def test_hydra_commands_require_literal_override_separator(tmp_path):
    with pytest.raises(Exception, match="literal '--'"):
        operate.main(_prefix(tmp_path, "smoke", "attempt-0001"))


def test_status_and_handoff_reject_identifier_path_traversal(tmp_path):
    with pytest.raises(Exception, match="path separators"):
        operate.main(
            [
                "status",
                "--run-root",
                str(tmp_path),
                "--run-id",
                "../escape",
                "--attempt-id",
                "attempt-0001",
            ]
        )
