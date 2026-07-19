from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from operations import lifecycle, runner
from operations.artifacts import (
    AttemptError,
    atomic_write_json,
    create_attempt,
    process_identity,
    sha256_file,
)
from operations.preflight import PreflightError
from operations.handoff import (
    HandoffValidationError,
    _wandb_evidence,
    _validate_attempt_artifacts,
    generate_handoff,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def _attempt(tmp_path):
    attempt = create_attempt(
        run_root=tmp_path,
        run_id="run-001",
        attempt_id="attempt-0001",
        action="smoke",
        executor="host",
        device="cpu",
        retry_from=None,
    )
    attempt.transition("preflighting")
    attempt.transition("ready")
    attempt.transition("running")
    return attempt


def _container_preflight(root: Path, image_id: str, *, wandb_mode: str = "disabled"):
    return {
        "checks": {
            "device": {"image_id": image_id},
            "wandb": {"mode": wandb_mode},
            "container_mounts": {
                "status": "passed",
                "required": True,
                "mounts": [
                    {
                        "source": str(root.resolve()),
                        "destination": str(root.resolve()),
                        "read_only": False,
                        "kind": "directory",
                        "purposes": ["repository"],
                    }
                ],
            },
        }
    }


def test_operational_hydra_authority_rejects_parent_mapping_replacement(tmp_path):
    args = SimpleNamespace(action="train", device="cuda")
    attempt = create_attempt(
        run_root=tmp_path,
        run_id="run-001",
        attempt_id="attempt-0001",
        action="train",
        executor="host",
        device="cuda",
        retry_from=None,
    )

    with pytest.raises(AttemptError, match="belongs to llm-scratch-ops"):
        runner._operational_overrides(
            args,
            ["runtime={device:cpu}"],
            attempt=attempt,
        )
    with pytest.raises(AttemptError, match="belongs to llm-scratch-ops"):
        runner._operational_overrides(
            args,
            ["measurement.output_path=/external/measurement.json"],
            attempt=attempt,
        )
    overrides = runner._operational_overrides(args, [], attempt=attempt)
    assert "measurement.output_path=measurement.json" in overrides


def test_dangerous_run_root_is_rejected_before_attempt_creation(tmp_path, monkeypatch):
    args = SimpleNamespace(
        action="smoke",
        retry_from=None,
        run_root=tmp_path,
        run_id="run-001",
        attempt_id="attempt-0001",
        executor="host",
        device="cpu",
    )
    monkeypatch.setattr(
        runner,
        "reject_writable_git_overlap",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PreflightError("run root overlaps Git metadata")
        ),
    )
    monkeypatch.setattr(
        runner,
        "create_attempt",
        lambda **_kwargs: pytest.fail("attempt must not be created before path validation"),
    )

    with pytest.raises(PreflightError, match="overlaps Git metadata"):
        runner.dispatch(args, [], root_dir=ROOT_DIR)


def test_train_selected_profile_is_ops_owned_and_composes(tmp_path):
    args = SimpleNamespace(action="train", device="cpu", profile="gate_overfit")
    attempt = create_attempt(
        run_root=tmp_path,
        run_id="run-001",
        attempt_id="attempt-0001",
        action="train",
        executor="host",
        device="cpu",
        retry_from=None,
    )
    selected = runner._selected_profile(args, root_dir=ROOT_DIR)
    overrides = runner._operational_overrides(
        args,
        [],
        attempt=attempt,
        selected_profile=selected,
    )

    assert selected == "gate_overfit"
    assert runner._compose(ROOT_DIR, overrides).profile.name == "gate_overfit"
    with pytest.raises(AttemptError, match="belongs to llm-scratch-ops"):
        runner._operational_overrides(
            args,
            ["profile=pretrain_streaming"],
            attempt=attempt,
            selected_profile=selected,
        )


def test_package_qualified_profile_override_cannot_bypass_ops_authority(tmp_path):
    args = SimpleNamespace(action="train", device="cpu", profile="pretrain_streaming")
    attempt = create_attempt(
        run_root=tmp_path,
        run_id="run-001",
        attempt_id="attempt-0001",
        action="train",
        executor="host",
        device="cpu",
        retry_from=None,
    )

    with pytest.raises(AttemptError, match="package-qualified"):
        runner._operational_overrides(
            args,
            ["+profile@_global_=smoke_overfit"],
            attempt=attempt,
            selected_profile="pretrain_streaming",
        )


def test_resume_derives_and_composes_checkpoint_owned_profile(tmp_path, monkeypatch):
    args = SimpleNamespace(
        action="resume",
        device="cpu",
        checkpoint=tmp_path / "checkpoint.pt",
    )
    attempt = create_attempt(
        run_root=tmp_path,
        run_id="run-001",
        attempt_id="attempt-0001",
        action="resume",
        executor="host",
        device="cpu",
        retry_from=None,
    )
    checkpoint = SimpleNamespace(
        payload={"state": {"resolved_config": {"profile": {"name": "pretrain_streaming"}}}}
    )
    monkeypatch.setattr(runner, "load_checkpoint_for_generation", lambda _path: checkpoint)

    selected = runner._selected_profile(args, root_dir=ROOT_DIR)
    overrides = runner._operational_overrides(
        args,
        [],
        attempt=attempt,
        selected_profile=selected,
    )

    assert selected == "pretrain_streaming"
    assert runner._compose(ROOT_DIR, overrides).profile.name == "pretrain_streaming"


@pytest.mark.parametrize("action", ["preflight", "train", "resume"])
def test_real_pretraining_requires_explicit_experiment_record(tmp_path, action):
    attempt = create_attempt(
        run_root=tmp_path,
        run_id="RUN-001",
        attempt_id=f"attempt-{action}",
        action=action,
        executor="host",
        device="cpu",
        retry_from=None,
    )
    cfg = runner._compose(ROOT_DIR, ["profile=pretrain_streaming"])

    with pytest.raises(AttemptError, match="requires --experiment-record"):
        runner._attempt_declaration(cfg, attempt=attempt, experiment_record=None)


def test_explicit_budget_must_equal_resolved_training_limits_and_device(tmp_path):
    attempt = create_attempt(
        run_root=tmp_path,
        run_id="RUN-001",
        attempt_id="attempt-train",
        action="train",
        executor="host",
        device="cpu",
        retry_from=None,
    )
    cfg = runner._compose(ROOT_DIR, ["profile=pretrain_streaming", "training.max_steps=1"])
    declaration = {
        "schema_version": 1,
        "ticket": "RUN-001",
        "predeclared_question": {
            "hypothesis": "fixture hypothesis",
            "expected_result": "fixture result",
            "success_condition": "fixture success",
            "failure_condition": "fixture failure",
            "stop_condition": "fixture stop",
            "baseline": "fixture baseline",
        },
        "planned_budget": {
            "training": {
                "epochs": 1,
                "max_steps": 2,
                "max_tokens": None,
                "max_time": None,
            },
            "device": "cpu",
        },
    }
    path = tmp_path / "declaration.json"
    path.write_text(json.dumps(declaration), encoding="utf-8")

    with pytest.raises(AttemptError, match="must exactly match"):
        runner._attempt_declaration(cfg, attempt=attempt, experiment_record=path)


def test_corrupt_resume_checkpoint_retains_failed_command_lineage(tmp_path, monkeypatch):
    parent = create_attempt(
        run_root=tmp_path,
        run_id="run-001",
        attempt_id="attempt-failed",
        action="train",
        executor="host",
        device="cpu",
        retry_from=None,
    )
    parent.transition("preflighting")
    atomic_write_json(parent.path / "result.json", {"outcome": "failed"})
    parent.transition("failed", outcome="failed")
    checkpoint = tmp_path / "corrupt.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    args = SimpleNamespace(
        action="resume",
        run_root=tmp_path,
        run_id="run-001",
        attempt_id="attempt-resume",
        executor="host",
        device="cpu",
        image=None,
        retry_from="attempt-failed",
        checkpoint=checkpoint,
    )
    monkeypatch.setattr(
        runner.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("child launched")),
    )
    monkeypatch.setattr(runner, "reject_writable_git_overlap", lambda *_args, **_kwargs: None)

    assert runner.dispatch(args, [], root_dir=ROOT_DIR) == 1

    attempt = tmp_path / "run-001" / "attempts" / "attempt-resume"
    command = json.loads((attempt / "command.json").read_text(encoding="utf-8"))
    state = json.loads((attempt / "state.json").read_text(encoding="utf-8"))
    diagnosis = json.loads((attempt / "diagnosis.json").read_text(encoding="utf-8"))
    assert command["checkpoint_path"] == str(checkpoint.resolve())
    assert command["requested_hydra_overrides"] == []
    assert state["status"] == "failed"
    assert state["retry_from"]["attempt_id"] == "attempt-failed"
    assert "cannot derive resume profile" in diagnosis["summary"]
    for name in (
        "declaration.json",
        "resolved_config.yaml",
        "preflight.json",
        "plan.json",
    ):
        assert (attempt / name).is_file()

    handoff_path = generate_handoff(
        runner.Attempt(tmp_path, "run-001", "attempt-resume"),
        root_dir=ROOT_DIR,
    )
    handoff = json.loads(handoff_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert handoff["results"]["outcome"] == "failed"
    assert handoff["scientific_identity"]["profile"] == "unresolved_prelaunch"
    checkpoint_evidence = handoff["scientific_identity"]["checkpoint"]
    assert checkpoint_evidence["status"] == "failed"
    assert checkpoint_evidence["physical_identity"]["sha256"]


def test_failed_output_handoff_retains_exact_corrupt_checkpoint_bytes(tmp_path):
    attempt = create_attempt(
        run_root=tmp_path,
        run_id="run-001",
        attempt_id="attempt-failed-checkpoint",
        action="train",
        executor="host",
        device="cpu",
        retry_from=None,
    )
    checkpoint = attempt.path / "work" / "checkpoints" / "corrupt.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"corrupt checkpoint evidence")
    record = {
        "path": str(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": sha256_file(checkpoint),
        "verified": False,
        "verification_error": "checkpoint payload is corrupt",
    }
    result = {
        "action": "train",
        "outcome": "failed",
        "checkpoint_status": {"files": [record]},
        "outputs": {"files": []},
    }

    _validate_attempt_artifacts(result, preflight={"checks": {}}, attempt=attempt)

    checkpoint.write_bytes(b"changed")
    with pytest.raises(HandoffValidationError, match="checkpoint .* (size|hash) changed"):
        _validate_attempt_artifacts(result, preflight={"checks": {}}, attempt=attempt)


def test_invalid_loss_is_detected_even_when_json_parser_accepts_nan(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        json.dumps({"optimizer_step": 1, "target_tokens": 8, "train/loss": 1.0})
        + "\n"
        + '{"optimizer_step": 2, "target_tokens": 16, "train/loss": NaN}\n',
        encoding="utf-8",
    )

    summary = runner._metrics_summary(metrics)

    assert summary["invalid_numeric"] is True
    assert "train/loss" in summary["invalid_numeric_evidence"][0]


def test_nonzero_nonfinite_counter_is_an_invalid_run(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(
        json.dumps(
            {
                "optimizer_step": 1,
                "target_tokens": 8,
                "train/loss": 1.0,
                "optimizer/nonfinite_count": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert runner._metrics_summary(metrics)["invalid_numeric"] is True


def test_disk_watchdog_checks_every_filesystem_and_stops_despite_pid_record_mutation(
    tmp_path, monkeypatch
):
    attempt = _attempt(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    trusted = process_identity(process.pid)
    atomic_write_json(attempt.path / "pid.json", trusted)
    atomic_write_json(
        attempt.path / "pid.json",
        {**trusted, "proc_start_ticks": trusted["proc_start_ticks"] + 1},
    )
    safe = tmp_path / "safe"
    cache = tmp_path / "cache"
    safe.mkdir()
    cache.mkdir()

    def disk_usage(path):
        free = 200_000_000_000 if Path(path) == safe else 99_000_000_000
        return SimpleNamespace(total=300_000_000_000, used=0, free=free)

    monkeypatch.setattr(lifecycle.shutil, "disk_usage", disk_usage)
    watchdog = {
        "triggered": False,
        "filesystems": [
            {
                "device": 1,
                "path": str(safe),
                "effective_live_floor_bytes": 120_000_000_000,
                "minimum_observed_free_bytes": None,
            },
            {
                "device": 2,
                "path": str(cache),
                "effective_live_floor_bytes": 120_000_000_000,
                "minimum_observed_free_bytes": None,
            },
        ],
    }

    with pytest.raises(AttemptError, match="on-disk PID ownership evidence changed"):
        lifecycle.wait_with_disk_watchdog(
            process,
            attempt=attempt,
            watchdog=watchdog,
            container_record=None,
            trusted_ownership=trusted,
        )

    assert process.poll() is not None
    assert watchdog["device"] == 2
    kinds = [json.loads(path.read_text())["kind"] for path in attempt.events_dir.glob("*.json")]
    assert "ownership_evidence_mutated" in kinds


def test_disk_sampling_failure_stops_trusted_child(tmp_path, monkeypatch):
    attempt = _attempt(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    trusted = process_identity(process.pid)
    atomic_write_json(attempt.path / "pid.json", trusted)
    monkeypatch.setattr(
        lifecycle.shutil,
        "disk_usage",
        lambda _path: (_ for _ in ()).throw(OSError("statvfs unavailable")),
    )

    exit_code = lifecycle.wait_with_disk_watchdog(
        process,
        attempt=attempt,
        watchdog={
            "triggered": False,
            "filesystems": [
                {
                    "device": 1,
                    "path": str(tmp_path),
                    "effective_live_floor_bytes": 120_000_000_000,
                    "minimum_observed_free_bytes": None,
                }
            ],
        },
        container_record=None,
        trusted_ownership=trusted,
    )

    assert exit_code != 0
    assert process.poll() is not None


def test_external_signal_request_stops_exact_owned_process_group(tmp_path):
    attempt = _attempt(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    trusted = process_identity(process.pid)
    atomic_write_json(attempt.path / "pid.json", trusted)
    watchdog = {
        "triggered": False,
        "filesystems": [
            {
                "device": tmp_path.stat().st_dev,
                "path": str(tmp_path),
                "effective_live_floor_bytes": 0,
                "minimum_observed_free_bytes": None,
            }
        ],
    }

    exit_code = lifecycle.wait_with_disk_watchdog(
        process,
        attempt=attempt,
        watchdog=watchdog,
        container_record=None,
        trusted_ownership=trusted,
        stop_request={"signal": signal.SIGTERM, "signal_name": "SIGTERM"},
    )

    assert exit_code != 0
    assert process.poll() is not None
    assert watchdog["reason"] == "external_signal"
    assert watchdog["signal_name"] == "SIGTERM"


def test_signal_capture_records_first_request_and_restores_handlers():
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_hup = signal.getsignal(signal.SIGHUP)

    with runner._capture_stop_signals() as request:
        term_handler = signal.getsignal(signal.SIGTERM)
        hup_handler = signal.getsignal(signal.SIGHUP)
        term_handler(signal.SIGTERM, None)
        hup_handler(signal.SIGHUP, None)
        assert request == {"signal": signal.SIGTERM, "signal_name": "SIGTERM"}

    assert signal.getsignal(signal.SIGTERM) == previous_term
    assert signal.getsignal(signal.SIGHUP) == previous_hup


def test_dispatch_captures_preflight_signal_and_terminalizes_attempt(tmp_path, monkeypatch):
    args = SimpleNamespace(
        action="smoke",
        retry_from=None,
        run_root=tmp_path,
        run_id="run-001",
        attempt_id="attempt-signaled",
        executor="host",
        device="cpu",
        image=None,
        checkpoint=None,
        experiment_record=None,
    )

    def signal_during_preflight(*_args, **_kwargs):
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler), "SIGTERM must be captured throughout preflight"
        handler(signal.SIGTERM, None)
        return {}

    monkeypatch.setattr(runner, "run_preflight", signal_during_preflight)
    monkeypatch.setattr(
        runner,
        "_execute_with_stop_request",
        lambda *_args, **_kwargs: pytest.fail("child must not launch after preflight signal"),
    )

    assert runner.dispatch(args, [], root_dir=ROOT_DIR) == 128 + signal.SIGTERM

    attempt = runner.Attempt(tmp_path, "run-001", "attempt-signaled")
    state = attempt.state()
    result = json.loads((attempt.path / "result.json").read_text(encoding="utf-8"))
    assert state["status"] == "stopped"
    assert result["outcome"] == "stopped"
    assert result["watchdog"] == {
        "triggered": True,
        "reason": "external_signal",
        "signal": signal.SIGTERM,
        "signal_name": "SIGTERM",
    }
    signal_events = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in attempt.events_dir.glob("*.json")
        if json.loads(path.read_text(encoding="utf-8"))["kind"] == "external_signal_recorded"
    ]
    assert signal_events[-1]["phase"] == "prelaunch"
    generate_handoff(attempt, root_dir=ROOT_DIR)


def test_container_label_anomaly_still_stops_immutable_created_id(tmp_path, monkeypatch):
    attempt = _attempt(tmp_path)
    calls = []

    class Process:
        pid = 123

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 137

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(
        lifecycle,
        "container_identity",
        lambda _record: {
            "available": True,
            "id_matches": True,
            "image_matches": True,
            "labels_match": False,
        },
    )
    monkeypatch.setattr(lifecycle.subprocess, "run", run)
    record = {
        "id": "a" * 64,
        "image_id": "sha256:" + "b" * 64,
        "labels": {"llm-scratch.ops.attempt_id": attempt.attempt_id},
    }

    errors = lifecycle.stop_owned_process(
        Process(),
        attempt=attempt,
        container_record=record,
        trusted_ownership=None,
    )

    assert errors == ["container ownership labels changed after launch"]
    assert ["docker", "stop", "--time", "10", "a" * 64] in calls


def test_container_inspect_failure_forces_owned_id_removal_and_records_error(tmp_path, monkeypatch):
    attempt = _attempt(tmp_path)
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        if command[:2] == ["docker", "inspect"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="daemon unavailable")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lifecycle.subprocess, "run", run)
    record = {
        "id": "a" * 64,
        "image_id": "sha256:" + "b" * 64,
        "labels": {},
        "removed": False,
    }

    errors = lifecycle.capture_and_remove_container(attempt, record)

    assert errors == ["docker inspect failed during finalization: daemon unavailable"]
    assert ["docker", "rm", "--force", "a" * 64] in calls
    assert json.loads((attempt.path / "container.json").read_text())["removed"] is True


def test_container_post_create_identity_failure_force_removes_exact_id(tmp_path, monkeypatch):
    attempt = _attempt(tmp_path)
    container_id = "a" * 64
    image_id = "sha256:" + "b" * 64
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        if command[:2] == ["docker", "create"]:
            return SimpleNamespace(returncode=0, stdout=container_id, stderr="")
        if command[:2] == ["docker", "inspect"]:
            payload = [
                {
                    "Id": container_id,
                    "Image": image_id,
                    "Config": {"Labels": {}},
                }
            ]
            return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lifecycle.subprocess, "run", run)

    with pytest.raises(AttemptError, match="exact attempt ownership identity"):
        lifecycle.create_container(
            attempt,
            args=SimpleNamespace(action="train", device="cpu"),
            root_dir=tmp_path,
            inner_command=[sys.executable, "src/train.py"],
            preflight=_container_preflight(tmp_path, image_id),
        )

    assert ["docker", "rm", "--force", container_id] in calls
    evidence = json.loads((attempt.path / "container.json").read_text())
    assert evidence["id"] == container_id
    assert evidence["removed"] is True
    assert evidence["post_create_identity"]["labels_match"] is False


def test_container_post_create_inspection_exception_still_force_removes_exact_id(
    tmp_path, monkeypatch
):
    attempt = _attempt(tmp_path)
    container_id = "a" * 64
    image_id = "sha256:" + "b" * 64
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        if command[:2] == ["docker", "create"]:
            return SimpleNamespace(returncode=0, stdout=container_id, stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lifecycle.subprocess, "run", run)
    monkeypatch.setattr(
        lifecycle,
        "container_identity",
        lambda _record: (_ for _ in ()).throw(OSError("docker transport disappeared")),
    )

    with pytest.raises(AttemptError, match="exact attempt ownership identity"):
        lifecycle.create_container(
            attempt,
            args=SimpleNamespace(action="train", device="cpu"),
            root_dir=tmp_path,
            inner_command=[sys.executable, "src/train.py"],
            preflight=_container_preflight(tmp_path, image_id),
        )

    assert ["docker", "rm", "--force", container_id] in calls
    evidence = json.loads((attempt.path / "container.json").read_text(encoding="utf-8"))
    assert evidence["removed"] is True
    assert "docker transport disappeared" in evidence["post_create_identity"]["error"]


def test_container_wandb_environment_is_forwarded_without_secret_evidence(tmp_path, monkeypatch):
    attempt = _attempt(tmp_path)
    container_id = "a" * 64
    image_id = "sha256:" + "b" * 64
    secret = "fixture-wandb-secret"
    labels = {
        "llm-scratch.ops.run_id": attempt.run_id,
        "llm-scratch.ops.attempt_id": attempt.attempt_id,
    }
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        if command[:2] == ["docker", "create"]:
            return SimpleNamespace(returncode=0, stdout=container_id, stderr="")
        payload = [
            {
                "Id": container_id,
                "Image": image_id,
                "Config": {"Labels": labels},
            }
        ]
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setenv("WANDB_API_KEY", secret)
    monkeypatch.setattr(lifecycle.subprocess, "run", run)
    record = lifecycle.create_container(
        attempt,
        args=SimpleNamespace(action="train", device="cpu"),
        root_dir=tmp_path,
        inner_command=[sys.executable, "src/train.py"],
        preflight=_container_preflight(tmp_path, image_id, wandb_mode="online"),
    )
    atomic_write_json(attempt.path / "container.json", record)

    create_command = calls[0]
    environment_index = create_command.index("--env")
    assert create_command[environment_index + 1] == "WANDB_API_KEY"
    assert secret not in json.dumps(calls)
    assert secret not in (attempt.path / "container.json").read_text(encoding="utf-8")


def test_generated_container_names_bind_full_case_sensitive_attempt_identity(tmp_path):
    case_a = lifecycle._container_name(runner.Attempt(tmp_path, "Run-001", "Attempt-0001"))
    case_b = lifecycle._container_name(runner.Attempt(tmp_path, "run-001", "Attempt-0001"))
    long_a = lifecycle._container_name(runner.Attempt(tmp_path, "r" * 127 + "A", "attempt-0001"))
    long_b = lifecycle._container_name(runner.Attempt(tmp_path, "r" * 127 + "B", "attempt-0001"))

    assert case_a != case_b
    assert long_a != long_b
    assert all(len(name) <= 128 for name in (case_a, case_b, long_a, long_b))


def test_smoke_child_environment_removes_wandb_credentials(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fixture-secret")
    monkeypatch.setenv("WANDB_BASE_URL", "https://wandb.invalid")

    environment = lifecycle.child_environment("smoke")

    assert "WANDB_API_KEY" not in environment
    assert "WANDB_BASE_URL" not in environment


def test_stale_proc_identity_never_signals_old_process_group(tmp_path, monkeypatch):
    attempt = _attempt(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    trusted = process_identity(process.pid)
    atomic_write_json(attempt.path / "pid.json", trusted)
    killpg_calls = []

    monkeypatch.setattr(
        lifecycle,
        "process_identity",
        lambda _pid: {**trusted, "proc_start_ticks": trusted["proc_start_ticks"] + 1},
    )
    monkeypatch.setattr(
        lifecycle.os,
        "killpg",
        lambda *args: killpg_calls.append(args),
    )

    errors = lifecycle.stop_owned_process(
        process,
        attempt=attempt,
        container_record=None,
        trusted_ownership=trusted,
    )

    assert process.poll() is not None
    assert killpg_calls == []
    assert errors == [
        "live /proc identity differs from trusted child lifecycle; "
        "descendant process ownership is uncertain"
    ]
    kinds = [json.loads(path.read_text())["kind"] for path in attempt.events_dir.glob("*.json")]
    assert "process_group_ownership_uncertain" in kinds


def test_missing_proc_capture_stops_only_exact_direct_child(tmp_path, monkeypatch):
    attempt = _attempt(tmp_path)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    killpg_calls = []
    monkeypatch.setattr(
        lifecycle.os,
        "killpg",
        lambda *args: killpg_calls.append(args),
    )

    errors = lifecycle.stop_owned_process(
        process,
        attempt=attempt,
        container_record=None,
        trusted_ownership=None,
    )

    assert process.poll() is not None
    assert killpg_calls == []
    assert errors == [
        "trusted process-group ownership is unavailable; descendant process ownership is uncertain"
    ]
    kinds = [json.loads(path.read_text())["kind"] for path in attempt.events_dir.glob("*.json")]
    assert "process_group_ownership_unavailable" in kinds


def test_owned_process_group_descendant_is_killed_after_leader_exits(tmp_path):
    attempt = _attempt(tmp_path)
    child_pid_path = tmp_path / "descendant.pid"
    child_code = (
        "import os,signal,sys,time; "
        "from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "Path(sys.argv[1]).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    parent_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}, sys.argv[1]]); "
        "time.sleep(30)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", parent_code, str(child_pid_path)],
        start_new_session=True,
    )
    trusted = process_identity(process.pid)
    atomic_write_json(attempt.path / "pid.json", trusted)
    try:
        for _ in range(200):
            if child_pid_path.is_file():
                break
            time.sleep(0.01)
        assert child_pid_path.is_file()

        errors = lifecycle.stop_owned_process(
            process,
            attempt=attempt,
            container_record=None,
            trusted_ownership=trusted,
        )

        assert errors == []
        assert process.poll() is not None
        assert lifecycle._live_process_group_members(int(trusted["process_group_id"])) == []
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_watchdog_reaps_descendants_when_leader_exits_naturally(tmp_path, monkeypatch):
    attempt = _attempt(tmp_path)
    child_pid_path = tmp_path / "natural-exit-descendant.pid"
    child_code = (
        "import os,signal,sys,time; "
        "from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "Path(sys.argv[1]).write_text(str(os.getpid())); "
        "time.sleep(30)"
    )
    parent_code = (
        "import pathlib,subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}, sys.argv[1]]); "
        "path=pathlib.Path(sys.argv[1]); "
        "deadline=time.monotonic()+5; "
        "\nwhile not path.is_file() and time.monotonic()<deadline: time.sleep(0.01)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", parent_code, str(child_pid_path)],
        start_new_session=True,
    )
    trusted = process_identity(process.pid)
    atomic_write_json(attempt.path / "pid.json", trusted)
    monkeypatch.setattr(lifecycle, "_PROCESS_GROUP_TERM_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(lifecycle, "_PROCESS_GROUP_KILL_GRACE_SECONDS", 0.2)
    process_group_id = int(trusted["process_group_id"])
    try:
        exit_code = lifecycle.wait_with_disk_watchdog(
            process,
            attempt=attempt,
            watchdog={
                "triggered": False,
                "filesystems": [
                    {
                        "device": tmp_path.stat().st_dev,
                        "path": str(tmp_path),
                        "effective_live_floor_bytes": 0,
                        "minimum_observed_free_bytes": None,
                    }
                ],
            },
            container_record=None,
            trusted_ownership=trusted,
        )

        assert exit_code == 0
        assert lifecycle._live_process_group_members(process_group_id) == []
        kinds = [json.loads(path.read_text())["kind"] for path in attempt.events_dir.glob("*.json")]
        assert "process_group_descendants_after_leader_exit" in kinds
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if lifecycle._live_process_group_members(process_group_id):
            try:
                lifecycle.os.killpg(process_group_id, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_handoff_rejects_semantically_invalid_wandb_evidence(tmp_path):
    attempt = _attempt(tmp_path)
    evidence = attempt.path / "work" / "wandb_events.jsonl"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recorded_at_utc": "2026-07-19T00:00:00+00:00",
                "action": "unrelated",
                "outcome": "nonsense",
                "mode": "disabled",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(HandoffValidationError, match="lifecycle action/outcome"):
        _wandb_evidence(
            attempt,
            preflight={"checks": {"wandb": {"mode": "disabled"}}},
            action="smoke",
            outcome="succeeded",
        )


def test_handoff_accepts_failed_online_wandb_initialization(tmp_path):
    attempt = _attempt(tmp_path)
    evidence = attempt.path / "work" / "wandb_events.jsonl"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recorded_at_utc": "2026-07-19T00:00:00+00:00",
                "action": "init",
                "outcome": "failed",
                "mode": "online",
                "error": {"type": "RuntimeError", "message": "service unavailable"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = _wandb_evidence(
        attempt,
        preflight={"checks": {"wandb": {"mode": "online"}}},
        action="train",
        outcome="succeeded",
    )

    assert result["initializations"][0]["outcome"] == "failed"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"recorded_at_utc": "not-a-time"}, "recorded_at_utc is invalid"),
        ({"project": "wrong-project"}, "configured identity"),
        ({"entity": "wrong-entity"}, "configured identity"),
    ],
)
def test_handoff_rejects_unbound_successful_wandb_initialization(tmp_path, mutation, message):
    attempt = _attempt(tmp_path)
    evidence = attempt.path / "work" / "wandb_events.jsonl"
    evidence.parent.mkdir(parents=True)
    record = {
        "schema_version": 1,
        "recorded_at_utc": "2026-07-19T00:00:00+00:00",
        "action": "init",
        "outcome": "succeeded",
        "mode": "online",
        "project": "expected-project",
        "entity": "expected-entity",
        "run_id": "run-id",
        "run_url": "https://wandb.example/run-id",
        **mutation,
    }
    evidence.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(HandoffValidationError, match=message):
        _wandb_evidence(
            attempt,
            preflight={
                "checks": {
                    "wandb": {
                        "mode": "online",
                        "project": "expected-project",
                        "entity": "expected-entity",
                    }
                }
            },
            action="train",
            outcome="succeeded",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"project": "wrong-project"}, "configured identity"),
        ({"run_id": 123}, "configured identity"),
        ({"run_url": {"not": "a URL"}}, "valid run URL"),
    ],
)
def test_handoff_rejects_invalid_successful_wandb_identity_when_attempt_failed(
    tmp_path, mutation, message
):
    attempt = _attempt(tmp_path)
    evidence = attempt.path / "work" / "wandb_events.jsonl"
    evidence.parent.mkdir(parents=True)
    record = {
        "schema_version": 1,
        "recorded_at_utc": "2026-07-19T00:00:00+00:00",
        "action": "init",
        "outcome": "succeeded",
        "mode": "online",
        "project": "expected-project",
        "entity": "expected-entity",
        "run_id": "run-id",
        "run_url": "https://wandb.example/run-id",
        **mutation,
    }
    evidence.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(HandoffValidationError, match=message):
        _wandb_evidence(
            attempt,
            preflight={
                "checks": {
                    "wandb": {
                        "mode": "online",
                        "project": "expected-project",
                        "entity": "expected-entity",
                    }
                }
            },
            action="train",
            outcome="failed",
        )


def test_malformed_terminal_container_inspect_still_force_removes_exact_id(tmp_path, monkeypatch):
    attempt = _attempt(tmp_path)
    container_id = "a" * 64
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        if command[:2] == ["docker", "inspect"]:
            return SimpleNamespace(returncode=0, stdout="not-json", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lifecycle.subprocess, "run", run)
    record = {
        "id": container_id,
        "image_id": "sha256:" + "b" * 64,
        "labels": {},
        "removed": False,
    }

    errors = lifecycle.capture_and_remove_container(attempt, record)

    assert ["docker", "rm", "--force", container_id] in calls
    assert record["removed"] is True
    assert errors and errors[0].startswith("container finalization: JSONDecodeError")


def test_lifecycle_errors_force_failed_result_and_are_preserved(tmp_path):
    attempt = _attempt(tmp_path)
    (attempt.path / "stdout.log").write_text("", encoding="utf-8")
    (attempt.path / "stderr.log").write_text("", encoding="utf-8")

    result = runner._execution_result(
        attempt,
        action="smoke",
        exit_code=0,
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:00:01Z",
        elapsed_seconds=1.0,
        watchdog={"triggered": False, "filesystems": []},
        lifecycle_errors=["container evidence write failed"],
    )

    assert result["outcome"] == "failed"
    assert result["lifecycle_errors"] == ["container evidence write failed"]
    assert "container evidence" in result["diagnosis"]["summary"]
