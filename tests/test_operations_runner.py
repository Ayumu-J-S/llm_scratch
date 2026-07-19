from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from operations import lifecycle, runner
from operations.artifacts import (
    AttemptError,
    atomic_write_json,
    create_attempt,
    process_identity,
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
            preflight={"checks": {"device": {"image_id": image_id}}},
        )

    assert ["docker", "rm", "--force", container_id] in calls
    evidence = json.loads((attempt.path / "container.json").read_text())
    assert evidence["id"] == container_id
    assert evidence["removed"] is True
    assert evidence["post_create_identity"]["labels_match"] is False


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
        preflight={
            "checks": {
                "device": {"image_id": image_id},
                "wandb": {"mode": "online"},
            }
        },
    )
    atomic_write_json(attempt.path / "container.json", record)

    create_command = calls[0]
    environment_index = create_command.index("--env")
    assert create_command[environment_index + 1] == "WANDB_API_KEY"
    assert secret not in json.dumps(calls)
    assert secret not in (attempt.path / "container.json").read_text(encoding="utf-8")


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
