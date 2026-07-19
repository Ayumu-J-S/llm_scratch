import importlib.util
import json
import os
from pathlib import Path

import hydra
import pytest
from omegaconf import OmegaConf

from dgx.planning import build_matrix_plan


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "run_dgx_measurements", ROOT / "scripts/run_dgx_measurements.py"
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
_container_command = RUNNER._container_command
_preflight_storage = RUNNER._preflight_storage
_run_config_authorities = RUNNER._run_config_authorities
_resource_budget = RUNNER._resource_budget
_resolved_role_config = RUNNER._resolved_role_config
_run_commands = RUNNER._run_commands
_selected_entry = RUNNER._selected_entry
_shard_cache_identity = RUNNER._shard_cache_identity


def _config():
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        config = hydra.compose(config_name="dgx")
    return OmegaConf.to_container(config, resolve=True)


def test_matrix_container_is_user_owned_offline_and_tracking_disabled(tmp_path):
    cfg = _config()
    entry = build_matrix_plan(cfg)[0]
    command = _container_command(
        cfg,
        entry,
        commit="a" * 40,
        image_id=cfg["image"]["expected_id"],
        plan_id="plan",
        output_root=tmp_path,
        cache_root=tmp_path,
        role="matrix",
    )
    assert command[command.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    assert cfg["image"]["expected_id"] in command
    assert cfg["image"]["name"] not in command
    assert "--network=none" in command
    assert "WANDB_MODE=disabled" in command
    assert "WANDB_DISABLED=true" in command
    assert "WANDB_API_KEY" not in command
    assert "wandb.mode=disabled" in command
    assert "wandb.watch.enabled=false" in command
    assert "wandb.artifact.policy=none" in command


def test_pilot_uses_selected_baseline_online_without_serializing_auth(monkeypatch, tmp_path):
    cfg = _config()
    entry = {key: value for key, value in build_matrix_plan(cfg)[0].items() if key != "repetition"}
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    command = _container_command(
        cfg,
        entry,
        commit="a" * 40,
        image_id=cfg["image"]["expected_id"],
        plan_id="plan",
        output_root=tmp_path,
        cache_root=tmp_path,
        role="pilot",
    )
    assert "--network=none" not in command
    assert any(item.endswith(":/evidence/home/.netrc:ro") for item in command)
    assert not any("api_key" in item.lower() or "apikey" in item.lower() for item in command)
    assert "profile=pretrain_baseline" in command
    assert "training.max_time=1800" in command
    assert "data.streaming.validation.max_target_tokens=65536" in command
    assert "training.validation_every_n_tokens=5000000" in command
    assert "training.checkpoint_every_n_tokens=2500000" in command
    assert "training.milestone_every_n_tokens=100000000" in command
    assert "training.log_every_n_steps=25" in command
    assert "wandb.mode=online" in command
    assert "wandb.watch.enabled=false" in command
    assert "wandb.artifact.policy=none" in command


def test_decomposition_commands_are_exact_selected_profile_and_offline(tmp_path):
    cfg = _config()
    entry = {**build_matrix_plan(cfg)[0], "repetition": 2}
    for role in ("model-only", "loader-only"):
        output_root = tmp_path / role
        output_root.mkdir()
        command = _container_command(
            cfg,
            entry,
            commit="a" * 40,
            image_id=cfg["image"]["expected_id"],
            plan_id="plan",
            output_root=output_root,
            cache_root=tmp_path,
            role=role,
        )
        assert "scripts/measure_dgx_decomposition.py" in command
        assert command[command.index("--role") + 1] == role
        assert command[command.index("--repetition") + 1] == "2"
        assert command[command.index("--warmup-optimizer-steps") + 1] == "10"
        assert command[command.index("--measured-optimizer-steps") + 1] == "20"
        assert command[command.index("--min-available-memory-bytes") + 1] == "64000000000"
        assert command[command.index("--min-free-disk-bytes") + 1] == "120000000000"
        assert command[command.index("--post-plan-free-reserve-bytes") + 1] == "100000000000"
        assert int(command[command.index("--max-in-flight-atomic-write-bytes") + 1]) < (
            20_000_000_000
        )
        assert command[command.index("--max-swap-in-pages") + 1] == "0"
        assert "--network=none" in command
        assert "profile=pretrain_baseline" in command
        assert "wandb.mode=disabled" in command
        assert "measurement.enabled=false" in command


def test_every_execution_role_has_a_full_config_authority():
    cfg = _config()
    entry = {**build_matrix_plan(cfg)[0], "repetition": 1}
    roles = [
        (entry, "matrix"),
        (entry, "model-only"),
        (entry, "loader-only"),
        (entry, "pilot"),
    ]
    authorities = _run_config_authorities(cfg, roles)
    assert len(authorities) == 4
    assert len({item["authority_key"] for item in authorities}) == 4
    assert all(len(item["canonical_config_sha256"]) == 64 for item in authorities)
    assert all(len(item["experiment_config_sha256"]) == 64 for item in authorities)
    assert all(item["max_in_flight_atomic_write_bytes"] < 20_000_000_000 for item in authorities)
    assert all(item["effective_min_free_disk_bytes"] == 120_000_000_000 for item in authorities)


def test_atomic_write_budget_raises_floor_when_projection_exceeds_static_buffer():
    cfg = _config()
    entry = build_matrix_plan(cfg)[0]
    resolved = _resolved_role_config(cfg, entry, "matrix")
    resolved["model"]["num_layers"] = 200
    budget = _resource_budget(resolved, 120_000_000_000)
    assert budget["max_in_flight_atomic_write_bytes"] > 20_000_000_000
    assert budget["effective_min_free_disk_bytes"] == (
        100_000_000_000 + budget["max_in_flight_atomic_write_bytes"]
    )


def test_failed_role_stops_runner_and_marks_execution_incomplete(monkeypatch, tmp_path):
    run_dirs = [tmp_path / "first", tmp_path / "second"]
    commands = []
    for run_dir in run_dirs:
        run_dir.mkdir()
        commands.append(["docker", "run", f"{run_dir}:/evidence"])
    calls = []

    def failed_first(command, **_kwargs):
        calls.append(command)
        return type("Result", (), {"returncode": 23})()

    monkeypatch.setattr(RUNNER.subprocess, "run", failed_first)
    assert _run_commands(commands, tmp_path) == 23
    assert calls == [commands[0]]
    execution = json.loads((tmp_path / "execution.json").read_text(encoding="utf-8"))
    assert execution["complete"] is False
    assert execution["return_code"] == 23
    assert len(execution["attempts"]) == 1


def test_shard_cache_identity_is_content_bound_and_deterministic(tmp_path):
    (tmp_path / "a.shard").write_bytes(b"a")
    (tmp_path / "b.shard").write_bytes(b"b")
    first = _shard_cache_identity(tmp_path)
    second = _shard_cache_identity(tmp_path)
    assert first == second
    (tmp_path / "b.shard").write_bytes(b"changed")
    assert _shard_cache_identity(tmp_path)["sha256"] != first["sha256"]


def test_storage_preflight_aggregates_growth_on_one_filesystem(monkeypatch, tmp_path):
    output = tmp_path / "output"
    cache = tmp_path / "cache"
    output.mkdir()
    cache.mkdir()
    monkeypatch.setattr(RUNNER, "_filesystem_device", lambda _path: 1)
    monkeypatch.setattr(
        RUNNER.shutil, "disk_usage", lambda _path: type("Usage", (), {"free": 298_000_000_000})()
    )
    with pytest.raises(RuntimeError, match="hard floor"):
        _preflight_storage(
            output_parent=output,
            cache_root=cache,
            cache_existing_bytes=1_000_000_000,
            cache_max_bytes=80_000_000_000,
            output_growth_bytes=100_000_000_000,
            minimum_free_bytes=120_000_000_000,
        )


def test_storage_preflight_accounts_split_filesystems_independently(monkeypatch, tmp_path):
    output = tmp_path / "output"
    cache = tmp_path / "cache"
    output.mkdir()
    cache.mkdir()
    monkeypatch.setattr(RUNNER, "_filesystem_device", lambda path: 1 if path == output else 2)
    free = {output: 220_000_000_000, cache: 199_000_000_000}
    monkeypatch.setattr(
        RUNNER.shutil,
        "disk_usage",
        lambda path: type("Usage", (), {"free": free[path]})(),
    )
    _preflight_storage(
        output_parent=output,
        cache_root=cache,
        cache_existing_bytes=1_000_000_000,
        cache_max_bytes=80_000_000_000,
        output_growth_bytes=100_000_000_000,
        minimum_free_bytes=120_000_000_000,
    )


def test_selected_entry_requires_a_passing_summary_for_current_commit(monkeypatch, tmp_path):
    cfg = _config()
    plan = build_matrix_plan(cfg)
    selected = {key: value for key, value in plan[0].items() if key != "repetition"}
    summary_path = tmp_path / "summary.json"
    commit = "a" * 40
    matrix_config = json.loads(json.dumps(cfg))
    matrix_config["mode"] = "matrix"
    matrix_config["selected_candidate"] = None
    matrix_config["matrix_summary_path"] = None
    matrix_plan = {
        "schema_version": 3,
        "ticket": "DGX-001",
        "config": matrix_config,
        "git_commit": commit,
        "image_id": cfg["image"]["expected_id"],
        "runs": plan,
        "selected": None,
        "matrix_summary_identity": None,
    }
    matrix_plan["plan_id"] = RUNNER._canonical_sha256(matrix_plan)
    (tmp_path / "plan.json").write_text(json.dumps(matrix_plan), encoding="utf-8")
    summary = {
        "schema_version": 3,
        "verdict": "PASS",
        "git_commit": "b" * 40,
        "image_id": cfg["image"]["expected_id"],
        "plan_id": matrix_plan["plan_id"],
        "selection_rule": cfg["selection"],
        "selected": {**selected, "conservative_tokens_per_second": 10.0},
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    cfg["selected_candidate"] = selected["candidate_id"]
    cfg["matrix_summary_path"] = str(summary_path)
    monkeypatch.setattr(RUNNER, "_git", lambda *args: commit)
    with pytest.raises(RuntimeError, match="exact passing matrix"):
        _selected_entry(cfg, plan)

    summary["git_commit"] = commit
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert _selected_entry(cfg, plan) == selected

    summary["plan_id"] = "f" * 64
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact passing matrix"):
        _selected_entry(cfg, plan)

    summary["plan_id"] = matrix_plan["plan_id"]
    summary["selection_rule"] = {**cfg["selection"], "max_slowdown_fraction": 0.99}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact passing matrix"):
        _selected_entry(cfg, plan)

    summary["selection_rule"] = cfg["selection"]
    matrix_plan["config"]["selection"]["max_slowdown_fraction"] = 0.99
    unsigned = dict(matrix_plan)
    unsigned.pop("plan_id")
    matrix_plan["plan_id"] = RUNNER._canonical_sha256(unsigned)
    (tmp_path / "plan.json").write_text(json.dumps(matrix_plan), encoding="utf-8")
    summary["plan_id"] = matrix_plan["plan_id"]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(RuntimeError, match="exact passing matrix"):
        _selected_entry(cfg, plan)
