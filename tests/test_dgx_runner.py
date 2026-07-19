import importlib.util
import json
import os
from pathlib import Path

import hydra
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
        assert command[command.index("--max-swap-in-pages") + 1] == "0"
        assert "--network=none" in command
        assert "profile=pretrain_baseline" in command
        assert "wandb.mode=disabled" in command
        assert "measurement.enabled=false" in command


def test_shard_cache_identity_is_content_bound_and_deterministic(tmp_path):
    (tmp_path / "a.shard").write_bytes(b"a")
    (tmp_path / "b.shard").write_bytes(b"b")
    first = _shard_cache_identity(tmp_path)
    second = _shard_cache_identity(tmp_path)
    assert first == second
    (tmp_path / "b.shard").write_bytes(b"changed")
    assert _shard_cache_identity(tmp_path)["sha256"] != first["sha256"]


def test_selected_entry_requires_a_passing_summary_for_current_commit(monkeypatch, tmp_path):
    cfg = _config()
    plan = build_matrix_plan(cfg)
    selected = {key: value for key, value in plan[0].items() if key != "repetition"}
    summary_path = tmp_path / "summary.json"
    summary = {
        "schema_version": 3,
        "verdict": "PASS",
        "git_commit": "b" * 40,
        "selected": {**selected, "conservative_tokens_per_second": 10.0},
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    cfg["selected_candidate"] = selected["candidate_id"]
    cfg["matrix_summary_path"] = str(summary_path)
    monkeypatch.setattr(RUNNER, "_git", lambda *args: "a" * 40)
    try:
        _selected_entry(cfg, plan)
    except RuntimeError as error:
        assert "for this commit" in str(error)
    else:
        raise AssertionError("a summary from another commit must be rejected")

    summary["git_commit"] = "a" * 40
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert _selected_entry(cfg, plan) == selected
