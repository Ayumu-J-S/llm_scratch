from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from operations import preflight as preflight_module
from operations.preflight import (
    LIVE_DISK_FLOOR_BYTES,
    POST_PLAN_RESERVE_BYTES,
    PreflightError,
    _storage_check,
    _wandb_check,
    run_preflight,
)
from operations.runner import _compose
from training.checkpoint import checkpoint_config_sha256


ROOT_DIR = Path(__file__).resolve().parents[1]


def _cfg(*overrides: str):
    return _compose(ROOT_DIR, ["profile=smoke_overfit", *overrides])


def _clean_git(monkeypatch):
    monkeypatch.setattr(
        preflight_module,
        "_git_check",
        lambda _root: {
            "sha": "a" * 40,
            "branch": "fixture",
            "dirty": False,
            "worktree_status": [],
            "lock_sha256": "b" * 64,
        },
    )


def test_missing_wandb_credentials_blocks_online_action_after_local_checks(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    monkeypatch.setattr(preflight_module, "_wandb_credential_visible", lambda _executor: False)
    cfg = _cfg("wandb.mode=online")

    report = run_preflight(
        cfg,
        root_dir=ROOT_DIR,
        run_root=tmp_path,
        action="train",
        executor="host",
        device="cpu",
        image=None,
    )

    assert report["local_checks_complete"] is True
    assert report["checks"]["configuration"]["status"] == "passed"
    assert report["checks"]["storage"]["status"] == "passed"
    assert report["checks"]["wandb"]["status"] == "blocked_online"
    assert report["ready"] is False
    assert "wandb login" in report["login_prompt"]


def test_missing_wandb_credentials_do_not_block_local_config_check(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    monkeypatch.setattr(preflight_module, "_wandb_credential_visible", lambda _executor: False)

    report = run_preflight(
        _cfg("wandb.mode=online"),
        root_dir=ROOT_DIR,
        run_root=tmp_path,
        action="config-check",
        executor="host",
        device="cpu",
        image=None,
    )

    assert report["ready"] is True
    assert report["login_prompt"]


def test_visible_wandb_quota_below_reserve_blocks_upload(tmp_path, monkeypatch):
    snapshot = tmp_path / "wandb-usage.json"
    snapshot.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "entity": "sunday-research",
                "plan": "Free",
                "used_bytes": 90,
                "limit_bytes": 100,
                "retention": "visible fixture",
                "source": "fixture",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(preflight_module, "_wandb_credential_visible", lambda _executor: True)
    cfg = _cfg(
        "wandb.mode=online",
        "wandb.artifact.policy=final",
        f"wandb.artifact.usage_snapshot_path={snapshot}",
        "wandb.artifact.reserve_bytes=20",
    )

    result = _wandb_check(cfg)

    assert result["status"] == "blocked_online"
    assert result["quota"]["status"] == "insufficient"


def test_container_wandb_auth_requires_environment_transport(monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(
        preflight_module.netrc,
        "netrc",
        lambda: SimpleNamespace(authenticators=lambda _host: ("user", None, "netrc-secret")),
    )

    assert preflight_module._wandb_credential_visible("host") is True
    assert preflight_module._wandb_credential_visible("container") is False

    monkeypatch.setenv("WANDB_API_KEY", "environment-secret")
    assert preflight_module._wandb_credential_visible("container") is True
    result = _wandb_check(_cfg("wandb.mode=online"), executor="container")
    assert result["status"] == "passed"
    assert result["credential_transport"] == "environment"


def test_container_missing_wandb_auth_prompts_for_environment_key(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(
        preflight_module,
        "_device_check",
        lambda **_kwargs: {"selected": "cpu", "image_id": "sha256:" + "a" * 64},
    )

    report = run_preflight(
        _cfg("wandb.mode=online"),
        root_dir=ROOT_DIR,
        run_root=tmp_path,
        action="train",
        executor="container",
        device="cpu",
        image="fixture",
    )

    assert report["ready"] is False
    assert "Export WANDB_API_KEY" in report["login_prompt"]


def test_evaluation_uses_evaluation_owned_wandb_configuration(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "fixture-secret")
    cfg = _compose(
        ROOT_DIR,
        ["profile=evaluation", "evaluation.wandb.mode=online"],
    )

    result = _wandb_check(cfg, executor="host")

    assert result["mode"] == "online"
    assert result["status"] == "passed"


def test_real_actions_reject_dirty_git_but_smoke_remains_available(tmp_path, monkeypatch):
    monkeypatch.setattr(
        preflight_module,
        "_git_check",
        lambda _root: {
            "sha": "a" * 40,
            "branch": "fixture",
            "dirty": True,
            "worktree_status": [" M fixture"],
            "lock_sha256": "b" * 64,
        },
    )
    cfg = _cfg()

    train = run_preflight(
        cfg,
        root_dir=ROOT_DIR,
        run_root=tmp_path,
        action="train",
        executor="host",
        device="cpu",
        image=None,
    )
    smoke = run_preflight(
        cfg,
        root_dir=ROOT_DIR,
        run_root=tmp_path,
        action="smoke",
        executor="host",
        device="cpu",
        image=None,
    )

    assert not train["ready"]
    assert any("clean worktree" in error for error in train["local_errors"])
    assert smoke["ready"]


def test_storage_rejects_below_120gb_live_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(
        preflight_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=1_000_000_000_000, used=0, free=119_999_999_999),
    )

    with pytest.raises(PreflightError, match="requires live floor"):
        _storage_check(_cfg(), root_dir=ROOT_DIR, run_root=tmp_path, action="smoke")


def test_storage_projects_remaining_cache_growth_once_per_path(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "existing").write_bytes(b"x" * 25)
    plain = OmegaConf.to_container(_cfg(), resolve=True)
    cache_config = {
        "dir": str(cache),
        "max_size_bytes": 100,
        "min_free_bytes": 0,
    }
    plain["data"]["streaming"] = {"cache": cache_config}
    plain["benchmark"] = {"cache": dict(cache_config)}
    monkeypatch.setattr(
        preflight_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=2_000_000_000_000, used=0, free=1_000_000_000_000),
    )

    result = _storage_check(
        OmegaConf.create(plain), root_dir=ROOT_DIR, run_root=tmp_path, action="eval"
    )

    assert result["filesystems"][0]["projected_additional_bytes"] == 1_000_000_075


def test_atomic_write_projection_dynamically_raises_live_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight_module, "estimate_parameter_count", lambda _cfg: 200_000_000)
    monkeypatch.setattr(
        preflight_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=2_000_000_000_000, used=0, free=1_000_000_000_000),
    )

    result = _storage_check(_cfg(), root_dir=ROOT_DIR, run_root=tmp_path, action="eval")

    expected = POST_PLAN_RESERVE_BYTES + 200_000_000 * 128 + 4_000_000_000
    assert expected > LIVE_DISK_FLOOR_BYTES
    assert result["effective_run_live_floor_bytes"] == expected


def test_eval_uses_verified_checkpoint_owned_config_for_manifest_and_storage(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    checkpoint_cfg = OmegaConf.to_container(_cfg(), resolve=True)
    identity = checkpoint_config_sha256(checkpoint_cfg)
    fake = SimpleNamespace(
        payload={
            "kind": "final",
            "identity": {"config_sha256": identity},
            "state": {"resolved_config": checkpoint_cfg},
        },
        physical_identity={
            "path": str(tmp_path / "final.pt"),
            "sha256": "c" * 64,
            "size_bytes": 123,
            "device": 1,
            "inode": 2,
            "mtime_ns": 3,
            "ctime_ns": 4,
        },
    )
    monkeypatch.setattr(preflight_module, "load_checkpoint_for_generation", lambda _path: fake)
    evaluation = _compose(
        ROOT_DIR,
        [
            "profile=evaluation",
            f"evaluation.checkpoint_path={tmp_path / 'final.pt'}",
            f"evaluation.output_path={tmp_path / 'evaluation.json'}",
            "evaluation.device=cpu",
            "runtime.device=cpu",
        ],
    )

    report = run_preflight(
        evaluation,
        root_dir=ROOT_DIR,
        run_root=tmp_path,
        action="eval",
        executor="host",
        device="cpu",
        image=None,
        checkpoint_path=tmp_path / "final.pt",
    )

    assert report["checks"]["checkpoint"]["status"] == "passed"
    assert report["checks"]["manifests"]["data_manifests"][0]["split"] == "memorization"
    assert report["checks"]["storage"]["parameter_count"] > 0
