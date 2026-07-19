from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from operations import preflight as preflight_module
from operations.lifecycle import watchdog_targets
from operations.preflight import (
    LIVE_DISK_FLOOR_BYTES,
    POST_PLAN_RESERVE_BYTES,
    PreflightError,
    _container_mount_check,
    _milestone_checkpoint_bound,
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


def test_cpu_bf16_training_profile_is_blocked_during_preflight(tmp_path, monkeypatch):
    _clean_git(monkeypatch)

    report = run_preflight(
        _cfg("training.precision=bf16"),
        root_dir=ROOT_DIR,
        run_root=tmp_path,
        action="train",
        executor="host",
        device="cpu",
        image=None,
    )

    assert report["ready"] is False
    assert report["checks"]["device"]["status"] == "failed"
    assert "training.precision=bf16 requires device=cuda" in report["checks"]["device"]["error"]


def test_checkpoint_owned_bf16_blocks_cpu_evaluation_preflight(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    checkpoint_path = tmp_path / "final.pt"
    checkpoint_path.write_bytes(b"fixture")
    checkpoint_cfg = OmegaConf.to_container(
        _cfg("training.precision=bf16"),
        resolve=True,
    )
    identity = checkpoint_config_sha256(checkpoint_cfg)
    fake = SimpleNamespace(
        payload={
            "kind": "final",
            "identity": {"config_sha256": identity},
            "state": {"resolved_config": checkpoint_cfg},
        },
        physical_identity={
            "path": str(checkpoint_path),
            "sha256": "c" * 64,
            "size_bytes": checkpoint_path.stat().st_size,
            "device": checkpoint_path.stat().st_dev,
            "inode": checkpoint_path.stat().st_ino,
            "mtime_ns": checkpoint_path.stat().st_mtime_ns,
            "ctime_ns": checkpoint_path.stat().st_ctime_ns,
        },
    )
    monkeypatch.setattr(preflight_module, "load_checkpoint_for_generation", lambda _path: fake)
    cfg = _compose(
        ROOT_DIR,
        [
            "profile=evaluation",
            f"evaluation.checkpoint_path={checkpoint_path}",
            "evaluation.device=cpu",
            "runtime.device=cpu",
        ],
    )

    report = run_preflight(
        cfg,
        root_dir=ROOT_DIR,
        run_root=tmp_path,
        action="eval",
        executor="host",
        device="cpu",
        image=None,
        checkpoint_path=checkpoint_path,
    )

    assert report["checks"]["checkpoint"]["status"] == "passed"
    assert report["ready"] is False
    assert report["checks"]["device"]["status"] == "failed"
    assert "training.precision=bf16 requires device=cuda" in report["checks"]["device"]["error"]


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


def test_storage_rejects_post_plan_free_below_effective_live_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight_module, "estimate_parameter_count", lambda _cfg: 1)
    monkeypatch.setattr(
        preflight_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            total=500_000_000_000,
            used=0,
            free=120_500_000_000,
        ),
    )

    with pytest.raises(PreflightError, match="effective post-plan floor 120000000000"):
        _storage_check(_cfg(), root_dir=ROOT_DIR, run_root=tmp_path, action="eval")


def test_storage_forecasts_every_accumulating_milestone(tmp_path, monkeypatch):
    monkeypatch.setattr(
        preflight_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            total=10_000_000_000_000,
            used=0,
            free=9_000_000_000_000,
        ),
    )
    cfg = _cfg(
        "training.max_steps=100",
        "training.milestone_every_n_steps=1",
        "artifacts.keep_last_n=2",
    )

    result = _storage_check(cfg, root_dir=ROOT_DIR, run_root=tmp_path, action="train")

    assert result["milestone_checkpoint_bound"] == 100
    assert result["checkpoint_plan_bytes"] == result["maximum_atomic_write_bytes"] * 105


def test_accumulating_milestones_require_a_finite_step_or_token_bound():
    plain = OmegaConf.to_container(_cfg("training.milestone_every_n_steps=1"), resolve=True)

    with pytest.raises(PreflightError, match="finitely forecastable"):
        _milestone_checkpoint_bound(plain, action="train")


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


def test_corrupt_checkpoint_preflight_retains_raw_physical_identity(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    checkpoint = tmp_path / "corrupt.pt"
    checkpoint.write_bytes(b"corrupt checkpoint evidence")
    evaluation = _compose(
        ROOT_DIR,
        [
            "profile=evaluation",
            f"evaluation.checkpoint_path={checkpoint}",
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
        checkpoint_path=checkpoint,
    )

    record = report["checks"]["checkpoint"]
    assert record["status"] == "failed"
    assert record["physical_identity"]["path"] == str(checkpoint.resolve())
    assert record["physical_identity"]["size_bytes"] == len(b"corrupt checkpoint evidence")
    assert len(record["physical_identity"]["sha256"]) == 64


def test_resume_preflight_rejects_incomplete_full_state(tmp_path, monkeypatch):
    _clean_git(monkeypatch)
    checkpoint_path = tmp_path / "recovery.pt"
    checkpoint_path.write_bytes(b"fixture")
    checkpoint_cfg = OmegaConf.to_container(_cfg(), resolve=True)
    identity = checkpoint_config_sha256(checkpoint_cfg)
    fake = SimpleNamespace(
        payload={
            "kind": "recovery",
            "identity": {"config_sha256": identity},
            "state": {
                "model": {},
                "counters": {"optimizer_step": 0, "target_tokens": 0},
                "stream_cursor": {},
                "resolved_config": checkpoint_cfg,
            },
        },
        physical_identity={
            "path": str(checkpoint_path),
            "sha256": "c" * 64,
            "size_bytes": checkpoint_path.stat().st_size,
            "device": checkpoint_path.stat().st_dev,
            "inode": checkpoint_path.stat().st_ino,
            "mtime_ns": checkpoint_path.stat().st_mtime_ns,
            "ctime_ns": checkpoint_path.stat().st_ctime_ns,
        },
    )
    monkeypatch.setattr(preflight_module, "load_checkpoint_for_generation", lambda _path: fake)

    report = run_preflight(
        _cfg(),
        root_dir=ROOT_DIR,
        run_root=tmp_path,
        action="resume",
        executor="host",
        device="cpu",
        image=None,
        checkpoint_path=checkpoint_path,
    )

    assert report["ready"] is False
    assert report["checks"]["checkpoint"]["status"] == "failed"
    assert "missing full-state entries" in report["checks"]["checkpoint"]["error"]


def test_checkpoint_benchmark_includes_action_owned_cache_in_all_resource_gates(
    tmp_path, monkeypatch
):
    _clean_git(monkeypatch)
    checkpoint_path = tmp_path / "final.pt"
    checkpoint_path.write_bytes(b"fixture")
    benchmark_cache = tmp_path / "benchmark-cache"
    benchmark_cache.mkdir()
    checkpoint_cfg = OmegaConf.to_container(_cfg(), resolve=True)
    identity = checkpoint_config_sha256(checkpoint_cfg)
    fake = SimpleNamespace(
        payload={
            "kind": "final",
            "identity": {"config_sha256": identity},
            "state": {"resolved_config": checkpoint_cfg},
        },
        physical_identity={
            "path": str(checkpoint_path),
            "sha256": "c" * 64,
            "size_bytes": checkpoint_path.stat().st_size,
            "device": checkpoint_path.stat().st_dev,
            "inode": checkpoint_path.stat().st_ino,
            "mtime_ns": checkpoint_path.stat().st_mtime_ns,
            "ctime_ns": checkpoint_path.stat().st_ctime_ns,
        },
    )
    monkeypatch.setattr(preflight_module, "load_checkpoint_for_generation", lambda _path: fake)
    monkeypatch.setattr(
        preflight_module,
        "_device_check",
        lambda **_kwargs: {"selected": "cpu", "image_id": "sha256:" + "a" * 64},
    )
    benchmark = _compose(
        ROOT_DIR,
        [
            "profile=benchmark",
            f"benchmark.checkpoint_path={checkpoint_path}",
            f"benchmark.output_root={tmp_path / 'results'}",
            f"benchmark.cache.dir={benchmark_cache}",
            "benchmark.cache.max_size_bytes=1000000",
            "benchmark.cache.min_free_bytes=200000000000",
            "benchmark.device=cpu",
            "runtime.device=cpu",
        ],
    )

    report = run_preflight(
        benchmark,
        root_dir=ROOT_DIR,
        run_root=tmp_path,
        action="benchmark",
        executor="container",
        device="cpu",
        image="fixture",
        checkpoint_path=checkpoint_path,
    )

    assert report["ready"] is True
    cache_record = next(
        record
        for record in report["checks"]["cache"]["caches"]
        if record["path"] == str(benchmark_cache.resolve())
    )
    assert cache_record["owners"] == ["action"]
    mounts = {record["source"]: record for record in report["checks"]["container_mounts"]["mounts"]}
    assert mounts[str(benchmark_cache.resolve())]["read_only"] is False
    filesystem = next(
        record
        for record in report["checks"]["storage"]["filesystems"]
        if record["device"] == benchmark_cache.stat().st_dev
    )
    assert filesystem["projected_additional_bytes"] == 1_001_000_000
    watchdog = next(
        record
        for record in watchdog_targets(report)
        if record["device"] == benchmark_cache.stat().st_dev
    )
    assert watchdog["effective_live_floor_bytes"] == 200_000_000_000


def test_container_mount_plan_binds_external_inputs_and_linked_git_metadata(tmp_path):
    run_root = tmp_path / "runs"
    cache = tmp_path / "external-cache"
    tokenizer = tmp_path / "tokenizer.json"
    data_manifest = tmp_path / "data.json"
    usage = tmp_path / "wandb-usage.json"
    checkpoint = tmp_path / "final.pt"
    measurement_dir = tmp_path / "measurement-evidence"
    for directory in (run_root, cache, measurement_dir):
        directory.mkdir()
    for path in (tokenizer, data_manifest, usage, checkpoint):
        path.write_text("fixture", encoding="utf-8")
    plain = OmegaConf.to_container(_cfg(), resolve=True)
    plain["tokenizer"]["manifest_path"] = str(tokenizer)
    plain["wandb"]["artifact"]["usage_snapshot_path"] = str(usage)
    plain["measurement"]["enabled"] = True
    plain["measurement"]["output_path"] = str(measurement_dir / "measurement.json")

    result = _container_mount_check(
        OmegaConf.create(plain),
        root_dir=ROOT_DIR,
        run_root=run_root,
        executor="container",
        checkpoint_path=checkpoint,
        manifests={"data_manifests": [{"path": str(data_manifest)}]},
        cache={"caches": [{"path": str(cache)}]},
    )

    mounts = {record["source"]: record for record in result["mounts"]}
    common_git = preflight_module._git_absolute_path(ROOT_DIR, "--git-common-dir")
    assert mounts[str(ROOT_DIR.resolve())]["read_only"] is False
    assert mounts[str(run_root.resolve())]["read_only"] is False
    assert mounts[str(cache.resolve())]["read_only"] is False
    assert mounts[str(measurement_dir.resolve())]["read_only"] is False
    assert mounts[str(common_git)]["read_only"] is True
    for path in (tokenizer, data_manifest, usage, checkpoint):
        assert mounts[str(path.resolve())]["read_only"] is True


def test_resume_container_mounts_checkpoint_bound_measurement_evidence(tmp_path):
    run_root = tmp_path / "current" / "runs"
    run_root.mkdir(parents=True)
    checkpoint = tmp_path / "prior" / "work" / "checkpoints" / "recovery.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("fixture", encoding="utf-8")
    measurement = checkpoint.parent.parent / "measurement.json"
    measurement.write_text("{}", encoding="utf-8")
    plain = OmegaConf.to_container(_cfg(), resolve=True)
    plain["measurement"]["enabled"] = True
    plain["measurement"]["output_path"] = "measurement.json"

    result = _container_mount_check(
        OmegaConf.create(plain),
        root_dir=ROOT_DIR,
        run_root=run_root,
        executor="container",
        checkpoint_path=checkpoint,
        manifests={"data_manifests": []},
        cache={"caches": []},
        action="resume",
    )

    mounts = {record["source"]: record for record in result["mounts"]}
    assert mounts[str(measurement.resolve())]["read_only"] is True
    assert "resume_measurement_evidence" in mounts[str(measurement.resolve())]["purposes"]


def test_resume_container_keeps_shared_absolute_measurement_path_writable(tmp_path):
    run_root = tmp_path / "current" / "runs"
    run_root.mkdir(parents=True)
    checkpoint = tmp_path / "prior" / "checkpoints" / "recovery.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("fixture", encoding="utf-8")
    measurement = tmp_path / "external-measurement" / "measurement.json"
    measurement.parent.mkdir()
    measurement.write_text("{}", encoding="utf-8")
    plain = OmegaConf.to_container(_cfg(), resolve=True)
    plain["measurement"]["enabled"] = True
    plain["measurement"]["output_path"] = str(measurement)

    result = _container_mount_check(
        OmegaConf.create(plain),
        root_dir=ROOT_DIR,
        run_root=run_root,
        executor="container",
        checkpoint_path=checkpoint,
        manifests={"data_manifests": []},
        cache={"caches": []},
        action="resume",
    )

    mounts = {record["source"]: record for record in result["mounts"]}
    assert str(measurement.resolve()) not in mounts
    parent = mounts[str(measurement.parent.resolve())]
    assert parent["read_only"] is False
    assert "measurement_output" in parent["purposes"]
    assert "resume_measurement_evidence_and_output" in parent["purposes"]


def test_container_mount_plan_uses_current_wandb_configuration(tmp_path):
    run_root = tmp_path / "runs"
    run_root.mkdir()
    old_usage = tmp_path / "missing-old-usage.json"
    current_usage = tmp_path / "current-usage.json"
    current_usage.write_text("fixture", encoding="utf-8")
    authority = OmegaConf.to_container(_cfg(), resolve=True)
    authority["wandb"]["artifact"]["usage_snapshot_path"] = str(old_usage)
    operational = OmegaConf.to_container(_cfg(), resolve=True)
    operational["wandb"]["artifact"]["usage_snapshot_path"] = str(current_usage)

    result = _container_mount_check(
        OmegaConf.create(authority),
        operational_cfg=OmegaConf.create(operational),
        root_dir=ROOT_DIR,
        run_root=run_root,
        executor="container",
        checkpoint_path=None,
        manifests={"data_manifests": []},
        cache={"caches": []},
    )

    mounts = {record["source"]: record for record in result["mounts"]}
    assert str(old_usage.resolve()) not in mounts
    assert mounts[str(current_usage.resolve())]["read_only"] is True


def test_container_mount_plan_rejects_unbound_missing_external_cache(tmp_path):
    missing = tmp_path / "missing-cache"

    with pytest.raises(PreflightError, match="must already exist"):
        _container_mount_check(
            _cfg(),
            root_dir=ROOT_DIR,
            run_root=tmp_path,
            executor="container",
            checkpoint_path=None,
            manifests={"data_manifests": []},
            cache={"caches": [{"path": str(missing)}]},
        )


def test_container_mount_plan_creates_missing_repository_internal_cache(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    git_dir = root / ".git"
    git_dir.mkdir(parents=True)
    tokenizer = root / "tokenizer.json"
    tokenizer.write_text("fixture", encoding="utf-8")
    internal_cache = root / "ignored" / "stream-loader-cache"
    plain = OmegaConf.to_container(_cfg(), resolve=True)
    plain["tokenizer"]["manifest_path"] = str(tokenizer)

    monkeypatch.setattr(
        preflight_module,
        "_git_absolute_path",
        lambda _root, _argument: git_dir,
    )
    monkeypatch.setattr(preflight_module, "_git_ignored", lambda _root, _path: True)

    result = _container_mount_check(
        OmegaConf.create(plain),
        root_dir=root,
        run_root=root / "runs",
        executor="container",
        checkpoint_path=None,
        manifests={"data_manifests": []},
        cache={"caches": [{"path": str(internal_cache)}]},
    )

    assert internal_cache.is_dir()
    mounts = {record["source"]: record for record in result["mounts"]}
    assert mounts[str(internal_cache.resolve())]["read_only"] is False


def test_container_mount_plan_refuses_to_create_nonignored_internal_cache(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    git_dir = root / ".git"
    git_dir.mkdir(parents=True)
    tokenizer = root / "tokenizer.json"
    tokenizer.write_text("fixture", encoding="utf-8")
    internal_cache = root / "tracked-location" / "cache"
    plain = OmegaConf.to_container(_cfg(), resolve=True)
    plain["tokenizer"]["manifest_path"] = str(tokenizer)
    monkeypatch.setattr(
        preflight_module,
        "_git_absolute_path",
        lambda _root, _argument: git_dir,
    )
    monkeypatch.setattr(preflight_module, "_git_ignored", lambda _root, _path: False)

    with pytest.raises(PreflightError, match="must be Git-ignored"):
        _container_mount_check(
            OmegaConf.create(plain),
            root_dir=root,
            run_root=root / "runs",
            executor="container",
            checkpoint_path=None,
            manifests={"data_manifests": []},
            cache={"caches": [{"path": str(internal_cache)}]},
        )

    assert not internal_cache.exists()


@pytest.mark.parametrize("writable_kind", ["cache", "run_root"])
@pytest.mark.parametrize("nested", [False, True])
def test_container_mount_plan_rejects_writable_git_metadata_overlap(
    tmp_path, monkeypatch, writable_kind, nested
):
    common_git = tmp_path / "common-git"
    git_dir = common_git / "worktrees" / "fixture"
    git_dir.mkdir(parents=True)
    writable = common_git / "nested-cache" if nested else common_git
    writable.mkdir(exist_ok=True)
    ordinary_runs = tmp_path / "ordinary-runs"
    ordinary_runs.mkdir()

    def git_path(_root, argument):
        return git_dir if argument == "--git-dir" else common_git

    monkeypatch.setattr(preflight_module, "_git_absolute_path", git_path)
    run_root = writable if writable_kind == "run_root" else ordinary_runs
    caches = [{"path": str(writable)}] if writable_kind == "cache" else []

    with pytest.raises(PreflightError, match=f"writable {writable_kind}.*Git metadata"):
        _container_mount_check(
            _cfg(),
            root_dir=ROOT_DIR,
            run_root=run_root,
            executor="container",
            checkpoint_path=None,
            manifests={"data_manifests": []},
            cache={"caches": caches},
        )


@pytest.mark.parametrize("nested", [False, True])
def test_host_cache_preflight_rejects_git_metadata_overlap(tmp_path, monkeypatch, nested):
    common_git = tmp_path / "common-git"
    git_dir = common_git / "worktrees" / "fixture"
    git_dir.mkdir(parents=True)
    cache = common_git / "nested-cache" if nested else common_git
    cache.mkdir(exist_ok=True)
    plain = OmegaConf.to_container(_cfg(), resolve=True)
    plain["data"]["streaming"] = {
        "cache": {
            "dir": str(cache),
            "max_size_bytes": 1_000_000,
            "min_free_bytes": 0,
        }
    }

    def git_path(_root, argument):
        return git_dir if argument == "--git-dir" else common_git

    monkeypatch.setattr(preflight_module, "_git_absolute_path", git_path)

    with pytest.raises(PreflightError, match="writable cache.*Git metadata"):
        preflight_module._cache_check(OmegaConf.create(plain), ROOT_DIR)
