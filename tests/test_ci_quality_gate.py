from __future__ import annotations

import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_offline_smoke():
    spec = importlib.util.spec_from_file_location(
        "offline_smoke", ROOT / "scripts/offline_smoke.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pr_workflow_exposes_each_matching_local_quality_target():
    workflow = (ROOT / ".github/workflows/pr-quality.yml").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    for target in ("ci-sync", "ci-lint", "ci-test", "ci-config", "ci-lock", "ci-offline-smoke"):
        assert f"make {target}" in workflow
        assert f"{target}:" in makefile
    assert "ci-cpu: ci-sync ci-lint ci-test ci-config ci-lock ci-offline-smoke" in makefile
    assert "--locked --no-default-groups --group dev" in makefile
    assert "--no-sync" in makefile


def test_network_integration_is_not_a_pull_request_trigger():
    workflow = (ROOT / ".github/workflows/network-integration.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "schedule:" in workflow
    assert "pull_request:" not in workflow
    assert "RUN_HF_DATASET_INTEGRATION" in workflow


def test_offline_smoke_environment_removes_credentials_and_loads_socket_guard(
    tmp_path, monkeypatch
):
    module = load_offline_smoke()
    monkeypatch.setenv("WANDB_API_KEY", "should-not-reach-smoke")
    monkeypatch.setenv("HF_TOKEN", "should-not-reach-smoke")
    monkeypatch.setenv("GITHUB_TOKEN", "should-not-reach-smoke")

    environment = module.offline_environment(tmp_path)

    assert "WANDB_API_KEY" not in environment
    assert "HF_TOKEN" not in environment
    assert "GITHUB_TOKEN" not in environment
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["HF_DATASETS_OFFLINE"] == "1"
    assert environment["UV_OFFLINE"] == "1"
    assert environment["WANDB_MODE"] == "disabled"
    assert environment["PYTHONPATH"].split(os.pathsep, 1)[0] == str(tmp_path)
    assert "_guarded_connect" in module.NETWORK_GUARD
    assert "_guarded_connect_ex" in module.NETWORK_GUARD
    assert "_guarded_sendto" in module.NETWORK_GUARD
