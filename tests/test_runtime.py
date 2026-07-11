from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

from runtime.device import select_device
from runtime.environment import collect_environment, unmet_requirements


def test_cpu_is_an_explicit_supported_device():
    assert select_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("requested", ["auto", "gpu", "CUDA", "cuda:0", ""])
def test_unknown_device_values_fail(requested):
    with pytest.raises(ValueError, match="exactly 'cuda' or 'cpu'"):
        select_device(requested)


def test_unavailable_cuda_fails_closed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        select_device("cuda")


def test_cuda_with_zero_visible_devices_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="zero visible devices"):
        select_device("cuda")


def test_cuda_initialization_failure_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def fail_init():
        raise RuntimeError("driver initialization failed")

    monkeypatch.setattr(torch.cuda, "init", fail_init)
    with pytest.raises(RuntimeError, match="driver initialization failed"):
        select_device("cuda")


@pytest.mark.parametrize("device_name", ["", "   "])
def test_cuda_without_readable_device_name_fails(monkeypatch, device_name):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: device_name)
    with pytest.raises(RuntimeError, match="no readable name"):
        select_device("cuda")


def test_cuda_device_name_query_failure_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "init", lambda: None)

    def fail_name_query(_index):
        raise RuntimeError("device name query failed")

    monkeypatch.setattr(torch.cuda, "get_device_name", fail_name_query)
    with pytest.raises(RuntimeError, match="device name query failed"):
        select_device("cuda")


def test_cuda_preflight_accepts_initialized_named_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "init", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA GB10")
    assert select_device("cuda") == torch.device("cuda")


def test_environment_report_is_explicit_about_host_cpu_runtime():
    report = collect_environment()
    assert report["architecture"]
    assert report["torch"]["version"]
    assert report["torch"]["module"]
    assert report["os_release"]["name"]
    assert report["os_release"]["version_id"]
    assert "unified CPU/GPU memory" in report["memory_interpretation"]
    failures = unmet_requirements(report, require_cuda=True, require_bf16=True)
    if not report["cuda"]["available"]:
        assert "CUDA is required but unavailable" in failures


def test_environment_json_cli_is_parseable():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "scripts/diagnose_environment.py", "--json"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    report = json.loads(result.stdout)
    assert report["requirements"]["passed"]
    assert report["os_release"]["name"]
