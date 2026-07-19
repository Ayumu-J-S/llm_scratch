from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).parents[1]


def test_installed_evaluate_command_composes_repository_config():
    command = Path(sys.executable).with_name("llm-scratch-evaluate")
    assert command.is_file(), "project console scripts must be installed before tests"
    environment = os.environ.copy()
    environment["HYDRA_FULL_ERROR"] = "1"

    result = subprocess.run(
        [str(command), "profile=evaluation", "evaluation.device=cpu"],
        cwd=ROOT_DIR,
        env=environment,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "ConfigPreflightError" in output
    assert "missing required config key(s) at evaluation: checkpoint_path" in output
    assert "MissingConfigException" not in output
    assert "Primary config module 'config' not found" not in output
