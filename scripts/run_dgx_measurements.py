#!/usr/bin/env python3
"""Plan or execute the exact DGX-001 candidate matrix in the pinned image."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dgx.planning import build_matrix_plan, training_overrides, validate_dgx_config


ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _image_id(name: str) -> str:
    return subprocess.run(
        ["docker", "image", "inspect", name, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _preflight(cfg: dict) -> tuple[str, str, Path]:
    commit = _git("rev-parse", "HEAD")
    if cfg.get("expected_commit") != commit:
        raise RuntimeError(f"expected_commit must match the exact current head: {commit}")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("DGX matrix requires a clean tracked worktree")
    image_name = str(cfg["image"]["name"])
    image_id = _image_id(image_name)
    if image_id != cfg["image"]["expected_id"]:
        raise RuntimeError(
            f"container image identity mismatch: {image_id} != {cfg['image']['expected_id']}"
        )
    if not cfg.get("output_root"):
        raise RuntimeError("output_root is required outside plan mode")
    output_root = Path(str(cfg["output_root"])).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    return commit, image_id, output_root


def _container_command(
    cfg: dict,
    entry: dict,
    *,
    commit: str,
    image_id: str,
    output_root: Path,
    pilot: bool = False,
) -> list[str]:
    run_name = (
        f"pilot-{entry['candidate_id']}"
        if pilot
        else f"{entry['candidate_id']}-r{entry['repetition']}"
    )
    run_dir = output_root / run_name
    run_dir.mkdir(parents=False, exist_ok=False)
    cache_dir = ROOT / "data" / "stream_loader_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    git_common_dir = Path(_git("rev-parse", "--git-common-dir")).resolve()
    gates = cfg["gates"]
    warmup = int(
        cfg["pilot"]["warmup_optimizer_steps"] if pilot else cfg["matrix"]["warmup_optimizer_steps"]
    )
    measured = int(cfg["matrix"]["measured_optimizer_steps"])
    overrides = training_overrides(entry, output_path="/evidence/measurement.json")
    if pilot:
        overrides = [
            item
            for item in overrides
            if not item.startswith("training.max_steps=")
            and not item.startswith("data.streaming.train.max_target_tokens=")
        ]
        overrides.extend(
            [
                "training.max_steps=null",
                f"training.max_time={int(cfg['pilot']['duration_seconds'])}",
                "data.streaming.train.max_target_tokens=1073741824",
            ]
        )
    overrides.append("data.streaming.cache.dir=/cache")
    arguments = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--ipc=host",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "--network=none",
        "--env",
        f"DGX_GIT_COMMIT={commit}",
        "--env",
        f"DGX_IMAGE_ID={image_id}",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "HF_DATASETS_OFFLINE=1",
        "--env",
        "WANDB_MODE=disabled",
        "--env",
        "WANDB_DISABLED=true",
        "--env",
        f"PYTHONPATH={ROOT / 'src'}",
        "--env",
        "GIT_CONFIG_COUNT=1",
        "--env",
        "GIT_CONFIG_KEY_0=safe.directory",
        "--env",
        f"GIT_CONFIG_VALUE_0={ROOT}",
        "--volume",
        f"{ROOT}:{ROOT}:ro",
        "--volume",
        f"{git_common_dir}:{git_common_dir}:ro",
        "--volume",
        f"{cache_dir}:/cache",
        "--volume",
        f"{run_dir}:/evidence",
        "--workdir",
        str(ROOT),
        str(cfg["image"]["name"]),
        "python",
        "scripts/measure_dgx.py",
        "--output-dir",
        "/evidence",
        "--candidate-id",
        str(entry["candidate_id"]),
        "--repetition",
        str(entry.get("repetition", 1)),
        "--warmup-optimizer-steps",
        str(warmup),
        "--measured-optimizer-steps",
        str(measured),
        "--min-available-memory-bytes",
        str(gates["min_available_memory_bytes"]),
        "--min-free-disk-bytes",
        str(gates["min_free_disk_bytes"]),
        "--max-temperature-c",
        str(gates["max_temperature_c"]),
    ]
    if pilot:
        arguments.extend(["--pilot", "--sample"])
    arguments.append("--")
    arguments.extend(overrides)
    return arguments


@hydra.main(version_base=None, config_path="../config", config_name="dgx")
def main(config: DictConfig) -> None:
    cfg = validate_dgx_config(config)
    plan = build_matrix_plan(cfg)
    if cfg.get("mode") == "plan":
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if cfg.get("mode") not in {"matrix", "pilot"}:
        raise RuntimeError("mode must be plan, matrix, or pilot")
    commit, image_id, output_root = _preflight(cfg)
    _write_json(
        output_root / "plan.json",
        {
            "config": OmegaConf.to_container(config, resolve=True),
            "git_commit": commit,
            "image_id": image_id,
            "runs": plan,
        },
    )
    if cfg["mode"] == "pilot":
        selected = str(cfg.get("selected_candidate") or "")
        matches = [entry for entry in plan if entry["candidate_id"] == selected]
        if not matches:
            raise RuntimeError("pilot mode requires a selected_candidate from the matrix")
        commands = [
            _container_command(
                cfg,
                matches[0],
                commit=commit,
                image_id=image_id,
                output_root=output_root,
                pilot=True,
            )
        ]
    else:
        commands = [
            _container_command(
                cfg,
                entry,
                commit=commit,
                image_id=image_id,
                output_root=output_root,
            )
            for entry in plan
        ]
    _write_json(output_root / "commands.json", commands)
    for command in commands:
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
