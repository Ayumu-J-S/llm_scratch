#!/usr/bin/env python3
"""Plan or execute the exact DGX-001 protocol in the pinned image."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dgx.planning import (
    build_matrix_plan,
    summarize_decomposition,
    summarize_pilot,
    training_overrides,
    validate_dgx_config,
)
from runtime.reproducibility import canonical_config_sha256, experiment_config_sha256


ROOT = Path(__file__).resolve().parent.parent
POST_PLAN_FREE_RESERVE_BYTES = 100_000_000_000
ATOMIC_WRITE_FIXED_BUDGET_BYTES = 4_000_000_000
ATOMIC_WRITE_BYTES_PER_PARAMETER = 128
PROTOCOL_CONFIG_KEYS = (
    "schema_version",
    "image",
    "matrix",
    "decomposition",
    "pilot",
    "gates",
    "selection",
)


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


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _shard_cache_identity(cache_root: Path) -> dict[str, object]:
    entries = []
    total_bytes = 0
    for path in sorted(cache_root.glob("*.shard")):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"cache shard must be one regular file: {path}")
        size = path.stat().st_size
        total_bytes += size
        entries.append({"name": path.name, "size_bytes": size, "sha256": _sha256_file(path)})
    if not entries:
        raise RuntimeError(f"cache_root has no verified *.shard files: {cache_root}")
    return {
        "root": str(cache_root),
        "files": len(entries),
        "size_bytes": total_bytes,
        "sha256": _canonical_sha256(entries),
        "entries": entries,
    }


def _source_identity() -> dict[str, object]:
    paths = [
        ROOT / "uv.lock",
        ROOT / "assets/tokenizers/llm-jp-v1/manifest.json",
        ROOT / "data/manifests/fineweb2-ja-jpn-jpan.manifest.json",
        ROOT / "data/manifests/fineweb-en-sample-10bt.manifest.json",
    ]
    files = []
    for path in paths:
        entry = {
            "path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        if path.name.endswith("manifest.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            entry["fingerprint"] = payload.get("manifest_fingerprint", payload.get("fingerprint"))
        files.append(entry)
    return {"sha256": _canonical_sha256(files), "files": files}


def _data_cache_max_bytes() -> int:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        profile = hydra.compose(config_name="train", overrides=["profile=dgx_candidate"])
    return int(profile.data.streaming.cache.max_size_bytes)


def _resolved_role_config(cfg: dict, entry: dict, role: str) -> dict:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        resolved = hydra.compose(config_name="train", overrides=_role_overrides(cfg, entry, role))
    plain = OmegaConf.to_container(resolved, resolve=True)
    if not isinstance(plain, dict):
        raise RuntimeError("DGX role configuration did not resolve to a mapping")
    return plain


def _model_parameter_count(resolved: dict) -> int:
    tokenizer = resolved.get("tokenizer", {})
    manifest_path = ROOT / str(tokenizer.get("manifest_path", ""))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != tokenizer.get("expected_fingerprint"):
        raise RuntimeError("DGX tokenizer manifest differs from the resolved authority")
    vocab_size = int(manifest["runtime"]["vocab_size"])
    embed_size = int(resolved["model"]["embed_size"])
    num_layers = int(resolved["model"]["num_layers"])
    # Exact parameter count for SimpleDecoderTransformer: token embedding and
    # untied LM head plus 12E^2 + 13E parameters per decoder block.
    return (
        2 * vocab_size * embed_size
        + vocab_size
        + num_layers * (12 * embed_size * embed_size + 13 * embed_size)
    )


def _resource_budget(resolved: dict, configured_floor_bytes: int) -> dict[str, int]:
    parameter_count = _model_parameter_count(resolved)
    # The checkpoint contains FP32 model weights (4 B/parameter) and AdamW
    # moments (8 B/parameter).  A 128 B/parameter budget plus 4 GB fixed
    # allowance conservatively covers serialization/container/config/cursor
    # overhead.  If it ever exceeds the static 20 GB buffer, the operational
    # floor rises instead of consuming the human-required 100 GB reserve.
    max_in_flight = (
        parameter_count * ATOMIC_WRITE_BYTES_PER_PARAMETER + ATOMIC_WRITE_FIXED_BUDGET_BYTES
    )
    effective_floor = max(int(configured_floor_bytes), POST_PLAN_FREE_RESERVE_BYTES + max_in_flight)
    return {
        "parameter_count": parameter_count,
        "max_in_flight_atomic_write_bytes": max_in_flight,
        "post_plan_free_reserve_bytes": POST_PLAN_FREE_RESERVE_BYTES,
        "effective_min_free_disk_bytes": effective_floor,
    }


def _role_resource_budget(cfg: dict, entry: dict, role: str) -> dict[str, int]:
    return _resource_budget(
        _resolved_role_config(cfg, entry, role), int(cfg["gates"]["min_free_disk_bytes"])
    )


def _maximum_operational_floor(cfg: dict) -> int:
    return max(
        _role_resource_budget(cfg, entry, "matrix")["effective_min_free_disk_bytes"]
        for entry in build_matrix_plan(cfg)
    )


def _filesystem_device(path: Path) -> int:
    return path.stat().st_dev


def _preflight_storage(
    *,
    output_parent: Path,
    cache_root: Path,
    cache_existing_bytes: int,
    cache_max_bytes: int,
    output_growth_bytes: int,
    minimum_free_bytes: int,
) -> None:
    allocations = (
        (output_parent, int(output_growth_bytes), "output"),
        (
            cache_root,
            max(0, int(cache_max_bytes) - int(cache_existing_bytes)),
            "cache",
        ),
    )
    filesystems: dict[int, dict] = {}
    for path, growth_bytes, role in allocations:
        device = _filesystem_device(path)
        group = filesystems.setdefault(
            device,
            {"path": path, "growth_bytes": 0, "roles": []},
        )
        group["growth_bytes"] += growth_bytes
        group["roles"].append(role)
    for group in filesystems.values():
        free_bytes = int(shutil.disk_usage(group["path"]).free)
        required_free = int(minimum_free_bytes) + int(group["growth_bytes"])
        if free_bytes < required_free:
            roles = ", ".join(sorted(group["roles"]))
            raise RuntimeError(
                f"free disk {free_bytes} for {roles} is below projected DGX preflight "
                f"{required_free}, including the {minimum_free_bytes}-byte hard floor"
            )


def _preflight(cfg: dict) -> tuple[str, str, Path, Path, dict[str, object]]:
    commit = _git("rev-parse", "HEAD")
    if cfg.get("expected_commit") != commit:
        raise RuntimeError(f"expected_commit must match the exact current head: {commit}")
    status = _git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(f"DGX evidence requires a clean worktree:\n{status}")
    image_name = str(cfg["image"]["name"])
    image_id = _image_id(image_name)
    if image_id != cfg["image"]["expected_id"]:
        raise RuntimeError(
            f"container image identity mismatch: {image_id} != {cfg['image']['expected_id']}"
        )
    if not cfg.get("output_root"):
        raise RuntimeError("output_root is required outside plan mode")
    output_root = Path(str(cfg["output_root"])).resolve()
    if output_root.exists():
        raise RuntimeError(f"DGX output_root must not already exist: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.get("cache_root"):
        raise RuntimeError("cache_root is required for offline real-data measurement")
    cache_root = Path(str(cfg["cache_root"])).resolve()
    if not cache_root.is_dir() or not os.access(cache_root, os.R_OK | os.W_OK | os.X_OK):
        raise RuntimeError(f"cache_root is not a user-readable/writable directory: {cache_root}")
    cache_identity = _shard_cache_identity(cache_root)
    reserve = int(cfg["gates"]["post_matrix_evidence_reserve_bytes"])
    if cfg["mode"] == "matrix":
        reserve = int(cfg["gates"]["matrix_output_reserve_bytes"])
    _preflight_storage(
        output_parent=output_root.parent,
        cache_root=cache_root,
        cache_existing_bytes=int(cache_identity["size_bytes"]),
        cache_max_bytes=_data_cache_max_bytes(),
        output_growth_bytes=reserve,
        minimum_free_bytes=_maximum_operational_floor(cfg),
    )
    netrc = Path.home() / ".netrc"
    if cfg["mode"] == "pilot" and not os.environ.get("WANDB_API_KEY") and not netrc.is_file():
        raise RuntimeError("selected online pilot requires WANDB_API_KEY or a host ~/.netrc")
    output_root.mkdir(mode=0o700)
    return commit, image_id, output_root, cache_root, cache_identity


def _selected_entry(cfg: dict, plan: list[dict]) -> dict:
    selected = str(cfg.get("selected_candidate") or "")
    matches = [entry for entry in plan if entry["candidate_id"] == selected]
    if not matches:
        raise RuntimeError(f"selected_candidate is not in the matrix: {selected!r}")
    entry = {key: value for key, value in matches[0].items() if key != "repetition"}
    summary_path_value = cfg.get("matrix_summary_path")
    if not summary_path_value:
        raise RuntimeError("pilot/decompose requires matrix_summary_path")
    summary_path = Path(str(summary_path_value)).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    matrix_plan_path = summary_path.parent / "plan.json"
    matrix_plan = json.loads(matrix_plan_path.read_text(encoding="utf-8"))
    unsigned_matrix_plan = dict(matrix_plan)
    matrix_plan_id = unsigned_matrix_plan.pop("plan_id", None)
    if matrix_plan_id != _canonical_sha256(unsigned_matrix_plan):
        raise RuntimeError("matrix selection authority has an invalid immutable plan hash")
    active_protocol = {key: cfg[key] for key in PROTOCOL_CONFIG_KEYS}
    matrix_protocol = {key: matrix_plan.get("config", {}).get(key) for key in PROTOCOL_CONFIG_KEYS}
    current_commit = _git("rev-parse", "HEAD")
    if (
        summary.get("schema_version") != 3
        or summary.get("verdict") != "PASS"
        or summary.get("plan_id") != matrix_plan_id
        or summary.get("git_commit") != current_commit
        or summary.get("image_id") != cfg["image"]["expected_id"]
        or summary.get("selection_rule") != cfg["selection"]
        or summary.get("selected", {}).get("candidate_id") != selected
        or matrix_plan.get("git_commit") != current_commit
        or matrix_plan.get("image_id") != cfg["image"]["expected_id"]
        or matrix_plan.get("schema_version") != 3
        or matrix_plan.get("ticket") != "DGX-001"
        or matrix_plan.get("config", {}).get("mode") != "matrix"
        or matrix_protocol != active_protocol
        or matrix_plan.get("runs") != plan
        or matrix_plan.get("selected") is not None
        or matrix_plan.get("matrix_summary_identity") is not None
    ):
        raise RuntimeError(
            "selected_candidate is not authorized by the exact passing matrix plan/summary"
        )
    summary_shape = {
        key: int(summary["selected"][key])
        for key in (
            "num_layers",
            "embed_size",
            "num_heads",
            "sequence_length",
            "batch_size",
            "gradient_accumulation_steps",
        )
    }
    if summary_shape != {key: int(entry[key]) for key in summary_shape}:
        raise RuntimeError("matrix summary selected shape differs from the declared matrix")
    cfg["matrix_summary_identity"] = {
        "path": str(summary_path),
        "sha256": _sha256_file(summary_path),
        "matrix_plan_path": str(matrix_plan_path),
        "matrix_plan_sha256": _sha256_file(matrix_plan_path),
        "matrix_plan_id": matrix_plan_id,
        "matrix_protocol_sha256": _canonical_sha256(matrix_protocol),
        "git_commit": current_commit,
        "image_id": cfg["image"]["expected_id"],
        "selection_rule": cfg["selection"],
        "selected": selected,
        "end_to_end_tokens_per_second": summary["selected"]["conservative_tokens_per_second"],
    }
    return entry


def _validate_committed_selected_profile(entry: dict) -> None:
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        profile = hydra.compose(config_name="train", overrides=["profile=pretrain_baseline"])
    observed = {
        "num_layers": int(profile.model.num_layers),
        "embed_size": int(profile.model.embed_size),
        "num_heads": int(profile.model.num_heads),
        "sequence_length": int(profile.training.sequence_length),
        "batch_size": int(profile.training.batch_size),
        "gradient_accumulation_steps": int(profile.training.gradient_accumulation_steps),
    }
    expected = {key: int(entry[key]) for key in observed}
    if observed != expected:
        raise RuntimeError(
            f"committed pretrain_baseline shape is not the selected candidate: {observed} != {expected}"
        )


def _pilot_overrides(cfg: dict, entry: dict) -> list[str]:
    pilot = cfg["pilot"]
    return [
        "profile=pretrain_baseline",
        f"model.num_layers={entry['num_layers']}",
        f"model.embed_size={entry['embed_size']}",
        f"model.num_heads={entry['num_heads']}",
        f"training.sequence_length={entry['sequence_length']}",
        f"training.batch_size={entry['batch_size']}",
        f"training.gradient_accumulation_steps={entry['gradient_accumulation_steps']}",
        "training.max_steps=null",
        f"training.max_time={int(pilot['duration_seconds'])}",
        "data.streaming.train.max_target_tokens=1073741824",
        f"data.streaming.validation.max_target_tokens={int(pilot['validation_target_tokens'])}",
        f"training.validation_every_n_tokens={int(pilot['validation_every_target_tokens'])}",
        f"training.checkpoint_every_n_tokens={int(pilot['recovery_every_target_tokens'])}",
        f"training.milestone_every_n_tokens={int(pilot['milestone_every_target_tokens'])}",
        f"training.log_every_n_steps={int(pilot['log_every_optimizer_steps'])}",
        "measurement.enabled=true",
        f"measurement.warmup_optimizer_steps={int(pilot['warmup_optimizer_steps'])}",
        "measurement.output_path=/evidence/measurement.json",
        "wandb.mode=online",
        "wandb.watch.enabled=false",
        "wandb.artifact.policy=none",
    ]


def _authority_key(role: str, entry: dict) -> str:
    return f"{role}:{entry['candidate_id']}:r{int(entry.get('repetition', 1))}"


def _role_overrides(cfg: dict, entry: dict, role: str) -> list[str]:
    if role == "matrix":
        return [
            *training_overrides(entry, output_path="/evidence/measurement.json"),
            "data.streaming.cache.dir=/cache",
        ]
    overrides = _pilot_overrides(cfg, entry)
    if role in {"model-only", "loader-only"}:
        overrides = [
            item
            for item in overrides
            if not item.startswith("training.max_time=")
            and not item.startswith("measurement.output_path=")
            and not item.startswith("wandb.mode=")
        ]
        overrides.extend(
            [
                "training.max_time=null",
                "measurement.enabled=false",
                "wandb.mode=disabled",
                "wandb.watch.enabled=false",
                "wandb.artifact.policy=none",
            ]
        )
    overrides.append("data.streaming.cache.dir=/cache")
    return overrides


def _run_config_authorities(cfg: dict, roles: list[tuple[dict, str]]) -> list[dict]:
    authorities = []
    for entry, role in roles:
        plain = _resolved_role_config(cfg, entry, role)
        resource_budget = _resource_budget(plain, int(cfg["gates"]["min_free_disk_bytes"]))
        authorities.append(
            {
                "authority_key": _authority_key(role, entry),
                "role": role,
                "candidate_id": entry["candidate_id"],
                "repetition": int(entry.get("repetition", 1)),
                "canonical_config_sha256": canonical_config_sha256(plain),
                "experiment_config_sha256": experiment_config_sha256(plain),
                **resource_budget,
            }
        )
    return authorities


def _container_command(
    cfg: dict,
    entry: dict,
    *,
    commit: str,
    image_id: str,
    plan_id: str,
    output_root: Path,
    cache_root: Path,
    role: str,
) -> list[str]:
    repetition = int(entry.get("repetition", 1))
    resource_budget = _role_resource_budget(cfg, entry, role)
    if role == "matrix":
        run_name = f"{entry['candidate_id']}-r{repetition}"
    elif role == "pilot":
        run_name = f"pilot-{entry['candidate_id']}"
    else:
        run_name = f"{role}-{entry['candidate_id']}-r{repetition}"
    run_dir = output_root / run_name
    run_dir.mkdir(mode=0o700)
    (run_dir / "home").mkdir(mode=0o700)
    (run_dir / "wandb").mkdir(mode=0o700)
    git_common_dir = Path(_git("rev-parse", "--git-common-dir")).resolve()
    gates = cfg["gates"]
    uid_gid = f"{os.getuid()}:{os.getgid()}"
    common = [
        "docker",
        "run",
        "--rm",
        "--user",
        uid_gid,
        "--gpus",
        "all",
        "--ipc=host",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
    ]
    if role != "pilot":
        common.extend(["--network=none"])
    common.extend(
        [
            "--env",
            f"DGX_GIT_COMMIT={commit}",
            "--env",
            f"DGX_IMAGE_ID={image_id}",
            "--env",
            "HF_HUB_OFFLINE=1",
            "--env",
            "HF_DATASETS_OFFLINE=1",
            "--env",
            f"PYTHONPATH={ROOT / 'src'}",
            "--env",
            "HOME=/evidence/home",
            "--env",
            "WANDB_DIR=/evidence/wandb",
            "--env",
            "GIT_CONFIG_COUNT=1",
            "--env",
            "GIT_CONFIG_KEY_0=safe.directory",
            "--env",
            f"GIT_CONFIG_VALUE_0={ROOT}",
        ]
    )
    if role == "pilot":
        if os.environ.get("WANDB_API_KEY"):
            common.extend(["--env", "WANDB_API_KEY"])
        else:
            common.extend(["--volume", f"{Path.home() / '.netrc'}:/evidence/home/.netrc:ro"])
        common.extend(["--env", "WANDB_MODE=online"])
    else:
        common.extend(["--env", "WANDB_MODE=disabled", "--env", "WANDB_DISABLED=true"])
    common.extend(
        [
            "--volume",
            f"{ROOT}:{ROOT}:ro",
            "--volume",
            f"{git_common_dir}:{git_common_dir}:ro",
            "--volume",
            f"{cache_root}:/cache",
            "--volume",
            f"{run_dir}:/evidence",
            "--workdir",
            str(ROOT),
            image_id,
            "python",
        ]
    )
    if role in {"model-only", "loader-only"}:
        overrides = _role_overrides(cfg, entry, role)
        return common + [
            "scripts/measure_dgx_decomposition.py",
            "--output-dir",
            "/evidence",
            "--role",
            role,
            "--candidate-id",
            str(entry["candidate_id"]),
            "--repetition",
            str(repetition),
            "--git-commit",
            commit,
            "--image-id",
            image_id,
            "--plan-id",
            plan_id,
            "--warmup-optimizer-steps",
            str(cfg["decomposition"]["warmup_optimizer_steps"]),
            "--measured-optimizer-steps",
            str(cfg["decomposition"]["measured_optimizer_steps"]),
            "--min-available-memory-bytes",
            str(gates["min_available_memory_bytes"]),
            "--min-free-disk-bytes",
            str(gates["min_free_disk_bytes"]),
            "--post-plan-free-reserve-bytes",
            str(resource_budget["post_plan_free_reserve_bytes"]),
            "--max-in-flight-atomic-write-bytes",
            str(resource_budget["max_in_flight_atomic_write_bytes"]),
            "--max-temperature-c",
            str(gates["max_temperature_c"]),
            "--max-swap-in-pages",
            str(gates["max_swap_in_pages"]),
            "--max-swap-out-pages",
            str(gates["max_swap_out_pages"]),
            "--",
            *overrides,
        ]
    if role == "pilot":
        overrides = _role_overrides(cfg, entry, role)
        pilot_flag = ["--pilot", "--sample"]
        warmup = int(cfg["pilot"]["warmup_optimizer_steps"])
    else:
        overrides = _role_overrides(cfg, entry, role)
        pilot_flag = []
        warmup = int(cfg["matrix"]["warmup_optimizer_steps"])
    return common + [
        "scripts/measure_dgx.py",
        "--output-dir",
        "/evidence",
        "--candidate-id",
        str(entry["candidate_id"]),
        "--repetition",
        str(repetition),
        "--git-commit",
        commit,
        "--image-id",
        image_id,
        "--plan-id",
        plan_id,
        "--role",
        role,
        "--warmup-optimizer-steps",
        str(warmup),
        "--measured-optimizer-steps",
        str(cfg["matrix"]["measured_optimizer_steps"]),
        "--min-available-memory-bytes",
        str(gates["min_available_memory_bytes"]),
        "--min-free-disk-bytes",
        str(gates["min_free_disk_bytes"]),
        "--post-plan-free-reserve-bytes",
        str(resource_budget["post_plan_free_reserve_bytes"]),
        "--max-in-flight-atomic-write-bytes",
        str(resource_budget["max_in_flight_atomic_write_bytes"]),
        "--max-temperature-c",
        str(gates["max_temperature_c"]),
        "--max-swap-in-pages",
        str(gates["max_swap_in_pages"]),
        "--max-swap-out-pages",
        str(gates["max_swap_out_pages"]),
        *pilot_flag,
        "--",
        *overrides,
    ]


def _run_commands(commands: list[list[str]], output_root: Path) -> int:
    attempts = []
    return_code = 0
    for index, command in enumerate(commands):
        evidence_mount = next(item for item in command if item.endswith(":/evidence"))
        run_dir = Path(evidence_mount.removesuffix(":/evidence"))
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        started = time.time()
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            result = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        attempt = {
            "index": index,
            "run_dir": str(run_dir),
            "started_unix_seconds": started,
            "ended_unix_seconds": time.time(),
            "return_code": result.returncode,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        attempts.append(attempt)
        _write_json(output_root / "execution.json", {"attempts": attempts, "complete": False})
        if result.returncode != 0:
            return_code = result.returncode
            break
    _write_json(
        output_root / "execution.json",
        {"attempts": attempts, "complete": return_code == 0, "return_code": return_code},
    )
    return return_code


@hydra.main(version_base=None, config_path="../config", config_name="dgx")
def main(config: DictConfig) -> None:
    cfg = validate_dgx_config(config)
    matrix_plan = build_matrix_plan(cfg)
    if cfg.get("mode") == "plan":
        print(json.dumps(matrix_plan, indent=2, sort_keys=True))
        return
    if cfg.get("mode") not in {"matrix", "decompose", "pilot"}:
        raise RuntimeError("mode must be plan, matrix, decompose, or pilot")
    commit, image_id, output_root, cache_root, cache_before = _preflight(cfg)
    selected = None
    if cfg["mode"] in {"decompose", "pilot"}:
        selected = _selected_entry(cfg, matrix_plan)
        _validate_committed_selected_profile(selected)
    if cfg["mode"] == "matrix":
        roles = [(entry, "matrix") for entry in matrix_plan]
    elif cfg["mode"] == "decompose":
        assert selected is not None
        roles = [
            ({**selected, "repetition": repetition}, role)
            for repetition in range(1, int(cfg["decomposition"]["repetitions"]) + 1)
            for role in ("model-only", "loader-only")
        ]
    else:
        assert selected is not None
        roles = [(selected, "pilot")]
    source_identity = _source_identity()
    plan_payload = {
        "schema_version": 3,
        "ticket": "DGX-001",
        "config": OmegaConf.to_container(config, resolve=True),
        "git_commit": commit,
        "image_id": image_id,
        "host_user": {"uid": os.getuid(), "gid": os.getgid()},
        "source_identity": source_identity,
        "cache_before": cache_before,
        "data_cache_max_bytes": _data_cache_max_bytes(),
        "runs": matrix_plan,
        "run_config_authorities": _run_config_authorities(cfg, roles),
        "selected": selected,
        "matrix_summary_identity": cfg.get("matrix_summary_identity"),
    }
    plan_id = _canonical_sha256(plan_payload)
    plan_payload["plan_id"] = plan_id
    _write_json(output_root / "plan.json", plan_payload)
    commands = [
        _container_command(
            cfg,
            entry,
            commit=commit,
            image_id=image_id,
            plan_id=plan_id,
            output_root=output_root,
            cache_root=cache_root,
            role=role,
        )
        for entry, role in roles
    ]
    _write_json(output_root / "commands.json", commands)
    return_code = _run_commands(commands, output_root)
    cache_after = _shard_cache_identity(cache_root)
    cache_unchanged = cache_after == cache_before
    _write_json(
        output_root / "cache-integrity.json",
        {"before": cache_before, "after": cache_after, "unchanged": cache_unchanged},
    )
    if return_code != 0:
        raise SystemExit(return_code)
    if not cache_unchanged:
        raise RuntimeError("offline DGX measurement changed the verified *.shard cache")
    if cfg["mode"] == "decompose":
        summary = summarize_decomposition(output_root, cfg)
        _write_json(output_root / "dgx-decomposition-summary.json", summary)
        if summary["verdict"] != "PASS":
            raise SystemExit(1)
    elif cfg["mode"] == "pilot":
        summary = summarize_pilot(output_root, cfg)
        _write_json(output_root / "dgx-pilot-summary.json", summary)
        if summary["verdict"] != "PASS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
