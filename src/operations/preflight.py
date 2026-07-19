"""Non-destructive configuration, identity, storage, GPU, and W&B preflight."""

from __future__ import annotations

import json
import netrc
import os
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from operations.artifacts import sha256_file, utc_now
from runtime.config import (
    validate_benchmark_config,
    validate_evaluation_config,
    validate_training_config,
)
from tokenizer.canonical import CanonicalTokenizer
from train import validate_profile_manifests
from training.wandb_tracking import load_usage_snapshot
from training.checkpoint import (
    checkpoint_config_sha256,
    load_checkpoint_for_generation,
    require_exact_stream_resume_state,
)


LIVE_DISK_FLOOR_BYTES = 120_000_000_000
POST_PLAN_RESERVE_BYTES = 100_000_000_000
CHECKPOINT_BYTES_PER_PARAMETER = 128
CHECKPOINT_FIXED_HEADROOM_BYTES = 4_000_000_000
LOG_HEADROOM_BYTES = 1_000_000_000


class PreflightError(RuntimeError):
    """A required local or launch-specific precondition did not pass."""


def run_preflight(
    cfg: DictConfig,
    *,
    root_dir: Path,
    run_root: Path,
    action: str,
    executor: str,
    device: str,
    image: str | None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Run every local check even if an online-only W&B gate fails."""

    checks: dict[str, Any] = {}
    local_errors: list[str] = []
    online_errors: list[str] = []
    authority_cfg: Mapping[str, Any] | DictConfig = cfg
    if checkpoint_path is not None:
        try:
            checkpoint = load_checkpoint_for_generation(checkpoint_path)
            checkpoint_cfg = _mapping(
                checkpoint.payload["state"].get("resolved_config"),
                "checkpoint resolved_config",
            )
            expected_config = checkpoint.payload["identity"].get("config_sha256")
            if checkpoint_config_sha256(checkpoint_cfg) != expected_config:
                raise PreflightError("checkpoint-owned config does not match checkpoint identity")
            if action == "resume" and checkpoint_config_sha256(cfg) != expected_config:
                raise PreflightError(
                    "resume Hydra config differs from checkpoint experiment identity"
                )
            if action == "resume":
                require_exact_stream_resume_state(checkpoint.payload["state"])
            authority_cfg = checkpoint_cfg
            checks["checkpoint"] = {
                "status": "passed",
                "physical_identity": checkpoint.physical_identity,
                "kind": checkpoint.payload["kind"],
                "config_sha256": expected_config,
            }
        except Exception as error:
            checks["checkpoint"] = {"status": "failed", "error": str(error)}
            local_errors.append(f"checkpoint: {error}")

    for name, function in (
        ("git", lambda: _git_check(root_dir)),
        ("configuration", lambda: _configuration_check(cfg)),
        ("manifests", lambda: _manifest_check(authority_cfg)),
        ("cache", lambda: _cache_check(authority_cfg, root_dir)),
        (
            "storage",
            lambda: _storage_check(
                authority_cfg,
                root_dir=root_dir,
                run_root=run_root,
                action=action,
            ),
        ),
        (
            "container_mounts",
            lambda: _container_mount_check(
                authority_cfg,
                root_dir=root_dir,
                run_root=run_root,
                executor=executor,
                checkpoint_path=checkpoint_path,
                manifests=checks.get("manifests", {}),
                cache=checks.get("cache", {}),
            ),
        ),
        (
            "device",
            lambda: _device_check(
                root_dir=root_dir,
                executor=executor,
                device=device,
                image=image,
                mounts=checks.get("container_mounts", {}).get("mounts", []),
            ),
        ),
    ):
        try:
            checks[name] = {"status": "passed", **function()}
        except Exception as error:
            message = f"{name}: {error}"
            checks[name] = {"status": "failed", "error": str(error)}
            local_errors.append(message)

    git_record = checks.get("git", {})
    profile_purpose = str(cfg.profile.get("purpose", ""))
    clean_required = action in {"train", "resume", "eval", "benchmark"} or (
        action == "preflight" and profile_purpose != "memorization_smoke"
    )
    if clean_required and git_record.get("dirty") is True:
        message = f"git: action {action} requires a clean worktree"
        git_record["status"] = "failed"
        git_record["error"] = message
        local_errors.append(message)

    try:
        wandb = _wandb_check(cfg, executor=executor)
        checks["wandb"] = wandb
        if wandb["status"] == "blocked_online":
            online_errors.extend(str(item) for item in wandb["blocking_reasons"])
    except Exception as error:
        checks["wandb"] = {"status": "failed", "error": str(error)}
        local_errors.append(f"wandb: {error}")

    online_action = action in {"preflight", "train", "resume", "eval", "benchmark"} and (
        _wandb_mode(cfg) == "online"
    )
    ready = not local_errors and (not online_action or not online_errors)
    return {
        "schema_version": 1,
        "checked_at": utc_now(),
        "action": action,
        "executor": executor,
        "device": device,
        "ready": ready,
        "local_checks_complete": True,
        "local_errors": local_errors,
        "online_errors": online_errors,
        "login_prompt": (
            (
                "Export WANDB_API_KEY and repeat preflight; all local checks completed."
                if executor == "container"
                else "Run `wandb login` and repeat preflight; all local checks completed."
            )
            if any("credential" in error for error in online_errors)
            else None
        ),
        "checks": checks,
    }


def require_ready(report: Mapping[str, Any]) -> None:
    if report.get("ready") is not True:
        errors = list(report.get("local_errors", [])) + list(report.get("online_errors", []))
        raise PreflightError("preflight blocked launch: " + "; ".join(map(str, errors)))


def estimate_parameter_count(cfg: Mapping[str, Any] | DictConfig) -> int:
    plain = _plain(cfg)
    model = _mapping(plain.get("model"), "model")
    tokenizer_cfg = _mapping(plain.get("tokenizer"), "tokenizer")
    manifest_path = Path(str(tokenizer_cfg.get("manifest_path", "")))
    if not manifest_path.is_absolute():
        manifest_path = Path(__file__).resolve().parents[2] / manifest_path
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        vocab_size = int(
            _mapping(manifest.get("runtime"), "tokenizer manifest runtime")["vocab_size"]
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise PreflightError(f"cannot derive tokenizer vocabulary size: {manifest_path}") from error
    embed = _positive_int(model.get("embed_size"), "model.embed_size")
    layers = _positive_int(model.get("num_layers"), "model.num_layers")
    # Untied token embedding and LM head, then the exact conventional decoder
    # block used by SimpleDecoderTransformer (MHA, two norms, 4x FFN).
    return 2 * vocab_size * embed + vocab_size + layers * (12 * embed * embed + 13 * embed)


def _configuration_check(cfg: DictConfig) -> dict[str, Any]:
    profile = str(cfg.profile.get("purpose", ""))
    if profile == "evaluation":
        validate_evaluation_config(cfg)
    elif profile == "benchmark":
        validate_benchmark_config(cfg)
    else:
        validate_training_config(cfg)
    rendered = OmegaConf.to_yaml(cfg, resolve=True)
    return {
        "profile": str(cfg.profile.name),
        "resolved_sha256": _sha256_bytes(rendered.encode("utf-8")),
    }


def _manifest_check(cfg: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    plain = _plain(cfg)
    profile = _mapping(plain.get("profile"), "profile")
    purpose = str(profile.get("purpose", ""))
    if purpose not in {"evaluation", "benchmark"}:
        validate_profile_manifests(OmegaConf.create(plain))
    tokenizer_cfg = _mapping(plain.get("tokenizer"), "tokenizer")
    tokenizer = CanonicalTokenizer.from_config(tokenizer_cfg)
    manifests: list[dict[str, Any]] = []
    data = plain.get("data", {})
    if isinstance(data, Mapping) and data.get("mode") == "memorization_smoke":
        smoke = data.get("memorization", {})
        if isinstance(smoke, Mapping):
            path = _rooted(str(smoke["manifest_path"]), Path(__file__).resolve().parents[2])
            manifests.append(
                {
                    "split": "memorization",
                    "name": "memorization",
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "fingerprint": smoke.get("expected_fingerprint"),
                }
            )
    streaming = data.get("streaming", {}) if isinstance(data, Mapping) else {}
    if isinstance(streaming, Mapping):
        for split in ("train", "validation"):
            split_cfg = streaming.get(split, {})
            if not isinstance(split_cfg, Mapping):
                continue
            for source in split_cfg.get("sources", split_cfg.get("datasets", [])):
                if not isinstance(source, Mapping) or source.get("type", "hf") != "manifest":
                    continue
                path = _rooted(str(source["manifest_path"]), Path(__file__).resolve().parents[2])
                manifests.append(
                    {
                        "split": split,
                        "name": source.get("name"),
                        "path": str(path),
                        "sha256": sha256_file(path),
                        "fingerprint": source.get("expected_fingerprint"),
                    }
                )
    return {
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "data_manifests": manifests,
    }


def _git_check(root_dir: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root_dir, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()

    status = [
        line for line in git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    ]
    return {
        "sha": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(status),
        "worktree_status": status,
        "lock_sha256": sha256_file(root_dir / "uv.lock"),
    }


def _cache_check(cfg: Mapping[str, Any] | DictConfig, root_dir: Path) -> dict[str, Any]:
    caches = _cache_configs(cfg)
    records = []
    for cache in caches:
        path = _rooted(str(cache["dir"]), root_dir)
        reject_writable_git_overlap(root_dir, path, purpose="cache")
        records.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "max_size_bytes": int(cache["max_size_bytes"]),
                "current_size_bytes": _directory_size(path),
                "configured_min_free_bytes": int(cache.get("min_free_bytes", 0)),
                "filesystem_free_bytes": shutil.disk_usage(_existing_ancestor(path)).free,
            }
        )
    return {"caches": records}


def _storage_check(
    cfg: Mapping[str, Any] | DictConfig,
    *,
    root_dir: Path,
    run_root: Path,
    action: str,
) -> dict[str, Any]:
    parameter_count = estimate_parameter_count(cfg)
    maximum_atomic_write = (
        parameter_count * CHECKPOINT_BYTES_PER_PARAMETER + CHECKPOINT_FIXED_HEADROOM_BYTES
    )
    plain = _plain(cfg)
    artifacts = plain.get("artifacts", {})
    keep_last = int(artifacts.get("keep_last_n", 0)) if isinstance(artifacts, Mapping) else 0
    # Recovery rotation + best/final/milestone + one atomic temporary write.
    checkpoint_plan = (
        maximum_atomic_write * (keep_last + 4)
        if action
        in {
            "smoke",
            "train",
            "resume",
            "preflight",
        }
        else 0
    )
    projected_by_device: defaultdict[int, int] = defaultdict(int)
    locations: dict[int, Path] = {}

    run_ancestor = _existing_ancestor(run_root)
    run_device = run_ancestor.stat().st_dev
    projected_by_device[run_device] += checkpoint_plan + LOG_HEADROOM_BYTES
    locations[run_device] = run_ancestor
    configured_floor_by_device: defaultdict[int, int] = defaultdict(int)

    unique_cache_growth: dict[Path, tuple[int, int]] = {}
    for cache in _cache_configs(cfg):
        cache_path = _rooted(str(cache["dir"]), root_dir)
        current_size = _directory_size(cache_path)
        remaining = max(0, int(cache["max_size_bytes"]) - current_size)
        configured_floor = int(cache.get("min_free_bytes", 0))
        prior_remaining, prior_floor = unique_cache_growth.get(cache_path, (0, 0))
        unique_cache_growth[cache_path] = (
            max(prior_remaining, remaining),
            max(prior_floor, configured_floor),
        )
    for cache_path, (remaining, configured_floor) in unique_cache_growth.items():
        ancestor = _existing_ancestor(cache_path)
        device_id = ancestor.stat().st_dev
        projected_by_device[device_id] += remaining
        configured_floor_by_device[device_id] = max(
            configured_floor_by_device[device_id], configured_floor
        )
        locations[device_id] = ancestor

    effective_live_floor = max(
        LIVE_DISK_FLOOR_BYTES,
        POST_PLAN_RESERVE_BYTES + maximum_atomic_write,
        configured_floor_by_device.get(run_device, 0),
    )
    filesystems = []
    for device_id, planned_bytes in sorted(projected_by_device.items()):
        free = shutil.disk_usage(locations[device_id]).free
        post_plan = free - planned_bytes
        live_floor = max(
            LIVE_DISK_FLOOR_BYTES,
            configured_floor_by_device.get(device_id, 0),
            POST_PLAN_RESERVE_BYTES + maximum_atomic_write
            if device_id == run_device
            else POST_PLAN_RESERVE_BYTES,
        )
        record = {
            "device": device_id,
            "path": str(locations[device_id]),
            "free_bytes": free,
            "projected_additional_bytes": planned_bytes,
            "projected_post_plan_free_bytes": post_plan,
            "effective_live_floor_bytes": live_floor,
            "post_plan_reserve_bytes": POST_PLAN_RESERVE_BYTES,
        }
        filesystems.append(record)
        if free < live_floor:
            raise PreflightError(
                f"filesystem {locations[device_id]} has {free} bytes free; "
                f"requires live floor {live_floor}"
            )
        if post_plan < POST_PLAN_RESERVE_BYTES:
            raise PreflightError(
                f"filesystem {locations[device_id]} projects {post_plan} bytes free; "
                f"requires {POST_PLAN_RESERVE_BYTES} post-plan reserve"
            )
    return {
        "parameter_count": parameter_count,
        "maximum_atomic_write_bytes": maximum_atomic_write,
        "checkpoint_plan_bytes": checkpoint_plan,
        "log_headroom_bytes": LOG_HEADROOM_BYTES,
        "effective_run_live_floor_bytes": effective_live_floor,
        "filesystems": filesystems,
    }


def _container_mount_check(
    cfg: Mapping[str, Any] | DictConfig,
    *,
    root_dir: Path,
    run_root: Path,
    executor: str,
    checkpoint_path: Path | None,
    manifests: Any,
    cache: Any,
) -> dict[str, Any]:
    if executor != "container":
        return {"required": False, "mounts": []}

    records: dict[str, dict[str, Any]] = {}
    protected_git_paths: list[Path] = []

    def add(path: Path, *, read_only: bool, purpose: str, require_directory: bool) -> None:
        destination = path.expanduser().resolve()
        if not read_only and purpose in {"run_root", "cache"} and any(
            destination == protected or destination.is_relative_to(protected)
            for protected in protected_git_paths
        ):
            raise PreflightError(
                f"container writable {purpose} path overlaps protected Git metadata: "
                f"{destination}"
            )
        if not destination.exists():
            kind = "directory" if require_directory else "file"
            raise PreflightError(f"container {purpose} {kind} must already exist: {destination}")
        if require_directory and not destination.is_dir():
            raise PreflightError(f"container {purpose} path is not a directory: {destination}")
        if not require_directory and not destination.is_file():
            raise PreflightError(f"container {purpose} path is not a regular file: {destination}")
        key = str(destination)
        existing = records.get(key)
        if existing is None:
            records[key] = {
                "source": key,
                "destination": key,
                "read_only": read_only,
                "kind": "directory" if require_directory else "file",
                "purposes": [purpose],
            }
            return
        if existing["kind"] != ("directory" if require_directory else "file"):
            raise PreflightError(f"container mount kind conflict at {destination}")
        if existing["read_only"] is not read_only:
            raise PreflightError(f"container mount access conflict at {destination}")
        if purpose not in existing["purposes"]:
            existing["purposes"].append(purpose)

    root = root_dir.expanduser().resolve()
    runs = run_root.expanduser().resolve()
    add(root, read_only=False, purpose="repository", require_directory=True)
    git_marker, git_dir, common_git = _git_metadata_paths(root)
    protected_git_paths.extend(dict.fromkeys((git_marker, git_dir, common_git)))
    if git_marker.is_file():
        add(
            git_marker,
            read_only=True,
            purpose="git_worktree_pointer",
            require_directory=False,
        )
    add(git_dir, read_only=True, purpose="git_worktree_metadata", require_directory=True)
    add(common_git, read_only=True, purpose="git_metadata", require_directory=True)
    if not runs.is_relative_to(root):
        add(runs, read_only=False, purpose="run_root", require_directory=True)

    cache_record = cache if isinstance(cache, Mapping) else {}
    for item in cache_record.get("caches", []):
        if isinstance(item, Mapping):
            add(
                Path(str(item["path"])),
                read_only=False,
                purpose="cache",
                require_directory=True,
            )

    manifest_record = manifests if isinstance(manifests, Mapping) else {}
    plain = _plain(cfg)
    tokenizer_cfg = _mapping(plain.get("tokenizer"), "tokenizer")
    add(
        _rooted(str(tokenizer_cfg["manifest_path"]), root),
        read_only=True,
        purpose="tokenizer_manifest",
        require_directory=False,
    )
    for item in manifest_record.get("data_manifests", []):
        if isinstance(item, Mapping):
            add(
                Path(str(item["path"])),
                read_only=True,
                purpose="data_manifest",
                require_directory=False,
            )

    wandb_cfg = _wandb_configuration(OmegaConf.create(plain))
    artifact_cfg = wandb_cfg.get("artifact", {})
    if isinstance(artifact_cfg, Mapping) and artifact_cfg.get("usage_snapshot_path"):
        add(
            _rooted(str(artifact_cfg["usage_snapshot_path"]), root),
            read_only=True,
            purpose="wandb_usage_snapshot",
            require_directory=False,
        )
    if checkpoint_path is not None:
        add(
            checkpoint_path,
            read_only=True,
            purpose="checkpoint_input",
            require_directory=False,
        )

    mounts = sorted(
        records.values(),
        key=lambda item: (len(Path(str(item["destination"])).parts), str(item["destination"])),
    )
    return {"required": True, "mounts": mounts}


def reject_writable_git_overlap(root_dir: Path, path: Path, *, purpose: str) -> None:
    """Reject a writable operator path before it can create files in Git metadata."""

    candidate = path.expanduser().resolve()
    protected = _git_metadata_paths(root_dir.expanduser().resolve())
    if any(candidate == item or candidate.is_relative_to(item) for item in protected):
        raise PreflightError(
            f"writable {purpose} path overlaps protected Git metadata: {candidate}"
        )


def _git_metadata_paths(root_dir: Path) -> tuple[Path, Path, Path]:
    return (
        (root_dir / ".git").resolve(),
        _git_absolute_path(root_dir, "--git-dir"),
        _git_absolute_path(root_dir, "--git-common-dir"),
    )


def _git_absolute_path(root_dir: Path, argument: str) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", argument],
        cwd=root_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def _docker_mount_flags(mounts: Any) -> list[str]:
    if not isinstance(mounts, list):
        raise PreflightError("container mount plan must be a list")
    flags: list[str] = []
    for item in mounts:
        record = _mapping(item, "container mount")
        source = Path(str(record.get("source", "")))
        destination = Path(str(record.get("destination", "")))
        if not source.is_absolute() or not destination.is_absolute():
            raise PreflightError("container mounts require absolute source and destination paths")
        specification = f"type=bind,src={source},dst={destination}"
        if record.get("read_only") is True:
            specification += ",readonly"
        flags.extend(["--mount", specification])
    return flags


def _device_check(
    *,
    root_dir: Path,
    executor: str,
    device: str,
    image: str | None,
    mounts: Any = (),
) -> dict[str, Any]:
    if executor == "host":
        if device == "cuda":
            if not torch.cuda.is_available():
                raise PreflightError("CUDA was explicitly requested but is unavailable on host")
            if not torch.cuda.is_bf16_supported():
                raise PreflightError("explicit host CUDA device does not support BF16")
        return {
            "selected": device,
            "cuda_available": torch.cuda.is_available(),
            "bf16_supported": (
                torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
            ),
        }
    if executor != "container" or not image:
        raise PreflightError("executor must be explicit host or container with an image")
    command = ["docker", "image", "inspect", "--format", "{{.Id}}", image]
    image_result = subprocess.run(command, cwd=root_dir, capture_output=True, text=True)
    if image_result.returncode != 0 or not image_result.stdout.strip().startswith("sha256:"):
        raise PreflightError(f"container image is unavailable or unpinned locally: {image}")
    image_id = image_result.stdout.strip()
    mount_flags = _docker_mount_flags(mounts)
    git_probe = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            *mount_flags,
            "--workdir",
            str(root_dir),
            "--env",
            "GIT_OPTIONAL_LOCKS=0",
            "--entrypoint",
            "git",
            image_id,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        cwd=root_dir,
        capture_output=True,
        text=True,
    )
    if git_probe.returncode != 0:
        raise PreflightError(
            "container cannot inspect the exact Git worktree: " + git_probe.stderr.strip()
        )
    if device == "cuda":
        diagnostic = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--gpus",
                "all",
                *mount_flags,
                "--workdir",
                str(root_dir),
                "--entrypoint",
                "python",
                image_id,
                "scripts/diagnose_environment.py",
                "--json",
                "--require-cuda",
                "--require-bf16",
            ],
            cwd=root_dir,
            capture_output=True,
            text=True,
        )
        if diagnostic.returncode != 0:
            raise PreflightError(
                "container CUDA/BF16 diagnostic failed: " + diagnostic.stderr.strip()
            )
    return {
        "selected": device,
        "image": image,
        "image_id": image_id,
        "git_probe": "passed",
    }


def _wandb_check(cfg: DictConfig, *, executor: str = "host") -> dict[str, Any]:
    wandb_cfg = _wandb_configuration(cfg)
    mode = str(wandb_cfg.get("mode", "disabled"))
    artifact = wandb_cfg.get("artifact", {})
    policy = str(artifact.get("policy", "none")) if isinstance(artifact, Mapping) else "none"
    identity = {
        "mode": mode,
        "project": wandb_cfg.get("project"),
        "entity": wandb_cfg.get("entity"),
        "artifact_policy": policy,
    }
    if mode != "online":
        return {
            "status": "passed",
            **identity,
            "credentials": "not_required",
            "quota": "not_required",
            "blocking_reasons": [],
        }

    blocking: list[str] = []
    credential_visible = _wandb_credential_visible(executor)
    if not credential_visible:
        blocking.append("W&B credential is missing")
    quota: dict[str, Any] = {"policy": policy, "status": "not_required"}
    if policy != "none":
        snapshot_path = artifact.get("usage_snapshot_path")
        if not snapshot_path:
            blocking.append("W&B quota visibility snapshot is required for artifact upload")
            quota["status"] = "missing"
        else:
            try:
                snapshot = load_usage_snapshot(
                    snapshot_path,
                    expected_entity=str(wandb_cfg.get("entity", "")),
                    max_age_seconds=float(artifact.get("max_usage_age_seconds", 0)),
                )
                required = int(artifact.get("reserve_bytes", 0))
                available = snapshot.limit_bytes - snapshot.used_bytes
                quota = {
                    "status": "visible",
                    "plan": snapshot.plan,
                    "used_bytes": snapshot.used_bytes,
                    "limit_bytes": snapshot.limit_bytes,
                    "retention": snapshot.retention,
                    "configured_reserve_bytes": required,
                }
                if available < required:
                    blocking.append("visible W&B quota is below configured reserve")
                    quota["status"] = "insufficient"
            except (OSError, TypeError, ValueError) as error:
                blocking.append(f"W&B quota visibility snapshot is invalid: {error}")
                quota = {"status": "invalid", "error": str(error), "policy": policy}
    return {
        "status": "blocked_online" if blocking else "passed",
        **identity,
        "credentials": "visible" if credential_visible else "missing",
        "credential_transport": (
            "environment"
            if credential_visible and os.environ.get("WANDB_API_KEY")
            else "host_netrc"
            if credential_visible and executor == "host"
            else "none"
        ),
        "quota": quota,
        "blocking_reasons": blocking,
    }


def _wandb_credential_visible(executor: str = "host") -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    if executor == "container":
        return False
    try:
        authenticators = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False
    return authenticators is not None and bool(authenticators[2])


def _wandb_mode(cfg: DictConfig) -> str:
    return str(_wandb_configuration(cfg).get("mode", "disabled"))


def _wandb_configuration(cfg: DictConfig) -> Mapping[str, Any]:
    plain = _plain(cfg)
    profile = plain.get("profile", {})
    purpose = str(profile.get("purpose", "")) if isinstance(profile, Mapping) else ""
    if purpose in {"evaluation", "benchmark"}:
        owner = plain.get(purpose, {})
        value = owner.get("wandb", {}) if isinstance(owner, Mapping) else {}
    else:
        value = plain.get("wandb", {})
    return value if isinstance(value, Mapping) else {}


def _cache_configs(cfg: Mapping[str, Any] | DictConfig) -> list[dict[str, Any]]:
    plain = _plain(cfg)
    values = []
    streaming = plain.get("data", {}).get("streaming", {})
    if isinstance(streaming, Mapping) and isinstance(streaming.get("cache"), Mapping):
        values.append(dict(streaming["cache"]))
    benchmark = plain.get("benchmark", {})
    if isinstance(benchmark, Mapping) and isinstance(benchmark.get("cache"), Mapping):
        values.append(dict(benchmark["cache"]))
    return [value for value in values if "dir" in value and "max_size_bytes" in value]


def _plain(cfg: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    if not isinstance(value, Mapping):
        raise PreflightError("configuration must resolve to a mapping")
    return dict(value)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreflightError(f"{label} must be a mapping")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PreflightError(f"{label} must be a positive integer")
    return value


def _rooted(value: str, root_dir: Path) -> Path:
    path = Path(value).expanduser()
    return (root_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _existing_ancestor(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists():
        if candidate == candidate.parent:
            raise PreflightError(f"no existing ancestor for path: {path}")
        candidate = candidate.parent
    return candidate


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for directory, _subdirectories, filenames in os.walk(path, followlinks=False):
        for filename in filenames:
            candidate = Path(directory) / filename
            try:
                if not candidate.is_symlink():
                    total += candidate.stat().st_size
            except FileNotFoundError:
                continue
    return total


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()
