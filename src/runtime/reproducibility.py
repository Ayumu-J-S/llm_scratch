"""Run identity, immutable-input checks, and process-wide RNG setup.

This module deliberately keeps the reproducibility contract independent from
the trainer.  A run can therefore retain enough evidence to be audited even
when W&B is disabled or training stops before the first optimizer step.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import stat
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from data.identity import canonical_json_bytes
from runtime.environment import collect_environment


class ReproducibilityError(ValueError):
    """Raised when a run cannot satisfy its immutable-input contract."""


class ManifestMismatchError(ReproducibilityError):
    """Raised when an input or persisted run manifest changed after capture."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("reproducibility.seed must be an integer")
    if seed < 0:
        raise ValueError("reproducibility.seed must be non-negative")
    return seed


def seed_everything(seed: int, *, deterministic: bool = True) -> torch.Generator:
    """Seed Python, NumPy, Torch CPU/CUDA, and return a DataLoader generator.

    Model construction must happen after this call.  ``torch.Generator`` is
    returned so each DataLoader can use an explicit, inspectable RNG rather
    than depending on process-global state.  Deterministic mode fails on an
    operation without a deterministic implementation.
    """

    seed = _positive_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Fail rather than silently weakening the requested deterministic mode.
        # This does not promise bitwise equality across platforms or versions.
        torch.use_deterministic_algorithms(True, warn_only=False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    return torch.Generator().manual_seed(seed)


def dataloader_worker_init_fn(worker_id: int) -> None:
    """Seed Python and NumPy in a DataLoader worker from Torch's worker seed."""

    del worker_id  # The seed is assigned by DataLoader's generator.
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def dataloader_generator(seed: int, *, stream: str = "train") -> torch.Generator:
    """Derive stable, independent DataLoader RNG streams from one Hydra seed."""

    seed = _positive_seed(seed)
    stream_bytes = stream.encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(stream_bytes).digest()[:8], "big")
    return torch.Generator().manual_seed((seed + offset) % (2**63 - 1))


def _git(root_dir: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root_dir,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise ReproducibilityError(
                f"unable to inspect git repository: git {' '.join(args)}"
            ) from error
        return result.stdout.strip()

    def run_bytes(*args: str) -> bytes:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root_dir,
                check=True,
                capture_output=True,
                timeout=10,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise ReproducibilityError(
                f"unable to inspect git repository bytes: git {' '.join(args)}"
            ) from error
        return result.stdout

    status = run("status", "--porcelain", "--untracked-files=all")
    sha = run("rev-parse", "HEAD")
    tracked = run_bytes("ls-files", "-z")
    _reject_non_regular_tracked_paths(root_dir, tracked)
    tracked_diff = run_bytes("diff", "--binary", "--no-ext-diff", "HEAD", "--")
    untracked = run_bytes("ls-files", "--others", "--exclude-standard", "-z")
    worktree_digest = hashlib.sha256()
    _hash_field(worktree_digest, b"revision", b"git-worktree-content-v1")
    _hash_field(worktree_digest, b"tracked-diff", tracked_diff)
    for raw_path in sorted(path for path in untracked.split(b"\0") if path):
        _hash_untracked_path(worktree_digest, root_dir, raw_path)
    if (
        run("rev-parse", "HEAD") != sha
        or run("status", "--porcelain", "--untracked-files=all") != status
        or run_bytes("ls-files", "-z") != tracked
        or run_bytes("diff", "--binary", "--no-ext-diff", "HEAD", "--") != tracked_diff
        or run_bytes("ls-files", "--others", "--exclude-standard", "-z") != untracked
    ):
        raise ReproducibilityError("git worktree changed while its content identity was captured")
    _reject_non_regular_tracked_paths(root_dir, tracked)
    return {
        "sha": sha,
        "dirty": bool(status),
        "status": status.splitlines(),
        "worktree_content_sha256": worktree_digest.hexdigest(),
    }


def _hash_field(digest: Any, label: bytes, payload: bytes) -> None:
    digest.update(len(label).to_bytes(8, "big"))
    digest.update(label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _hash_untracked_path(digest: Any, root_dir: Path, raw_path: bytes) -> None:
    path = root_dir / os.fsdecode(raw_path)
    try:
        before = path.lstat()
    except OSError as error:
        raise ReproducibilityError(f"unable to hash untracked path: {path}") from error
    if not stat.S_ISREG(before.st_mode):
        raise ReproducibilityError(
            f"untracked evaluator path must be a regular file, not a symlink or special path: {path}"
        )
    _hash_field(digest, b"untracked-path", raw_path)
    _hash_field(digest, b"untracked-mode", str(before.st_mode).encode())
    try:
        file_digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                file_digest.update(chunk)
        _hash_field(digest, b"untracked-file-sha256", file_digest.digest())
        after = path.lstat()
    except OSError as error:
        raise ReproducibilityError(f"unable to hash untracked path: {path}") from error
    observed_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    observed_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    )
    if observed_after != observed_before:
        raise ReproducibilityError(f"untracked path changed while it was hashed: {path}")


def _reject_non_regular_tracked_paths(root_dir: Path, tracked: bytes) -> None:
    for raw_path in (path for path in tracked.split(b"\0") if path):
        path = root_dir / os.fsdecode(raw_path)
        try:
            observed = path.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ReproducibilityError(
                f"unable to inspect tracked evaluator path: {path}"
            ) from error
        if not stat.S_ISREG(observed.st_mode):
            raise ReproducibilityError(
                f"tracked evaluator path must be a regular file, not a symlink or special path: {path}"
            )


def collect_git_identity(root_dir: str | Path) -> dict[str, Any]:
    """Return the observable commit and dirty state for an evaluator or run."""

    return _git(Path(root_dir).resolve())


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _experiment_identity_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only the explicit operational resume selector for run identity."""

    normalized = _plain(cfg)
    if not isinstance(normalized, dict):
        raise ReproducibilityError("run identity requires a mapping configuration")
    artifacts = normalized.get("artifacts")
    if isinstance(artifacts, Mapping):
        normalized_artifacts = dict(artifacts)
        normalized_artifacts.pop("resume_path", None)
        normalized["artifacts"] = normalized_artifacts
    return normalized


def _manifest_payload(path: Path, expected_fingerprint: str | None = None) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReproducibilityError(f"cannot read manifest {path}") from error
    if not isinstance(payload, dict):
        raise ReproducibilityError(f"manifest {path} must contain a JSON object")
    actual = payload.get("manifest_fingerprint", payload.get("fingerprint"))
    if expected_fingerprint is not None and actual != expected_fingerprint:
        raise ManifestMismatchError(
            f"manifest fingerprint mismatch for {path}: expected {expected_fingerprint}, got {actual}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "fingerprint": actual,
        "payload": payload,
    }


def _collect_manifest_configs(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    data = cfg.get("data", {})
    if not isinstance(data, Mapping):
        return []
    mode = data.get("mode")
    if mode == "memorization_smoke":
        smoke = data.get("memorization", {})
        values = (
            [{"type": "manifest", "name": "memorization", **dict(smoke)}]
            if isinstance(smoke, Mapping)
            else []
        )
    elif mode == "streaming":
        streaming = data.get("streaming", {})
        values = []
        if isinstance(streaming, Mapping):
            for split in ("train", "validation"):
                split_cfg = streaming.get(split, {})
                if isinstance(split_cfg, Mapping):
                    sources = split_cfg.get("sources", split_cfg.get("datasets", []))
                    if isinstance(sources, list):
                        values.extend(sources)
    else:
        values = []
    return [dict(value) for value in values if isinstance(value, Mapping)]


def validate_immutable_inputs(cfg: Mapping[str, Any], *, real_run: bool) -> None:
    """Reject mutable remote inputs before a real run can consume them."""

    tokenizer = cfg.get("tokenizer", {})
    if not isinstance(tokenizer, Mapping) or not tokenizer.get("manifest_path"):
        raise ReproducibilityError("runs require a pinned tokenizer manifest")
    for source in _collect_manifest_configs(cfg):
        source_type = source.get("type", source.get("source", "hf"))
        if source_type == "manifest":
            if not source.get("manifest_path") or not source.get("expected_fingerprint"):
                raise ReproducibilityError(
                    "manifest data sources require path and expected_fingerprint"
                )
        elif real_run:
            revision = source.get("revision")
            if source_type == "hf" and isinstance(revision, str) and len(revision) == 40:
                if all(character in "0123456789abcdef" for character in revision):
                    continue
            raise ReproducibilityError(
                "real runs reject mutable remote tokenizer/dataset inputs; use an immutable manifest or 40-hex revision"
            )


def _copy_manifest(path: Path, destination: Path, expected: str | None) -> dict[str, Any]:
    manifest = _manifest_payload(path, expected)
    destination.write_text(
        json.dumps(manifest["payload"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest["run_path"] = destination.name
    manifest["run_sha256"] = sha256_file(destination)
    return manifest


def write_run_manifest(
    *,
    cfg: Mapping[str, Any],
    run_dir: str | Path,
    root_dir: str | Path,
    resolved_config_path: str | Path,
    tokenizer_manifest_path: str | Path,
    tokenizer_expected_fingerprint: str | None,
    data_manifest_configs: list[Mapping[str, Any]] | None = None,
) -> Path:
    """Persist a self-contained run identity and immutable input snapshots."""

    run_path = Path(run_dir)
    root = Path(root_dir).resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    git = _git(root)
    real_run = str(cfg.get("profile", {}).get("purpose", "")) == "pretraining"
    reproducibility_cfg = cfg.get("reproducibility", {})
    reject_dirty = (
        bool(reproducibility_cfg.get("reject_dirty", True))
        if isinstance(reproducibility_cfg, Mapping)
        else True
    )
    if real_run and reject_dirty and git["dirty"]:
        raise ReproducibilityError(
            "real runs require a clean git worktree; commit or explicitly use a smoke profile first"
        )
    validate_immutable_inputs(cfg, real_run=real_run)

    config_path = Path(resolved_config_path)
    lock_path = root / "uv.lock"
    if not config_path.is_file() or not lock_path.is_file():
        raise ReproducibilityError("run identity requires resolved_config.yaml and uv.lock")
    config_snapshot = run_path / "resolved_config.yaml"
    if config_path.resolve() != config_snapshot.resolve():
        config_snapshot.write_bytes(config_path.read_bytes())
    config_path = config_snapshot
    tokenizer_path = Path(tokenizer_manifest_path).resolve()
    tokenizer_snapshot = _copy_manifest(
        tokenizer_path,
        run_path / "tokenizer_manifest.json",
        tokenizer_expected_fingerprint,
    )

    configs = (
        data_manifest_configs
        if data_manifest_configs is not None
        else _collect_manifest_configs(cfg)
    )
    snapshots: list[dict[str, Any]] = []
    for index, source in enumerate(configs):
        if source.get("type", source.get("source", "hf")) != "manifest":
            continue
        path = Path(str(source["manifest_path"])).resolve()
        snapshot = _copy_manifest(
            path, run_path / f"data_manifest_{index}.json", source.get("expected_fingerprint")
        )
        snapshot["name"] = source.get("name")
        snapshot["selection"] = source.get("selection")
        snapshots.append(snapshot)

    config_hash = sha256_file(config_path)
    lock_hash = sha256_file(lock_path)
    identity_payload = {
        "git_sha": git["sha"],
        # The resolved config file remains recorded byte-for-byte below. Only
        # the explicit operational recovery selector is normalized for the
        # experiment ID, so resuming the same run does not manufacture a new
        # identity while every experiment-affecting config change still does.
        "config_sha256": sha256_bytes(canonical_json_bytes(_experiment_identity_config(cfg))),
        "lock_sha256": lock_hash,
        "seed": int(cfg.get("reproducibility", {}).get("seed", 0)),
        "tokenizer_fingerprint": tokenizer_snapshot["fingerprint"],
        "data_fingerprints": [item["fingerprint"] for item in snapshots],
    }
    experiment_id = "exp-" + sha256_bytes(canonical_json_bytes(identity_payload))[:20]
    payload = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "git": git,
        "config": {"path": config_path.name, "sha256": config_hash},
        "experiment_identity": {
            "config_sha256": identity_payload["config_sha256"],
            "operational_exclusions": ["artifacts.resume_path"],
        },
        "lock": {"path": "uv.lock", "sha256": lock_hash},
        "seed": int(cfg.get("reproducibility", {}).get("seed", 0)),
        "deterministic_algorithms": bool(cfg.get("reproducibility", {}).get("deterministic", True)),
        "hardware_software": collect_environment(),
        "tokenizer": tokenizer_snapshot,
        "data": snapshots,
    }
    destination = run_path / "run_manifest.json"
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destination


def verify_run_manifest(
    run_dir: str | Path, *, root_dir: str | Path | None = None
) -> dict[str, Any]:
    """Verify captured files and, when supplied, the source lock and Git SHA."""

    run_path = Path(run_dir)
    manifest_path = run_path / "run_manifest.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestMismatchError(f"invalid run manifest: {manifest_path}") from error
    config_entry = payload.get("config")
    config_file = (
        run_path / str(config_entry.get("path", "resolved_config.yaml"))
        if isinstance(config_entry, Mapping)
        else run_path / "resolved_config.yaml"
    )
    if (
        not config_file.is_file()
        or not isinstance(config_entry, Mapping)
        or sha256_file(config_file) != config_entry.get("sha256")
    ):
        raise ManifestMismatchError(f"resolved configuration changed: {config_file}")
    lock_entry = payload.get("lock")
    if root_dir is not None:
        if not isinstance(lock_entry, Mapping):
            raise ManifestMismatchError("run manifest is missing dependency-lock identity")
        lock_file = Path(root_dir).resolve() / str(lock_entry.get("path", "uv.lock"))
        if not lock_file.is_file() or sha256_file(lock_file) != lock_entry.get("sha256"):
            raise ManifestMismatchError(f"dependency lock changed: {lock_file}")
        recorded_git = payload.get("git") if isinstance(payload.get("git"), Mapping) else None
        if recorded_git is None:
            raise ManifestMismatchError("run manifest is missing source Git identity")
        recorded_sha = recorded_git.get("sha")
        current_git = _git(Path(root_dir).resolve())
        if not isinstance(recorded_sha, str) or current_git["sha"] != recorded_sha:
            raise ManifestMismatchError(
                f"source Git commit changed: expected {recorded_sha}, got {current_git['sha']}"
            )
        recorded_dirty = recorded_git.get("dirty")
        if not isinstance(recorded_dirty, bool) or current_git["dirty"] != recorded_dirty:
            raise ManifestMismatchError(
                "source Git worktree dirty state changed: "
                f"expected {recorded_dirty}, got {current_git['dirty']}"
            )
        recorded_status = recorded_git.get("status")
        if not isinstance(recorded_status, list) or current_git["status"] != recorded_status:
            raise ManifestMismatchError(
                "source Git worktree status changed: "
                f"expected {recorded_status!r}, got {current_git['status']!r}"
            )
        recorded_content = recorded_git.get("worktree_content_sha256")
        current_content = current_git.get("worktree_content_sha256")
        if (
            not isinstance(recorded_content, str)
            or not isinstance(current_content, str)
            or current_content != recorded_content
        ):
            raise ManifestMismatchError("source Git worktree content changed")
    entries = [payload.get("tokenizer")] + list(payload.get("data", []))
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ManifestMismatchError("run manifest contains an invalid input entry")
        run_file = run_path / str(entry.get("run_path", ""))
        expected = entry.get("run_sha256")
        if not run_file.is_file() or sha256_file(run_file) != expected:
            raise ManifestMismatchError(f"captured input manifest changed: {run_file}")
    return payload
