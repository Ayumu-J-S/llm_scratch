"""Verified, full-state checkpoints for the single-process training path.

Recovery checkpoints deliberately use a small direct file protocol: write a
unique temporary file, load it back, then replace the final name atomically.
The latest usable recovery is selected by verification rather than by trusting
the filename, so a torn/corrupt newest file cannot hide an older verified
state.  Best, final, and milestone files live outside recovery rotation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from data.identity import canonical_json_bytes


CHECKPOINT_SCHEMA_VERSION = 1
_RECOVERY_PATTERN = re.compile(r"^recovery-step-(\d{12})\.pt$")
_RUN_LINEAGE_PATTERN = re.compile(r"^run-[0-9a-f]{32}$")


class CheckpointError(RuntimeError):
    """Base error for a checkpoint that cannot safely be used."""


class CheckpointVerificationError(CheckpointError):
    """The checkpoint was missing, corrupt, or structurally incomplete."""


class CheckpointCompatibilityError(CheckpointError):
    """A verified checkpoint belongs to a different experiment identity."""


def require_exact_stream_resume_state(state: Mapping[str, Any]) -> None:
    """Reject a resume that cannot prove its next training sample is exact."""

    cursor = state.get("stream_cursor")
    if not isinstance(cursor, Mapping):
        raise CheckpointCompatibilityError(
            "exact resume requires a cursor-aware streaming train loader; "
            "this checkpoint has no stream cursor"
        )


@dataclass(frozen=True)
class ResumeCheckpoint:
    """A verified resume payload and the recovery path selected for it."""

    path: Path
    payload: dict[str, Any]
    rejected_paths: tuple[Path, ...]


@dataclass(frozen=True)
class CheckpointWriteMeasurement:
    """Observable local pause/write/verification evidence for one save."""

    path: Path
    size_bytes: int
    write_seconds: float
    verification_seconds: float
    pause_seconds: float

    @property
    def write_bytes_per_second(self) -> float:
        return self.size_bytes / self.write_seconds if self.write_seconds > 0 else float("inf")


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A verified payload and physical identity captured through one open file."""

    payload: dict[str, Any]
    physical_identity: dict[str, Any]


def capture_rng_state() -> dict[str, Any]:
    """Capture every process RNG that can affect a resumed single-process run."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore :func:`capture_rng_state` after model/data construction."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = required.difference(state)
    if missing:
        raise CheckpointVerificationError(f"checkpoint RNG state is missing {sorted(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state["torch_cuda"]
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise CheckpointCompatibilityError(
                "checkpoint contains CUDA RNG state but CUDA is unavailable for resume"
            )
        torch.cuda.set_rng_state_all(cuda_state)


def build_checkpoint_identity(
    cfg: DictConfig | Mapping[str, Any], *, run_manifest_path: str | Path | None = None
) -> dict[str, Any]:
    """Build the immutable identity checked before a resume mutates training.

    ``artifacts.resume_path`` is an operational input selecting *where* to
    recover from.  It is deliberately excluded from the experiment identity;
    model/data/tokenizer/config differences remain incompatible.
    """

    config = _plain_config(cfg)

    run_manifest: dict[str, Any] = {}
    if run_manifest_path is not None:
        path = Path(run_manifest_path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CheckpointError(
                f"cannot read run manifest for checkpoint identity: {path}"
            ) from error
        if not isinstance(value, dict):
            raise CheckpointError(f"run manifest for checkpoint identity is not an object: {path}")
        run_manifest = value

    tokenizer = run_manifest.get("tokenizer", {})
    data = run_manifest.get("data", [])
    data_fingerprints = [
        item.get("fingerprint")
        for item in data
        if isinstance(item, Mapping) and item.get("fingerprint") is not None
    ]
    configured_data_fingerprints = configured_manifest_fingerprints(config)
    if run_manifest_path is not None and data_fingerprints != configured_data_fingerprints:
        if data_fingerprints or configured_data_fingerprints:
            raise CheckpointError(
                "configured and captured data manifest fingerprints are out of order or differ"
            )
    if configured_data_fingerprints:
        data_fingerprints = configured_data_fingerprints
    identity = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "config_sha256": checkpoint_config_sha256(config),
        "model_config": copy.deepcopy(config.get("model", {})),
        "tokenizer_fingerprint": tokenizer.get("fingerprint")
        if isinstance(tokenizer, Mapping)
        else None,
        "data_fingerprints": data_fingerprints,
    }
    if run_manifest_path is not None:
        git = run_manifest.get("git")
        lock = run_manifest.get("lock")
        experiment_id = run_manifest.get("experiment_id")
        run_lineage_id = run_manifest.get("run_lineage_id")
        if (
            not isinstance(experiment_id, str)
            or not experiment_id
            or not isinstance(run_lineage_id, str)
            or _RUN_LINEAGE_PATTERN.fullmatch(run_lineage_id) is None
            or not isinstance(git, Mapping)
            or not isinstance(git.get("sha"), str)
            or not git["sha"]
            or not isinstance(lock, Mapping)
            or not isinstance(lock.get("sha256"), str)
            or not lock["sha256"]
        ):
            raise CheckpointError(
                "run manifest for checkpoint identity requires experiment_id, run_lineage_id, "
                "git.sha, and lock.sha256"
            )
        # These are the recorded immutable run values, not inferred substitutes.
        identity["experiment_id"] = experiment_id
        identity["run_lineage_id"] = run_lineage_id
        identity["git_sha"] = git["sha"]
        identity["lock_sha256"] = lock["sha256"]
    return identity


def checkpoint_config_sha256(cfg: DictConfig | Mapping[str, Any]) -> str:
    """Hash training identity while excluding operational evidence controls."""

    config = _plain_config(cfg)
    config.pop("measurement", None)
    artifacts = config.get("artifacts")
    if isinstance(artifacts, dict):
        artifacts = copy.deepcopy(artifacts)
        artifacts.pop("resume_path", None)
        config["artifacts"] = artifacts
    config.pop("wandb", None)
    return _sha256_json(config)


def configured_manifest_fingerprints(cfg: DictConfig | Mapping[str, Any]) -> list[str]:
    """Return configured manifest fingerprints in train-then-validation order."""

    config = _plain_config(cfg)
    data = config.get("data", {})
    if not isinstance(data, Mapping):
        return []
    if data.get("mode") == "memorization_smoke":
        smoke = data.get("memorization", {})
        fingerprint = smoke.get("expected_fingerprint") if isinstance(smoke, Mapping) else None
        return [str(fingerprint)] if fingerprint else []
    if data.get("mode") != "streaming":
        return []

    result: list[str] = []
    streaming = data.get("streaming", {})
    if not isinstance(streaming, Mapping):
        return result
    for split_name in ("train", "validation"):
        split = streaming.get(split_name, {})
        if not isinstance(split, Mapping):
            continue
        sources = split.get("sources", split.get("datasets", []))
        if not isinstance(sources, list):
            continue
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            source_type = source.get("type", source.get("source", "hf"))
            fingerprint = source.get("expected_fingerprint")
            if source_type == "manifest" and fingerprint:
                result.append(str(fingerprint))
    return result


def verify_checkpoint_config_identity(
    state: Mapping[str, Any], identity: Mapping[str, Any]
) -> None:
    """Prove that a full-state checkpoint config belongs to its envelope."""

    resolved_config = state.get("resolved_config")
    expected = identity.get("config_sha256")
    if not isinstance(resolved_config, Mapping) or not isinstance(expected, str):
        raise CheckpointVerificationError(
            "full-state checkpoint requires resolved_config and identity.config_sha256"
        )
    actual = checkpoint_config_sha256(resolved_config)
    if actual != expected:
        raise CheckpointCompatibilityError(
            "checkpoint resolved_config does not match identity.config_sha256"
        )
    expected_data = identity.get("data_fingerprints")
    configured_data = configured_manifest_fingerprints(resolved_config)
    if expected_data != configured_data:
        raise CheckpointCompatibilityError(
            "checkpoint resolved_config data manifests do not match identity.data_fingerprints"
        )


def build_logical_checkpoint_identity(
    checkpoint_identity: Mapping[str, Any], counters: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the logical model identity shared by training and evaluation."""

    optimizer_step = counters.get("optimizer_step")
    target_tokens = counters.get("target_tokens")
    if (
        isinstance(optimizer_step, bool)
        or not isinstance(optimizer_step, int)
        or optimizer_step < 0
        or isinstance(target_tokens, bool)
        or not isinstance(target_tokens, int)
        or target_tokens < 0
    ):
        raise CheckpointVerificationError("logical checkpoint counters are invalid")
    return {
        "checkpoint_identity": copy.deepcopy(dict(checkpoint_identity)),
        "optimizer_step": optimizer_step,
        "target_tokens": target_tokens,
    }


class CheckpointManager:
    """Own verified checkpoint files and recovery rotation for one run."""

    def __init__(
        self, directory: str | Path, *, keep_last_n: int, identity: Mapping[str, Any]
    ) -> None:
        if isinstance(keep_last_n, bool) or not isinstance(keep_last_n, int) or keep_last_n < 1:
            raise ValueError("artifacts.keep_last_n must be a positive integer")
        self.directory = Path(directory)
        self.keep_last_n = keep_last_n
        self.identity = copy.deepcopy(dict(identity))
        self.last_write_measurement: CheckpointWriteMeasurement | None = None

    def save_recovery(self, payload: Mapping[str, Any]) -> Path:
        step = _checkpoint_step(payload)
        destination = self.directory / f"recovery-step-{step:012d}.pt"
        self.last_write_measurement = self._atomic_write(
            destination, self._checkpoint_payload("recovery", payload)
        )
        self._rotate_recovery(exclude=destination)
        return destination

    def save_best(self, payload: Mapping[str, Any]) -> Path:
        destination = self.directory / "best.pt"
        self.last_write_measurement = self._atomic_write(
            destination, self._checkpoint_payload("best", payload)
        )
        return destination

    def save_final(self, payload: Mapping[str, Any]) -> Path:
        destination = self.directory / "final.pt"
        self.last_write_measurement = self._atomic_write(
            destination, self._checkpoint_payload("final", payload)
        )
        return destination

    def save_milestone(self, payload: Mapping[str, Any]) -> Path:
        step = _checkpoint_step(payload)
        destination = self.directory / f"milestone-step-{step:012d}.pt"
        self.last_write_measurement = self._atomic_write(
            destination, self._checkpoint_payload("milestone", payload)
        )
        return destination

    def load_resume(self, resume_path: str | Path) -> ResumeCheckpoint:
        """Load a compatible recovery, skipping corrupt newer recovery files."""

        candidates = self._resume_candidates(resume_path)
        if not candidates:
            raise CheckpointVerificationError(
                f"no recovery checkpoint found for resume path: {resume_path}"
            )
        rejected: list[Path] = []
        for path in candidates:
            try:
                payload = self._read_verified(path)
            except CheckpointVerificationError:
                rejected.append(path)
                continue
            self._validate_identity(payload)
            return ResumeCheckpoint(path=path, payload=payload, rejected_paths=tuple(rejected))
        names = ", ".join(str(path) for path in rejected)
        raise CheckpointVerificationError(
            f"no verified recovery checkpoint remains; rejected: {names}"
        )

    def _checkpoint_payload(self, kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        # ``model.state_dict()`` intentionally shares parameter storage. A
        # synchronous torch.save can serialize that stable post-update state
        # directly; deep-copying it would create an avoidable full-model peak
        # on the one-machine UMA path.
        state = dict(payload)
        existing_identity = state.pop("identity", None)
        if existing_identity is not None and existing_identity != self.identity:
            raise CheckpointCompatibilityError(
                "checkpoint payload identity differs from manager identity"
            )
        if "resolved_config" in state:
            verify_checkpoint_config_identity(state, self.identity)
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": kind,
            "identity": copy.deepcopy(self.identity),
            "state": state,
        }

    def _atomic_write(
        self, destination: Path, payload: Mapping[str, Any]
    ) -> CheckpointWriteMeasurement:
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        pause_started = time.monotonic()
        try:
            torch.save(dict(payload), temporary)
            _fsync_file(temporary)
            write_finished = time.monotonic()
            verified = self._read_verified(temporary)
            if verified["kind"] != payload["kind"] or verified["identity"] != self.identity:
                raise CheckpointVerificationError(
                    "checkpoint read-back did not preserve its verified header"
                )
            verification_finished = time.monotonic()
            os.replace(temporary, destination)
            _fsync_directory(self.directory)
            completed = time.monotonic()
            return CheckpointWriteMeasurement(
                path=destination,
                size_bytes=destination.stat().st_size,
                write_seconds=write_finished - pause_started,
                verification_seconds=verification_finished - write_finished,
                pause_seconds=completed - pause_started,
            )
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    def _read_verified(self, path: Path) -> dict[str, Any]:
        try:
            payload = _torch_load(path)
        except Exception as error:  # torch exposes several format-specific error types.
            raise CheckpointVerificationError(
                f"unable to read checkpoint {path}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise CheckpointVerificationError(f"checkpoint {path} is not a mapping")
        required = {"schema_version", "kind", "identity", "state"}
        missing = required.difference(payload)
        if missing:
            raise CheckpointVerificationError(f"checkpoint {path} is missing {sorted(missing)}")
        if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointVerificationError(
                f"checkpoint {path} has unsupported schema {payload['schema_version']!r}"
            )
        if payload["kind"] not in {"recovery", "best", "final", "milestone"}:
            raise CheckpointVerificationError(
                f"checkpoint {path} has unknown kind {payload['kind']!r}"
            )
        if not isinstance(payload["identity"], Mapping) or not isinstance(
            payload["state"], Mapping
        ):
            raise CheckpointVerificationError(f"checkpoint {path} has invalid identity or state")
        if "resolved_config" in payload["state"]:
            verify_checkpoint_config_identity(payload["state"], payload["identity"])
        _checkpoint_step(payload["state"])
        return payload

    def _validate_identity(self, payload: Mapping[str, Any]) -> None:
        observed = dict(payload["identity"])
        if observed != self.identity:
            changed = sorted(
                key
                for key in set(observed).union(self.identity)
                if observed.get(key) != self.identity.get(key)
            )
            raise CheckpointCompatibilityError(
                "checkpoint is incompatible with the resolved experiment identity; "
                f"changed fields: {', '.join(changed)}"
            )

    def _resume_candidates(self, resume_path: str | Path) -> list[Path]:
        raw = str(resume_path)
        if raw == "latest":
            path = self.directory
        else:
            path = Path(resume_path)
            # Hydra values are often relative. Bind them to the configured
            # checkpoint directory, never to an incidental process CWD.
            if not path.is_absolute():
                path = self.directory / path
        if path.is_dir():
            return self._recovery_paths(path)
        if not path.exists():
            raise CheckpointVerificationError(f"resume checkpoint does not exist: {path}")
        match = _RECOVERY_PATTERN.match(path.name)
        if match is None:
            return [path]
        siblings = self._recovery_paths(path.parent)
        return [path, *(candidate for candidate in siblings if candidate != path)]

    def _rotate_recovery(self, *, exclude: Path) -> None:
        recoveries = self._recovery_paths(self.directory)
        for path in recoveries[self.keep_last_n :]:
            if path == exclude:
                continue
            path.unlink()
        _fsync_directory(self.directory)

    @staticmethod
    def _recovery_paths(directory: Path) -> list[Path]:
        if not directory.is_dir():
            return []
        entries: list[tuple[int, Path]] = []
        for path in directory.iterdir():
            match = _RECOVERY_PATTERN.match(path.name)
            if match is not None and path.is_file():
                entries.append((int(match.group(1)), path))
        return [path for _, path in sorted(entries, reverse=True)]


def load_run_lineage_from_resume(resume_path: str | Path, *, checkpoint_dir: str | Path) -> str:
    """Read the unique run lineage from the newest verified resume candidate."""

    probe = CheckpointManager(checkpoint_dir, keep_last_n=1, identity={})
    candidates = probe._resume_candidates(resume_path)
    if not candidates:
        raise CheckpointVerificationError(
            f"no recovery checkpoint found for resume path: {resume_path}"
        )
    rejected: list[Path] = []
    for path in candidates:
        try:
            payload = probe._read_verified(path)
        except CheckpointVerificationError:
            rejected.append(path)
            continue
        lineage = payload["identity"].get("run_lineage_id")
        if not isinstance(lineage, str) or _RUN_LINEAGE_PATTERN.fullmatch(lineage) is None:
            raise CheckpointCompatibilityError(
                "resume checkpoint is missing its unique run_lineage_id"
            )
        return lineage
    names = ", ".join(str(path) for path in rejected)
    raise CheckpointVerificationError(
        f"no verified recovery checkpoint remains for lineage; rejected: {names}"
    )


def _checkpoint_step(payload: Mapping[str, Any]) -> int:
    counters = payload.get("counters")
    if not isinstance(counters, Mapping):
        raise CheckpointVerificationError("checkpoint state is missing counters")
    step = counters.get("optimizer_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise CheckpointVerificationError(
            "checkpoint optimizer_step must be a non-negative integer"
        )
    return step


def _plain_config(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    value = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else dict(cfg)
    if not isinstance(value, dict):
        raise TypeError("checkpoint identity requires a mapping configuration")
    return copy.deepcopy(value)


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Older PyTorch does not expose weights_only.
        return torch.load(path, map_location="cpu")


def load_checkpoint_for_generation(path: str | Path) -> LoadedCheckpoint:
    """Load the verified inference-relevant part of a full-state checkpoint.

    Generation deliberately accepts only repository checkpoint files.  The
    checkpoint's own resolved config and canonical-tokenizer configuration are
    the reconstruction authority, so callers cannot quietly substitute model
    dimensions or tokenizer settings with CLI arguments.
    """

    checkpoint_path = Path(path).resolve()
    try:
        with checkpoint_path.open("rb") as handle:
            stat = os.fstat(handle.fileno())
            digest = hashlib.sha256()
            size_bytes = 0
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size_bytes += len(chunk)
            if size_bytes != stat.st_size:
                raise CheckpointVerificationError(
                    f"checkpoint {checkpoint_path} changed while its identity was captured"
                )
            if size_bytes == 0:
                raise CheckpointVerificationError(f"checkpoint {checkpoint_path} is empty")
            physical_identity = {
                "path": str(checkpoint_path),
                "sha256": digest.hexdigest(),
                "size_bytes": size_bytes,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            }
            handle.seek(0)
            try:
                payload = _torch_load_handle(handle)
            except Exception as error:  # Torch exposes several format-specific error types.
                raise CheckpointVerificationError(
                    f"unable to read checkpoint {checkpoint_path}: {error}"
                ) from error
            final_stat = os.fstat(handle.fileno())
            if (
                final_stat.st_dev,
                final_stat.st_ino,
                final_stat.st_size,
                final_stat.st_mtime_ns,
            ) != (
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
            ):
                raise CheckpointVerificationError(
                    f"checkpoint {checkpoint_path} changed while it was loaded"
                )
    except OSError as error:
        raise CheckpointVerificationError(
            f"generation checkpoint does not exist or is not a file: {checkpoint_path}"
        ) from error
    if not isinstance(payload, dict):
        raise CheckpointVerificationError(f"checkpoint {checkpoint_path} is not a mapping")
    required = {"schema_version", "kind", "identity", "state"}
    missing = required.difference(payload)
    if missing:
        raise CheckpointVerificationError(
            f"checkpoint {checkpoint_path} is missing {sorted(missing)}"
        )
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointVerificationError(
            f"checkpoint {checkpoint_path} has unsupported schema {payload['schema_version']!r}"
        )
    if payload["kind"] not in {"recovery", "best", "final", "milestone"}:
        raise CheckpointVerificationError(
            f"checkpoint {checkpoint_path} has unknown kind {payload['kind']!r}"
        )
    identity = payload["identity"]
    state = payload["state"]
    if not isinstance(identity, Mapping) or not isinstance(state, Mapping):
        raise CheckpointVerificationError(
            f"checkpoint {checkpoint_path} has invalid identity or state"
        )
    _checkpoint_step(state)
    required_state = {"model", "resolved_config", "run_identity"}
    missing_state = required_state.difference(state)
    if missing_state:
        raise CheckpointVerificationError(
            f"checkpoint is not a full-state generation checkpoint; missing {sorted(missing_state)}"
        )
    if not isinstance(state["model"], Mapping):
        raise CheckpointVerificationError("checkpoint model state must be a mapping")
    if not isinstance(state["resolved_config"], Mapping):
        raise CheckpointVerificationError("checkpoint resolved_config must be a mapping")
    if not isinstance(state["run_identity"], Mapping):
        raise CheckpointVerificationError("checkpoint run_identity must be a mapping")
    verify_checkpoint_config_identity(state, identity)
    if dict(state["run_identity"]) != dict(identity):
        raise CheckpointCompatibilityError(
            "checkpoint envelope identity differs from its full-state run_identity"
        )
    return LoadedCheckpoint(payload=payload, physical_identity=physical_identity)


def _torch_load_handle(handle: BinaryIO) -> Any:
    """Deserialize from the same open checkpoint file used to hash its bytes."""

    try:
        return torch.load(handle, map_location="cpu", weights_only=False)
    except TypeError:  # Older PyTorch does not expose weights_only.
        handle.seek(0)
        return torch.load(handle, map_location="cpu")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
