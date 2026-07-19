"""Strict inference-only model artifacts derived from verified checkpoints."""

from __future__ import annotations

import hashlib
import math
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from models.simple_decoder_transformer import SimpleDecoderTransformer
from tokenizer.canonical import CanonicalTokenizer
from training.checkpoint import LoadedCheckpoint, load_checkpoint_for_generation


MODEL_ARTIFACT_SCHEMA_VERSION = 1
MODEL_ARTIFACT_TOP_LEVEL_KEYS = {
    "schema_version",
    "artifact_type",
    "model",
    "tokenizer",
    "source_checkpoint",
    "counters",
}
MODEL_KEYS = {"architecture", "state_dict"}
ARCHITECTURE_KEYS = {
    "embed_size",
    "num_heads",
    "num_layers",
    "dropout",
    "max_sequence_length",
}
TOKENIZER_KEYS = {"manifest_path", "expected_fingerprint"}
SOURCE_CHECKPOINT_KEYS = {
    "kind",
    "sha256",
    "size_bytes",
    "logical_identity",
}
LOGICAL_IDENTITY_KEYS = {
    "schema_version",
    "config_sha256",
    "experiment_id",
    "git_sha",
    "lock_sha256",
    "data_fingerprints",
}
COUNTER_KEYS = {"optimizer_step", "target_tokens"}


@dataclass(frozen=True)
class ModelArtifactFile:
    path: str
    sha256: str
    size_bytes: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class ReconstructedModelArtifact:
    model: SimpleDecoderTransformer
    tokenizer: CanonicalTokenizer
    payload: dict[str, Any]
    physical_identity: ModelArtifactFile


def stage_model_artifact(
    source_checkpoint: str | Path,
    *,
    destination_dir: str | Path,
    expected_source_sha256: str,
    expected_source_size_bytes: int,
    expected_source_device: int,
    expected_source_inode: int,
    expected_source_mtime_ns: int,
    expected_source_ctime_ns: int,
) -> ModelArtifactFile:
    """Write and verify one allowlisted model-only package.

    The source is loaded through the repository checkpoint verifier. Only model
    tensors and the minimum reconstruction/provenance fields are copied; full
    training state is deliberately not traversed or forwarded.
    """

    loaded = load_checkpoint_for_generation(source_checkpoint)
    source_identity = loaded.physical_identity
    if (
        source_identity["sha256"],
        source_identity["size_bytes"],
        source_identity["device"],
        source_identity["inode"],
        source_identity["mtime_ns"],
        source_identity["ctime_ns"],
    ) != (
        expected_source_sha256,
        expected_source_size_bytes,
        expected_source_device,
        expected_source_inode,
        expected_source_mtime_ns,
        expected_source_ctime_ns,
    ):
        raise ValueError("source checkpoint changed after artifact admission")
    source_binding = (
        str(loaded.payload["kind"]),
        str(source_identity["sha256"]),
        int(source_identity["size_bytes"]),
    )
    payload = _artifact_payload(loaded)
    directory = Path(destination_dir)
    directory.mkdir(parents=True, mode=0o700, exist_ok=True)
    directory.chmod(0o700)
    destination = (
        directory / f"model-{loaded.physical_identity['sha256'][:16]}-{uuid.uuid4().hex}.pt"
    )
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.chmod(0o600)
        _fsync_file(temporary)
        os.replace(temporary, destination)
        _fsync_directory(directory)
        # The full checkpoint can include optimizer and cursor state. Drop it
        # before reloading/reconstructing the smaller model package so optional
        # tracking does not retain both payloads at once.
        del payload
        del loaded
        reconstructed = reconstruct_model_artifact(destination)
        _assert_source_binding(reconstructed.payload, source_binding)
        return reconstructed.physical_identity
    except BaseException:
        for path in (temporary, destination):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def reconstruct_model_artifact(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> ReconstructedModelArtifact:
    """Strictly load an inference artifact and reconstruct its model/tokenizer."""

    artifact_path = Path(path).resolve()
    payload, physical = _load_with_identity(artifact_path)
    _validate_payload(payload)
    tokenizer_config = payload["tokenizer"]
    tokenizer = CanonicalTokenizer.from_config(tokenizer_config)
    architecture = payload["model"]["architecture"]
    if tokenizer.fingerprint != tokenizer_config["expected_fingerprint"]:
        raise ValueError("model artifact tokenizer fingerprint does not match its manifest")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested for a model artifact but is unavailable")
    model = SimpleDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_size=architecture["embed_size"],
        num_heads=architecture["num_heads"],
        max_len=architecture["max_sequence_length"],
        num_layers=architecture["num_layers"],
        dropout=architecture["dropout"],
        pad_token_id=tokenizer.pad_token_id,
    )
    expected_state = model.state_dict()
    observed_state = payload["model"]["state_dict"]
    if set(observed_state) != set(expected_state):
        raise ValueError("model artifact tensor names do not match its architecture")
    for name, expected in expected_state.items():
        observed = observed_state[name]
        if observed.shape != expected.shape or observed.dtype != expected.dtype:
            raise ValueError(
                f"model artifact tensor {name!r} shape/dtype does not match its architecture"
            )
    try:
        model.load_state_dict(observed_state, strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            "model artifact tensors do not match its allowlisted architecture"
        ) from error
    model.to(resolved_device)
    model.eval()
    return ReconstructedModelArtifact(
        model=model,
        tokenizer=tokenizer,
        payload=payload,
        physical_identity=physical,
    )


def remove_staged_model_artifact(artifact: ModelArtifactFile | None) -> None:
    """Remove a local model-only staging file after W&B no longer needs it."""

    if artifact is not None:
        Path(artifact.path).unlink(missing_ok=True)


def verify_staged_model_artifact(artifact: ModelArtifactFile) -> None:
    """Re-hash, validate, and reconstruct an unchanged staging file."""

    observed = reconstruct_model_artifact(artifact.path).physical_identity
    if observed != artifact:
        raise ValueError("model artifact physical identity changed")


def _artifact_payload(loaded: LoadedCheckpoint) -> dict[str, Any]:
    envelope = loaded.payload
    state = _mapping(envelope["state"], "checkpoint state")
    config = _mapping(state["resolved_config"], "checkpoint resolved_config")
    identity = _mapping(envelope["identity"], "checkpoint identity")
    model_config = _mapping(config.get("model"), "checkpoint model config")
    identity_model_config = _mapping(identity.get("model_config"), "checkpoint model identity")
    if dict(model_config) != dict(identity_model_config):
        raise ValueError("checkpoint model identity differs from its resolved model config")
    training_config = _mapping(config.get("training"), "checkpoint training config")
    tokenizer_config = _mapping(config.get("tokenizer"), "checkpoint tokenizer config")
    if set(tokenizer_config) != TOKENIZER_KEYS:
        raise ValueError("checkpoint tokenizer config is not the canonical allowlist")
    expected_fingerprint = _nonempty_string(
        tokenizer_config.get("expected_fingerprint"),
        "tokenizer expected_fingerprint",
    )
    if identity.get("tokenizer_fingerprint") != expected_fingerprint:
        raise ValueError("checkpoint tokenizer identity differs from its tokenizer config")
    counters = _mapping(state.get("counters"), "checkpoint counters")
    model_state = _mapping(state.get("model"), "checkpoint model state")
    tensor_state: dict[str, torch.Tensor] = {}
    for name, value in model_state.items():
        if not isinstance(name, str) or not name or not isinstance(value, torch.Tensor):
            raise ValueError("checkpoint model state must contain only named tensors")
        tensor_state[name] = value.detach().cpu()
    if not tensor_state:
        raise ValueError("checkpoint model state cannot be empty")
    physical = loaded.physical_identity
    return {
        "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "inference_model",
        "model": {
            "architecture": {
                "embed_size": _positive_int(model_config.get("embed_size"), "model.embed_size"),
                "num_heads": _positive_int(model_config.get("num_heads"), "model.num_heads"),
                "num_layers": _positive_int(model_config.get("num_layers"), "model.num_layers"),
                "dropout": _dropout(model_config.get("dropout")),
                "max_sequence_length": _positive_int(
                    training_config.get("sequence_length"),
                    "training.sequence_length",
                ),
            },
            "state_dict": tensor_state,
        },
        "tokenizer": {
            "manifest_path": _nonempty_string(
                tokenizer_config.get("manifest_path"),
                "tokenizer manifest_path",
            ),
            "expected_fingerprint": expected_fingerprint,
        },
        "source_checkpoint": {
            "kind": _nonempty_string(envelope.get("kind"), "checkpoint kind"),
            "sha256": _sha256(physical.get("sha256"), "checkpoint sha256"),
            "size_bytes": _positive_int(physical.get("size_bytes"), "checkpoint size_bytes"),
            "logical_identity": {
                "schema_version": _positive_int(
                    identity.get("schema_version"),
                    "checkpoint identity schema_version",
                ),
                "config_sha256": _sha256(
                    identity.get("config_sha256"),
                    "checkpoint config_sha256",
                ),
                "experiment_id": _optional_string(identity.get("experiment_id")),
                "git_sha": _optional_string(identity.get("git_sha")),
                "lock_sha256": _optional_sha256(identity.get("lock_sha256")),
                "data_fingerprints": _sha256_list(identity.get("data_fingerprints", [])),
            },
        },
        "counters": {
            "optimizer_step": _nonnegative_int(
                counters.get("optimizer_step"),
                "checkpoint optimizer_step",
            ),
            "target_tokens": _nonnegative_int(
                counters.get("target_tokens"),
                "checkpoint target_tokens",
            ),
        },
    }


def _validate_payload(payload: Any) -> None:
    top = _exact_mapping(payload, MODEL_ARTIFACT_TOP_LEVEL_KEYS, "model artifact")
    if top["schema_version"] != MODEL_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported model artifact schema")
    if top["artifact_type"] != "inference_model":
        raise ValueError("model artifact type must be inference_model")
    model = _exact_mapping(top["model"], MODEL_KEYS, "model artifact model")
    architecture = _exact_mapping(
        model["architecture"],
        ARCHITECTURE_KEYS,
        "model artifact architecture",
    )
    _positive_int(architecture["embed_size"], "model artifact embed_size")
    _positive_int(architecture["num_heads"], "model artifact num_heads")
    _positive_int(architecture["num_layers"], "model artifact num_layers")
    _dropout(architecture["dropout"])
    _positive_int(architecture["max_sequence_length"], "model artifact max_sequence_length")
    state = _mapping(model["state_dict"], "model artifact state_dict")
    if not state or any(
        not isinstance(name, str) or not name or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("model artifact state_dict must contain only named tensors")
    tokenizer = _exact_mapping(top["tokenizer"], TOKENIZER_KEYS, "model artifact tokenizer")
    _nonempty_string(tokenizer["manifest_path"], "model artifact tokenizer manifest_path")
    _sha256(tokenizer["expected_fingerprint"], "model artifact tokenizer fingerprint")
    source = _exact_mapping(
        top["source_checkpoint"],
        SOURCE_CHECKPOINT_KEYS,
        "model artifact source_checkpoint",
    )
    _nonempty_string(source["kind"], "model artifact checkpoint kind")
    _sha256(source["sha256"], "model artifact checkpoint sha256")
    _positive_int(source["size_bytes"], "model artifact checkpoint size_bytes")
    logical = _exact_mapping(
        source["logical_identity"],
        LOGICAL_IDENTITY_KEYS,
        "model artifact logical_identity",
    )
    _positive_int(logical["schema_version"], "model artifact logical schema_version")
    _sha256(logical["config_sha256"], "model artifact config_sha256")
    _optional_string(logical["experiment_id"])
    _optional_string(logical["git_sha"])
    _optional_sha256(logical["lock_sha256"])
    _sha256_list(logical["data_fingerprints"])
    counters = _exact_mapping(top["counters"], COUNTER_KEYS, "model artifact counters")
    _nonnegative_int(counters["optimizer_step"], "model artifact optimizer_step")
    _nonnegative_int(counters["target_tokens"], "model artifact target_tokens")


def _assert_source_binding(
    payload: Mapping[str, Any],
    expected: tuple[str, str, int],
) -> None:
    source = payload["source_checkpoint"]
    if (source["kind"], source["sha256"], source["size_bytes"]) != expected:
        raise ValueError("model artifact source identity differs from its verified checkpoint")


def _load_with_identity(path: Path) -> tuple[dict[str, Any], ModelArtifactFile]:
    try:
        with path.open("rb") as handle:
            stat = os.fstat(handle.fileno())
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            if handle.tell() != stat.st_size or stat.st_size < 1:
                raise ValueError("model artifact changed while its identity was captured")
            handle.seek(0)
            payload = torch.load(handle, map_location="cpu", weights_only=True)
            final = os.fstat(handle.fileno())
            if (
                final.st_dev,
                final.st_ino,
                final.st_size,
                final.st_mtime_ns,
                final.st_ctime_ns,
            ) != (
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            ):
                raise ValueError("model artifact changed while it was loaded")
    except OSError as error:
        raise ValueError(f"model artifact is unavailable: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("model artifact must contain a mapping")
    return payload, ModelArtifactFile(
        path=str(path),
        sha256=digest.hexdigest(),
        size_bytes=stat.st_size,
        device=stat.st_dev,
        inode=stat.st_ino,
        mtime_ns=stat.st_mtime_ns,
        ctime_ns=stat.st_ctime_ns,
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _exact_mapping(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    result = _mapping(value, label)
    if set(result) != keys:
        raise ValueError(f"{label} keys must be exactly {sorted(keys)}")
    return result


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _dropout(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("model artifact dropout must be a finite number in [0, 1)")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result < 1:
        raise ValueError("model artifact dropout must be a finite number in [0, 1)")
    return result


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, "optional identity value")


def _sha256(value: Any, label: str) -> str:
    result = _nonempty_string(value, label)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _optional_sha256(value: Any) -> str | None:
    if value is None:
        return None
    return _sha256(value, "optional identity SHA-256")


def _sha256_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("data_fingerprints must be a list")
    return [_sha256(item, "data fingerprint") for item in value]


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
