"""Validation for the small set of canonical Hydra training profiles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf


class ConfigPreflightError(ValueError):
    """Raised when a profile is empty, unsafe, or contains a critical typo."""


_TOP_LEVEL = {
    "tokenizer",
    "profile",
    "runtime",
    "reproducibility",
    "data",
    "training",
    "model",
    "artifacts",
    "wandb",
}
_RUNTIME = {"device"}
_REPRODUCIBILITY = {"seed", "deterministic", "reject_dirty"}
_PROFILE = {"name", "purpose", "task"}
_DATA = {"mode", "memorization", "streaming"}
_STREAMING = {
    "output_mode",
    "max_tokens",
    "sequence_length",
    "add_eos",
    "preserve_metadata",
    "require_manifests",
    "seed",
    "prefetch",
    "retry",
    "cache",
    "train",
    "validation",
    "sources",
    "datasets",
}
_SPLIT = {"max_tokens", "add_eos", "preserve_metadata", "require_manifests", "seed", "prefetch", "retry", "cache", "sources", "datasets"}
_SOURCE = {
    "name",
    "type",
    "source",
    "ratio",
    "manifest_path",
    "expected_fingerprint",
    "selection",
    "path",
    "url",
    "revision",
    "repo_id",
    "config_name",
    "split",
    "data_files",
    "text_field",
    "metadata_fields",
    "max_text_chars",
    "timeout_seconds",
    "trust_remote_code",
    "documents",
    "iterable",
}
_TRAINING = {
    "sequence_length",
    "epochs",
    "batch_size",
    "shuffle",
    "optimizer",
    "scheduler",
    "max_steps",
    "max_tokens",
    "max_time",
    "log_every_n_steps",
    "validation_every_n_steps",
    "checkpoint_every_n_steps",
    "milestone_every_n_steps",
    "cadence",
    "ignore_index",
}
_MODEL = {"embed_size", "num_heads", "num_layers", "dropout"}
_ARTIFACTS = {"checkpoints_dir"}
_WANDB = {"enabled", "project", "entity", "name", "mode", "log_model_every_n_epoch"}


def _plain(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else config
    if not isinstance(value, Mapping):
        raise ConfigPreflightError("Hydra configuration must resolve to a mapping")
    return dict(value)


def _check_keys(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigPreflightError(f"unknown critical config key(s) at {path}: {', '.join(unknown)}")


def _required(mapping: Mapping[str, Any], keys: tuple[str, ...], path: str) -> None:
    missing = [key for key in keys if key not in mapping or mapping[key] is None]
    if missing:
        raise ConfigPreflightError(f"missing required config key(s) at {path}: {', '.join(missing)}")


def _check_nested(mapping: Mapping[str, Any], key: str, allowed: set[str], path: str) -> None:
    value = mapping.get(key)
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ConfigPreflightError(f"{path}.{key} must be a mapping")
    _check_keys(value, allowed, f"{path}.{key}")


def _source_identity(source: Mapping[str, Any]) -> tuple[Any, ...]:
    """Identity used to reject accidental train/validation duplication."""

    return (
        source.get("name"),
        source.get("type", source.get("source", "hf")),
        source.get("manifest_path"),
        source.get("expected_fingerprint"),
        source.get("path"),
        source.get("url"),
        source.get("repo_id"),
        source.get("revision"),
        source.get("config_name"),
        source.get("split"),
        source.get("selection"),
    )


def _sources(split: Mapping[str, Any], path: str) -> list[dict[str, Any]]:
    sources = split.get("sources", split.get("datasets"))
    if not isinstance(sources, list) or not sources:
        raise ConfigPreflightError(f"{path}.sources must contain at least one source")
    result: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ConfigPreflightError(f"{path}.sources[{index}] must be a mapping")
        _check_keys(source, _SOURCE, f"{path}.sources[{index}]")
        if not source.get("name"):
            raise ConfigPreflightError(f"{path}.sources[{index}].name is required")
        if source["name"] in names:
            raise ConfigPreflightError(f"duplicate source name in {path}: {source['name']}")
        names.add(str(source["name"]))
        source_type = source.get("type", source.get("source", "hf"))
        if source_type == "manifest":
            _required(source, ("manifest_path", "expected_fingerprint", "selection"), f"{path}.sources[{index}]")
            if source["selection"] not in {"train", "validation", "all"}:
                raise ConfigPreflightError(f"{path}.sources[{index}].selection is invalid")
        result.append(dict(source))
    return result


def validate_training_config(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    """Validate a composed config before tokenizer/data/model initialization."""

    cfg = _plain(config)
    _check_keys(cfg, _TOP_LEVEL, "<root>")
    _required(cfg, ("profile", "runtime", "data", "training", "model", "tokenizer"), "<root>")
    for key, allowed in (("profile", _PROFILE), ("runtime", _RUNTIME), ("reproducibility", _REPRODUCIBILITY), ("data", _DATA), ("training", _TRAINING), ("model", _MODEL), ("artifacts", _ARTIFACTS), ("wandb", _WANDB)):
        _check_nested(cfg, key, allowed, "<root>")

    reproducibility = _plain(cfg["reproducibility"])
    _required(reproducibility, ("seed",), "reproducibility")
    if (
        isinstance(reproducibility["seed"], bool)
        or not isinstance(reproducibility["seed"], int)
        or reproducibility["seed"] < 0
    ):
        raise ConfigPreflightError("reproducibility.seed must be a non-negative integer")

    profile = _plain(cfg["profile"])
    _required(profile, ("name",), "profile")
    profile_name = str(profile["name"])
    if profile_name == "evaluation" or profile.get("purpose") == "evaluation" or profile.get("task") == "evaluate_checkpoint":
        raise ConfigPreflightError(
            "profile=evaluation is composition-only until the evaluation ticket; "
            "the training entrypoint cannot run it"
        )
    data = _plain(cfg["data"])
    _required(data, ("mode",), "data")
    expected_profile = {
        "smoke_overfit": ("memorization_smoke", "memorization_smoke"),
        "pretrain_streaming": ("streaming", "pretraining"),
    }.get(profile_name)
    if expected_profile is None:
        raise ConfigPreflightError(f"unknown training profile: {profile_name!r}")
    expected_mode, expected_purpose = expected_profile
    if data["mode"] != expected_mode or profile.get("purpose") != expected_purpose:
        raise ConfigPreflightError(
            f"profile {profile_name!r} must use data.mode={expected_mode!r} "
            f"and profile.purpose={expected_purpose!r}"
        )
    training = _plain(cfg["training"])
    _required(training, ("sequence_length", "epochs", "batch_size"), "training")
    if int(training["sequence_length"]) < 2 or int(training["batch_size"]) < 1 or int(training["epochs"]) < 1:
        raise ConfigPreflightError("training sequence_length, batch_size, and epochs must be positive")
    for budget_name in ("max_steps", "max_tokens", "max_time"):
        budget = training.get(budget_name)
        if budget is not None and float(budget) <= 0:
            raise ConfigPreflightError(f"training.{budget_name} must be positive when configured")
    cadence = training.get("cadence")
    if cadence is not None:
        if not isinstance(cadence, Mapping):
            raise ConfigPreflightError("training.cadence must be a mapping")
        _check_keys(
            cadence,
            {"log_every_n_steps", "validation_every_n_steps", "checkpoint_every_n_steps", "milestone_every_n_steps"},
            "training.cadence",
        )
    for cadence_name in (
        "log_every_n_steps",
        "validation_every_n_steps",
        "checkpoint_every_n_steps",
        "milestone_every_n_steps",
    ):
        cadence_value = training.get(cadence_name)
        if cadence_value is not None and int(cadence_value) < 1:
            raise ConfigPreflightError(f"training.{cadence_name} must be positive when configured")
    if data["mode"] == "memorization_smoke":
        if profile.get("purpose") != "memorization_smoke":
            raise ConfigPreflightError("memorization_smoke is only allowed in the smoke_overfit profile")
        memory = data.get("memorization")
        if not isinstance(memory, Mapping):
            raise ConfigPreflightError("data.memorization is required for memorization_smoke")
        _check_keys(memory, {"manifest_path", "expected_fingerprint"}, "data.memorization")
        _required(memory, ("manifest_path", "expected_fingerprint"), "data.memorization")
    elif data["mode"] == "streaming":
        streaming = data.get("streaming")
        if not isinstance(streaming, Mapping):
            raise ConfigPreflightError("data.streaming is required for streaming profiles")
        _check_keys(streaming, _STREAMING, "data.streaming")
        if streaming.get("require_manifests") is not True:
            raise ConfigPreflightError("real streaming profiles must set data.streaming.require_manifests=true")
        train = streaming.get("train")
        validation = streaming.get("validation")
        if not isinstance(train, Mapping) or not isinstance(validation, Mapping):
            raise ConfigPreflightError("data.streaming.train and data.streaming.validation are required")
        _check_keys(train, _SPLIT, "data.streaming.train")
        _check_keys(validation, _SPLIT, "data.streaming.validation")
        train_sources = _sources(train, "data.streaming.train")
        validation_sources = _sources(validation, "data.streaming.validation")
        if train_sources == validation_sources or {
            _source_identity(item) for item in train_sources
        } == {_source_identity(item) for item in validation_sources}:
            raise ConfigPreflightError("real train and validation sources must be distinct")
    else:
        raise ConfigPreflightError(f"unsupported data.mode: {data['mode']!r}")
    return cfg
