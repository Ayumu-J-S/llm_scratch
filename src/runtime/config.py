"""Validation for the small set of canonical Hydra training profiles."""

from __future__ import annotations

import math
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
    "evaluation",
    "benchmark",
    "measurement",
}
_RUNTIME = {"device"}
_REPRODUCIBILITY = {"seed", "deterministic", "reject_dirty"}
_PROFILE = {"name", "purpose", "task"}
_DATA = {"mode", "memorization", "streaming"}
_STREAMING = {
    "output_mode",
    "max_tokens",
    "max_target_tokens",
    "mixture_basis",
    "sequence_length",
    "add_eos",
    "preserve_metadata",
    "measure_performance",
    "require_manifests",
    "seed",
    "prefetch",
    "retry",
    "cache",
    "repeat",
    "horizon",
    "shuffle",
    "shuffle_buffer_size",
    "train",
    "validation",
    "sources",
    "datasets",
}
_SPLIT = {
    "max_tokens",
    "max_target_tokens",
    "mixture_basis",
    "add_eos",
    "preserve_metadata",
    "measure_performance",
    "require_manifests",
    "seed",
    "prefetch",
    "retry",
    "cache",
    "repeat",
    "horizon",
    "shuffle",
    "shuffle_buffer_size",
    "sources",
    "datasets",
}
_CACHE = {"dir", "max_size_bytes", "min_free_bytes", "wait_timeout_seconds"}
_PREFETCH = {"enabled", "buffer_size", "mode"}
_RETRY = {"max_attempts", "initial_delay_seconds", "max_delay_seconds", "multiplier"}
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
    "log_every_n_tokens",
    "validation_every_n_steps",
    "validation_every_n_tokens",
    "checkpoint_every_n_steps",
    "checkpoint_every_n_tokens",
    "milestone_every_n_steps",
    "milestone_every_n_tokens",
    "cadence",
    "ignore_index",
    "precision",
    "gradient_accumulation_steps",
    "max_grad_norm",
}
_MODEL = {"embed_size", "num_heads", "num_layers", "dropout"}
_ARTIFACTS = {"checkpoints_dir", "keep_last_n", "resume_path"}
_WANDB = {"enabled", "project", "entity", "name", "mode", "log_model_every_n_epoch"}
_EVALUATION = {"checkpoint_path", "output_path", "device", "wandb"}
_EVALUATION_WANDB = {"enabled", "project", "entity", "name", "mode"}
_BENCHMARK = {"checkpoint_path", "output_path", "device", "cache", "wandb"}
_BENCHMARK_CACHE = {
    "dir",
    "max_size_bytes",
    "min_free_bytes",
    "wait_timeout_seconds",
    "timeout_seconds",
}
_BENCHMARK_WANDB = {"enabled", "project", "entity", "name", "mode"}
_MEASUREMENT = {"enabled", "warmup_optimizer_steps", "cuda_events", "output_path"}


def _plain(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    value = (
        OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else config
    )
    if not isinstance(value, Mapping):
        raise ConfigPreflightError("Hydra configuration must resolve to a mapping")
    return dict(value)


def _check_keys(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigPreflightError(
            f"unknown critical config key(s) at {path}: {', '.join(unknown)}"
        )


def _required(mapping: Mapping[str, Any], keys: tuple[str, ...], path: str) -> None:
    missing = [key for key in keys if key not in mapping or mapping[key] is None]
    if missing:
        raise ConfigPreflightError(
            f"missing required config key(s) at {path}: {', '.join(missing)}"
        )


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
            _required(
                source,
                ("manifest_path", "expected_fingerprint", "selection"),
                f"{path}.sources[{index}]",
            )
            if source["selection"] not in {"train", "validation", "all"}:
                raise ConfigPreflightError(f"{path}.sources[{index}].selection is invalid")
        result.append(dict(source))
    return result


def validate_evaluation_config(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    """Validate standalone operational controls without overriding checkpoint authority."""

    cfg = _plain(config)
    _check_keys(cfg, _TOP_LEVEL, "<root>")
    _required(cfg, ("profile", "evaluation"), "<root>")
    profile = _plain(cfg["profile"])
    if (
        profile.get("name") != "evaluation"
        or profile.get("purpose") != "evaluation"
        or profile.get("task") != "evaluate_checkpoint"
    ):
        raise ConfigPreflightError("standalone evaluation requires profile=evaluation")
    evaluation = _plain(cfg["evaluation"])
    _check_keys(evaluation, _EVALUATION, "evaluation")
    _required(evaluation, ("checkpoint_path", "output_path", "device"), "evaluation")
    for key in ("checkpoint_path", "output_path", "device"):
        if not isinstance(evaluation[key], str) or not evaluation[key].strip():
            raise ConfigPreflightError(f"evaluation.{key} must be a non-empty string")
    if evaluation["device"] not in {"cpu", "cuda"}:
        raise ConfigPreflightError("evaluation.device must be either 'cpu' or 'cuda'")
    wandb_config = evaluation.get("wandb", {})
    if not isinstance(wandb_config, Mapping):
        raise ConfigPreflightError("evaluation.wandb must be a mapping")
    _check_keys(wandb_config, _EVALUATION_WANDB, "evaluation.wandb")
    enabled = wandb_config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigPreflightError("evaluation.wandb.enabled must be boolean")
    mode = wandb_config.get("mode", "online")
    if mode not in {"online", "offline"}:
        raise ConfigPreflightError("evaluation.wandb.mode must be either 'online' or 'offline'")
    return cfg


def validate_evaluation_checkpoint_runtime(
    config: Mapping[str, Any] | DictConfig,
    checkpoint_config: Mapping[str, Any] | DictConfig,
) -> None:
    """Reject evaluator device choices incompatible with checkpoint-owned precision."""

    evaluation = _plain(_plain(config)["evaluation"])
    training = _plain(_plain(checkpoint_config)["training"])
    device = str(evaluation["device"])
    precision = str(training.get("precision", "fp32"))
    if device == "cpu" and precision == "bf16":
        raise ConfigPreflightError(
            "evaluation.device=cpu is incompatible with checkpoint-owned "
            "training.precision=bf16; use evaluation.device=cuda or evaluate an "
            "FP32 checkpoint"
        )


def validate_benchmark_config(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    """Validate benchmark operational controls without exposing data access authority."""

    cfg = _plain(config)
    _check_keys(cfg, _TOP_LEVEL, "<root>")
    _required(cfg, ("profile", "benchmark"), "<root>")
    profile = _plain(cfg["profile"])
    if (
        profile.get("name") != "benchmark"
        or profile.get("purpose") != "benchmark"
        or profile.get("task") != "benchmark_checkpoint"
    ):
        raise ConfigPreflightError("standalone benchmark requires profile=benchmark")
    benchmark = _plain(cfg["benchmark"])
    _check_keys(benchmark, _BENCHMARK, "benchmark")
    _required(benchmark, ("checkpoint_path", "output_path", "device", "cache"), "benchmark")
    for key in ("checkpoint_path", "output_path", "device"):
        if not isinstance(benchmark[key], str) or not benchmark[key].strip():
            raise ConfigPreflightError(f"benchmark.{key} must be a non-empty string")
    if benchmark["device"] not in {"cpu", "cuda"}:
        raise ConfigPreflightError("benchmark.device must be either 'cpu' or 'cuda'")
    cache = benchmark["cache"]
    if not isinstance(cache, Mapping):
        raise ConfigPreflightError("benchmark.cache must be a mapping")
    _check_keys(cache, _BENCHMARK_CACHE, "benchmark.cache")
    _required(
        cache,
        ("dir", "max_size_bytes", "min_free_bytes", "wait_timeout_seconds", "timeout_seconds"),
        "benchmark.cache",
    )
    if not isinstance(cache["dir"], str) or not cache["dir"].strip():
        raise ConfigPreflightError("benchmark.cache.dir must be a non-empty path string")
    for field in ("max_size_bytes", "min_free_bytes"):
        value = cache[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigPreflightError(f"benchmark.cache.{field} must be a non-negative integer")
    if cache["max_size_bytes"] < 1:
        raise ConfigPreflightError("benchmark.cache.max_size_bytes must be positive")
    for field in ("wait_timeout_seconds", "timeout_seconds"):
        try:
            valid = math.isfinite(float(cache[field])) and float(cache[field]) > 0
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ConfigPreflightError(f"benchmark.cache.{field} must be positive and finite")
    wandb_config = benchmark.get("wandb", {})
    if not isinstance(wandb_config, Mapping):
        raise ConfigPreflightError("benchmark.wandb must be a mapping")
    _check_keys(wandb_config, _BENCHMARK_WANDB, "benchmark.wandb")
    if not isinstance(wandb_config.get("enabled", False), bool):
        raise ConfigPreflightError("benchmark.wandb.enabled must be boolean")
    if wandb_config.get("mode", "online") not in {"online", "offline"}:
        raise ConfigPreflightError("benchmark.wandb.mode must be either 'online' or 'offline'")
    return cfg


def validate_benchmark_checkpoint_runtime(
    config: Mapping[str, Any] | DictConfig,
    checkpoint_config: Mapping[str, Any] | DictConfig,
) -> None:
    """Reject benchmark device choices incompatible with checkpoint-owned precision."""

    benchmark = _plain(_plain(config)["benchmark"])
    training = _plain(_plain(checkpoint_config)["training"])
    device = str(benchmark["device"])
    precision = str(training.get("precision", "fp32"))
    if device == "cpu" and precision == "bf16":
        raise ConfigPreflightError(
            "benchmark.device=cpu is incompatible with checkpoint-owned "
            "training.precision=bf16; use benchmark.device=cuda or benchmark an FP32 checkpoint"
        )


def validate_training_config(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    """Validate a composed config before tokenizer/data/model initialization."""

    cfg = _plain(config)
    _check_keys(cfg, _TOP_LEVEL, "<root>")
    _required(cfg, ("profile", "runtime", "data", "training", "model", "tokenizer"), "<root>")
    for key, allowed in (
        ("profile", _PROFILE),
        ("runtime", _RUNTIME),
        ("reproducibility", _REPRODUCIBILITY),
        ("data", _DATA),
        ("training", _TRAINING),
        ("model", _MODEL),
        ("artifacts", _ARTIFACTS),
        ("wandb", _WANDB),
        ("evaluation", _EVALUATION),
        ("benchmark", _BENCHMARK),
        ("measurement", _MEASUREMENT),
    ):
        _check_nested(cfg, key, allowed, "<root>")

    reproducibility = _plain(cfg["reproducibility"])
    _required(reproducibility, ("seed",), "reproducibility")
    if (
        isinstance(reproducibility["seed"], bool)
        or not isinstance(reproducibility["seed"], int)
        or reproducibility["seed"] < 0
    ):
        raise ConfigPreflightError("reproducibility.seed must be a non-negative integer")

    measurement = _plain(cfg.get("measurement", {}))
    if measurement:
        for key in ("enabled", "cuda_events"):
            value = measurement.get(key)
            if not isinstance(value, bool):
                raise ConfigPreflightError(f"measurement.{key} must be boolean")
        warmup_steps = measurement.get("warmup_optimizer_steps")
        if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
            raise ConfigPreflightError(
                "measurement.warmup_optimizer_steps must be a non-negative integer"
            )
        output_path = measurement.get("output_path")
        if output_path is not None and (
            not isinstance(output_path, str) or not output_path.strip()
        ):
            raise ConfigPreflightError(
                "measurement.output_path must be null or a non-empty path string"
            )

    profile = _plain(cfg["profile"])
    _required(profile, ("name",), "profile")
    profile_name = str(profile["name"])
    if (
        profile_name in {"evaluation", "benchmark"}
        or profile.get("purpose") in {"evaluation", "benchmark"}
        or profile.get("task") in {"evaluate_checkpoint", "benchmark_checkpoint"}
    ):
        raise ConfigPreflightError(
            "evaluation and benchmark profiles are standalone-only; "
            "the training entrypoint cannot run them"
        )
    data = _plain(cfg["data"])
    _required(data, ("mode",), "data")
    expected_profile = {
        "smoke_overfit": ("memorization_smoke", "memorization_smoke"),
        "pretrain_streaming": ("streaming", "pretraining"),
        "stability_smoke": ("streaming", "pretraining"),
        "gate_overfit": ("streaming", "memorization_gate"),
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
    if (
        int(training["sequence_length"]) < 2
        or int(training["batch_size"]) < 1
        or int(training["epochs"]) < 1
    ):
        raise ConfigPreflightError(
            "training sequence_length, batch_size, and epochs must be positive"
        )
    if training.get("precision", "fp32") not in {"fp32", "bf16"}:
        raise ConfigPreflightError("training.precision must be either 'fp32' or 'bf16'")
    accumulation_steps = training.get("gradient_accumulation_steps", 1)
    if (
        isinstance(accumulation_steps, bool)
        or not isinstance(accumulation_steps, int)
        or accumulation_steps < 1
    ):
        raise ConfigPreflightError(
            "training.gradient_accumulation_steps must be a positive integer"
        )
    max_grad_norm = training.get("max_grad_norm")
    if max_grad_norm is not None:
        try:
            valid_max_grad_norm = math.isfinite(float(max_grad_norm)) and float(max_grad_norm) > 0.0
        except (TypeError, ValueError):
            valid_max_grad_norm = False
        if not valid_max_grad_norm:
            raise ConfigPreflightError(
                "training.max_grad_norm must be a positive finite number or null"
            )
    optimizer = training.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise ConfigPreflightError("training.optimizer must be a mapping")
    _required(optimizer, ("_target_", "lr", "betas", "eps", "weight_decay"), "training.optimizer")
    if optimizer.get("_target_") != "torch.optim.AdamW":
        raise ConfigPreflightError("training.optimizer._target_ must be torch.optim.AdamW")
    scheduler = training.get("scheduler")
    if not isinstance(scheduler, Mapping):
        raise ConfigPreflightError("training.scheduler must be a mapping")
    if scheduler.get("enabled", True):
        _required(
            scheduler,
            ("_target_", "interval", "warmup_steps", "decay_steps", "min_lr_ratio"),
            "training.scheduler",
        )
        if scheduler.get("_target_") != "training.optimization.WarmupCosineScheduler":
            raise ConfigPreflightError("enabled training.scheduler must use WarmupCosineScheduler")
        if scheduler.get("interval") != "step":
            raise ConfigPreflightError("enabled training.scheduler.interval must be 'step'")
        warmup_steps = scheduler["warmup_steps"]
        decay_steps = scheduler["decay_steps"]
        if (
            isinstance(warmup_steps, bool)
            or not isinstance(warmup_steps, int)
            or warmup_steps < 0
            or isinstance(decay_steps, bool)
            or not isinstance(decay_steps, int)
            or decay_steps < 1
            or decay_steps < warmup_steps
        ):
            raise ConfigPreflightError(
                "scheduler warmup_steps/decay_steps must be ordered non-negative/positive integers"
            )
    for budget_name in ("max_steps", "max_tokens", "max_time"):
        budget = training.get(budget_name)
        if budget is None:
            continue
        if budget_name in {"max_steps", "max_tokens"}:
            if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
                raise ConfigPreflightError(
                    f"training.{budget_name} must be a positive integer when configured"
                )
        elif float(budget) <= 0:
            raise ConfigPreflightError(f"training.{budget_name} must be positive when configured")
    cadence = training.get("cadence")
    if cadence is not None:
        if not isinstance(cadence, Mapping):
            raise ConfigPreflightError("training.cadence must be a mapping")
        _check_keys(
            cadence,
            {
                "log_every_n_steps",
                "log_every_n_tokens",
                "validation_every_n_steps",
                "validation_every_n_tokens",
                "checkpoint_every_n_steps",
                "checkpoint_every_n_tokens",
                "milestone_every_n_steps",
                "milestone_every_n_tokens",
            },
            "training.cadence",
        )
    for cadence_name in (
        "log_every_n_steps",
        "log_every_n_tokens",
        "validation_every_n_steps",
        "validation_every_n_tokens",
        "checkpoint_every_n_steps",
        "checkpoint_every_n_tokens",
        "milestone_every_n_steps",
        "milestone_every_n_tokens",
    ):
        cadence_value = training.get(cadence_name)
        if cadence_value is not None and (
            isinstance(cadence_value, bool)
            or not isinstance(cadence_value, int)
            or cadence_value < 1
        ):
            raise ConfigPreflightError(f"training.{cadence_name} must be positive when configured")
    artifacts = _plain(cfg.get("artifacts", {}))
    if artifacts:
        _required(artifacts, ("checkpoints_dir", "keep_last_n"), "artifacts")
        if (
            not isinstance(artifacts["checkpoints_dir"], str)
            or not artifacts["checkpoints_dir"].strip()
        ):
            raise ConfigPreflightError("artifacts.checkpoints_dir must be a non-empty path string")
        keep_last_n = artifacts["keep_last_n"]
        if isinstance(keep_last_n, bool) or not isinstance(keep_last_n, int) or keep_last_n < 1:
            raise ConfigPreflightError("artifacts.keep_last_n must be a positive integer")
        resume_path = artifacts.get("resume_path")
        if resume_path is not None and (
            not isinstance(resume_path, str) or not resume_path.strip()
        ):
            raise ConfigPreflightError(
                "artifacts.resume_path must be null or a non-empty checkpoint path"
            )
    if data["mode"] == "memorization_smoke":
        if profile.get("purpose") != "memorization_smoke":
            raise ConfigPreflightError(
                "memorization_smoke is only allowed in the smoke_overfit profile"
            )
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
        for key, allowed in (("cache", _CACHE), ("prefetch", _PREFETCH), ("retry", _RETRY)):
            _check_nested(streaming, key, allowed, "data.streaming")
        cache = streaming.get("cache")
        if isinstance(cache, Mapping):
            for field in ("max_size_bytes", "min_free_bytes"):
                value = cache.get(field)
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                ):
                    raise ConfigPreflightError(
                        f"data.streaming.cache.{field} must be a non-negative integer"
                    )
            if int(cache.get("max_size_bytes", 0)) < 1:
                raise ConfigPreflightError("data.streaming.cache.max_size_bytes must be positive")
            if float(cache.get("wait_timeout_seconds", 30.0)) <= 0:
                raise ConfigPreflightError(
                    "data.streaming.cache.wait_timeout_seconds must be positive"
                )
        if streaming.get("require_manifests") is not True:
            raise ConfigPreflightError(
                "real streaming profiles must set data.streaming.require_manifests=true"
            )
        train = streaming.get("train")
        validation = streaming.get("validation")
        if not isinstance(train, Mapping) or not isinstance(validation, Mapping):
            raise ConfigPreflightError(
                "data.streaming.train and data.streaming.validation are required"
            )
        _check_keys(train, _SPLIT, "data.streaming.train")
        _check_keys(validation, _SPLIT, "data.streaming.validation")
        train_sources = _sources(train, "data.streaming.train")
        validation_sources = _sources(validation, "data.streaming.validation")
        for split_name, split in (("train", train), ("validation", validation)):
            basis = split.get("mixture_basis", streaming.get("mixture_basis", "tokenizer_tokens"))
            if basis not in {"tokenizer_tokens", "trained_targets"}:
                raise ConfigPreflightError(f"data.streaming.{split_name}.mixture_basis is invalid")
            target_budget = split.get("max_target_tokens", streaming.get("max_target_tokens"))
            if basis == "trained_targets" and (
                isinstance(target_budget, bool)
                or not isinstance(target_budget, int)
                or target_budget < 1
            ):
                raise ConfigPreflightError(
                    f"data.streaming.{split_name} trained_targets mixture requires "
                    "a positive max_target_tokens"
                )
            if basis == "trained_targets" and target_budget % int(training["sequence_length"]) != 0:
                raise ConfigPreflightError(
                    f"data.streaming.{split_name}.max_target_tokens must be divisible by "
                    "training.sequence_length so every packed batch has a full target window"
                )
        train_identities = {_source_identity(item) for item in train_sources}
        validation_identities = {_source_identity(item) for item in validation_sources}
        if train_sources == validation_sources or train_identities & validation_identities:
            raise ConfigPreflightError("real train and validation sources must be distinct")
    else:
        raise ConfigPreflightError(f"unsupported data.mode: {data['mode']!r}")
    return cfg
