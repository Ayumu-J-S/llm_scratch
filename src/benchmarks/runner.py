"""Checkpoint-owned BENCH-001 execution and compact evidence output."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from benchmarks.contamination import scan_checkpoint_training_data
from benchmarks.scoring import score_suite, validate_suite_context
from benchmarks.suite import (
    CANONICAL_REGISTRY_FINGERPRINT,
    CANONICAL_REGISTRY_PATH,
    FINAL_ACKNOWLEDGEMENT,
    BenchmarkAccess,
    load_suite,
)
from data.identity import canonical_json_bytes
from data.stream_loader.cache import BoundedShardCache
from generation.sampler import CheckpointSampler
from runtime.config import (
    validate_benchmark_checkpoint_runtime,
    validate_benchmark_config,
    validate_training_config,
)
from runtime.device import select_device
from runtime.evaluation import (
    apply_evaluation_determinism_policy as _apply_evaluation_determinism_policy,
)
from runtime.environment import collect_environment
from runtime.reproducibility import collect_git_identity, sha256_file
from training.checkpoint import load_checkpoint_for_generation
from training.wandb_tracking import call_bounded, finish_run_bounded


ROOT_DIR = Path(__file__).resolve().parents[2]
SUITE_REGISTRY_PATH = CANONICAL_REGISTRY_PATH
SUITE_REGISTRY_FINGERPRINT = CANONICAL_REGISTRY_FINGERPRINT


class BenchmarkContaminationError(RuntimeError):
    """Scoring was blocked after retaining the completed contamination report."""

    def __init__(self, output_path: Path) -> None:
        super().__init__(f"benchmark contamination was detected; blocked evidence: {output_path}")
        self.output_path = output_path


def run_benchmark(
    cfg: DictConfig,
    *,
    access: BenchmarkAccess = "dev",
    final_acknowledgement: str | None = None,
    registry_path: str | Path = SUITE_REGISTRY_PATH,
    expected_registry_fingerprint: str = SUITE_REGISTRY_FINGERPRINT,
) -> Path:
    """Evaluate one checkpoint after a complete contamination gate.

    The public development entrypoint never supplies ``access=final``.  The
    reserved entrypoint obtains its acknowledgement from the environment
    before Hydra parses overrides, so no Hydra key can grant final-test access.
    Registry overrides exist only for offline acceptance fixtures and are not
    exposed by either console command.
    """

    determinism_policy = _apply_evaluation_determinism_policy()
    validate_benchmark_config(cfg)
    if access == "final" and final_acknowledgement != FINAL_ACKNOWLEDGEMENT:
        raise PermissionError(
            f"reserved final benchmark requires BENCHMARK_FINAL_ACK={FINAL_ACKNOWLEDGEMENT}"
        )
    if access not in {"dev", "final"}:
        raise ValueError("benchmark access must be dev or final")
    benchmark_cfg = cfg.benchmark
    checkpoint_path = _root_or_cwd_path(str(benchmark_cfg.checkpoint_path))
    configured_output_root, configured_output_path = _output_paths(benchmark_cfg)
    sampler, checkpoint_cfg = _load_benchmark_sampler(checkpoint_path, cfg=cfg)
    cache_path = _repository_path(str(benchmark_cfg.cache.dir))
    output_path = (
        None
        if configured_output_path is None
        else _validated_benchmark_output_path(
            checkpoint_path,
            configured_output_root,
            configured_output_path,
            checkpoint_config=checkpoint_cfg,
            cache_path=cache_path,
        )
    )
    cache = _benchmark_cache(benchmark_cfg.cache)
    suite = load_suite(
        registry_path,
        expected_fingerprint=expected_registry_fingerprint,
        access=access,
        cache=cache,
        timeout_seconds=float(benchmark_cfg.cache.timeout_seconds),
    )
    context_preflight = validate_suite_context(sampler, suite)
    evaluator_identity = _evaluator_identity()
    evaluation_identity = _evaluation_identity(
        sampler,
        suite.identity(),
        context_preflight=context_preflight,
        determinism_policy=determinism_policy,
        evaluator_identity=evaluator_identity,
    )
    evaluation_identity_sha256 = hashlib.sha256(
        canonical_json_bytes(evaluation_identity)
    ).hexdigest()
    if output_path is None:
        output_path = _validated_benchmark_output_path(
            checkpoint_path,
            configured_output_root,
            _identity_bound_output_path(
                configured_output_root,
                access=access,
                evaluation_identity_sha256=evaluation_identity_sha256,
            ),
            checkpoint_config=checkpoint_cfg,
            cache_path=cache_path,
        )
    contamination = scan_checkpoint_training_data(
        sampler.resolved_config,
        suite,
        fallback_cache=cache,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "repository_checkpoint_benchmark",
        "status": "blocked_contamination" if contamination["contaminated"] else "complete",
        "evaluation_identity": evaluation_identity,
        "evaluation_identity_sha256": evaluation_identity_sha256,
        "contamination": contamination,
        "tasks": {},
    }
    if contamination["contaminated"]:
        _write_json_atomic(output_path, result)
        raise BenchmarkContaminationError(output_path)

    result["tasks"] = score_suite(sampler, suite)
    _write_json_atomic(output_path, result)
    _maybe_log_wandb(
        benchmark_cfg,
        result,
        local_result_identity={
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "size_bytes": output_path.stat().st_size,
        },
    )
    return output_path


def _load_benchmark_sampler(
    checkpoint_path: Path,
    *,
    cfg: DictConfig,
) -> tuple[CheckpointSampler, DictConfig]:
    """Build the sampler without retaining optimizer-bearing checkpoint state."""

    loaded = load_checkpoint_for_generation(checkpoint_path)
    checkpoint_cfg = OmegaConf.create(loaded.payload["state"]["resolved_config"])
    validate_training_config(checkpoint_cfg)
    validate_benchmark_checkpoint_runtime(cfg, checkpoint_cfg)
    device = select_device(str(cfg.benchmark.device))
    sampler = CheckpointSampler.from_loaded_checkpoint(
        checkpoint_path,
        loaded,
        device=device,
    )
    return sampler, checkpoint_cfg


def _evaluation_identity(
    sampler: CheckpointSampler,
    suite_identity: Mapping[str, Any],
    *,
    context_preflight: Mapping[str, Any],
    determinism_policy: Mapping[str, Any],
    evaluator_identity: Mapping[str, Any],
) -> dict[str, Any]:
    physical = sampler.physical_checkpoint_identity
    return {
        "checkpoint": {
            "kind": sampler.checkpoint_kind,
            "logical": sampler.logical_checkpoint_identity,
            "physical": {
                "sha256": physical["sha256"],
                "size_bytes": physical["size_bytes"],
            },
        },
        "tokenizer_fingerprint": sampler.tokenizer.fingerprint,
        "device": str(sampler.device),
        "precision": str(sampler.resolved_config.get("training", {}).get("precision", "fp32")),
        "context_preflight": dict(context_preflight),
        "determinism_policy": dict(determinism_policy),
        "evaluator": dict(evaluator_identity),
        "suite": dict(suite_identity),
    }


def _evaluator_identity() -> dict[str, Any]:
    """Bind scores to the executable revision, dependency lock, and runtime stack."""

    environment = collect_environment()
    cuda = environment["cuda"]
    torch_identity = environment["torch"]
    return {
        "git": collect_git_identity(ROOT_DIR),
        "lock_sha256": sha256_file(ROOT_DIR / "uv.lock"),
        "environment": {
            "os": environment["os"],
            "os_release": environment["os_release"],
            "architecture": environment["architecture"],
            "python": environment["python"],
            "torch": {
                "version": torch_identity["version"],
                "compiled_cuda": torch_identity["compiled_cuda"],
            },
            "cuda": {
                "available": cuda["available"],
                "runtime_version": cuda["runtime_version"],
                "driver_version": cuda["driver_version"],
                "devices": cuda["devices"],
                "bf16_supported": cuda["bf16_supported"],
            },
            "container_image": environment["container_image"],
        },
    }


def _maybe_log_wandb(
    benchmark_cfg: DictConfig,
    result: Mapping[str, Any],
    *,
    local_result_identity: Mapping[str, Any],
) -> None:
    wandb_cfg = benchmark_cfg.get("wandb", {}) or {}
    mode = str(wandb_cfg.get("mode", "disabled"))
    if mode == "disabled":
        return
    identity = result["evaluation_identity"]
    run = None
    try:
        init_timeout = float(wandb_cfg.get("init_timeout_seconds", 10.0))
        finish_timeout = float(wandb_cfg.get("finish_timeout_seconds", 30.0))
        if mode == "online":
            login_succeeded = call_bounded(
                lambda: wandb.login(
                    force=True,
                    verify=True,
                    timeout=max(1, math.ceil(init_timeout)),
                ),
                timeout_seconds=init_timeout,
                operation="W&B benchmark login verification",
            )
            if not login_succeeded:
                raise RuntimeError("verified W&B login did not succeed")
        run = call_bounded(
            lambda: wandb.init(
                project=wandb_cfg.get("project"),
                entity=wandb_cfg.get("entity"),
                name=wandb_cfg.get("name"),
                mode=mode,
                config={
                    "benchmark_evaluation_identity_sha256": result["evaluation_identity_sha256"],
                    "benchmark_suite": identity["suite"],
                    "checkpoint_identity": identity["checkpoint"],
                    "tokenizer_fingerprint": identity["tokenizer_fingerprint"],
                },
                settings=wandb.Settings(init_timeout=init_timeout),
            ),
            timeout_seconds=init_timeout,
            operation="W&B benchmark initialization",
            on_late_result=lambda late_run: (
                finish_run_bounded(late_run, timeout_seconds=finish_timeout)
                if late_run is not None
                else None
            ),
        )
        if run is None:
            raise RuntimeError("wandb.init returned no run")
        rows = []
        summary: dict[str, Any] = {
            "benchmark/evaluation_identity_sha256": result["evaluation_identity_sha256"],
            "benchmark/local_result_identity": dict(local_result_identity),
            "benchmark/contamination": {
                "scan_complete": result["contamination"]["scan_complete"],
                "scanned_documents": result["contamination"]["scanned_documents"],
                "match_counts": result["contamination"]["match_counts"],
            },
        }
        for task_name, task_result in sorted(result["tasks"].items()):
            metric = str(task_result["primary_metric"])
            value = float(task_result[metric])
            rows.append(
                [
                    task_name,
                    identity["suite"]["access"],
                    metric,
                    value,
                    int(task_result["correct"]),
                    int(task_result["total"]),
                    identity["suite"]["protocol_sha256"],
                ]
            )
            summary[f"benchmark/{task_name}/{metric}"] = value
            summary[f"benchmark/{task_name}/correct"] = int(task_result["correct"])
            summary[f"benchmark/{task_name}/total"] = int(task_result["total"])
        call_bounded(
            lambda: run.summary.update(summary),
            timeout_seconds=init_timeout,
            operation="W&B benchmark summary update",
        )
        table = call_bounded(
            lambda: wandb.Table(
                columns=[
                    "task",
                    "access",
                    "metric",
                    "value",
                    "correct",
                    "total",
                    "protocol_sha256",
                ],
                data=rows,
            ),
            timeout_seconds=init_timeout,
            operation="W&B benchmark table construction",
        )
        call_bounded(
            lambda: run.log({"benchmark/results": table}),
            timeout_seconds=init_timeout,
            operation="W&B benchmark table log",
        )
    except Exception as error:
        logger.warning(
            "W&B benchmark logging failed after local result commit: {}",
            error,
        )
    finally:
        if run is not None:
            try:
                finish_run_bounded(
                    run,
                    timeout_seconds=float(wandb_cfg.get("finish_timeout_seconds", 30.0)),
                )
            except Exception as error:
                logger.warning(
                    "W&B benchmark finish failed after local result commit: {}",
                    error,
                )


def _benchmark_cache(cache_cfg: DictConfig) -> BoundedShardCache:
    cache_path = _repository_path(str(cache_cfg.dir))
    return BoundedShardCache(
        cache_path,
        max_size_bytes=int(cache_cfg.max_size_bytes),
        min_free_bytes=int(cache_cfg.min_free_bytes),
        wait_timeout_seconds=float(cache_cfg.wait_timeout_seconds),
    )


def _output_paths(benchmark_cfg: DictConfig) -> tuple[Path, Path | None]:
    root = Path(str(benchmark_cfg.output_root))
    configured_root = root if root.is_absolute() else ROOT_DIR / root
    configured_path = benchmark_cfg.get("output_path")
    if configured_path is None:
        return configured_root, None
    path = Path(str(configured_path))
    return configured_root, path if path.is_absolute() else configured_root / path


def _identity_bound_output_path(
    output_root: Path,
    *,
    access: BenchmarkAccess,
    evaluation_identity_sha256: str,
) -> Path:
    """Derive a collision-resistant result name from the complete score identity."""

    return output_root / f"{access}-{evaluation_identity_sha256}.json"


def _root_or_cwd_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    root_candidate = (ROOT_DIR / path).resolve()
    return root_candidate if root_candidate.exists() else (Path.cwd() / path).resolve()


def _repository_path(value: str) -> Path:
    """Resolve a configured repository-owned path from one stable base."""

    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT_DIR / path).resolve()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one complete JSON result without ever replacing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(
                f"result output already exists and will not be replaced: {path}"
            ) from error
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_benchmark_output_path(
    checkpoint_path: Path,
    configured_output_root: Path,
    output_path: Path,
    *,
    checkpoint_config: DictConfig,
    cache_path: Path,
) -> Path:
    """Return one new JSON path inside the dedicated benchmark-result namespace."""

    checkpoint = checkpoint_path.resolve()
    root_candidate = configured_output_root.absolute()
    output_root = root_candidate.resolve()
    candidate = output_path.absolute()
    output = candidate.resolve()
    checkpoint_roots: set[Path] = set()
    configured_root = Path(str(checkpoint_config.artifacts.checkpoints_dir))
    if configured_root.is_absolute():
        checkpoint_roots.add(configured_root.resolve())
    else:
        configured_parts = tuple(part for part in configured_root.parts if part not in {"", "."})
        if (
            configured_parts
            and checkpoint.parent.parts[-len(configured_parts) :] == configured_parts
        ):
            checkpoint_roots.add(checkpoint.parent)
    protected_names = {"checkpoint", "checkpoints"}
    if checkpoint.parent.name.casefold() in protected_names:
        checkpoint_roots.add(checkpoint.parent)
    if (
        output == checkpoint
        or any(root == output or root in output.parents for root in checkpoint_roots)
        or any(part.casefold() in protected_names for part in output.parts[:-1])
    ):
        raise ValueError("benchmark output path must be outside checkpoint namespaces")
    if output.suffix != ".json":
        raise ValueError("benchmark output path must use a .json suffix")
    try:
        existing = candidate.lstat()
    except FileNotFoundError:
        existing = None
    except OSError as error:
        raise ValueError("benchmark output path cannot be inspected safely") from error
    if existing is not None:
        if candidate.is_symlink():
            raise ValueError("benchmark output path must not be a symlink")
        if not candidate.is_file():
            raise ValueError("benchmark output path must be a regular file")
        if existing.st_nlink != 1:
            raise ValueError("benchmark output path must not be a hardlink")
        if os.path.samefile(checkpoint, candidate):
            raise ValueError("benchmark output and checkpoint must not share an inode")
        raise ValueError("benchmark output path must not overwrite an existing file")
    if root_candidate != output_root:
        raise ValueError("benchmark output root must not traverse a symlink")
    if root_candidate.exists() and not root_candidate.is_dir():
        raise ValueError("benchmark output root must be a directory")
    if output == output_root or output_root not in output.parents:
        raise ValueError("benchmark output path must be inside its configured output root")
    repository_output_root = (ROOT_DIR / "outputs/benchmark-results").resolve()
    if output_root == ROOT_DIR or (
        ROOT_DIR in output_root.parents
        and output_root != repository_output_root
        and repository_output_root not in output_root.parents
    ):
        raise ValueError(
            "repository-local benchmark output roots must be inside outputs/benchmark-results"
        )
    repository_protected = tuple(
        (ROOT_DIR / name).resolve()
        for name in (".git", "assets", "config", "data", "docs", "src", "tests")
    )
    if any(root == output_root or root in output_root.parents for root in repository_protected):
        raise ValueError("benchmark output root must be outside repository input namespaces")
    resolved_cache = cache_path.resolve()
    if (
        resolved_cache == output_root
        or resolved_cache in output_root.parents
        or output_root in resolved_cache.parents
    ):
        raise ValueError("benchmark output root must be separate from benchmark cache storage")
    protected_root_names = {"artifact", "artifacts", "cache", "checkpoint", "checkpoints"}
    if any(part.casefold() in protected_root_names for part in output_root.parts):
        raise ValueError(
            "benchmark output root must be outside cache/checkpoint/artifact namespaces"
        )
    return output
