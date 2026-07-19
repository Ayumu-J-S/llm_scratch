"""Checkpoint-owned BENCH-001 execution and compact evidence output."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import wandb
from hydra.core.hydra_config import HydraConfig
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
from runtime.environment import collect_environment
from runtime.reproducibility import collect_git_identity, sha256_file
from training.checkpoint import load_checkpoint_for_generation


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

    validate_benchmark_config(cfg)
    if access == "final" and final_acknowledgement != FINAL_ACKNOWLEDGEMENT:
        raise PermissionError(
            f"reserved final benchmark requires BENCHMARK_FINAL_ACK={FINAL_ACKNOWLEDGEMENT}"
        )
    if access not in {"dev", "final"}:
        raise ValueError("benchmark access must be dev or final")
    benchmark_cfg = cfg.benchmark
    checkpoint_path = _root_or_cwd_path(str(benchmark_cfg.checkpoint_path))
    output_path = _output_path(benchmark_cfg).resolve()
    _reject_output_checkpoint_collision(checkpoint_path, output_path)

    loaded = load_checkpoint_for_generation(checkpoint_path)
    checkpoint_cfg = OmegaConf.create(loaded.payload["state"]["resolved_config"])
    validate_training_config(checkpoint_cfg)
    validate_benchmark_checkpoint_runtime(cfg, checkpoint_cfg)
    device = select_device(str(benchmark_cfg.device))
    sampler = CheckpointSampler.from_loaded_checkpoint(
        checkpoint_path,
        loaded,
        device=device,
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
    evaluation_identity = _evaluation_identity(
        sampler,
        suite.identity(),
        context_preflight=context_preflight,
        evaluator_identity=_evaluator_identity(),
    )
    evaluation_identity_sha256 = hashlib.sha256(
        canonical_json_bytes(evaluation_identity)
    ).hexdigest()
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


def _evaluation_identity(
    sampler: CheckpointSampler,
    suite_identity: Mapping[str, Any],
    *,
    context_preflight: Mapping[str, Any],
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
    if not wandb_cfg.get("enabled", False):
        return
    identity = result["evaluation_identity"]
    run = wandb.init(
        project=wandb_cfg.get("project"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name"),
        mode=wandb_cfg.get("mode", "online"),
        config={
            "benchmark_evaluation_identity_sha256": result["evaluation_identity_sha256"],
            "benchmark_suite": identity["suite"],
            "checkpoint_identity": identity["checkpoint"],
            "tokenizer_fingerprint": identity["tokenizer_fingerprint"],
        },
    )
    try:
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
        run.summary.update(summary)
        table = wandb.Table(
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
        )
        run.log({"benchmark/results": table})
    finally:
        run.finish()


def _benchmark_cache(cache_cfg: DictConfig) -> BoundedShardCache:
    cache_path = _root_or_cwd_path(str(cache_cfg.dir))
    return BoundedShardCache(
        cache_path,
        max_size_bytes=int(cache_cfg.max_size_bytes),
        min_free_bytes=int(cache_cfg.min_free_bytes),
        wait_timeout_seconds=float(cache_cfg.wait_timeout_seconds),
    )


def _output_path(benchmark_cfg: DictConfig) -> Path:
    path = Path(str(benchmark_cfg.output_path))
    if path.is_absolute():
        return path
    try:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        output_dir = Path.cwd()
    return output_dir / path


def _root_or_cwd_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    root_candidate = (ROOT_DIR / path).resolve()
    return root_candidate if root_candidate.exists() else (Path.cwd() / path).resolve()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
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
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _reject_output_checkpoint_collision(checkpoint_path: Path, output_path: Path) -> None:
    checkpoint = checkpoint_path.resolve()
    output = output_path.resolve()
    if checkpoint == output:
        raise ValueError("benchmark output path must not be the checkpoint path")
    try:
        if output.exists() and os.path.samefile(checkpoint, output):
            raise ValueError("benchmark output and checkpoint must not share an inode")
    except FileNotFoundError:
        return
