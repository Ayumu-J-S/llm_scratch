#!/usr/bin/env python3
"""Measure selected-profile model and loader capacity without changing training."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Sequence

import hydra
import torch

from dgx.telemetry import TelemetrySampler, system_sample
from runtime.config import validate_training_config
from runtime.device import select_device
from runtime.reproducibility import seed_everything, write_run_manifest
from train import (
    ROOT_DIR,
    build_streaming_dataloader,
    build_tokenizer_config,
    prepare_trainer,
    save_resolved_config,
    to_plain_config,
    validate_streaming_dataloaders,
)
from training.trainer import Trainer


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--role", choices=("model-only", "loader-only"), required=True)
    command.add_argument("--candidate-id", required=True)
    command.add_argument("--repetition", type=int, required=True)
    command.add_argument("--git-commit", required=True)
    command.add_argument("--image-id", required=True)
    command.add_argument("--plan-id", required=True)
    command.add_argument("--warmup-optimizer-steps", type=int, required=True)
    command.add_argument("--measured-optimizer-steps", type=int, required=True)
    command.add_argument("--telemetry-interval-seconds", type=float, default=1.0)
    command.add_argument("--min-available-memory-bytes", type=int, required=True)
    command.add_argument("--min-free-disk-bytes", type=int, required=True)
    command.add_argument("--post-plan-free-reserve-bytes", type=int, required=True)
    command.add_argument("--max-in-flight-atomic-write-bytes", type=int, required=True)
    command.add_argument("--max-temperature-c", type=float, required=True)
    command.add_argument("--max-swap-in-pages", type=int, required=True)
    command.add_argument("--max-swap-out-pages", type=int, required=True)
    command.add_argument("overrides", nargs=argparse.REMAINDER)
    return command


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compose(overrides: Sequence[str]):
    normalized = list(overrides)
    if normalized[:1] == ["--"]:
        normalized = normalized[1:]
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT_DIR / "config")):
        return hydra.compose(config_name="train", overrides=normalized)


def _environment() -> dict:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "scripts" / "diagnose_environment.py"),
            "--json",
            "--require-cuda",
            "--require-bf16",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return json.loads(result.stdout)


def _effective_min_free_disk_bytes(args: argparse.Namespace) -> int:
    if args.min_free_disk_bytes != 120_000_000_000:
        raise RuntimeError("DGX operational free-disk floor must be exactly 120 GB")
    if args.post_plan_free_reserve_bytes != 100_000_000_000:
        raise RuntimeError("DGX post-plan free-disk reserve must be exactly 100 GB")
    if args.max_in_flight_atomic_write_bytes <= 0:
        raise RuntimeError("DGX maximum in-flight atomic-write budget must be positive")
    return max(
        args.min_free_disk_bytes,
        args.post_plan_free_reserve_bytes + args.max_in_flight_atomic_write_bytes,
    )


def _preflight(args: argparse.Namespace, output_dir: Path, effective_floor_bytes: int) -> dict:
    sample = system_sample(output_dir, (Path("/cache"),))
    if sample["host"]["memory_available_bytes"] < args.min_available_memory_bytes:
        raise RuntimeError("available UMA is below the hard preflight floor")
    if sample["host"]["disk_free_bytes"] < effective_floor_bytes:
        raise RuntimeError("free disk is below the hard preflight floor")
    temperature = sample["gpu"]["temperature_c"]
    if temperature is None or temperature > args.max_temperature_c:
        raise RuntimeError("GPU temperature is unavailable or above the hard preflight ceiling")
    return sample


def _identity(cfg, output_dir: Path) -> dict:
    resolved_path = save_resolved_config(cfg, run_dir=output_dir)
    tokenizer = build_tokenizer_config(cfg)
    manifest_path = write_run_manifest(
        cfg=to_plain_config(cfg),
        run_dir=output_dir,
        root_dir=ROOT_DIR,
        resolved_config_path=resolved_path,
        tokenizer_manifest_path=ROOT_DIR / tokenizer["manifest_path"],
        tokenizer_expected_fingerprint=tokenizer.get("expected_fingerprint"),
    )
    return {
        "resolved_config": resolved_path.name,
        "resolved_config_sha256": _sha256(resolved_path),
        "run_manifest": manifest_path.name,
        "run_manifest_sha256": _sha256(manifest_path),
    }


def _model_only(cfg, output_dir: Path, warmup: int, measured: int) -> list[dict]:
    trainer = prepare_trainer(cfg, run_dir=output_dir)
    iterator = iter(trainer.train_loader)
    try:
        source_batch = next(iterator)
    finally:
        trainer._close_train_iterator(iterator)
    batch = {key: value.to(trainer.device) for key, value in source_batch.items()}
    rows = []
    total_steps = warmup + measured
    for step in range(1, total_steps + 1):
        torch.cuda.reset_peak_memory_stats(trainer.device)
        torch.cuda.synchronize(trainer.device)
        started = time.perf_counter()
        loss_sum, target_tokens, micro_batches, gradient_norm, clipped, lr, _ = (
            trainer._train_update(
                batch,
                iterator=itertools.repeat(batch),
                first_batch_index=(step - 1) * int(cfg.training.gradient_accumulation_steps) + 1,
            )
        )
        torch.cuda.synchronize(trainer.device)
        wall_seconds = time.perf_counter() - started
        rows.append(
            {
                "optimizer_step": step,
                "warmup": step <= warmup,
                "wall_seconds": wall_seconds,
                "target_tokens": target_tokens,
                "loss": loss_sum / target_tokens,
                "gradient_norm": gradient_norm,
                "micro_batches": micro_batches,
                "clipped": clipped,
                "learning_rate": lr,
                "pytorch_peak_allocated_bytes": torch.cuda.max_memory_allocated(trainer.device),
                "pytorch_peak_reserved_bytes": torch.cuda.max_memory_reserved(trainer.device),
            }
        )
    return rows


def _loader_only(cfg, output_dir: Path, warmup: int, measured: int) -> list[dict]:
    validate_training_config(cfg)
    seed_everything(
        int(cfg.reproducibility.seed),
        deterministic=bool(cfg.reproducibility.deterministic),
    )
    device = select_device(cfg.runtime.device)
    train_loader = build_streaming_dataloader(cfg, "train", device=device)
    validation_loader = build_streaming_dataloader(cfg, "validation", device=device)
    validate_streaming_dataloaders(train_loader, validation_loader)
    rows = []
    iterator = iter(train_loader)
    try:
        for step in range(1, warmup + measured + 1):
            started = time.perf_counter()
            target_tokens = 0
            batch_latencies = []
            for _ in range(int(cfg.training.gradient_accumulation_steps)):
                batch_started = time.perf_counter()
                batch = next(iterator)
                batch_latencies.append(time.perf_counter() - batch_started)
                target_tokens += int((batch["labels"] != -100).sum().item())
            rows.append(
                {
                    "optimizer_step": step,
                    "warmup": step <= warmup,
                    "wall_seconds": time.perf_counter() - started,
                    "target_tokens": target_tokens,
                    "batch_latencies_seconds": batch_latencies,
                    "micro_batches": int(cfg.training.gradient_accumulation_steps),
                }
            )
    finally:
        Trainer._close_train_iterator(iterator)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    effective_disk_floor = _effective_min_free_disk_bytes(args)
    record: dict = {
        "schema_version": 3,
        "ticket": "DGX-001",
        "status": "failed",
        "role": args.role,
        "candidate_id": args.candidate_id,
        "repetition": args.repetition,
        "git_commit": args.git_commit,
        "image_id": args.image_id,
        "plan_id": args.plan_id,
        "warmup_optimizer_steps": args.warmup_optimizer_steps,
        "measured_optimizer_steps": args.measured_optimizer_steps,
        "telemetry_interval_seconds": args.telemetry_interval_seconds,
        "started_unix_seconds": started,
        "storage_safety": {
            "configured_min_free_disk_bytes": args.min_free_disk_bytes,
            "post_plan_free_reserve_bytes": args.post_plan_free_reserve_bytes,
            "max_in_flight_atomic_write_bytes": args.max_in_flight_atomic_write_bytes,
            "effective_min_free_disk_bytes": effective_disk_floor,
        },
    }
    sampler: TelemetrySampler | None = None
    try:
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("DGX decomposition requires CUDA BF16")
        cfg = _compose(args.overrides)
        if cfg.profile.name != "pretrain_baseline":
            raise RuntimeError("DGX decomposition requires profile=pretrain_baseline")
        if cfg.wandb.mode != "disabled" or cfg.measurement.enabled:
            raise RuntimeError("DGX decomposition must disable W&B and trainer measurement")
        record["environment"] = _environment()
        record["preflight"] = _preflight(args, output_dir, effective_disk_floor)
        record.update(
            {
                "num_layers": int(cfg.model.num_layers),
                "embed_size": int(cfg.model.embed_size),
                "num_heads": int(cfg.model.num_heads),
                "sequence_length": int(cfg.training.sequence_length),
                "batch_size": int(cfg.training.batch_size),
                "gradient_accumulation_steps": int(cfg.training.gradient_accumulation_steps),
                "effective_target_tokens_per_step": (
                    int(cfg.training.sequence_length)
                    * int(cfg.training.batch_size)
                    * int(cfg.training.gradient_accumulation_steps)
                ),
            }
        )
        sampler = TelemetrySampler(
            output_dir / "system.jsonl",
            interval_seconds=args.telemetry_interval_seconds,
            hard_limits={
                "min_available_memory_bytes": args.min_available_memory_bytes,
                "min_free_disk_bytes": effective_disk_floor,
                "max_temperature_c": args.max_temperature_c,
                "max_swap_in_pages": args.max_swap_in_pages,
                "max_swap_out_pages": args.max_swap_out_pages,
            },
            interrupt_on_violation=True,
            additional_disk_paths=(Path("/cache"),),
        )
        record["telemetry_started_monotonic_seconds"] = time.monotonic()
        sampler.start()
        if args.role == "model-only":
            rows = _model_only(
                cfg, output_dir, args.warmup_optimizer_steps, args.measured_optimizer_steps
            )
        else:
            rows = _loader_only(
                cfg, output_dir, args.warmup_optimizer_steps, args.measured_optimizer_steps
            )
        sampler.stop()
        record["telemetry_ended_monotonic_seconds"] = time.monotonic()
        record.update(_identity(cfg, output_dir))
        record["rows"] = rows
        record["status"] = "succeeded"
    except BaseException as error:
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()
    finally:
        if sampler is not None and sampler._thread is not None and sampler._thread.is_alive():
            try:
                sampler.stop()
            except BaseException as error:
                record.setdefault("telemetry_stop_error", f"{type(error).__name__}: {error}")
        if sampler is not None:
            record.setdefault("telemetry_ended_monotonic_seconds", time.monotonic())
            record["telemetry_samples"] = sampler.samples
            record["telemetry_errors"] = sampler.errors
            record["telemetry_violations"] = sampler.violations
        record["ended_unix_seconds"] = time.time()
        record["wall_seconds"] = record["ended_unix_seconds"] - started
        _atomic_json(output_dir / "decomposition.json", record)
    return 0 if record["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
