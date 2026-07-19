#!/usr/bin/env python3
"""Run one canonical DGX-001 training arm and retain local evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Sequence

import hydra
import torch
from dgx.telemetry import TelemetrySampler, system_sample
from generation.sampler import CheckpointSampler
from train import ROOT_DIR, prepare_trainer
from training.checkpoint import load_checkpoint_for_generation
from utils.model import count_parameters


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--candidate-id", required=True)
    command.add_argument("--repetition", required=True, type=int)
    command.add_argument("--git-commit", default=os.environ.get("DGX_GIT_COMMIT"))
    command.add_argument("--image-id", default=os.environ.get("DGX_IMAGE_ID"))
    command.add_argument("--plan-id", required=True)
    command.add_argument("--role", choices=("matrix", "pilot"), required=True)
    command.add_argument("--warmup-optimizer-steps", required=True, type=int)
    command.add_argument("--measured-optimizer-steps", required=True, type=int)
    command.add_argument("--telemetry-interval-seconds", type=float, default=1.0)
    command.add_argument("--min-available-memory-bytes", required=True, type=int)
    command.add_argument("--min-free-disk-bytes", required=True, type=int)
    command.add_argument("--post-plan-free-reserve-bytes", required=True, type=int)
    command.add_argument("--max-in-flight-atomic-write-bytes", required=True, type=int)
    command.add_argument("--max-temperature-c", required=True, type=float)
    command.add_argument("--max-swap-in-pages", type=int, default=0)
    command.add_argument("--max-swap-out-pages", type=int, default=0)
    command.add_argument("--pilot", action="store_true")
    command.add_argument("--sample", action="store_true")
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


def _wandb_evidence(checkpoint_dir: Path) -> dict:
    path = checkpoint_dir / "wandb_events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    init = [row for row in rows if row.get("action") == "init"]
    run = init[-1] if init else {}
    critical_failures = sorted(
        {
            str(row.get("action"))
            for row in rows
            if row.get("outcome") == "failed"
            and row.get("action") in {"log", "summary", "runtime_summary", "final_summary"}
        }
    )
    return {
        "path": str(path.relative_to(checkpoint_dir.parent)),
        "sha256": _sha256(path),
        "rows": len(rows),
        "init_status": init[-1].get("outcome") if init else None,
        "mode": run.get("mode"),
        "run_id": run.get("run_id"),
        "run_url": run.get("run_url"),
        "watch_disabled": any(
            row.get("action") == "watch" and row.get("outcome") == "disabled" for row in rows
        ),
        "finish_succeeded": any(
            row.get("action") == "finish" and row.get("outcome") == "succeeded" for row in rows
        ),
        "successful_scalar_logs": sum(
            row.get("action") == "log" and row.get("outcome") == "succeeded" for row in rows
        ),
        "runtime_summary_succeeded": any(
            row.get("action") == "runtime_summary" and row.get("outcome") == "succeeded"
            for row in rows
        ),
        "final_summary_succeeded": any(
            row.get("action") == "final_summary" and row.get("outcome") == "succeeded"
            for row in rows
        ),
        "artifact_uploads": sum(
            row.get("action") == "artifact" and row.get("outcome") == "uploaded" for row in rows
        ),
        "critical_failures": critical_failures,
    }


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
    if not args.git_commit or len(args.git_commit) != 40:
        raise RuntimeError("one exact 40-character git commit is required")
    if not args.image_id or not args.image_id.startswith("sha256:"):
        raise RuntimeError("one exact container image ID is required")
    if args.repetition < 1:
        raise ValueError("repetition must be positive")
    if args.warmup_optimizer_steps < 0 or args.measured_optimizer_steps < 1:
        raise ValueError("warmup/measured optimizer steps are invalid")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("DGX-001 requires CUDA with BF16 support")
    sample = system_sample(output_dir, (Path("/cache"),))
    if sample["host"]["memory_available_bytes"] < args.min_available_memory_bytes:
        raise RuntimeError("available UMA is below the hard preflight floor")
    if sample["host"]["disk_free_bytes"] < effective_floor_bytes:
        raise RuntimeError("free disk is below the hard preflight floor")
    temperature = sample["gpu"]["temperature_c"]
    if temperature is None or temperature > args.max_temperature_c:
        raise RuntimeError("GPU temperature is unavailable or above the hard preflight ceiling")
    return sample


def _samples(checkpoint: Path) -> list[dict]:
    sampler = CheckpointSampler.from_checkpoint(checkpoint, device="cuda")
    results = []
    for prompt in ("小さな言語モデルは", "A small language model"):
        results.append(sampler.generate(prompt, max_new_tokens=16).metadata())
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_path = output_dir / "run.json"
    started = time.time()
    effective_disk_floor = _effective_min_free_disk_bytes(args)
    record: dict = {
        "schema_version": 3,
        "ticket": "DGX-001",
        "status": "failed",
        "role": args.role,
        "plan_id": args.plan_id,
        "candidate_id": args.candidate_id,
        "repetition": args.repetition,
        "git_commit": args.git_commit,
        "image_id": args.image_id,
        "telemetry_interval_seconds": args.telemetry_interval_seconds,
        "warmup_optimizer_steps": args.warmup_optimizer_steps,
        "measured_optimizer_steps": args.measured_optimizer_steps,
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
        record["preflight"] = _preflight(args, output_dir, effective_disk_floor)
        record["environment"] = _environment()
        cfg = _compose(args.overrides)
        expected_profile = "pretrain_baseline" if args.role == "pilot" else "dgx_candidate"
        if cfg.profile.name != expected_profile:
            raise RuntimeError(f"{args.role} requires profile={expected_profile}")
        if cfg.runtime.device != "cuda" or cfg.training.precision != "bf16":
            raise RuntimeError("DGX measurements require explicit CUDA BF16")
        if not cfg.reproducibility.deterministic:
            raise RuntimeError("DGX candidate measurements require strict determinism")
        expected_steps = args.warmup_optimizer_steps + args.measured_optimizer_steps
        if args.role == "matrix" and cfg.training.max_steps != expected_steps:
            raise RuntimeError(
                f"training.max_steps must equal warmup + measured steps ({expected_steps})"
            )
        if args.role == "pilot" and (
            not args.pilot or cfg.training.max_steps is not None or cfg.training.max_time is None
        ):
            raise RuntimeError("pilot measurements require max_steps=null and an explicit max_time")
        if args.role == "matrix" and args.pilot:
            raise RuntimeError("matrix measurements cannot set --pilot")
        output_path = Path(cfg.measurement.output_path)
        if not output_path.is_absolute() or output_path != output_dir / "measurement.json":
            raise RuntimeError("measurement.output_path must be the exact absolute evidence path")
        trainer = prepare_trainer(cfg, run_dir=output_dir)
        effective_tokens = (
            int(cfg.training.sequence_length)
            * int(cfg.training.batch_size)
            * int(cfg.training.gradient_accumulation_steps)
        )
        record.update(
            {
                "profile": str(cfg.profile.name),
                "parameter_count": count_parameters(trainer.model),
                "num_layers": int(cfg.model.num_layers),
                "embed_size": int(cfg.model.embed_size),
                "num_heads": int(cfg.model.num_heads),
                "sequence_length": int(cfg.training.sequence_length),
                "batch_size": int(cfg.training.batch_size),
                "gradient_accumulation_steps": int(cfg.training.gradient_accumulation_steps),
                "effective_target_tokens_per_step": effective_tokens,
                "resolved_config": "resolved_config.yaml",
                "resolved_config_sha256": _sha256(output_dir / "resolved_config.yaml"),
                "run_manifest": "run_manifest.json",
                "run_manifest_sha256": _sha256(output_dir / "run_manifest.json"),
                "wandb_policy": {
                    "mode": str(cfg.wandb.mode),
                    "watch_enabled": bool(cfg.wandb.watch.enabled),
                    "artifact_policy": str(cfg.wandb.artifact.policy),
                    "log_every_n_steps": int(cfg.training.log_every_n_steps),
                },
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
        metrics = trainer.fit()
        sampler.stop()
        record["telemetry_ended_monotonic_seconds"] = time.monotonic()
        final_checkpoint = output_dir / cfg.artifacts.checkpoints_dir / "final.pt"
        loaded = load_checkpoint_for_generation(final_checkpoint)
        if loaded.physical_identity["size_bytes"] > args.max_in_flight_atomic_write_bytes:
            raise RuntimeError("observed checkpoint exceeded its atomic-write safety budget")
        record["checkpoint_verified"] = loaded.payload["kind"] == "final"
        record["checkpoint"] = str(final_checkpoint.relative_to(output_dir))
        record["checkpoint_physical_identity"] = loaded.physical_identity
        record["checkpoint_identity"] = loaded.payload["identity"]
        record["final_checkpoint_boundary"] = (
            loaded.payload["state"].get("measurement_evidence", {}).get("checkpoint_boundary")
        )
        record["final_optimizer_step"] = trainer.optimizer_step
        record["final_target_tokens"] = trainer.target_tokens
        record["final_elapsed_seconds"] = trainer.elapsed_seconds
        record["metric_records"] = len(metrics)
        if args.sample:
            _atomic_json(output_dir / "samples.json", _samples(final_checkpoint))
            record["samples"] = "samples.json"
        record["wandb_evidence"] = _wandb_evidence(output_dir / cfg.artifacts.checkpoints_dir)
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
        _atomic_json(run_path, record)
    if record["status"] != "succeeded":
        print(record.get("error", "DGX measurement failed"), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
