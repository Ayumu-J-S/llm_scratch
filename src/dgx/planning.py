"""Deterministic DGX-001 matrix construction, evidence gates, and selection."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


SECONDS = {"1_hour": 3600, "24_hours": 86400, "7_days": 604800}


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else value


def validate_dgx_config(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    cfg = _plain(config)
    if not isinstance(cfg, Mapping):
        raise ValueError("DGX config must resolve to a mapping")
    required = {"image", "matrix", "pilot", "gates", "selection"}
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"DGX config is missing: {', '.join(missing)}")
    matrix = cfg["matrix"]
    if not isinstance(matrix, Mapping):
        raise ValueError("matrix must be a mapping")
    repetitions = matrix.get("repetitions")
    warmup = matrix.get("warmup_optimizer_steps")
    measured = matrix.get("measured_optimizer_steps")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (repetitions, measured)
    ):
        raise ValueError(
            "matrix repetitions and measured_optimizer_steps must be positive integers"
        )
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("matrix warmup_optimizer_steps must be a non-negative integer")
    models = matrix.get("model_presets")
    contexts = matrix.get("context_presets")
    if not isinstance(models, list) or not models or not isinstance(contexts, list) or not contexts:
        raise ValueError("matrix model_presets and context_presets must be non-empty lists")
    model_ids = [str(item["id"]) for item in models]
    context_ids = [str(item["id"]) for item in contexts]
    if len(set(model_ids)) != len(model_ids) or len(set(context_ids)) != len(context_ids):
        raise ValueError("DGX preset ids must be unique")
    effective_tokens = {
        int(item["sequence_length"])
        * int(item["batch_size"])
        * int(item["gradient_accumulation_steps"])
        for item in contexts
    }
    if len(effective_tokens) != 1:
        raise ValueError("context presets must keep effective target tokens per update equal")
    if int(cfg["selection"]["target_tokens_7d"]) < 1:
        raise ValueError("selection.target_tokens_7d must be positive")
    return dict(cfg)


def build_matrix_plan(config: Mapping[str, Any] | DictConfig) -> list[dict[str, Any]]:
    cfg = validate_dgx_config(config)
    matrix = cfg["matrix"]
    steps = int(matrix["warmup_optimizer_steps"]) + int(matrix["measured_optimizer_steps"])
    arms: list[dict[str, Any]] = []
    for model in matrix["model_presets"]:
        for context in matrix["context_presets"]:
            candidate_id = f"{model['id']}-{context['id']}"
            tokens_per_step = (
                int(context["sequence_length"])
                * int(context["batch_size"])
                * int(context["gradient_accumulation_steps"])
            )
            arms.append(
                {
                    "candidate_id": candidate_id,
                    "model_id": str(model["id"]),
                    "context_id": str(context["id"]),
                    "num_layers": int(model["num_layers"]),
                    "embed_size": int(model["embed_size"]),
                    "num_heads": int(model["num_heads"]),
                    "sequence_length": int(context["sequence_length"]),
                    "batch_size": int(context["batch_size"]),
                    "gradient_accumulation_steps": int(context["gradient_accumulation_steps"]),
                    "effective_target_tokens_per_step": tokens_per_step,
                    "max_steps": steps,
                    "max_target_tokens": tokens_per_step * steps,
                }
            )
    plan: list[dict[str, Any]] = []
    for repetition in range(1, int(matrix["repetitions"]) + 1):
        # Rotate each repeat so no candidate always inherits the cold or hot slot.
        offset = ((repetition - 1) * max(1, len(arms) // 3)) % len(arms)
        for arm in arms[offset:] + arms[:offset]:
            plan.append({**arm, "repetition": repetition})
    return plan


def training_overrides(entry: Mapping[str, Any], *, output_path: str) -> list[str]:
    return [
        "profile=dgx_candidate",
        f"model.num_layers={entry['num_layers']}",
        f"model.embed_size={entry['embed_size']}",
        f"model.num_heads={entry['num_heads']}",
        f"training.sequence_length={entry['sequence_length']}",
        f"training.batch_size={entry['batch_size']}",
        f"training.gradient_accumulation_steps={entry['gradient_accumulation_steps']}",
        f"training.max_steps={entry['max_steps']}",
        f"data.streaming.train.max_target_tokens={entry['max_target_tokens']}",
        f"measurement.output_path={output_path}",
        "wandb.mode=disabled",
        "wandb.watch.enabled=false",
        "wandb.artifact.policy=none",
    ]


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _finite_training(metrics: list[dict[str, Any]]) -> bool:
    steps = [row for row in metrics if row.get("event") == "step"]
    if not steps:
        return False
    keys = ("train/loss_step", "optimizer/gradient_norm", "optimizer/lr")
    return all(
        all(isinstance(row.get(key), (int, float)) and math.isfinite(row[key]) for key in keys)
        for row in steps
    )


def summarize_run(run_dir: Path, gates: Mapping[str, Any]) -> dict[str, Any]:
    run = _read_json(run_dir / "run.json")
    measurement = _read_json(run_dir / "measurement.json")
    telemetry = _read_jsonl(run_dir / "system.jsonl")
    metrics = _read_jsonl(run_dir / "checkpoints" / "metrics.jsonl")
    measured = [
        row
        for row in measurement.get("rows", [])
        if row.get("event") == "optimizer_step" and row.get("warmup") is False
    ]
    wall_seconds = sum(float(row["step_wall_seconds"]) for row in measured)
    target_tokens = sum(int(row["target_tokens_step"]) for row in measured)
    step_times = [float(row["step_wall_seconds"]) for row in measured]
    data_wait_seconds = sum(float(row["host_seconds"]["data_wait"]) for row in measured)
    phase_cuda_seconds: defaultdict[str, float] = defaultdict(float)
    for row in measured:
        for key, value in row.get("cuda_milliseconds", {}).items():
            phase_cuda_seconds[key] += float(value) / 1000.0
    allocated = [int(row["pytorch_allocated_bytes"]) for row in measured]
    reserved = [int(row["pytorch_reserved_bytes"]) for row in measured]
    host_rows = [row["host"] for row in telemetry]
    gpu_rows = [row["gpu"] for row in telemetry]
    observed_seconds = float(run["telemetry_ended_monotonic_seconds"]) - float(
        run["telemetry_started_monotonic_seconds"]
    )
    interval = float(run.get("telemetry_interval_seconds", 1.0))
    expected_samples = max(1.0, observed_seconds / interval + 1.0)
    checkpoint_rows = [row for row in measurement.get("rows", []) if "checkpoint_seconds" in row]
    checkpoint_size = max(
        (int(row.get("checkpoint/size_bytes", 0)) for row in checkpoint_rows), default=0
    )
    temperatures = [
        float(row["temperature_c"]) for row in gpu_rows if row.get("temperature_c") is not None
    ]
    swap_in_delta = host_rows[-1]["swap_in_pages"] - host_rows[0]["swap_in_pages"]
    swap_out_delta = host_rows[-1]["swap_out_pages"] - host_rows[0]["swap_out_pages"]
    allocator_window = min(5, max(1, len(allocated) // 2))
    # Batch/source shapes can make the live allocation oscillate. A sustained
    # leak raises the lower envelope; comparing peak-to-trough would reject a
    # stable periodic pattern as if it were monotonic growth.
    allocator_growth = max(
        0,
        min(allocated[-allocator_window:]) - min(allocated[:allocator_window]),
    )
    run_gates = {
        "run_succeeded": run.get("status") == "succeeded",
        "measurement_complete": measurement.get("complete") is True,
        "cuda_events": measurement.get("device") == "cuda"
        and measurement.get("cuda_events") is True,
        "measured_steps": len(measured) >= int(run["measured_optimizer_steps"]),
        "finite_training": _finite_training(metrics),
        "telemetry_error_free": not run.get("telemetry_errors"),
        "sampler_coverage": len(telemetry) / expected_samples
        >= float(gates["min_sampler_coverage"]),
        "available_memory": min(row["memory_available_bytes"] for row in host_rows)
        >= int(gates["min_available_memory_bytes"]),
        "free_disk": min(row["disk_free_bytes"] for row in host_rows)
        >= int(gates["min_free_disk_bytes"]),
        "swap_in": swap_in_delta <= int(gates["max_swap_in_pages"]),
        "swap_out": swap_out_delta <= int(gates["max_swap_out_pages"]),
        "thermal": bool(temperatures) and max(temperatures) <= float(gates["max_temperature_c"]),
        "allocator_stable": allocator_growth <= int(gates["max_allocator_growth_bytes"]),
        "data_wait": data_wait_seconds / wall_seconds <= float(gates["max_data_wait_fraction"]),
        "checkpoint_verified": bool(run.get("checkpoint_verified")) and checkpoint_size > 0,
    }
    if not measured or not telemetry or wall_seconds <= 0:
        raise ValueError(f"incomplete measurement evidence in {run_dir}")
    return {
        "candidate_id": run["candidate_id"],
        "repetition": int(run["repetition"]),
        "git_commit": run["git_commit"],
        "image_id": run["image_id"],
        "parameter_count": int(run["parameter_count"]),
        "num_layers": int(run["num_layers"]),
        "embed_size": int(run["embed_size"]),
        "num_heads": int(run["num_heads"]),
        "sequence_length": int(run["sequence_length"]),
        "batch_size": int(run["batch_size"]),
        "gradient_accumulation_steps": int(run["gradient_accumulation_steps"]),
        "effective_target_tokens_per_step": int(run["effective_target_tokens_per_step"]),
        "measured_optimizer_steps": len(measured),
        "target_tokens": target_tokens,
        "tokens_per_second": target_tokens / wall_seconds,
        "step_time_seconds": {
            "median": statistics.median(step_times),
            "p95": _percentile(step_times, 0.95),
            "max": max(step_times),
        },
        "data_wait_seconds": data_wait_seconds,
        "data_wait_fraction": data_wait_seconds / wall_seconds,
        "cuda_phase_seconds": dict(sorted(phase_cuda_seconds.items())),
        "pytorch_peak_allocated_bytes": max(allocated),
        "pytorch_peak_reserved_bytes": max(reserved),
        "pytorch_allocator_baseline_growth_bytes": allocator_growth,
        "host_min_available_memory_bytes": min(row["memory_available_bytes"] for row in host_rows),
        "host_peak_process_rss_bytes": max(row["process_rss_bytes"] for row in host_rows),
        "host_min_free_disk_bytes": min(row["disk_free_bytes"] for row in host_rows),
        "max_temperature_c": max(temperatures),
        "swap_in_pages": swap_in_delta,
        "swap_out_pages": swap_out_delta,
        "checkpoint_size_bytes": checkpoint_size,
        "checkpoint_seconds": sum(float(row["checkpoint_seconds"]) for row in checkpoint_rows),
        "validation_seconds": sum(
            float(row.get("full_event_pause_seconds", 0.0))
            for row in measurement.get("rows", [])
            if row.get("event") == "validation"
        ),
        "sampler_coverage": min(1.0, len(telemetry) / expected_samples),
        "gates": run_gates,
        "passed": all(run_gates.values()),
        "evidence_dir": str(run_dir),
    }


def _candidate_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    first = runs[0]
    throughputs = [float(run["tokens_per_second"]) for run in runs]
    data_wait = [float(run["data_wait_fraction"]) for run in runs]
    allocated = [int(run["pytorch_peak_allocated_bytes"]) for run in runs]
    checkpoint_sizes = [int(run["checkpoint_size_bytes"]) for run in runs]
    conservative = min(throughputs)
    cuda_phases: defaultdict[str, float] = defaultdict(float)
    for run in runs:
        for key, value in run["cuda_phase_seconds"].items():
            cuda_phases[key] += float(value)
    bottleneck = max(cuda_phases, key=cuda_phases.get) if cuda_phases else "unattributed"
    if statistics.median(data_wait) > 0.05:
        bottleneck = "data_wait"
    return {
        key: first[key]
        for key in (
            "candidate_id",
            "parameter_count",
            "num_layers",
            "embed_size",
            "num_heads",
            "sequence_length",
            "batch_size",
            "gradient_accumulation_steps",
            "effective_target_tokens_per_step",
        )
    } | {
        "repetitions": len(runs),
        "all_runs_passed": all(run["passed"] for run in runs),
        "throughput": {
            "min": min(throughputs),
            "median": statistics.median(throughputs),
            "max": max(throughputs),
            "spread_fraction": (max(throughputs) - min(throughputs))
            / statistics.median(throughputs),
        },
        "conservative_tokens_per_second": conservative,
        "token_budgets": {
            label: math.floor(conservative * seconds) for label, seconds in SECONDS.items()
        },
        "data_wait_fraction_median": statistics.median(data_wait),
        "pytorch_peak_allocated_bytes_max": max(allocated),
        "host_min_available_memory_bytes": min(
            int(run["host_min_available_memory_bytes"]) for run in runs
        ),
        "host_min_free_disk_bytes": min(int(run["host_min_free_disk_bytes"]) for run in runs),
        "checkpoint_size_bytes_max": max(checkpoint_sizes),
        "checkpoint_seconds_median": statistics.median(
            float(run["checkpoint_seconds"]) for run in runs
        ),
        "validation_seconds_median": statistics.median(
            float(run["validation_seconds"]) for run in runs
        ),
        "named_bottleneck": bottleneck,
        "runs": runs,
    }


def select_candidate(
    candidates: list[dict[str, Any]], selection: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = int(selection["target_tokens_7d"])
    eligible = [
        candidate
        for candidate in candidates
        if candidate["all_runs_passed"] and candidate["token_budgets"]["7_days"] >= target
    ]
    if not eligible:
        raise ValueError("no candidate passes every hard gate and the conservative 7-day target")
    fastest = max(candidate["conservative_tokens_per_second"] for candidate in eligible)
    speed_floor = fastest * (1.0 - float(selection["max_slowdown_fraction"]))
    eligible = [
        candidate
        for candidate in eligible
        if candidate["conservative_tokens_per_second"] >= speed_floor
    ]
    largest_layers = max(candidate["num_layers"] for candidate in eligible)
    eligible = [candidate for candidate in eligible if candidate["num_layers"] == largest_layers]
    model_fastest = max(candidate["conservative_tokens_per_second"] for candidate in eligible)
    context_floor = model_fastest * float(selection["context_throughput_floor_fraction"])
    eligible = [
        candidate
        for candidate in eligible
        if candidate["conservative_tokens_per_second"] >= context_floor
    ]
    largest_context = max(candidate["sequence_length"] for candidate in eligible)
    eligible = [
        candidate for candidate in eligible if candidate["sequence_length"] == largest_context
    ]
    near_fastest = max(candidate["conservative_tokens_per_second"] for candidate in eligible) * (
        1.0 - float(selection["near_fastest_fraction"])
    )
    eligible = [
        candidate
        for candidate in eligible
        if candidate["conservative_tokens_per_second"] >= near_fastest
    ]
    selected = min(
        eligible,
        key=lambda item: (
            item["pytorch_peak_allocated_bytes_max"],
            -item["conservative_tokens_per_second"],
            item["candidate_id"],
        ),
    )
    return selected, eligible


def summarize_evidence(
    evidence_root: Path, config: Mapping[str, Any] | DictConfig
) -> dict[str, Any]:
    cfg = validate_dgx_config(config)
    run_dirs = sorted(path.parent for path in Path(evidence_root).glob("*/run.json"))
    if not run_dirs:
        raise ValueError(f"no DGX run evidence found under {evidence_root}")
    runs = [summarize_run(path, cfg["gates"]) for path in run_dirs]
    commits = {run["git_commit"] for run in runs}
    images = {run["image_id"] for run in runs}
    if len(commits) != 1 or len(images) != 1:
        raise ValueError("all candidate runs must use one exact commit and image")
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["candidate_id"]].append(run)
    expected_repetitions = int(cfg["matrix"]["repetitions"])
    candidates = []
    for candidate_id, candidate_runs in sorted(grouped.items()):
        repetitions = {run["repetition"] for run in candidate_runs}
        if repetitions != set(range(1, expected_repetitions + 1)):
            raise ValueError(f"candidate {candidate_id} lacks the declared repetitions")
        candidates.append(_candidate_summary(candidate_runs))
    expected_candidates = {entry["candidate_id"] for entry in build_matrix_plan(cfg)}
    if set(grouped) != expected_candidates:
        missing = sorted(expected_candidates - set(grouped))
        extra = sorted(set(grouped) - expected_candidates)
        raise ValueError(f"candidate matrix mismatch; missing={missing}, extra={extra}")
    selected, finalists = select_candidate(candidates, cfg["selection"])
    checkpoint_size = int(selected["checkpoint_size_bytes_max"])
    full_run_seconds = SECONDS["7_days"]
    checkpoint_cadence = 2_500_000
    planned_recovery_writes = math.ceil(
        selected["conservative_tokens_per_second"] * full_run_seconds / checkpoint_cadence
    )
    storage_peak = checkpoint_size * 6  # 3 rotating + atomic temp + best + final/milestone.
    storage_passed = selected["host_min_free_disk_bytes"] >= (
        storage_peak * float(cfg["gates"]["storage_headroom_ratio"])
    )
    return {
        "schema_version": 1,
        "ticket": "DGX-001",
        "git_commit": next(iter(commits)),
        "image_id": next(iter(images)),
        "measurement_conditions": {
            "repetitions": expected_repetitions,
            "warmup_optimizer_steps": int(cfg["matrix"]["warmup_optimizer_steps"]),
            "measured_optimizer_steps": int(cfg["matrix"]["measured_optimizer_steps"]),
            "precision": "bf16",
            "deterministic": True,
            "cuda_events": True,
            "effective_target_tokens_equal_across_contexts": True,
        },
        "candidates": candidates,
        "selection_rule": dict(cfg["selection"]),
        "selection_finalists": [item["candidate_id"] for item in finalists],
        "selected": selected,
        "plan": {
            "token_budgets": selected["token_budgets"],
            "one_hour_max_time_seconds": 3600,
            "validation_every_target_tokens": 5_000_000,
            "recovery_checkpoint_every_target_tokens": checkpoint_cadence,
            "milestone_every_target_tokens": 100_000_000,
            "planned_7d_recovery_writes": planned_recovery_writes,
            "checkpoint_size_bytes": checkpoint_size,
            "peak_checkpoint_storage_bytes": storage_peak,
            "storage_headroom_passed": storage_passed,
            "wandb_mode": "online",
            "wandb_watch": False,
            "wandb_artifact_policy": "none",
        },
        "named_bottleneck": selected["named_bottleneck"],
        "verdict": "PASS" if storage_passed else "FAIL",
    }
