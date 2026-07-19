"""Deterministic DGX-001 planning, evidence validation, and selection."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
from runtime.reproducibility import canonical_config_sha256


SECONDS = {"1_hour": 3600, "24_hours": 86400, "7_days": 604800}
MEASUREMENT_TOP_LEVEL_KEYS = {
    "schema_version",
    "checkpoint_identity",
    "measurement_evidence_id",
    "complete",
    "segments",
    "checkpoint_boundaries",
}
MEASUREMENT_SEGMENT_KEYS = {
    "segment_index",
    "start_counters",
    "end_counters",
    "resumed_from",
    "parent_boundary_id",
    "measurement",
    "complete",
    "rows",
}
MEASUREMENT_BOUNDARY_KEYS = {
    "boundary_index",
    "boundary_id",
    "evidence_id",
    "segment_index",
    "kind",
    "counters",
    "status",
    "checkpoint_path",
}
MEASUREMENT_BINDING_KEYS = MEASUREMENT_BOUNDARY_KEYS - {"status", "checkpoint_path"}
COUNTER_KEYS = {"optimizer_step", "target_tokens", "elapsed_seconds"}
CHECKPOINT_KINDS = {"recovery", "best", "final", "milestone"}
REQUIRED_CUDA_PHASES = {"forward", "backward", "optimizer"}
PROTOCOL_CONFIG_KEYS = {
    "schema_version",
    "image",
    "matrix",
    "decomposition",
    "pilot",
    "gates",
    "selection",
}


def _plain(value: Any) -> Any:
    return OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else value


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _filesystem_device(path: Path) -> int:
    return path.stat().st_dev


def _storage_headroom(
    *,
    cache_root: Path,
    output_root: Path,
    cache_existing_bytes: int,
    cache_planned_bytes: int,
    output_existing_bytes: int,
    output_planned_bytes: int,
    headroom_ratio: float,
    post_plan_free_reserve_bytes: int,
) -> dict[str, Any]:
    allocations = (
        {
            "role": "cache",
            "path": cache_root,
            "existing_bytes": cache_existing_bytes,
            "planned_bytes": cache_planned_bytes,
        },
        {
            "role": "output",
            "path": output_root,
            "existing_bytes": output_existing_bytes,
            "planned_bytes": output_planned_bytes,
        },
    )
    filesystems: dict[int, dict[str, Any]] = {}
    for allocation in allocations:
        path = allocation["path"]
        if not path.is_dir():
            raise ValueError(f"DGX storage root does not exist: {path}")
        device = _filesystem_device(path)
        group = filesystems.setdefault(
            device,
            {
                "device": device,
                "representative_path": str(path),
                "roles": [],
                "existing_bytes": 0,
                "planned_bytes": 0,
            },
        )
        group["roles"].append(allocation["role"])
        group["existing_bytes"] += int(allocation["existing_bytes"])
        group["planned_bytes"] += int(allocation["planned_bytes"])
    reports = []
    for group in filesystems.values():
        free_bytes = int(shutil.disk_usage(group["representative_path"]).free)
        capacity_bytes = free_bytes + int(group["existing_bytes"])
        required_capacity_bytes = math.ceil(float(headroom_ratio) * group["planned_bytes"])
        projected_free_bytes = capacity_bytes - int(group["planned_bytes"])
        reports.append(
            {
                **group,
                "roles": sorted(group["roles"]),
                "free_bytes": free_bytes,
                "capacity_bytes": capacity_bytes,
                "required_capacity_bytes": required_capacity_bytes,
                "projected_free_bytes": projected_free_bytes,
                "post_plan_free_reserve_bytes": int(post_plan_free_reserve_bytes),
                "headroom_passed": capacity_bytes >= required_capacity_bytes,
                "post_plan_reserve_passed": projected_free_bytes
                >= int(post_plan_free_reserve_bytes),
            }
        )
    reports.sort(key=lambda item: item["roles"])
    return {
        "same_filesystem": len(reports) == 1,
        "headroom_ratio": float(headroom_ratio),
        "filesystems": reports,
        "headroom_passed": all(item["headroom_passed"] for item in reports),
        "post_plan_reserve_passed": all(item["post_plan_reserve_passed"] for item in reports),
        "passed": all(
            item["headroom_passed"] and item["post_plan_reserve_passed"] for item in reports
        ),
    }


def validate_dgx_config(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    cfg = _plain(config)
    if not isinstance(cfg, Mapping):
        raise ValueError("DGX config must resolve to a mapping")
    if cfg.get("schema_version") != 3:
        raise ValueError("DGX-001 accepts only orchestration schema_version 3")
    required = {"image", "matrix", "decomposition", "pilot", "gates", "selection"}
    missing = sorted(required - set(cfg))
    if missing:
        raise ValueError(f"DGX config is missing: {', '.join(missing)}")
    matrix = cfg["matrix"]
    if not isinstance(matrix, Mapping):
        raise ValueError("matrix must be a mapping")
    if (
        matrix.get("repetitions") != 3
        or matrix.get("warmup_optimizer_steps") != 10
        or matrix.get("measured_optimizer_steps") != 20
    ):
        raise ValueError("final DGX matrix is exactly 3 repetitions of 10 warmup + 20 measured")
    models = matrix.get("model_presets")
    contexts = matrix.get("context_presets")
    if not isinstance(models, list) or len(models) != 3:
        raise ValueError("final DGX matrix requires exactly three model presets")
    if not isinstance(contexts, list) or len(contexts) != 3:
        raise ValueError("final DGX matrix requires exactly three context presets")
    model_ids = [str(item["id"]) for item in models]
    context_ids = [str(item["id"]) for item in contexts]
    if len(set(model_ids)) != 3 or len(set(context_ids)) != 3:
        raise ValueError("DGX preset IDs must be unique")
    effective_tokens = {
        int(item["sequence_length"])
        * int(item["batch_size"])
        * int(item["gradient_accumulation_steps"])
        for item in contexts
    }
    if effective_tokens != {32768}:
        raise ValueError("every final DGX arm must train exactly 32,768 targets per update")
    decomposition = cfg["decomposition"]
    if (
        not isinstance(decomposition, Mapping)
        or decomposition.get("repetitions") != 3
        or decomposition.get("warmup_optimizer_steps") != 10
        or decomposition.get("measured_optimizer_steps") != 20
    ):
        raise ValueError("decomposition is exactly 3 repetitions of 10 warmup + 20 measured")
    if int(cfg["pilot"]["duration_seconds"]) != 1800:
        raise ValueError("selected-profile pilot must be capped at 1,800 seconds")
    finalization_reserve = int(cfg["pilot"]["finalization_reserve_seconds"])
    if finalization_reserve != 120:
        raise ValueError("pilot finalization reserve is exactly 120 seconds")
    if int(cfg["selection"]["target_tokens_7d"]) != 1_000_000_000:
        raise ValueError("DGX selection requires the one-billion-target seven-day floor")
    spread = float(cfg["selection"]["max_repeat_spread_fraction"])
    if not 0.0 <= spread <= 0.25:
        raise ValueError("repeatability spread threshold must be between 0 and 25%")
    gpu_coverage = float(cfg["gates"]["min_gpu_metric_coverage"])
    if not 0.0 < gpu_coverage <= 1.0:
        raise ValueError("required GPU metric coverage must be in (0, 1]")
    if int(cfg["gates"]["min_free_disk_bytes"]) != 120_000_000_000:
        raise ValueError("DGX operational free-disk watchdog is exactly 120 GB")
    if int(cfg["gates"]["post_plan_free_reserve_bytes"]) != 100_000_000_000:
        raise ValueError("DGX post-plan storage reserve is exactly 100 GB")
    return dict(cfg)


def build_matrix_plan(config: Mapping[str, Any] | DictConfig) -> list[dict[str, Any]]:
    cfg = validate_dgx_config(config)
    matrix = cfg["matrix"]
    steps = int(matrix["warmup_optimizer_steps"]) + int(matrix["measured_optimizer_steps"])
    arms: list[dict[str, Any]] = []
    for model in matrix["model_presets"]:
        for context in matrix["context_presets"]:
            target_tokens = (
                int(context["sequence_length"])
                * int(context["batch_size"])
                * int(context["gradient_accumulation_steps"])
            )
            arms.append(
                {
                    "candidate_id": f"{model['id']}-{context['id']}",
                    "model_id": str(model["id"]),
                    "context_id": str(context["id"]),
                    "num_layers": int(model["num_layers"]),
                    "embed_size": int(model["embed_size"]),
                    "num_heads": int(model["num_heads"]),
                    "sequence_length": int(context["sequence_length"]),
                    "batch_size": int(context["batch_size"]),
                    "gradient_accumulation_steps": int(context["gradient_accumulation_steps"]),
                    "effective_target_tokens_per_step": target_tokens,
                    "max_steps": steps,
                    "max_target_tokens": target_tokens * steps,
                }
            )
    plan: list[dict[str, Any]] = []
    for repetition in range(1, 4):
        offset = ((repetition - 1) * 3) % len(arms)
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


def _validated_counters(value: Any, label: str) -> dict[str, int | float]:
    if not isinstance(value, Mapping) or set(value) != COUNTER_KEYS:
        raise ValueError(f"{label} counters are invalid")
    step = value["optimizer_step"]
    tokens = value["target_tokens"]
    elapsed = value["elapsed_seconds"]
    if (
        isinstance(step, bool)
        or not isinstance(step, int)
        or step < 0
        or isinstance(tokens, bool)
        or not isinstance(tokens, int)
        or tokens < 0
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or elapsed < 0
    ):
        raise ValueError(f"{label} counters are invalid")
    return {"optimizer_step": step, "target_tokens": tokens, "elapsed_seconds": float(elapsed)}


def _authority_key(role: str, candidate_id: str, repetition: int) -> str:
    return f"{role}:{candidate_id}:r{repetition}"


def _validate_measurement_v3(
    run_dir: Path, run: Mapping[str, Any], measurement: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if set(measurement) != MEASUREMENT_TOP_LEVEL_KEYS or measurement.get("schema_version") != 3:
        raise ValueError("DGX accepts only the exact trainer measurement schema v3")
    evidence_id = measurement.get("measurement_evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError("measurement evidence chain ID is invalid")
    if measurement.get("checkpoint_identity") != run.get("checkpoint_identity"):
        raise ValueError("measurement and final checkpoint identities differ")
    if measurement.get("complete") is not True:
        raise ValueError("measurement evidence is not complete")
    segments = measurement.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("measurement evidence has no segments")
    if len(segments) != 1:
        raise ValueError("DGX matrix/pilot evidence requires exactly one complete fresh segment")
    boundaries = measurement.get("checkpoint_boundaries")
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError("measurement evidence has no checkpoint boundaries")
    validated_boundaries: list[dict[str, Any]] = []
    boundary_ids: set[str] = set()
    for index, boundary in enumerate(boundaries):
        if not isinstance(boundary, Mapping) or set(boundary) != MEASUREMENT_BOUNDARY_KEYS:
            raise ValueError("measurement checkpoint boundary has an invalid shape")
        if boundary["boundary_index"] != index or boundary["evidence_id"] != evidence_id:
            raise ValueError("measurement checkpoint boundary ordering/identity is invalid")
        if boundary["boundary_id"] in boundary_ids:
            raise ValueError("measurement checkpoint boundary IDs are not unique")
        boundary_ids.add(boundary["boundary_id"])
        if boundary["kind"] not in CHECKPOINT_KINDS or boundary["status"] != "committed":
            raise ValueError("complete DGX evidence requires every checkpoint boundary committed")
        if not isinstance(boundary["checkpoint_path"], str) or not boundary["checkpoint_path"]:
            raise ValueError("committed measurement checkpoint boundary has no path")
        _validated_counters(boundary["counters"], f"checkpoint boundary {index}")
        validated_boundaries.append(dict(boundary))
    boundaries_by_id = {item["boundary_id"]: item for item in validated_boundaries}
    rows: list[dict[str, Any]] = []
    prior_end: dict[str, int | float] | None = None
    for index, segment in enumerate(segments):
        if (
            not isinstance(segment, Mapping)
            or set(segment) != MEASUREMENT_SEGMENT_KEYS
            or segment["segment_index"] != index
        ):
            raise ValueError("measurement segment shape/order is invalid")
        start = _validated_counters(segment["start_counters"], f"segment {index} start")
        end = _validated_counters(segment["end_counters"], f"segment {index} end")
        if any(end[key] < start[key] for key in COUNTER_KEYS):
            raise ValueError("measurement segment counters move backwards")
        parent_id = segment["parent_boundary_id"]
        resumed = segment["resumed_from"]
        if index == 0:
            if (
                parent_id is not None
                or resumed is not None
                or start
                != {
                    "optimizer_step": 0,
                    "target_tokens": 0,
                    "elapsed_seconds": 0.0,
                }
            ):
                raise ValueError("DGX measurement must begin with one fresh zero-counter segment")
        else:
            if not isinstance(parent_id, str) or parent_id not in boundaries_by_id:
                raise ValueError("resumed measurement segment has no committed parent boundary")
            parent = boundaries_by_id[parent_id]
            if (
                parent["segment_index"] >= index
                or _validated_counters(parent["counters"], "parent boundary") != start
            ):
                raise ValueError("measurement segment parent lineage is invalid")
            if not isinstance(resumed, Mapping):
                raise ValueError("resumed measurement segment lacks resume evidence")
            if prior_end is not None and any(start[key] != prior_end[key] for key in COUNTER_KEYS):
                raise ValueError("measurement evidence chain counters are discontinuous")
        segment_rows = segment["rows"]
        if not isinstance(segment_rows, list):
            raise ValueError("measurement segment rows are invalid")
        settings = segment["measurement"]
        if (
            not isinstance(settings, Mapping)
            or set(settings) != {"warmup_optimizer_steps", "cuda_events", "device", "output_path"}
            or settings["warmup_optimizer_steps"] != run.get("warmup_optimizer_steps")
            or settings["cuda_events"] is not True
            or settings["device"] != "cuda"
            or not isinstance(settings["output_path"], str)
            or not settings["output_path"]
        ):
            raise ValueError("measurement segment settings are invalid")
        prior_step = int(start["optimizer_step"])
        prior_tokens = int(start["target_tokens"])
        for row in segment_rows:
            if not isinstance(row, Mapping):
                raise ValueError("measurement contains a non-mapping row")
            step = row.get("optimizer_step")
            tokens = row.get("target_tokens")
            if (
                isinstance(step, bool)
                or not isinstance(step, int)
                or isinstance(tokens, bool)
                or not isinstance(tokens, int)
                or step < prior_step
                or tokens < prior_tokens
                or step > end["optimizer_step"]
                or tokens > end["target_tokens"]
            ):
                raise ValueError("measurement row counters are invalid")
            prior_step, prior_tokens = step, tokens
            rows.append(dict(row))
        if segment.get("complete") is not (index == len(segments) - 1):
            raise ValueError("only the final measurement segment may be complete")
        prior_end = end
    segment_ranges = {
        int(segment["segment_index"]): (
            _validated_counters(segment["start_counters"], "segment boundary start"),
            _validated_counters(segment["end_counters"], "segment boundary end"),
        )
        for segment in segments
    }
    for boundary in validated_boundaries:
        segment_index = boundary["segment_index"]
        if (
            isinstance(segment_index, bool)
            or not isinstance(segment_index, int)
            or segment_index not in segment_ranges
        ):
            raise ValueError("checkpoint boundary references an invalid segment")
        start, end = segment_ranges[segment_index]
        counters = _validated_counters(boundary["counters"], "checkpoint boundary")
        if any(not start[key] <= counters[key] <= end[key] for key in COUNTER_KEYS):
            raise ValueError("checkpoint boundary counters fall outside its segment")
    final_binding = run.get("final_checkpoint_boundary")
    if not isinstance(final_binding, Mapping) or set(final_binding) != MEASUREMENT_BINDING_KEYS:
        raise ValueError("final checkpoint lacks its measurement boundary binding")
    matching = [
        boundary
        for boundary in validated_boundaries
        if {key: boundary[key] for key in final_binding} == dict(final_binding)
    ]
    if len(matching) != 1 or matching[0]["kind"] != "final":
        raise ValueError("final checkpoint binding has no unique committed final boundary")
    physical = run.get("checkpoint_physical_identity")
    if not isinstance(physical, Mapping):
        raise ValueError("run lacks final checkpoint physical identity")
    final_path = run_dir / str(run.get("checkpoint"))
    if (
        not final_path.is_file()
        or final_path.stat().st_size != physical.get("size_bytes")
        or _sha256_file(final_path) != physical.get("sha256")
        or matching[0]["checkpoint_path"] != physical.get("path")
    ):
        raise ValueError("committed final boundary is not bound to the physical final checkpoint")
    if _validated_counters(matching[0]["counters"], "final boundary") != {
        "optimizer_step": int(run["final_optimizer_step"]),
        "target_tokens": int(run["final_target_tokens"]),
        "elapsed_seconds": float(run["final_elapsed_seconds"]),
    }:
        raise ValueError("final checkpoint boundary counters differ from the completed run")
    return rows


def _validate_manifest_and_config(
    run_dir: Path,
    run: Mapping[str, Any],
    expected: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    role: str,
) -> DictConfig:
    resolved_path = run_dir / str(run.get("resolved_config"))
    manifest_path = run_dir / str(run.get("run_manifest"))
    if _sha256_file(resolved_path) != run.get("resolved_config_sha256") or _sha256_file(
        manifest_path
    ) != run.get("run_manifest_sha256"):
        raise ValueError("resolved config or run-manifest physical identity changed")
    manifest = _read_json(manifest_path)
    if manifest.get("git") != {"sha": plan["git_commit"], "dirty": False, "status": []}:
        raise ValueError("run manifest is not the clean planned Git commit")
    source_files = {item["path"]: item for item in plan["source_identity"]["files"]}
    if manifest.get("lock", {}).get("sha256") != source_files["uv.lock"]["sha256"]:
        raise ValueError("run lock identity differs from the immutable plan")
    tokenizer_source = source_files["assets/tokenizers/llm-jp-v1/manifest.json"]
    if manifest.get("tokenizer", {}).get("fingerprint") != tokenizer_source["fingerprint"]:
        raise ValueError("run tokenizer identity differs from the immutable plan")
    expected_data = {
        source_files["data/manifests/fineweb2-ja-jpn-jpan.manifest.json"]["fingerprint"],
        source_files["data/manifests/fineweb-en-sample-10bt.manifest.json"]["fingerprint"],
    }
    if {item.get("fingerprint") for item in manifest.get("data", [])} != expected_data:
        raise ValueError("run data fingerprints differ from the immutable plan")
    cfg = OmegaConf.load(resolved_path)
    authority_key = _authority_key(
        role, str(run.get("candidate_id")), int(run.get("repetition", 0))
    )
    authorities = [
        item
        for item in plan.get("run_config_authorities", [])
        if isinstance(item, Mapping) and item.get("authority_key") == authority_key
    ]
    if len(authorities) != 1:
        raise ValueError("run lacks one immutable resolved-config authority")
    authority = authorities[0]
    expected_storage_safety = {
        "configured_min_free_disk_bytes": int(plan["config"]["gates"]["min_free_disk_bytes"]),
        "post_plan_free_reserve_bytes": int(authority["post_plan_free_reserve_bytes"]),
        "max_in_flight_atomic_write_bytes": int(authority["max_in_flight_atomic_write_bytes"]),
        "effective_min_free_disk_bytes": int(authority["effective_min_free_disk_bytes"]),
    }
    if run.get("storage_safety") != expected_storage_safety:
        raise ValueError("run storage-safety budget differs from immutable plan authority")
    if role in {"matrix", "pilot"} and run.get("parameter_count") != authority.get(
        "parameter_count"
    ):
        raise ValueError("observed parameter count differs from immutable storage authority")
    plain_cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(plain_cfg, Mapping) or canonical_config_sha256(plain_cfg) != authority.get(
        "canonical_config_sha256"
    ):
        raise ValueError("resolved run config differs from immutable full-config authority")
    if manifest.get("config") != {
        "path": resolved_path.name,
        "sha256": run.get("resolved_config_sha256"),
    } or manifest.get("experiment_identity", {}).get("config_sha256") != authority.get(
        "experiment_config_sha256"
    ):
        raise ValueError("run manifest config identity differs from immutable plan authority")
    observed_shape = {
        "num_layers": int(cfg.model.num_layers),
        "embed_size": int(cfg.model.embed_size),
        "num_heads": int(cfg.model.num_heads),
        "sequence_length": int(cfg.training.sequence_length),
        "batch_size": int(cfg.training.batch_size),
        "gradient_accumulation_steps": int(cfg.training.gradient_accumulation_steps),
    }
    if observed_shape != {key: int(expected[key]) for key in observed_shape}:
        raise ValueError("resolved run shape differs from the immutable matrix entry")
    if (
        cfg.runtime.device != "cuda"
        or cfg.training.precision != "bf16"
        or cfg.reproducibility.deterministic is not True
        or cfg.data.streaming.cache.dir != "/cache"
    ):
        raise ValueError("resolved run device/precision/determinism/cache differs from plan")
    if role == "matrix":
        if (
            cfg.profile.name != "dgx_candidate"
            or cfg.training.max_steps != 30
            or cfg.data.streaming.train.max_target_tokens != 983040
            or cfg.wandb.mode != "disabled"
        ):
            raise ValueError("matrix resolved config differs from its exact 30-step protocol")
    elif role == "pilot":
        pilot = plan["config"]["pilot"]
        pilot_training_seconds = int(pilot["duration_seconds"]) - int(
            pilot["finalization_reserve_seconds"]
        )
        if (
            cfg.profile.name != "pretrain_baseline"
            or cfg.training.max_steps is not None
            or cfg.training.max_time != pilot_training_seconds
            or cfg.data.streaming.validation.max_target_tokens != pilot["validation_target_tokens"]
            or cfg.training.validation_every_n_tokens != pilot["validation_every_target_tokens"]
            or cfg.training.checkpoint_every_n_tokens != pilot["recovery_every_target_tokens"]
            or cfg.training.milestone_every_n_tokens != pilot["milestone_every_target_tokens"]
            or cfg.training.log_every_n_steps != pilot["log_every_optimizer_steps"]
            or cfg.wandb.mode != "online"
            or cfg.wandb.watch.enabled
            or cfg.wandb.artifact.policy != "none"
        ):
            raise ValueError("pilot resolved config differs from selected baseline protocol")
    elif role in {"model-only", "loader-only"}:
        if (
            cfg.profile.name != "pretrain_baseline"
            or cfg.training.max_steps is not None
            or cfg.training.max_time is not None
            or cfg.measurement.enabled
            or cfg.wandb.mode != "disabled"
            or cfg.wandb.watch.enabled
            or cfg.wandb.artifact.policy != "none"
        ):
            raise ValueError("decomposition resolved config differs from the selected protocol")
    return cfg


def _telemetry_summary(
    rows: list[dict[str, Any]], run: Mapping[str, Any], gates: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, bool]]:
    if len(rows) < 2:
        raise ValueError("DGX evidence has fewer than two telemetry samples")
    times = [float(row["monotonic_seconds"]) for row in rows]
    gaps = [right - left for left, right in zip(times, times[1:])]
    if any(gap <= 0 for gap in gaps):
        raise ValueError("telemetry timestamps are not strictly increasing")
    interval = float(run.get("telemetry_interval_seconds", 1.0))
    start = float(run["telemetry_started_monotonic_seconds"])
    end = float(run["telemetry_ended_monotonic_seconds"])
    expected = max(1.0, (end - start) / interval + 1.0)
    max_gap = max(gaps)
    max_gap_allowed = interval * float(gates["max_telemetry_gap_factor"])
    host = [row["host"] for row in rows]
    gpu = [row["gpu"] for row in rows]

    def required_values(key: str) -> tuple[list[float], float]:
        values = []
        for row in gpu:
            value = row.get(key)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(f"DGX telemetry {key} contains an invalid observation")
            values.append(float(value))
        return values, len(values) / len(gpu)

    clocks, clock_coverage = required_values("sm_clock_mhz")
    power, power_coverage = required_values("power_watts")
    utilization, utilization_coverage = required_values("gpu_utilization_percent")
    temperatures, temperature_coverage = required_values("temperature_c")
    summary = {
        "samples": len(rows),
        "expected_samples": expected,
        "coverage": min(1.0, len(rows) / expected),
        "first_sample_offset_seconds": times[0] - start,
        "last_sample_offset_seconds": end - times[-1],
        "max_gap_seconds": max_gap,
        "host": {
            "min_available_memory_bytes": min(row["memory_available_bytes"] for row in host),
            "max_process_rss_bytes": max(row["process_rss_bytes"] for row in host),
            "max_process_peak_rss_bytes": max(row["process_peak_rss_bytes"] for row in host),
            "max_swap_used_bytes": max(row["swap_used_bytes"] for row in host),
            "swap_in_pages_delta": host[-1]["swap_in_pages"] - host[0]["swap_in_pages"],
            "swap_out_pages_delta": host[-1]["swap_out_pages"] - host[0]["swap_out_pages"],
            "page_faults_delta": host[-1]["page_faults"] - host[0]["page_faults"],
            "process_read_bytes_delta": host[-1]["process_read_bytes"]
            - host[0]["process_read_bytes"],
            "process_write_bytes_delta": host[-1]["process_write_bytes"]
            - host[0]["process_write_bytes"],
            "disk_read_sectors_delta": host[-1]["disk_read_sectors"] - host[0]["disk_read_sectors"],
            "disk_written_sectors_delta": host[-1]["disk_written_sectors"]
            - host[0]["disk_written_sectors"],
            "disk_io_milliseconds_delta": host[-1]["disk_io_milliseconds"]
            - host[0]["disk_io_milliseconds"],
            "network_rx_bytes_delta": host[-1]["network_rx_bytes"] - host[0]["network_rx_bytes"],
            "network_tx_bytes_delta": host[-1]["network_tx_bytes"] - host[0]["network_tx_bytes"],
            "max_load_1m": max(row["load_1m"] for row in host),
            "min_free_disk_bytes": min(row["disk_free_bytes"] for row in host),
        },
        "gpu": {
            "max_temperature_c": max(temperatures) if temperatures else None,
            "sm_clock_mhz_min": min(clocks) if clocks else None,
            "sm_clock_mhz_median": statistics.median(clocks) if clocks else None,
            "power_watts_max": max(power) if power else None,
            "power_watts_median": statistics.median(power) if power else None,
            "utilization_percent_median": statistics.median(utilization) if utilization else None,
            "utilization_percent_p95": _percentile(utilization, 0.95) if utilization else None,
            "metric_coverage": {
                "temperature_c": temperature_coverage,
                "sm_clock_mhz": clock_coverage,
                "power_watts": power_coverage,
                "gpu_utilization_percent": utilization_coverage,
            },
        },
    }
    temporal = (
        summary["coverage"] >= float(gates["min_sampler_coverage"])
        and summary["first_sample_offset_seconds"] <= max_gap_allowed
        and summary["last_sample_offset_seconds"] <= max_gap_allowed
        and max_gap <= max_gap_allowed
    )
    telemetry_gates = {
        "telemetry_error_free": not run.get("telemetry_errors"),
        "telemetry_watchdog_clear": not run.get("telemetry_violations"),
        "telemetry_temporal_coverage": temporal,
        "available_memory": summary["host"]["min_available_memory_bytes"]
        >= int(gates["min_available_memory_bytes"]),
        "free_disk": summary["host"]["min_free_disk_bytes"] >= int(gates["min_free_disk_bytes"]),
        "swap_in": summary["host"]["swap_in_pages_delta"] <= int(gates["max_swap_in_pages"]),
        "swap_out": summary["host"]["swap_out_pages_delta"] <= int(gates["max_swap_out_pages"]),
        "thermal": summary["gpu"]["max_temperature_c"] is not None
        and summary["gpu"]["max_temperature_c"] <= float(gates["max_temperature_c"]),
        "gpu_temperature_observed": temperature_coverage >= float(gates["min_gpu_metric_coverage"]),
        "gpu_clock_observed": clock_coverage >= float(gates["min_gpu_metric_coverage"]),
        "gpu_power_observed": power_coverage >= float(gates["min_gpu_metric_coverage"]),
        "gpu_utilization_observed": utilization_coverage >= float(gates["min_gpu_metric_coverage"]),
    }
    return summary, telemetry_gates


def _finite_training(metrics: list[dict[str, Any]], expected_steps: int | None = None) -> bool:
    steps = [row for row in metrics if row.get("event") == "step"]
    if not steps or (expected_steps is not None and len(steps) != expected_steps):
        return False
    keys = ("train/loss_step", "optimizer/gradient_norm", "optimizer/lr")
    return all(
        all(isinstance(row.get(key), (int, float)) and math.isfinite(row[key]) for key in keys)
        for row in steps
    )


def _validated_cuda_timings(row: Mapping[str, Any]) -> dict[str, float]:
    timings = row.get("cuda_milliseconds")
    if not isinstance(timings, Mapping) or not timings:
        raise ValueError("measured optimizer row lacks CUDA timing observations")
    if not REQUIRED_CUDA_PHASES.issubset(timings):
        missing = ", ".join(sorted(REQUIRED_CUDA_PHASES - set(timings)))
        raise ValueError(f"measured optimizer row lacks required CUDA phases: {missing}")
    validated: dict[str, float] = {}
    for key, value in timings.items():
        if (
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError("measured optimizer row has an invalid CUDA timing observation")
        validated[key] = float(value)
    return validated


def summarize_run(
    run_dir: Path,
    gates: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    plan: Mapping[str, Any],
    role: str = "matrix",
) -> dict[str, Any]:
    run = _read_json(run_dir / "run.json")
    if (
        run.get("schema_version") != 3
        or run.get("status") != "succeeded"
        or run.get("role") != role
        or run.get("plan_id") != plan["plan_id"]
        or run.get("git_commit") != plan["git_commit"]
        or run.get("image_id") != plan["image_id"]
    ):
        raise ValueError(f"run identity does not match its immutable plan: {run_dir}")
    for key in (
        "candidate_id",
        "repetition",
        "num_layers",
        "embed_size",
        "num_heads",
        "sequence_length",
        "batch_size",
        "gradient_accumulation_steps",
        "effective_target_tokens_per_step",
    ):
        if run.get(key) != expected.get(key):
            raise ValueError(f"run {key} differs from immutable plan entry: {run_dir}")
    _validate_manifest_and_config(run_dir, run, expected, plan, role=role)
    measurement = _read_json(run_dir / "measurement.json")
    rows = _validate_measurement_v3(run_dir, run, measurement)
    optimizer_rows = [row for row in rows if row.get("event") == "optimizer_step"]
    declared_measured = int(run["measured_optimizer_steps"])
    if role == "matrix":
        expected_steps = list(range(1, 31))
        if [row.get("optimizer_step") for row in optimizer_rows] != expected_steps:
            raise ValueError("matrix requires exactly optimizer steps 1..30")
        if [row.get("warmup") for row in optimizer_rows] != [step <= 10 for step in expected_steps]:
            raise ValueError("matrix warmup flags differ from exact 10+20 protocol")
        if any(row.get("target_tokens_step") != 32768 for row in optimizer_rows):
            raise ValueError("matrix optimizer updates must each train exactly 32,768 targets")
        if [row.get("target_tokens") for row in optimizer_rows] != [
            32768 * step for step in expected_steps
        ]:
            raise ValueError("matrix cumulative target counters differ from exact protocol")
        if run.get("final_optimizer_step") != 30 or run.get("final_target_tokens") != 983040:
            raise ValueError("matrix final counters differ from 30 updates / 983,040 targets")
    measured = [row for row in optimizer_rows if row.get("warmup") is False]
    if (role == "matrix" and len(measured) != declared_measured) or (
        role == "pilot" and len(measured) < declared_measured
    ):
        raise ValueError("DGX evidence does not contain the exact declared measured steps")
    first_measured_step = int(measured[0]["optimizer_step"])
    last_measured_step = int(measured[-1]["optimizer_step"])
    target_tokens = sum(int(row["target_tokens_step"]) for row in measured)
    if role == "matrix" and target_tokens != 655360:
        raise ValueError("matrix measured window must contain exactly 655,360 trained targets")
    step_wall_seconds = sum(float(row["step_wall_seconds"]) for row in measured)
    scheduled = {
        "scheduled_log": sum(
            float(row["scheduled_log_seconds"])
            for row in rows
            if row.get("event") == "scheduled_log"
            and first_measured_step <= row["optimizer_step"] <= last_measured_step
        ),
        "validation": sum(
            float(row["full_event_pause_seconds"])
            for row in rows
            if row.get("event") == "validation"
            and first_measured_step <= row["optimizer_step"] <= last_measured_step
        ),
        "recovery_checkpoint": sum(
            float(row["checkpoint_seconds"])
            for row in rows
            if row.get("event") == "checkpoint"
            and first_measured_step <= row["optimizer_step"] <= last_measured_step
        ),
        "milestone_checkpoint": sum(
            float(row["checkpoint_seconds"])
            for row in rows
            if row.get("event") == "milestone"
            and first_measured_step <= row["optimizer_step"] <= last_measured_step
        ),
    }
    scheduled_pause_seconds = sum(scheduled.values())
    decision_wall_seconds = step_wall_seconds + scheduled_pause_seconds
    if decision_wall_seconds <= 0:
        raise ValueError("DGX measured wall-time denominator is not positive")
    step_times = [float(row["step_wall_seconds"]) for row in measured]
    data_wait_seconds = sum(float(row["host_seconds"]["data_wait"]) for row in measured)
    phase_cuda: defaultdict[str, float] = defaultdict(float)
    for row in measured:
        for key, value in _validated_cuda_timings(row).items():
            phase_cuda[key] += value / 1000.0
    allocated = [int(row["pytorch_allocated_bytes"]) for row in measured]
    reserved = [int(row["pytorch_reserved_bytes"]) for row in measured]
    allocator_window = min(5, max(1, len(allocated) // 2))
    allocator_growth = max(
        0, min(allocated[-allocator_window:]) - min(allocated[:allocator_window])
    )
    telemetry, telemetry_gates = _telemetry_summary(
        _read_jsonl(run_dir / "system.jsonl"), run, gates
    )
    metrics = _read_jsonl(run_dir / "checkpoints" / "metrics.jsonl")
    checkpoint_rows = [row for row in rows if "checkpoint/size_bytes" in row]
    authority_key = _authority_key(role, str(run["candidate_id"]), int(run["repetition"]))
    atomic_write_budget = next(
        int(item["max_in_flight_atomic_write_bytes"])
        for item in plan["run_config_authorities"]
        if item["authority_key"] == authority_key
    )
    if any(int(row["checkpoint/size_bytes"]) > atomic_write_budget for row in checkpoint_rows):
        raise ValueError("observed checkpoint exceeded immutable atomic-write safety budget")
    final_checkpoint_rows = [row for row in rows if row.get("event") == "final_checkpoint"]
    validation_pauses = [
        float(row["full_event_pause_seconds"]) for row in rows if row.get("event") == "validation"
    ]
    recovery_pauses = [
        float(row["checkpoint_seconds"]) for row in rows if row.get("event") == "checkpoint"
    ]
    final_pauses = [
        float(row["checkpoint_seconds"])
        for row in final_checkpoint_rows
        if "checkpoint_seconds" in row
    ]
    run_gates = {
        "run_succeeded": True,
        "measurement_schema_v3_complete": True,
        "measurement_exact_steps": True,
        "finite_training": _finite_training(
            metrics, expected_steps=30 if role == "matrix" else None
        ),
        **telemetry_gates,
        "allocator_stable": allocator_growth <= int(gates["max_allocator_growth_bytes"]),
        "data_wait": data_wait_seconds / step_wall_seconds
        <= float(gates["max_data_wait_fraction"]),
        "checkpoint_verified": bool(run.get("checkpoint_verified")) and bool(final_checkpoint_rows),
    }
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
        "executed_optimizer_steps": len(optimizer_rows),
        "measured_optimizer_steps": len(measured),
        "executed_target_tokens": sum(int(row["target_tokens_step"]) for row in optimizer_rows),
        "measured_target_tokens": target_tokens,
        "step_wall_seconds": step_wall_seconds,
        "scheduled_pause_seconds": scheduled_pause_seconds,
        "scheduled_pause_breakdown_seconds": scheduled,
        "decision_wall_seconds": decision_wall_seconds,
        "compute_tokens_per_second": target_tokens / step_wall_seconds,
        "tokens_per_second": target_tokens / decision_wall_seconds,
        "step_time_seconds": {
            "median": statistics.median(step_times),
            "p95": _percentile(step_times, 0.95),
            "max": max(step_times),
        },
        "data_wait_seconds": data_wait_seconds,
        "data_wait_fraction": data_wait_seconds / step_wall_seconds,
        "cuda_phase_seconds": dict(sorted(phase_cuda.items())),
        "pytorch_peak_allocated_bytes": max(allocated),
        "pytorch_peak_reserved_bytes": max(reserved),
        "pytorch_allocator_baseline_growth_bytes": allocator_growth,
        "checkpoint_size_bytes": max(
            (int(row["checkpoint/size_bytes"]) for row in checkpoint_rows), default=0
        ),
        "scheduled_log_pause_seconds": [
            float(row["scheduled_log_seconds"])
            for row in rows
            if row.get("event") == "scheduled_log"
        ],
        "validation_pause_seconds": validation_pauses,
        "recovery_checkpoint_pause_seconds": recovery_pauses,
        "final_checkpoint_pause_seconds": final_pauses,
        "telemetry": telemetry,
        "gates": run_gates,
        "passed": all(run_gates.values()),
        "evidence_dir": str(run_dir),
    }


def _project_token_budget(
    duration_seconds: int,
    *,
    compute_tokens_per_second: float,
    logging_seconds: float,
    validation_seconds: float,
    recovery_seconds: float,
    milestone_seconds: float,
    final_seconds: float,
    effective_target_tokens_per_step: int,
    pilot: Mapping[str, Any],
) -> int:
    def elapsed(tokens: int) -> float:
        optimizer_steps = math.ceil(tokens / effective_target_tokens_per_step) if tokens else 0
        return (
            tokens / compute_tokens_per_second
            + (optimizer_steps // int(pilot["log_every_optimizer_steps"])) * logging_seconds
            + (tokens // int(pilot["validation_every_target_tokens"])) * validation_seconds
            + (tokens // int(pilot["recovery_every_target_tokens"])) * recovery_seconds
            + (tokens // int(pilot["milestone_every_target_tokens"])) * milestone_seconds
            + final_seconds
        )

    low = 0
    high = max(1, math.floor(compute_tokens_per_second * duration_seconds))
    while low < high:
        middle = (low + high + 1) // 2
        if elapsed(middle) <= duration_seconds:
            low = middle
        else:
            high = middle - 1
    return low


def _candidate_summary(runs: list[dict[str, Any]], cfg: Mapping[str, Any]) -> dict[str, Any]:
    first = runs[0]
    throughputs = [float(run["tokens_per_second"]) for run in runs]
    compute_throughputs = [float(run["compute_tokens_per_second"]) for run in runs]
    logging = max(
        (pause for run in runs for pause in run["scheduled_log_pause_seconds"]), default=0.0
    )
    spread = (max(throughputs) - min(throughputs)) / statistics.median(throughputs)
    validation = max(
        (pause for run in runs for pause in run["validation_pause_seconds"]), default=0.0
    )
    recovery = max(
        (pause for run in runs for pause in run["recovery_checkpoint_pause_seconds"]),
        default=0.0,
    )
    final = max(
        (pause for run in runs for pause in run["final_checkpoint_pause_seconds"]), default=0.0
    )
    checkpoint = max(recovery, final)
    max_step = max(float(run["step_time_seconds"]["max"]) for run in runs)
    required_finalization_reserve = math.ceil(
        2.0 * (max_step + logging + validation + recovery + checkpoint + final)
    )
    configured_finalization_reserve = int(cfg["pilot"]["finalization_reserve_seconds"])
    conservative_compute = min(compute_throughputs)
    token_budgets = {
        label: _project_token_budget(
            seconds,
            compute_tokens_per_second=conservative_compute,
            logging_seconds=logging,
            validation_seconds=validation,
            recovery_seconds=recovery,
            milestone_seconds=checkpoint,
            final_seconds=final,
            effective_target_tokens_per_step=int(first["effective_target_tokens_per_step"]),
            pilot=cfg["pilot"],
        )
        for label, seconds in SECONDS.items()
    }
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
        "repeatability_passed": spread <= float(cfg["selection"]["max_repeat_spread_fraction"]),
        "finalization_reserve_passed": (
            required_finalization_reserve <= configured_finalization_reserve
        ),
        "finalization_reserve": {
            "configured_seconds": configured_finalization_reserve,
            "required_seconds": required_finalization_reserve,
            "safety_factor": 2.0,
            "components_seconds": {
                "optimizer_step_max": max_step,
                "scheduled_log_max": logging,
                "validation_max": validation,
                "recovery_checkpoint_max": recovery,
                "milestone_checkpoint_max": checkpoint,
                "final_checkpoint_max": final,
            },
        },
        "throughput": {
            "min": min(throughputs),
            "median": statistics.median(throughputs),
            "max": max(throughputs),
            "spread_fraction": spread,
        },
        "compute_throughput": {
            "min": min(compute_throughputs),
            "median": statistics.median(compute_throughputs),
            "max": max(compute_throughputs),
        },
        "conservative_tokens_per_second": min(throughputs),
        "conservative_compute_tokens_per_second": conservative_compute,
        "token_budgets": token_budgets,
        "data_wait_fraction_median": statistics.median(
            float(run["data_wait_fraction"]) for run in runs
        ),
        "pytorch_peak_allocated_bytes_max": max(
            int(run["pytorch_peak_allocated_bytes"]) for run in runs
        ),
        "host_min_available_memory_bytes": min(
            int(run["telemetry"]["host"]["min_available_memory_bytes"]) for run in runs
        ),
        "host_min_free_disk_bytes": min(
            int(run["telemetry"]["host"]["min_free_disk_bytes"]) for run in runs
        ),
        "checkpoint_size_bytes_max": max(int(run["checkpoint_size_bytes"]) for run in runs),
        "projected_overhead_seconds": {
            "scheduled_log_each": logging,
            "validation_each": validation,
            "recovery_each": recovery,
            "milestone_each": checkpoint,
            "final_once": final,
        },
        "runs": runs,
    }


def select_candidate(
    candidates: list[dict[str, Any]], selection: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    target = int(selection["target_tokens_7d"])
    eligible = [
        candidate
        for candidate in candidates
        if candidate["all_runs_passed"]
        and candidate["repeatability_passed"]
        and candidate["finalization_reserve_passed"]
        and candidate["token_budgets"]["7_days"] >= target
    ]
    if not eligible:
        raise ValueError("no candidate passes every gate, repeatability, and seven-day floor")
    fastest = max(candidate["conservative_tokens_per_second"] for candidate in eligible)
    speed_floor = fastest * (1.0 - float(selection["max_slowdown_fraction"]))
    eligible = [
        candidate
        for candidate in eligible
        if candidate["conservative_tokens_per_second"] >= speed_floor
    ]
    deepest = max(candidate["num_layers"] for candidate in eligible)
    eligible = [candidate for candidate in eligible if candidate["num_layers"] == deepest]
    depth_fastest = max(candidate["conservative_tokens_per_second"] for candidate in eligible)
    context_floor = depth_fastest * float(selection["context_throughput_floor_fraction"])
    eligible = [
        candidate
        for candidate in eligible
        if candidate["conservative_tokens_per_second"] >= context_floor
    ]
    longest = max(candidate["sequence_length"] for candidate in eligible)
    eligible = [candidate for candidate in eligible if candidate["sequence_length"] == longest]
    if len(eligible) != 1:
        raise ValueError("selection rule did not produce one unique candidate")
    return eligible[0], eligible


def _load_plan(evidence_root: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    plan = _read_json(evidence_root / "plan.json")
    if plan.get("schema_version") != 3 or plan.get("ticket") != "DGX-001":
        raise ValueError("DGX plan does not use schema version 3")
    claimed = plan.get("plan_id")
    unsigned = dict(plan)
    unsigned.pop("plan_id", None)
    if claimed != _canonical_sha256(unsigned):
        raise ValueError("DGX immutable plan hash does not verify")
    planned_protocol = {key: plan["config"][key] for key in PROTOCOL_CONFIG_KEYS}
    active_protocol = {key: cfg[key] for key in PROTOCOL_CONFIG_KEYS}
    if planned_protocol != active_protocol:
        raise ValueError("DGX summary config differs from the executed protocol")
    cache = _read_json(evidence_root / "cache-integrity.json")
    if cache.get("unchanged") is not True or cache.get("before") != plan.get("cache_before"):
        raise ValueError("DGX cache before/after identity did not remain exact")
    expected_runs = build_matrix_plan(plan["config"])
    if plan.get("runs") != expected_runs:
        raise ValueError("DGX plan run list differs from the exact matrix")
    mode = plan.get("config", {}).get("mode")
    if mode == "matrix":
        expected_authority_keys = {
            _authority_key("matrix", entry["candidate_id"], entry["repetition"])
            for entry in expected_runs
        }
    elif mode == "decompose":
        selected = plan.get("selected")
        if not isinstance(selected, Mapping):
            raise ValueError("DGX decomposition plan has no selected candidate")
        expected_authority_keys = {
            _authority_key(role, str(selected["candidate_id"]), repetition)
            for role in ("model-only", "loader-only")
            for repetition in range(1, 4)
        }
    elif mode == "pilot":
        selected = plan.get("selected")
        if not isinstance(selected, Mapping):
            raise ValueError("DGX pilot plan has no selected candidate")
        expected_authority_keys = {_authority_key("pilot", str(selected["candidate_id"]), 1)}
    else:
        raise ValueError("DGX evidence plan has an invalid execution mode")
    authorities = plan.get("run_config_authorities")
    if not isinstance(authorities, list):
        raise ValueError("DGX plan lacks immutable run-config authorities")
    observed_authority_keys = []
    for authority in authorities:
        if (
            not isinstance(authority, Mapping)
            or set(authority)
            != {
                "authority_key",
                "role",
                "candidate_id",
                "repetition",
                "canonical_config_sha256",
                "experiment_config_sha256",
                "parameter_count",
                "max_in_flight_atomic_write_bytes",
                "post_plan_free_reserve_bytes",
                "effective_min_free_disk_bytes",
            }
            or authority["authority_key"]
            != _authority_key(
                str(authority["role"]),
                str(authority["candidate_id"]),
                int(authority["repetition"]),
            )
            or not all(
                isinstance(authority[key], str) and len(authority[key]) == 64
                for key in ("canonical_config_sha256", "experiment_config_sha256")
            )
            or any(
                isinstance(authority[key], bool)
                or not isinstance(authority[key], int)
                or authority[key] <= 0
                for key in (
                    "parameter_count",
                    "max_in_flight_atomic_write_bytes",
                    "post_plan_free_reserve_bytes",
                    "effective_min_free_disk_bytes",
                )
            )
            or authority["post_plan_free_reserve_bytes"] != 100_000_000_000
            or authority["effective_min_free_disk_bytes"]
            != max(
                120_000_000_000,
                authority["post_plan_free_reserve_bytes"]
                + authority["max_in_flight_atomic_write_bytes"],
            )
        ):
            raise ValueError("DGX plan contains an invalid run-config authority")
        observed_authority_keys.append(authority["authority_key"])
    if (
        len(observed_authority_keys) != len(set(observed_authority_keys))
        or set(observed_authority_keys) != expected_authority_keys
    ):
        raise ValueError("DGX plan run-config authorities differ from its exact execution roles")
    return plan


def _validate_matrix_authority(plan: Mapping[str, Any]) -> dict[str, Any]:
    authority = plan.get("matrix_summary_identity")
    if not isinstance(authority, Mapping):
        raise ValueError("DGX auxiliary evidence lacks matrix selection authority")
    summary_path = Path(str(authority.get("path")))
    matrix_plan_path = Path(str(authority.get("matrix_plan_path")))
    if (
        not summary_path.is_file()
        or _sha256_file(summary_path) != authority.get("sha256")
        or not matrix_plan_path.is_file()
        or _sha256_file(matrix_plan_path) != authority.get("matrix_plan_sha256")
    ):
        raise ValueError("matrix selection authority changed after the auxiliary plan")
    matrix_plan = _read_json(matrix_plan_path)
    unsigned = dict(matrix_plan)
    matrix_plan_id = unsigned.pop("plan_id", None)
    matrix_protocol = {key: matrix_plan.get("config", {}).get(key) for key in PROTOCOL_CONFIG_KEYS}
    auxiliary_protocol = {key: plan.get("config", {}).get(key) for key in PROTOCOL_CONFIG_KEYS}
    if (
        matrix_plan_id != _canonical_sha256(unsigned)
        or matrix_plan_id != authority.get("matrix_plan_id")
        or _canonical_sha256(matrix_protocol) != authority.get("matrix_protocol_sha256")
        or matrix_protocol != auxiliary_protocol
        or matrix_plan.get("config", {}).get("mode") != "matrix"
        or matrix_plan.get("git_commit") != plan.get("git_commit")
        or matrix_plan.get("image_id") != plan.get("image_id")
        or matrix_plan.get("runs") != build_matrix_plan(matrix_plan["config"])
        or matrix_plan.get("selected") is not None
        or matrix_plan.get("matrix_summary_identity") is not None
    ):
        raise ValueError("matrix selection source plan differs from auxiliary protocol authority")
    summary = _read_json(summary_path)
    if (
        summary.get("schema_version") != 3
        or summary.get("verdict") != "PASS"
        or summary.get("plan_id") != matrix_plan_id
        or summary.get("git_commit") != authority.get("git_commit")
        or summary.get("image_id") != authority.get("image_id")
        or summary.get("selection_rule") != authority.get("selection_rule")
        or summary.get("selection_rule") != matrix_plan["config"]["selection"]
        or summary.get("selected", {}).get("candidate_id") != authority.get("selected")
        or summary.get("selected", {}).get("conservative_compute_tokens_per_second")
        != authority.get("end_to_end_compute_tokens_per_second")
    ):
        raise ValueError("matrix selection authority is not one passing exact summary")
    return summary


def _committed_profile_matches(
    selected: Mapping[str, Any], *, finalization_reserve_seconds: int
) -> bool:
    root = Path(__file__).resolve().parents[2]
    with hydra.initialize_config_dir(version_base=None, config_dir=str(root / "config")):
        cfg = hydra.compose(config_name="train", overrides=["profile=pretrain_baseline"])
    keys = (
        "num_layers",
        "embed_size",
        "num_heads",
        "sequence_length",
        "batch_size",
        "gradient_accumulation_steps",
    )
    observed = {
        "num_layers": int(cfg.model.num_layers),
        "embed_size": int(cfg.model.embed_size),
        "num_heads": int(cfg.model.num_heads),
        "sequence_length": int(cfg.training.sequence_length),
        "batch_size": int(cfg.training.batch_size),
        "gradient_accumulation_steps": int(cfg.training.gradient_accumulation_steps),
    }
    return observed == {key: int(selected[key]) for key in keys} and int(cfg.training.max_time) == (
        SECONDS["1_hour"] - finalization_reserve_seconds
    )


def summarize_evidence(
    evidence_root: Path, config: Mapping[str, Any] | DictConfig
) -> dict[str, Any]:
    cfg = validate_dgx_config(config)
    root = Path(evidence_root)
    plan = _load_plan(root, cfg)
    expected_entries = build_matrix_plan(cfg)
    expected_by_key = {
        (entry["candidate_id"], entry["repetition"]): entry for entry in expected_entries
    }
    run_dirs = sorted(path.parent for path in root.glob("*/run.json"))
    if len(run_dirs) != 27:
        raise ValueError("final DGX matrix requires exactly 27 run directories")
    runs = []
    observed_keys: set[tuple[str, int]] = set()
    for run_dir in run_dirs:
        raw = _read_json(run_dir / "run.json")
        key = (str(raw.get("candidate_id")), int(raw.get("repetition", 0)))
        if key in observed_keys or key not in expected_by_key:
            raise ValueError("DGX matrix contains a duplicate or unplanned candidate/repetition")
        observed_keys.add(key)
        runs.append(
            summarize_run(
                run_dir,
                cfg["gates"],
                expected=expected_by_key[key],
                plan=plan,
                role="matrix",
            )
        )
    if observed_keys != set(expected_by_key):
        raise ValueError("DGX matrix is missing one or more planned runs")
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["candidate_id"]].append(run)
    candidates = [
        _candidate_summary(sorted(items, key=lambda item: item["repetition"]), cfg)
        for _, items in sorted(grouped.items())
    ]
    selected, finalists = select_candidate(candidates, cfg["selection"])
    full_tokens = int(selected["token_budgets"]["7_days"])
    checkpoint_size = int(selected["checkpoint_size_bytes_max"])
    milestone_copies = math.ceil(full_tokens / int(cfg["pilot"]["milestone_every_target_tokens"]))
    checkpoint_copies = 3 + 1 + 1 + 1 + milestone_copies
    cache_bytes = int(plan["data_cache_max_bytes"])
    current_evidence_bytes = _tree_size(root)
    evidence_bytes = current_evidence_bytes + int(
        cfg["gates"]["post_matrix_evidence_reserve_bytes"]
    )
    checkpoint_footprint = checkpoint_size * checkpoint_copies
    output_footprint = checkpoint_footprint + evidence_bytes
    storage = _storage_headroom(
        cache_root=Path(str(plan["cache_before"]["root"])),
        output_root=root,
        cache_existing_bytes=int(plan["cache_before"]["size_bytes"]),
        cache_planned_bytes=cache_bytes,
        output_existing_bytes=current_evidence_bytes,
        output_planned_bytes=output_footprint,
        headroom_ratio=float(cfg["gates"]["storage_headroom_ratio"]),
        post_plan_free_reserve_bytes=int(cfg["gates"]["post_plan_free_reserve_bytes"]),
    )
    storage_passed = bool(storage["passed"])
    profile_matches = _committed_profile_matches(
        selected,
        finalization_reserve_seconds=int(cfg["pilot"]["finalization_reserve_seconds"]),
    )
    verdict = "PASS" if storage_passed and profile_matches else "FAIL"
    return {
        "schema_version": 3,
        "ticket": "DGX-001",
        "plan_id": plan["plan_id"],
        "git_commit": plan["git_commit"],
        "image_id": plan["image_id"],
        "measurement_conditions": {
            "arms": 9,
            "repetitions": 3,
            "executed_optimizer_steps_per_run": 30,
            "warmup_optimizer_steps": 10,
            "measured_optimizer_steps": 20,
            "executed_target_tokens": 26_542_080,
            "measured_target_tokens": 17_694_720,
            "precision": "bf16",
            "deterministic": True,
            "measurement_schema_version": 3,
            "wandb": "disabled",
            "cache_unchanged": True,
        },
        "candidates": candidates,
        "selection_rule": dict(cfg["selection"]),
        "selection_finalists": [item["candidate_id"] for item in finalists],
        "selected": selected,
        "committed_pretrain_baseline_matches_selected": profile_matches,
        "plan": {
            "token_budgets": selected["token_budgets"],
            "one_hour_wall_budget_seconds": 3600,
            "one_hour_training_max_time_seconds": (
                3600 - int(cfg["pilot"]["finalization_reserve_seconds"])
            ),
            "pilot_wall_budget_seconds": int(cfg["pilot"]["duration_seconds"]),
            "pilot_training_max_time_seconds": (
                int(cfg["pilot"]["duration_seconds"])
                - int(cfg["pilot"]["finalization_reserve_seconds"])
            ),
            "finalization_reserve": selected["finalization_reserve"],
            "validation_every_target_tokens": cfg["pilot"]["validation_every_target_tokens"],
            "recovery_checkpoint_every_target_tokens": cfg["pilot"]["recovery_every_target_tokens"],
            "milestone_every_target_tokens": cfg["pilot"]["milestone_every_target_tokens"],
            "checkpoint_size_bytes": checkpoint_size,
            "checkpoint_copies": {
                "rotating_recovery": 3,
                "atomic_temporary": 1,
                "best": 1,
                "final": 1,
                "milestones": milestone_copies,
                "total": checkpoint_copies,
            },
            "cache_bytes": cache_bytes,
            "checkpoint_footprint_bytes": checkpoint_footprint,
            "matrix_and_future_evidence_bytes": evidence_bytes,
            "output_footprint_bytes": output_footprint,
            "storage_filesystems": storage["filesystems"],
            "storage_same_filesystem": storage["same_filesystem"],
            "storage_headroom_ratio": storage["headroom_ratio"],
            "storage_headroom_passed": storage["headroom_passed"],
            "post_plan_free_reserve_bytes": cfg["gates"]["post_plan_free_reserve_bytes"],
            "post_plan_reserve_passed": storage["post_plan_reserve_passed"],
            "wandb_mode": "online",
            "wandb_watch": False,
            "wandb_artifact_policy": "none",
            "wandb_scalar_cadence_optimizer_steps": 25,
        },
        "bottleneck_status": "pending selected-profile decomposition",
        "verdict": verdict,
    }


def _decomposition_conclusion(
    *, model_ratio: float, loader_ratio: float, min_loader_ratio: float
) -> tuple[bool, str]:
    loader_headroom_passed = loader_ratio >= min_loader_ratio
    if not loader_headroom_passed:
        return False, "insufficient loader headroom; long run blocked"
    if loader_ratio <= model_ratio:
        return True, "data loader is the nearer measured ceiling"
    return True, "model forward/backward/optimizer is the nearer measured ceiling"


def summarize_decomposition(
    evidence_root: Path, config: Mapping[str, Any] | DictConfig
) -> dict[str, Any]:
    cfg = validate_dgx_config(config)
    root = Path(evidence_root)
    plan = _load_plan(root, cfg)
    _validate_matrix_authority(plan)
    authority = plan["matrix_summary_identity"]
    selected = plan["selected"]
    raw_paths = sorted(root.glob("*/decomposition.json"))
    if len(raw_paths) != 6:
        raise ValueError("decomposition requires exactly 3 model-only and 3 loader-only runs")
    results: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    expected_keys = {
        (role, repetition) for role in ("model-only", "loader-only") for repetition in range(1, 4)
    }
    observed_keys = set()
    for path in raw_paths:
        raw = _read_json(path)
        key = (raw.get("role"), raw.get("repetition"))
        if key in observed_keys or key not in expected_keys:
            raise ValueError("decomposition contains duplicate/unplanned evidence")
        observed_keys.add(key)
        if (
            raw.get("schema_version") != 3
            or raw.get("status") != "succeeded"
            or raw.get("plan_id") != plan["plan_id"]
            or raw.get("candidate_id") != selected["candidate_id"]
            or raw.get("git_commit") != plan["git_commit"]
            or raw.get("image_id") != plan["image_id"]
            or raw.get("warmup_optimizer_steps") != 10
            or raw.get("measured_optimizer_steps") != 20
            or raw.get("telemetry_errors")
            or raw.get("telemetry_violations")
        ):
            raise ValueError("decomposition run identity/health differs from plan")
        for shape_key in (
            "num_layers",
            "embed_size",
            "num_heads",
            "sequence_length",
            "batch_size",
            "gradient_accumulation_steps",
            "effective_target_tokens_per_step",
        ):
            if raw.get(shape_key) != selected.get(shape_key):
                raise ValueError("decomposition shape differs from selected candidate")
        _validate_manifest_and_config(path.parent, raw, selected, plan, role=str(raw["role"]))
        telemetry, telemetry_gates = _telemetry_summary(
            _read_jsonl(path.parent / "system.jsonl"), raw, cfg["gates"]
        )
        if not all(telemetry_gates.values()):
            raise ValueError("decomposition telemetry fails the DGX health/coverage gates")
        rows = raw.get("rows")
        if not isinstance(rows, list) or len(rows) != 30:
            raise ValueError("decomposition run must contain exactly 30 updates")
        measured = [row for row in rows if row.get("warmup") is False]
        if (
            len(measured) != 20
            or any(row.get("target_tokens") != 32768 for row in rows)
            or any(not math.isfinite(float(row["wall_seconds"])) for row in rows)
        ):
            raise ValueError("decomposition rows differ from exact target protocol")
        wall = sum(float(row["wall_seconds"]) for row in measured)
        result = {
            "role": raw["role"],
            "repetition": raw["repetition"],
            "target_tokens": 655360,
            "wall_seconds": wall,
            "tokens_per_second": 655360 / wall,
            "telemetry": telemetry,
            "evidence": str(path),
        }
        if raw["role"] == "model-only":
            if any(
                not math.isfinite(float(row["loss"]))
                or not math.isfinite(float(row["gradient_norm"]))
                for row in rows
            ):
                raise ValueError("model-only decomposition contains non-finite training")
        results[raw["role"]].append(result)
    if observed_keys != expected_keys:
        raise ValueError("decomposition is incomplete")
    end_to_end = float(authority["end_to_end_compute_tokens_per_second"])
    summaries = {}
    for role, items in sorted(results.items()):
        rates = [item["tokens_per_second"] for item in items]
        summaries[role] = {
            "runs": sorted(items, key=lambda item: item["repetition"]),
            "throughput": {
                "min": min(rates),
                "median": statistics.median(rates),
                "max": max(rates),
                "spread_fraction": (max(rates) - min(rates)) / statistics.median(rates),
            },
            "supply_ratio_min": min(rates) / end_to_end,
        }
    model_ratio = summaries["model-only"]["supply_ratio_min"]
    loader_ratio = summaries["loader-only"]["supply_ratio_min"]
    threshold = float(cfg["gates"]["min_loader_supply_ratio"])
    loader_headroom_passed, bottleneck = _decomposition_conclusion(
        model_ratio=model_ratio,
        loader_ratio=loader_ratio,
        min_loader_ratio=threshold,
    )
    return {
        "schema_version": 3,
        "ticket": "DGX-001",
        "plan_id": plan["plan_id"],
        "git_commit": plan["git_commit"],
        "image_id": plan["image_id"],
        "selected_candidate": selected["candidate_id"],
        "matrix_summary": dict(authority),
        "end_to_end_compute_tokens_per_second": end_to_end,
        "minimum_loader_supply_ratio": threshold,
        "decomposition": summaries,
        "loader_headroom_passed": loader_headroom_passed,
        "named_bottleneck": bottleneck,
        "verdict": "PASS" if loader_headroom_passed else "FAIL",
    }


def _validated_wandb_evidence(run_dir: Path, run: Mapping[str, Any]) -> dict[str, Any]:
    recorded = run.get("wandb_evidence")
    if not isinstance(recorded, Mapping):
        raise ValueError("pilot lacks W&B evidence identity")
    relative_path = Path(str(recorded.get("path", "")))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("pilot W&B evidence path is not run-relative")
    path = run_dir / relative_path
    if not path.is_file() or _sha256_file(path) != recorded.get("sha256"):
        raise ValueError("pilot W&B evidence physical identity changed")
    rows = _read_jsonl(path)
    if len(rows) != recorded.get("rows") or any(
        not isinstance(row, Mapping) or row.get("schema_version") != 1 for row in rows
    ):
        raise ValueError("pilot W&B evidence record shape/count is invalid")
    init = [
        row for row in rows if row.get("action") == "init" and row.get("outcome") == "succeeded"
    ]
    finished = [
        row for row in rows if row.get("action") == "finish" and row.get("outcome") == "succeeded"
    ]
    critical_failures = sorted(
        {
            str(row.get("action"))
            for row in rows
            if row.get("outcome") == "failed"
            and row.get("action") in {"log", "summary", "runtime_summary"}
        }
    )
    successful_init = init[-1] if len(init) == 1 else {}
    return {
        "path": str(relative_path),
        "sha256": recorded.get("sha256"),
        "rows": len(rows),
        "init_status": "succeeded" if len(init) == 1 else None,
        "mode": successful_init.get("mode"),
        "run_id": successful_init.get("run_id"),
        "run_url": successful_init.get("run_url"),
        "watch_disabled": any(
            row.get("action") == "watch" and row.get("outcome") == "disabled" for row in rows
        ),
        "finish_succeeded": len(finished) == 1,
        "artifact_uploads": sum(
            row.get("action") == "artifact" and row.get("outcome") == "uploaded" for row in rows
        ),
        "critical_failures": critical_failures,
    }


def summarize_pilot(evidence_root: Path, config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    cfg = validate_dgx_config(config)
    root = Path(evidence_root)
    plan = _load_plan(root, cfg)
    _validate_matrix_authority(plan)
    selected = plan.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("pilot has no summary-selected matrix entry")
    run_dirs = sorted(path.parent for path in root.glob("*/run.json"))
    if len(run_dirs) != 1:
        raise ValueError("selected-profile pilot requires exactly one run")
    expected = {**dict(selected), "repetition": 1}
    result = summarize_run(run_dirs[0], cfg["gates"], expected=expected, plan=plan, role="pilot")
    run = _read_json(run_dirs[0] / "run.json")
    wandb = _validated_wandb_evidence(run_dirs[0], run)
    pilot_gates = {
        "training_gates": result["passed"],
        "duration_representative": float(run["final_elapsed_seconds"])
        >= (
            int(cfg["pilot"]["duration_seconds"])
            - int(cfg["pilot"]["finalization_reserve_seconds"])
        ),
        "training_and_final_checkpoint_within_wall_budget": (
            float(run["final_elapsed_seconds"])
            + max(result["final_checkpoint_pause_seconds"], default=0.0)
            <= int(cfg["pilot"]["duration_seconds"])
        ),
        "validation_observed": bool(result["validation_pause_seconds"]),
        "recovery_observed": bool(result["recovery_checkpoint_pause_seconds"]),
        "sample_observed": (run_dirs[0] / str(run.get("samples"))).is_file(),
        "wandb_online_visible": wandb.get("init_status") == "succeeded"
        and wandb.get("mode") == "online"
        and isinstance(wandb.get("run_id"), str)
        and isinstance(wandb.get("run_url"), str),
        "wandb_finished": wandb.get("finish_succeeded") is True,
        "wandb_watch_off": wandb.get("watch_disabled") is True,
        "wandb_no_artifacts": wandb.get("artifact_uploads") == 0,
        "wandb_logging_healthy": not wandb.get("critical_failures"),
    }
    return {
        "schema_version": 3,
        "ticket": "DGX-001",
        "plan_id": plan["plan_id"],
        "git_commit": plan["git_commit"],
        "image_id": plan["image_id"],
        "selected_candidate": selected["candidate_id"],
        "run": result,
        "wandb": wandb,
        "gates": pilot_gates,
        "verdict": "PASS" if all(pilot_gates.values()) else "FAIL",
    }
