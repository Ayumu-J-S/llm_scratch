#!/usr/bin/env python3
"""Verify and summarize the predeclared WB-001 DGX R2 matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


MATRIX = (
    (1, 1, "disabled"),
    (1, 2, "offline-off"),
    (1, 3, "offline-on"),
    (2, 1, "offline-on"),
    (2, 2, "disabled"),
    (2, 3, "offline-off"),
    (3, 1, "offline-off"),
    (3, 2, "offline-on"),
    (3, 3, "disabled"),
)
ARM_CONFIG = {
    "disabled": ("disabled", False),
    "offline-off": ("offline", False),
    "offline-on": ("offline", True),
}
ROOT = Path(__file__).resolve().parents[3]


class Gates:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def require(self, name: str, passed: bool, detail: Any = None) -> None:
        self.checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            self.failures.append(name)

    def note(self, name: str, triggered: bool, detail: Any) -> None:
        if triggered:
            self.warnings.append(name)
            self.checks.append({"name": name, "passed": True, "warning": True, "detail": detail})


def nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 < percentile <= 1.0:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(float(value) for value in values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def parse_size_bytes(value: str) -> int:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTPE]?i?B)\s*", value, re.IGNORECASE)
    if match is None:
        raise ValueError(f"unsupported size: {value!r}")
    unit = match.group(2)
    prefix = unit[0].upper() if unit.lower() not in {"b", "ib"} else ""
    exponent = "KMGTPE".find(prefix) + 1 if prefix else 0
    return int(
        float(match.group(1)) * ((1024 if unit.lower().endswith("ib") else 1000) ** exponent)
    )


def parse_vmstat(text: str) -> list[dict[str, int]]:
    lines = [line.split() for line in text.splitlines() if line.strip()]
    header_index = next(
        (
            index
            for index, fields in enumerate(lines)
            if fields and fields[0] == "r" and {"swpd", "free", "si", "so"}.issubset(fields)
        ),
        None,
    )
    if header_index is None:
        return []
    header = lines[header_index]
    header = header[: header.index("UTC")] if "UTC" in header else header
    rows = []
    for fields in lines[header_index + 1 :]:
        try:
            rows.append({name: int(fields[index]) for index, name in enumerate(header)})
        except (IndexError, ValueError):
            continue
    return rows[1:] if rows else []  # vmstat's first row is the since-boot average.


def normalized_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(config))
    wandb = result.get("wandb")
    if isinstance(wandb, dict):
        wandb.pop("mode", None)
        wandb.pop("name", None)
        if isinstance(wandb.get("watch"), dict):
            wandb["watch"].pop("enabled", None)
    return result


def paired_regression_percent(baseline: float, candidate: float) -> float:
    if baseline <= 0.0 or candidate <= 0.0:
        raise ValueError("paired throughput values must be positive")
    return 100.0 * (baseline - candidate) / baseline


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, Mapping):
            digest.update(b"mapping:")
            for key in sorted(item, key=repr):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for member in item:
                update(member)
        elif hasattr(item, "dtype") and hasattr(item, "shape") and hasattr(item, "tobytes"):
            digest.update(f"array:{item.dtype}:{tuple(item.shape)}:".encode())
            digest.update(item.tobytes())
        else:
            digest.update(f"{type(item).__name__}:{item!r}:".encode())

    update(value)
    return digest.hexdigest()


# Kept as a test-facing spelling for the pure digest invariant.
_canonical_digest = _digest


def _env(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            result[key] = value
    return result


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _config(path: Path) -> dict[str, Any]:
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"expected resolved mapping: {path}")
    return value


def _checkpoint(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)["state"]
    counters = {key: value for key, value in state["counters"].items() if key != "elapsed_seconds"}
    resume = {
        key: state[key]
        for key in (
            "optimizer",
            "scheduler",
            "precision",
            "event_state",
            "rng",
            "stream_cursor",
            "run_identity",
        )
    }
    resume["counters"] = counters
    return {
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
        "model_digest": _digest(state["model"]),
        "resume_digest": _digest(resume),
        "cursor_digest": _digest(state["stream_cursor"]),
        "counters": counters,
    }


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite(member) for member in value.values())
    if isinstance(value, list):
        return all(_finite(member) for member in value)
    return True


def summarize_steps(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in rows if row.get("event") == "optimizer_step" and not row["warmup"]]
    if not rows:
        raise ValueError("no post-warmup optimizer rows")
    seconds = [float(row["step_wall_seconds"]) for row in rows]
    targets = [int(row["target_tokens_step"]) for row in rows]
    wait = [float(row["host_seconds"]["data_wait"]) for row in rows]
    result = {
        "steps": len(rows),
        "target_tokens": sum(targets),
        "wall_seconds": sum(seconds),
        "target_tokens_per_second": sum(targets) / sum(seconds),
        "step_median_seconds": statistics.median(seconds),
        "step_p95_seconds": nearest_rank_percentile(seconds, 0.95),
        "step_max_seconds": max(seconds),
        "data_wait_fraction": sum(wait) / sum(seconds),
    }
    for name, key in (
        ("allocated", "pytorch_allocated_bytes"),
        ("reserved", "pytorch_reserved_bytes"),
    ):
        values = [int(row[key]) for row in rows]
        result[name] = {
            "maximum": max(values),
            "first_20_median": statistics.median(values[:20]),
            "last_20_median": statistics.median(values[-20:]),
        }
    result["host_phase_median_seconds"] = {
        phase: statistics.median(float(row["host_seconds"].get(phase, 0.0)) for row in rows)
        for phase in sorted({key for row in rows for key in row["host_seconds"]})
    }
    result["cuda_phase_median_milliseconds"] = {
        phase: statistics.median(float(row["cuda_milliseconds"].get(phase, 0.0)) for row in rows)
        for phase in sorted({key for row in rows for key in row["cuda_milliseconds"]})
    }
    return result


def _gpu(path: Path, wall: float) -> dict[str, Any]:
    rows = [line.split(",") for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if len(row) == 11]
    expected = max(1, math.ceil(wall / 0.2))

    def stats(index: int) -> dict[str, float] | None:
        values = []
        for row in rows:
            try:
                values.append(float(row[index].strip()))
            except ValueError:
                pass
        return (
            {"min": min(values), "median": statistics.median(values), "max": max(values)}
            if values
            else None
        )

    return {
        "samples": len(rows),
        "expected": expected,
        "coverage": len(rows) / expected,
        "utilization_percent": stats(2),
        "sm_clock_mhz": stats(6),
        "power_w": stats(9),
        "temperature_c": stats(10),
    }


def _host(path: Path, wall: float) -> dict[str, Any]:
    rows = parse_vmstat(path.read_text(encoding="utf-8"))
    expected = max(1, math.ceil(wall))
    longest = current = 0
    for row in rows:
        current = current + 1 if row.get("si", 0) > 0 or row.get("so", 0) > 0 else 0
        longest = max(longest, current)
    available = [row.get("free", 0) + row.get("buff", 0) + row.get("cache", 0) for row in rows]
    return {
        "samples": len(rows),
        "expected": expected,
        "coverage": len(rows) / expected,
        "minimum_free_buffer_cache_bytes": min(available) * 1024 if available else None,
        "longest_swap_io_run": longest,
    }


def _container(path: Path, wall: float) -> dict[str, Any]:
    rows = [line.split("|") for line in path.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if len(row) == 7]
    memory = []
    for row in rows:
        try:
            memory.append(parse_size_bytes(row[2].split("/", 1)[0].strip()))
        except ValueError:
            pass
    expected = max(1, math.ceil(wall))
    return {
        "samples": len(rows),
        "expected": expected,
        "coverage": len(rows) / expected,
        "max_memory_bytes": max(memory) if memory else None,
    }


def _inventory(root: Path) -> bool:
    inventory = root / "artifact-sha256.txt"
    listed = set()
    for line in inventory.read_text(encoding="utf-8").splitlines():
        digest, path = line.split(maxsplit=1)
        artifact = Path(path)
        listed.add(artifact.resolve())
        if not artifact.is_file() or _sha(artifact) != digest:
            return False
    actual = {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path != inventory and not path.is_symlink()
    }
    return listed == actual


def _mount(inspect: Mapping[str, Any], destination: str) -> Mapping[str, Any] | None:
    return next((mount for mount in inspect["Mounts"] if mount["Destination"] == destination), None)


def _lifecycle(arm: str, events: Sequence[Mapping[str, Any]]) -> tuple[bool, list[list[Any]]]:
    pairs = [[row.get("action"), row.get("outcome")] for row in events]
    common = all(pair[1] != "failed" for pair in pairs) and ["artifact", "uploaded"] not in pairs
    if arm == "disabled":
        expected = ["init", "disabled"] in pairs and not any(
            pair[0] in {"watch", "unwatch", "finish"} for pair in pairs
        )
    elif arm == "offline-off":
        expected = all(
            pair in pairs
            for pair in (["init", "succeeded"], ["watch", "disabled"], ["finish", "succeeded"])
        ) and not any(pair[0] == "unwatch" for pair in pairs)
    else:
        expected = all(
            pair in pairs
            for pair in (
                ["init", "succeeded"],
                ["watch", "succeeded"],
                ["unwatch", "succeeded"],
                ["finish", "succeeded"],
            )
        )
    if arm != "disabled":
        expected = expected and any(
            row.get("action") == "init" and row.get("outcome") == "succeeded" and row.get("run_id")
            for row in events
        )
    return common and expected, pairs


def _wandb_storage(root: Path, arm: str) -> tuple[bool, dict[str, Any]]:
    files = [
        path
        for name in ("wandb", "wandb-cache", "wandb-config", "wandb-data")
        for path in (root / name).rglob("*")
        if path.is_file() and not path.is_symlink()
    ]
    forbidden = [
        str(path.relative_to(root))
        for path in files
        if path.suffix == ".pt"
        or path.name in {"bilingual.jsonl", "final.pt", "best.pt"}
        or path.name.startswith(("recovery-step-", "milestone-step-"))
    ]
    run_files = [
        path
        for path in files
        if path.suffix == ".wandb"
        or any(part.startswith(("run-", "offline-run-")) for part in path.parts)
    ]
    expected = not run_files if arm == "disabled" else bool(run_files)
    return expected and not forbidden, {
        "file_count": len(files),
        "run_file_count": len(run_files),
        "bytes": sum(path.stat().st_size for path in files),
        "forbidden": forbidden,
    }


def _hardware(manifest: Mapping[str, Any]) -> dict[str, Any]:
    environment = manifest["hardware_software"]
    return {
        "host": environment["host"],
        "os": environment["os"],
        "architecture": environment["architecture"],
        "python": environment["python"],
        "torch": environment["torch"],
        "driver": environment["cuda"]["driver_version"],
        "devices": environment["cuda"]["devices"],
        "bf16": environment["cuda"]["bf16_supported"],
        "image": environment["container_image"],
    }


def _run(
    root: Path, repetition: int, position: int, arm: str, matrix: Mapping[str, str], gates: Gates
) -> dict[str, Any]:
    run_id = f"r{repetition}-p{position}-{arm}"
    prefix = run_id + ":"
    conditions = _env(root / "conditions.env")
    inspect = json.loads((root / "container-inspect.json").read_text(encoding="utf-8"))[0]
    config = _config(root / "hydra/resolved_config.yaml")
    manifest = _json(root / "hydra/run_manifest.json")
    metrics = _jsonl(root / "checkpoints/metrics.jsonl")
    events = _jsonl(root / "checkpoints/wandb_events.jsonl")
    measurement = _json(root / "measurement.json")
    checkpoint = _checkpoint(root / "checkpoints/final.pt")
    wall = (int(conditions["end_unix_ns"]) - int(conditions["start_unix_ns"])) / 1e9

    workspace, cache, evidence = (
        _mount(inspect, path) for path in ("/workspace", "/cache", "/evidence")
    )
    gates.require(prefix + "inventory", _inventory(root))
    gates.require(
        prefix + "identity_isolation",
        conditions["run_id"] == run_id
        and conditions["repetition"] == str(repetition)
        and conditions["position"] == str(position)
        and conditions["arm"] == arm
        and conditions["commit"] == matrix["measured_commit"]
        and conditions["image_id"] == matrix["image_id"]
        and conditions["exit_code"] == "0"
        and manifest["git"]["sha"] == matrix["measured_commit"]
        and inspect["Image"] == matrix["image_id"]
        and inspect["HostConfig"]["NetworkMode"] == "none"
        and workspace is not None
        and workspace["RW"] is False
        and workspace["Source"] == matrix["source_root"]
        and cache is not None
        and cache["RW"] is True
        and cache["Source"] == matrix["cache_root"]
        and evidence is not None
        and evidence["RW"] is True
        and evidence["Source"] == str(root),
    )
    gates.require(
        prefix + "cache_unchanged",
        conditions["cache_unchanged"] == "true"
        and conditions["cache_before"] == matrix["cache_baseline_sha256"]
        and conditions["cache_after"] == matrix["cache_baseline_sha256"],
    )

    mode, watch = ARM_CONFIG[arm]
    wandb = config["wandb"]
    gates.require(
        prefix + "arm_config",
        wandb["mode"] == mode
        and wandb["watch"]["enabled"] is watch
        and wandb["artifact"]["policy"] == "none",
    )
    training = config["training"]
    targets_per_step = (
        int(training["batch_size"])
        * int(training["sequence_length"])
        * int(training["gradient_accumulation_steps"])
    )
    expected_targets = targets_per_step * 100
    measured = [row for row in measurement["rows"] if row.get("event") == "optimizer_step"]
    steps = [row for row in metrics if row.get("event") == "step"]
    logs = [row for row in metrics if row.get("event") == "log"]
    scheduled = [row for row in measurement["rows"] if row.get("event") == "scheduled_log"]
    validations = [row for row in metrics if row.get("event") == "validation"]
    final = [row for row in metrics if row.get("event") == "final_checkpoint"]
    log_steps = list(range(10, 101, 10))
    gates.require(
        prefix + "fixed_work",
        config["profile"]["name"] == "stability_smoke"
        and training["max_steps"] == 100
        and training["precision"] == "bf16"
        and measurement["complete"] is True
        and measurement["warmup_optimizer_steps"] == 10
        and measurement["cuda_events"] is True
        and len(measured) == len(steps) == 100
        and [row["optimizer_step"] for row in measured] == list(range(1, 101))
        and [row["optimizer_step"] for row in steps] == list(range(1, 101))
        and all(row["warmup"] is (index <= 10) for index, row in enumerate(measured, 1))
        and all(row["target_tokens_step"] == targets_per_step for row in measured)
        and sum(row["train/target_tokens_step"] for row in steps) == expected_targets
        and len(final) == 1
        and final[0]["target_tokens"] == expected_targets
        and checkpoint["counters"]["optimizer_step"] == 100
        and checkpoint["counters"]["target_tokens"] == expected_targets
        and _finite(metrics),
    )
    gates.require(
        prefix + "cadence_validation",
        [row["optimizer_step"] for row in logs] == log_steps
        and [row["optimizer_step"] for row in scheduled] == log_steps
        and [row["optimizer_step"] for row in validations] == [100]
        and (root / "checkpoints/best.pt").is_file(),
    )
    lifecycle_ok, lifecycle = _lifecycle(arm, events)
    storage_ok, storage = _wandb_storage(root, arm)
    gates.require(prefix + "wandb_lifecycle", lifecycle_ok, lifecycle)
    gates.require(prefix + "wandb_storage", storage_ok, storage)

    performance = summarize_steps(measured)
    resources = {
        "gpu": _gpu(root / "gpu.csv", wall),
        "host": _host(root / "host-vmstat.txt", wall),
        "container": _container(root / "container-stats.txt", wall),
    }
    for name, resource in resources.items():
        gates.require(prefix + name + "_coverage", resource["coverage"] >= 0.9, resource)
    gates.require(prefix + "swap", resources["host"]["longest_swap_io_run"] < 3, resources["host"])
    for name in ("allocated", "reserved"):
        memory = performance[name]
        limit = memory["first_20_median"] + max(128 * 1024**2, 0.05 * memory["first_20_median"])
        gates.require(prefix + name + "_stable", memory["last_20_median"] <= limit, memory)
    gates.require(prefix + "data_wait_fail", performance["data_wait_fraction"] <= 0.10)
    gates.note(
        prefix + "data_wait_investigate",
        0.05 < performance["data_wait_fraction"] <= 0.10,
        performance["data_wait_fraction"],
    )
    tail = performance["step_p95_seconds"] / performance["step_median_seconds"]
    gates.note(prefix + "step_tail_investigate", tail > 1.5, tail)

    trajectory = [
        {key: value for key, value in row.items() if key != "elapsed_seconds"} for row in steps
    ]
    return {
        "run_id": run_id,
        "repetition": repetition,
        "position": position,
        "arm": arm,
        "normalized_config": _digest(normalized_config(config)),
        "manifest": {
            "experiment_id": manifest["experiment_id"],
            "git_sha": manifest["git"]["sha"],
            "tokenizer": manifest["tokenizer"]["fingerprint"],
            "data": [item["fingerprint"] for item in manifest["data"]],
            "hardware": _digest(_hardware(manifest)),
        },
        "trajectory": _digest(trajectory),
        "checkpoint": checkpoint,
        "performance": performance,
        "scheduled_log_seconds": {
            "total": sum(row["scheduled_log_seconds"] for row in scheduled),
            "median": statistics.median(row["scheduled_log_seconds"] for row in scheduled),
            "max": max(row["scheduled_log_seconds"] for row in scheduled),
        },
        "resources": resources,
        "container_wall_seconds": wall,
        "wandb": {"lifecycle": lifecycle, "storage": storage},
        "disk": {
            "free_before_bytes": int(conditions["free_before_bytes"]),
            "free_after_bytes": int(conditions["free_after_bytes"]),
            "checkpoint_files": len(list((root / "checkpoints").glob("*.pt"))),
            "checkpoint_bytes": sum(
                path.stat().st_size for path in (root / "checkpoints").glob("*.pt")
            ),
        },
        "hashes": {
            name: _sha(root / path)
            for name, path in {
                "config": "hydra/resolved_config.yaml",
                "manifest": "hydra/run_manifest.json",
                "measurement": "measurement.json",
                "metrics": "checkpoints/metrics.jsonl",
                "wandb_events": "checkpoints/wandb_events.jsonl",
                "gpu": "gpu.csv",
                "vmstat": "host-vmstat.txt",
                "container": "container-stats.txt",
            }.items()
        },
    }


def build_summary(
    evidence: Path, *, expected_commit: str, expected_image_id: str
) -> dict[str, Any]:
    matrix = _env(evidence / "matrix.env")
    gates = Gates()
    diagnose = _json(evidence / "diagnose.json")
    prime = _env(evidence / "cache-prime/conditions.env")
    gates.require(
        "matrix_identity",
        matrix["ticket"] == "WB-001"
        and matrix["measured_commit"] == expected_commit
        and matrix["image_id"] == expected_image_id
        and diagnose["cuda"]["available"] is True
        and diagnose["cuda"]["bf16_supported"] is True
        and matrix["runner_sha256"] == _sha(ROOT / "docs/experiments/evidence/run_wb001_dgx.sh")
        and matrix["verifier_sha256"] == _sha(Path(__file__).resolve()),
    )
    gates.require(
        "cache_prime",
        prime["exit_code"] == "0"
        and prime["cache_after"] == matrix["cache_baseline_sha256"]
        and _inventory(evidence / "cache-prime"),
    )
    runs = [
        _run(evidence / f"r{rep}-p{position}-{arm}", rep, position, arm, matrix, gates)
        for rep, position, arm in MATRIX
    ]
    for name, values in {
        "normalized_config": [run["normalized_config"] for run in runs],
        "experiment_id": [run["manifest"]["experiment_id"] for run in runs],
        "manifest_git": [run["manifest"]["git_sha"] for run in runs],
        "tokenizer": [run["manifest"]["tokenizer"] for run in runs],
        "data": [_digest(run["manifest"]["data"]) for run in runs],
        "hardware": [run["manifest"]["hardware"] for run in runs],
        "trajectory": [run["trajectory"] for run in runs],
        "model": [run["checkpoint"]["model_digest"] for run in runs],
        "resume": [run["checkpoint"]["resume_digest"] for run in runs],
        "cursor": [run["checkpoint"]["cursor_digest"] for run in runs],
    }.items():
        gates.require(name + "_exact", len(set(values)) == 1)

    pairs = []
    for repetition in (1, 2, 3):
        arms = {run["arm"]: run for run in runs if run["repetition"] == repetition}
        throughput = {
            arm: run["performance"]["target_tokens_per_second"] for arm, run in arms.items()
        }
        pairs.append(
            {
                "repetition": repetition,
                "offline_off_vs_disabled": paired_regression_percent(
                    throughput["disabled"], throughput["offline-off"]
                ),
                "offline_on_vs_disabled": paired_regression_percent(
                    throughput["disabled"], throughput["offline-on"]
                ),
                "watch_vs_offline_off": paired_regression_percent(
                    throughput["offline-off"], throughput["offline-on"]
                ),
            }
        )
    aggregates = {}
    for name in ("offline_off_vs_disabled", "offline_on_vs_disabled", "watch_vs_offline_off"):
        values = [pair[name] for pair in pairs]
        aggregates[name] = {
            "values": values,
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
        gates.require(
            name + "_under_10_percent", aggregates[name]["median"] < 10.0, aggregates[name]
        )
        gates.note(
            name + "_investigate_5_percent",
            aggregates[name]["median"] >= 5.0,
            aggregates[name],
        )

    verdict = "FAIL" if gates.failures else "PASS WITH NOTE" if gates.warnings else "PASS"
    return {
        "schema_version": 1,
        "ticket": "WB-001",
        "review_size": "R2",
        "measured_commit": expected_commit,
        "image_id": expected_image_id,
        "matrix": [
            {
                "run_id": f"r{rep}-p{position}-{arm}",
                "repetition": rep,
                "position": position,
                "arm": arm,
            }
            for rep, position, arm in MATRIX
        ],
        "runs": runs,
        "pairs": pairs,
        "paired_aggregates": aggregates,
        "gates": gates.checks,
        "failures": gates.failures,
        "warnings": gates.warnings,
        "verdict": verdict,
        "limitations": [
            "network=none and artifact policy none make no online auth, quota, retention, or upload claim",
            "DGX Spark unified-memory headroom is interpreted from host, container, and allocator evidence rather than nvidia-smi memory alone",
            "W&B binary run files are inventoried and hashed but not decoded by this verifier",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-image-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    evidence = args.evidence_root.resolve()
    output = args.output.resolve() if args.output else evidence / "wb001-r2-summary.json"
    summary = build_summary(
        evidence, expected_commit=args.expected_commit, expected_image_id=args.expected_image_id
    )
    temporary = output.with_name("." + output.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "verdict": summary["verdict"]}, sort_keys=True))
    return 1 if summary["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
