import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import hydra
import pytest
from omegaconf import OmegaConf

from dgx.planning import (
    PROTOCOL_CONFIG_KEYS,
    _decomposition_conclusion,
    _decomposition_decision,
    _online_adjusted_projection,
    _project_token_budget,
    _telemetry_summary,
    _validate_matrix_authority,
    _validated_wandb_evidence,
    build_matrix_plan,
    select_candidate,
    summarize_evidence,
    summarize_run,
)
from runtime.config import validate_training_config
from runtime.reproducibility import canonical_config_sha256, experiment_config_sha256


ROOT = Path(__file__).resolve().parent.parent
COMMIT = "a" * 40
IMAGE = "sha256:" + "b" * 64
PLAN_ID = ""


def compose_train(profile: str, overrides: list[str] | None = None):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        return hydra.compose(
            config_name="train", overrides=[f"profile={profile}", *(overrides or [])]
        )


def compose_dgx(overrides: list[str] | None = None):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        return hydra.compose(config_name="dgx", overrides=overrides or [])


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_dgx_profiles_are_real_cuda_bf16_and_baseline_is_bounded():
    smoke = compose_train("dgx_smoke")
    candidate = compose_train("dgx_candidate")
    baseline = compose_train("pretrain_baseline")
    for config in (smoke, candidate, baseline):
        validate_training_config(config)
        assert config.runtime.device == "cuda"
        assert config.training.precision == "bf16"
        assert config.data.mode == "streaming"
        assert config.data.streaming.require_manifests is True
        assert config.reproducibility.deterministic is True
        assert config.wandb.watch.enabled is False
        assert config.wandb.artifact.policy == "none"
    assert smoke.measurement.enabled is True
    assert candidate.measurement.enabled is True
    assert baseline.measurement.enabled is True
    assert baseline.measurement.cuda_events is True
    assert baseline.training.max_time == 3480
    assert baseline.training.max_time + compose_dgx().pilot.finalization_reserve_seconds == 3600
    assert baseline.training.validation_every_n_tokens == 5_000_000
    assert baseline.training.checkpoint_every_n_tokens == 2_500_000
    assert baseline.training.milestone_every_n_tokens == 100_000_000
    assert baseline.training.log_every_n_steps == 25
    assert baseline.model.num_layers == 18
    assert baseline.training.sequence_length == 1024
    assert baseline.training.batch_size == 8
    assert baseline.wandb.mode == "online"


def test_matrix_is_exact_nine_by_three_with_equal_work_and_rotated_repeats():
    config = compose_dgx()
    plan = build_matrix_plan(config)
    assert len(plan) == 27
    assert len({entry["candidate_id"] for entry in plan}) == 9
    assert {entry["effective_target_tokens_per_step"] for entry in plan} == {32768}
    assert {entry["max_steps"] for entry in plan} == {30}
    assert {entry["max_target_tokens"] for entry in plan} == {983040}
    assert {entry["repetition"] for entry in plan} == {1, 2, 3}
    first_ids = [
        next(entry["candidate_id"] for entry in plan if entry["repetition"] == repetition)
        for repetition in (1, 2, 3)
    ]
    assert len(set(first_ids)) == 3
    assert config.gates.min_free_disk_bytes == 120_000_000_000
    assert config.gates.post_plan_free_reserve_bytes == 100_000_000_000


def test_dgx_config_rejects_unsupported_conservative_quantile():
    config = compose_dgx(["selection.conservative_throughput_quantile=median"])
    with pytest.raises(ValueError, match="slowest repetition"):
        build_matrix_plan(config)


def _source_identity(config) -> dict:
    tokenizer = str(config.tokenizer.expected_fingerprint)
    ja = str(config.data.streaming.train.sources[0].expected_fingerprint)
    en = str(config.data.streaming.train.sources[1].expected_fingerprint)
    files = [
        {"path": "uv.lock", "size_bytes": 1, "sha256": "lock"},
        {
            "path": "assets/tokenizers/llm-jp-v1/manifest.json",
            "size_bytes": 1,
            "sha256": "tokenizer",
            "fingerprint": tokenizer,
        },
        {
            "path": "data/manifests/fineweb2-ja-jpn-jpan.manifest.json",
            "size_bytes": 1,
            "sha256": "ja",
            "fingerprint": ja,
        },
        {
            "path": "data/manifests/fineweb-en-sample-10bt.manifest.json",
            "size_bytes": 1,
            "sha256": "en",
            "fingerprint": en,
        },
    ]
    return {"sha256": _canonical_sha256(files), "files": files}


def _write_plan(root: Path, config) -> dict:
    cfg = (
        OmegaConf.to_container(config, resolve=True)
        if OmegaConf.is_config(config)
        else dict(config)
    )
    train = compose_train("dgx_candidate")
    cache_root = root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = {
        "root": str(cache_root),
        "files": 2,
        "size_bytes": 1_000_000_000,
        "sha256": "cache",
        "entries": [],
    }
    matrix_runs = build_matrix_plan(cfg)
    authorities = []
    for entry in matrix_runs:
        resolved = _candidate_config(entry)
        plain = OmegaConf.to_container(resolved, resolve=True)
        parameter_count = 100_000_000 + entry["num_layers"]
        max_in_flight = parameter_count * 128 + 4_000_000_000
        authorities.append(
            {
                "authority_key": (f"matrix:{entry['candidate_id']}:r{entry['repetition']}"),
                "role": "matrix",
                "candidate_id": entry["candidate_id"],
                "repetition": entry["repetition"],
                "canonical_config_sha256": canonical_config_sha256(plain),
                "experiment_config_sha256": experiment_config_sha256(plain),
                "parameter_count": parameter_count,
                "max_in_flight_atomic_write_bytes": max_in_flight,
                "post_plan_free_reserve_bytes": 100_000_000_000,
                "effective_min_free_disk_bytes": max(
                    120_000_000_000, 100_000_000_000 + max_in_flight
                ),
            }
        )
    payload = {
        "schema_version": 3,
        "ticket": "DGX-001",
        "config": cfg,
        "git_commit": COMMIT,
        "image_id": IMAGE,
        "host_user": {"uid": 1000, "gid": 1000},
        "source_identity": _source_identity(train),
        "cache_before": cache,
        "data_cache_max_bytes": 80_000_000_000,
        "runs": matrix_runs,
        "run_config_authorities": authorities,
        "selected": None,
        "matrix_summary_identity": None,
    }
    payload["plan_id"] = _canonical_sha256(payload)
    _write_json(root / "plan.json", payload)
    _write_json(
        root / "cache-integrity.json",
        {"before": cache, "after": cache, "unchanged": True},
    )
    return payload


def _candidate_config(entry: dict):
    return compose_train(
        "dgx_candidate",
        [
            f"model.num_layers={entry['num_layers']}",
            f"model.embed_size={entry['embed_size']}",
            f"model.num_heads={entry['num_heads']}",
            f"training.sequence_length={entry['sequence_length']}",
            f"training.batch_size={entry['batch_size']}",
            f"training.gradient_accumulation_steps={entry['gradient_accumulation_steps']}",
            "training.max_steps=30",
            "data.streaming.train.max_target_tokens=983040",
            "data.streaming.cache.dir=/cache",
            "measurement.output_path=/evidence/measurement.json",
            "wandb.mode=disabled",
        ],
    )


def _telemetry(swap_out: int = 0, *, gap: float = 1.0) -> list[dict]:
    rows = []
    for index, moment in enumerate((0.0, gap, gap + 1.0)):
        rows.append(
            {
                "monotonic_seconds": moment,
                "collection_duration_seconds": 0.0,
                "host": {
                    "memory_available_bytes": 900_000_000_000,
                    "process_rss_bytes": 2_000_000_000,
                    "process_peak_rss_bytes": 3_000_000_000,
                    "swap_used_bytes": 0,
                    "swap_in_pages": 0,
                    "swap_out_pages": swap_out if index == 2 else 0,
                    "page_faults": index,
                    "process_read_bytes": index,
                    "process_write_bytes": index,
                    "disk_read_sectors": index,
                    "disk_written_sectors": index,
                    "disk_io_milliseconds": index,
                    "network_rx_bytes": index,
                    "network_tx_bytes": index,
                    "load_1m": 1.0,
                    "disk_free_bytes": 900_000_000_000,
                },
                "gpu": {
                    "temperature_c": 65.0,
                    "sm_clock_mhz": 1800.0,
                    "power_watts": 100.0,
                    "gpu_utilization_percent": 90.0,
                },
            }
        )
    return rows


def _throughput(entry: dict) -> float:
    values = {
        "p70-ctx1024": 18_000,
        "p70-ctx2048": 12_000,
        "p70-ctx4096": 6_000,
        "p85-ctx1024": 14_000,
        "p85-ctx2048": 9_000,
        "p85-ctx4096": 4_000,
        "p99-ctx1024": 11_000,
        "p99-ctx2048": 7_000,
        "p99-ctx4096": 3_000,
    }
    return values[entry["candidate_id"]] * (1.0 - 0.01 * (entry["repetition"] - 1))


def _fake_run(
    root: Path,
    entry: dict,
    plan: dict,
    throughput: float,
    *,
    swap_out: int = 0,
    telemetry_gap: float = 1.0,
) -> Path:
    run_dir = root / f"{entry['candidate_id']}-r{entry['repetition']}"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "final.pt"
    checkpoint.write_bytes(f"{entry['candidate_id']}-{entry['repetition']}".encode())
    physical_path = "/evidence/checkpoints/final.pt"
    config = _candidate_config(entry)
    resolved = run_dir / "resolved_config.yaml"
    OmegaConf.save(config, resolved, resolve=True)
    source = {item["path"]: item for item in plan["source_identity"]["files"]}
    authority_key = f"matrix:{entry['candidate_id']}:r{entry['repetition']}"
    authority = next(
        item for item in plan["run_config_authorities"] if item["authority_key"] == authority_key
    )
    manifest = {
        "git": {"sha": COMMIT, "dirty": False, "status": []},
        "lock": {"sha256": source["uv.lock"]["sha256"]},
        "tokenizer": {
            "fingerprint": source["assets/tokenizers/llm-jp-v1/manifest.json"]["fingerprint"]
        },
        "data": [
            {
                "fingerprint": source["data/manifests/fineweb2-ja-jpn-jpan.manifest.json"][
                    "fingerprint"
                ]
            },
            {
                "fingerprint": source["data/manifests/fineweb-en-sample-10bt.manifest.json"][
                    "fingerprint"
                ]
            },
        ],
        "config": {"path": resolved.name, "sha256": _sha256(resolved)},
        "experiment_identity": {
            "config_sha256": authority["experiment_config_sha256"],
            "operational_exclusions": ["artifacts.resume_path", "measurement", "wandb"],
        },
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    rows = []
    for step in range(1, 31):
        wall = 32768 / throughput
        rows.append(
            {
                "event": "optimizer_step",
                "optimizer_step": step,
                "target_tokens": step * 32768,
                "target_tokens_step": 32768,
                "warmup": step <= 10,
                "step_wall_seconds": wall,
                "host_seconds": {"data_wait": wall * 0.01},
                "cuda_milliseconds": {
                    "forward": wall * 250,
                    "backward": wall * 600,
                    "optimizer": wall * 100,
                },
                "pytorch_allocated_bytes": 10_000_000_000 + step,
                "pytorch_reserved_bytes": 11_000_000_000,
            }
        )
        if step == 20:
            rows.extend(
                [
                    {
                        "event": "validation",
                        "optimizer_step": 20,
                        "target_tokens": 655360,
                        "full_event_pause_seconds": 3.0,
                    },
                    {
                        "event": "scheduled_log",
                        "optimizer_step": 20,
                        "target_tokens": 655360,
                        "scheduled_log_seconds": 0.1,
                    },
                    {
                        "event": "checkpoint",
                        "optimizer_step": 20,
                        "target_tokens": 655360,
                        "checkpoint_seconds": 2.0,
                        "checkpoint/size_bytes": 1_000_000_000,
                    },
                ]
            )
        if step == 30:
            rows.extend(
                [
                    {
                        "event": "scheduled_log",
                        "optimizer_step": 30,
                        "target_tokens": 983040,
                        "scheduled_log_seconds": 0.1,
                    },
                    {
                        "event": "final_checkpoint",
                        "optimizer_step": 30,
                        "target_tokens": 983040,
                        "checkpoint_seconds": 2.0,
                        "checkpoint/size_bytes": 1_000_000_000,
                    },
                ]
            )
    evidence_id = "evidence"
    checkpoint_identity = {"experiment_id": "fixture"}
    boundary_bindings = [
        {
            "boundary_index": index,
            "boundary_id": f"boundary-{kind}",
            "evidence_id": evidence_id,
            "segment_index": 0,
            "kind": kind,
            "counters": {
                "optimizer_step": step,
                "target_tokens": step * 32768,
                "elapsed_seconds": float(step),
            },
        }
        for index, (kind, step) in enumerate((("best", 20), ("recovery", 20), ("final", 30)))
    ]
    boundaries = [
        {**binding, "status": "committed", "checkpoint_path": physical_path}
        for binding in boundary_bindings
    ]
    _write_json(
        run_dir / "measurement.json",
        {
            "schema_version": 3,
            "checkpoint_identity": checkpoint_identity,
            "measurement_evidence_id": evidence_id,
            "complete": True,
            "segments": [
                {
                    "segment_index": 0,
                    "start_counters": {
                        "optimizer_step": 0,
                        "target_tokens": 0,
                        "elapsed_seconds": 0.0,
                    },
                    "end_counters": {
                        "optimizer_step": 30,
                        "target_tokens": 983040,
                        "elapsed_seconds": 30.0,
                    },
                    "resumed_from": None,
                    "parent_boundary_id": None,
                    "measurement": {
                        "warmup_optimizer_steps": 10,
                        "cuda_events": True,
                        "device": "cuda",
                        "output_path": "/evidence/measurement.json",
                    },
                    "complete": True,
                    "rows": rows,
                }
            ],
            "checkpoint_boundaries": boundaries,
        },
    )
    metric_rows = [
        {
            "event": "step",
            "optimizer_step": step,
            "train/loss_step": 10.0,
            "optimizer/gradient_norm": 1.0,
            "optimizer/lr": 0.0003,
        }
        for step in range(1, 31)
    ]
    (checkpoints / "metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in metric_rows), encoding="utf-8"
    )
    telemetry = _telemetry(swap_out, gap=telemetry_gap)
    (run_dir / "system.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in telemetry), encoding="utf-8"
    )
    run = {
        "schema_version": 3,
        "status": "succeeded",
        "role": "matrix",
        "plan_id": plan["plan_id"],
        "candidate_id": entry["candidate_id"],
        "repetition": entry["repetition"],
        "git_commit": COMMIT,
        "image_id": IMAGE,
        "parameter_count": 100_000_000 + entry["num_layers"],
        "storage_safety": {
            "configured_min_free_disk_bytes": 120_000_000_000,
            "post_plan_free_reserve_bytes": authority["post_plan_free_reserve_bytes"],
            "max_in_flight_atomic_write_bytes": authority["max_in_flight_atomic_write_bytes"],
            "effective_min_free_disk_bytes": authority["effective_min_free_disk_bytes"],
        },
        "num_layers": entry["num_layers"],
        "embed_size": entry["embed_size"],
        "num_heads": entry["num_heads"],
        "sequence_length": entry["sequence_length"],
        "batch_size": entry["batch_size"],
        "gradient_accumulation_steps": entry["gradient_accumulation_steps"],
        "effective_target_tokens_per_step": 32768,
        "warmup_optimizer_steps": 10,
        "measured_optimizer_steps": 20,
        "final_optimizer_step": 30,
        "final_target_tokens": 983040,
        "final_elapsed_seconds": 30.0,
        "checkpoint_verified": True,
        "checkpoint": "checkpoints/final.pt",
        "checkpoint_physical_identity": {
            "path": physical_path,
            "sha256": _sha256(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
        },
        "checkpoint_identity": checkpoint_identity,
        "final_checkpoint_boundary": boundary_bindings[-1],
        "resolved_config": resolved.name,
        "resolved_config_sha256": _sha256(resolved),
        "run_manifest": "run_manifest.json",
        "run_manifest_sha256": _sha256(run_dir / "run_manifest.json"),
        "telemetry_interval_seconds": 1.0,
        "telemetry_started_monotonic_seconds": 0.0,
        "telemetry_ended_monotonic_seconds": telemetry[-1]["monotonic_seconds"],
        "telemetry_errors": [],
        "telemetry_violations": [],
    }
    _write_json(run_dir / "run.json", run)
    return run_dir


def _matrix_fixture(root: Path):
    config = compose_dgx(["mode=matrix"])
    plan = _write_plan(root, config)
    for entry in build_matrix_plan(config):
        _fake_run(root, entry, plan, _throughput(entry))
    return config, plan


def test_summary_gates_exact_matrix_and_selects_committed_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dgx.planning.shutil.disk_usage", lambda _path: SimpleNamespace(free=1_000_000_000_000)
    )
    config, _ = _matrix_fixture(tmp_path)
    summary = summarize_evidence(tmp_path, config)
    assert summary["schema_version"] == 3
    assert summary["verdict"] == "PASS"
    assert summary["selected"]["candidate_id"] == "p70-ctx1024"
    assert summary["measurement_conditions"]["executed_target_tokens"] == 26_542_080
    assert summary["measurement_conditions"]["measured_target_tokens"] == 17_694_720
    assert summary["committed_pretrain_baseline_matches_selected"] is True
    assert summary["plan"]["token_budgets"]["1_hour"] > 0
    assert summary["plan"]["token_budgets"]["7_days"] >= 1_000_000_000
    assert summary["plan"]["checkpoint_copies"]["milestones"] > 0
    assert summary["plan"]["storage_headroom_passed"] is True
    assert summary["plan"]["storage_same_filesystem"] is True
    assert summary["plan"]["one_hour_wall_budget_seconds"] == 3600
    assert summary["plan"]["one_hour_training_max_time_seconds"] == 3480
    assert summary["plan"]["token_budget_training_seconds"]["1_hour"] == 3480
    selected = summary["selected"]
    overhead = selected["projected_overhead_seconds"]
    assert summary["plan"]["token_budgets"]["1_hour"] == _project_token_budget(
        3480,
        compute_tokens_per_second=selected["conservative_compute_tokens_per_second"],
        logging_seconds=overhead["scheduled_log_each"],
        validation_seconds=overhead["validation_each"],
        recovery_seconds=overhead["recovery_each"],
        milestone_seconds=overhead["milestone_each"],
        final_seconds=0.0,
        effective_target_tokens_per_step=selected["effective_target_tokens_per_step"],
        pilot=OmegaConf.to_container(config.pilot, resolve=True),
    )
    assert summary["plan"]["pilot_wall_budget_seconds"] == 1800
    assert summary["plan"]["pilot_training_max_time_seconds"] == 1680
    assert summary["plan"]["finalization_reserve"]["configured_seconds"] == 120
    assert summary["selected"]["projected_overhead_seconds"]["scheduled_log_each"] == 0.1


def test_summary_fails_explicit_low_storage_without_weakening_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dgx.planning.shutil.disk_usage", lambda _path: SimpleNamespace(free=1_000_000)
    )
    config, _ = _matrix_fixture(tmp_path)
    summary = summarize_evidence(tmp_path, config)
    assert summary["verdict"] == "FAIL"
    assert summary["plan"]["storage_headroom_passed"] is False
    assert summary["plan"]["storage_same_filesystem"] is True


def test_summary_accounts_cache_and_output_filesystems_separately(tmp_path, monkeypatch):
    config, _ = _matrix_fixture(tmp_path)
    monkeypatch.setattr(
        "dgx.planning._filesystem_device",
        lambda path: 1 if Path(path).name == "cache" else 2,
    )
    monkeypatch.setattr(
        "dgx.planning.shutil.disk_usage",
        lambda path: SimpleNamespace(
            free=1_000_000_000_000 if Path(path).name == "cache" else 1_000_000
        ),
    )
    summary = summarize_evidence(tmp_path, config)
    assert summary["verdict"] == "FAIL"
    assert summary["plan"]["storage_same_filesystem"] is False
    reports = {tuple(item["roles"]): item for item in summary["plan"]["storage_filesystems"]}
    assert reports[("cache",)]["headroom_passed"] is True
    assert reports[("output",)]["headroom_passed"] is False


def test_post_plan_free_reserve_is_separate_from_headroom_ratio(tmp_path, monkeypatch):
    config, _ = _matrix_fixture(tmp_path)
    monkeypatch.setattr(
        "dgx.planning._filesystem_device",
        lambda path: 1 if Path(path).name == "cache" else 2,
    )
    monkeypatch.setattr(
        "dgx.planning.shutil.disk_usage",
        lambda path: SimpleNamespace(
            free=160_000_000_000 if Path(path).name == "cache" else 1_000_000_000_000
        ),
    )
    summary = summarize_evidence(tmp_path, config)
    reports = {tuple(item["roles"]): item for item in summary["plan"]["storage_filesystems"]}
    assert reports[("cache",)]["headroom_passed"] is True
    assert reports[("cache",)]["post_plan_reserve_passed"] is False
    assert summary["plan"]["storage_headroom_passed"] is True
    assert summary["plan"]["post_plan_reserve_passed"] is False
    assert summary["verdict"] == "FAIL"


def test_primary_denominator_includes_scheduled_pauses_exactly_once(tmp_path):
    config = OmegaConf.to_container(compose_dgx(), resolve=True)
    plan = _write_plan(tmp_path, config)
    entry = build_matrix_plan(config)[0]
    run_dir = _fake_run(tmp_path, entry, plan, 18_000)
    result = summarize_run(run_dir, config["gates"], expected=entry, plan=plan, role="matrix")
    assert result["scheduled_pause_breakdown_seconds"] == {
        "scheduled_log": pytest.approx(0.2),
        "validation": 3.0,
        "recovery_checkpoint": 2.0,
        "milestone_checkpoint": 0,
    }
    assert result["decision_wall_seconds"] == pytest.approx(result["step_wall_seconds"] + 5.2)
    assert result["tokens_per_second"] < result["compute_tokens_per_second"]
    assert result["data_wait_fraction"] == pytest.approx(0.01)


def test_data_wait_gate_uses_optimizer_step_wall_not_scheduled_pause(tmp_path):
    config = OmegaConf.to_container(compose_dgx(), resolve=True)
    plan = _write_plan(tmp_path, config)
    entry = build_matrix_plan(config)[0]
    run_dir = _fake_run(tmp_path, entry, plan, 18_000)
    measurement_path = run_dir / "measurement.json"
    measurement = json.loads(measurement_path.read_text())
    rows = measurement["segments"][0]["rows"]
    for row in rows:
        if row.get("event") == "optimizer_step" and row.get("warmup") is False:
            row["host_seconds"]["data_wait"] = row["step_wall_seconds"] * 0.11
        if row.get("event") == "validation":
            row["full_event_pause_seconds"] = 1000.0
    _write_json(measurement_path, measurement)
    result = summarize_run(run_dir, config["gates"], expected=entry, plan=plan, role="matrix")
    assert result["data_wait_fraction"] == pytest.approx(0.11)
    assert result["gates"]["data_wait"] is False


def test_projected_token_budget_charges_scheduled_logging():
    pilot = {
        "log_every_optimizer_steps": 2,
        "validation_every_target_tokens": 1_000_000,
        "recovery_every_target_tokens": 1_000_000,
        "milestone_every_target_tokens": 1_000_000,
    }
    without_logging = _project_token_budget(
        50,
        compute_tokens_per_second=100.0,
        logging_seconds=0.0,
        validation_seconds=0.0,
        recovery_seconds=0.0,
        milestone_seconds=0.0,
        final_seconds=0.0,
        effective_target_tokens_per_step=100,
        pilot=pilot,
    )
    with_logging = _project_token_budget(
        50,
        compute_tokens_per_second=100.0,
        logging_seconds=10.0,
        validation_seconds=0.0,
        recovery_seconds=0.0,
        milestone_seconds=0.0,
        final_seconds=0.0,
        effective_target_tokens_per_step=100,
        pilot=pilot,
    )
    assert without_logging == 5000
    assert with_logging < without_logging
    assert with_logging % 100 == 0


def test_online_log_latency_reprojects_and_regates_every_matrix_candidate():
    config = OmegaConf.to_container(compose_dgx(), resolve=True)
    candidates = []
    for candidate_id, depth, compute in (("shallow", 18, 20_000.0), ("deep", 26, 18_000.0)):
        candidates.append(
            {
                "candidate_id": candidate_id,
                "num_layers": depth,
                "sequence_length": 1024,
                "effective_target_tokens_per_step": 32768,
                "all_runs_passed": True,
                "repeatability_passed": True,
                "finalization_reserve_passed": True,
                "conservative_tokens_per_second": compute,
                "conservative_compute_tokens_per_second": compute,
                "token_budgets": {"1_hour": 1, "24_hours": 1, "7_days": 1_000_000_000},
                "projected_overhead_seconds": {
                    "scheduled_log_each": 0.1,
                    "validation_each": 1.0,
                    "recovery_each": 1.0,
                    "milestone_each": 1.0,
                    "final_once": 1.0,
                },
            }
        )
    projection = _online_adjusted_projection(
        {"candidates": candidates},
        {
            "scheduled_log_pause_seconds": [2.0, 3.0],
            "validation_pause_seconds": [4.0],
            "recovery_checkpoint_pause_seconds": [5.0],
            "final_checkpoint_pause_seconds": [6.0],
        },
        config,
    )
    assert projection["online_scheduled_log_max_seconds"] == 3.0
    assert len(projection["candidates"]) == 2
    assert projection["selected"]["candidate_id"] == "deep"
    for candidate, adjusted in zip(candidates, projection["candidates"], strict=True):
        assert adjusted["projected_overhead_seconds"]["scheduled_log_each"] == 3.0
        assert adjusted["projected_overhead_seconds"]["validation_each"] == 4.0
        assert adjusted["projected_overhead_seconds"]["recovery_each"] == 5.0
        assert adjusted["projected_overhead_seconds"]["final_once"] == 6.0
        assert adjusted["token_budgets"]["7_days"] % 32768 == 0
        assert adjusted["token_budgets"]["7_days"] < int(
            candidate["conservative_compute_tokens_per_second"] * 604800
        )


def test_only_loader_supply_threshold_blocks_decomposition():
    passed, bottleneck = _decomposition_conclusion(
        model_ratio=0.8, loader_ratio=1.3, min_loader_ratio=1.2
    )
    assert passed is True
    assert bottleneck == "model forward/backward/optimizer is the nearer measured ceiling"
    failed, reason = _decomposition_conclusion(
        model_ratio=2.0, loader_ratio=1.1, min_loader_ratio=1.2
    )
    assert failed is False
    assert reason == "insufficient loader headroom; long run blocked"


def test_unstable_decomposition_cannot_pass_or_name_a_bottleneck():
    passed, bottleneck = _decomposition_decision(
        model_ratio=1.5,
        loader_ratio=1.5,
        min_loader_ratio=1.2,
        repeatability_passed=False,
    )
    assert passed is False
    assert bottleneck is None


def test_schema_v1_measurement_is_rejected_without_fallback(tmp_path):
    config = OmegaConf.to_container(compose_dgx(), resolve=True)
    plan = _write_plan(tmp_path, config)
    entry = build_matrix_plan(config)[0]
    run_dir = _fake_run(tmp_path, entry, plan, 18_000)
    measurement_path = run_dir / "measurement.json"
    measurement = json.loads(measurement_path.read_text())
    measurement["schema_version"] = 1
    _write_json(measurement_path, measurement)
    with pytest.raises(ValueError, match="only the exact trainer measurement schema v3"):
        summarize_run(run_dir, config["gates"], expected=entry, plan=plan)


@pytest.mark.parametrize(
    "cuda_timings",
    [
        {},
        {"forward": 1.0, "backward": 2.0},
        {"forward": 1.0, "backward": float("nan"), "optimizer": 1.0},
        {"forward": 1.0, "backward": 2.0, "optimizer": -1.0},
    ],
)
def test_measured_rows_require_complete_finite_cuda_timings(tmp_path, cuda_timings):
    config = OmegaConf.to_container(compose_dgx(), resolve=True)
    plan = _write_plan(tmp_path, config)
    entry = build_matrix_plan(config)[0]
    run_dir = _fake_run(tmp_path, entry, plan, 18_000)
    measurement_path = run_dir / "measurement.json"
    measurement = json.loads(measurement_path.read_text())
    measured = next(
        row
        for row in measurement["segments"][0]["rows"]
        if row.get("event") == "optimizer_step" and row.get("warmup") is False
    )
    measured["cuda_milliseconds"] = cuda_timings
    _write_json(measurement_path, measurement)
    with pytest.raises(ValueError, match="CUDA"):
        summarize_run(run_dir, config["gates"], expected=entry, plan=plan)


def test_final_checkpoint_physical_binding_is_a_hard_gate(tmp_path):
    config = OmegaConf.to_container(compose_dgx(), resolve=True)
    plan = _write_plan(tmp_path, config)
    entry = build_matrix_plan(config)[0]
    run_dir = _fake_run(tmp_path, entry, plan, 18_000)
    (run_dir / "checkpoints/final.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="physical final checkpoint"):
        summarize_run(run_dir, config["gates"], expected=entry, plan=plan)


def test_mislabeled_matrix_shape_is_rejected(tmp_path):
    config = OmegaConf.to_container(compose_dgx(), resolve=True)
    plan = _write_plan(tmp_path, config)
    entry = build_matrix_plan(config)[0]
    run_dir = _fake_run(tmp_path, entry, plan, 18_000)
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text())
    run["num_layers"] += 1
    _write_json(run_path, run)
    with pytest.raises(ValueError, match="differs from immutable plan"):
        summarize_run(run_dir, config["gates"], expected=entry, plan=plan)


def test_run_cannot_self_report_a_weaker_storage_watchdog(tmp_path):
    config = OmegaConf.to_container(compose_dgx(), resolve=True)
    plan = _write_plan(tmp_path, config)
    entry = build_matrix_plan(config)[0]
    run_dir = _fake_run(tmp_path, entry, plan, 18_000)
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text())
    run["storage_safety"]["effective_min_free_disk_bytes"] = 1
    _write_json(run_path, run)
    with pytest.raises(ValueError, match="storage-safety budget"):
        summarize_run(run_dir, config["gates"], expected=entry, plan=plan)


def test_observed_checkpoint_cannot_exceed_atomic_write_budget(tmp_path):
    config = OmegaConf.to_container(compose_dgx(), resolve=True)
    plan = _write_plan(tmp_path, config)
    entry = build_matrix_plan(config)[0]
    run_dir = _fake_run(tmp_path, entry, plan, 18_000)
    measurement_path = run_dir / "measurement.json"
    measurement = json.loads(measurement_path.read_text())
    authority = next(
        item
        for item in plan["run_config_authorities"]
        if item["authority_key"] == f"matrix:{entry['candidate_id']}:r{entry['repetition']}"
    )
    checkpoint = next(
        row for row in measurement["segments"][0]["rows"] if row.get("event") == "checkpoint"
    )
    checkpoint["checkpoint/size_bytes"] = authority["max_in_flight_atomic_write_bytes"] + 1
    _write_json(measurement_path, measurement)
    with pytest.raises(ValueError, match="atomic-write safety budget"):
        summarize_run(run_dir, config["gates"], expected=entry, plan=plan)


@pytest.mark.parametrize(
    ("config_path", "value"),
    [
        ("model.dropout", 0.9),
        ("training.optimizer.lr", 0.9),
        ("training.scheduler.warmup_steps", 999),
        ("data.streaming.train.sources.0.ratio", 0.9),
        ("reproducibility.seed", 999),
    ],
)
def test_self_consistent_scientific_config_mutation_is_rejected(tmp_path, config_path, value):
    config = OmegaConf.to_container(compose_dgx(), resolve=True)
    plan = _write_plan(tmp_path, config)
    entry = build_matrix_plan(config)[0]
    run_dir = _fake_run(tmp_path, entry, plan, 18_000)
    resolved_path = run_dir / "resolved_config.yaml"
    resolved = OmegaConf.load(resolved_path)
    OmegaConf.update(resolved, config_path, value)
    OmegaConf.save(resolved, resolved_path, resolve=True)
    plain = OmegaConf.to_container(resolved, resolve=True)

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["sha256"] = _sha256(resolved_path)
    manifest["experiment_identity"]["config_sha256"] = experiment_config_sha256(plain)
    _write_json(manifest_path, manifest)

    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text())
    run["resolved_config_sha256"] = _sha256(resolved_path)
    run["run_manifest_sha256"] = _sha256(manifest_path)
    _write_json(run_path, run)

    with pytest.raises(ValueError, match="full-config authority"):
        summarize_run(run_dir, config["gates"], expected=entry, plan=plan)


def test_swap_growth_and_telemetry_gap_are_hard_gates(tmp_path):
    config = OmegaConf.to_container(compose_dgx(), resolve=True)
    plan = _write_plan(tmp_path, config)
    entry = build_matrix_plan(config)[0]
    swap = _fake_run(tmp_path, entry, plan, 18_000, swap_out=1)
    result = summarize_run(swap, config["gates"], expected=entry, plan=plan)
    assert result["gates"]["swap_out"] is False
    assert result["passed"] is False

    gap_root = tmp_path / "gap"
    gap_plan = _write_plan(gap_root, config)
    gap_run = _fake_run(gap_root, entry, gap_plan, 18_000, telemetry_gap=4.0)
    gap_result = summarize_run(gap_run, config["gates"], expected=entry, plan=gap_plan)
    assert gap_result["gates"]["telemetry_temporal_coverage"] is False


def test_completion_based_telemetry_coverage_accounts_collection_overhead():
    config = OmegaConf.to_container(compose_dgx(), resolve=True)
    template = _telemetry()[0]
    telemetry = []
    for index in range(50):
        row = json.loads(json.dumps(template))
        row["monotonic_seconds"] = 0.2 + index * 1.2
        row["collection_duration_seconds"] = 0.2
        telemetry.append(row)
    summary, gates = _telemetry_summary(
        telemetry,
        {
            "telemetry_interval_seconds": 1.0,
            "telemetry_started_monotonic_seconds": 0.0,
            "telemetry_ended_monotonic_seconds": 60.0,
            "telemetry_errors": [],
            "telemetry_violations": [],
        },
        config["gates"],
    )
    assert summary["effective_cadence_seconds"] == pytest.approx(1.2)
    assert summary["coverage"] == pytest.approx(50 / (61 / 1.2))
    assert gates["telemetry_temporal_coverage"] is True


def test_telemetry_coverage_rejects_missing_collection_duration():
    config = OmegaConf.to_container(compose_dgx(), resolve=True)
    telemetry = _telemetry()
    del telemetry[0]["collection_duration_seconds"]
    with pytest.raises(ValueError, match="collection duration"):
        _telemetry_summary(
            telemetry,
            {
                "telemetry_interval_seconds": 1.0,
                "telemetry_started_monotonic_seconds": 0.0,
                "telemetry_ended_monotonic_seconds": 2.0,
                "telemetry_errors": [],
                "telemetry_violations": [],
            },
            config["gates"],
        )


@pytest.mark.parametrize(
    ("field", "gate"),
    [
        ("sm_clock_mhz", "gpu_clock_observed"),
        ("power_watts", "gpu_power_observed"),
        ("gpu_utilization_percent", "gpu_utilization_observed"),
    ],
)
def test_required_gpu_telemetry_cannot_be_missing_or_all_none(tmp_path, field, gate):
    config = OmegaConf.to_container(compose_dgx(), resolve=True)
    for missing in (False, True):
        root = tmp_path / ("missing" if missing else "none") / field
        plan = _write_plan(root, config)
        entry = build_matrix_plan(config)[0]
        run_dir = _fake_run(root, entry, plan, 18_000)
        telemetry_path = run_dir / "system.jsonl"
        telemetry = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
        for row in telemetry:
            if missing:
                row["gpu"].pop(field)
            else:
                row["gpu"][field] = None
        telemetry_path.write_text(
            "".join(json.dumps(row) + "\n" for row in telemetry), encoding="utf-8"
        )
        result = summarize_run(run_dir, config["gates"], expected=entry, plan=plan)
        assert result["gates"][gate] is False
        assert result["passed"] is False


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_required_gpu_telemetry_rejects_nonfinite_or_negative_values(tmp_path, value):
    config = OmegaConf.to_container(compose_dgx(), resolve=True)
    plan = _write_plan(tmp_path, config)
    entry = build_matrix_plan(config)[0]
    run_dir = _fake_run(tmp_path, entry, plan, 18_000)
    telemetry_path = run_dir / "system.jsonl"
    telemetry = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    telemetry[0]["gpu"]["power_watts"] = value
    telemetry_path.write_text(
        "".join(json.dumps(row) + "\n" for row in telemetry), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="power_watts"):
        summarize_run(run_dir, config["gates"], expected=entry, plan=plan)


def test_periodic_allocator_peaks_do_not_look_like_monotonic_growth(tmp_path):
    config = OmegaConf.to_container(compose_dgx(), resolve=True)
    plan = _write_plan(tmp_path, config)
    entry = build_matrix_plan(config)[0]
    run_dir = _fake_run(tmp_path, entry, plan, 18_000)
    measurement_path = run_dir / "measurement.json"
    measurement = json.loads(measurement_path.read_text())
    measured = [
        row
        for row in measurement["segments"][0]["rows"]
        if row.get("event") == "optimizer_step" and row.get("warmup") is False
    ]
    low = 10_000_000_000
    high = low + 1_000_000_000
    for index, row in enumerate(measured):
        row["pytorch_allocated_bytes"] = high if index % 2 else low
    _write_json(measurement_path, measurement)
    result = summarize_run(run_dir, config["gates"], expected=entry, plan=plan)
    assert result["pytorch_allocator_baseline_growth_bytes"] == 0
    assert result["gates"]["allocator_stable"] is True

    for row in measured[-5:]:
        row["pytorch_allocated_bytes"] = low + 500_000_000
    _write_json(measurement_path, measurement)
    result = summarize_run(run_dir, config["gates"], expected=entry, plan=plan)
    assert result["gates"]["allocator_stable"] is False


def test_selection_requires_all_runs_repeatability_and_exact_rule():
    base = {
        "all_runs_passed": True,
        "repeatability_passed": True,
        "finalization_reserve_passed": True,
        "token_budgets": {"7_days": 1_100_000_000},
        "conservative_tokens_per_second": 100.0,
        "num_layers": 18,
        "sequence_length": 1024,
        "candidate_id": "shallow",
    }
    deepest = {
        **base,
        "candidate_id": "deep",
        "num_layers": 26,
        "conservative_tokens_per_second": 85.0,
    }
    long = {
        **deepest,
        "candidate_id": "deep-long",
        "sequence_length": 2048,
        "conservative_tokens_per_second": 80.0,
    }
    selected, _ = select_candidate(
        [base, deepest, long],
        {
            "target_tokens_7d": 1_000_000_000,
            "max_slowdown_fraction": 0.20,
            "context_throughput_floor_fraction": 0.85,
        },
    )
    assert selected["candidate_id"] == "deep-long"
    with pytest.raises(ValueError, match="no candidate"):
        select_candidate(
            [{**base, "finalization_reserve_passed": False}],
            {
                "target_tokens_7d": 1_000_000_000,
                "max_slowdown_fraction": 0.20,
                "context_throughput_floor_fraction": 0.85,
            },
        )


def test_auxiliary_authority_requires_exact_matrix_plan_protocol_and_selection(tmp_path):
    matrix_root = tmp_path / "matrix"
    matrix_config = OmegaConf.to_container(compose_dgx(["mode=matrix"]), resolve=True)
    matrix_plan = _write_plan(matrix_root, matrix_config)
    selected = {
        key: value
        for key, value in build_matrix_plan(matrix_config)[0].items()
        if key != "repetition"
    }
    summary_path = matrix_root / "dgx-summary.json"
    summary = {
        "schema_version": 3,
        "verdict": "PASS",
        "plan_id": matrix_plan["plan_id"],
        "git_commit": COMMIT,
        "image_id": IMAGE,
        "selection_rule": matrix_config["selection"],
        "selected": {
            **selected,
            "conservative_tokens_per_second": 9.0,
            "conservative_compute_tokens_per_second": 10.0,
        },
    }

    def write_summary() -> None:
        _write_json(summary_path, summary)

    def authority() -> dict:
        protocol = {key: matrix_plan["config"][key] for key in PROTOCOL_CONFIG_KEYS}
        return {
            "path": str(summary_path),
            "sha256": _sha256(summary_path),
            "matrix_plan_path": str(matrix_root / "plan.json"),
            "matrix_plan_sha256": _sha256(matrix_root / "plan.json"),
            "matrix_plan_id": matrix_plan["plan_id"],
            "matrix_protocol_sha256": _canonical_sha256(protocol),
            "git_commit": COMMIT,
            "image_id": IMAGE,
            "selection_rule": matrix_config["selection"],
            "selected": selected["candidate_id"],
            "end_to_end_compute_tokens_per_second": 10.0,
        }

    write_summary()
    auxiliary_config = json.loads(json.dumps(matrix_config))
    auxiliary_config["mode"] = "decompose"
    auxiliary_plan = {
        "config": auxiliary_config,
        "git_commit": COMMIT,
        "image_id": IMAGE,
        "matrix_summary_identity": authority(),
    }
    assert _validate_matrix_authority(auxiliary_plan)["plan_id"] == matrix_plan["plan_id"]

    summary["plan_id"] = "f" * 64
    write_summary()
    auxiliary_plan["matrix_summary_identity"] = authority()
    with pytest.raises(ValueError, match="one passing exact summary"):
        _validate_matrix_authority(auxiliary_plan)

    summary["plan_id"] = matrix_plan["plan_id"]
    summary["selection_rule"] = {
        **matrix_config["selection"],
        "max_slowdown_fraction": 0.99,
    }
    write_summary()
    auxiliary_plan["matrix_summary_identity"] = authority()
    with pytest.raises(ValueError, match="one passing exact summary"):
        _validate_matrix_authority(auxiliary_plan)


def test_wandb_evidence_requires_successful_scalar_runtime_and_final_summaries(tmp_path):
    evidence_path = tmp_path / "checkpoints" / "wandb_events.jsonl"
    evidence_path.parent.mkdir(parents=True)
    rows = [
        {
            "schema_version": 1,
            "action": "init",
            "outcome": "succeeded",
            "mode": "online",
            "run_id": "run-id",
            "run_url": "https://wandb.example/runs/run-id",
        },
        {"schema_version": 1, "action": "watch", "outcome": "disabled"},
        {"schema_version": 1, "action": "finish", "outcome": "succeeded"},
    ]

    def record() -> dict:
        evidence_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return {
            "wandb_evidence": {
                "path": "checkpoints/wandb_events.jsonl",
                "sha256": _sha256(evidence_path),
                "rows": len(rows),
            }
        }

    evidence = _validated_wandb_evidence(tmp_path, record())
    assert evidence["critical_failures"] == []
    assert evidence["successful_scalar_logs"] == 0
    assert evidence["runtime_summary_succeeded"] is False
    assert evidence["final_summary_succeeded"] is False
    rows.extend(
        [
            {
                "schema_version": 1,
                "action": "log",
                "outcome": "succeeded",
                "optimizer_step": 25,
            },
            {"schema_version": 1, "action": "runtime_summary", "outcome": "succeeded"},
            {"schema_version": 1, "action": "final_summary", "outcome": "succeeded"},
        ]
    )
    evidence = _validated_wandb_evidence(tmp_path, record())
    assert evidence["successful_scalar_logs"] == 1
    assert evidence["runtime_summary_succeeded"] is True
    assert evidence["final_summary_succeeded"] is True
    rows.append(
        {
            "schema_version": 1,
            "action": "log",
            "outcome": "failed",
            "error": {"type": "TimeoutError", "message": "bounded timeout"},
        }
    )
    rows.append(
        {
            "schema_version": 1,
            "action": "summary",
            "outcome": "failed",
            "error": {"type": "RuntimeError", "message": "unavailable"},
        }
    )
    rows.append(
        {
            "schema_version": 1,
            "action": "final_summary",
            "outcome": "failed",
            "error": {"type": "RuntimeError", "message": "unavailable"},
        }
    )
    evidence = _validated_wandb_evidence(tmp_path, record())
    assert evidence["critical_failures"] == ["final_summary", "log", "summary"]
