import json
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from dgx.planning import build_matrix_plan, summarize_evidence, summarize_run
from runtime.config import validate_training_config


ROOT = Path(__file__).resolve().parent.parent


def compose_train(profile: str):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        return hydra.compose(config_name="train", overrides=[f"profile={profile}"])


def compose_dgx():
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        return hydra.compose(config_name="dgx")


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
    assert baseline.measurement.enabled is False
    assert baseline.training.max_time == 3600
    assert baseline.training.validation_every_n_tokens == 5_000_000
    assert baseline.training.checkpoint_every_n_tokens == 2_500_000
    assert baseline.training.milestone_every_n_tokens == 100_000_000
    assert baseline.wandb.mode == "online"


def test_matrix_crosses_depth_and_context_with_equal_work_and_rotated_repeats():
    plan = build_matrix_plan(compose_dgx())
    assert len(plan) == 27
    assert len({entry["candidate_id"] for entry in plan}) == 9
    assert {entry["effective_target_tokens_per_step"] for entry in plan} == {32768}
    assert {entry["repetition"] for entry in plan} == {1, 2, 3}
    first_ids = [
        next(entry["candidate_id"] for entry in plan if entry["repetition"] == repetition)
        for repetition in (1, 2, 3)
    ]
    assert len(set(first_ids)) == 3


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _fake_run(root: Path, entry: dict, throughput: float, *, swap_out: int = 0) -> Path:
    run_dir = root / f"{entry['candidate_id']}-r{entry['repetition']}"
    rows = []
    for step in range(1, entry["max_steps"] + 1):
        wall = entry["effective_target_tokens_per_step"] / throughput
        rows.append(
            {
                "event": "optimizer_step",
                "optimizer_step": step,
                "target_tokens": step * entry["effective_target_tokens_per_step"],
                "target_tokens_step": entry["effective_target_tokens_per_step"],
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
    rows.append(
        {
            "event": "validation",
            "validation_event_seconds": 3.0,
        }
    )
    rows.append(
        {
            "event": "final_checkpoint",
            "checkpoint_seconds": 2.0,
            "checkpoint/size_bytes": 1_000_000_000,
        }
    )
    _write_json(
        run_dir / "measurement.json",
        {
            "schema_version": 1,
            "complete": True,
            "device": "cuda",
            "cuda_events": True,
            "rows": rows,
        },
    )
    metrics = [
        {
            "event": "step",
            "train/loss_step": 10.0,
            "optimizer/gradient_norm": 1.0,
            "optimizer/lr": 0.0003,
        }
    ]
    metrics_path = run_dir / "checkpoints" / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("".join(json.dumps(row) + "\n" for row in metrics))
    telemetry = []
    for index in range(3):
        telemetry.append(
            {
                "monotonic_seconds": float(index),
                "host": {
                    "memory_available_bytes": 100_000_000_000,
                    "disk_free_bytes": 200_000_000_000,
                    "swap_in_pages": 0,
                    "swap_out_pages": swap_out if index == 2 else 0,
                    "process_rss_bytes": 2_000_000_000,
                },
                "gpu": {"temperature_c": 65.0},
            }
        )
    system_path = run_dir / "system.jsonl"
    system_path.write_text("".join(json.dumps(row) + "\n" for row in telemetry))
    _write_json(
        run_dir / "run.json",
        {
            "status": "succeeded",
            "candidate_id": entry["candidate_id"],
            "repetition": entry["repetition"],
            "git_commit": "a" * 40,
            "image_id": "sha256:" + "b" * 64,
            "parameter_count": 100_000_000 + entry["num_layers"],
            "num_layers": entry["num_layers"],
            "embed_size": entry["embed_size"],
            "num_heads": entry["num_heads"],
            "sequence_length": entry["sequence_length"],
            "batch_size": entry["batch_size"],
            "gradient_accumulation_steps": entry["gradient_accumulation_steps"],
            "effective_target_tokens_per_step": entry["effective_target_tokens_per_step"],
            "measured_optimizer_steps": 20,
            "checkpoint_verified": True,
            "telemetry_interval_seconds": 1.0,
            "telemetry_started_monotonic_seconds": 0.0,
            "telemetry_ended_monotonic_seconds": 2.0,
            "telemetry_errors": [],
        },
    )
    return run_dir


def _throughput(entry: dict) -> float:
    values = {
        "p70-ctx1024": 5000,
        "p70-ctx2048": 4600,
        "p70-ctx4096": 4200,
        "p85-ctx1024": 4300,
        "p85-ctx2048": 4100,
        "p85-ctx4096": 3500,
        "p99-ctx1024": 1600,
        "p99-ctx2048": 1450,
        "p99-ctx4096": 1200,
    }
    return values[entry["candidate_id"]] * (1.0 - 0.01 * (entry["repetition"] - 1))


def test_summary_applies_triplicate_gates_and_selects_committed_baseline_shape(tmp_path):
    config = compose_dgx()
    for entry in build_matrix_plan(config):
        _fake_run(tmp_path, entry, _throughput(entry))
    summary = summarize_evidence(tmp_path, config)
    assert summary["verdict"] == "PASS"
    assert summary["selected"]["candidate_id"] == "p85-ctx2048"
    assert summary["selected"]["num_layers"] == 26
    assert summary["selected"]["sequence_length"] == 2048
    assert summary["selected"]["batch_size"] == 4
    assert summary["plan"]["token_budgets"]["1_hour"] > 0
    assert summary["plan"]["token_budgets"]["24_hours"] > 0
    assert summary["plan"]["token_budgets"]["7_days"] >= 1_000_000_000
    assert summary["named_bottleneck"] == "backward"

    baseline = compose_train("pretrain_baseline")
    assert baseline.model.num_layers == summary["selected"]["num_layers"]
    assert baseline.training.sequence_length == summary["selected"]["sequence_length"]
    assert baseline.training.batch_size == summary["selected"]["batch_size"]


def test_swap_growth_is_a_hard_run_gate(tmp_path):
    config = OmegaConf.to_container(compose_dgx(), resolve=True)
    entry = build_matrix_plan(config)[0]
    run_dir = _fake_run(tmp_path, entry, 4000, swap_out=1)
    result = summarize_run(run_dir, config["gates"])
    assert result["gates"]["swap_out"] is False
    assert result["passed"] is False
