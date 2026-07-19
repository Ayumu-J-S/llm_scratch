import importlib.util
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MEASURE = _load_script("measure_dgx")
DECOMPOSE = _load_script("measure_dgx_decomposition")


class InterruptingSampler:
    instances = []

    def __init__(self, _path, **kwargs):
        self.kwargs = kwargs
        self.samples = 1
        self.errors = []
        self.violations = ["free disk fell below the hard floor"]
        self._thread = None
        self.__class__.instances.append(self)

    def start(self):
        if not self.kwargs["interrupt_on_violation"]:
            raise AssertionError("DGX role did not arm fail-closed interruption")
        raise KeyboardInterrupt("simulated in-flight hard resource violation")

    def stop(self):
        return None


def _common_args(output: Path, role: str) -> list[str]:
    return [
        "--output-dir",
        str(output),
        "--role",
        role,
        "--candidate-id",
        "p70-ctx1024",
        "--repetition",
        "1",
        "--git-commit",
        "a" * 40,
        "--image-id",
        "sha256:" + "b" * 64,
        "--plan-id",
        "plan",
        "--warmup-optimizer-steps",
        "10",
        "--measured-optimizer-steps",
        "20",
        "--min-available-memory-bytes",
        "64000000000",
        "--min-free-disk-bytes",
        "120000000000",
        "--post-plan-free-reserve-bytes",
        "100000000000",
        "--max-in-flight-atomic-write-bytes",
        "17000000000",
        "--max-temperature-c",
        "80",
        "--max-swap-in-pages",
        "0",
        "--max-swap-out-pages",
        "0",
    ]


@pytest.mark.parametrize("role", ["matrix", "pilot"])
def test_training_roles_fail_closed_and_preserve_low_disk_evidence(monkeypatch, tmp_path, role):
    output = tmp_path / role
    max_steps = 30 if role == "matrix" else None
    profile = "dgx_candidate" if role == "matrix" else "pretrain_baseline"
    config = OmegaConf.create(
        {
            "profile": {"name": profile},
            "runtime": {"device": "cuda"},
            "reproducibility": {"deterministic": True},
            "training": {
                "precision": "bf16",
                "max_steps": max_steps,
                "max_time": None if role == "matrix" else 1680,
                "sequence_length": 1024,
                "batch_size": 8,
                "gradient_accumulation_steps": 4,
                "log_every_n_steps": 25,
            },
            "model": {"num_layers": 18, "embed_size": 384, "num_heads": 6},
            "measurement": {"output_path": str(output / "measurement.json")},
            "artifacts": {"checkpoints_dir": "checkpoints"},
            "wandb": {
                "mode": "online" if role == "pilot" else "disabled",
                "watch": {"enabled": False},
                "artifact": {"policy": "none"},
            },
        }
    )

    class Trainer:
        model = object()

        def fit(self):
            raise AssertionError("workload continued after hard resource violation")

    def prepare(_cfg, run_dir):
        (run_dir / "resolved_config.yaml").write_text("profile: fixture\n", encoding="utf-8")
        (run_dir / "run_manifest.json").write_text("{}\n", encoding="utf-8")
        return Trainer()

    InterruptingSampler.instances.clear()
    monkeypatch.setattr(MEASURE, "TelemetrySampler", InterruptingSampler)
    monkeypatch.setattr(MEASURE, "_preflight", lambda *_args: {"safe": True})
    monkeypatch.setattr(MEASURE, "_environment", lambda: {"cuda": True})
    monkeypatch.setattr(MEASURE, "_compose", lambda _overrides: config)
    monkeypatch.setattr(MEASURE, "prepare_trainer", prepare)
    monkeypatch.setattr(MEASURE, "count_parameters", lambda _model: 70_828_682)
    argv = _common_args(output, role)
    if role == "pilot":
        argv.extend(["--pilot", "--sample"])

    assert MEASURE.main(argv) == 1
    record = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["telemetry_violations"] == ["free disk fell below the hard floor"]
    assert record["storage_safety"]["effective_min_free_disk_bytes"] == 120_000_000_000
    assert InterruptingSampler.instances[-1].kwargs["hard_limits"]["min_free_disk_bytes"] == (
        120_000_000_000
    )


@pytest.mark.parametrize("role", ["model-only", "loader-only"])
def test_decomposition_roles_fail_closed_and_preserve_low_disk_evidence(
    monkeypatch, tmp_path, role
):
    output = tmp_path / role
    config = OmegaConf.create(
        {
            "profile": {"name": "pretrain_baseline"},
            "wandb": {"mode": "disabled"},
            "measurement": {
                "enabled": role == "model-only",
                "cuda_events": role == "model-only",
            },
            "model": {"num_layers": 18, "embed_size": 384, "num_heads": 6},
            "training": {
                "sequence_length": 1024,
                "batch_size": 8,
                "gradient_accumulation_steps": 4,
            },
        }
    )

    InterruptingSampler.instances.clear()
    monkeypatch.setattr(DECOMPOSE, "TelemetrySampler", InterruptingSampler)
    monkeypatch.setattr(DECOMPOSE.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(DECOMPOSE.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(DECOMPOSE, "_compose", lambda _overrides: config)
    monkeypatch.setattr(DECOMPOSE, "_environment", lambda: {"cuda": True})
    monkeypatch.setattr(DECOMPOSE, "_preflight", lambda *_args: {"safe": True})

    assert DECOMPOSE.main(_common_args(output, role)) == 1
    record = json.loads((output / "decomposition.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["telemetry_violations"] == ["free disk fell below the hard floor"]
    assert record["storage_safety"]["effective_min_free_disk_bytes"] == 120_000_000_000
    assert InterruptingSampler.instances[-1].kwargs["hard_limits"]["min_free_disk_bytes"] == (
        120_000_000_000
    )
