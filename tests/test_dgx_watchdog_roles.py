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
        "--expected-architecture",
        "aarch64",
        "--expected-gpu-name",
        "NVIDIA GB10",
        "--expected-device-count",
        "1",
        "--expected-compute-capability-major",
        "12",
        "--expected-compute-capability-minor",
        "1",
        "--min-unified-memory-bytes",
        "120000000000",
        "--max-unified-memory-bytes",
        "140000000000",
        "--require-equal-host-device-memory",
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
    monkeypatch.setattr(MEASURE, "validate_dgx_spark_environment", lambda *_args: {"target": True})
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
    monkeypatch.setattr(
        DECOMPOSE, "validate_dgx_spark_environment", lambda *_args: {"target": True}
    )

    assert DECOMPOSE.main(_common_args(output, role)) == 1
    record = json.loads((output / "decomposition.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["telemetry_violations"] == ["free disk fell below the hard floor"]
    assert record["storage_safety"]["effective_min_free_disk_bytes"] == 120_000_000_000
    assert InterruptingSampler.instances[-1].kwargs["hard_limits"]["min_free_disk_bytes"] == (
        120_000_000_000
    )


def test_pilot_telemetry_covers_checkpoint_verification_and_sampling(monkeypatch, tmp_path):
    output = tmp_path / "pilot-ordering"
    events = []
    config = OmegaConf.create(
        {
            "profile": {"name": "pretrain_baseline"},
            "runtime": {"device": "cuda"},
            "reproducibility": {"deterministic": True},
            "training": {
                "precision": "bf16",
                "max_steps": None,
                "max_time": 1680,
                "sequence_length": 1024,
                "batch_size": 8,
                "gradient_accumulation_steps": 4,
                "log_every_n_steps": 25,
            },
            "model": {"num_layers": 18, "embed_size": 384, "num_heads": 6},
            "measurement": {"output_path": str(output / "measurement.json")},
            "artifacts": {"checkpoints_dir": "checkpoints"},
            "wandb": {
                "mode": "online",
                "watch": {"enabled": False},
                "artifact": {"policy": "none"},
            },
        }
    )

    class OrderingSampler:
        def __init__(self, _path, **_kwargs):
            self._thread = None
            self.samples = 2
            self.errors = []
            self.violations = []
            self.active = False

        def start(self):
            self.active = True
            events.append("telemetry_start")

        def stop(self):
            assert self.active
            events.append("telemetry_stop")
            self.active = False

    class Trainer:
        model = object()
        optimizer_step = 30
        target_tokens = 983_040
        elapsed_seconds = 1680.0

        def fit(self):
            events.append("fit")
            return []

    def prepare(_cfg, run_dir):
        (run_dir / "resolved_config.yaml").write_text("profile: fixture\n", encoding="utf-8")
        (run_dir / "run_manifest.json").write_text("{}\n", encoding="utf-8")
        checkpoints = run_dir / "checkpoints"
        checkpoints.mkdir()
        (checkpoints / "final.pt").write_bytes(b"checkpoint")
        return Trainer()

    loaded = type(
        "Loaded",
        (),
        {
            "physical_identity": {"size_bytes": 10, "sha256": "fixture"},
            "payload": {
                "kind": "final",
                "identity": {"experiment_id": "fixture"},
                "state": {"measurement_evidence": {"checkpoint_boundary": {"kind": "final"}}},
            },
        },
    )()

    def verify(_path):
        events.append("checkpoint_verify")
        return loaded

    def sample(_path):
        events.append("sample")
        return []

    def wandb(_path):
        events.append("wandb_evidence")
        return {"finish_succeeded": True}

    monkeypatch.setattr(MEASURE, "TelemetrySampler", OrderingSampler)
    monkeypatch.setattr(MEASURE, "_preflight", lambda *_args: {"safe": True})
    monkeypatch.setattr(MEASURE, "_environment", lambda: {"cuda": True})
    monkeypatch.setattr(MEASURE, "validate_dgx_spark_environment", lambda *_args: {"target": True})
    monkeypatch.setattr(MEASURE, "_compose", lambda _overrides: config)
    monkeypatch.setattr(MEASURE, "prepare_trainer", prepare)
    monkeypatch.setattr(MEASURE, "count_parameters", lambda _model: 70_828_682)
    monkeypatch.setattr(MEASURE, "load_checkpoint_for_generation", verify)
    monkeypatch.setattr(MEASURE, "_samples", sample)
    monkeypatch.setattr(MEASURE, "_wandb_evidence", wandb)

    argv = [*_common_args(output, "pilot"), "--pilot", "--sample"]
    assert MEASURE.main(argv) == 0
    assert events == [
        "telemetry_start",
        "fit",
        "checkpoint_verify",
        "sample",
        "wandb_evidence",
        "telemetry_stop",
    ]
