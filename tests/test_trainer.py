from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import training.trainer as trainer_module
import training.optimization as optimization_module
from training.optimization import WarmupCosineScheduler
from training.trainer import Trainer


class FixedLogitModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 4) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.arange(vocab_size, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(*inputs.shape, -1)


class ListLoader:
    def __init__(self, batches: list[dict[str, torch.Tensor]]) -> None:
        self.batches = batches

    def __iter__(self):
        yield from self.batches


class ClosingStreamIterator:
    def __init__(self, batches: list[dict[str, torch.Tensor]]) -> None:
        self.batches = batches
        self.position = 0
        self.dataset_stream = ClosingDatasetStream()
        self._dataset_fetcher = SimpleNamespace(dataset_iter=self.dataset_stream)

    def __iter__(self):
        return self

    def __next__(self):
        if self.position >= len(self.batches):
            raise StopIteration
        batch = self.batches[self.position]
        self.position += 1
        return batch


class ClosingStreamLoader:
    def __init__(self, batches: list[dict[str, torch.Tensor]]) -> None:
        self.batches = batches
        self.iterator: ClosingStreamIterator | None = None

    def __iter__(self):
        self.iterator = ClosingStreamIterator(self.batches)
        return self.iterator


class ClosingDatasetStream:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class NaNGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full_like(grad_output, float("nan"))


class NaNGradientModel(FixedLogitModel):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = NaNGradient.apply(self.logits)
        return logits.expand(*inputs.shape, -1)


class NonFiniteValidationModel(FixedLogitModel):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.logits
        if not self.training:
            logits = logits.masked_fill(torch.ones_like(logits, dtype=torch.bool), float("nan"))
        return logits.expand(*inputs.shape, -1)


def _trainer(
    tmp_path: Path,
    batches,
    *,
    measurement: dict | None = None,
    checkpoint_dir: Path | None = None,
    **training_overrides,
) -> Trainer:
    model = FixedLogitModel()
    cfg = OmegaConf.create(
        {
            "training": {
                "epochs": 1,
                "max_steps": None,
                "max_tokens": None,
                "max_time": None,
                "log_every_n_steps": 1,
                "validation_every_n_steps": None,
                "checkpoint_every_n_steps": None,
                "milestone_every_n_steps": None,
                "scheduler": {"interval": "epoch"},
                **training_overrides,
            },
            "wandb": {"enabled": False},
            "measurement": measurement or {"enabled": False},
        }
    )
    return Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        scheduler=None,
        train_loader=ListLoader(batches),
        validation_loader_factory=lambda: ListLoader(batches),
        checkpoint_dir=checkpoint_dir or tmp_path,
        cfg=cfg,
        device=torch.device("cpu"),
    )


def _batch(labels: list[list[int]]) -> dict[str, torch.Tensor]:
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return {"inputs": torch.zeros_like(labels_tensor), "labels": labels_tensor}


def _recipe_trainer(
    tmp_path: Path,
    *,
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler=None,
    **training_overrides,
) -> Trainer:
    cfg = OmegaConf.create(
        {
            "training": {
                "epochs": 1,
                "batch_size": 1,
                "sequence_length": 2,
                "max_steps": 1,
                "max_tokens": None,
                "max_time": None,
                "precision": "fp32",
                "gradient_accumulation_steps": 1,
                "max_grad_norm": None,
                "log_every_n_steps": 1,
                "validation_every_n_steps": None,
                "checkpoint_every_n_steps": None,
                "milestone_every_n_steps": None,
                "scheduler": {"interval": "step"},
                **training_overrides,
            },
            "wandb": {"enabled": False},
        }
    )
    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=ListLoader(batches),
        validation_loader_factory=lambda: ListLoader(batches),
        checkpoint_dir=tmp_path,
        cfg=cfg,
        device=torch.device("cpu"),
    )


def test_partial_batch_metric_is_token_weighted(tmp_path: Path):
    batch = _batch([[0, 1, -100], [2, -100, 3]])
    trainer = _trainer(tmp_path, [batch])
    trainer.fit()

    expected = torch.nn.functional.cross_entropy(
        trainer.model.logits.detach().expand(4, -1),
        torch.tensor([0, 1, 2, 3]),
    ).item()
    step_record = next(item for item in trainer.metrics if "train/loss_step" in item)
    assert step_record["train/loss_step"] == pytest.approx(expected)
    assert step_record["train/target_tokens_step"] == 4
    assert trainer.target_tokens == 4
    assert "timing" not in step_record
    assert not (tmp_path / "measurement.json").exists()


def test_benchmark_measurement_is_explicit_and_flushed_once(tmp_path: Path):
    trainer = _trainer(
        tmp_path,
        [_batch([[0, 1]]) for _ in range(2)],
        max_steps=2,
        measurement={
            "enabled": True,
            "warmup_optimizer_steps": 1,
            "cuda_events": True,
            "output_path": str(tmp_path / "timing.json"),
        },
        checkpoint_dir=tmp_path / "checkpoints",
    )

    trainer.fit()

    payload = json.loads((tmp_path / "timing.json").read_text(encoding="utf-8"))
    assert payload["complete"] is True
    assert payload["cuda_events"] is False
    steps = [row for row in payload["rows"] if row["event"] == "optimizer_step"]
    assert [row["optimizer_step"] for row in steps] == [1, 2]
    assert [row["warmup"] for row in steps] == [True, False]
    assert all(row["data_wait_calls"] == 1 for row in steps)
    assert all(row["step_wall_seconds"] >= row["host_seconds"]["data_wait"] for row in steps)
    assert all(row["cuda_milliseconds"] == {} for row in steps)
    assert not (tmp_path / "timing.jsonl").exists()


def test_benchmark_measurement_separates_validation_from_optimizer_steps(tmp_path: Path):
    trainer = _trainer(
        tmp_path,
        [_batch([[0, 1]]) for _ in range(2)],
        max_steps=2,
        validation_every_n_steps=1,
        measurement={
            "enabled": True,
            "warmup_optimizer_steps": 0,
            "cuda_events": False,
            "output_path": str(tmp_path / "timing.json"),
        },
        checkpoint_dir=tmp_path / "checkpoints",
    )

    trainer.fit()

    rows = json.loads((tmp_path / "timing.json").read_text(encoding="utf-8"))["rows"]
    steps = [row for row in rows if row["event"] == "optimizer_step"]
    validations = [row for row in rows if row["event"] == "validation"]
    assert len(steps) == len(validations) == 2
    assert all(
        step["wall_end_unix_ns"] <= validation["wall_start_unix_ns"]
        for step, validation in zip(steps, validations, strict=True)
    )
    assert all(validation["full_event_pause_seconds"] >= 0.0 for validation in validations)
    assert all(
        abs(validation["unattributed_seconds"])
        <= max(0.005, validation["full_event_pause_seconds"] * 0.01)
        for validation in validations
    )


@pytest.mark.parametrize(
    "artifact_name",
    [
        "final.pt",
        "best.pt",
        "recovery-step-000000000001.pt",
        "milestone-step-000000000001.pt",
        ".final.pt.review.tmp",
    ],
)
def test_measurement_output_rejects_every_checkpoint_namespace_before_training(
    tmp_path: Path, artifact_name: str
):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    artifact = checkpoint_dir / artifact_name
    artifact.write_bytes(b"checkpoint sentinel")

    with pytest.raises(ValueError, match="outside the checkpoint directory"):
        _trainer(
            tmp_path,
            [_batch([[0, 1]])],
            measurement={"enabled": True, "output_path": str(artifact)},
            checkpoint_dir=checkpoint_dir,
            max_steps=1,
        )

    assert artifact.read_bytes() == b"checkpoint sentinel"


def test_measurement_output_rejects_external_checkpoint_hardlink_before_training(
    tmp_path: Path,
):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    artifact = checkpoint_dir / "final.pt"
    artifact.write_bytes(b"checkpoint sentinel")
    measurement_path = tmp_path / "timing.json"
    os.link(artifact, measurement_path)

    with pytest.raises(ValueError, match="share an inode"):
        _trainer(
            tmp_path,
            [_batch([[0, 1]])],
            measurement={"enabled": True, "output_path": str(measurement_path)},
            checkpoint_dir=checkpoint_dir,
            max_steps=1,
        )

    assert artifact.read_bytes() == b"checkpoint sentinel"
    assert measurement_path.read_bytes() == b"checkpoint sentinel"


def test_max_steps_boundary_does_not_fetch_or_update_extra_batch(tmp_path: Path):
    batches = [_batch([[0, 1]]) for _ in range(5)]
    trainer = _trainer(tmp_path, batches, max_steps=2)
    trainer.fit()
    assert trainer.optimizer_step == 2
    assert trainer.target_tokens == 4
    assert max(item["optimizer_step"] for item in trainer.metrics) == 2


def test_budget_stop_closes_the_active_stream_generator(tmp_path: Path):
    loader = ClosingStreamLoader([_batch([[0, 1]]) for _ in range(3)])
    trainer = _trainer(tmp_path, [_batch([[0, 1]])], max_steps=1)
    trainer.train_loader = loader

    trainer.fit()

    assert loader.iterator is not None
    assert loader.iterator.dataset_stream.closed is True


def test_max_tokens_stops_at_exact_partial_batch_boundary(tmp_path: Path):
    trainer = _trainer(tmp_path, [_batch([[0, 1, 2], [1, 2, 3]])], max_tokens=5)
    trainer.fit()
    assert trainer.optimizer_step == 1
    assert trainer.target_tokens == 5
    step_record = next(item for item in trainer.metrics if "train/loss_step" in item)
    assert step_record["train/target_tokens_step"] == 5


def test_validation_and_checkpoint_cadences_are_independent(tmp_path: Path, monkeypatch):
    batches = [_batch([[0, 1]]) for _ in range(4)]
    trainer = _trainer(
        tmp_path,
        batches,
        max_steps=4,
        validation_every_n_steps=2,
        checkpoint_every_n_steps=3,
    )
    evaluations: list[int] = []
    original_evaluate = trainer._evaluate

    def evaluate():
        evaluations.append(trainer.optimizer_step)
        return original_evaluate()

    monkeypatch.setattr(trainer, "_evaluate", evaluate)
    trainer.fit()
    assert evaluations == [2, 4]
    assert len(list(tmp_path.glob("recovery-step-*.pt"))) == 1
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "final.pt").is_file()
    assert trainer._last_checkpoint_step == 3
    checkpoint_record = next(item for item in trainer.metrics if item.get("event") == "checkpoint")
    assert checkpoint_record["checkpoint/size_bytes"] > 0
    assert checkpoint_record["checkpoint/write_seconds"] >= 0.0
    assert checkpoint_record["checkpoint/verification_seconds"] >= 0.0
    assert checkpoint_record["checkpoint/pause_seconds"] >= 0.0


def test_token_cadence_records_boundaries_and_local_metrics(tmp_path: Path):
    batches = [_batch([[0, 1]]) for _ in range(3)]
    trainer = _trainer(
        tmp_path,
        batches,
        log_every_n_steps=None,
        log_every_n_tokens=4,
        validation_every_n_steps=None,
        validation_every_n_tokens=4,
        checkpoint_every_n_steps=None,
        checkpoint_every_n_tokens=4,
    )
    trainer.fit()
    assert [item["target_tokens"] for item in trainer.metrics if item.get("event") == "log"] == [4]
    assert [
        item["target_tokens"] for item in trainer.metrics if item.get("event") == "validation"
    ] == [4]
    assert [
        item["target_tokens"] for item in trainer.metrics if item.get("event") == "checkpoint"
    ] == [4]
    assert any(
        item.get("event") == "epoch_summary" and "train/loss" in item for item in trainer.metrics
    )
    assert any(
        item.get("event") == "epoch_summary" and "train/perplexity" in item
        for item in trainer.metrics
    )
    assert (tmp_path / "metrics.jsonl").exists()


def test_milestone_checkpoint_is_saved_once_per_step_across_epoch_end(tmp_path: Path):
    trainer = _trainer(
        tmp_path,
        [_batch([[0, 1]]) for _ in range(2)],
        max_steps=2,
        milestone_every_n_steps=1,
    )

    trainer.fit()

    milestones = [item for item in trainer.metrics if item.get("event") == "milestone"]
    assert [item["optimizer_step"] for item in milestones] == [1, 2]
    assert sorted(path.name for path in tmp_path.glob("milestone-step-*.pt")) == [
        "milestone-step-000000000001.pt",
        "milestone-step-000000000002.pt",
    ]


def test_nonfinite_gradient_records_context_before_counters_advance(tmp_path: Path):
    trainer = _trainer(tmp_path, [_batch([[0, 1]])])
    trainer.model = NaNGradientModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)
    with pytest.raises(FloatingPointError, match="non-finite gradients"):
        trainer.fit()
    assert trainer.optimizer_step == 0
    assert trainer.target_tokens == 0
    failure = next(item for item in trainer.metrics if item.get("event") == "nonfinite_gradients")
    assert failure["batch_index"] == 1


def test_nonfinite_validation_records_context_and_stops(tmp_path: Path):
    trainer = _trainer(tmp_path, [_batch([[0, 1]])])
    trainer.model = NonFiniteValidationModel()
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.0)
    with pytest.raises(FloatingPointError, match="non-finite validation"):
        trainer.fit()
    assert trainer.optimizer_step == 1
    failure = next(item for item in trainer.metrics if item.get("event") == "nonfinite_validation")
    assert failure["batch_index"] == 1
    assert failure.get("preceding_checkpoint") is None


def test_fractional_step_and_token_budgets_are_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="max_steps.*positive integer"):
        _trainer(tmp_path, [_batch([[0, 1]])], max_steps=1.5)
    with pytest.raises(ValueError, match="max_tokens.*positive integer"):
        _trainer(tmp_path, [_batch([[0, 1]])], max_tokens=1.5)


def test_reused_checkpoint_dir_truncates_metrics_per_run(tmp_path: Path):
    first = _trainer(tmp_path, [_batch([[0, 1]]) for _ in range(2)], max_steps=2)
    first.fit()
    first_lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(first_lines) == len(first.metrics)
    assert any(json.loads(line)["optimizer_step"] == 2 for line in first_lines)

    second = _trainer(tmp_path, [_batch([[0, 1]])], max_steps=1)
    second.fit()
    second_lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in second_lines]
    assert len(second_lines) == len(second.metrics)
    assert all(record["optimizer_step"] == 1 for record in records)
    assert len(second_lines) < len(first_lines)


def test_wandb_init_failure_preserves_previous_metrics(tmp_path: Path, monkeypatch):
    metrics_path = tmp_path / "metrics.jsonl"
    previous = '{"optimizer_step": 99, "event": "previous-run"}\n'
    metrics_path.write_text(previous, encoding="utf-8")
    trainer = _trainer(tmp_path, [_batch([[0, 1]])])
    trainer.cfg.wandb.enabled = True

    def fail_wandb_init(**kwargs):
        raise RuntimeError("simulated W&B initialization failure")

    monkeypatch.setattr(trainer_module.wandb, "init", fail_wandb_init)
    with pytest.raises(RuntimeError, match="W&B initialization failure"):
        trainer.fit()
    assert metrics_path.read_text(encoding="utf-8") == previous


def test_accumulation_matches_one_combined_batch_with_dropout_disabled(tmp_path: Path):
    accumulated_model = FixedLogitModel()
    combined_model = FixedLogitModel()
    combined_model.load_state_dict(accumulated_model.state_dict())

    accumulated = _recipe_trainer(
        tmp_path / "accumulated",
        model=accumulated_model,
        batches=[_batch([[0, 1]]), _batch([[2, 3]])],
        optimizer=torch.optim.SGD(accumulated_model.parameters(), lr=0.1),
        gradient_accumulation_steps=2,
    )
    combined = _recipe_trainer(
        tmp_path / "combined",
        model=combined_model,
        batches=[_batch([[0, 1], [2, 3]])],
        optimizer=torch.optim.SGD(combined_model.parameters(), lr=0.1),
        batch_size=2,
        gradient_accumulation_steps=1,
    )

    accumulated.fit()
    combined.fit()

    assert torch.allclose(accumulated_model.logits, combined_model.logits, rtol=0, atol=1e-7)
    record = next(item for item in accumulated.metrics if item.get("event") == "step")
    assert record["train/effective_target_tokens_update"] == 4
    assert record["train/effective_target_tokens_configured"] == 4
    assert record["train/micro_batches_per_update"] == 2


def test_global_norm_clipping_is_recorded_and_limits_the_update(tmp_path: Path):
    clipped_model = FixedLogitModel()
    unclipped_model = FixedLogitModel()
    unclipped_model.load_state_dict(clipped_model.state_dict())
    initial = clipped_model.logits.detach().clone()
    batch = _batch([[0, 1]])

    clipped = _recipe_trainer(
        tmp_path / "clipped",
        model=clipped_model,
        batches=[batch],
        optimizer=torch.optim.SGD(clipped_model.parameters(), lr=0.1),
        max_grad_norm=1.0e-4,
    )
    unclipped = _recipe_trainer(
        tmp_path / "unclipped",
        model=unclipped_model,
        batches=[batch],
        optimizer=torch.optim.SGD(unclipped_model.parameters(), lr=0.1),
    )

    clipped.fit()
    unclipped.fit()

    record = next(item for item in clipped.metrics if item.get("event") == "step")
    assert record["optimizer/gradient_norm"] > 1.0e-4
    assert record["optimizer/gradient_clipped"] is True
    assert torch.linalg.vector_norm(clipped_model.logits - initial) < torch.linalg.vector_norm(
        unclipped_model.logits - initial
    )


def test_scheduler_advances_once_per_optimizer_update(tmp_path: Path):
    model = FixedLogitModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=2,
        decay_steps=4,
        min_lr_ratio=0.1,
    )
    trainer = _recipe_trainer(
        tmp_path,
        model=model,
        batches=[_batch([[0, 1]]) for _ in range(6)],
        optimizer=optimizer,
        scheduler=scheduler,
        max_steps=3,
        gradient_accumulation_steps=2,
    )

    trainer.fit()

    assert trainer.optimizer_step == 3
    assert scheduler.optimizer_steps == 3
    step_records = [item for item in trainer.metrics if item.get("event") == "step"]
    assert [item["optimizer/lr_used"] for item in step_records] == pytest.approx([0.05, 0.1, 0.1])
    assert step_records[-1]["optimizer/lr"] == pytest.approx(0.055)


def test_bf16_autocast_requests_cuda_and_cpu_requires_fp32(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_autocast(**kwargs):
        captured.update(kwargs)
        return nullcontext()

    monkeypatch.setattr(optimization_module.torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(optimization_module.torch, "autocast", fake_autocast)
    with optimization_module.autocast_context(torch.device("cuda"), "bf16"):
        pass
    assert captured == {"device_type": "cuda", "dtype": torch.bfloat16}

    cpu_model = FixedLogitModel()
    with pytest.raises(ValueError, match="requires runtime.device=cuda"):
        _recipe_trainer(
            tmp_path,
            model=cpu_model,
            batches=[_batch([[0, 1]])],
            optimizer=torch.optim.SGD(cpu_model.parameters(), lr=0.1),
            precision="bf16",
        )
