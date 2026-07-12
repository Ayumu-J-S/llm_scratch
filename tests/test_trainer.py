from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

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


def _trainer(tmp_path: Path, batches, **training_overrides) -> Trainer:
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
        }
    )
    return Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        scheduler=None,
        train_loader=ListLoader(batches),
        validation_loader=ListLoader(batches),
        checkpoint_dir=tmp_path,
        cfg=cfg,
        device=torch.device("cpu"),
    )


def _batch(labels: list[list[int]]) -> dict[str, torch.Tensor]:
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return {"inputs": torch.zeros_like(labels_tensor), "labels": labels_tensor}


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


def test_max_steps_boundary_does_not_fetch_or_update_extra_batch(tmp_path: Path):
    batches = [_batch([[0, 1]]) for _ in range(5)]
    trainer = _trainer(tmp_path, batches, max_steps=2)
    trainer.fit()
    assert trainer.optimizer_step == 2
    assert trainer.target_tokens == 4
    assert max(item["optimizer_step"] for item in trainer.metrics) == 2


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
    assert len(list(tmp_path.glob("model_last.pth"))) == 1
    assert trainer._last_checkpoint_step == 3


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
    assert [item["target_tokens"] for item in trainer.metrics if item.get("event") == "validation"] == [4]
    assert [item["target_tokens"] for item in trainer.metrics if item.get("event") == "checkpoint"] == [4]
    assert any(item.get("event") == "epoch_summary" and "train/loss" in item for item in trainer.metrics)
    assert any(item.get("event") == "epoch_summary" and "train/perplexity" in item for item in trainer.metrics)
    assert (tmp_path / "metrics.jsonl").exists()


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


def test_fractional_step_and_token_budgets_are_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="max_steps.*positive integer"):
        _trainer(tmp_path, [_batch([[0, 1]])], max_steps=1.5)
    with pytest.raises(ValueError, match="max_tokens.*positive integer"):
        _trainer(tmp_path, [_batch([[0, 1]])], max_tokens=1.5)
