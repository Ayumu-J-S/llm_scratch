from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from data.streaming_dataset import causal_lm_collate_fn, target_sources_from_spans
from evaluation.scoring import CausalLMScorer
from training.trainer import Trainer


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(*inputs.shape, -1)


class ClosingIterator:
    def __init__(self, batches):
        self.batches = batches
        self.closed = False
        self.position = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.position >= len(self.batches):
            raise StopIteration
        batch = self.batches[self.position]
        self.position += 1
        return batch

    def close(self):
        self.closed = True


class ClosingLoader:
    def __init__(self, batches):
        self.batches = batches
        self.iterator = None

    def __iter__(self):
        self.iterator = ClosingIterator(self.batches)
        return self.iterator


class NaNModel(FixedLogitModel):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.full((*inputs.shape, self.logits.numel()), float("nan"))


def _batch(labels, sources=None):
    labels = torch.tensor(labels, dtype=torch.long)
    batch = {"inputs": torch.zeros_like(labels), "labels": labels}
    if sources is not None:
        batch["target_sources"] = sources
    return batch


def _scorer():
    return CausalLMScorer(device=torch.device("cpu"), precision="fp32")


def test_known_logits_match_token_weighted_nll_and_perplexity():
    model = FixedLogitModel([0.0, 1.0, 2.0])
    batch = _batch([[0, 1, 2]])
    result = _scorer().score(model, [batch])

    expected = torch.nn.functional.cross_entropy(
        model.logits.detach().expand(3, -1), torch.tensor([0, 1, 2])
    ).item()
    assert result.nll == pytest.approx(expected)
    assert result.perplexity == pytest.approx(torch.exp(torch.tensor(expected)).item())
    assert result.target_tokens == 3
    assert result.evaluated_windows == 1


def test_partial_and_ignored_labels_reconcile_by_corpus():
    model = FixedLogitModel([0.0, 1.0, 2.0, 3.0])
    batch = _batch(
        [[0, -100, 1], [2, 3, -100]],
        [["ja", "ja", "en"], ["en", "en", "en"]],
    )
    result = _scorer().score(model, [batch])

    assert result.target_tokens == 4
    assert result.by_corpus["ja"].target_tokens == 1
    assert result.by_corpus["en"].target_tokens == 3
    assert result.nll == pytest.approx(
        sum(score.nll * score.target_tokens for score in result.by_corpus.values()) / 4
    )


def test_source_spans_propagate_sources_at_causal_label_boundary():
    spans = [
        {"source": "ja", "start": 0, "end": 2},
        {"source": "en", "start": 2, "end": 4},
    ]
    assert target_sources_from_spans(spans, target_count=3) == ["ja", "en", "en"]
    batch = causal_lm_collate_fn(
        [{"input_ids": torch.tensor([10, 11, 12, 13]), "target_sources": ["ja", "en", "en"], "source_spans": spans}]
    )
    assert batch["target_sources"] == [["ja", "en", "en"]]


def test_fixed_window_score_is_batching_independent_and_repeated():
    model = FixedLogitModel([0.0, 1.0, 2.0])
    windows = [
        _batch([[0, 1]], [["ja", "ja"]]),
        _batch([[1, 2]], [["en", "en"]]),
    ]
    one_batch = {
        "inputs": torch.cat([window["inputs"] for window in windows]),
        "labels": torch.cat([window["labels"] for window in windows]),
        "target_sources": ["ja", "ja"],
    }
    one_batch["target_sources"] = [*windows[0]["target_sources"], *windows[1]["target_sources"]]
    scorer = _scorer()
    batched = scorer.score(model, [one_batch])
    split = scorer.score(model, windows)
    repeated = scorer.score(model, lambda: windows)

    assert batched.evaluated_window_sha256 == split.evaluated_window_sha256
    assert batched.evaluated_token_sha256 == split.evaluated_token_sha256
    assert batched.by_corpus == split.by_corpus == repeated.by_corpus


def test_scoring_closes_iterator_when_model_fails():
    loader = ClosingLoader([_batch([[0, 1]])])
    with pytest.raises(FloatingPointError, match="non-finite"):
        _scorer().score(NaNModel([0.0, 1.0]), loader)
    assert loader.iterator is not None
    assert loader.iterator.closed is True


def test_scorer_restores_model_mode():
    model = FixedLogitModel([0.0, 1.0])
    model.eval()
    _scorer().score(model, [_batch([[0, 1]])])
    assert model.training is False
    model.train()
    _scorer().score(model, [_batch([[0, 1]])])
    assert model.training is True


def test_trainer_memorization_metrics_have_no_validation_namespace(tmp_path):
    model = FixedLogitModel([0.0, 1.0])
    batch = _batch([[0, 1]])
    cfg = OmegaConf.create(
        {
            "data": {"mode": "memorization_smoke"},
            "training": {
                "epochs": 1,
                "max_steps": 1,
                "max_tokens": None,
                "max_time": None,
                "batch_size": 1,
                "sequence_length": 2,
                "precision": "fp32",
                "gradient_accumulation_steps": 1,
                "max_grad_norm": None,
                "validation_every_n_steps": 1,
                "log_every_n_steps": None,
                "checkpoint_every_n_steps": None,
                "milestone_every_n_steps": None,
                "scheduler": {"interval": "step"},
            },
            "wandb": {"enabled": False},
        }
    )
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        scheduler=None,
        train_loader=[batch],
        validation_loader=[batch],
        checkpoint_dir=tmp_path,
        cfg=cfg,
        device=torch.device("cpu"),
    )

    trainer.fit()

    record = next(item for item in trainer.metrics if item.get("event") == "memorization")
    assert "memorization/loss" in record
    assert not any(key.startswith("validation/") for key in record)
    assert record["memorization/physical_checkpoint_identity"]["sha256"]
