from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import hydra
import pytest
import torch
from omegaconf import OmegaConf, open_dict

import evaluate as evaluate_module
import training.checkpoint as checkpoint_module
from data.streaming_dataset import causal_lm_collate_fn, target_sources_from_spans
from evaluation.scoring import CausalLMScorer, manifest_identities
from models.simple_decoder_transformer import SimpleDecoderTransformer
from runtime.config import ConfigPreflightError
from tokenizer.canonical import CanonicalTokenizer
from train import build_validation_loader_factory
from training.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointManager,
    build_checkpoint_identity,
    build_logical_checkpoint_identity,
    configured_manifest_fingerprints,
    load_checkpoint_for_generation,
)
from training.trainer import Trainer


CONFIG_DIR = Path(__file__).parents[1] / "config"
TOKENIZER_CONFIG = {
    "manifest_path": "assets/tokenizers/llm-jp-v1/manifest.json",
    "expected_fingerprint": "12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b",
}


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


class FailingCloseIterator(ClosingIterator):
    def close(self):
        super().close()
        raise RuntimeError("iterator close failed")


class FailingCloseLoader(ClosingLoader):
    def __iter__(self):
        self.iterator = FailingCloseIterator(self.batches)
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


def _compose(*overrides: str):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return hydra.compose(config_name="train", overrides=list(overrides))


def _streaming_checkpoint_config():
    config = _compose("profile=evaluation")
    with open_dict(config):
        config.profile.name = "pretrain_streaming"
        config.profile.purpose = "pretraining"
        del config.profile.task
        config.runtime.device = "cpu"
        config.model.embed_size = 8
        config.model.num_heads = 2
        config.model.num_layers = 1
        config.model.dropout = 0.0
        config.training.precision = "fp32"
        config.training.gradient_accumulation_steps = 1
        config.training.max_grad_norm = None
        config.wandb.enabled = False
    return config


def _milestone_checkpoint(tmp_path: Path):
    config = _streaming_checkpoint_config()
    tokenizer = CanonicalTokenizer.from_config(config.tokenizer)
    model = SimpleDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_size=config.model.embed_size,
        num_heads=config.model.num_heads,
        max_len=config.training.sequence_length,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        pad_token_id=tokenizer.pad_token_id,
    )
    identity = build_checkpoint_identity(config)
    identity["tokenizer_fingerprint"] = tokenizer.fingerprint
    counters = {"optimizer_step": 7, "target_tokens": 19, "elapsed_seconds": 1.25}
    checkpoint = CheckpointManager(tmp_path, keep_last_n=1, identity=identity).save_milestone(
        {
            "model": model.state_dict(),
            "counters": counters,
            "resolved_config": OmegaConf.to_container(config, resolve=True),
            "run_identity": identity,
        }
    )
    return checkpoint, config, model, identity, counters


def test_known_logits_match_token_weighted_nll_and_perplexity():
    model = FixedLogitModel([0.0, 1.0, 2.0])
    batch = _batch([[0, 1, 2]])
    result = _scorer().score(model, lambda: [batch])

    expected = torch.nn.functional.cross_entropy(
        model.logits.detach().expand(3, -1), torch.tensor([0, 1, 2])
    ).item()
    assert result.nll == pytest.approx(expected)
    assert result.perplexity == pytest.approx(torch.exp(torch.tensor(expected)).item())
    assert result.aggregate.nll_sum == pytest.approx(expected * 3)
    assert result.aggregate.perplexity_overflow is False
    assert result.target_tokens == 3
    assert result.evaluated_windows == 1
    assert model.logits.grad is None
    for field in (
        "loader_construction_seconds",
        "iterator_creation_seconds",
        "data_wait_seconds",
        "host_device_preparation_seconds",
        "forward_seconds",
        "loss_seconds",
        "validation_seconds",
    ):
        assert result.timing[field] >= 0.0


def test_partial_and_ignored_labels_reconcile_by_corpus():
    model = FixedLogitModel([0.0, 1.0, 2.0, 3.0])
    batch = _batch(
        [[0, -100, 1], [2, 3, -100]],
        [["ja", "ja", "en"], ["en", "en", "en"]],
    )
    result = _scorer().score(model, lambda: [batch])

    assert result.target_tokens == 4
    assert result.by_corpus["ja"].target_tokens == 1
    assert result.by_corpus["en"].target_tokens == 3
    assert result.nll == pytest.approx(
        sum(score.nll * score.target_tokens for score in result.by_corpus.values()) / 4
    )
    assert result.aggregate.nll_sum == pytest.approx(
        sum(score.nll_sum for score in result.by_corpus.values())
    )


def test_source_spans_propagate_sources_at_causal_label_boundary():
    spans = [
        {"source": "ja", "start": 0, "end": 2},
        {"source": "en", "start": 2, "end": 4},
    ]
    assert target_sources_from_spans(spans, target_count=3) == ["ja", "en", "en"]
    batch = causal_lm_collate_fn(
        [
            {
                "input_ids": torch.tensor([10, 11, 12, 13]),
                "target_sources": ["ja", "en", "en"],
                "source_spans": spans,
            }
        ]
    )
    assert batch["target_sources"] == [["ja", "en", "en"]]


@pytest.mark.parametrize(
    "spans",
    [
        [{"source": "ja", "start": 0, "end": 2}],
        [
            {"source": "ja", "start": 0, "end": 4},
            {"source": "en", "start": 2, "end": 4},
        ],
    ],
)
def test_source_spans_fail_when_a_target_is_missing_or_double_attributed(spans):
    with pytest.raises(ValueError, match="exactly once"):
        target_sources_from_spans(spans, target_count=3)


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
    batched = scorer.score(model, lambda: [one_batch])
    split = scorer.score(model, lambda: windows)
    repeated = scorer.score(model, lambda: windows)

    assert batched.evaluated_window_sha256 == split.evaluated_window_sha256
    assert batched.evaluated_token_sha256 == split.evaluated_token_sha256
    assert batched.by_corpus == split.by_corpus == repeated.by_corpus

    context_changed = {
        **one_batch,
        "inputs": one_batch["inputs"].clone(),
    }
    context_changed["inputs"][0, 0] = 2
    reassigned = {**one_batch, "target_sources": [["en", "ja"], ["en", "en"]]}
    changed_context_result = scorer.score(model, lambda: [context_changed])
    reassigned_result = scorer.score(model, lambda: [reassigned])
    assert changed_context_result.evaluated_window_sha256 != batched.evaluated_window_sha256
    assert reassigned_result.evaluated_window_sha256 != batched.evaluated_window_sha256
    assert reassigned_result.evaluated_token_sha256 != batched.evaluated_token_sha256


def test_validation_attribution_must_match_declared_manifests():
    model = FixedLogitModel([0.0, 1.0])
    with pytest.raises(ValueError, match="does not match the actual validation loader"):
        _scorer().score(
            model,
            lambda: [_batch([[0, 1]], [["en", "en"]])],
            manifest_identity={"ja": {"selection": "validation"}},
        )


def test_perplexity_overflow_is_standards_safe_json(tmp_path):
    result = _scorer().score(FixedLogitModel([0.0, 1000.0]), lambda: [_batch([[0]])])
    assert result.perplexity is None
    assert result.aggregate.perplexity_overflow is True

    destination = tmp_path / "evaluation.json"
    evaluate_module._write_json_atomic(destination, result.as_dict())
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["aggregate"]["perplexity"] is None
    assert "Infinity" not in destination.read_text(encoding="utf-8")


def test_scoring_closes_iterator_when_model_fails():
    loader = ClosingLoader([_batch([[0, 1]])])
    with pytest.raises(FloatingPointError, match="non-finite"):
        _scorer().score(NaNModel([0.0, 1.0]), lambda: loader)
    assert loader.iterator is not None
    assert loader.iterator.closed is True


def test_scoring_rejects_zero_valid_targets():
    with pytest.raises(ValueError, match="zero target tokens"):
        _scorer().score(FixedLogitModel([0.0, 1.0]), lambda: [_batch([[-100, -100]])])


def test_scorer_restores_model_mode():
    model = FixedLogitModel([0.0, 1.0])
    model.eval()
    _scorer().score(model, lambda: [_batch([[0, 1]])])
    assert model.training is False
    model.train()
    _scorer().score(model, lambda: [_batch([[0, 1]])])
    assert model.training is True


def test_scorer_restores_model_mode_when_iterator_close_fails():
    model = FixedLogitModel([0.0, 1.0])
    model.train()

    with pytest.raises(RuntimeError, match="iterator close failed"):
        _scorer().score(model, lambda: FailingCloseLoader([_batch([[0, 1]])]))

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
        validation_loader_factory=lambda: [batch],
        checkpoint_dir=tmp_path,
        cfg=cfg,
        device=torch.device("cpu"),
    )

    trainer.fit()

    record = next(item for item in trainer.metrics if item.get("event") == "memorization")
    assert "memorization/loss" in record
    assert not any(key.startswith("validation/") for key in record)
    assert record.get("memorization/physical_checkpoint_identity") is None
    assert not (tmp_path / "best.pt").exists()


def test_standalone_milestone_matches_shared_training_time_score(tmp_path):
    checkpoint, checkpoint_config, model, identity, counters = _milestone_checkpoint(
        tmp_path / "checkpoints"
    )
    loader_factory = build_validation_loader_factory(checkpoint_config, device=torch.device("cpu"))
    manifest_loader = loader_factory()
    live_result = _scorer().score(
        model,
        loader_factory,
        namespace="validation",
        logical_checkpoint_identity=build_logical_checkpoint_identity(identity, counters),
        configured_data_fingerprints=configured_manifest_fingerprints(checkpoint_config),
        manifest_identity=manifest_identities(manifest_loader.dataset.resolved_manifests),
    )

    evaluation_config = _compose("profile=evaluation")
    with open_dict(evaluation_config):
        evaluation_config.evaluation.checkpoint_path = str(checkpoint)
        evaluation_config.evaluation.output_path = str(tmp_path / "standalone.json")
        evaluation_config.evaluation.device = "cpu"
    result_path = evaluate_module.evaluate_checkpoint(evaluation_config)
    standalone = json.loads(result_path.read_text(encoding="utf-8"))
    result = standalone["result"]

    assert result["aggregate"] == live_result.aggregate.as_dict()
    assert result["by_corpus"] == {
        name: score.as_dict() for name, score in live_result.by_corpus.items()
    }
    assert result["evaluated_window_sha256"] == live_result.evaluated_window_sha256
    assert result["evaluated_token_sha256"] == live_result.evaluated_token_sha256
    assert result["manifest_identity"] == live_result.manifest_identity
    assert standalone["checkpoint"]["kind"] == "milestone"
    assert standalone["checkpoint"]["logical"] == live_result.logical_checkpoint_identity
    assert standalone["checkpoint"]["physical"]["path"] == str(checkpoint.resolve())
    assert len(standalone["checkpoint"]["physical"]["sha256"]) == 64
    evaluator_run = standalone["evaluator_run"]
    assert len(evaluator_run["git"]["sha"]) == 40
    assert isinstance(evaluator_run["git"]["dirty"], bool)
    assert evaluator_run["resolved_config"]["evaluation"]["checkpoint_path"] == str(checkpoint)
    assert len(evaluator_run["resolved_config_sha256"]) == 64
    assert evaluator_run["lock"]["path"].endswith("uv.lock")
    assert len(evaluator_run["lock"]["sha256"]) == 64
    assert evaluator_run["environment"]["os_release"]["name"]
    assert evaluator_run["environment"]["python"]
    assert evaluator_run["environment"]["torch"]["version"]
    assert "container_image" in evaluator_run["environment"]
    rendered = result_path.read_text(encoding="utf-8")
    assert '"inputs"' not in rendered
    assert '"labels"' not in rendered


def test_checkpoint_snapshot_payload_and_identity_survive_atomic_replacement(tmp_path, monkeypatch):
    checkpoint, *_ = _milestone_checkpoint(tmp_path / "checkpoints")
    original_bytes = checkpoint.read_bytes()
    replacement_payload = copy.deepcopy(
        torch.load(checkpoint, map_location="cpu", weights_only=False)
    )
    replacement_payload["state"]["counters"]["optimizer_step"] = 8
    replacement_payload["state"]["counters"]["target_tokens"] = 23
    replacement = tmp_path / "replacement.pt"
    torch.save(replacement_payload, replacement)
    replacement_bytes = replacement.read_bytes()
    assert replacement_bytes != original_bytes

    deserialize_handle = checkpoint_module._torch_load_handle

    def replace_then_deserialize(handle):
        os.replace(replacement, checkpoint)
        return deserialize_handle(handle)

    monkeypatch.setattr(checkpoint_module, "_torch_load_handle", replace_then_deserialize)
    loaded = checkpoint_module.load_checkpoint_for_generation(checkpoint)

    assert loaded.payload["state"]["counters"]["optimizer_step"] == 7
    assert loaded.physical_identity["sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert loaded.physical_identity["size_bytes"] == len(original_bytes)
    assert loaded.physical_identity["inode"] != checkpoint.stat().st_ino
    assert (
        hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        == hashlib.sha256(replacement_bytes).hexdigest()
    )
    assert loaded.physical_identity["sha256"] != hashlib.sha256(checkpoint.read_bytes()).hexdigest()


@pytest.mark.parametrize("collision_kind", ["same_path", "same_inode"])
def test_standalone_rejects_output_checkpoint_collisions_before_evaluation(
    tmp_path, collision_kind
):
    checkpoint, *_ = _milestone_checkpoint(tmp_path / "checkpoints")
    original = checkpoint.read_bytes()
    if collision_kind == "same_path":
        output_path = checkpoint
        expected = "checkpoint path"
    else:
        output_path = tmp_path / "checkpoint-hardlink.json"
        os.link(checkpoint, output_path)
        expected = "share an inode"
    evaluation_config = _compose("profile=evaluation")
    with open_dict(evaluation_config):
        evaluation_config.evaluation.checkpoint_path = str(checkpoint)
        evaluation_config.evaluation.output_path = str(output_path)
        evaluation_config.evaluation.device = "cpu"

    with pytest.raises(ValueError, match=expected):
        evaluate_module.evaluate_checkpoint(evaluation_config)

    assert checkpoint.read_bytes() == original


def test_standalone_wandb_summary_records_compact_identities_and_local_hash(tmp_path, monkeypatch):
    checkpoint, *_ = _milestone_checkpoint(tmp_path / "checkpoints")

    class Summary:
        def __init__(self):
            self.values = {}

        def update(self, values):
            self.values.update(values)

    class Run:
        def __init__(self):
            self.summary = Summary()
            self.finished = False

        def finish(self):
            self.finished = True

    run = Run()
    monkeypatch.setattr(evaluate_module.wandb, "init", lambda **kwargs: run)
    evaluation_config = _compose("profile=evaluation")
    output_path = tmp_path / "wandb-standalone.json"
    with open_dict(evaluation_config):
        evaluation_config.evaluation.checkpoint_path = str(checkpoint)
        evaluation_config.evaluation.output_path = str(output_path)
        evaluation_config.evaluation.device = "cpu"
        evaluation_config.evaluation.wandb.enabled = True
        evaluation_config.evaluation.wandb.mode = "offline"

    result_path = evaluate_module.evaluate_checkpoint(evaluation_config)
    local_bytes = result_path.read_bytes()
    local_payload = json.loads(local_bytes)
    values = run.summary.values

    assert run.finished is True
    assert values["evaluation/scorer_identity"] == {
        "revision": local_payload["result"]["scorer_revision"]
    }
    assert values["evaluation/checkpoint_identity"]["kind"] == "milestone"
    assert values["evaluation/checkpoint_identity"]["logical"]
    assert (
        values["evaluation/checkpoint_identity"]["physical"]["sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert values["evaluation/manifest_identity"]
    assert values["evaluation/local_result_identity"] == {
        "path": str(result_path),
        "sha256": hashlib.sha256(local_bytes).hexdigest(),
        "size_bytes": len(local_bytes),
    }


def test_standalone_preflight_rejects_cpu_for_bf16_checkpoint(tmp_path, monkeypatch):
    checkpoint_config = _streaming_checkpoint_config()
    with open_dict(checkpoint_config):
        checkpoint_config.runtime.device = "cuda"
        checkpoint_config.training.precision = "bf16"
    tokenizer = CanonicalTokenizer.from_config(checkpoint_config.tokenizer)
    model = SimpleDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_size=checkpoint_config.model.embed_size,
        num_heads=checkpoint_config.model.num_heads,
        max_len=checkpoint_config.training.sequence_length,
        num_layers=checkpoint_config.model.num_layers,
        dropout=checkpoint_config.model.dropout,
        pad_token_id=tokenizer.pad_token_id,
    )
    identity = build_checkpoint_identity(checkpoint_config)
    identity["tokenizer_fingerprint"] = tokenizer.fingerprint
    checkpoint = CheckpointManager(
        tmp_path / "bf16-checkpoint", keep_last_n=1, identity=identity
    ).save_milestone(
        {
            "model": model.state_dict(),
            "counters": {"optimizer_step": 1, "target_tokens": 8, "elapsed_seconds": 1.0},
            "resolved_config": OmegaConf.to_container(checkpoint_config, resolve=True),
            "run_identity": identity,
        }
    )
    evaluation_config = _compose("profile=evaluation")
    with open_dict(evaluation_config):
        evaluation_config.evaluation.checkpoint_path = str(checkpoint)
        evaluation_config.evaluation.output_path = str(tmp_path / "must-not-exist.json")
        evaluation_config.evaluation.device = "cpu"

    monkeypatch.setattr(
        evaluate_module,
        "select_device",
        lambda *_args, **_kwargs: pytest.fail("device selection must follow compatibility preflight"),
    )
    with pytest.raises(
        ConfigPreflightError,
        match=r"evaluation\.device=cpu.*checkpoint-owned training\.precision=bf16",
    ):
        evaluate_module.evaluate_checkpoint(evaluation_config)
    assert not (tmp_path / "must-not-exist.json").exists()


def test_checkpoint_resolved_config_tampering_is_rejected_before_evaluation(tmp_path):
    checkpoint, *_ = _milestone_checkpoint(tmp_path / "checkpoints")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["state"]["resolved_config"]["training"]["sequence_length"] += 1
    torch.save(payload, checkpoint)

    with pytest.raises(CheckpointCompatibilityError, match="config_sha256"):
        load_checkpoint_for_generation(checkpoint)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["state"]["resolved_config"]["training"]["sequence_length"] -= 1
    payload["identity"]["data_fingerprints"] = ["tampered"]
    torch.save(payload, checkpoint)
    with pytest.raises(CheckpointCompatibilityError, match="data manifests"):
        load_checkpoint_for_generation(checkpoint)


def test_scorer_rejects_manifest_tampering_in_actual_loader(tmp_path):
    checkpoint, checkpoint_config, model, identity, counters = _milestone_checkpoint(
        tmp_path / "checkpoints"
    )
    del checkpoint
    loader_factory = build_validation_loader_factory(checkpoint_config, device=torch.device("cpu"))

    def tampered_factory():
        loader = loader_factory()
        name, manifest = next(iter(loader.dataset.resolved_manifests.items()))
        loader.dataset.resolved_manifests = {name: replace(manifest, manifest_fingerprint="c" * 64)}
        return loader

    with pytest.raises(ValueError, match="fingerprint changed"):
        _scorer().score(
            model,
            tampered_factory,
            namespace="validation",
            logical_checkpoint_identity=build_logical_checkpoint_identity(identity, counters),
            configured_data_fingerprints=configured_manifest_fingerprints(checkpoint_config),
        )


def test_scorer_rejects_wrong_split_or_missing_loader_manifest_metadata(tmp_path):
    _, checkpoint_config, model, identity, counters = _milestone_checkpoint(
        tmp_path / "checkpoints"
    )
    loader_factory = build_validation_loader_factory(checkpoint_config, device=torch.device("cpu"))

    def wrong_split_factory():
        loader = loader_factory()
        name, manifest = next(iter(loader.dataset.resolved_manifests.items()))
        loader.dataset.resolved_manifests = {name: replace(manifest, selection="train")}
        return loader

    with pytest.raises(ValueError, match="selection 'train'.*expected 'validation'"):
        _scorer().score(
            model,
            wrong_split_factory,
            namespace="validation",
            logical_checkpoint_identity=build_logical_checkpoint_identity(identity, counters),
            configured_data_fingerprints=configured_manifest_fingerprints(checkpoint_config),
        )

    def missing_metadata_factory():
        loader = loader_factory()
        del loader.dataset.config
        return loader

    with pytest.raises(ValueError, match="missing required manifest identity metadata"):
        _scorer().score(
            model,
            missing_metadata_factory,
            namespace="validation",
            logical_checkpoint_identity=build_logical_checkpoint_identity(identity, counters),
            configured_data_fingerprints=configured_manifest_fingerprints(checkpoint_config),
        )


def test_standalone_held_out_evaluation_rejects_memorization_checkpoint(tmp_path):
    checkpoint_config = _compose("profile=smoke_overfit")
    with open_dict(checkpoint_config):
        checkpoint_config.model.embed_size = 8
        checkpoint_config.model.num_heads = 2
        checkpoint_config.model.num_layers = 1
        checkpoint_config.model.dropout = 0.0
    tokenizer = CanonicalTokenizer.from_config(checkpoint_config.tokenizer)
    model = SimpleDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_size=checkpoint_config.model.embed_size,
        num_heads=checkpoint_config.model.num_heads,
        max_len=checkpoint_config.training.sequence_length,
        num_layers=checkpoint_config.model.num_layers,
        dropout=checkpoint_config.model.dropout,
        pad_token_id=tokenizer.pad_token_id,
    )
    identity = build_checkpoint_identity(checkpoint_config)
    identity["tokenizer_fingerprint"] = tokenizer.fingerprint
    checkpoint = CheckpointManager(
        tmp_path / "memorization", keep_last_n=1, identity=identity
    ).save_final(
        {
            "model": model.state_dict(),
            "counters": {
                "optimizer_step": 1,
                "target_tokens": 8,
                "elapsed_seconds": 1.0,
            },
            "resolved_config": OmegaConf.to_container(checkpoint_config, resolve=True),
            "run_identity": identity,
        }
    )
    evaluation_config = _compose("profile=evaluation")
    with open_dict(evaluation_config):
        evaluation_config.evaluation.checkpoint_path = str(checkpoint)
        evaluation_config.evaluation.output_path = str(tmp_path / "must-not-exist.json")
        evaluation_config.evaluation.device = "cpu"

    with pytest.raises(ValueError, match="held-out evaluation requires"):
        evaluate_module.evaluate_checkpoint(evaluation_config)
    assert not (tmp_path / "must-not-exist.json").exists()
