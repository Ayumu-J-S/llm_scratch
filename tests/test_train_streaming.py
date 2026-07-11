from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

import train as train_module
from data.manifests import ManifestError
from tokenizer.bpe import BPETokenizer


def test_streaming_validation_keeps_partial_batch(tmp_path):
    tokenizer_dir = tmp_path / "tokenizers"
    tokenizer_dir.mkdir()
    tokenizer = BPETokenizer(special_tokens=["<eos>"])
    tokenizer.train("the model learns from data. the model predicts the next token.", vocab_size=40)
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))

    streaming_split = {
        "max_tokens": "max",
        "add_eos": False,
        "seed": 1,
        "sources": [
            {
                "name": "smoke",
                "type": "manifest",
                "ratio": 1.0,
                "manifest_path": str(
                    Path(__file__).parent
                    / "fixtures"
                    / "data_manifests"
                    / "memorization.manifest.json"
                ),
                "expected_fingerprint": (
                    "00c3797a7d0eda13950fd699a60c45fcd388829f016479caaeb369438767bd31"
                ),
                "selection": "all",
            }
        ],
    }
    cfg = OmegaConf.create(
        {
            "data": {
                "streaming": {
                    "require_manifests": True,
                    "train": streaming_split,
                    "validation": streaming_split,
                }
            },
            "training": {
                "sequence_length": 3,
                "batch_size": 100,
            },
            "artifacts": {
                "tokenizers_dir": str(tokenizer_dir),
                "tokenizer_filename": "tokenizer.json",
            },
        }
    )

    train_batches = list(train_module.build_streaming_dataloader(cfg, "train"))
    validation_batches = list(train_module.build_streaming_dataloader(cfg, "validation"))

    assert train_batches == []
    assert len(validation_batches) == 1
    assert validation_batches[0]["inputs"].shape[1] == 3
    assert validation_batches[0]["labels"].shape == validation_batches[0]["inputs"].shape


def test_default_smoke_requires_the_pinned_memorization_manifest():
    with pytest.raises(ValueError, match="data.mode=memorization_smoke"):
        train_module.resolve_memorization_smoke(
            OmegaConf.create({"mode": "local_text", "purpose": "memorization_smoke"})
        )

    fixture = Path(__file__).parent / "fixtures" / "data_manifests"
    manifest = train_module.resolve_memorization_smoke(
        OmegaConf.create(
            {
                "mode": "memorization_smoke",
                "memorization": {
                    "manifest_path": str(fixture / "memorization.manifest.json"),
                    "expected_fingerprint": (
                        "00c3797a7d0eda13950fd699a60c45fcd388829f016479caaeb369438767bd31"
                    ),
                },
            }
        )
    )
    assert manifest.purpose.value == "memorization_smoke"
    assert len(manifest.documents) == 2

    tokenizer_text = (Path(__file__).parents[1] / "data" / "inputLearnText.txt").read_text(
        encoding="utf-8"
    )
    tokenizer = BPETokenizer(special_tokens=["<pad>", "<bos>", "<eos>"])
    tokenizer.train(tokenizer_text, vocab_size=len(set(tokenizer_text)) + 3)
    assert train_module.resolved_manifest_token_ids(manifest, tokenizer)


def test_streaming_train_validation_same_membership_fails_before_iteration(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "data_manifests"
    source = {
        "type": "manifest",
        "manifest_path": str(fixture / "bilingual.manifest.json"),
        "expected_fingerprint": (
            "47cca88c4a5595e27eb5d60d99918fb77c30b23f7c0ae98024153f25e14ffc19"
        ),
        "selection": "train",
        "ratio": 1.0,
    }
    cfg = OmegaConf.create(
        {
            "data": {
                "streaming": {
                    "require_manifests": True,
                    "train": {
                        "max_tokens": "max",
                        "add_eos": False,
                        "sources": [{"name": "train", **source}],
                    },
                    "validation": {
                        "max_tokens": "max",
                        "add_eos": False,
                        "sources": [{"name": "validation", **source}],
                    },
                }
            },
            "training": {"sequence_length": 3, "batch_size": 1},
            "artifacts": {
                "tokenizers_dir": str(tmp_path / "unused"),
                "tokenizer_filename": "unused.json",
            },
        }
    )
    train_loader = train_module.build_streaming_dataloader(cfg, "train")
    validation_loader = train_module.build_streaming_dataloader(cfg, "validation")
    with pytest.raises(ManifestError, match="document_id overlap"):
        train_module.validate_streaming_dataloaders(train_loader, validation_loader)


def test_training_streaming_rejects_false_and_empty_manifest_bypasses(tmp_path):
    memory_split = {
        "max_tokens": "max",
        "add_eos": False,
        "sources": [
            {
                "name": "bypass",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "abcd"}],
            }
        ],
    }
    base = {
        "training": {"sequence_length": 3, "batch_size": 1},
        "artifacts": {
            "tokenizers_dir": str(tmp_path / "unused"),
            "tokenizer_filename": "unused.json",
        },
    }
    false_cfg = OmegaConf.create(
        {
            **base,
            "data": {
                "streaming": {
                    "require_manifests": False,
                    "train": memory_split,
                    "validation": memory_split,
                }
            },
        }
    )
    with pytest.raises(ValueError, match="require_manifests=false"):
        train_module.build_streaming_dataloader(false_cfg, "train")

    absent_cfg = OmegaConf.create(
        {
            **base,
            "data": {"streaming": {"train": memory_split, "validation": memory_split}},
        }
    )
    train_loader = train_module.build_streaming_dataloader(absent_cfg, "train")
    validation_loader = train_module.build_streaming_dataloader(absent_cfg, "validation")
    with pytest.raises(ManifestError, match="resolved train manifests"):
        train_module.validate_streaming_dataloaders(train_loader, validation_loader)


def test_canonical_streaming_fixture_budgets_do_not_exhaust():
    cfg = OmegaConf.load(Path(__file__).parents[1] / "config" / "train.yaml")
    assert cfg.data.streaming.require_manifests is True
    assert cfg.data.streaming.train.max_tokens == "max"
    assert cfg.data.streaming.validation.max_tokens == "max"
