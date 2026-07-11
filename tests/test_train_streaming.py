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
    tokenizer.train("abcd", vocab_size=5)
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))

    streaming_split = {
        "max_tokens": 4,
        "add_eos": False,
        "seed": 1,
        "sources": [
            {
                "name": "docs",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "abcd"}],
            }
        ],
    }
    cfg = OmegaConf.create(
        {
            "data": {
                "streaming": {
                    "train": streaming_split,
                    "validation": streaming_split,
                }
            },
            "training": {
                "sequence_length": 3,
                "batch_size": 2,
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
    assert validation_batches[0]["inputs"].shape == (1, 3)
    assert validation_batches[0]["labels"].shape == (1, 3)


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
                        "adb4dadc6f2e37f340504c8368be44a7b0c7a73b337c34d5457f2f2629bc0256"
                    ),
                },
            }
        )
    )
    assert manifest.purpose.value == "memorization_smoke"
    assert len(manifest.documents) == 2


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
