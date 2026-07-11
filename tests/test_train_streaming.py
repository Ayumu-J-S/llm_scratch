from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

import train as train_module
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

    device = torch.device("cpu")
    train_batches = list(train_module.build_streaming_dataloader(cfg, "train", device=device))
    validation_batches = list(
        train_module.build_streaming_dataloader(cfg, "validation", device=device)
    )

    assert train_batches == []
    assert len(validation_batches) == 1
    assert validation_batches[0]["inputs"].shape == (1, 3)
    assert validation_batches[0]["labels"].shape == (1, 3)


def test_cuda_request_fails_before_tokenizer_or_data(monkeypatch):
    cfg = OmegaConf.create({"runtime": {"device": "cuda"}})
    tokenizer_touched = False

    def fail_if_tokenizer_is_loaded(*args, **kwargs):
        nonlocal tokenizer_touched
        tokenizer_touched = True
        raise AssertionError("tokenizer must not be loaded")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(train_module, "load_tokenizer", fail_if_tokenizer_is_loaded)

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        train_module.main.__wrapped__(cfg)

    assert not tokenizer_touched
