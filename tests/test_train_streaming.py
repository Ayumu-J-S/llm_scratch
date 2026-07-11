from __future__ import annotations

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

    train_batches = list(train_module.build_streaming_dataloader(cfg, "train"))
    validation_batches = list(train_module.build_streaming_dataloader(cfg, "validation"))

    assert train_batches == []
    assert len(validation_batches) == 1
    assert validation_batches[0]["inputs"].shape == (1, 3)
    assert validation_batches[0]["labels"].shape == (1, 3)
