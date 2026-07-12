from __future__ import annotations

from omegaconf import OmegaConf

import train as train_module


def test_streaming_validation_keeps_partial_batch():
    streaming_split = {
        "max_tokens": 4,
        "add_eos": False,
        "seed": 1,
        "sources": [
            {
                "name": "docs",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "Hello, world!"}],
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
            "tokenizer": {
                "manifest_path": "assets/tokenizers/llm-jp-v1/manifest.json",
                "expected_fingerprint": (
                    "12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b"
                ),
            },
        }
    )

    train_batches = list(train_module.build_streaming_dataloader(cfg, "train"))
    validation_batches = list(train_module.build_streaming_dataloader(cfg, "validation"))

    assert train_batches == []
    assert len(validation_batches) == 1
    assert validation_batches[0]["inputs"].shape == (1, 3)
    assert validation_batches[0]["labels"].shape == (1, 3)
