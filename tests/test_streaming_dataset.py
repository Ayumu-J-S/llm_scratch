from __future__ import annotations

import torch

from data.streaming_dataset import (
    StreamingTokenDataset,
    causal_lm_collate_fn,
    create_streaming_token_dataloader,
)
from tokenizer.canonical import CanonicalTokenizer


TOKENIZER_CONFIG = {
    "manifest_path": "assets/tokenizers/llm-jp-v1/manifest.json",
    "expected_fingerprint": "12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b",
}
TOKENIZER = CanonicalTokenizer.from_config(TOKENIZER_CONFIG)


def configured(config):
    config["tokenizer"] = dict(TOKENIZER_CONFIG)
    return config


def test_streaming_token_dataset_yields_only_input_ids():
    text = "日本語とEnglish。"
    config = configured(
        {
            "max_tokens": 4,
            "add_eos": False,
            "sources": [
                {
                    "name": "docs",
                    "type": "memory",
                    "ratio": 1.0,
                    "documents": [{"text": text}],
                }
            ],
        }
    )

    sample = list(
        StreamingTokenDataset(
            config=config,
            sequence_length=3,
        )
    )[0]

    assert set(sample) == {"input_ids"}
    assert sample["input_ids"].dtype == torch.long
    assert sample["input_ids"].tolist() == TOKENIZER.encode(text)[:4]


def test_causal_lm_collate_fn_shifts_input_ids():
    batch = causal_lm_collate_fn(
        [
            {"input_ids": torch.tensor([2, 3, 4, 5])},
            {"input_ids": torch.tensor([3, 4, 5, 6])},
        ]
    )

    assert batch["inputs"].tolist() == [[2, 3, 4], [3, 4, 5]]
    assert batch["labels"].tolist() == [[3, 4, 5], [4, 5, 6]]


def test_streaming_dataloader_returns_trainer_batch_shape():
    text = "記号: []{}<> /\\ +-=_"
    token_ids = TOKENIZER.encode(text)
    config = configured(
        {
            "max_tokens": len(token_ids),
            "add_eos": False,
            "datasets": [
                {
                    "name": "docs",
                    "type": "memory",
                    "ratio": 1.0,
                    "documents": [{"text": text}],
                }
            ],
        }
    )

    loader = create_streaming_token_dataloader(
        config=config,
        sequence_length=3,
        batch_size=2,
    )
    batch = next(iter(loader))

    assert batch["inputs"].shape == (2, 3)
    assert batch["labels"].shape == (2, 3)
    expected_windows = [token_ids[:4], token_ids[4:8]]
    assert batch["inputs"].tolist() == [window[:-1] for window in expected_windows]
    assert batch["labels"].tolist() == [window[1:] for window in expected_windows]


def test_streaming_token_dataset_packs_across_documents_with_eos():
    config = configured(
        {
            "max_tokens": 6,
            "add_eos": True,
            "datasets": [
                {
                    "name": "docs",
                    "type": "memory",
                    "ratio": 1.0,
                    "documents": [{"text": "ab"}, {"text": "cd"}],
                }
            ],
        }
    )

    sample = list(
        StreamingTokenDataset(
            config=config,
            sequence_length=3,
        )
    )[0]

    expected = TOKENIZER.encode("ab") + [TOKENIZER.eos_token_id]
    expected += TOKENIZER.encode("cd") + [TOKENIZER.eos_token_id]
    assert sample["input_ids"].tolist() == expected[:4]


def test_streaming_token_dataset_loads_canonical_tokenizer_from_config():
    text = "日本語とEnglish。"
    config = configured(
        {
            "max_tokens": 4,
            "add_eos": False,
            "datasets": [
                {
                    "name": "docs",
                    "type": "memory",
                    "ratio": 1.0,
                    "documents": [{"text": text}],
                }
            ],
        }
    )

    sample = list(StreamingTokenDataset(config=config, sequence_length=3))[0]

    assert sample["input_ids"].tolist() == TOKENIZER.encode(text)[:4]
