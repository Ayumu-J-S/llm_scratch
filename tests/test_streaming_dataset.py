from __future__ import annotations

import torch
from tokenizers import Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split

from data.streaming_dataset import (
    StreamingTokenDataset,
    causal_lm_collate_fn,
    create_streaming_token_dataloader,
)
from tokenizer.bpe import BPETokenizer


CHAR_VOCAB = {
    "<unk>": 0,
    "<eos>": 1,
    "a": 2,
    "b": 3,
    "c": 4,
    "d": 5,
    "e": 6,
    "f": 7,
}


def make_tokenizers_tokenizer():
    tokenizer = Tokenizer(WordLevel(CHAR_VOCAB, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")
    tokenizer.decoder = Fuse()
    return tokenizer


def test_streaming_token_dataset_yields_only_input_ids():
    config = {
        "max_tokens": 4,
        "add_eos": False,
        "sources": [
            {
                "name": "docs",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "abcd"}],
            }
        ],
    }

    sample = list(
        StreamingTokenDataset(
            config=config,
            sequence_length=3,
            tokenizer=make_tokenizers_tokenizer(),
        )
    )[0]

    assert set(sample) == {"input_ids"}
    assert sample["input_ids"].dtype == torch.long
    assert sample["input_ids"].tolist() == [2, 3, 4, 5]


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
    config = {
        "max_tokens": 8,
        "add_eos": False,
        "datasets": [
            {
                "name": "docs",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "abcdefab"}],
            }
        ],
    }

    loader = create_streaming_token_dataloader(
        config=config,
        sequence_length=3,
        batch_size=2,
        tokenizer=make_tokenizers_tokenizer(),
    )
    batch = next(iter(loader))

    assert batch["inputs"].shape == (2, 3)
    assert batch["labels"].shape == (2, 3)
    assert batch["inputs"].tolist() == [[2, 3, 4], [6, 7, 2]]
    assert batch["labels"].tolist() == [[3, 4, 5], [7, 2, 3]]


def test_streaming_token_dataset_packs_across_documents_with_eos():
    config = {
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

    sample = list(
        StreamingTokenDataset(
            config=config,
            sequence_length=3,
            tokenizer=make_tokenizers_tokenizer(),
        )
    )[0]

    assert sample["input_ids"].tolist() == [2, 3, 1, 4]


def test_streaming_token_dataset_loads_project_bpe_tokenizer_from_config(tmp_path):
    tokenizer = BPETokenizer(special_tokens=["<eos>"])
    tokenizer.train("abcd", vocab_size=5)
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    config = {
        "tokenizer": {"kind": "bpe", "path": str(tokenizer_path)},
        "max_tokens": 4,
        "add_eos": False,
        "datasets": [
            {
                "name": "docs",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "abcd"}],
            }
        ],
    }

    sample = list(StreamingTokenDataset(config=config, sequence_length=3))[0]

    assert len(sample["input_ids"]) == 4
