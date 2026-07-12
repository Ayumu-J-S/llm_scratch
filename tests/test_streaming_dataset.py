from __future__ import annotations

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split

from data.streaming_dataset import (
    StreamingTokenDataset,
    causal_lm_collate_fn,
    create_streaming_token_dataloader,
)
from models.simple_decoder_transformer import SimpleDecoderTransformer
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
    "g": 8,
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
    assert batch["inputs"].tolist() == [[2, 3, 4], [5, 6, 7]]
    assert batch["labels"].tolist() == [[3, 4, 5], [6, 7, 2]]


def test_ticket_example_windows_collate_to_exact_causal_targets():
    config = {
        "max_tokens": 7,
        "add_eos": False,
        "datasets": [
            {
                "name": "docs",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "abcdefg"}],
            }
        ],
    }

    samples = list(
        StreamingTokenDataset(
            config=config,
            sequence_length=3,
            tokenizer=make_tokenizers_tokenizer(),
        )
    )
    batch = causal_lm_collate_fn(samples)

    assert [sample["input_ids"].tolist() for sample in samples] == [
        [2, 3, 4, 5],
        [5, 6, 7, 8],
    ]
    assert batch["inputs"].tolist() == [[2, 3, 4], [5, 6, 7]]
    assert batch["labels"].tolist() == [[3, 4, 5], [6, 7, 8]]


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


def test_real_streaming_batch_updates_tiny_decoder_with_finite_gradients():
    torch.manual_seed(0)
    config = {
        "max_tokens": 7,
        "add_eos": False,
        "datasets": [
            {
                "name": "docs",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "abcdefg"}],
            }
        ],
    }
    dataloader = create_streaming_token_dataloader(
        config=config,
        sequence_length=3,
        batch_size=2,
        tokenizer=make_tokenizers_tokenizer(),
    )
    model = SimpleDecoderTransformer(
        vocab_size=len(CHAR_VOCAB),
        embed_size=8,
        num_heads=2,
        max_len=3,
        num_layers=1,
        dropout=0.0,
        dim_feedforward=16,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    batch = next(iter(dataloader))
    before = model.lm_head.weight.detach().clone()

    logits = model(batch["inputs"])
    loss = F.cross_entropy(logits.flatten(0, 1), batch["labels"].flatten())
    optimizer.zero_grad()
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert torch.isfinite(loss)
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)

    optimizer.step()
    assert not torch.equal(before, model.lm_head.weight)
