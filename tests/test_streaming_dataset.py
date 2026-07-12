from __future__ import annotations

import torch
import torch.nn.functional as F

from data.streaming_dataset import (
    StreamingTokenDataset,
    causal_lm_collate_fn,
    create_streaming_token_dataloader,
)
from models.simple_decoder_transformer import SimpleDecoderTransformer
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
    text = "Hello, world!"
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
    expected_windows = [token_ids[:4], token_ids[3:7]]
    assert batch["inputs"].tolist() == [window[:-1] for window in expected_windows]
    assert batch["labels"].tolist() == [window[1:] for window in expected_windows]


def test_ticket_example_windows_collate_to_exact_causal_targets():
    text = "改行\nタブ\tspace"
    token_ids = TOKENIZER.encode(text)
    config = configured(
        {
            "max_tokens": 7,
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

    samples = list(
        StreamingTokenDataset(
            config=config,
            sequence_length=3,
        )
    )
    batch = causal_lm_collate_fn(samples)

    assert [sample["input_ids"].tolist() for sample in samples] == [
        token_ids[:4],
        token_ids[3:7],
    ]
    assert batch["inputs"].tolist() == [token_ids[:3], token_ids[3:6]]
    assert batch["labels"].tolist() == [token_ids[1:4], token_ids[4:7]]


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
    text = "Hello, world!"
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


def test_real_streaming_batch_updates_tiny_decoder_with_finite_gradients():
    torch.manual_seed(0)
    text = "改行\nタブ\tspace"
    config = configured(
        {
            "max_tokens": 7,
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
    dataloader = create_streaming_token_dataloader(
        config=config,
        sequence_length=3,
        batch_size=2,
    )
    model = SimpleDecoderTransformer(
        vocab_size=TOKENIZER.vocab_size,
        embed_size=8,
        num_heads=2,
        max_len=3,
        num_layers=1,
        dropout=0.0,
        dim_feedforward=16,
        pad_token_id=TOKENIZER.pad_token_id,
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
