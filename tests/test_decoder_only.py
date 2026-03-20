import torch

from src.datasets.text_dataset import create_autoregressive_training_data
from src.models.simple_decoder_transformer import SimpleDecoderTransformer


def test_create_autoregressive_training_data_uses_shifted_next_token_targets():
    inputs, labels = create_autoregressive_training_data(
        token_ids=[10, 11, 12, 13],
        seq_len=3,
        bos_token_id=1,
        eos_token_id=2,
    )

    assert inputs.tolist() == [
        [1, 10, 11],
        [10, 11, 12],
        [11, 12, 13],
    ]
    assert labels.tolist() == [
        [10, 11, 12],
        [11, 12, 13],
        [12, 13, 2],
    ]



def test_simple_decoder_transformer_returns_vocab_logits():
    model = SimpleDecoderTransformer(
        vocab_size=32,
        embed_size=16,
        num_heads=4,
        max_len=4,
        num_layers=2,
        pad_token_id=0,
    )
    tokens = torch.tensor([[1, 5, 6, 7], [1, 8, 9, 0]])

    logits = model(tokens)

    assert logits.shape == (2, 4, 32)
