import torch
import torch.nn as nn

from models.embedding import TransformerEmbedding


class SimpleDecoderTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        num_heads,
        max_len,
        num_layers=4,
        dropout=0.1,
        dim_feedforward=None,
        pad_token_id=0,
    ):
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = embed_size * 4

        self.max_len = max_len
        self.pad_token_id = pad_token_id
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            embed_size=embed_size,
            max_len=max_len,
            pad_token_id=pad_token_id,
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            encoder_layer=decoder_layer,
            num_layers=num_layers,
        )
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, tokens, key_padding_mask=None):
        sequence_length = tokens.size(1)
        if sequence_length > self.max_len:
            raise ValueError(
                f"Sequence length exceeds max_len={self.max_len}: {sequence_length}"
            )

        if key_padding_mask is None:
            key_padding_mask = self.make_padding_mask(tokens)

        hidden = self.embedding(tokens)
        causal_mask = self.generate_square_subsequent_mask(
            sequence_length,
            device=tokens.device,
        )
        hidden = self.decoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask,
        )
        return self.lm_head(hidden)

    def make_padding_mask(self, tokens):
        return tokens.eq(self.pad_token_id)

    @staticmethod
    def generate_square_subsequent_mask(size, device):
        return torch.triu(
            torch.ones(size, size, device=device, dtype=torch.bool),
            diagonal=1,
        )
