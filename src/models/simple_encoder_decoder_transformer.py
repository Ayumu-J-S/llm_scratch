import torch
import torch.nn as nn

from models.embedding import TransformerEmbedding


class SimpleEncoderDecoderTransformer(nn.Module):
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
        self.source_embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            embed_size=embed_size,
            max_len=max_len,
            pad_token_id=pad_token_id,
        )
        self.target_embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            embed_size=embed_size,
            max_len=max_len,
            pad_token_id=pad_token_id,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
        )
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(
        self,
        source_tokens,
        target_tokens,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
    ):
        memory = self.encode(
            source_tokens,
            src_key_padding_mask=src_key_padding_mask,
        )
        return self.decode(
            memory=memory,
            target_tokens=target_tokens,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

    def encode(self, source_tokens, src_key_padding_mask=None):
        source_length = source_tokens.size(1)
        if source_length > self.max_len:
            raise ValueError(
                f"Source sequence length exceeds max_len={self.max_len}: "
                f"{source_length}"
            )

        if src_key_padding_mask is None:
            src_key_padding_mask = self.make_padding_mask(source_tokens)

        source_hidden = self.source_embedding(source_tokens)
        return self.encoder(
            source_hidden,
            src_key_padding_mask=src_key_padding_mask,
        )

    def decode(
        self,
        memory,
        target_tokens,
        memory_key_padding_mask=None,
        tgt_key_padding_mask=None,
    ):
        target_length = target_tokens.size(1)
        if target_length > self.max_len:
            raise ValueError(
                f"Target sequence length exceeds max_len={self.max_len}: "
                f"{target_length}"
            )

        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self.make_padding_mask(target_tokens)

        target_hidden = self.target_embedding(target_tokens)
        target_mask = self.generate_square_subsequent_mask(
            target_length,
            device=target_tokens.device,
        )

        hidden = self.decoder(
            tgt=target_hidden,
            memory=memory,
            tgt_mask=target_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
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
