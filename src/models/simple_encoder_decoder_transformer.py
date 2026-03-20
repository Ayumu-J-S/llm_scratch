import math

import torch
import torch.nn as nn


class SimpleEncoderDecoderTransformer(nn.Module):
    """
    Minimal encoder-decoder Transformer.

    Notes
    - `target_tokens` are decoder inputs, not the training labels directly.
      In teacher forcing, decoder inputs are usually:
          [BOS, y1, y2, ..., y_{n-1}]
      while labels are:
          [y1, y2, ..., y_n, EOS]
    - For padded batches, padding masks are important so attention ignores PAD tokens.
    """

    def __init__(
        self,
        vocab_size,
        embed_size,
        num_heads,
        max_len,
        num_layers=10,
        dropout=0.0,
        dim_feedforward=None,
        pad_token_id=0,
    ):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = embed_size * 4

        self.max_len = max_len
        self.pad_token_id = pad_token_id

        self.source_embedding = nn.Embedding(vocab_size, embed_size)
        self.target_embedding = nn.Embedding(vocab_size, embed_size)
        self.register_buffer("pe", self.positional_encoding(max_len, embed_size))

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
        """
        Args
            source_tokens: LongTensor of shape (batch, src_len)
            target_tokens: LongTensor of shape (batch, tgt_len)
                These are decoder inputs, typically shifted-right targets.
            src_key_padding_mask: BoolTensor of shape (batch, src_len)
                True where source token is PAD.
            tgt_key_padding_mask: BoolTensor of shape (batch, tgt_len)
                True where target token is PAD.

        Returns
            logits: FloatTensor of shape (batch, tgt_len, vocab_size)
        """
        src_len = source_tokens.size(1)
        tgt_len = target_tokens.size(1)

        if src_len > self.max_len or tgt_len > self.max_len:
            raise ValueError(
                f"Sequence length exceeds max_len={self.max_len}: "
                f"src_len={src_len}, tgt_len={tgt_len}"
            )

        if src_key_padding_mask is None:
            src_key_padding_mask = self.make_padding_mask(source_tokens)
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self.make_padding_mask(target_tokens)

        source_positions = self.pe[:src_len, :].unsqueeze(0)
        target_positions = self.pe[:tgt_len, :].unsqueeze(0)

        source_embedded = self.source_embedding(source_tokens) + source_positions
        target_embedded = self.target_embedding(target_tokens) + target_positions

        memory = self.encoder(
            source_embedded,
            src_key_padding_mask=src_key_padding_mask,
        )

        target_mask = self.generate_square_subsequent_mask(
            tgt_len,
            device=target_tokens.device,
        )

        hidden = self.decoder(
            tgt=target_embedded,
            memory=memory,
            tgt_mask=target_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return self.lm_head(hidden)

    def encode(self, source_tokens, src_key_padding_mask=None):
        src_len = source_tokens.size(1)
        if src_len > self.max_len:
            raise ValueError(
                f"Source sequence length exceeds max_len={self.max_len}: src_len={src_len}"
            )

        if src_key_padding_mask is None:
            src_key_padding_mask = self.make_padding_mask(source_tokens)

        source_positions = self.pe[:src_len, :].unsqueeze(0)
        source_embedded = self.source_embedding(source_tokens) + source_positions

        return self.encoder(
            source_embedded,
            src_key_padding_mask=src_key_padding_mask,
        )

    def decode(
        self,
        memory,
        target_tokens,
        memory_key_padding_mask=None,
        tgt_key_padding_mask=None,
    ):
        tgt_len = target_tokens.size(1)
        if tgt_len > self.max_len:
            raise ValueError(
                f"Target sequence length exceeds max_len={self.max_len}: tgt_len={tgt_len}"
            )

        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self.make_padding_mask(target_tokens)

        target_positions = self.pe[:tgt_len, :].unsqueeze(0)
        target_embedded = self.target_embedding(target_tokens) + target_positions

        target_mask = self.generate_square_subsequent_mask(
            tgt_len,
            device=target_tokens.device,
        )

        hidden = self.decoder(
            tgt=target_embedded,
            memory=memory,
            tgt_mask=target_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.lm_head(hidden)

    def make_padding_mask(self, tokens):
        return tokens.eq(self.pad_token_id)

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        return torch.triu(
            torch.ones(sz, sz, device=device, dtype=torch.bool),
            diagonal=1,
        )

    @staticmethod
    def positional_encoding(max_len, embed_size):
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_size)
        )
        pe = torch.zeros(max_len, embed_size, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe
