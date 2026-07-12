import torch
import torch.nn as nn

from models.embedding import TransformerEmbedding


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, dim_feedforward):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(embed_size)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_size, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_size),
        )
        self.feedforward_norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden, causal_mask, key_padding_mask=None):
        attention_output, _ = self.self_attention(
            hidden,
            hidden,
            hidden,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        hidden = self.attention_norm(hidden + self.dropout(attention_output))
        feedforward_output = self.feedforward(hidden)
        return self.feedforward_norm(hidden + self.dropout(feedforward_output))


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
        pad_token_id=None,
    ):
        super().__init__()

        if isinstance(pad_token_id, bool) or (
            pad_token_id is not None and not isinstance(pad_token_id, int)
        ):
            raise TypeError("pad_token_id must be an integer or None")
        if pad_token_id is not None and not 0 <= pad_token_id < vocab_size:
            raise ValueError(f"pad_token_id must be in [0, vocab_size): {pad_token_id}")

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
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size=embed_size,
                    num_heads=num_heads,
                    dropout=dropout,
                    dim_feedforward=dim_feedforward,
                )
                for _ in range(num_layers)
            ]
        )
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, tokens):
        if not isinstance(tokens, torch.Tensor):
            raise TypeError("tokens must be a torch.Tensor")
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be rank 2, got rank {tokens.ndim}")
        if tokens.numel() == 0:
            raise ValueError("tokens must contain a non-empty batch and sequence")
        if tokens.dtype != torch.long:
            raise TypeError(f"tokens must have dtype torch.long, got {tokens.dtype}")

        sequence_length = tokens.size(1)
        if sequence_length > self.max_len:
            raise ValueError(f"Sequence length exceeds max_len={self.max_len}: {sequence_length}")

        key_padding_mask = self.make_padding_mask(tokens)
        attention_padding_mask = self.make_attention_padding_mask(key_padding_mask)

        hidden = self.embedding(tokens)
        if key_padding_mask is not None:
            hidden = hidden.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        causal_mask = self.generate_square_subsequent_mask(
            sequence_length,
            device=tokens.device,
        )
        for layer in self.layers:
            hidden = layer(
                hidden,
                causal_mask=causal_mask,
                key_padding_mask=attention_padding_mask,
            )
            if key_padding_mask is not None:
                hidden = hidden.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        logits = self.lm_head(hidden)
        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return logits

    def make_padding_mask(self, tokens):
        if self.pad_token_id is None:
            return None
        return tokens.eq(self.pad_token_id)

    @staticmethod
    def make_attention_padding_mask(key_padding_mask):
        if key_padding_mask is None:
            return None

        all_padding = key_padding_mask.all(dim=1, keepdim=True)
        sentinel = torch.zeros_like(key_padding_mask)
        sentinel[:, :1] = True
        return key_padding_mask & ~(all_padding & sentinel)

    @staticmethod
    def generate_square_subsequent_mask(size, device):
        return torch.triu(
            torch.ones(size, size, device=device, dtype=torch.bool),
            diagonal=1,
        )
