from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from models.embedding import SinusoidalPositionalEncoding
from models.simple_decoder_transformer import SimpleDecoderTransformer
from tokenizer.canonical import CanonicalTokenizer
from utils.model import get_parameter_counts


def make_tiny_model(*, pad_token_id=0):
    return SimpleDecoderTransformer(
        vocab_size=8,
        embed_size=16,
        num_heads=4,
        max_len=6,
        num_layers=1,
        dropout=0.0,
        dim_feedforward=32,
        pad_token_id=pad_token_id,
    )


def test_output_shape_and_maximum_context():
    model = make_tiny_model()
    tokens = torch.tensor(
        [
            [1, 2, 3, 4, 1, 2],
            [2, 3, 4, 1, 2, 3],
            [3, 4, 1, 2, 3, 4],
            [4, 1, 2, 3, 4, 1],
        ]
    )

    assert model(tokens).shape == (4, 6, 8)

    with pytest.raises(ValueError, match="Sequence length exceeds max_len=6"):
        model(torch.ones((1, 7), dtype=torch.long))


def test_future_tokens_do_not_change_prefix_logits():
    torch.manual_seed(17)
    model = make_tiny_model().eval()
    first = torch.tensor([[1, 2, 3, 4, 1, 2]])
    second = torch.tensor([[1, 2, 3, 4, 4, 1]])

    with torch.no_grad():
        first_logits = model(first)
        second_logits = model(second)

    torch.testing.assert_close(
        first_logits[:, :4],
        second_logits[:, :4],
        atol=1e-6,
        rtol=1e-5,
    )
    assert not torch.allclose(
        first_logits[:, 4:],
        second_logits[:, 4:],
        atol=1e-6,
        rtol=1e-5,
    )


def test_supported_batch_has_finite_loss_and_gradients():
    torch.manual_seed(17)
    model = make_tiny_model()
    tokens = torch.tensor([[1, 2, 3, 4, 1, 2], [4, 3, 2, 1, 4, 3]])
    labels = torch.tensor([[2, 3, 4, 1, 2, 3], [3, 2, 1, 4, 3, 2]])

    loss = F.cross_entropy(model(tokens).flatten(0, 1), labels.flatten())
    loss.backward()

    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        assert gradient is not None, f"missing gradient for {name}"
        assert torch.isfinite(gradient).all(), f"non-finite gradient for {name}"
        assert torch.count_nonzero(gradient) > 0, f"zero gradient tensor for {name}"


def test_padding_is_inferred_and_padded_query_logits_are_zero():
    torch.manual_seed(17)
    model = make_tiny_model().eval()
    unpadded = torch.tensor([[1, 2, 3]])
    padded = torch.tensor([[1, 2, 3, 0, 0, 0]])

    with torch.no_grad():
        unpadded_logits = model(unpadded)
        padded_logits = model(padded)

    torch.testing.assert_close(padded_logits[:, :3], unpadded_logits)
    torch.testing.assert_close(padded_logits[:, 3:], torch.zeros(1, 3, 8))


def test_all_padding_rows_are_finite_and_zero():
    model = make_tiny_model().eval()

    with torch.no_grad():
        logits = model(torch.zeros((2, 6), dtype=torch.long))

    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits, torch.zeros(2, 6, 8))


def test_padded_objective_is_finite_and_pad_embedding_gradient_is_zero():
    torch.manual_seed(17)
    model = make_tiny_model()
    tokens = torch.tensor([[1, 2, 3, 0, 0, 0], [4, 1, 2, 3, 0, 0]])
    labels = torch.tensor([[2, 3, 4, 0, 0, 0], [1, 2, 3, 4, 0, 0]])

    logits = model(tokens)
    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=0)
    loss.backward()

    assert torch.isfinite(loss)
    pad_gradient = model.embedding.token.embedding.weight.grad[0]
    assert torch.equal(pad_gradient, torch.zeros_like(pad_gradient))


def test_external_padding_mask_injection_is_rejected():
    model = make_tiny_model()
    tokens = torch.tensor([[1, 2, 3, 4, 1, 2]])
    external_mask = torch.zeros_like(tokens, dtype=torch.bool)

    with pytest.raises(TypeError, match="unexpected keyword argument 'key_padding_mask'"):
        model(tokens, key_padding_mask=external_mask)


@pytest.mark.parametrize(
    ("tokens", "exception", "message"),
    [
        (torch.ones(6, dtype=torch.long), ValueError, "tokens must be rank 2"),
        (torch.empty((0, 6), dtype=torch.long), ValueError, "tokens must contain"),
        (torch.empty((1, 0), dtype=torch.long), ValueError, "tokens must contain"),
        (torch.ones((1, 6)), TypeError, "tokens must have dtype torch.long"),
        ([[1, 2, 3]], TypeError, "tokens must be a torch.Tensor"),
    ],
)
def test_bad_token_inputs_fail_clearly(tokens, exception, message):
    model = make_tiny_model()

    with pytest.raises(exception, match=message):
        model(tokens)


@pytest.mark.parametrize("pad_token_id", [True, 0.0, "0"])
def test_pad_token_id_must_be_a_non_boolean_integer(pad_token_id):
    with pytest.raises(TypeError, match="pad_token_id must be an integer or None"):
        make_tiny_model(pad_token_id=pad_token_id)


@pytest.mark.parametrize("pad_token_id", [-1, 8])
def test_pad_token_id_must_be_inside_vocabulary(pad_token_id):
    with pytest.raises(ValueError, match=r"pad_token_id must be in \[0, vocab_size\)"):
        make_tiny_model(pad_token_id=pad_token_id)


def test_tiny_parameter_count_matches_independent_oracle():
    model = make_tiny_model()
    raw_total = sum(parameter.numel() for parameter in model.parameters())
    counts = get_parameter_counts(model)

    assert raw_total == 2_488
    assert counts.total == raw_total
    assert counts.trainable == raw_total


def test_canonical_model_parameter_count_and_padding_match_tokenizer():
    root_dir = Path(__file__).resolve().parents[1]
    tokenizer_config = OmegaConf.load(root_dir / "config/tokenizer/canonical.yaml")
    tokenizer = CanonicalTokenizer.from_config(tokenizer_config)
    model = SimpleDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_size=384,
        num_heads=6,
        max_len=64,
        num_layers=6,
        dropout=0.1,
        pad_token_id=tokenizer.pad_token_id,
    )
    raw_total = sum(parameter.numel() for parameter in model.parameters())
    counts = get_parameter_counts(model)

    assert tokenizer.vocab_size == 50_570
    assert tokenizer.pad_token_id == 4
    assert raw_total == 10_646_784 + tokenizer.vocab_size * 769
    assert raw_total == 49_535_114
    assert counts.total == raw_total
    assert counts.trainable == raw_total
    assert model.embedding.token.embedding.padding_idx == tokenizer.pad_token_id


def test_conventional_architecture_contract_remains_explicit_and_untied():
    model = SimpleDecoderTransformer(
        vocab_size=8,
        embed_size=16,
        num_heads=4,
        max_len=6,
        num_layers=2,
        dropout=0.0,
        dim_feedforward=32,
        pad_token_id=0,
    )

    assert isinstance(model.embedding.position, SinusoidalPositionalEncoding)
    assert isinstance(model.lm_head, nn.Linear)
    assert model.lm_head.weight is not model.embedding.token.embedding.weight
    assert model.lm_head.weight.data_ptr() != model.embedding.token.embedding.weight.data_ptr()
    for layer in model.layers:
        assert isinstance(layer.self_attention, nn.MultiheadAttention)
        assert isinstance(layer.attention_norm, nn.LayerNorm)
        assert isinstance(layer.feedforward_norm, nn.LayerNorm)
        assert isinstance(layer.feedforward[1], nn.GELU)


class _ZeroAttention(nn.Module):
    def forward(self, query, key, value, **kwargs):
        del key, value, kwargs
        return torch.zeros_like(query), None


class _ZeroFeedforward(nn.Module):
    def forward(self, hidden):
        return torch.zeros_like(hidden)


def test_decoder_block_residual_paths_preserve_hidden_when_sublayers_are_zero():
    block = make_tiny_model().layers[0]
    block.self_attention = _ZeroAttention()
    block.attention_norm = nn.Identity()
    block.feedforward = _ZeroFeedforward()
    block.feedforward_norm = nn.Identity()
    hidden = torch.randn(2, 6, 16)
    causal_mask = torch.zeros(6, 6, dtype=torch.bool)

    output = block(hidden, causal_mask)

    torch.testing.assert_close(output, hidden)


def test_fixed_tiny_batch_overfits_within_predeclared_budget():
    torch.manual_seed(17)
    model = make_tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02, weight_decay=0.0)
    inputs = torch.tensor(
        [
            [1, 2, 3, 4, 1, 2],
            [2, 3, 4, 1, 2, 3],
            [3, 4, 1, 2, 3, 4],
            [4, 1, 2, 3, 4, 1],
        ]
    )
    labels = torch.tensor(
        [
            [2, 3, 4, 1, 2, 3],
            [3, 4, 1, 2, 3, 4],
            [4, 1, 2, 3, 4, 1],
            [1, 2, 3, 4, 1, 2],
        ]
    )
    losses = []

    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs).flatten(0, 1), labels.flatten())
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())

    assert losses[-1] < losses[0]
    assert losses[-1] <= 0.02
