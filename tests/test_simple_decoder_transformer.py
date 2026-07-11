import pytest
import torch
import torch.nn.functional as F

from models.simple_decoder_transformer import SimpleDecoderTransformer
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
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) > 0 for gradient in gradients)


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


def test_default_parameter_count_remains_unchanged():
    model = SimpleDecoderTransformer(
        vocab_size=512,
        embed_size=384,
        num_heads=6,
        max_len=64,
        num_layers=6,
        dropout=0.1,
        pad_token_id=0,
    )
    raw_total = sum(parameter.numel() for parameter in model.parameters())
    counts = get_parameter_counts(model)

    assert raw_total == 11_040_512
    assert counts.total == raw_total
    assert counts.trainable == raw_total


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
