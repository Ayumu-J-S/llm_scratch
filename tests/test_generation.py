from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import generate
from generation.sampler import BASE_MODEL_CONTINUATION_LABEL, CheckpointSampler, SamplingError
from models.simple_decoder_transformer import SimpleDecoderTransformer
from tokenizer.canonical import CanonicalTokenizer
from training.checkpoint import CheckpointManager


TOKENIZER_CONFIG = {
    "manifest_path": "assets/tokenizers/llm-jp-v1/manifest.json",
    "expected_fingerprint": "12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b",
}


def _checkpoint(
    tmp_path: Path,
    *,
    logits: dict[int, float],
    max_len: int = 8,
) -> tuple[Path, CanonicalTokenizer]:
    tokenizer = CanonicalTokenizer.from_config(TOKENIZER_CONFIG)
    config = {
        "model": {"embed_size": 8, "num_heads": 2, "num_layers": 1, "dropout": 0.0},
        "training": {"sequence_length": max_len},
        "tokenizer": TOKENIZER_CONFIG,
    }
    identity = {
        "schema_version": 1,
        "config_sha256": "fixture-config",
        "model_config": config["model"],
        "tokenizer_fingerprint": tokenizer.fingerprint,
        "data_fingerprints": ["fixture-data"],
    }
    model = SimpleDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_size=8,
        num_heads=2,
        max_len=max_len,
        num_layers=1,
        dropout=0.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for token_id, bias in logits.items():
            model.lm_head.bias[token_id] = bias
    manager = CheckpointManager(tmp_path, keep_last_n=1, identity=identity)
    checkpoint = manager.save_final(
        {
            "model": model.state_dict(),
            "counters": {"optimizer_step": 7, "target_tokens": 19, "elapsed_seconds": 0.0},
            "resolved_config": config,
            "run_identity": identity,
        }
    )
    return checkpoint, tokenizer


def _normal_token(tokenizer: CanonicalTokenizer) -> int:
    special_ids = {
        tokenizer.unk_token_id,
        tokenizer.bos_token_id,
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
    }
    return next(
        token_id
        for token_id in tokenizer.encode("base model continuation")
        if token_id not in special_ids
    )


def test_checkpoint_round_trip_reconstructs_model_and_labels_base_continuation(tmp_path: Path):
    checkpoint, tokenizer = _checkpoint(
        tmp_path, logits={_normal_token(CanonicalTokenizer.from_config(TOKENIZER_CONFIG)): 5.0}
    )

    sampler = CheckpointSampler.from_checkpoint(checkpoint)
    result = sampler.generate("base", max_new_tokens=3)

    assert sampler.model.max_len == 8
    assert sampler.tokenizer.fingerprint == tokenizer.fingerprint
    assert result.label == BASE_MODEL_CONTINUATION_LABEL
    assert result.checkpoint_kind == "final"
    assert result.checkpoint_optimizer_step == 7
    assert result.tokenizer_fingerprint == tokenizer.fingerprint
    assert len(result.generated_token_ids) == 3
    assert result.stop_reason == "max_new_tokens"


def test_greedy_and_seeded_top_k_sampling_are_reproducible(tmp_path: Path):
    tokenizer = CanonicalTokenizer.from_config(TOKENIZER_CONFIG)
    first = _normal_token(tokenizer)
    second = next(
        token_id for token_id in tokenizer.encode("another continuation") if token_id != first
    )
    checkpoint, _ = _checkpoint(tmp_path, logits={first: 1.0, second: 0.9})
    sampler = CheckpointSampler.from_checkpoint(checkpoint)

    greedy_first = sampler.generate("base", max_new_tokens=3)
    greedy_second = sampler.generate("base", max_new_tokens=3)
    sampled_first = sampler.generate("base", max_new_tokens=5, temperature=0.7, top_k=2, seed=123)
    sampled_second = sampler.generate("base", max_new_tokens=5, temperature=0.7, top_k=2, seed=123)

    assert (
        greedy_first.generated_token_ids
        == greedy_second.generated_token_ids
        == (first, first, first)
    )
    assert sampled_first.generated_token_ids == sampled_second.generated_token_ids
    assert sampled_first.temperature == 0.7
    assert sampled_first.top_k == 2
    assert sampled_first.seed == 123


def test_eos_and_context_limits_are_enforced(tmp_path: Path):
    tokenizer = CanonicalTokenizer.from_config(TOKENIZER_CONFIG)
    eos_checkpoint, _ = _checkpoint(tmp_path / "eos", logits={tokenizer.eos_token_id: 10.0})
    eos_result = CheckpointSampler.from_checkpoint(eos_checkpoint).generate(
        "base", max_new_tokens=5
    )
    assert eos_result.generated_token_ids == (tokenizer.eos_token_id,)
    assert eos_result.stop_reason == "eos"

    prompt = "context"
    prompt_length = len(tokenizer.encode(prompt))
    continuation_token = _normal_token(tokenizer)
    context_checkpoint, _ = _checkpoint(
        tmp_path / "context", logits={continuation_token: 10.0}, max_len=prompt_length + 1
    )
    context_result = CheckpointSampler.from_checkpoint(context_checkpoint).generate(
        prompt, max_new_tokens=5
    )
    assert context_result.max_new_tokens_allowed_by_context == 1
    assert context_result.generated_token_ids == (continuation_token,)
    assert context_result.stop_reason == "context_limit"


def test_cli_outputs_metadata_without_architecture_arguments(tmp_path: Path, capsys):
    tokenizer = CanonicalTokenizer.from_config(TOKENIZER_CONFIG)
    checkpoint, _ = _checkpoint(tmp_path, logits={_normal_token(tokenizer): 5.0})

    assert (
        generate.main(
            ["--checkpoint", str(checkpoint), "--prompt", "base", "--max-new-tokens", "2", "--json"]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["label"] == BASE_MODEL_CONTINUATION_LABEL
    assert output["checkpoint_path"] == str(checkpoint.resolve())
    assert output["prompt"] == "base"
    assert output["stop_reason"] == "max_new_tokens"


def test_sampler_rejects_manual_sampling_knobs_without_a_seed(tmp_path: Path):
    tokenizer = CanonicalTokenizer.from_config(TOKENIZER_CONFIG)
    checkpoint, _ = _checkpoint(tmp_path, logits={_normal_token(tokenizer): 5.0})
    sampler = CheckpointSampler.from_checkpoint(checkpoint)

    with pytest.raises(SamplingError, match="seed is required"):
        sampler.generate("base", max_new_tokens=1, temperature=1.0)
    with pytest.raises(SamplingError, match="top_k and seed require"):
        sampler.generate("base", max_new_tokens=1, top_k=1)
