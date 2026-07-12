import json
from pathlib import Path

import pytest
import torch

from models.simple_decoder_transformer import SimpleDecoderTransformer
from runtime.reproducibility import (
    ManifestMismatchError,
    ReproducibilityError,
    dataloader_generator,
    dataloader_worker_init_fn,
    seed_everything,
    validate_immutable_inputs,
    verify_run_manifest,
    write_run_manifest,
)
from data.text_dataset import create_autoregressive_dataloader


ROOT = Path(__file__).parents[1]
TOKENIZER = ROOT / "assets/tokenizers/llm-jp-v1/manifest.json"
TOKENIZER_FINGERPRINT = "12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b"
DATA = ROOT / "tests/fixtures/data_manifests/memorization.manifest.json"
DATA_FINGERPRINT = "00c3797a7d0eda13950fd699a60c45fcd388829f016479caaeb369438767bd31"


def _fixture_trace(seed: int):
    seed_everything(seed)
    loader = create_autoregressive_dataloader(
        list(range(20)) * 4,
        seq_len=4,
        batch_size=3,
        shuffle=True,
        generator=dataloader_generator(seed, stream="train"),
        worker_init_fn=dataloader_worker_init_fn,
    )
    model = SimpleDecoderTransformer(
        vocab_size=20,
        embed_size=8,
        num_heads=2,
        max_len=4,
        num_layers=1,
        dropout=0.2,
    )
    losses = []
    batches = []
    for batch in loader:
        batches.append((batch["inputs"].clone(), batch["labels"].clone()))
        logits = model(batch["inputs"])
        losses.append(torch.nn.functional.cross_entropy(logits.flatten(0, 1), batch["labels"].flatten()).item())
    return batches, losses


def test_same_seed_reproduces_initial_batches_and_loss_sequence():
    first_batches, first_losses = _fixture_trace(123)
    second_batches, second_losses = _fixture_trace(123)
    assert len(first_batches) == len(second_batches)
    for first, second in zip(first_batches, second_batches):
        assert torch.equal(first[0], second[0])
        assert torch.equal(first[1], second[1])
    assert first_losses == pytest.approx(second_losses, rel=0, abs=0)


def _manifest_config():
    return {
        "profile": {"purpose": "memorization_smoke"},
        "reproducibility": {"seed": 123, "deterministic": True, "reject_dirty": True},
        "tokenizer": {"manifest_path": str(TOKENIZER)},
        "data": {
            "mode": "memorization_smoke",
            "memorization": {
                "manifest_path": str(DATA),
                "expected_fingerprint": DATA_FINGERPRINT,
            },
        },
    }


def test_run_manifest_is_self_contained_and_mutation_is_explicit(tmp_path):
    resolved = tmp_path / "resolved_config.yaml"
    resolved.write_text("seed: 123\n", encoding="utf-8")
    run_path = write_run_manifest(
        cfg=_manifest_config(),
        run_dir=tmp_path,
        root_dir=ROOT,
        resolved_config_path=resolved,
        tokenizer_manifest_path=TOKENIZER,
        tokenizer_expected_fingerprint=TOKENIZER_FINGERPRINT,
    )
    payload = verify_run_manifest(tmp_path)
    assert run_path.name == "run_manifest.json"
    assert payload["experiment_id"].startswith("exp-")
    assert (tmp_path / "tokenizer_manifest.json").is_file()
    assert (tmp_path / "data_manifest_0.json").is_file()

    captured = tmp_path / "tokenizer_manifest.json"
    captured.write_text(captured.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ManifestMismatchError, match="captured input manifest changed"):
        verify_run_manifest(tmp_path)


def test_real_run_rejects_mutable_remote_data():
    config = _manifest_config()
    config["profile"]["purpose"] = "pretraining"
    config["data"] = {
        "mode": "streaming",
        "streaming": {
            "train": {
                "sources": [{"name": "remote", "type": "url_jsonl", "url": "https://example.invalid/data.jsonl"}]
            },
            "validation": {"sources": []},
        },
    }
    with pytest.raises(ReproducibilityError, match="mutable remote"):
        validate_immutable_inputs(config, real_run=True)


def test_manifest_payload_is_valid_json():
    assert isinstance(json.loads(TOKENIZER.read_text(encoding="utf-8")), dict)
