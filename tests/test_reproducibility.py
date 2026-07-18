import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import hydra

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
from training.checkpoint import build_checkpoint_identity
from data.text_dataset import create_autoregressive_dataloader
import train as train_module


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
        losses.append(
            torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), batch["labels"].flatten()
            ).item()
        )
    return batches, losses


def test_same_seed_reproduces_initial_batches_and_loss_sequence():
    first_batches, first_losses = _fixture_trace(123)
    second_batches, second_losses = _fixture_trace(123)
    assert len(first_batches) == len(second_batches)
    for first, second in zip(first_batches, second_batches):
        assert torch.equal(first[0], second[0])
        assert torch.equal(first[1], second[1])
    assert first_losses == pytest.approx(second_losses, rel=0, abs=0)


def test_deterministic_toggle_is_explicit():
    seed_everything(123, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    seed_everything(123, deterministic=False)
    assert not torch.are_deterministic_algorithms_enabled()
    seed_everything(123, deterministic=True)


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
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    resolved = source_dir / "resolved_config.yaml"
    resolved.write_text("seed: 123\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_path = write_run_manifest(
        cfg=_manifest_config(),
        run_dir=run_dir,
        root_dir=ROOT,
        resolved_config_path=resolved,
        tokenizer_manifest_path=TOKENIZER,
        tokenizer_expected_fingerprint=TOKENIZER_FINGERPRINT,
    )
    payload = verify_run_manifest(run_dir)
    assert run_path.name == "run_manifest.json"
    assert payload["experiment_id"].startswith("exp-")
    assert (run_dir / "resolved_config.yaml").is_file()
    assert (run_dir / "tokenizer_manifest.json").is_file()
    assert (run_dir / "data_manifest_0.json").is_file()

    captured = run_dir / "tokenizer_manifest.json"
    captured.write_text(captured.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ManifestMismatchError, match="captured input manifest changed"):
        verify_run_manifest(run_dir)


def test_resume_path_is_the_only_normalized_run_identity_config_delta(monkeypatch, tmp_path):
    config = _manifest_config()
    config["artifacts"] = {
        "checkpoints_dir": "checkpoints",
        "keep_last_n": 2,
        "resume_path": None,
    }
    resumed = deepcopy(config)
    resumed["artifacts"]["resume_path"] = "/tmp/previous/recovery-step-000000000010.pt"
    monkeypatch.setattr(
        "runtime.reproducibility._git",
        lambda root: {"sha": "a" * 40, "dirty": False, "status": []},
    )
    first_source = tmp_path / "first-source"
    resumed_source = tmp_path / "resumed-source"
    first_source.mkdir()
    resumed_source.mkdir()
    first_resolved = first_source / "resolved_config.yaml"
    resumed_resolved = resumed_source / "resolved_config.yaml"
    first_resolved.write_text("artifacts:\n  resume_path: null\n", encoding="utf-8")
    resumed_resolved.write_text(
        "artifacts:\n  resume_path: /tmp/previous/recovery-step-000000000010.pt\n",
        encoding="utf-8",
    )
    first_run = tmp_path / "first-run"
    resumed_run = tmp_path / "resumed-run"
    first_manifest_path = write_run_manifest(
        cfg=config,
        run_dir=first_run,
        root_dir=ROOT,
        resolved_config_path=first_resolved,
        tokenizer_manifest_path=TOKENIZER,
        tokenizer_expected_fingerprint=TOKENIZER_FINGERPRINT,
    )
    resumed_manifest_path = write_run_manifest(
        cfg=resumed,
        run_dir=resumed_run,
        root_dir=ROOT,
        resolved_config_path=resumed_resolved,
        tokenizer_manifest_path=TOKENIZER,
        tokenizer_expected_fingerprint=TOKENIZER_FINGERPRINT,
    )
    first_manifest = json.loads(first_manifest_path.read_text(encoding="utf-8"))
    resumed_manifest = json.loads(resumed_manifest_path.read_text(encoding="utf-8"))

    assert first_manifest["config"]["sha256"] != resumed_manifest["config"]["sha256"]
    assert first_manifest["experiment_id"] == resumed_manifest["experiment_id"]
    assert build_checkpoint_identity(config, run_manifest_path=first_manifest_path) == (
        build_checkpoint_identity(resumed, run_manifest_path=resumed_manifest_path)
    )


def test_verify_run_manifest_rejects_dirty_source_worktree(monkeypatch, tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    resolved = source_dir / "resolved_config.yaml"
    resolved.write_text("seed: 123\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    clean_git = {"sha": "a" * 40, "dirty": False, "status": []}
    dirty_git = {"sha": "a" * 40, "dirty": True, "status": [" M src/train.py"]}
    monkeypatch.setattr("runtime.reproducibility._git", lambda root: clean_git)
    write_run_manifest(
        cfg=_manifest_config(),
        run_dir=run_dir,
        root_dir=ROOT,
        resolved_config_path=resolved,
        tokenizer_manifest_path=TOKENIZER,
        tokenizer_expected_fingerprint=TOKENIZER_FINGERPRINT,
    )

    monkeypatch.setattr("runtime.reproducibility._git", lambda root: dirty_git)
    with pytest.raises(ManifestMismatchError, match="dirty state changed"):
        verify_run_manifest(run_dir, root_dir=ROOT)


def test_verify_run_manifest_accepts_matching_clean_source_worktree(monkeypatch, tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    resolved = source_dir / "resolved_config.yaml"
    resolved.write_text("seed: 123\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    clean_git = {"sha": "a" * 40, "dirty": False, "status": []}
    monkeypatch.setattr("runtime.reproducibility._git", lambda root: clean_git)
    write_run_manifest(
        cfg=_manifest_config(),
        run_dir=run_dir,
        root_dir=ROOT,
        resolved_config_path=resolved,
        tokenizer_manifest_path=TOKENIZER,
        tokenizer_expected_fingerprint=TOKENIZER_FINGERPRINT,
    )

    payload = verify_run_manifest(run_dir, root_dir=ROOT)
    assert payload["git"]["dirty"] is False


def test_real_run_rejects_mutable_remote_data():
    config = _manifest_config()
    config["profile"]["purpose"] = "pretraining"
    config["data"] = {
        "mode": "streaming",
        "streaming": {
            "train": {
                "sources": [
                    {
                        "name": "remote",
                        "type": "url_jsonl",
                        "url": "https://example.invalid/data.jsonl",
                    }
                ]
            },
            "validation": {"sources": []},
        },
    }
    with pytest.raises(ReproducibilityError, match="mutable remote"):
        validate_immutable_inputs(config, real_run=True)


def test_dirty_real_run_fails_before_tokenizer_or_data(monkeypatch):
    config_dir = ROOT / "config"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = hydra.compose(
            config_name="train",
            overrides=["profile=pretrain_streaming", "runtime.device=cpu"],
        )

    monkeypatch.setattr(
        "runtime.reproducibility._git",
        lambda root: {"sha": "a" * 40, "dirty": True, "status": ["?? sentinel"]},
    )
    tokenizer_touched = False

    def fail_if_tokenizer_is_loaded(*args, **kwargs):
        nonlocal tokenizer_touched
        tokenizer_touched = True
        raise AssertionError("dirty real run must stop before tokenizer initialization")

    monkeypatch.setattr(train_module.CanonicalTokenizer, "from_config", fail_if_tokenizer_is_loaded)
    with pytest.raises(ReproducibilityError, match="clean git worktree"):
        train_module.main.__wrapped__(config)
    assert not tokenizer_touched


def test_manifest_payload_is_valid_json():
    assert isinstance(json.loads(TOKENIZER.read_text(encoding="utf-8")), dict)
