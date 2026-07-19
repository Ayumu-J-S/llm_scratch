import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import hydra

from models.simple_decoder_transformer import SimpleDecoderTransformer
from runtime.reproducibility import (
    ManifestMismatchError,
    ReproducibilityError,
    collect_git_identity,
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


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


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


def test_git_identity_binds_distinct_tracked_and_untracked_dirty_bytes(tmp_path: Path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.name", "Fixture")
    _git(tmp_path, "config", "user.email", "fixture@example.invalid")
    tracked = tmp_path / "evaluator.py"
    tracked.write_text("value = 'clean'\n", encoding="utf-8")
    _git(tmp_path, "add", "evaluator.py")
    _git(tmp_path, "commit", "-qm", "fixture")

    tracked.write_text("value = 'first'\n", encoding="utf-8")
    first_tracked = collect_git_identity(tmp_path)
    tracked.write_text("value = 'other'\n", encoding="utf-8")
    other_tracked = collect_git_identity(tmp_path)
    assert first_tracked["sha"] == other_tracked["sha"]
    assert first_tracked["status"] == other_tracked["status"]
    assert first_tracked["worktree_content_sha256"] != (other_tracked["worktree_content_sha256"])

    tracked.write_text("value = 'clean'\n", encoding="utf-8")
    untracked = tmp_path / "untracked_evaluator.py"
    untracked.write_text("value = 'first'\n", encoding="utf-8")
    first_untracked = collect_git_identity(tmp_path)
    untracked.write_text("value = 'other'\n", encoding="utf-8")
    other_untracked = collect_git_identity(tmp_path)
    assert first_untracked["status"] == other_untracked["status"]
    assert (
        first_untracked["worktree_content_sha256"] != (other_untracked["worktree_content_sha256"])
    )


@pytest.mark.parametrize("tracked", [False, True])
def test_git_identity_rejects_symlinked_evaluator_paths(tmp_path: Path, tracked: bool):
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Fixture")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    committed = repository / "committed.py"
    committed.write_text("value = 'clean'\n", encoding="utf-8")
    _git(repository, "add", "committed.py")
    _git(repository, "commit", "-qm", "fixture")
    target = tmp_path / "mutable-target.py"
    target.write_text("value = 'first'\n", encoding="utf-8")
    linked = repository / "linked_evaluator.py"
    linked.symlink_to(target)
    if tracked:
        _git(repository, "add", "linked_evaluator.py")
        _git(repository, "commit", "-qm", "tracked symlink")

    with pytest.raises(ReproducibilityError, match="regular file, not a symlink"):
        collect_git_identity(repository)
    target.write_text("value = 'other'\n", encoding="utf-8")
    with pytest.raises(ReproducibilityError, match="regular file, not a symlink"):
        collect_git_identity(repository)


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
    assert payload["run_lineage_id"].startswith("run-")
    assert (run_dir / "resolved_config.yaml").is_file()
    assert (run_dir / "tokenizer_manifest.json").is_file()
    assert (run_dir / "data_manifest_0.json").is_file()

    captured = run_dir / "tokenizer_manifest.json"
    captured.write_text(captured.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ManifestMismatchError, match="captured input manifest changed"):
        verify_run_manifest(run_dir)


def test_operational_controls_preserve_recipe_but_not_independent_run_lineage(
    monkeypatch, tmp_path
):
    config = _manifest_config()
    config["artifacts"] = {
        "checkpoints_dir": "checkpoints",
        "keep_last_n": 2,
        "resume_path": None,
    }
    config["measurement"] = {
        "enabled": False,
        "warmup_optimizer_steps": 10,
        "cuda_events": True,
        "output_path": None,
    }
    resumed = deepcopy(config)
    resumed["artifacts"]["resume_path"] = "/tmp/previous/recovery-step-000000000010.pt"
    resumed["measurement"] = {
        "enabled": True,
        "warmup_optimizer_steps": 3,
        "cuda_events": False,
        "output_path": "/tmp/measurement.json",
    }
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
    first_resolved.write_text(
        "artifacts:\n  resume_path: null\nmeasurement:\n  enabled: false\n",
        encoding="utf-8",
    )
    resumed_resolved.write_text(
        "artifacts:\n  resume_path: /tmp/previous/recovery-step-000000000010.pt\n"
        "measurement:\n  enabled: true\n",
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
    assert first_manifest["run_lineage_id"] != resumed_manifest["run_lineage_id"]
    assert first_manifest["experiment_identity"] == resumed_manifest["experiment_identity"]
    assert first_manifest["experiment_identity"]["operational_exclusions"] == [
        "artifacts.resume_path",
        "measurement",
        "wandb",
    ]
    assert build_checkpoint_identity(config, run_manifest_path=first_manifest_path) != (
        build_checkpoint_identity(resumed, run_manifest_path=resumed_manifest_path)
    )

    inherited_run = tmp_path / "inherited-resume-run"
    inherited_manifest_path = write_run_manifest(
        cfg=resumed,
        run_dir=inherited_run,
        root_dir=ROOT,
        resolved_config_path=resumed_resolved,
        tokenizer_manifest_path=TOKENIZER,
        tokenizer_expected_fingerprint=TOKENIZER_FINGERPRINT,
        run_lineage_id=first_manifest["run_lineage_id"],
    )
    inherited_manifest = json.loads(inherited_manifest_path.read_text(encoding="utf-8"))
    assert inherited_manifest["run_lineage_id"] == first_manifest["run_lineage_id"]
    assert build_checkpoint_identity(config, run_manifest_path=first_manifest_path) == (
        build_checkpoint_identity(resumed, run_manifest_path=inherited_manifest_path)
    )

    with pytest.raises(ReproducibilityError, match="fresh launch.*already contains"):
        write_run_manifest(
            cfg=resumed,
            run_dir=first_run,
            root_dir=ROOT,
            resolved_config_path=resumed_resolved,
            tokenizer_manifest_path=TOKENIZER,
            tokenizer_expected_fingerprint=TOKENIZER_FINGERPRINT,
        )
    retained_manifest_path = write_run_manifest(
        cfg=resumed,
        run_dir=first_run,
        root_dir=ROOT,
        resolved_config_path=resumed_resolved,
        tokenizer_manifest_path=TOKENIZER,
        tokenizer_expected_fingerprint=TOKENIZER_FINGERPRINT,
        run_lineage_id=first_manifest["run_lineage_id"],
    )
    retained_manifest = json.loads(retained_manifest_path.read_text(encoding="utf-8"))
    assert retained_manifest["run_lineage_id"] == first_manifest["run_lineage_id"]


def test_prepare_trainer_rejects_fresh_launch_into_existing_run(tmp_path: Path, monkeypatch):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        config = hydra.compose(config_name="train", overrides=["profile=smoke_overfit"])
    run_dir = tmp_path / "occupied-run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        train_module,
        "save_resolved_config",
        lambda *_args, **_kwargs: pytest.fail("occupied run must fail before evidence mutation"),
    )

    with pytest.raises(train_module.ConfigPreflightError, match="fresh launch into an occupied run"):
        train_module.prepare_trainer(config, run_dir=run_dir)


def test_prepare_trainer_rejects_concurrent_same_directory_launch(tmp_path: Path, monkeypatch):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        config = hydra.compose(config_name="train", overrides=["profile=smoke_overfit"])
    run_dir = tmp_path / "colliding-run"
    run_dir.mkdir()
    (run_dir / ".run-preparation.lock").write_bytes(b"")
    monkeypatch.setattr(
        train_module,
        "save_resolved_config",
        lambda *_args, **_kwargs: pytest.fail("colliding run must fail before evidence mutation"),
    )

    with pytest.raises(train_module.ConfigPreflightError, match="already being prepared"):
        train_module.prepare_trainer(config, run_dir=run_dir)


def test_operational_wandb_controls_do_not_change_checkpoint_compatibility():
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        disabled = hydra.compose(config_name="train", overrides=["profile=smoke_overfit"])
        online = hydra.compose(
            config_name="train",
            overrides=[
                "profile=smoke_overfit",
                "wandb.mode=online",
                "wandb.watch.enabled=true",
                "wandb.artifact.policy=final",
                "wandb.artifact.usage_snapshot_path=/operator/usage.json",
            ],
        )

    assert build_checkpoint_identity(disabled) == build_checkpoint_identity(online)


def test_verify_run_manifest_rejects_dirty_source_worktree(monkeypatch, tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    resolved = source_dir / "resolved_config.yaml"
    resolved.write_text("seed: 123\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    clean_git = {
        "sha": "a" * 40,
        "dirty": False,
        "status": [],
        "worktree_content_sha256": "1" * 64,
    }
    dirty_git = {
        "sha": "a" * 40,
        "dirty": True,
        "status": [" M src/train.py"],
        "worktree_content_sha256": "2" * 64,
    }
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
    clean_git = {
        "sha": "a" * 40,
        "dirty": False,
        "status": [],
        "worktree_content_sha256": "1" * 64,
    }
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


def test_verify_run_manifest_rejects_same_status_dirty_byte_mutation(tmp_path: Path):
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.name", "Fixture")
    _git(repository, "config", "user.email", "fixture@example.invalid")
    (repository / "uv.lock").write_bytes((ROOT / "uv.lock").read_bytes())
    tracked = repository / "evaluator.py"
    tracked.write_text("value = 'clean'\n", encoding="utf-8")
    _git(repository, "add", "uv.lock", "evaluator.py")
    _git(repository, "commit", "-qm", "fixture")

    tracked.write_text("value = 'first'\n", encoding="utf-8")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    resolved = source_dir / "resolved_config.yaml"
    resolved.write_text("seed: 123\n", encoding="utf-8")
    config = _manifest_config()
    config["reproducibility"]["reject_dirty"] = False
    run_dir = tmp_path / "run"
    write_run_manifest(
        cfg=config,
        run_dir=run_dir,
        root_dir=repository,
        resolved_config_path=resolved,
        tokenizer_manifest_path=TOKENIZER,
        tokenizer_expected_fingerprint=TOKENIZER_FINGERPRINT,
    )

    before = collect_git_identity(repository)
    tracked.write_text("value = 'other'\n", encoding="utf-8")
    after = collect_git_identity(repository)
    assert before["sha"] == after["sha"]
    assert before["status"] == after["status"]
    assert before["worktree_content_sha256"] != after["worktree_content_sha256"]
    with pytest.raises(ManifestMismatchError, match="worktree content changed"):
        verify_run_manifest(run_dir, root_dir=repository)


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
