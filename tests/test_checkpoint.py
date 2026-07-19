from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import training.checkpoint as checkpoint_module
from training.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointManager,
    CheckpointVerificationError,
    build_checkpoint_identity,
    require_exact_stream_resume_state,
)
from training.optimization import WarmupCosineScheduler
from training.trainer import Trainer
from data.streaming_dataset import StreamingTokenDataset, causal_lm_collate_fn
from torch.utils.data import DataLoader


class InterruptedRun(RuntimeError):
    pass


class CursorDataset:
    """A small stream whose cursor and all process RNGs affect each batch."""

    def __init__(self, total_batches: int, *, vocab_size: int = 5) -> None:
        self.total_batches = total_batches
        self.vocab_size = vocab_size
        self.position = 0

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position}

    def load_state_dict(self, state: dict[str, int]) -> None:
        position = state.get("position")
        if not isinstance(position, int) or not 0 <= position <= self.total_batches:
            raise ValueError("invalid test stream cursor")
        self.position = position

    def next_batch(self) -> dict[str, torch.Tensor]:
        if self.position >= self.total_batches:
            raise StopIteration
        self.position += 1
        # Exercise the Python, NumPy, and Torch process RNG states that a
        # full-state checkpoint must restore before consuming the suffix.
        target = (
            random.randrange(self.vocab_size)
            + int(np.random.randint(0, self.vocab_size))
            + int(torch.randint(0, self.vocab_size, ()).item())
        ) % self.vocab_size
        labels = torch.tensor([[target, (target + 1) % self.vocab_size]], dtype=torch.long)
        return {"inputs": torch.zeros_like(labels), "labels": labels}


class CursorLoader:
    def __init__(self, dataset: CursorDataset) -> None:
        self.dataset = dataset

    def __iter__(self):
        while self.dataset.position < self.dataset.total_batches:
            yield self.dataset.next_batch()


class TinyModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 5) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.randn(vocab_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(*inputs.shape, -1)


def _identity() -> dict[str, object]:
    return {
        "schema_version": 1,
        "config_sha256": "fixture-config",
        "model_config": {"vocab_size": 5},
        "tokenizer_fingerprint": "fixture-tokenizer",
        "data_fingerprints": ["fixture-data"],
        "experiment_id": "fixture-run",
    }


def _state(step: int) -> dict[str, object]:
    return {"counters": {"optimizer_step": step, "target_tokens": step * 2, "elapsed_seconds": 0.0}}


def _trainer(
    directory: Path,
    *,
    resume_path: str | Path | None = None,
    measurement: dict[str, object] | None = None,
) -> Trainer:
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, betas=(0.9, 0.95))
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=1,
        decay_steps=6,
        min_lr_ratio=0.1,
    )
    cfg = OmegaConf.create(
        {
            "training": {
                "epochs": 1,
                "batch_size": 1,
                "sequence_length": 2,
                "max_steps": 6,
                "max_tokens": None,
                "max_time": None,
                "precision": "fp32",
                "gradient_accumulation_steps": 1,
                "max_grad_norm": None,
                "log_every_n_steps": 1,
                "validation_every_n_steps": None,
                "checkpoint_every_n_steps": 1,
                "milestone_every_n_steps": 2,
                "scheduler": {"interval": "step"},
            },
            "artifacts": {"checkpoints_dir": "checkpoints", "keep_last_n": 2, "resume_path": None},
            "wandb": {"mode": "disabled"},
            "measurement": measurement or {"enabled": False},
        }
    )
    dataset = CursorDataset(total_batches=6)
    return Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=CursorLoader(dataset),
        validation_loader_factory=lambda: CursorLoader(CursorDataset(total_batches=1)),
        checkpoint_dir=directory,
        cfg=cfg,
        device=torch.device("cpu"),
        checkpoint_identity=build_checkpoint_identity(cfg),
        resume_path=resume_path,
    )


def _seed_all() -> None:
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)


def test_interrupted_resume_matches_uninterrupted_model_optimizer_scheduler_and_suffix(
    tmp_path: Path,
):
    _seed_all()
    uninterrupted = _trainer(tmp_path / "uninterrupted")
    uninterrupted.fit()

    _seed_all()
    interrupted_dir = tmp_path / "interrupted"
    interrupted = _trainer(interrupted_dir)
    original_events = interrupted._run_events

    def stop_after_verified_recovery(*, epoch_end: bool, train_loss: float | None = None) -> None:
        original_events(epoch_end=epoch_end, train_loss=train_loss)
        if interrupted.optimizer_step == 3 and not epoch_end:
            raise InterruptedRun("simulated process interruption after verified step 3")

    interrupted._run_events = stop_after_verified_recovery  # type: ignore[method-assign]
    with pytest.raises(InterruptedRun, match="verified step 3"):
        interrupted.fit()
    assert (interrupted_dir / "recovery-step-000000000003.pt").is_file()

    # A fresh process starts from the original seeds and builds new objects;
    # Trainer must restore the checkpoint RNG/cursor only after construction.
    _seed_all()
    resumed = _trainer(interrupted_dir, resume_path="latest")
    resumed.fit()

    assert resumed.optimizer_step == uninterrupted.optimizer_step == 6
    assert resumed.target_tokens == uninterrupted.target_tokens
    assert resumed.elapsed_seconds >= 0.0
    assert resumed.scheduler.state_dict() == uninterrupted.scheduler.state_dict()
    for resumed_parameter, full_parameter in zip(
        resumed.model.parameters(), uninterrupted.model.parameters()
    ):
        assert torch.equal(resumed_parameter, full_parameter)
    resumed_steps = [item for item in resumed.metrics if item.get("event") == "step"]
    full_steps = [item for item in uninterrupted.metrics if item.get("event") == "step"]
    assert [item["train/loss_step"] for item in resumed_steps] == pytest.approx(
        [item["train/loss_step"] for item in full_steps[3:]], rel=0, abs=0
    )


def test_cross_directory_resume_recovers_or_explicitly_selects_prior_best(tmp_path: Path):
    source_dir = tmp_path / "source"
    source = _trainer(source_dir)
    source.optimizer_step = 1
    source.target_tokens = 2
    source._best_validation_loss = -1.0
    source._best_checkpoint_step = 1
    best = source._save_best_checkpoint()
    source._best_checkpoint_path = best
    source.optimizer_step = 2
    source.target_tokens = 4
    recovery = source._save_checkpoint()

    recovered = _trainer(tmp_path / "recovered", resume_path=recovery)
    assert recovered._best_checkpoint_path == best
    assert recovered._expected_best_checkpoint_path == best

    best.unlink()
    missing = _trainer(tmp_path / "missing", resume_path=recovery)
    missing.cfg.wandb.artifact = {"policy": "best"}

    class RecordingArtifacts:
        def __init__(self) -> None:
            self.decisions = []

        def reset_local_evidence(self) -> None:
            pass

        def start(self, model) -> None:
            pass

        def log(self, values) -> None:
            pass

        def update_summary(self, values) -> None:
            pass

        def consider_artifact(self, **decision):
            self.decisions.append(decision)
            return {"outcome": "blocked", "block_reason": "checkpoint_identity_unavailable"}

        def finish(self) -> None:
            pass

    tracking = RecordingArtifacts()
    missing.wandb = tracking
    missing.fit()

    assert missing._best_checkpoint_path is None
    assert missing._expected_best_checkpoint_path == best
    assert tracking.decisions == [{"reason": "best", "checkpoint_path": best, "step": 1}]


def test_measurement_evidence_preserves_verified_segments_across_resume(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    first_path = tmp_path / "first-measurement.json"
    resumed_path = tmp_path / "resumed-measurement.json"
    measurement = {
        "enabled": True,
        "warmup_optimizer_steps": 1,
        "cuda_events": False,
        "output_path": str(first_path),
    }
    _seed_all()
    interrupted = _trainer(checkpoint_dir, measurement=measurement)
    original_events = interrupted._run_events

    def stop_after_verified_recovery(*, epoch_end: bool, train_loss: float | None = None) -> None:
        original_events(epoch_end=epoch_end, train_loss=train_loss)
        if interrupted.optimizer_step == 3 and not epoch_end:
            raise InterruptedRun("simulated interruption with measurement evidence")

    interrupted._run_events = stop_after_verified_recovery  # type: ignore[method-assign]
    with pytest.raises(InterruptedRun, match="measurement evidence"):
        interrupted.fit()

    first_payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert first_payload["complete"] is False
    assert len(first_payload["segments"]) == 1
    first_segment = first_payload["segments"][0]
    assert first_segment["complete"] is False
    assert first_segment["end_counters"]["optimizer_step"] == 3

    resumed_measurement = {
        "enabled": True,
        "warmup_optimizer_steps": 0,
        "cuda_events": False,
        "output_path": str(resumed_path),
    }
    _seed_all()
    resumed = _trainer(
        checkpoint_dir,
        resume_path="latest",
        measurement=resumed_measurement,
    )
    resumed.fit()

    payload = json.loads(resumed_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert payload["complete"] is True
    assert payload["segments"][0] == first_segment
    assert len(payload["segments"]) == 2
    suffix = payload["segments"][1]
    assert suffix["segment_index"] == 1
    assert suffix["start_counters"]["optimizer_step"] == 3
    assert suffix["end_counters"]["optimizer_step"] == 6
    assert suffix["resumed_from"]["counters"] == suffix["start_counters"]
    assert suffix["resumed_from"]["prior_measurement"] == {
        "status": "verified",
        "path": str(first_path),
    }
    assert suffix["complete"] is True
    suffix_steps = [
        row["optimizer_step"] for row in suffix["rows"] if row["event"] == "optimizer_step"
    ]
    assert suffix_steps == [4, 5, 6]
    assert json.loads(first_path.read_text(encoding="utf-8")) == first_payload


@pytest.mark.parametrize(
    ("save_method", "kind"),
    [
        ("_save_checkpoint", "recovery"),
        ("_save_best_checkpoint", "best"),
        ("_save_final_checkpoint", "final"),
        ("_save_milestone_checkpoint", "milestone"),
    ],
)
def test_every_checkpoint_kind_embeds_a_durable_measurement_boundary_before_metrics(
    tmp_path: Path, save_method: str, kind: str
):
    checkpoint_dir = tmp_path / kind / "checkpoints"
    measurement_path = tmp_path / kind / "measurement.json"
    measurement = {
        "enabled": True,
        "warmup_optimizer_steps": 0,
        "cuda_events": False,
        "output_path": str(measurement_path),
    }
    _seed_all()
    trainer = _trainer(checkpoint_dir, measurement=measurement)
    trainer.optimizer_step = 1
    trainer.target_tokens = 2
    trainer.train_loader.dataset.position = 1
    trainer._initialize_measurements()

    checkpoint_path = getattr(trainer, save_method)()

    artifact = json.loads(measurement_path.read_text(encoding="utf-8"))
    boundary = artifact["checkpoint_boundaries"][-1]
    assert boundary["kind"] == kind
    assert boundary["status"] == "committed"
    assert boundary["counters"] == trainer.counters
    assert artifact["segments"][0]["rows"] == []
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["kind"] == kind
    assert checkpoint["state"]["measurement_evidence"]["checkpoint_boundary"] == {
        key: boundary[key]
        for key in (
            "boundary_index",
            "boundary_id",
            "evidence_id",
            "segment_index",
            "kind",
            "counters",
        )
    }

    _seed_all()
    resumed = _trainer(
        checkpoint_dir,
        resume_path=checkpoint_path,
        measurement=measurement,
    )
    resumed._initialize_measurements()
    assert resumed.optimizer_step == 1


def test_pending_boundary_with_matching_verified_checkpoint_survives_hard_interruption(
    tmp_path: Path, monkeypatch
):
    checkpoint_dir = tmp_path / "checkpoints"
    measurement_path = tmp_path / "measurement.json"
    measurement = {
        "enabled": True,
        "warmup_optimizer_steps": 0,
        "cuda_events": False,
        "output_path": str(measurement_path),
    }
    _seed_all()
    interrupted = _trainer(checkpoint_dir, measurement=measurement)
    interrupted.optimizer_step = 3
    interrupted.target_tokens = 6
    interrupted.train_loader.dataset.position = 3
    interrupted._initialize_measurements()
    original_flush = interrupted._flush_measurements
    flush_calls = 0

    def interrupt_after_verified_save() -> None:
        nonlocal flush_calls
        flush_calls += 1
        if flush_calls == 2:
            raise InterruptedRun("hard interruption before boundary promotion")
        original_flush()

    monkeypatch.setattr(interrupted, "_flush_measurements", interrupt_after_verified_save)
    with pytest.raises(InterruptedRun, match="boundary promotion"):
        interrupted._save_checkpoint()

    artifact = json.loads(measurement_path.read_text(encoding="utf-8"))
    assert artifact["checkpoint_boundaries"][-1]["status"] == "pending"
    checkpoint_path = checkpoint_dir / "recovery-step-000000000003.pt"
    assert checkpoint_path.is_file()

    _seed_all()
    resumed = _trainer(checkpoint_dir, resume_path="latest", measurement=measurement)
    resumed._initialize_measurements()
    assert resumed.optimizer_step == 3


def test_resume_rejects_checkpoint_payload_kind_that_differs_from_boundary(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    measurement_path = tmp_path / "measurement.json"
    measurement = {
        "enabled": True,
        "warmup_optimizer_steps": 0,
        "cuda_events": False,
        "output_path": str(measurement_path),
    }
    _seed_all()
    trainer = _trainer(checkpoint_dir, measurement=measurement)
    trainer.optimizer_step = 1
    trainer.target_tokens = 2
    trainer.train_loader.dataset.position = 1
    trainer._initialize_measurements()
    checkpoint_path = trainer._save_checkpoint()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["kind"] = "milestone"
    torch.save(checkpoint, checkpoint_path)

    _seed_all()
    with pytest.raises(CheckpointCompatibilityError, match="checkpoint kind differs"):
        _trainer(checkpoint_dir, resume_path=checkpoint_path, measurement=measurement)


def test_failed_checkpoint_save_leaves_no_accepted_boundary_and_older_resume_works(
    tmp_path: Path, monkeypatch
):
    checkpoint_dir = tmp_path / "checkpoints"
    measurement_path = tmp_path / "measurement.json"
    measurement = {
        "enabled": True,
        "warmup_optimizer_steps": 0,
        "cuda_events": False,
        "output_path": str(measurement_path),
    }
    _seed_all()
    trainer = _trainer(checkpoint_dir, measurement=measurement)
    trainer.optimizer_step = 1
    trainer.target_tokens = 2
    trainer.train_loader.dataset.position = 1
    trainer._initialize_measurements()
    older = trainer._save_checkpoint()
    trainer.optimizer_step = 2
    trainer.target_tokens = 4
    trainer.train_loader.dataset.position = 2

    monkeypatch.setattr(
        trainer.checkpoints,
        "save_recovery",
        lambda _state: (_ for _ in ()).throw(OSError("injected checkpoint write failure")),
    )
    with pytest.raises(OSError, match="injected checkpoint write failure"):
        trainer._save_checkpoint()

    artifact = json.loads(measurement_path.read_text(encoding="utf-8"))
    assert [boundary["status"] for boundary in artifact["checkpoint_boundaries"]] == [
        "committed",
        "failed",
    ]
    assert artifact["checkpoint_boundaries"][-1]["checkpoint_path"] is None
    assert list(checkpoint_dir.glob("recovery-step-*.pt")) == [older]

    _seed_all()
    resumed = _trainer(checkpoint_dir, resume_path="latest", measurement=measurement)
    resumed.fit()
    assert resumed.optimizer_step == 6
    resumed_artifact = json.loads(measurement_path.read_text(encoding="utf-8"))
    older_boundary_id = resumed_artifact["checkpoint_boundaries"][0]["boundary_id"]
    assert resumed_artifact["segments"][0]["end_counters"]["optimizer_step"] == 2
    assert resumed_artifact["segments"][1]["start_counters"]["optimizer_step"] == 1
    assert resumed_artifact["segments"][1]["parent_boundary_id"] == older_boundary_id

    _seed_all()
    second_resume = _trainer(checkpoint_dir, resume_path="latest", measurement=measurement)
    second_resume._initialize_measurements()
    assert second_resume.optimizer_step == 6


@pytest.mark.parametrize("mismatch", ["identity", "evidence_chain", "resume_boundary"])
def test_resume_rejects_mismatched_prior_measurement_evidence_before_training(
    tmp_path: Path, mismatch: str
):
    checkpoint_dir = tmp_path / "checkpoints"
    measurement_path = tmp_path / "measurement.json"
    measurement = {
        "enabled": True,
        "warmup_optimizer_steps": 0,
        "cuda_events": False,
        "output_path": str(measurement_path),
    }
    _seed_all()
    interrupted = _trainer(checkpoint_dir, measurement=measurement)
    original_events = interrupted._run_events

    def stop_after_verified_recovery(*, epoch_end: bool, train_loss: float | None = None) -> None:
        original_events(epoch_end=epoch_end, train_loss=train_loss)
        if interrupted.optimizer_step == 3 and not epoch_end:
            raise InterruptedRun("simulated interruption before evidence mismatch")

    interrupted._run_events = stop_after_verified_recovery  # type: ignore[method-assign]
    with pytest.raises(InterruptedRun, match="evidence mismatch"):
        interrupted.fit()

    payload = json.loads(measurement_path.read_text(encoding="utf-8"))
    if mismatch == "identity":
        payload["checkpoint_identity"] = {"stale": True}
        expected_error = "checkpoint identity"
    elif mismatch == "evidence_chain":
        payload["measurement_evidence_id"] = "unrelated-evidence-chain"
        expected_error = "chain identity"
    else:
        checkpoint_boundary = next(
            boundary
            for boundary in payload["checkpoint_boundaries"]
            if boundary["kind"] == "recovery" and boundary["counters"]["optimizer_step"] == 3
        )
        checkpoint_boundary["kind"] = "milestone"
        expected_error = "resume boundary"
    measurement_path.write_text(json.dumps(payload), encoding="utf-8")

    _seed_all()
    resumed = _trainer(checkpoint_dir, resume_path="latest", measurement=measurement)
    parameters_before = [parameter.detach().clone() for parameter in resumed.model.parameters()]
    with pytest.raises(ValueError, match=expected_error):
        resumed.fit()

    assert resumed.optimizer_step == 3
    assert all(
        torch.equal(before, after)
        for before, after in zip(parameters_before, resumed.model.parameters(), strict=True)
    )


def test_corrupt_newest_recovery_falls_back_to_previous_verified_checkpoint(tmp_path: Path):
    manager = CheckpointManager(tmp_path, keep_last_n=3, identity=_identity())
    first = manager.save_recovery(_state(1))
    newest = manager.save_recovery(_state(2))
    newest.write_bytes(b"not a torch checkpoint")

    resumed = manager.load_resume("latest")

    assert resumed.path == first
    assert resumed.payload["state"]["counters"]["optimizer_step"] == 1
    assert resumed.rejected_paths == (newest,)


def test_atomic_failure_preserves_verified_recovery_and_does_not_rotate(
    tmp_path: Path, monkeypatch
):
    manager = CheckpointManager(tmp_path, keep_last_n=1, identity=_identity())
    previous = manager.save_recovery(_state(1))

    def write_corrupt_temporary_payload(_payload, destination):
        Path(destination).write_bytes(b"corrupt temporary checkpoint")

    monkeypatch.setattr(checkpoint_module.torch, "save", write_corrupt_temporary_payload)
    with pytest.raises(CheckpointVerificationError):
        manager.save_recovery(_state(2))

    assert previous.is_file()
    assert not (tmp_path / "recovery-step-000000000002.pt").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_rotation_runs_only_after_verified_replacement_and_protects_retention_classes(
    tmp_path: Path,
):
    manager = CheckpointManager(tmp_path, keep_last_n=1, identity=_identity())
    old_recovery = manager.save_recovery(_state(1))
    best = manager.save_best(_state(1))
    final = manager.save_final(_state(1))
    milestone = manager.save_milestone(_state(1))

    replacement = manager.save_recovery(_state(2))

    assert replacement.is_file()
    assert not old_recovery.exists()
    assert best.is_file()
    assert final.is_file()
    assert milestone.is_file()
    assert len(list(tmp_path.glob("recovery-step-*.pt"))) == 1


def test_incompatible_identity_rejects_before_model_or_optimizer_state_is_applied(tmp_path: Path):
    writer = CheckpointManager(tmp_path, keep_last_n=2, identity=_identity())
    writer.save_recovery(_state(1))
    incompatible = {**_identity(), "model_config": {"vocab_size": 7}}

    with pytest.raises(CheckpointCompatibilityError, match="model_config"):
        CheckpointManager(tmp_path, keep_last_n=2, identity=incompatible).load_resume("latest")


def test_resume_without_a_stream_cursor_is_rejected_before_training():
    with pytest.raises(CheckpointCompatibilityError, match="cursor-aware streaming"):
        require_exact_stream_resume_state({"stream_cursor": None})


def test_resume_and_measurement_are_operational_but_model_config_remains_critical(
    tmp_path: Path,
):
    base = OmegaConf.create(
        {
            "model": {"embed_size": 8},
            "tokenizer": {"name": "fixture"},
            "data": {"mode": "memorization_smoke"},
            "artifacts": {"checkpoints_dir": "checkpoints", "keep_last_n": 2, "resume_path": None},
            "measurement": {
                "enabled": False,
                "warmup_optimizer_steps": 10,
                "cuda_events": True,
                "output_path": None,
            },
        }
    )
    resumed = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    resumed.artifacts.resume_path = "/tmp/previous/recovery-step-000000000010.pt"
    measured = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    measured.measurement.enabled = True
    measured.measurement.warmup_optimizer_steps = 3
    measured.measurement.cuda_events = False
    measured.measurement.output_path = "/tmp/measurement.json"
    changed_model = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
    changed_model.model.embed_size = 16

    assert build_checkpoint_identity(base) == build_checkpoint_identity(resumed)
    assert build_checkpoint_identity(base) == build_checkpoint_identity(measured)
    assert build_checkpoint_identity(base) != build_checkpoint_identity(changed_model)

    manager = CheckpointManager(tmp_path, keep_last_n=2, identity=build_checkpoint_identity(base))
    recovery = manager.save_recovery(_state(1))
    resumed_checkpoint = CheckpointManager(
        tmp_path, keep_last_n=2, identity=build_checkpoint_identity(measured)
    ).load_resume(recovery)
    assert resumed_checkpoint.path == recovery


def test_checkpoint_identity_requires_recorded_run_identity_fields(tmp_path: Path):
    config = OmegaConf.create(
        {
            "model": {"embed_size": 8},
            "tokenizer": {"name": "fixture"},
            "data": {"mode": "streaming"},
            "artifacts": {"checkpoints_dir": "checkpoints", "keep_last_n": 2, "resume_path": None},
        }
    )
    manifest = {
        "experiment_id": "exp-recorded-run",
        "git": {"sha": "a" * 40},
        "lock": {"sha256": "b" * 64},
        "tokenizer": {"fingerprint": "tokenizer"},
        "data": [],
    }
    first = tmp_path / "first.json"
    same = tmp_path / "same.json"
    first.write_text(json.dumps(manifest), encoding="utf-8")
    same.write_text(json.dumps(manifest), encoding="utf-8")

    baseline = build_checkpoint_identity(config, run_manifest_path=first)
    assert baseline == build_checkpoint_identity(config, run_manifest_path=same)

    for name, mutation, expected_field in (
        ("experiment", ("experiment_id", "exp-different-run"), "experiment_id"),
        ("git", ("git", {"sha": "c" * 40}), "git_sha"),
        ("lock", ("lock", {"sha256": "d" * 64}), "lock_sha256"),
    ):
        changed = dict(manifest)
        changed[mutation[0]] = mutation[1]
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        changed_identity = build_checkpoint_identity(config, run_manifest_path=path)
        assert changed_identity != baseline
        manager = CheckpointManager(tmp_path / name, keep_last_n=2, identity=baseline)
        manager.save_recovery(_state(1))
        with pytest.raises(CheckpointCompatibilityError, match=expected_field):
            CheckpointManager(
                tmp_path / name, keep_last_n=2, identity=changed_identity
            ).load_resume("latest")


def test_checkpoint_identity_reconciles_configured_manifest_order(tmp_path: Path):
    config = OmegaConf.create(
        {
            "data": {
                "mode": "streaming",
                "streaming": {
                    "train": {
                        "sources": [
                            {"name": "train", "type": "manifest", "expected_fingerprint": "a" * 64}
                        ]
                    },
                    "validation": {
                        "sources": [
                            {
                                "name": "validation",
                                "type": "manifest",
                                "expected_fingerprint": "b" * 64,
                            }
                        ]
                    },
                },
            }
        }
    )
    manifest = {
        "experiment_id": "exp-ordered-manifests",
        "git": {"sha": "a" * 40},
        "lock": {"sha256": "b" * 64},
        "tokenizer": {"fingerprint": "tokenizer"},
        "data": [{"fingerprint": "b" * 64}, {"fingerprint": "a" * 64}],
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CheckpointError, match="out of order"):
        build_checkpoint_identity(config, run_manifest_path=path)


def test_checkpoint_identity_rejects_incomplete_run_manifest(tmp_path: Path):
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps({"experiment_id": "exp-only"}), encoding="utf-8")
    with pytest.raises(CheckpointError, match="experiment_id, git.sha, and lock.sha256"):
        build_checkpoint_identity(OmegaConf.create({}), run_manifest_path=path)


def test_relative_resume_path_is_resolved_from_checkpoint_directory(tmp_path: Path):
    manager = CheckpointManager(tmp_path, keep_last_n=2, identity=_identity())
    path = manager.save_recovery(_state(1))

    assert manager.load_resume(path.name).path == path


def test_streaming_dataset_cursor_replays_exact_next_prefetched_batch():
    config = {
        "tokenizer": {
            "manifest_path": "assets/tokenizers/llm-jp-v1/manifest.json",
            "expected_fingerprint": "12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b",
        },
        "max_tokens": "max",
        "add_eos": False,
        "seed": 71,
        "horizon": {"repeat": False, "shuffle": False},
        "prefetch": {"enabled": True, "mode": "thread", "buffer_size": 3},
        "datasets": [
            {
                "name": "fixture",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": f"stream resume document {index}"} for index in range(20)],
            }
        ],
    }

    def batches(dataset: StreamingTokenDataset):
        loader = DataLoader(dataset, batch_size=2, collate_fn=causal_lm_collate_fn)
        return list(loader)

    expected = batches(StreamingTokenDataset(config, sequence_length=4))
    interrupted_dataset = StreamingTokenDataset(config, sequence_length=4)
    interrupted_loader = DataLoader(
        interrupted_dataset,
        batch_size=2,
        collate_fn=causal_lm_collate_fn,
    )
    iterator = iter(interrupted_loader)
    prefix = [next(iterator)]
    cursor = interrupted_dataset.state_dict()
    Trainer._close_train_iterator(iterator)

    resumed_dataset = StreamingTokenDataset(config, sequence_length=4)
    resumed_dataset.load_state_dict(cursor)
    actual = prefix + batches(resumed_dataset)

    assert len(actual) == len(expected)
    for observed, uninterrupted in zip(actual, expected):
        assert torch.equal(observed["inputs"], uninterrupted["inputs"])
        assert torch.equal(observed["labels"], uninterrupted["labels"])


def test_checkpoint_after_completed_stream_pass_resumes_the_next_pass(tmp_path: Path):
    config = {
        "tokenizer": {
            "manifest_path": "assets/tokenizers/llm-jp-v1/manifest.json",
            "expected_fingerprint": "12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b",
        },
        "max_tokens": "max",
        "add_eos": False,
        "seed": 73,
        "horizon": {"repeat": True, "shuffle": True, "shuffle_buffer_size": 3},
        "prefetch": {"enabled": True, "mode": "thread", "buffer_size": 3},
        "datasets": [
            {
                "name": "fixture",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": f"completed pass document {index}"} for index in range(20)],
            }
        ],
    }

    def batches(dataset: StreamingTokenDataset):
        loader = DataLoader(dataset, batch_size=2, collate_fn=causal_lm_collate_fn)
        return list(loader)

    uninterrupted = StreamingTokenDataset(config, sequence_length=4)
    completed_pass = batches(uninterrupted)
    terminal_cursor = uninterrupted.state_dict()
    expected_next_pass = batches(uninterrupted)
    assert completed_pass
    assert expected_next_pass
    assert terminal_cursor["pass_complete"] is True

    checkpoint_state = _state(1)
    checkpoint_state["stream_cursor"] = terminal_cursor
    manager = CheckpointManager(tmp_path, keep_last_n=2, identity=_identity())
    checkpoint_path = manager.save_recovery(checkpoint_state)
    restored_cursor = manager.load_resume(checkpoint_path).payload["state"]["stream_cursor"]

    resumed = StreamingTokenDataset(config, sequence_length=4)
    resumed.load_state_dict(restored_cursor)
    actual_next_pass = batches(resumed)

    assert len(actual_next_pass) == len(expected_next_pass)
    for observed, expected in zip(actual_next_pass, expected_next_pass):
        assert torch.equal(observed["inputs"], expected["inputs"])
        assert torch.equal(observed["labels"], expected["labels"])
