from __future__ import annotations

import copy
import io
import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import wandb
from wandb.integration.torch.wandb_torch import TorchHistory
from omegaconf import OmegaConf

from models.simple_decoder_transformer import SimpleDecoderTransformer
from tokenizer.canonical import CanonicalTokenizer
from training.checkpoint import (
    CheckpointManager,
    CheckpointVerificationError,
    build_checkpoint_identity,
    load_checkpoint_for_generation,
)
from training.model_artifact import (
    ARCHITECTURE_KEYS,
    COUNTER_KEYS,
    LOGICAL_IDENTITY_KEYS,
    MODEL_ARTIFACT_TOP_LEVEL_KEYS,
    MODEL_KEYS,
    SOURCE_CHECKPOINT_KEYS,
    TOKENIZER_KEYS,
    reconstruct_model_artifact,
)
from training.wandb_tracking import (
    WandbTracker,
    artifact_uploads_forbidden,
    call_bounded,
    finish_run_bounded,
    load_usage_snapshot,
    wandb_run_config,
)


class FakeSummary:
    def __init__(self) -> None:
        self.values = {}

    def update(self, values) -> None:
        self.values.update(values)


class FakeCommittedArtifact:
    def __init__(self, *, state="COMMITTED") -> None:
        self.state = state
        self.id = "artifact-id"
        self.name = "model-exp-test:v3"
        self.version = "v3"
        self.digest = "artifact-digest"

    def wait(self, timeout=None):
        assert timeout == 600
        return self


class FakeRun:
    def __init__(
        self,
        *,
        upload_error: Exception | None = None,
        artifact_result: object | None = None,
    ) -> None:
        self.id = "run-id"
        self.url = "https://wandb.example/runs/run-id"
        self.summary = FakeSummary()
        self.logs = []
        self.watches = []
        self.unwatches = []
        self.uploads = []
        self.finished = False
        self.upload_error = upload_error
        self.artifact_result = artifact_result or FakeCommittedArtifact()

    def watch(self, model, **kwargs) -> None:
        self.watches.append((model, kwargs))

    def unwatch(self, model) -> None:
        self.unwatches.append(model)

    def log(self, values) -> None:
        self.logs.append(values)

    def log_artifact(self, artifact, aliases):
        if self.upload_error is not None:
            raise self.upload_error
        self.uploads.append((artifact, aliases))
        return self.artifact_result

    def finish(self) -> None:
        self.finished = True


class FakeArtifact:
    def __init__(self, name, type, metadata):
        self.name = name
        self.type = type
        self.metadata = metadata
        self.files = []
        self.file_bytes = []

    def add_file(self, path, name, policy=None) -> None:
        self.files.append((path, name, policy))
        self.file_bytes.append(Path(path).read_bytes())


class FakeWandb:
    def __init__(
        self,
        *,
        entity: str = "research-team",
        login: bool = True,
        upload_error: Exception | None = None,
        artifact_result: object | None = None,
        string_teams: bool = False,
    ) -> None:
        self.entity = entity
        self.login_result = login
        self.run = FakeRun(upload_error=upload_error, artifact_result=artifact_result)
        self.login_calls = []
        self.init_calls = []
        self.string_teams = string_teams
        self.artifacts = []

    def Settings(self, **kwargs):
        return kwargs

    def login(self, **kwargs):
        self.login_calls.append(kwargs)
        return self.login_result

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run

    def Api(self, timeout):
        viewer = SimpleNamespace(username="operator", entity="operator", teams=[])
        if self.entity != "operator":
            viewer.teams = (
                [self.entity]
                if self.string_teams
                else [SimpleNamespace(name=self.entity, entity=self.entity)]
            )
        return SimpleNamespace(viewer=viewer)

    def Artifact(self, *args, **kwargs):
        artifact = FakeArtifact(*args, **kwargs)
        self.artifacts.append(artifact)
        return artifact


def _config(
    *,
    mode: str = "online",
    policy: str = "final",
    profile_name: str = "pretrain_streaming",
    purpose: str = "pretraining",
    snapshot_path: Path | None = None,
    reserve_bytes: int = 0,
    watch: bool = False,
):
    return OmegaConf.create(
        {
            "profile": {"name": profile_name, "purpose": purpose},
            "data": {"mode": "streaming"},
            "model": {
                "embed_size": 4,
                "num_heads": 1,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "training": {"sequence_length": 8},
            "tokenizer": {
                "manifest_path": "assets/tokenizers/llm-jp-v1/manifest.json",
                "expected_fingerprint": (
                    "12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b"
                ),
            },
            "wandb": {
                "mode": mode,
                "project": "llm-scratch",
                "entity": "research-team",
                "name": None,
                "init_timeout_seconds": 3,
                "log_timeout_seconds": 0.01,
                "watch": {"enabled": watch, "log": "gradients", "log_freq": 17},
                "artifact": {
                    "policy": policy,
                    "usage_snapshot_path": str(snapshot_path) if snapshot_path else None,
                    "max_usage_age_seconds": 900,
                    "reserve_bytes": reserve_bytes,
                    "upload_timeout_seconds": 600,
                },
            },
        }
    )


def _snapshot(
    path: Path,
    *,
    entity: str = "research-team",
    used_bytes: int = 10,
    limit_bytes: int = 100,
    captured_at: datetime | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "captured_at_utc": (captured_at or datetime.now(timezone.utc)).isoformat(),
                "entity": entity,
                "plan": "operator-visible-plan",
                "used_bytes": used_bytes,
                "limit_bytes": limit_bytes,
                "retention": "operator-visible-retention",
                "source": "W&B Billing UI export",
            }
        ),
        encoding="utf-8",
    )


def _tracker(
    tmp_path: Path,
    cfg,
    fake: FakeWandb,
    *,
    identity: dict | None = None,
) -> WandbTracker:
    checkpoint_identity = build_checkpoint_identity(cfg) if identity is None else identity
    if identity is None:
        checkpoint_identity.update(
            {
                "tokenizer_fingerprint": str(cfg.tokenizer.expected_fingerprint),
                "experiment_id": "exp-test",
                "git_sha": "a" * 40,
                "lock_sha256": "b" * 64,
            }
        )
    tracker = WandbTracker(
        cfg=cfg,
        evidence_dir=tmp_path,
        checkpoint_identity=checkpoint_identity,
        wandb_module=fake,
    )
    tracker.reset_local_evidence()
    return tracker


def _checkpoint(
    directory: Path,
    *,
    tracker: WandbTracker,
    kind: str,
    step: int,
    cfg=None,
    identity: dict | None = None,
    extra_state: dict | None = None,
) -> Path:
    resolved_cfg = tracker.cfg if cfg is None else cfg
    checkpoint_identity = tracker.checkpoint_identity if identity is None else identity
    manager = CheckpointManager(directory, keep_last_n=2, identity=checkpoint_identity)
    tokenizer = CanonicalTokenizer.from_config(resolved_cfg.tokenizer)
    model = SimpleDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_size=int(resolved_cfg.model.embed_size),
        num_heads=int(resolved_cfg.model.num_heads),
        max_len=int(resolved_cfg.training.sequence_length),
        num_layers=int(resolved_cfg.model.num_layers),
        dropout=float(resolved_cfg.model.dropout),
        pad_token_id=tokenizer.pad_token_id,
    )
    payload = {
        "model": model.state_dict(),
        "resolved_config": OmegaConf.to_container(resolved_cfg, resolve=True),
        "run_identity": dict(checkpoint_identity),
        "counters": {
            "optimizer_step": step,
            "target_tokens": step * 8,
            "elapsed_seconds": float(step),
        },
    }
    payload.update(extra_state or {})
    save = getattr(manager, f"save_{kind}")
    return save(payload)


@pytest.fixture(autouse=True)
def _controlled_ci_environment(monkeypatch):
    monkeypatch.delenv("CI", raising=False)


def test_usage_snapshot_requires_fresh_attributable_visible_values(tmp_path: Path):
    path = tmp_path / "usage.json"
    _snapshot(path)
    snapshot = load_usage_snapshot(
        path,
        expected_entity="research-team",
        max_age_seconds=900,
    )
    assert snapshot.used_bytes == 10
    assert snapshot.limit_bytes == 100

    _snapshot(path, captured_at=datetime.now(timezone.utc) - timedelta(hours=1))
    with pytest.raises(ValueError, match="stale"):
        load_usage_snapshot(path, expected_entity="research-team", max_age_seconds=900)

    _snapshot(path, entity="wrong-team")
    with pytest.raises(ValueError, match="entity"):
        load_usage_snapshot(path, expected_entity="research-team", max_age_seconds=900)


def test_bounded_call_enforces_wall_clock_timeout():
    release = threading.Event()
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="verification exceeded"):
        call_bounded(
            lambda: release.wait(1.0),
            timeout_seconds=0.01,
            operation="verification",
        )
    release.set()

    assert time.monotonic() - started < 0.2


def test_online_login_verification_is_wall_clock_bounded(tmp_path: Path):
    release = threading.Event()

    class BlockingLoginWandb(FakeWandb):
        def login(self, **kwargs):
            self.login_calls.append(kwargs)
            release.wait(1.0)
            return True

    cfg = _config()
    cfg.wandb.init_timeout_seconds = 0.01
    fake = BlockingLoginWandb()
    tracker = _tracker(tmp_path, cfg, fake)
    started = time.monotonic()

    tracker.start(object())
    release.set()

    assert time.monotonic() - started < 0.2
    assert tracker.run is None
    failure = next(
        json.loads(line)
        for line in (tmp_path / "wandb_events.jsonl").read_text(encoding="utf-8").splitlines()
        if '"action": "init"' in line and '"outcome": "failed"' in line
    )
    assert failure["error"]["type"] == "TimeoutError"


def test_wandb_failure_evidence_redacts_credentials(tmp_path: Path, monkeypatch):
    secret = "OPS_FAKE_WANDB_SECRET_SENTINEL"

    class RejectingWandb(FakeWandb):
        def login(self, **kwargs):
            self.login_calls.append(kwargs)
            raise RuntimeError(f"authentication rejected key={secret}")

    monkeypatch.setenv("WANDB_API_KEY", secret)
    tracker = _tracker(tmp_path, _config(), RejectingWandb())

    tracker.start(object())

    evidence = (tmp_path / "wandb_events.jsonl").read_text(encoding="utf-8")
    assert secret not in evidence
    assert "[REDACTED]" in evidence


def test_wandb_initialization_is_wall_clock_bounded_and_late_run_is_finished(tmp_path: Path):
    release = threading.Event()

    class BlockingInitWandb(FakeWandb):
        def init(self, **kwargs):
            self.init_calls.append(kwargs)
            release.wait(1.0)
            return self.run

    cfg = _config(mode="offline", policy="none")
    cfg.wandb.init_timeout_seconds = 0.01
    fake = BlockingInitWandb()
    tracker = _tracker(tmp_path, cfg, fake)
    started = time.monotonic()

    tracker.start(object())
    elapsed = time.monotonic() - started
    release.set()
    deadline = time.monotonic() + 1.0
    while not fake.run.finished and time.monotonic() < deadline:
        time.sleep(0.001)

    assert elapsed < 0.2
    assert tracker.run is None
    assert fake.run.finished is True
    failure = next(
        json.loads(line)
        for line in (tmp_path / "wandb_events.jsonl").read_text(encoding="utf-8").splitlines()
        if '"action": "init"' in line and '"outcome": "failed"' in line
    )
    assert failure["error"]["type"] == "TimeoutError"


def test_artifact_entity_verification_is_wall_clock_bounded(tmp_path: Path):
    release = threading.Event()

    class BlockingEntityWandb(FakeWandb):
        def Api(self, timeout):
            release.wait(1.0)
            return super().Api(timeout)

    snapshot = tmp_path / "usage.json"
    _snapshot(snapshot, limit_bytes=10_000_000)
    cfg = _config(snapshot_path=snapshot)
    cfg.wandb.init_timeout_seconds = 0.01
    fake = BlockingEntityWandb()
    tracker = _tracker(tmp_path, cfg, fake)
    tracker.start(object())
    checkpoint = _checkpoint(tmp_path / "checkpoint", tracker=tracker, kind="final", step=5)
    started = time.monotonic()

    decision = tracker.consider_artifact(reason="final", checkpoint_path=checkpoint, step=5)
    release.set()

    assert time.monotonic() - started < 0.2
    assert decision["block_reason"] == "authentication_or_entity_mismatch"
    assert decision["error"]["type"] == "TimeoutError"
    assert fake.artifacts == []


def test_wandb_config_keeps_dataset_references_but_never_inline_documents():
    cfg = _config()
    cfg.data = {
        "mode": "streaming",
        "streaming": {
            "repeat": True,
            "train": {
                "max_tokens": 100,
                "sources": [
                    {
                        "name": "fixture",
                        "type": "manifest",
                        "manifest_path": "data/manifest.json",
                        "expected_fingerprint": "a" * 64,
                        "selection": "train",
                        "documents": [{"text": "raw corpus text"}],
                        "iterable": ["raw corpus text"],
                    }
                ],
            },
            "validation": {"sources": []},
        },
    }

    safe = wandb_run_config(cfg)

    source = safe["data"]["streaming"]["train"]["sources"][0]
    assert source["manifest_path"] == "data/manifest.json"
    assert source["expected_fingerprint"] == "a" * 64
    assert "documents" not in source
    assert "iterable" not in source
    assert "raw corpus text" not in json.dumps(safe)


@pytest.mark.parametrize(
    ("mode", "policy", "reason", "profile", "purpose", "expected"),
    [
        (
            "disabled",
            "final",
            "final",
            "pretrain_streaming",
            "pretraining",
            "online_run_unavailable",
        ),
        (
            "offline",
            "final",
            "final",
            "pretrain_streaming",
            "pretraining",
            "online_run_unavailable",
        ),
        ("online", "none", "final", "pretrain_streaming", "pretraining", "policy_none"),
        ("online", "best", "final", "pretrain_streaming", "pretraining", "policy_reason_mismatch"),
        (
            "online",
            "final",
            "final",
            "smoke_overfit",
            "memorization_smoke",
            "profile_forbids_artifacts",
        ),
        ("online", "final", "final", "stability_smoke", "pretraining", "profile_forbids_artifacts"),
    ],
)
def test_artifact_policy_and_profile_matrix(
    tmp_path: Path, mode, policy, reason, profile, purpose, expected
):
    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"checkpoint")
    snapshot = tmp_path / "usage.json"
    _snapshot(snapshot)
    fake = FakeWandb()
    tracker = _tracker(
        tmp_path,
        _config(
            mode=mode,
            policy=policy,
            profile_name=profile,
            purpose=purpose,
            snapshot_path=snapshot,
        ),
        fake,
    )
    tracker.start(object())

    decision = tracker.consider_artifact(reason=reason, checkpoint_path=checkpoint, step=5)

    assert decision["outcome"] == "blocked"
    assert decision["block_reason"] == expected
    assert fake.run.uploads == []


@pytest.mark.parametrize(
    ("snapshot_kind", "expected"),
    [
        ("missing", "usage_snapshot_missing"),
        ("stale", "usage_snapshot_invalid"),
        ("quota", "visible_quota_insufficient"),
        ("entity", "usage_snapshot_invalid"),
    ],
)
def test_unknown_stale_mismatched_or_insufficient_usage_fails_closed(
    tmp_path: Path, snapshot_kind, expected
):
    snapshot = tmp_path / "usage.json"
    if snapshot_kind == "stale":
        _snapshot(snapshot, captured_at=datetime.now(timezone.utc) - timedelta(hours=1))
    elif snapshot_kind == "quota":
        _snapshot(snapshot, used_bytes=95, limit_bytes=100)
    elif snapshot_kind == "entity":
        _snapshot(snapshot, entity="wrong")
    fake = FakeWandb()
    tracker = _tracker(
        tmp_path,
        _config(snapshot_path=None if snapshot_kind == "missing" else snapshot),
        fake,
    )
    tracker.start(object())
    checkpoint = _checkpoint(tmp_path / "checkpoints", tracker=tracker, kind="final", step=5)

    decision = tracker.consider_artifact(reason="final", checkpoint_path=checkpoint, step=5)

    assert decision["block_reason"] == expected
    assert fake.artifacts == []
    assert fake.run.uploads == []


@pytest.mark.parametrize("kind", ["best", "final", "milestone"])
def test_verified_login_entity_quota_upload_and_duplicate_are_recorded(tmp_path: Path, kind: str):
    snapshot = tmp_path / "usage.json"
    _snapshot(snapshot, used_bytes=10, limit_bytes=10_000_000)
    fake = FakeWandb()
    tracker = _tracker(
        tmp_path,
        _config(policy=kind, snapshot_path=snapshot, reserve_bytes=20),
        fake,
    )
    model = object()
    tracker.start(model)
    checkpoint = _checkpoint(tmp_path / "checkpoints", tracker=tracker, kind=kind, step=5)

    first = tracker.consider_artifact(reason=kind, checkpoint_path=checkpoint, step=5)

    assert fake.login_calls == [{"force": True, "verify": True, "timeout": 3}]
    assert first["outcome"] == "uploaded"
    staged_path, staged_name, staged_policy = fake.artifacts[0].files[0]
    assert Path(staged_path) != checkpoint
    assert not Path(staged_path).exists()
    assert staged_name == "model.pt"
    assert staged_policy == "immutable"
    assert first["checkpoint"]["checkpoint_kind"] == kind
    assert first["checkpoint"]["checkpoint_optimizer_step"] == 5
    assert first["checkpoint"]["size_bytes"] == checkpoint.stat().st_size
    assert first["checkpoint"]["sha256"]
    assert first["model_artifact"]["schema_version"] == 1
    assert first["model_artifact"]["sha256"] != first["checkpoint"]["sha256"]
    assert first["model_artifact"]["size_bytes"] <= first["checkpoint"]["size_bytes"]
    assert fake.artifacts[0].metadata["source_checkpoint_sha256"] == first["checkpoint"]["sha256"]
    assert fake.artifacts[0].metadata["model_artifact_sha256"] == first["model_artifact"]["sha256"]
    assert first["auth"]["outcome"] == "verified"
    assert first["projected_bytes"] == 10 + checkpoint.stat().st_size + 20
    assert first["retry_outcome"] == "not_needed"
    assert first["artifact"] == {
        "id": "artifact-id",
        "name": "model-exp-test:v3",
        "version": "v3",
        "digest": "artifact-digest",
        "aliases": [kind, "step-5", "latest"],
    }
    assert fake.run.summary.values["tracking/run_id"] == "run-id"
    assert fake.run.summary.values["runtime/architecture"]
    assert fake.run.summary.values["runtime/torch_version"]
    assert "runtime/container_image" in fake.run.summary.values
    assert fake.run.summary.values["artifact/id"] == "artifact-id"
    assert fake.run.summary.values["artifact/checkpoint_sha256"] == first["checkpoint"]["sha256"]
    assert fake.run.summary.values["artifact/model_sha256"] == first["model_artifact"]["sha256"]

    second = tracker.consider_artifact(reason=kind, checkpoint_path=checkpoint, step=5)
    tracker.finish()

    assert second["block_reason"] == "duplicate_checkpoint"
    assert len(fake.run.uploads) == 1
    assert fake.run.finished is True

    events = [
        json.loads(line)
        for line in (tmp_path / "wandb_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    init = next(event for event in events if event["action"] == "init")
    uploaded = next(
        event
        for event in events
        if event["action"] == "artifact" and event["outcome"] == "uploaded"
    )
    assert init["run_id"] == "run-id"
    assert init["run_url"] == "https://wandb.example/runs/run-id"
    assert uploaded["artifact"]["id"] == "artifact-id"


def test_uploaded_model_artifact_is_strict_private_and_reconstructs_inference(
    tmp_path: Path,
):
    raw_text = "RAW-DOCUMENT-SENTINEL-DO-NOT-UPLOAD"
    raw_metadata = "RAW-METADATA-SENTINEL-DO-NOT-UPLOAD"
    snapshot = tmp_path / "usage.json"
    _snapshot(snapshot, limit_bytes=20_000_000)
    cfg = _config(snapshot_path=snapshot)
    cfg.data.streaming = {
        "train": {
            "sources": [
                {
                    "name": "fixture",
                    "type": "manifest",
                    "manifest_path": "data/fixture.json",
                    "expected_fingerprint": "c" * 64,
                    "selection": "train",
                    "documents": [{"text": raw_text, "metadata": raw_metadata}],
                }
            ]
        },
        "validation": {"sources": []},
    }
    fake = FakeWandb()
    tracker = _tracker(tmp_path, cfg, fake)
    tracker.start(object())
    checkpoint = _checkpoint(
        tmp_path / "checkpoints",
        tracker=tracker,
        kind="final",
        step=5,
        extra_state={
            "optimizer": {"private": raw_text},
            "scheduler": {"private": raw_metadata},
            "rng": {"private": raw_text},
            "stream_cursor": {
                "shuffle_buffer": [{"text": raw_text, "metadata": {"private": raw_metadata}}]
            },
            "arbitrary_payload": {"private": raw_metadata},
        },
    )

    decision = tracker.consider_artifact(reason="final", checkpoint_path=checkpoint, step=5)

    assert decision["outcome"] == "uploaded"
    uploaded_bytes = fake.artifacts[0].file_bytes[0]
    assert raw_text.encode() not in uploaded_bytes
    assert raw_metadata.encode() not in uploaded_bytes
    payload = torch.load(io.BytesIO(uploaded_bytes), map_location="cpu", weights_only=True)
    assert set(payload) == MODEL_ARTIFACT_TOP_LEVEL_KEYS
    assert set(payload["model"]) == MODEL_KEYS
    assert set(payload["model"]["architecture"]) == ARCHITECTURE_KEYS
    assert set(payload["tokenizer"]) == TOKENIZER_KEYS
    assert set(payload["source_checkpoint"]) == SOURCE_CHECKPOINT_KEYS
    assert set(payload["source_checkpoint"]["logical_identity"]) == LOGICAL_IDENTITY_KEYS
    assert set(payload["counters"]) == COUNTER_KEYS
    assert "optimizer" not in payload
    assert "scheduler" not in payload
    assert "rng" not in payload
    assert "stream_cursor" not in payload
    assert "resolved_config" not in payload
    assert "measurement_evidence" not in payload
    assert "arbitrary_payload" not in payload

    manager = CheckpointManager(
        checkpoint.parent,
        keep_last_n=2,
        identity=tracker.checkpoint_identity,
    )
    resumed = manager.load_resume(checkpoint)
    assert resumed.payload["state"]["stream_cursor"]["shuffle_buffer"][0]["text"] == raw_text

    captured = tmp_path / "captured-model.pt"
    captured.write_bytes(uploaded_bytes)
    reconstructed = reconstruct_model_artifact(captured)
    source = load_checkpoint_for_generation(checkpoint).payload["state"]["model"]
    for name, tensor in reconstructed.model.state_dict().items():
        assert torch.equal(tensor, source[name])
    tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
    expected_model = SimpleDecoderTransformer(
        vocab_size=reconstructed.tokenizer.vocab_size,
        embed_size=cfg.model.embed_size,
        num_heads=cfg.model.num_heads,
        max_len=cfg.training.sequence_length,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        pad_token_id=reconstructed.tokenizer.pad_token_id,
    )
    expected_model.load_state_dict(source, strict=True)
    expected_model.eval()
    with torch.no_grad():
        assert torch.equal(reconstructed.model(tokens), expected_model(tokens))
    with pytest.raises(CheckpointVerificationError):
        manager.load_resume(captured)

    unexpected = copy.deepcopy(payload)
    unexpected["unexpected"] = raw_text
    invalid = tmp_path / "invalid-model.pt"
    torch.save(unexpected, invalid)
    with pytest.raises(ValueError, match="keys must be exactly"):
        reconstruct_model_artifact(invalid)


def test_successful_uploads_accumulate_against_the_same_visible_snapshot(tmp_path: Path):
    snapshot = tmp_path / "usage.json"
    fake = FakeWandb()
    tracker = _tracker(
        tmp_path,
        _config(policy="milestone", snapshot_path=snapshot),
        fake,
    )
    tracker.start(object())
    first_checkpoint = _checkpoint(tmp_path / "first", tracker=tracker, kind="milestone", step=5)
    second_checkpoint = _checkpoint(tmp_path / "second", tracker=tracker, kind="milestone", step=6)
    used_bytes = 10
    _snapshot(
        snapshot,
        used_bytes=used_bytes,
        limit_bytes=(
            used_bytes + first_checkpoint.stat().st_size + second_checkpoint.stat().st_size - 1
        ),
    )

    first = tracker.consider_artifact(reason="milestone", checkpoint_path=first_checkpoint, step=5)
    _snapshot(
        snapshot,
        used_bytes=used_bytes,
        limit_bytes=(
            used_bytes + first_checkpoint.stat().st_size + second_checkpoint.stat().st_size - 1
        ),
    )
    second = tracker.consider_artifact(
        reason="milestone", checkpoint_path=second_checkpoint, step=6
    )

    assert first["outcome"] == "uploaded"
    assert second["outcome"] == "blocked"
    assert second["block_reason"] == "visible_quota_insufficient"
    assert second["quota"]["used_bytes"] == used_bytes
    assert second["quota"]["reserved_by_tracker_bytes"] == first_checkpoint.stat().st_size
    assert second["projected_bytes"] == (
        used_bytes + first_checkpoint.stat().st_size + second_checkpoint.stat().st_size
    )
    assert len(fake.run.uploads) == 1


def test_cloud_submission_failure_retains_quota_reservation(tmp_path: Path):
    snapshot = tmp_path / "usage.json"
    fake = FakeWandb(upload_error=RuntimeError("submission state unknown"))
    tracker = _tracker(
        tmp_path,
        _config(policy="milestone", snapshot_path=snapshot),
        fake,
    )
    tracker.start(object())
    first_checkpoint = _checkpoint(tmp_path / "first", tracker=tracker, kind="milestone", step=5)
    second_checkpoint = _checkpoint(tmp_path / "second", tracker=tracker, kind="milestone", step=6)
    used_bytes = 10
    _snapshot(
        snapshot,
        used_bytes=used_bytes,
        limit_bytes=(
            used_bytes + first_checkpoint.stat().st_size + second_checkpoint.stat().st_size - 1
        ),
    )

    first = tracker.consider_artifact(reason="milestone", checkpoint_path=first_checkpoint, step=5)
    second = tracker.consider_artifact(
        reason="milestone", checkpoint_path=second_checkpoint, step=6
    )

    assert first["outcome"] == "upload_failed"
    assert Path(fake.artifacts[0].files[0][0]).is_file()
    assert second["block_reason"] == "visible_quota_insufficient"
    assert second["quota"]["reserved_by_tracker_bytes"] == first_checkpoint.stat().st_size


def test_committed_upload_cleanup_failure_keeps_uploaded_outcome(
    tmp_path: Path,
    monkeypatch,
):
    snapshot = tmp_path / "usage.json"
    _snapshot(snapshot, limit_bytes=10_000_000)
    fake = FakeWandb()
    tracker = _tracker(tmp_path, _config(snapshot_path=snapshot), fake)
    tracker.start(object())
    checkpoint = _checkpoint(tmp_path / "checkpoint", tracker=tracker, kind="final", step=5)

    def cleanup_failure(_artifact):
        raise PermissionError("simulated staging cleanup failure")

    monkeypatch.setattr(
        "training.wandb_tracking.remove_staged_model_artifact",
        cleanup_failure,
    )

    decision = tracker.consider_artifact(reason="final", checkpoint_path=checkpoint, step=5)

    assert decision["outcome"] == "uploaded"
    assert decision["retry_outcome"] == "not_needed"
    assert tracker._reserved_bytes_by_tracker == checkpoint.stat().st_size
    assert fake.run.summary.values["artifact/outcome"] == "uploaded"
    events = [
        json.loads(line)
        for line in (tmp_path / "wandb_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    artifact_events = [event for event in events if event["action"] == "artifact"]
    assert [event["outcome"] for event in artifact_events] == ["uploaded"]
    cleanup = next(event for event in events if event["action"] == "artifact_cleanup")
    assert cleanup["outcome"] == "failed"
    assert cleanup["context"] == "committed_upload"
    assert cleanup["error"]["type"] == "PermissionError"
    assert Path(fake.artifacts[0].files[0][0]).is_file()


def test_cleanup_and_cleanup_evidence_failure_cannot_mask_artifact_outcomes(
    tmp_path: Path,
    monkeypatch,
):
    snapshot = tmp_path / "usage.json"
    _snapshot(snapshot, limit_bytes=10_000_000)

    def cleanup_failure(_artifact):
        raise PermissionError("simulated staging cleanup failure")

    monkeypatch.setattr(
        "training.wandb_tracking.remove_staged_model_artifact",
        cleanup_failure,
    )

    committed_fake = FakeWandb()
    committed = _tracker(tmp_path / "committed", _config(snapshot_path=snapshot), committed_fake)
    committed.start(object())
    committed_checkpoint = _checkpoint(
        tmp_path / "committed-checkpoint",
        tracker=committed,
        kind="final",
        step=5,
    )
    committed_record_failure = committed._record_failure

    def fail_committed_cleanup_evidence(action, error, details=None):
        if action == "artifact_cleanup":
            raise OSError("simulated cleanup evidence failure")
        return committed_record_failure(action, error, details)

    committed._record_failure = fail_committed_cleanup_evidence
    committed_decision = committed.consider_artifact(
        reason="final",
        checkpoint_path=committed_checkpoint,
        step=5,
    )
    assert committed_decision["outcome"] == "uploaded"
    assert committed_decision["retry_outcome"] == "not_needed"
    assert committed._reserved_bytes_by_tracker == committed_checkpoint.stat().st_size

    preparation_fake = FakeWandb()
    preparation = _tracker(
        tmp_path / "preparation",
        _config(snapshot_path=snapshot),
        preparation_fake,
    )
    preparation.start(object())
    preparation_checkpoint = _checkpoint(
        tmp_path / "preparation-checkpoint",
        tracker=preparation,
        kind="final",
        step=5,
    )
    prepared = preparation_fake.Artifact

    def fail_add_file(*args, **kwargs):
        artifact = prepared(*args, **kwargs)
        artifact.add_file = lambda *args, **kwargs: (_ for _ in ()).throw(
            OSError("original local preparation failure")
        )
        return artifact

    preparation_fake.Artifact = fail_add_file
    preparation_record_failure = preparation._record_failure

    def fail_preparation_cleanup_evidence(action, error, details=None):
        if action == "artifact_cleanup":
            raise OSError("simulated cleanup evidence failure")
        return preparation_record_failure(action, error, details)

    preparation._record_failure = fail_preparation_cleanup_evidence
    preparation_decision = preparation.consider_artifact(
        reason="final",
        checkpoint_path=preparation_checkpoint,
        step=5,
    )
    assert preparation_decision["outcome"] == "upload_failed"
    assert preparation_decision["error"]["message"] == "original local preparation failure"
    assert preparation._reserved_bytes_by_tracker == 0
    assert preparation_fake.run.uploads == []


def test_blocking_cloud_submission_is_bounded_and_retains_reservation(tmp_path: Path):
    release = threading.Event()

    class BlockingUploadRun(FakeRun):
        def log_artifact(self, artifact, aliases):
            release.wait(1.0)
            return self.artifact_result

    snapshot = tmp_path / "usage.json"
    _snapshot(snapshot, limit_bytes=10_000_000)
    cfg = _config(policy="milestone", snapshot_path=snapshot)
    cfg.wandb.artifact.upload_timeout_seconds = 0.5
    fake = FakeWandb()
    fake.run = BlockingUploadRun()
    tracker = _tracker(tmp_path, cfg, fake)
    tracker.start(object())
    checkpoint = _checkpoint(tmp_path / "checkpoint", tracker=tracker, kind="milestone", step=5)
    started = time.monotonic()

    decision = tracker.consider_artifact(reason="milestone", checkpoint_path=checkpoint, step=5)
    release.set()

    assert time.monotonic() - started < 1.2
    assert decision["outcome"] == "upload_failed"
    assert tracker._reserved_bytes_by_tracker == checkpoint.stat().st_size
    assert Path(fake.artifacts[0].files[0][0]).is_file()
    failure = next(
        json.loads(line)
        for line in (tmp_path / "wandb_events.jsonl").read_text(encoding="utf-8").splitlines()
        if '"action": "artifact"' in line and '"outcome": "upload_failed"' in line
    )
    assert failure["error"]["type"] == "TimeoutError"


def test_local_artifact_preparation_failure_releases_quota_for_retry(tmp_path: Path):
    snapshot = tmp_path / "usage.json"
    _snapshot(snapshot, limit_bytes=10_000_000)
    fake = FakeWandb()
    tracker = _tracker(tmp_path, _config(policy="milestone", snapshot_path=snapshot), fake)
    tracker.start(object())
    checkpoint = _checkpoint(tmp_path / "checkpoint", tracker=tracker, kind="milestone", step=5)
    original_artifact = fake.Artifact
    attempts = 0

    def artifact(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        prepared = original_artifact(*args, **kwargs)
        if attempts == 1:
            prepared.add_file = lambda *args, **kwargs: (_ for _ in ()).throw(
                OSError("local staging failed")
            )
        return prepared

    fake.Artifact = artifact

    first = tracker.consider_artifact(reason="milestone", checkpoint_path=checkpoint, step=5)
    second = tracker.consider_artifact(reason="milestone", checkpoint_path=checkpoint, step=5)

    assert first["outcome"] == "upload_failed"
    assert second["outcome"] == "uploaded"
    assert tracker._reserved_bytes_by_tracker == checkpoint.stat().st_size
    assert len(fake.run.uploads) == 1


def test_blocking_artifact_preparation_is_bounded_and_releases_reservation(tmp_path: Path):
    release = threading.Event()

    class BlockingArtifact(FakeArtifact):
        def add_file(self, path, name, policy=None) -> None:
            release.wait(1.0)
            super().add_file(path, name, policy)

    class BlockingArtifactWandb(FakeWandb):
        def Artifact(self, *args, **kwargs):
            artifact = BlockingArtifact(*args, **kwargs)
            self.artifacts.append(artifact)
            return artifact

    snapshot = tmp_path / "usage.json"
    _snapshot(snapshot, limit_bytes=10_000_000)
    cfg = _config(snapshot_path=snapshot)
    cfg.wandb.artifact.upload_timeout_seconds = 0.01
    fake = BlockingArtifactWandb()
    tracker = _tracker(tmp_path, cfg, fake)
    tracker.start(object())
    checkpoint = _checkpoint(tmp_path / "checkpoint", tracker=tracker, kind="final", step=5)
    started = time.monotonic()

    decision = tracker.consider_artifact(reason="final", checkpoint_path=checkpoint, step=5)
    elapsed = time.monotonic() - started
    release.set()

    assert elapsed < 0.2
    assert decision["outcome"] == "upload_failed"
    assert decision["error"]["type"] == "TimeoutError"
    assert tracker._reserved_bytes_by_tracker == 0
    assert tracker._reserved_sha256 == set()
    assert fake.run.uploads == []


def test_quota_ledger_retains_strictest_values_across_snapshot_refreshes(tmp_path: Path):
    snapshot = tmp_path / "usage.json"
    fake = FakeWandb()
    tracker = _tracker(
        tmp_path,
        _config(policy="milestone", snapshot_path=snapshot),
        fake,
    )
    tracker.start(object())
    first_checkpoint = _checkpoint(tmp_path / "first", tracker=tracker, kind="milestone", step=5)
    second_checkpoint = _checkpoint(tmp_path / "second", tracker=tracker, kind="milestone", step=6)
    _snapshot(snapshot, used_bytes=10, limit_bytes=10_000_000)
    first = tracker.consider_artifact(reason="milestone", checkpoint_path=first_checkpoint, step=5)
    strict_limit = 20 + first_checkpoint.stat().st_size + second_checkpoint.stat().st_size - 1
    _snapshot(snapshot, used_bytes=20, limit_bytes=strict_limit)
    strict = tracker.consider_artifact(
        reason="milestone", checkpoint_path=second_checkpoint, step=6
    )
    _snapshot(snapshot, used_bytes=15, limit_bytes=strict_limit + 1_000_000)
    relaxed = tracker.consider_artifact(
        reason="milestone", checkpoint_path=second_checkpoint, step=6
    )

    assert first["outcome"] == "uploaded"
    assert strict["block_reason"] == "visible_quota_insufficient"
    assert relaxed["block_reason"] == "visible_quota_insufficient"
    assert relaxed["quota"]["effective_baseline_used_bytes"] == 20
    assert relaxed["quota"]["effective_limit_bytes"] == strict_limit
    assert relaxed["quota"]["reserved_by_tracker_bytes"] == first_checkpoint.stat().st_size


def test_concurrent_artifact_selection_reserves_quota_before_upload(tmp_path: Path):
    first_submitted = threading.Event()
    release_first = threading.Event()

    class BlockingFirstUploadRun(FakeRun):
        def log_artifact(self, artifact, aliases):
            self.uploads.append((artifact, aliases))
            if len(self.uploads) == 1:
                first_submitted.set()
                release_first.wait(1.0)
            return self.artifact_result

    snapshot = tmp_path / "usage.json"
    fake = FakeWandb()
    fake.run = BlockingFirstUploadRun()
    tracker = _tracker(
        tmp_path,
        _config(policy="milestone", snapshot_path=snapshot),
        fake,
    )
    tracker.start(object())
    first_checkpoint = _checkpoint(tmp_path / "first", tracker=tracker, kind="milestone", step=5)
    second_checkpoint = _checkpoint(tmp_path / "second", tracker=tracker, kind="milestone", step=6)
    _snapshot(
        snapshot,
        used_bytes=10,
        limit_bytes=(10 + first_checkpoint.stat().st_size + second_checkpoint.stat().st_size - 1),
    )
    decisions = []
    first_thread = threading.Thread(
        target=lambda: decisions.append(
            tracker.consider_artifact(reason="milestone", checkpoint_path=first_checkpoint, step=5)
        )
    )
    first_thread.start()
    assert first_submitted.wait(1.0)

    second = tracker.consider_artifact(
        reason="milestone", checkpoint_path=second_checkpoint, step=6
    )
    release_first.set()
    first_thread.join(1.0)

    assert second["block_reason"] == "visible_quota_insufficient"
    assert decisions[0]["outcome"] == "uploaded"
    assert len(fake.run.uploads) == 1


def test_missing_login_and_upload_failure_preserve_local_retry_evidence(tmp_path: Path):
    snapshot = tmp_path / "usage.json"
    _snapshot(snapshot, limit_bytes=10_000_000)
    early_gate_checkpoint = tmp_path / "not-validated.pt"
    early_gate_checkpoint.write_bytes(b"not reached")

    missing_login = FakeWandb(login=False)
    login_tracker = _tracker(tmp_path / "login", _config(snapshot_path=snapshot), missing_login)
    login_tracker.start(object())
    login_decision = login_tracker.consider_artifact(
        reason="final", checkpoint_path=early_gate_checkpoint, step=1
    )
    assert login_decision["block_reason"] == "online_run_unavailable"

    failing = FakeWandb(upload_error=RuntimeError("upload failed"))
    upload_tracker = _tracker(tmp_path / "upload", _config(snapshot_path=snapshot), failing)
    upload_tracker.start(object())
    checkpoint = _checkpoint(tmp_path / "checkpoints", tracker=upload_tracker, kind="final", step=2)
    upload_decision = upload_tracker.consider_artifact(
        reason="final", checkpoint_path=checkpoint, step=2
    )
    assert upload_decision["outcome"] == "upload_failed"
    assert upload_decision["retry_outcome"] == "operator_retry_required"
    assert checkpoint.is_file()
    assert "upload_failed" in (tmp_path / "upload" / "wandb_events.jsonl").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("corrupt", "unable to read checkpoint"),
        ("wrong_kind", "does not match artifact reason"),
        ("foreign_identity", "does not match the active run"),
        ("wrong_step", "does not match 6"),
    ],
)
def test_artifact_checkpoint_validation_rejects_untrusted_candidates(
    tmp_path: Path, case: str, message: str
):
    snapshot = tmp_path / "usage.json"
    _snapshot(snapshot, limit_bytes=10_000_000)
    policy = "best" if case == "wrong_kind" else "final"
    fake = FakeWandb()
    cfg = _config(policy=policy, snapshot_path=snapshot)
    tracker = _tracker(tmp_path, cfg, fake)
    tracker.start(object())

    reason = policy
    expected_step = 6 if case == "wrong_step" else 5
    if case == "corrupt":
        checkpoint = tmp_path / "corrupt.pt"
        checkpoint.write_bytes(b"not a repository checkpoint")
    elif case == "wrong_kind":
        checkpoint = _checkpoint(tmp_path / "wrong-kind", tracker=tracker, kind="final", step=5)
    elif case == "foreign_identity":
        foreign_cfg = _config(policy="final", snapshot_path=snapshot)
        foreign_cfg.profile.name = "foreign_pretrain"
        foreign_identity = build_checkpoint_identity(foreign_cfg)
        checkpoint = _checkpoint(
            tmp_path / "foreign",
            tracker=tracker,
            kind="final",
            step=5,
            cfg=foreign_cfg,
            identity=foreign_identity,
        )
    else:
        checkpoint = _checkpoint(tmp_path / "wrong-step", tracker=tracker, kind="final", step=5)

    decision = tracker.consider_artifact(
        reason=reason,
        checkpoint_path=checkpoint,
        step=expected_step,
    )

    assert decision["outcome"] == "blocked"
    assert decision["block_reason"] == "checkpoint_identity_unavailable"
    assert message in decision["error"]["message"]
    assert fake.run.uploads == []


def test_wandb_0251_public_viewer_string_teams_authorize_entity(tmp_path: Path):
    assert wandb.__version__ == "0.25.1"
    snapshot = tmp_path / "usage.json"
    _snapshot(snapshot, limit_bytes=10_000_000)
    fake = FakeWandb(string_teams=True)
    tracker = _tracker(tmp_path, _config(snapshot_path=snapshot), fake)
    tracker.start(object())
    checkpoint = _checkpoint(tmp_path / "checkpoints", tracker=tracker, kind="final", step=5)

    decision = tracker.consider_artifact(reason="final", checkpoint_path=checkpoint, step=5)

    assert decision["outcome"] == "uploaded"
    assert decision["auth"] == {
        "outcome": "verified",
        "viewer": "operator",
        "entity": "research-team",
    }


def test_ci_and_data_mode_deny_before_checkpoint_validation(tmp_path: Path, monkeypatch):
    raw = tmp_path / "intentionally-invalid.pt"
    raw.write_bytes(b"early gate must prevent parsing")
    snapshot = tmp_path / "usage.json"
    _snapshot(snapshot)

    monkeypatch.setenv("CI", "true")
    ci_tracker = _tracker(tmp_path / "ci", _config(snapshot_path=snapshot), FakeWandb())
    ci_tracker.start(object())
    ci_decision = ci_tracker.consider_artifact(reason="final", checkpoint_path=raw, step=1)
    assert ci_decision["block_reason"] == "profile_forbids_artifacts"

    monkeypatch.delenv("CI")
    data_cfg = _config(snapshot_path=snapshot)
    data_cfg.data.mode = "memorization_smoke"
    data_tracker = _tracker(tmp_path / "data", data_cfg, FakeWandb())
    data_tracker.start(object())
    data_decision = data_tracker.consider_artifact(reason="final", checkpoint_path=raw, step=1)
    assert data_decision["block_reason"] == "profile_forbids_artifacts"

    assert (
        artifact_uploads_forbidden(
            {"name": "pretrain_streaming", "purpose": "pretraining"},
            data_mode="streaming",
            environment={"CI": "0"},
        )
        is False
    )


@pytest.mark.parametrize(
    ("artifact_result", "message"),
    [
        (SimpleNamespace(state="COMMITTED"), "no completion wait"),
        (FakeCommittedArtifact(state="PENDING"), "did not reach COMMITTED"),
    ],
)
def test_artifact_upload_requires_wait_and_committed_state(
    tmp_path: Path, artifact_result: object, message: str
):
    snapshot = tmp_path / "usage.json"
    _snapshot(snapshot, limit_bytes=10_000_000)
    fake = FakeWandb(artifact_result=artifact_result)
    tracker = _tracker(tmp_path, _config(snapshot_path=snapshot), fake)
    tracker.start(object())
    checkpoint = _checkpoint(tmp_path / "checkpoints", tracker=tracker, kind="final", step=5)

    decision = tracker.consider_artifact(reason="final", checkpoint_path=checkpoint, step=5)

    assert decision["outcome"] == "upload_failed"
    assert decision["retry_outcome"] == "operator_retry_required"
    assert fake.run.summary.values["artifact/outcome"] == "upload_failed"
    failure = next(
        json.loads(line)
        for line in (tmp_path / "wandb_events.jsonl").read_text(encoding="utf-8").splitlines()
        if '"action": "artifact"' in line and '"outcome": "upload_failed"' in line
    )
    assert message in failure["error"]["message"]


def test_artifact_identity_race_is_detected_before_log_artifact(tmp_path: Path):
    snapshot = tmp_path / "usage.json"
    _snapshot(snapshot, limit_bytes=10_000_000)
    fake = FakeWandb()

    class MutatingArtifact(FakeArtifact):
        def add_file(self, path, name, policy=None) -> None:
            super().add_file(path, name, policy)
            with Path(path).open("ab") as handle:
                handle.write(b"changed-after-identity")

    fake.Artifact = MutatingArtifact
    tracker = _tracker(tmp_path, _config(snapshot_path=snapshot), fake)
    tracker.start(object())
    checkpoint = _checkpoint(tmp_path / "checkpoints", tracker=tracker, kind="final", step=5)

    decision = tracker.consider_artifact(reason="final", checkpoint_path=checkpoint, step=5)

    assert decision["outcome"] == "upload_failed"
    assert fake.run.uploads == []
    failure = next(
        json.loads(line)
        for line in (tmp_path / "wandb_events.jsonl").read_text(encoding="utf-8").splitlines()
        if '"action": "artifact"' in line and '"outcome": "upload_failed"' in line
    )
    assert "changed while W&B captured" in failure["error"]["message"]


def test_finish_is_bounded_and_tracker_persists_timeout(tmp_path: Path):
    release = threading.Event()

    class BlockingRun(FakeRun):
        def finish(self) -> None:
            release.wait(1.0)

    run = BlockingRun()
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="finish exceeded"):
        finish_run_bounded(run, timeout_seconds=0.01)
    assert time.monotonic() - started < 0.2

    cfg = _config(mode="offline", policy="none")
    cfg.wandb.finish_timeout_seconds = 0.01
    fake = FakeWandb()
    fake.run = BlockingRun()
    tracker = _tracker(tmp_path, cfg, fake)
    tracker.start(object())
    tracker.finish()
    release.set()

    events = [
        json.loads(line)
        for line in (tmp_path / "wandb_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    finish = next(event for event in events if event["action"] == "finish")
    assert finish["outcome"] == "failed"
    assert finish["error"]["type"] == "TimeoutError"


def test_blocking_scalar_log_is_bounded_and_opens_circuit_breaker(tmp_path: Path):
    release = threading.Event()
    sdk_lock = threading.Lock()

    class BlockingRun(FakeRun):
        def log(self, values) -> None:
            with sdk_lock:
                self.logs.append(values)
                release.wait(1.0)

        def finish(self) -> None:
            with sdk_lock:
                self.finished = True

    cfg = _config(mode="offline", policy="none")
    cfg.wandb.finish_timeout_seconds = 0.01
    fake = FakeWandb()
    fake.run = BlockingRun()
    tracker = _tracker(tmp_path, cfg, fake)
    tracker.start(object())

    started = time.monotonic()
    tracker.log({"optimizer_step": 1, "train/loss": 1.0})
    elapsed = time.monotonic() - started
    tracker.log({"optimizer_step": 2, "train/loss": 0.5})
    worker = tracker._scalar_log_worker
    finish_started = time.monotonic()
    tracker.finish()
    finish_elapsed = time.monotonic() - finish_started
    release.set()

    assert elapsed < 0.2
    assert finish_elapsed < 0.2
    assert fake.run.finished is False
    assert fake.run.logs == [{"optimizer_step": 1, "train/loss": 1.0}]
    assert tracker._scalar_log_worker is worker
    failure = next(
        json.loads(line)
        for line in (tmp_path / "wandb_events.jsonl").read_text(encoding="utf-8").splitlines()
        if '"action": "log"' in line and '"outcome": "failed"' in line
    )
    assert failure["error"]["type"] == "TimeoutError"
    assert failure["circuit_breaker"] == "opened"
    finish_failure = next(
        json.loads(line)
        for line in (tmp_path / "wandb_events.jsonl").read_text(encoding="utf-8").splitlines()
        if '"action": "finish"' in line and '"outcome": "failed"' in line
    )
    assert "scalar log worker exceeded" in finish_failure["error"]["message"]


def test_blocking_summary_update_is_bounded_and_opens_circuit_breaker(tmp_path: Path):
    release = threading.Event()

    class BlockingSummary(FakeSummary):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def update(self, values) -> None:
            self.calls += 1
            release.wait(1.0)
            super().update(values)

    cfg = _config(mode="offline", policy="none")
    cfg.wandb.log_timeout_seconds = 0.01
    fake = FakeWandb()
    fake.run.summary = BlockingSummary()
    tracker = _tracker(tmp_path, cfg, fake)

    started = time.monotonic()
    tracker.start(object())
    elapsed = time.monotonic() - started
    tracker.update_summary({"run/final_optimizer_step": 1})
    release.set()

    assert elapsed < 0.2
    assert fake.run.summary.calls == 1
    failure = next(
        json.loads(line)
        for line in (tmp_path / "wandb_events.jsonl").read_text(encoding="utf-8").splitlines()
        if '"action": "summary"' in line and '"outcome": "failed"' in line
    )
    assert failure["error"]["type"] == "TimeoutError"
    assert failure["circuit_breaker"] == "opened"


def test_scalar_log_worker_exits_after_normal_finish(tmp_path: Path):
    fake = FakeWandb()
    tracker = _tracker(
        tmp_path,
        _config(mode="offline", policy="none"),
        fake,
    )
    tracker.start(object())
    tracker.log({"optimizer_step": 1, "train/loss": 1.0})
    worker = tracker._scalar_log_worker
    assert worker is not None and worker.is_alive()

    tracker.finish()

    assert not worker.is_alive()
    assert fake.run.finished is True


def test_worker_stop_cancels_queued_scalar_log_without_sdk_call(tmp_path: Path):
    entered = threading.Event()
    release = threading.Event()

    class BlockingRun(FakeRun):
        def log(self, values) -> None:
            self.logs.append(values)
            entered.set()
            release.wait(1.0)

    cfg = _config(mode="offline", policy="none")
    cfg.wandb.log_timeout_seconds = 0.5
    cfg.wandb.finish_timeout_seconds = 0.01
    fake = FakeWandb()
    fake.run = BlockingRun()
    tracker = _tracker(tmp_path, cfg, fake)
    tracker.start(object())
    callers = [
        threading.Thread(
            target=tracker.log,
            args=({"optimizer_step": step, "train/loss": float(step)},),
        )
        for step in (1, 2)
    ]
    callers[0].start()
    assert entered.wait(1.0)
    callers[1].start()
    deadline = time.monotonic() + 1.0
    while tracker._scalar_log_queue.qsize() != 1 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert tracker._scalar_log_queue.qsize() == 1

    tracker.finish()
    release.set()
    for caller in callers:
        caller.join(1.0)
    worker = tracker._scalar_log_worker
    assert worker is not None
    worker.join(1.0)

    assert fake.run.logs == [{"optimizer_step": 1, "train/loss": 1.0}]
    assert not worker.is_alive()
    assert all(not caller.is_alive() for caller in callers)


def test_watch_is_off_by_default_and_optional_hooks_are_torn_down(tmp_path: Path):
    for enabled in (False, True):
        fake = FakeWandb()
        tracker = _tracker(
            tmp_path / str(enabled),
            _config(mode="offline", policy="none", watch=enabled),
            fake,
        )
        model = object()
        tracker.start(model)
        tracker.finish()
        assert len(fake.run.watches) == int(enabled)
        assert len(fake.run.unwatches) == int(enabled)
        if enabled:
            assert fake.run.watches[0][1] == {"log": "gradients", "log_freq": 17}


def test_blocking_watch_install_is_bounded_and_late_hooks_are_removed(tmp_path: Path):
    entered = threading.Event()
    release = threading.Event()
    late_finished = threading.Event()

    class BlockingWatchRun(FakeRun):
        def __init__(self) -> None:
            super().__init__()
            self.torch_history = TorchHistory()

        def watch(self, model, **kwargs) -> None:
            try:
                self.watches.append((model, kwargs))
                entered.set()
                release.wait(1.0)
                self.torch_history.add_log_parameters_hook(model, log_freq=kwargs["log_freq"])
                raise RuntimeError("late watch failure after installing hooks")
            finally:
                late_finished.set()

        def unwatch(self, model=None) -> None:
            self.unwatches.append(model)
            self.torch_history.unhook_all()

    cfg = _config(mode="offline", policy="none", watch=True)
    cfg.wandb.log_timeout_seconds = 0.01
    fake = FakeWandb()
    fake.run = BlockingWatchRun()
    tracker = _tracker(tmp_path, cfg, fake)
    model = torch.nn.Linear(2, 2)
    started = time.monotonic()

    tracker.start(model)
    elapsed = time.monotonic() - started
    assert entered.is_set()
    release.set()
    assert late_finished.wait(1.0)
    deadline = time.monotonic() + 1.0
    while fake.run.torch_history._hook_handles and time.monotonic() < deadline:
        time.sleep(0.001)

    assert elapsed < 0.2
    assert not model._forward_hooks
    assert fake.run.torch_history._hook_handles == {}
    assert fake.run.unwatches == [None, None]
    failure = next(
        json.loads(line)
        for line in (tmp_path / "wandb_events.jsonl").read_text(encoding="utf-8").splitlines()
        if '"action": "watch"' in line and '"outcome": "failed"' in line
    )
    assert failure["error"]["type"] == "TimeoutError"


def test_blocking_unwatch_is_bounded_and_finish_is_still_attempted(tmp_path: Path):
    release = threading.Event()

    class BlockingUnwatchRun(FakeRun):
        def unwatch(self, model=None) -> None:
            self.unwatches.append(model)
            release.wait(1.0)

    cfg = _config(mode="offline", policy="none", watch=True)
    cfg.wandb.finish_timeout_seconds = 0.01
    fake = FakeWandb()
    fake.run = BlockingUnwatchRun()
    tracker = _tracker(tmp_path, cfg, fake)
    tracker.start(torch.nn.Linear(2, 2))
    started = time.monotonic()

    tracker.finish()
    elapsed = time.monotonic() - started
    release.set()

    assert elapsed < 0.2
    assert fake.run.finished is True
    failure = next(
        json.loads(line)
        for line in (tmp_path / "wandb_events.jsonl").read_text(encoding="utf-8").splitlines()
        if '"action": "unwatch"' in line and '"outcome": "failed"' in line
    )
    assert failure["error"]["type"] == "TimeoutError"


def test_partial_watch_installation_is_torn_down_immediately(tmp_path: Path):
    class PartialWatchRun(FakeRun):
        def __init__(self) -> None:
            super().__init__()
            self.torch_history = TorchHistory()

        def watch(self, model, **kwargs) -> None:
            self.watches.append((model, kwargs))
            self.torch_history.add_log_parameters_hook(
                model,
                log_freq=kwargs["log_freq"],
            )
            raise RuntimeError("watch failed after installing hooks")

        def unwatch(self, model=None) -> None:
            self.unwatches.append(model)
            if model is None:
                self.torch_history.unhook_all()
                return
            for name in model._wandb_hook_names:
                self.torch_history.unhook(name)
            delattr(model, "_wandb_hook_names")

    fake = FakeWandb()
    fake.run = PartialWatchRun()
    tracker = _tracker(
        tmp_path,
        _config(mode="offline", policy="none", watch=True),
        fake,
    )
    model = torch.nn.Linear(2, 2)

    tracker.start(model)

    assert fake.run.unwatches == [None]
    assert not model._forward_hooks
    assert fake.run.torch_history._hook_handles == {}
    assert not hasattr(model, "_wandb_hook_names")
    failure = next(
        json.loads(line)
        for line in (tmp_path / "wandb_events.jsonl").read_text(encoding="utf-8").splitlines()
        if '"action": "watch"' in line and '"outcome": "failed"' in line
    )
    assert failure["error"]["message"] == "watch failed after installing hooks"


def test_model_specific_unwatch_failure_falls_back_to_all_hooks(tmp_path: Path):
    class FallbackCleanupRun(FakeRun):
        def __init__(self) -> None:
            super().__init__()
            self.torch_history = TorchHistory()

        def watch(self, model, **kwargs) -> None:
            self.watches.append((model, kwargs))
            self.torch_history.add_log_parameters_hook(
                model,
                log_freq=kwargs["log_freq"],
            )

        def unwatch(self, model=None) -> None:
            self.unwatches.append(model)
            if model is not None:
                raise KeyError("partial model hook registry")
            self.torch_history.unhook_all()

    fake = FakeWandb()
    fake.run = FallbackCleanupRun()
    tracker = _tracker(
        tmp_path,
        _config(mode="offline", policy="none", watch=True),
        fake,
    )
    model = torch.nn.Linear(2, 2)
    tracker.start(model)
    assert model._forward_hooks

    tracker.finish()

    assert fake.run.unwatches == [model, None]
    assert not model._forward_hooks
    assert fake.run.torch_history._hook_handles == {}
    assert not hasattr(model, "_wandb_hook_names")


def test_stuck_scalar_worker_does_not_prevent_watch_hook_cleanup(tmp_path: Path):
    release = threading.Event()

    class BlockingWatchedRun(FakeRun):
        def __init__(self) -> None:
            super().__init__()
            self.torch_history = TorchHistory()

        def watch(self, model, **kwargs) -> None:
            self.watches.append((model, kwargs))
            self.torch_history.add_log_parameters_hook(
                model,
                log_freq=kwargs["log_freq"],
            )

        def unwatch(self, model=None) -> None:
            self.unwatches.append(model)
            if model is None:
                self.torch_history.unhook_all()
                return
            for name in model._wandb_hook_names:
                self.torch_history.unhook(name)
            delattr(model, "_wandb_hook_names")

        def log(self, values) -> None:
            self.logs.append(values)
            release.wait(1.0)

    cfg = _config(mode="offline", policy="none", watch=True)
    cfg.wandb.finish_timeout_seconds = 0.01
    fake = FakeWandb()
    fake.run = BlockingWatchedRun()
    tracker = _tracker(tmp_path, cfg, fake)
    model = torch.nn.Linear(2, 2)
    tracker.start(model)
    assert model._forward_hooks
    tracker.log({"optimizer_step": 1, "train/loss": 1.0})

    tracker.finish()
    release.set()

    assert not model._forward_hooks
    assert fake.run.torch_history._hook_handles == {}
    assert not hasattr(model, "_wandb_hook_names")
    assert fake.run.unwatches == [model]
    assert fake.run.finished is True


@pytest.mark.parametrize(
    ("name", "purpose"),
    [
        ("smoke_overfit", "memorization_smoke"),
        ("stability_smoke", "pretraining"),
        ("ci", "pretraining"),
        ("gate_overfit", "memorization_gate"),
    ],
)
def test_smoke_ci_and_memorization_profiles_are_code_level_denied(name, purpose):
    assert artifact_uploads_forbidden({"name": name, "purpose": purpose}) is True
