"""Failure-isolated, quota-safe Weights & Biases tracking.

W&B receives compact metrics and selected model checkpoints.  Local metrics,
manifests, and checkpoints remain authoritative and continue to work when W&B
is disabled, offline, unavailable, or out of artifact quota.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import wandb
from omegaconf import DictConfig, OmegaConf

from runtime.environment import collect_environment
from training.checkpoint import load_checkpoint_for_generation
from training.model_artifact import (
    MODEL_ARTIFACT_SCHEMA_VERSION,
    ModelArtifactFile,
    remove_staged_model_artifact,
    stage_model_artifact,
    verify_staged_model_artifact,
)

ARTIFACT_REASONS = {"best", "final", "milestone"}
DATA_REFERENCE_FIELDS = {
    "name",
    "type",
    "source",
    "manifest_path",
    "expected_fingerprint",
    "selection",
    "path",
    "url",
    "repo_id",
    "revision",
    "config_name",
    "split",
    "data_files",
    "ratio",
}

_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|authorization|password|passwd|secret|token|key)"
    r"(\s*[:=]\s*)(?:bearer\s+)?(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_BEARER_CREDENTIAL = re.compile(r"(?i)\bbearer\s+[^\s,;]+")


def safe_external_error(error: Exception) -> dict[str, str]:
    """Return bounded external-service failure evidence with credentials removed."""

    message = str(error)
    sensitive_names = (
        "WANDB_API_KEY",
        "WANDB_ACCESS_TOKEN",
        "WANDB_TOKEN",
        "WANDB_PASSWORD",
    )
    for name in sensitive_names:
        value = os.environ.get(name)
        if value:
            message = message.replace(value, "[REDACTED]")
    message = _CREDENTIAL_ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]",
        message,
    )
    message = _BEARER_CREDENTIAL.sub("Bearer [REDACTED]", message)
    return {"type": type(error).__name__, "message": message[:500]}


def append_wandb_evidence(
    path: str | Path,
    *,
    action: str,
    outcome: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    """Append one fsynced W&B lifecycle record for standalone evaluators."""

    evidence_path = Path(path)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "outcome": outcome,
        **dict(details or {}),
    }
    descriptor = os.open(
        evidence_path,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o600,
    )
    try:
        os.write(descriptor, (json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class ArtifactCandidate:
    reason: str
    checkpoint_kind: str
    checkpoint_optimizer_step: int
    path: str
    sha256: str
    size_bytes: int
    device: int
    inode: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class UsageSnapshot:
    captured_at_utc: str
    entity: str
    plan: str
    used_bytes: int
    limit_bytes: int
    retention: str
    source: str


def artifact_uploads_forbidden(
    profile: Mapping[str, Any],
    *,
    data_mode: str = "",
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Return whether this profile is code-level denied from model upload."""

    name = str(profile.get("name", "")).lower()
    purpose = str(profile.get("purpose", "")).lower()
    denied_markers = ("ci", "smoke", "memorization")
    if any(
        marker in name or marker in purpose or marker in data_mode.lower()
        for marker in denied_markers
    ):
        return True
    ci_value = (os.environ if environment is None else environment).get("CI", "").strip().lower()
    return ci_value not in {"", "0", "false", "no", "off"}


def finish_run_bounded(run: Any, *, timeout_seconds: float) -> None:
    """Finish a W&B run without allowing an unbounded service wait."""

    call_bounded(
        run.finish,
        timeout_seconds=timeout_seconds,
        operation="W&B finish",
    )


def call_bounded(
    call: Callable[[], Any],
    *,
    timeout_seconds: float,
    operation: str,
    on_late_result: Callable[[Any], None] | None = None,
    on_late_error: Callable[[Exception], None] | None = None,
) -> Any:
    """Run one external SDK call behind an actual wall-clock timeout."""

    errors: list[Exception] = []
    results: list[Any] = []
    timed_out = threading.Event()
    state_lock = threading.Lock()

    def invoke() -> None:
        try:
            result = call()
            with state_lock:
                late = timed_out.is_set()
                if not late:
                    results.append(result)
            if late and on_late_result is not None:
                try:
                    on_late_result(result)
                except Exception:
                    pass
        except Exception as error:
            with state_lock:
                late = timed_out.is_set()
                if not late:
                    errors.append(error)
            if late and on_late_error is not None:
                try:
                    on_late_error(error)
                except Exception:
                    pass

    thread = threading.Thread(target=invoke, name="wandb-bounded-call", daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        with state_lock:
            if not results and not errors:
                timed_out.set()
                raise TimeoutError(f"{operation} exceeded {timeout_seconds:.3f} seconds")
    if errors:
        raise errors[0]
    return results[0]


def wandb_run_config(cfg: DictConfig) -> dict[str, Any]:
    """Return resolved run config with dataset contents reduced to references."""

    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("W&B run config must resolve to a mapping")
    data = resolved.get("data")
    if not isinstance(data, Mapping):
        return resolved
    safe_data: dict[str, Any] = {"mode": data.get("mode")}
    memorization = data.get("memorization")
    if isinstance(memorization, Mapping):
        safe_data["memorization"] = {
            key: memorization[key]
            for key in ("manifest_path", "expected_fingerprint")
            if key in memorization
        }
    streaming = data.get("streaming")
    if isinstance(streaming, Mapping):
        safe_streaming = {
            key: value
            for key, value in streaming.items()
            if key not in {"sources", "datasets", "train", "validation"}
        }
        for split_name in ("train", "validation"):
            split = streaming.get(split_name)
            if not isinstance(split, Mapping):
                continue
            safe_split = {
                key: value for key, value in split.items() if key not in {"sources", "datasets"}
            }
            sources = split.get("sources", split.get("datasets", []))
            safe_split["sources"] = [
                {key: source[key] for key in DATA_REFERENCE_FIELDS if key in source}
                for source in sources
                if isinstance(source, Mapping)
            ]
            safe_streaming[split_name] = safe_split
        safe_data["streaming"] = safe_streaming
    resolved["data"] = safe_data
    return resolved


def load_usage_snapshot(
    path: str | Path,
    *,
    expected_entity: str,
    max_age_seconds: float,
    now: datetime | None = None,
) -> UsageSnapshot:
    """Load an operator-captured, visible W&B usage/plan snapshot."""

    snapshot_path = Path(path)
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"usage snapshot is unreadable: {snapshot_path}") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("usage snapshot must use schema_version 1")
    required = {
        "captured_at_utc",
        "entity",
        "plan",
        "used_bytes",
        "limit_bytes",
        "retention",
        "source",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"usage snapshot is missing {missing}")
    entity = payload["entity"]
    if not isinstance(entity, str) or entity != expected_entity:
        raise ValueError("usage snapshot entity does not match wandb.entity")
    for key in ("plan", "retention", "source"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"usage snapshot {key} must be a non-empty string")
    used_bytes = payload["used_bytes"]
    limit_bytes = payload["limit_bytes"]
    if (
        isinstance(used_bytes, bool)
        or not isinstance(used_bytes, int)
        or used_bytes < 0
        or isinstance(limit_bytes, bool)
        or not isinstance(limit_bytes, int)
        or limit_bytes < 1
        or used_bytes > limit_bytes
    ):
        raise ValueError("usage snapshot byte counts are invalid")
    try:
        captured = datetime.fromisoformat(str(payload["captured_at_utc"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("usage snapshot captured_at_utc is invalid") from error
    if captured.tzinfo is None:
        raise ValueError("usage snapshot captured_at_utc must include a timezone")
    current = datetime.now(timezone.utc) if now is None else now.astimezone(timezone.utc)
    age = (current - captured.astimezone(timezone.utc)).total_seconds()
    if age < -60.0:
        raise ValueError("usage snapshot timestamp is in the future")
    if not math.isfinite(max_age_seconds) or max_age_seconds <= 0 or age > max_age_seconds:
        raise ValueError("usage snapshot is stale")
    return UsageSnapshot(
        captured_at_utc=str(payload["captured_at_utc"]),
        entity=entity,
        plan=str(payload["plan"]),
        used_bytes=used_bytes,
        limit_bytes=limit_bytes,
        retention=str(payload["retention"]),
        source=str(payload["source"]),
    )


class WandbTracker:
    """Own W&B side effects without making them training failure modes."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        evidence_dir: str | Path,
        checkpoint_identity: Mapping[str, Any],
        wandb_module=wandb,
    ) -> None:
        self.cfg = cfg
        self.wandb_cfg = cfg.get("wandb", {}) or {}
        self.evidence_dir = Path(evidence_dir)
        self.checkpoint_identity = dict(checkpoint_identity)
        self.wandb = wandb_module
        self.run = None
        self._uploaded_sha256: set[str] = set()
        self._reserved_sha256: set[str] = set()
        self._usage_snapshot: UsageSnapshot | None = None
        self._max_observed_used_bytes = 0
        self._min_observed_limit_bytes: int | None = None
        self._reserved_bytes_by_tracker = 0
        self._artifact_lock = threading.Lock()
        self._evidence_path = self.evidence_dir / "wandb_events.jsonl"
        self._watched_model = None
        self._scalar_logging_disabled = False
        self._summary_updates_disabled = False
        self._scalar_log_queue: queue.Queue[
            tuple[Any, dict[str, Any], threading.Event, list[Exception]] | None
        ] = queue.Queue(maxsize=1)
        self._scalar_log_worker: threading.Thread | None = None
        self._scalar_log_worker_lock = threading.Lock()
        self._scalar_log_state_lock = threading.Lock()
        self._scalar_log_stop = threading.Event()
        self._watch_cleanup_all = False

    @property
    def mode(self) -> str:
        return str(self.wandb_cfg.get("mode", "disabled"))

    def reset_local_evidence(self) -> None:
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        temporary = self._evidence_path.with_name(f".{self._evidence_path.name}.tmp")
        temporary.write_text("", encoding="utf-8")
        temporary.replace(self._evidence_path)

    def start(self, model) -> None:
        if self.mode == "disabled":
            self._record("init", "disabled")
            return
        try:
            init_timeout = float(self.wandb_cfg.get("init_timeout_seconds", 10.0))
            if self.mode == "online":
                login_succeeded = call_bounded(
                    lambda: self.wandb.login(
                        force=True,
                        verify=True,
                        timeout=max(1, math.ceil(init_timeout)),
                    ),
                    timeout_seconds=init_timeout,
                    operation="W&B login verification",
                )
                if not login_succeeded:
                    raise RuntimeError("verified W&B login did not succeed")
            self.run = call_bounded(
                lambda: self.wandb.init(
                    project=self.wandb_cfg.get("project"),
                    entity=self.wandb_cfg.get("entity"),
                    name=self.wandb_cfg.get("name"),
                    mode=self.mode,
                    config=wandb_run_config(self.cfg),
                    settings=self.wandb.Settings(init_timeout=init_timeout),
                ),
                timeout_seconds=init_timeout,
                operation="W&B initialization",
                on_late_result=self._finish_abandoned_run,
            )
            if self.run is None:
                raise RuntimeError("wandb.init returned no run")
            run_url = getattr(self.run, "url", None) if self.mode == "online" else None
            run_identity = {
                "mode": self.mode,
                "project": self.wandb_cfg.get("project"),
                "entity": self.wandb_cfg.get("entity"),
                "run_id": getattr(self.run, "id", None),
                "run_url": run_url,
            }
            self._record("init", "succeeded", run_identity)
        except Exception as error:  # External tracking cannot destroy local training.
            self.run = None
            self._record_failure("init", error)
            return

        watch = self.wandb_cfg.get("watch", {}) or {}
        if watch.get("enabled", False):
            self._watched_model = model
            watch_timeout = float(self.wandb_cfg.get("log_timeout_seconds", 5.0))
            run = self.run
            try:
                call_bounded(
                    lambda: run.watch(
                        model,
                        log=str(watch.get("log", "gradients")),
                        log_freq=int(watch.get("log_freq", 1000)),
                    ),
                    timeout_seconds=watch_timeout,
                    operation="W&B watch installation",
                    on_late_result=lambda _result: self._cleanup_late_watch(
                        run,
                        model,
                        timeout_seconds=watch_timeout,
                    ),
                    on_late_error=lambda _error: self._cleanup_late_watch(
                        run,
                        model,
                        timeout_seconds=watch_timeout,
                    ),
                )
                self._record("watch", "succeeded")
            except Exception as error:
                self._record_failure("watch", error)
                self._watch_cleanup_all = True
                self._teardown_watch(timeout_seconds=watch_timeout)
        else:
            self._record("watch", "disabled")

        summary = {
            **self._lineage_summary(),
            "tracking/mode": self.mode,
            "tracking/project": self.wandb_cfg.get("project"),
            "tracking/entity": self.wandb_cfg.get("entity"),
            "tracking/run_id": getattr(self.run, "id", None),
            "tracking/run_url": run_url,
        }
        try:
            summary.update(self._runtime_summary())
        except Exception as error:
            self._record_failure("runtime_summary", error)
        self.update_summary(summary)

    def log(self, values: Mapping[str, Any]) -> None:
        completed = threading.Event()
        errors: list[Exception] = []
        queue_full = False
        with self._scalar_log_state_lock:
            if self.run is None or self._scalar_logging_disabled:
                return
            self._start_scalar_log_worker()
            try:
                self._scalar_log_queue.put_nowait((self.run, dict(values), completed, errors))
            except queue.Full:
                queue_full = True
        if queue_full:
            self._disable_scalar_logging(RuntimeError("W&B scalar log queue is full"), values)
            return
        timeout_seconds = float(self.wandb_cfg.get("log_timeout_seconds", 5.0))
        if not completed.wait(timeout_seconds):
            self._disable_scalar_logging(
                TimeoutError(f"W&B scalar log exceeded {timeout_seconds:.3f} seconds"),
                values,
            )
            return
        if not errors:
            return
        self._disable_scalar_logging(errors[0], values)

    def _disable_scalar_logging(
        self,
        error: Exception,
        values: Mapping[str, Any],
    ) -> None:
        with self._scalar_log_state_lock:
            if self._scalar_logging_disabled:
                return
            self._scalar_logging_disabled = True
            self._request_scalar_log_worker_stop()
        self._record_failure(
            "log",
            error,
            {
                "optimizer_step": values.get("optimizer_step"),
                "circuit_breaker": "opened",
            },
        )

    def _start_scalar_log_worker(self) -> None:
        if self._scalar_log_worker is not None:
            return
        with self._scalar_log_worker_lock:
            if self._scalar_log_worker is not None:
                return

            def work() -> None:
                while True:
                    request = self._scalar_log_queue.get()
                    if request is None:
                        return
                    run, values, completed, errors = request
                    if self._scalar_log_stop.is_set():
                        errors.append(RuntimeError("W&B scalar log cancelled during shutdown"))
                        completed.set()
                        self._cancel_queued_scalar_logs()
                        return
                    try:
                        run.log(values)
                    except Exception as error:
                        errors.append(error)
                    finally:
                        completed.set()
                    if self._scalar_log_stop.is_set():
                        self._cancel_queued_scalar_logs()
                        return

            self._scalar_log_worker = threading.Thread(
                target=work,
                name="wandb-scalar-log",
                daemon=True,
            )
            self._scalar_log_worker.start()

    def update_summary(self, values: Mapping[str, Any]) -> None:
        if self.run is None or self._summary_updates_disabled:
            return
        run = self.run
        timeout_seconds = float(self.wandb_cfg.get("log_timeout_seconds", 5.0))
        try:
            call_bounded(
                lambda: run.summary.update(dict(values)),
                timeout_seconds=timeout_seconds,
                operation="W&B summary update",
            )
        except Exception as error:
            self._summary_updates_disabled = True
            self._record_failure(
                "summary",
                error,
                {"circuit_breaker": "opened"},
            )

    def consider_artifact(
        self, *, reason: str, checkpoint_path: str | Path, step: int
    ) -> dict[str, Any]:
        """Upload one selected checkpoint only after every safety gate passes."""

        decision: dict[str, Any] = {
            "policy": str((self.wandb_cfg.get("artifact", {}) or {}).get("policy", "none")),
            "reason": reason,
            "optimizer_step": step,
            "mode": self.mode,
            "auth": {"outcome": "not_checked"},
            "usage": {"outcome": "not_checked"},
            "quota": {"outcome": "not_checked"},
            "reserve_bytes": int(
                (self.wandb_cfg.get("artifact", {}) or {}).get("reserve_bytes", 0)
            ),
            "projected_bytes": None,
            "retry_outcome": "not_attempted",
        }
        if reason not in ARTIFACT_REASONS:
            return self._artifact_blocked(decision, "unsupported_reason")
        profile = self.cfg.get("profile", {}) or {}
        data = self.cfg.get("data", {}) or {}
        if artifact_uploads_forbidden(profile, data_mode=str(data.get("mode", ""))):
            return self._artifact_blocked(decision, "profile_forbids_artifacts")
        if decision["policy"] == "none":
            return self._artifact_blocked(decision, "policy_none")
        if decision["policy"] != reason:
            return self._artifact_blocked(decision, "policy_reason_mismatch")
        if self.mode != "online" or self.run is None:
            return self._artifact_blocked(decision, "online_run_unavailable")
        try:
            candidate = self._candidate(reason, checkpoint_path, expected_step=step)
            decision["checkpoint"] = asdict(candidate)
        except Exception as error:
            return self._artifact_blocked(decision, "checkpoint_identity_unavailable", error)
        artifact_cfg = self.wandb_cfg.get("artifact", {}) or {}
        usage_path = artifact_cfg.get("usage_snapshot_path")
        if not usage_path:
            return self._artifact_blocked(decision, "usage_snapshot_missing")
        try:
            snapshot = load_usage_snapshot(
                usage_path,
                expected_entity=str(self.wandb_cfg.get("entity")),
                max_age_seconds=float(artifact_cfg.get("max_usage_age_seconds", 900.0)),
            )
            decision["usage"] = {"outcome": "visible", **asdict(snapshot)}
        except Exception as error:
            return self._artifact_blocked(decision, "usage_snapshot_invalid", error)

        try:
            timeout = float(self.wandb_cfg.get("init_timeout_seconds", 10.0))
            auth = call_bounded(
                self._authenticated_entity,
                timeout_seconds=timeout,
                operation="W&B entity verification",
            )
            decision["auth"] = {"outcome": "verified", **auth}
        except Exception as error:
            return self._artifact_blocked(
                decision,
                "authentication_or_entity_mismatch",
                error,
            )

        reservation_created = False
        cloud_submission_started = False
        staged_model_artifact: ModelArtifactFile | None = None
        try:
            block_reason = None
            with self._artifact_lock:
                if self._usage_snapshot is None:
                    self._usage_snapshot = snapshot
                self._max_observed_used_bytes = max(
                    self._max_observed_used_bytes,
                    snapshot.used_bytes,
                )
                self._min_observed_limit_bytes = min(
                    value
                    for value in (
                        self._min_observed_limit_bytes,
                        snapshot.limit_bytes,
                    )
                    if value is not None
                )
                if candidate.sha256 in self._uploaded_sha256 | self._reserved_sha256:
                    block_reason = "duplicate_checkpoint"
                else:
                    baseline_used_bytes = self._max_observed_used_bytes
                    effective_limit_bytes = self._min_observed_limit_bytes
                    assert effective_limit_bytes is not None
                    projected = (
                        baseline_used_bytes
                        + self._reserved_bytes_by_tracker
                        + candidate.size_bytes
                        + decision["reserve_bytes"]
                    )
                    decision["projected_bytes"] = projected
                    decision["quota"] = {
                        "outcome": (
                            "allowed" if projected <= effective_limit_bytes else "exceeded"
                        ),
                        "used_bytes": snapshot.used_bytes,
                        "effective_baseline_used_bytes": baseline_used_bytes,
                        "reserved_by_tracker_bytes": self._reserved_bytes_by_tracker,
                        "limit_bytes": snapshot.limit_bytes,
                        "effective_limit_bytes": effective_limit_bytes,
                    }
                    if projected > effective_limit_bytes:
                        block_reason = "visible_quota_insufficient"
                    else:
                        self._reserved_sha256.add(candidate.sha256)
                        self._reserved_bytes_by_tracker += candidate.size_bytes
                        reservation_created = True
            if block_reason is not None:
                return self._artifact_blocked(decision, block_reason)

            upload_timeout = float(artifact_cfg.get("upload_timeout_seconds", 600))

            def prepare_artifact():
                model_artifact = stage_model_artifact(
                    candidate.path,
                    destination_dir=self.evidence_dir / ".wandb-model-staging",
                    expected_source_sha256=candidate.sha256,
                    expected_source_size_bytes=candidate.size_bytes,
                    expected_source_device=candidate.device,
                    expected_source_inode=candidate.inode,
                    expected_source_mtime_ns=candidate.mtime_ns,
                    expected_source_ctime_ns=candidate.ctime_ns,
                )
                try:
                    if model_artifact.size_bytes > candidate.size_bytes:
                        raise RuntimeError(
                            "model artifact exceeds its conservative checkpoint-size reservation"
                        )
                    if model_artifact.sha256 == candidate.sha256:
                        raise RuntimeError(
                            "model artifact physical identity must differ from its source checkpoint"
                        )
                    artifact = self.wandb.Artifact(
                        name=self._artifact_name(),
                        type="model",
                        metadata={
                            "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
                            "reason": reason,
                            "optimizer_step": step,
                            "source_checkpoint_sha256": candidate.sha256,
                            "source_checkpoint_size_bytes": candidate.size_bytes,
                            "model_artifact_sha256": model_artifact.sha256,
                            "model_artifact_size_bytes": model_artifact.size_bytes,
                        },
                    )
                    artifact.add_file(
                        model_artifact.path,
                        name="model.pt",
                        policy="immutable",
                    )
                    try:
                        verify_staged_model_artifact(model_artifact)
                    except Exception as error:
                        raise RuntimeError(
                            "model artifact changed while W&B captured it"
                        ) from error
                    return artifact, model_artifact
                except BaseException:
                    self._cleanup_staged_model_artifact(
                        model_artifact,
                        context="preparation_failure",
                    )
                    raise

            artifact, staged_model_artifact = call_bounded(
                prepare_artifact,
                timeout_seconds=upload_timeout,
                operation="W&B artifact preparation",
                on_late_result=lambda result: self._cleanup_staged_model_artifact(
                    result[1],
                    context="late_preparation_completion",
                ),
            )
            decision["model_artifact"] = {
                "schema_version": MODEL_ARTIFACT_SCHEMA_VERSION,
                "sha256": staged_model_artifact.sha256,
                "size_bytes": staged_model_artifact.size_bytes,
            }

            cloud_submission_started = True

            def submit_and_wait():
                logged = self.run.log_artifact(
                    artifact,
                    aliases=[reason, f"step-{step}", "latest"],
                )
                wait = getattr(logged, "wait", None)
                if not callable(wait):
                    raise RuntimeError("W&B artifact upload exposes no completion wait")
                return wait(timeout=upload_timeout)

            committed = call_bounded(
                submit_and_wait,
                timeout_seconds=upload_timeout,
                operation="W&B artifact upload",
            )
            if getattr(committed, "state", None) != "COMMITTED":
                raise RuntimeError("W&B artifact did not reach COMMITTED state")
            verify_staged_model_artifact(staged_model_artifact)
            with self._artifact_lock:
                self._uploaded_sha256.add(candidate.sha256)
            decision["outcome"] = "uploaded"
            decision["retry_outcome"] = "not_needed"
            decision["artifact"] = {
                "id": getattr(committed, "id", None),
                "name": getattr(committed, "name", None),
                "version": getattr(committed, "version", None),
                "digest": getattr(committed, "digest", None),
                "aliases": [reason, f"step-{step}", "latest"],
            }
            self._record("artifact", "uploaded", decision)
            self._update_artifact_summary(decision)
            self._cleanup_staged_model_artifact(
                staged_model_artifact,
                context="committed_upload",
            )
        except Exception as error:
            if reservation_created and not cloud_submission_started:
                with self._artifact_lock:
                    self._reserved_sha256.discard(candidate.sha256)
                    self._reserved_bytes_by_tracker -= candidate.size_bytes
                self._cleanup_staged_model_artifact(
                    staged_model_artifact,
                    context="pre_submission_failure",
                )
            decision["outcome"] = "upload_failed"
            decision["retry_outcome"] = "operator_retry_required"
            decision["error"] = self._safe_error(error)
            self._record_failure("artifact", error, decision)
            self._update_artifact_summary(decision)
        return decision

    def finish(self) -> None:
        if self.run is None:
            return
        finish_timeout = float(self.wandb_cfg.get("finish_timeout_seconds", 30.0))
        errors: list[Exception] = []
        try:
            worker_stopped = self._stop_scalar_log_worker(timeout_seconds=finish_timeout)
            if not worker_stopped:
                errors.append(
                    TimeoutError(
                        f"W&B scalar log worker exceeded {finish_timeout:.3f} seconds during finish"
                    )
                )
            self._teardown_watch(timeout_seconds=finish_timeout)
            try:
                finish_run_bounded(
                    self.run,
                    timeout_seconds=finish_timeout,
                )
            except Exception as error:
                errors.append(error)
            if errors:
                raise errors[0]
            self._record("finish", "succeeded")
        except Exception as error:
            self._record_failure("finish", error)
        finally:
            self._scalar_logging_disabled = True
            self.run = None
            self._watched_model = None

    def _stop_scalar_log_worker(self, *, timeout_seconds: float) -> bool:
        with self._scalar_log_state_lock:
            self._scalar_logging_disabled = True
            worker = self._scalar_log_worker
            if worker is None:
                return True
            self._request_scalar_log_worker_stop()
        worker.join(timeout_seconds)
        return not worker.is_alive()

    def _request_scalar_log_worker_stop(self) -> None:
        self._scalar_log_stop.set()
        try:
            self._scalar_log_queue.put_nowait(None)
        except queue.Full:
            pass

    def _cancel_queued_scalar_logs(self) -> None:
        while True:
            try:
                request = self._scalar_log_queue.get_nowait()
            except queue.Empty:
                return
            if request is None:
                continue
            _, _, completed, errors = request
            errors.append(RuntimeError("W&B scalar log cancelled during shutdown"))
            completed.set()

    def _teardown_watch(self, *, timeout_seconds: float | None = None) -> None:
        if self.run is None or self._watched_model is None:
            return
        model = self._watched_model
        run = self.run
        timeout = (
            float(self.wandb_cfg.get("log_timeout_seconds", 5.0))
            if timeout_seconds is None
            else timeout_seconds
        )
        try:
            if self._watch_cleanup_all:
                call_bounded(
                    run.unwatch,
                    timeout_seconds=timeout,
                    operation="W&B global unwatch",
                    on_late_result=lambda _result: self._clear_watch_bookkeeping(model),
                )
                self._clear_watch_bookkeeping(model)
            else:
                try:
                    call_bounded(
                        lambda: run.unwatch(model),
                        timeout_seconds=timeout,
                        operation="W&B model unwatch",
                    )
                except Exception:
                    self._watch_cleanup_all = True
                    call_bounded(
                        run.unwatch,
                        timeout_seconds=timeout,
                        operation="W&B global unwatch",
                        on_late_result=lambda _result: self._clear_watch_bookkeeping(model),
                    )
                    self._clear_watch_bookkeeping(model)
            self._watched_model = None
            self._watch_cleanup_all = False
            self._record("unwatch", "succeeded")
        except Exception as error:
            self._record_failure("unwatch", error)

    def _finish_abandoned_run(self, run: Any) -> None:
        if run is None:
            return
        try:
            finish_run_bounded(
                run,
                timeout_seconds=float(self.wandb_cfg.get("finish_timeout_seconds", 30.0)),
            )
        except Exception:
            pass

    @staticmethod
    def _cleanup_late_watch(
        run: Any,
        model: Any,
        *,
        timeout_seconds: float,
    ) -> None:
        try:
            call_bounded(
                run.unwatch,
                timeout_seconds=timeout_seconds,
                operation="W&B late global unwatch",
                on_late_result=lambda _result: WandbTracker._clear_watch_bookkeeping(model),
            )
        except Exception:
            return
        WandbTracker._clear_watch_bookkeeping(model)

    @staticmethod
    def _clear_watch_bookkeeping(model: Any) -> None:
        if hasattr(model, "_wandb_hook_names"):
            delattr(model, "_wandb_hook_names")

    def _candidate(
        self,
        reason: str,
        checkpoint_path: str | Path,
        *,
        expected_step: int,
    ) -> ArtifactCandidate:
        loaded = load_checkpoint_for_generation(checkpoint_path)
        payload = loaded.payload
        checkpoint_kind = str(payload["kind"])
        if checkpoint_kind != reason:
            raise ValueError(
                f"checkpoint kind {checkpoint_kind!r} does not match artifact reason {reason!r}"
            )
        if dict(payload["identity"]) != self.checkpoint_identity:
            raise ValueError("checkpoint identity does not match the active run")
        counters = payload["state"]["counters"]
        checkpoint_step = counters["optimizer_step"]
        if checkpoint_step != expected_step:
            raise ValueError(
                f"checkpoint optimizer_step {checkpoint_step} does not match {expected_step}"
            )
        physical = loaded.physical_identity
        return ArtifactCandidate(
            reason=reason,
            checkpoint_kind=checkpoint_kind,
            checkpoint_optimizer_step=checkpoint_step,
            path=str(physical["path"]),
            sha256=str(physical["sha256"]),
            size_bytes=int(physical["size_bytes"]),
            device=int(physical["device"]),
            inode=int(physical["inode"]),
            mtime_ns=int(physical["mtime_ns"]),
            ctime_ns=int(physical["ctime_ns"]),
        )

    def _authenticated_entity(self) -> dict[str, Any]:
        entity = self.wandb_cfg.get("entity")
        if not isinstance(entity, str) or not entity:
            raise ValueError("wandb.entity is required for artifact uploads")
        timeout = float(self.wandb_cfg.get("init_timeout_seconds", 10.0))
        api = call_bounded(
            lambda: self.wandb.Api(timeout=max(1, math.ceil(timeout))),
            timeout_seconds=timeout,
            operation="W&B entity authentication",
        )
        viewer = api.viewer
        username = getattr(viewer, "username", None)
        team_names = sorted(
            {
                str(value)
                for team in (getattr(viewer, "teams", None) or [])
                for value in (
                    team if isinstance(team, str) else None,
                    getattr(team, "name", None),
                    getattr(team, "entity", None),
                )
                if value
            }
        )
        viewer_entity = getattr(viewer, "entity", None)
        if entity not in {username, viewer_entity, *team_names}:
            raise PermissionError("authenticated W&B viewer cannot verify the configured entity")
        return {"viewer": str(username) if username else None, "entity": entity}

    def _lineage_summary(self) -> dict[str, Any]:
        data_fingerprints = self.checkpoint_identity.get("data_fingerprints", [])
        return {
            "lineage/experiment_id": self.checkpoint_identity.get("experiment_id"),
            "lineage/git_sha": self.checkpoint_identity.get("git_sha"),
            "lineage/config_sha256": self.checkpoint_identity.get("config_sha256"),
            "lineage/lock_sha256": self.checkpoint_identity.get("lock_sha256"),
            "lineage/tokenizer_fingerprint": self.checkpoint_identity.get("tokenizer_fingerprint"),
            "lineage/data_fingerprints": json.dumps(data_fingerprints, separators=(",", ":")),
        }

    @staticmethod
    def _runtime_summary() -> dict[str, Any]:
        environment = collect_environment()
        torch_identity = environment.get("torch", {}) or {}
        cuda_identity = environment.get("cuda", {}) or {}
        return {
            "runtime/host": environment.get("host"),
            "runtime/os": environment.get("os"),
            "runtime/architecture": environment.get("architecture"),
            "runtime/python": environment.get("python"),
            "runtime/torch_version": torch_identity.get("version"),
            "runtime/torch_compiled_cuda": torch_identity.get("compiled_cuda"),
            "runtime/cuda_available": cuda_identity.get("available"),
            "runtime/cuda_runtime_version": cuda_identity.get("runtime_version"),
            "runtime/cuda_driver_version": cuda_identity.get("driver_version"),
            "runtime/cuda_devices": json.dumps(
                cuda_identity.get("devices", []), separators=(",", ":")
            ),
            "runtime/container_image": json.dumps(
                environment.get("container_image", {}), separators=(",", ":")
            ),
        }

    def _update_artifact_summary(self, decision: Mapping[str, Any]) -> None:
        checkpoint = decision.get("checkpoint", {}) or {}
        model_artifact = decision.get("model_artifact", {}) or {}
        artifact = decision.get("artifact", {}) or {}
        self.update_summary(
            {
                "artifact/outcome": decision.get("outcome"),
                "artifact/block_reason": decision.get("block_reason"),
                "artifact/reason": decision.get("reason"),
                "artifact/optimizer_step": decision.get("optimizer_step"),
                "artifact/checkpoint_sha256": checkpoint.get("sha256"),
                "artifact/checkpoint_size_bytes": checkpoint.get("size_bytes"),
                "artifact/model_sha256": model_artifact.get("sha256"),
                "artifact/model_size_bytes": model_artifact.get("size_bytes"),
                "artifact/id": artifact.get("id"),
                "artifact/name": artifact.get("name"),
                "artifact/version": artifact.get("version"),
                "artifact/digest": artifact.get("digest"),
            }
        )

    def _cleanup_staged_model_artifact(
        self,
        artifact: ModelArtifactFile | None,
        *,
        context: str,
    ) -> None:
        if artifact is None:
            return
        try:
            remove_staged_model_artifact(artifact)
        except Exception as error:
            try:
                self._record_failure(
                    "artifact_cleanup",
                    error,
                    {
                        "context": context,
                        "model_artifact_sha256": artifact.sha256,
                        "model_artifact_size_bytes": artifact.size_bytes,
                        "staging_path": artifact.path,
                    },
                )
            except Exception:
                # Cleanup is non-authoritative. Even loss of its local failure
                # event must not rewrite a committed cloud outcome or mask the
                # original pre-submission error.
                pass

    def _artifact_name(self) -> str:
        experiment_id = self.checkpoint_identity.get("experiment_id")
        return f"model-{experiment_id}" if experiment_id else "model-checkpoint"

    def _artifact_blocked(
        self,
        decision: dict[str, Any],
        reason: str,
        error: Exception | None = None,
    ) -> dict[str, Any]:
        decision["outcome"] = "blocked"
        decision["block_reason"] = reason
        if error is not None:
            decision["error"] = self._safe_error(error)
        self._record("artifact", "blocked", decision)
        self._update_artifact_summary(decision)
        return decision

    def _record_failure(
        self,
        action: str,
        error: Exception,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        payload = dict(details or {})
        payload["error"] = self._safe_error(error)
        self._record(action, "failed", payload)

    @staticmethod
    def _safe_error(error: Exception) -> dict[str, str]:
        return safe_external_error(error)

    def _record(
        self,
        action: str,
        outcome: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": 1,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "outcome": outcome,
            "mode": self.mode,
            **dict(details or {}),
        }
        descriptor = os.open(
            self._evidence_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            os.write(descriptor, (json.dumps(record, sort_keys=True) + "\n").encode("utf-8"))
        finally:
            os.close(descriptor)
