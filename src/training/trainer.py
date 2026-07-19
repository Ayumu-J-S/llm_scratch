"""Small, deterministic training loop with explicit work counters.

The trainer deliberately owns the definition of a step and a target token.  A
batch can therefore be partial (padding or a token budget boundary) without
changing the reported objective: NLL is accumulated as a token-weighted sum.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau

from evaluation.scoring import CausalLMScorer, EvaluationResult
from training.checkpoint import (
    CheckpointCompatibilityError,
    CheckpointManager,
    ResumeCheckpoint,
    build_logical_checkpoint_identity,
    build_checkpoint_identity,
    capture_rng_state,
    configured_manifest_fingerprints,
    load_checkpoint_for_generation,
    require_exact_stream_resume_state,
    restore_rng_state,
)
from training.optimization import autocast_context, get_learning_rate
from training.wandb_tracking import WandbTracker


_CHECKPOINT_KINDS = {"recovery", "best", "final", "milestone"}
_MEASUREMENT_BOUNDARY_KEYS = {
    "boundary_index",
    "boundary_id",
    "evidence_id",
    "segment_index",
    "kind",
    "counters",
}
_MEASUREMENT_SEGMENT_KEYS = {
    "segment_index",
    "start_counters",
    "end_counters",
    "resumed_from",
    "parent_boundary_id",
    "measurement",
    "complete",
    "rows",
}


def _resolve_measurement_output_path(checkpoint_dir: Path, configured_output_path: Any) -> Path:
    """Resolve a measurement artifact outside every checkpoint namespace."""

    checkpoint_root = Path(checkpoint_dir).resolve()
    if configured_output_path:
        output_path = Path(str(configured_output_path))
        if not output_path.is_absolute():
            output_path = checkpoint_root.parent / output_path
    else:
        output_path = checkpoint_root.parent / "measurement.json"
    output_path = output_path.resolve()
    if output_path == checkpoint_root or checkpoint_root in output_path.parents:
        raise ValueError("measurement.output_path must resolve outside the checkpoint directory")
    if output_path.exists() and checkpoint_root.is_dir():
        for checkpoint_path in checkpoint_root.rglob("*"):
            if checkpoint_path.is_file() and output_path.samefile(checkpoint_path):
                raise ValueError(
                    "measurement.output_path must not share an inode with a checkpoint artifact"
                )
    return output_path


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Trainer:
    """Train a causal LM until an explicit step, token, or time budget.

    ``metrics`` is intentionally an in-memory/local record independent of W&B;
    each record is also appended to ``metrics.jsonl`` when a checkpoint
    directory is configured.  This makes an offline run auditable in exactly
    the same way as an online run.
    """

    def __init__(
        self,
        *,
        model,
        optimizer,
        scheduler,
        train_loader,
        validation_loader_factory: Callable[[], Any],
        checkpoint_dir: Path,
        cfg: DictConfig,
        device: torch.device,
        checkpoint_identity: dict[str, Any] | None = None,
        resume_path: str | Path | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.validation_loader_factory = validation_loader_factory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.cfg = cfg
        self.device = device
        self.scheduler_interval = self._get_scheduler_interval()

        # These are the authoritative units used by stopping, events, logs,
        # and downstream checkpoint work.  ``train_step`` remains a read-only
        # spelling for existing callers, but is never incremented separately.
        self.optimizer_step = 0
        self.target_tokens = 0
        self.elapsed_seconds = 0.0
        self.metrics: list[dict[str, Any]] = []
        self._last_validation_step: int | None = None
        self._last_checkpoint_step: int | None = None
        self._last_milestone_step: int | None = None
        self._last_log_step: int | None = None
        self._last_token_event_boundary: dict[str, int] = {}
        self._start_time: float | None = None
        self._latest_wandb_scalars: dict[str, Any] = {}
        self._best_checkpoint_path: Path | None = None
        self._expected_best_checkpoint_path = self.checkpoint_dir / "best.pt"
        self._best_checkpoint_step: int | None = None
        measurement = self.cfg.get("measurement", {}) or {}
        self._measurement_enabled = bool(measurement.get("enabled", False))
        self._measurement_warmup_steps = int(measurement.get("warmup_optimizer_steps", 10))
        self._measurement_cuda_events = bool(
            self._measurement_enabled
            and measurement.get("cuda_events", True)
            and self.device.type == "cuda"
        )
        self._measurement_path = _resolve_measurement_output_path(
            self.checkpoint_dir, measurement.get("output_path")
        )
        self._measurement_rows: list[dict[str, Any]] = []
        self._measurement_segments: list[dict[str, Any]] = []
        self._measurement_segment: dict[str, Any] | None = None
        self._measurement_checkpoint_boundaries: list[dict[str, Any]] = []
        self._active_measurement_checkpoint_boundary: dict[str, Any] | None = None
        self._measurement_completed = False
        self._measurement_evidence_id = uuid.uuid4().hex if self._measurement_enabled else None
        self._resume_measurement_config: dict[str, Any] | None = None
        self._resume_measurement_evidence_id: str | None = None
        self._resume_measurement_checkpoint_boundary: dict[str, Any] | None = None

        self.max_steps = self._positive_budget("max_steps")
        self.max_tokens = self._positive_budget("max_tokens")
        self.max_time = self._positive_budget("max_time")
        self.epochs = int(self._training_value("epochs", 1))
        if self.epochs < 1:
            raise ValueError("training.epochs must be positive")
        self.precision = str(self._training_value("precision", "fp32"))
        # Validate the selected precision before the first batch can mutate
        # optimizer state. CPU smoke runs stay explicitly FP32.
        with autocast_context(self.device, self.precision):
            pass
        self.gradient_accumulation_steps = self._positive_training_integer(
            "gradient_accumulation_steps", default=1
        )
        self.max_grad_norm = self._max_grad_norm()
        artifacts = self.cfg.get("artifacts", {}) or {}
        keep_last_n = artifacts.get("keep_last_n", 3)
        self.checkpoint_identity = (
            dict(checkpoint_identity)
            if checkpoint_identity is not None
            else build_checkpoint_identity(self.cfg)
        )
        self.checkpoints = CheckpointManager(
            self.checkpoint_dir,
            keep_last_n=keep_last_n,
            identity=self.checkpoint_identity,
        )
        self.wandb = WandbTracker(
            cfg=self.cfg,
            evidence_dir=self.checkpoint_dir,
            checkpoint_identity=self.checkpoint_identity,
        )
        configured_resume = artifacts.get("resume_path")
        self.resume_path = Path(resume_path) if resume_path is not None else configured_resume
        self._resumed_from: ResumeCheckpoint | None = None
        self._best_validation_loss: float | None = None
        self.validation_scorer = CausalLMScorer(
            device=self.device,
            precision=self.precision,
            ignore_index=int(self._training_value("ignore_index", -100)),
            measure_phase_timing=self._measurement_enabled,
            cuda_events=self._measurement_cuda_events,
        )
        if self.resume_path is not None:
            self._restore_checkpoint(self.resume_path)

    @property
    def train_step(self) -> int:
        """Compatibility readout for the authoritative optimizer step."""

        return self.optimizer_step

    def fit(self) -> list[dict[str, Any]]:
        """Run training and return the local metric records."""

        # A checkpoint directory can be reused by a later run. Metrics are
        # run-local evidence, not append-only checkpoint state. Validate local
        # measurement evidence before starting an external tracker.
        self.metrics.clear()
        self._latest_wandb_scalars.clear()
        if self._resumed_from is None:
            self._reset_local_metrics()
            self.wandb.reset_local_evidence()
        self._initialize_measurements()
        self.wandb.start(self.model)
        self._start_time = time.monotonic() - self.elapsed_seconds
        saw_batch = False

        logger.info("Training...")
        try:
            for epoch_index in range(self.epochs):
                self.model.train()
                self._latest_validation_loss = None
                epoch_loss_sum = 0.0
                epoch_tokens = 0
                epoch_saw_batch = False
                iterator = iter(self.train_loader)
                batch_index = 0
                try:
                    while not self._budget_reached():
                        self._update_elapsed()
                        step_started = time.perf_counter() if self._measurement_enabled else None
                        wall_start_unix_ns = time.time_ns() if self._measurement_enabled else None
                        if self._measurement_cuda_events:
                            torch.cuda.reset_peak_memory_stats(self.device)
                        wait_started = time.perf_counter() if self._measurement_enabled else None
                        try:
                            batch = next(iterator)
                        except StopIteration:
                            break
                        initial_data_wait_seconds = (
                            time.perf_counter() - wait_started if wait_started is not None else 0.0
                        )
                        batch_index += 1
                        (
                            loss_sum,
                            token_count,
                            micro_batches,
                            gradient_norm,
                            clipped,
                            learning_rate_used,
                            step_timing,
                        ) = self._train_update(
                            batch,
                            iterator=iterator,
                            first_batch_index=batch_index,
                        )
                        if step_timing is not None:
                            step_timing["host_seconds"]["data_wait"] += initial_data_wait_seconds
                            step_timing["data_wait_calls"] += 1
                        batch_index += micro_batches - 1
                        if token_count == 0:
                            break
                        epoch_saw_batch = saw_batch = True
                        epoch_loss_sum += loss_sum
                        epoch_tokens += token_count
                        self._update_elapsed()
                        metrics_started = time.perf_counter() if self._measurement_enabled else None
                        self._record_step_metrics(
                            loss_sum / token_count,
                            token_count,
                            micro_batches=micro_batches,
                            gradient_norm=gradient_norm,
                            clipped=clipped,
                            learning_rate_used=learning_rate_used,
                        )
                        if step_timing is not None:
                            step_timing["host_seconds"]["step_metrics"] += (
                                time.perf_counter() - metrics_started
                            )
                            self._finish_step_measurement(
                                step_timing,
                                step_started=step_started,
                                wall_start_unix_ns=wall_start_unix_ns,
                                target_tokens_step=token_count,
                                micro_batches=micro_batches,
                            )
                        self._run_events(epoch_end=False)
                        if self._budget_reached():
                            break
                finally:
                    self._close_train_iterator(iterator)

                if not epoch_saw_batch:
                    if not saw_batch:
                        raise ValueError("training loader is empty; no optimizer steps were taken")
                    break

                # With no explicit step cadence, epoch end is the event
                # boundary. Explicit cadences remain independent.
                train_loss = epoch_loss_sum / epoch_tokens
                self._record_metrics(
                    {
                        "event": "epoch_summary",
                        "epoch": epoch_index + 1,
                        "optimizer_step": self.optimizer_step,
                        "target_tokens": self.target_tokens,
                        "elapsed_seconds": self.elapsed_seconds,
                        "train/loss": train_loss,
                        "train/perplexity": _perplexity(train_loss),
                        "optimizer/lr": get_learning_rate(self.optimizer),
                    },
                    send_to_wandb=False,
                )
                self._run_events(epoch_end=True, train_loss=train_loss)
                if self.scheduler is not None and self.scheduler_interval == "epoch":
                    # Metric-free schedulers advance at the epoch boundary;
                    # ReduceLROnPlateau alone requires a validation metric.
                    validation_loss = self._latest_validation_loss
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        if validation_loss is None:
                            self._run_events(epoch_end=True, force_validation=True)
                            validation_loss = self._latest_validation_loss
                            if validation_loss is None:
                                raise RuntimeError("ReduceLROnPlateau validation produced no loss")
                        self._step_scheduler(validation_loss)
                    else:
                        self._step_scheduler()

                if self._budget_reached():
                    break

            if not saw_batch:
                raise ValueError("training loader is empty; no optimizer steps were taken")
            self._update_elapsed()
            final_checkpoint_started = time.perf_counter()
            final_path = self._save_final_checkpoint()
            final_checkpoint_seconds = time.perf_counter() - final_checkpoint_started
            self._record_metrics(
                {
                    "event": "final_checkpoint",
                    "optimizer_step": self.optimizer_step,
                    "target_tokens": self.target_tokens,
                    "elapsed_seconds": self.elapsed_seconds,
                    "checkpoint": str(final_path),
                    **self._checkpoint_measurement_metrics(),
                },
                send_to_wandb=False,
            )
            artifact_policy = str(
                ((self.cfg.get("wandb", {}) or {}).get("artifact", {}) or {}).get("policy", "none")
            )
            if artifact_policy == "best":
                best_artifact_path = (
                    self._best_checkpoint_path or self._expected_best_checkpoint_path
                )
                self.wandb.consider_artifact(
                    reason="best",
                    checkpoint_path=best_artifact_path,
                    step=(
                        self._best_checkpoint_step
                        if self._best_checkpoint_step is not None
                        else self.optimizer_step
                    ),
                )
            elif artifact_policy == "final":
                self.wandb.consider_artifact(
                    reason="final",
                    checkpoint_path=final_path,
                    step=self.optimizer_step,
                )
            self.wandb.update_summary(
                {
                    "run/final_optimizer_step": self.optimizer_step,
                    "run/final_target_tokens": self.target_tokens,
                    "run/final_elapsed_seconds": self.elapsed_seconds,
                    **self._latest_wandb_scalars,
                }
            )
            if self._measurement_enabled:
                self._measurement_rows.append(
                    {
                        "event": "final_checkpoint",
                        "optimizer_step": self.optimizer_step,
                        "target_tokens": self.target_tokens,
                        "checkpoint_seconds": final_checkpoint_seconds,
                        **self._checkpoint_measurement_metrics(),
                    }
                )
                self._measurement_completed = True
            return list(self.metrics)
        finally:
            self._flush_measurements()
            self.wandb.finish()

    def _train_update(
        self,
        first_batch: dict[str, torch.Tensor],
        *,
        iterator,
        first_batch_index: int,
    ) -> tuple[float, int, int, float, bool, float, dict[str, Any] | None]:
        """Accumulate a token-weighted gradient and perform one optimizer update."""

        ignore_index = int(self._training_value("ignore_index", -100))
        learning_rate_used = get_learning_rate(self.optimizer)
        self.optimizer.zero_grad(set_to_none=True)
        total_tokens = 0
        total_loss_sum: torch.Tensor | None = None
        micro_batches = 0
        batch = first_batch
        batch_index = first_batch_index
        timing = self._new_step_measurement() if self._measurement_enabled else None

        while micro_batches < self.gradient_accumulation_steps:
            phase = self._start_measurement_phase() if timing is not None else None
            input_batch = batch["inputs"].to(self.device)
            label_batch = batch["labels"].to(self.device)
            if timing is not None:
                self._end_measurement_phase(timing, "host_device_prepare", phase)
            phase = self._start_measurement_phase() if timing is not None else None
            with autocast_context(self.device, self.precision):
                logits = self.model(input_batch)
            if timing is not None:
                self._end_measurement_phase(timing, "forward", phase)
            phase = self._start_measurement_phase() if timing is not None else None
            with autocast_context(self.device, self.precision):
                flat_labels = label_batch.reshape(-1)
                flat_losses = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    flat_labels,
                    reduction="none",
                    ignore_index=ignore_index,
                )
            if timing is not None:
                self._end_measurement_phase(timing, "loss", phase)
            valid_indices = torch.nonzero(flat_labels != ignore_index, as_tuple=False).flatten()
            if valid_indices.numel() == 0:
                raise ValueError("training batch contains zero target tokens")

            remaining = self._remaining_tokens(total_tokens)
            if remaining is not None:
                valid_indices = valid_indices[:remaining]
                if valid_indices.numel() == 0:
                    break
            selected_loss_sum = flat_losses.index_select(0, valid_indices).sum()
            token_count = int(valid_indices.numel())
            micro_loss = selected_loss_sum / token_count
            if not torch.isfinite(micro_loss):
                self._record_numeric_failure("loss", batch_index)
                raise FloatingPointError(self._numeric_failure_message("loss", batch_index))
            phase = self._start_measurement_phase() if timing is not None else None
            selected_loss_sum.backward()
            if timing is not None:
                self._end_measurement_phase(timing, "backward", phase)
            total_loss_sum = (
                selected_loss_sum.detach()
                if total_loss_sum is None
                else total_loss_sum + selected_loss_sum.detach()
            )
            total_tokens += token_count
            micro_batches += 1

            if self._remaining_tokens(total_tokens) == 0:
                break
            if micro_batches == self.gradient_accumulation_steps:
                break
            wait_started = time.perf_counter() if timing is not None else None
            try:
                batch = next(iterator)
            except StopIteration:
                break
            if timing is not None:
                timing["host_seconds"]["data_wait"] += time.perf_counter() - wait_started
                timing["data_wait_calls"] += 1
            batch_index += 1

        if total_tokens == 0 or total_loss_sum is None:
            self.optimizer.zero_grad(set_to_none=True)
            return 0.0, 0, micro_batches, 0.0, False, learning_rate_used, timing

        phase = self._start_measurement_phase() if timing is not None else None
        self._scale_gradients(1.0 / total_tokens)
        if timing is not None:
            self._end_measurement_phase(timing, "gradient_scale", phase)
        phase = self._start_measurement_phase() if timing is not None else None
        if not self._gradients_are_finite():
            self._record_numeric_failure("gradients", batch_index)
            raise FloatingPointError(self._numeric_failure_message("gradients", batch_index))
        if timing is not None:
            self._end_measurement_phase(timing, "gradient_finite_check", phase)
        phase = self._start_measurement_phase() if timing is not None else None
        gradient_norm = self._global_gradient_norm()
        if not torch.isfinite(gradient_norm):
            self._record_numeric_failure("gradient_norm", batch_index)
            raise FloatingPointError(self._numeric_failure_message("gradient_norm", batch_index))
        gradient_norm_value = float(gradient_norm.item())
        if timing is not None:
            self._end_measurement_phase(timing, "gradient_norm_and_scalar_read", phase)
        clipped = self.max_grad_norm is not None and gradient_norm_value > self.max_grad_norm
        phase = self._start_measurement_phase() if timing is not None else None
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm, error_if_nonfinite=False
            )
        if timing is not None:
            self._end_measurement_phase(timing, "clipping", phase)
        phase = self._start_measurement_phase() if timing is not None else None
        self.optimizer.step()
        if timing is not None:
            self._end_measurement_phase(timing, "optimizer", phase)
        phase = self._start_measurement_phase() if timing is not None else None
        if not self._parameters_are_finite():
            self._record_numeric_failure("parameters", batch_index)
            raise FloatingPointError(self._numeric_failure_message("parameters", batch_index))
        if timing is not None:
            self._end_measurement_phase(timing, "parameter_finite_check", phase)
        self.optimizer_step += 1
        self.target_tokens += total_tokens
        if self.scheduler is not None and self.scheduler_interval == "step":
            # A scheduler observes the update only after optimizer.step and
            # after the authoritative step counter advances.
            phase = self._start_measurement_phase() if timing is not None else None
            self._step_scheduler()
            if timing is not None:
                self._end_measurement_phase(timing, "scheduler", phase)
        phase = self._start_measurement_phase() if timing is not None else None
        loss_sum_value = float(total_loss_sum.item())
        if timing is not None:
            self._end_measurement_phase(timing, "loss_scalar_read", phase)
        return (
            loss_sum_value,
            total_tokens,
            micro_batches,
            gradient_norm_value,
            clipped,
            learning_rate_used,
            timing,
        )

    def _evaluate(self) -> EvaluationResult:
        """Score one fresh fixed validation window pass through the shared scorer."""

        namespace = "memorization" if self._is_memorization_run() else "validation"
        try:
            return self.validation_scorer.score(
                self.model,
                self.validation_loader_factory,
                namespace=namespace,
                logical_checkpoint_identity=self._logical_checkpoint_identity(),
                configured_data_fingerprints=configured_manifest_fingerprints(self.cfg),
            )
        except FloatingPointError as error:
            self._record_numeric_failure("validation", getattr(error, "batch_index", None))
            raise

    def _run_events(
        self,
        *,
        epoch_end: bool,
        train_loss: float | None = None,
        force_validation: bool = False,
    ) -> None:
        step = self.optimizer_step
        if step < 1:
            return
        self._latest_validation_loss = getattr(self, "_latest_validation_loss", None)

        should_log = self._event_due("log_every_n_steps", "log_every_n_tokens", epoch_end)
        scheduled_log_pending = should_log and self._last_log_step != step

        should_validate = force_validation or self._event_due(
            "validation_every_n_steps", "validation_every_n_tokens", epoch_end
        )
        validation_for_scheduled_log: dict[str, Any] = {}
        if should_validate and self._last_validation_step != step:
            validation_event_started = time.perf_counter() if self._measurement_enabled else None
            validation_wall_start_unix_ns = time.time_ns() if self._measurement_enabled else None
            pre_boundary_sync_seconds = self._measurement_boundary_sync()
            scoring_started = time.perf_counter() if self._measurement_enabled else None
            validation_result = self._evaluate()
            scoring_seconds = (
                time.perf_counter() - scoring_started
                if scoring_started is not None
                else validation_result.pause_seconds
            )
            self._update_elapsed()
            validation_loss = validation_result.nll
            self._latest_validation_loss = validation_loss
            self._last_validation_step = step
            best_checkpoint_seconds = 0.0
            best_checkpoint_written = False
            best_checkpoint_measurement: dict[str, float | int] = {}
            if not self._is_memorization_run() and (
                self._best_validation_loss is None or validation_loss < self._best_validation_loss
            ):
                self._best_validation_loss = validation_loss
                self._best_checkpoint_step = step
                checkpoint_started = time.perf_counter()
                best_path = self._save_best_checkpoint()
                self._best_checkpoint_path = best_path
                best_checkpoint_seconds = time.perf_counter() - checkpoint_started
                best_checkpoint_written = True
                best_checkpoint_measurement = self._checkpoint_measurement_metrics()
                self._record_metrics(
                    {
                        "event": "best_checkpoint",
                        "optimizer_step": step,
                        "target_tokens": self.target_tokens,
                        "elapsed_seconds": self.elapsed_seconds,
                        f"{validation_result.namespace}/loss": validation_loss,
                        "checkpoint": str(best_path),
                        **self._checkpoint_measurement_metrics(),
                    },
                    send_to_wandb=False,
                )
            validation_metrics_started = time.perf_counter() if self._measurement_enabled else None
            compact_validation = self._record_validation_metrics(validation_result)
            if scheduled_log_pending:
                validation_for_scheduled_log = compact_validation
            else:
                validation_log_started = time.perf_counter() if self._measurement_enabled else None
                self.wandb.log(
                    {
                        "event": f"{validation_result.namespace}_log",
                        "optimizer_step": step,
                        "target_tokens": self.target_tokens,
                        "elapsed_seconds": self.elapsed_seconds,
                        **compact_validation,
                        **self._system_wandb_scalars(),
                    }
                )
                if validation_log_started is not None:
                    self._measurement_rows.append(
                        {
                            "event": "validation_log",
                            "optimizer_step": step,
                            "target_tokens": self.target_tokens,
                            "validation_log_seconds": (
                                time.perf_counter() - validation_log_started
                            ),
                        }
                    )
            validation_metrics_seconds = (
                time.perf_counter() - validation_metrics_started
                if validation_metrics_started is not None
                else 0.0
            )
            if validation_event_started is not None:
                post_boundary_sync_seconds = self._measurement_boundary_sync()
                full_event_pause_seconds = time.perf_counter() - validation_event_started
                attributed_seconds = (
                    pre_boundary_sync_seconds
                    + scoring_seconds
                    + best_checkpoint_seconds
                    + validation_metrics_seconds
                    + post_boundary_sync_seconds
                )
                self._measurement_rows.append(
                    {
                        "event": "validation",
                        "optimizer_step": step,
                        "target_tokens": self.target_tokens,
                        "trigger": self._event_trigger("validation", epoch_end),
                        "wall_start_unix_ns": validation_wall_start_unix_ns,
                        "wall_end_unix_ns": time.time_ns(),
                        "pre_boundary_sync_seconds": pre_boundary_sync_seconds,
                        "scoring_seconds": scoring_seconds,
                        "best_checkpoint_written": best_checkpoint_written,
                        "best_checkpoint_seconds": best_checkpoint_seconds,
                        "best_checkpoint_write_seconds": best_checkpoint_measurement.get(
                            "checkpoint/write_seconds", 0.0
                        ),
                        "best_checkpoint_verification_seconds": (
                            best_checkpoint_measurement.get("checkpoint/verification_seconds", 0.0)
                        ),
                        "validation_metrics_seconds": validation_metrics_seconds,
                        "post_boundary_sync_seconds": post_boundary_sync_seconds,
                        "full_event_pause_seconds": full_event_pause_seconds,
                        "unattributed_seconds": full_event_pause_seconds - attributed_seconds,
                        "evaluated_windows": validation_result.evaluated_windows,
                        "evaluated_target_tokens": validation_result.target_tokens,
                        "nll": validation_result.nll,
                        "manifest_identity": validation_result.manifest_identity,
                        "logical_checkpoint_identity": (
                            validation_result.logical_checkpoint_identity
                        ),
                        "evaluated_window_sha256": (validation_result.evaluated_window_sha256),
                        "evaluated_token_sha256": validation_result.evaluated_token_sha256,
                    }
                )

        if scheduled_log_pending:
            self._last_log_step = step
            latest_scalars = {
                key: value
                for key, value in self._latest_wandb_scalars.items()
                if not (key.startswith("validation/") or key.startswith("memorization/"))
            }
            if epoch_end and train_loss is not None:
                latest_scalars["train/nll"] = train_loss
                latest_scalars["train/perplexity"] = _perplexity(train_loss)
            values = {
                "event": "log",
                "optimizer_step": step,
                "target_tokens": self.target_tokens,
                "elapsed_seconds": self.elapsed_seconds,
                **latest_scalars,
                **validation_for_scheduled_log,
                **self._system_wandb_scalars(),
            }
            scheduled_log_started = time.perf_counter() if self._measurement_enabled else None
            self._record_metrics(values, send_to_wandb=True)
            if scheduled_log_started is not None:
                self._measurement_rows.append(
                    {
                        "event": "scheduled_log",
                        "optimizer_step": step,
                        "target_tokens": self.target_tokens,
                        "scheduled_log_seconds": time.perf_counter() - scheduled_log_started,
                    }
                )

        should_save = self._event_due(
            "checkpoint_every_n_steps", "checkpoint_every_n_tokens", epoch_end
        )
        if should_save and self._last_checkpoint_step != step:
            checkpoint_started = time.perf_counter()
            checkpoint_path = self._save_checkpoint()
            checkpoint_seconds = time.perf_counter() - checkpoint_started
            self._last_checkpoint_step = step
            self._record_metrics(
                {
                    "event": "checkpoint",
                    "optimizer_step": step,
                    "target_tokens": self.target_tokens,
                    "elapsed_seconds": self.elapsed_seconds,
                    "checkpoint": str(checkpoint_path),
                    **self._checkpoint_measurement_metrics(),
                },
                send_to_wandb=False,
            )
            if self._measurement_enabled:
                self._measurement_rows.append(
                    {
                        "event": "checkpoint",
                        "optimizer_step": step,
                        "target_tokens": self.target_tokens,
                        "checkpoint_seconds": checkpoint_seconds,
                        **self._checkpoint_measurement_metrics(),
                    }
                )

        # Milestones are retention-class checkpoints, not aliases for the
        # rotating recovery file or a W&B-only implementation detail.
        milestone_due = self._event_due(
            "milestone_every_n_steps", "milestone_every_n_tokens", False
        )
        milestone_path: Path | None = None
        if milestone_due and self._last_milestone_step != step:
            milestone_started = time.perf_counter()
            milestone_path = self._save_milestone_checkpoint()
            milestone_seconds = time.perf_counter() - milestone_started
            self._last_milestone_step = step
            self._record_metrics(
                {
                    "event": "milestone",
                    "optimizer_step": step,
                    "target_tokens": self.target_tokens,
                    "elapsed_seconds": self.elapsed_seconds,
                    "checkpoint": str(milestone_path),
                    **self._checkpoint_measurement_metrics(),
                },
                send_to_wandb=False,
            )
            if self._measurement_enabled:
                self._measurement_rows.append(
                    {
                        "event": "milestone",
                        "optimizer_step": step,
                        "target_tokens": self.target_tokens,
                        "checkpoint_seconds": milestone_seconds,
                        **self._checkpoint_measurement_metrics(),
                    }
                )
        artifact_policy = str(
            ((self.cfg.get("wandb", {}) or {}).get("artifact", {}) or {}).get("policy", "none")
        )
        if milestone_path is not None and artifact_policy == "milestone":
            self.wandb.consider_artifact(
                reason="milestone",
                checkpoint_path=milestone_path,
                step=step,
            )

    def _record_step_metrics(
        self,
        loss: float,
        token_count: int,
        *,
        micro_batches: int,
        gradient_norm: float,
        clipped: bool,
        learning_rate_used: float,
    ) -> None:
        values = {
            "event": "step",
            "optimizer_step": self.optimizer_step,
            "target_tokens": self.target_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            "train/loss_step": loss,
            "train/target_tokens_step": token_count,
            "train/effective_target_tokens_update": token_count,
            "train/effective_target_tokens_configured": self._configured_effective_tokens(),
            "train/micro_batches_per_update": micro_batches,
            "optimizer/gradient_norm": gradient_norm,
            "optimizer/gradient_clipped": clipped,
            "optimizer/lr_used": learning_rate_used,
            "optimizer/lr": get_learning_rate(self.optimizer),
        }
        self._latest_wandb_scalars.update(
            {
                "train/nll": loss,
                "train/perplexity": _perplexity(loss),
                "train/target_tokens_step": token_count,
                "train/micro_batches_per_update": micro_batches,
                "optimizer/gradient_norm": gradient_norm,
                "optimizer/gradient_clipped": int(clipped),
                "optimizer/nonfinite_count": 0,
                "optimizer/lr_used": learning_rate_used,
                "optimizer/lr": get_learning_rate(self.optimizer),
            }
        )
        self._record_metrics(values, send_to_wandb=False)

    def _record_validation_metrics(self, validation_result: EvaluationResult) -> dict[str, Any]:
        namespace = validation_result.namespace
        values: dict[str, Any] = {
            "event": namespace,
            "optimizer_step": self.optimizer_step,
            "target_tokens": self.target_tokens,
            "elapsed_seconds": self.elapsed_seconds,
            f"{namespace}/loss": validation_result.nll,
            f"{namespace}/perplexity": validation_result.perplexity,
            f"{namespace}/perplexity_overflow": validation_result.aggregate.perplexity_overflow,
            f"{namespace}/scorer_revision": validation_result.scorer_revision,
            f"{namespace}/target_tokens": validation_result.target_tokens,
            f"{namespace}/evaluated_windows": validation_result.evaluated_windows,
            f"{namespace}/evaluated_window_sha256": validation_result.evaluated_window_sha256,
            f"{namespace}/evaluated_token_sha256": validation_result.evaluated_token_sha256,
            f"{namespace}/pause_seconds": validation_result.pause_seconds,
            f"{namespace}/timing": validation_result.timing,
            f"{namespace}/evaluated_targets_per_second": (
                validation_result.evaluated_targets_per_second
            ),
            f"{namespace}/manifest_identity": validation_result.manifest_identity,
            f"{namespace}/logical_checkpoint_identity": (
                validation_result.logical_checkpoint_identity
            ),
            f"{namespace}/physical_checkpoint_identity": (
                validation_result.physical_checkpoint_identity
            ),
            f"{namespace}/by_corpus": {
                name: score.as_dict() for name, score in sorted(validation_result.by_corpus.items())
            },
        }
        compact = {
            f"{namespace}/nll": validation_result.nll,
            f"{namespace}/perplexity": validation_result.perplexity,
            f"{namespace}/perplexity_overflow": validation_result.aggregate.perplexity_overflow,
            f"{namespace}/target_tokens": validation_result.target_tokens,
            f"{namespace}/evaluated_windows": validation_result.evaluated_windows,
            f"{namespace}/pause_seconds": validation_result.pause_seconds,
            f"{namespace}/evaluated_targets_per_second": (
                validation_result.evaluated_targets_per_second
            ),
        }
        for name, score in sorted(validation_result.by_corpus.items()):
            compact[f"{namespace}/corpus/{name}/nll"] = score.nll
            compact[f"{namespace}/corpus/{name}/perplexity"] = score.perplexity
            compact[f"{namespace}/corpus/{name}/perplexity_overflow"] = score.perplexity_overflow
            compact[f"{namespace}/corpus/{name}/target_tokens"] = score.target_tokens
        self._latest_wandb_scalars.update(compact)
        self._record_metrics(
            values,
            send_to_wandb=False,
            preserve_none=(f"{namespace}/perplexity",),
        )
        return compact

    def _is_memorization_run(self) -> bool:
        data = self.cfg.get("data", {}) or {}
        profile = self.cfg.get("profile", {}) or {}
        return data.get("mode") == "memorization_smoke" or str(
            profile.get("purpose", "")
        ).startswith("memorization")

    def _logical_checkpoint_identity(self) -> dict[str, Any]:
        return build_logical_checkpoint_identity(self.checkpoint_identity, self.counters)

    def _record_metrics(
        self,
        values: dict[str, Any],
        *,
        send_to_wandb: bool,
        preserve_none: tuple[str, ...] = (),
    ) -> None:
        record = {
            key: value for key, value in values.items() if value is not None or key in preserve_none
        }
        self.metrics.append(record)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with (self.checkpoint_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if send_to_wandb:
            self.wandb.log(record)

    def _reset_local_metrics(self) -> None:
        """Atomically start a fresh local evidence stream after run init."""

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self.checkpoint_dir / "metrics.jsonl"
        temporary_path = metrics_path.with_name(f".{metrics_path.name}.tmp")
        temporary_path.write_text("", encoding="utf-8")
        temporary_path.replace(metrics_path)

    def _system_wandb_scalars(self) -> dict[str, float | int]:
        values: dict[str, float | int] = {
            "throughput/target_tokens_per_second": (
                self.target_tokens / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0
            )
        }
        try:
            status = Path("/proc/self/status").read_text(encoding="utf-8")
        except OSError:
            status = ""
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                values["system/rss_bytes"] = int(line.split()[1]) * 1024
            elif line.startswith("VmHWM:"):
                values["system/rss_peak_bytes"] = int(line.split()[1]) * 1024
        if self.device.type == "cuda":
            values.update(
                {
                    "system/cuda_peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(self.device)
                    ),
                    "system/cuda_peak_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(self.device)
                    ),
                }
            )
        return values

    def _new_step_measurement(self) -> dict[str, Any]:
        phases = (
            "data_wait",
            "host_device_prepare",
            "forward",
            "loss",
            "backward",
            "gradient_scale",
            "gradient_finite_check",
            "gradient_norm_and_scalar_read",
            "clipping",
            "optimizer",
            "parameter_finite_check",
            "loss_scalar_read",
            "scheduler",
            "step_metrics",
        )
        return {
            "host_seconds": dict.fromkeys(phases, 0.0),
            "cuda_event_pairs": {},
            "data_wait_calls": 0,
        }

    def _start_measurement_phase(
        self,
    ) -> tuple[float, torch.cuda.Event | None, torch.cuda.Event | None]:
        started = time.perf_counter()
        if not self._measurement_cuda_events:
            return started, None, None
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return started, start_event, end_event

    def _end_measurement_phase(
        self,
        timing: dict[str, Any] | None,
        name: str,
        phase: tuple[float, torch.cuda.Event | None, torch.cuda.Event | None] | None,
    ) -> None:
        if timing is None or phase is None:
            return
        started, start_event, end_event = phase
        if end_event is not None:
            end_event.record()
            timing["cuda_event_pairs"].setdefault(name, []).append((start_event, end_event))
        timing["host_seconds"][name] += time.perf_counter() - started

    def _measurement_boundary_sync(self) -> float:
        if not self._measurement_cuda_events:
            return 0.0
        started = time.perf_counter()
        torch.cuda.synchronize(self.device)
        return time.perf_counter() - started

    def _finish_step_measurement(
        self,
        timing: dict[str, Any],
        *,
        step_started: float | None,
        wall_start_unix_ns: int | None,
        target_tokens_step: int,
        micro_batches: int,
    ) -> None:
        if step_started is None or wall_start_unix_ns is None:
            raise RuntimeError("enabled measurement is missing its step boundary")
        boundary_sync_seconds = self._measurement_boundary_sync()
        wall_end_unix_ns = time.time_ns()
        step_wall_seconds = time.perf_counter() - step_started
        cuda_milliseconds: dict[str, float] = {}
        for name, pairs in timing.pop("cuda_event_pairs").items():
            cuda_milliseconds[name] = sum(start.elapsed_time(end) for start, end in pairs)
        host_seconds = timing["host_seconds"]
        attributed_host_seconds = sum(host_seconds.values()) + boundary_sync_seconds
        if self.device.type == "cuda":
            allocated_bytes = int(torch.cuda.max_memory_allocated(self.device))
            reserved_bytes = int(torch.cuda.max_memory_reserved(self.device))
        else:
            allocated_bytes = 0
            reserved_bytes = 0
        self._measurement_rows.append(
            {
                "event": "optimizer_step",
                "optimizer_step": self.optimizer_step,
                "target_tokens": self.target_tokens,
                "target_tokens_step": target_tokens_step,
                "micro_batches": micro_batches,
                "warmup": self.optimizer_step <= self._measurement_warmup_steps,
                "wall_start_unix_ns": wall_start_unix_ns,
                "wall_end_unix_ns": wall_end_unix_ns,
                "step_wall_seconds": step_wall_seconds,
                "data_wait_calls": timing["data_wait_calls"],
                "host_seconds": host_seconds,
                "cuda_milliseconds": cuda_milliseconds,
                "boundary_sync_seconds": boundary_sync_seconds,
                "host_reconciliation_error_seconds": (step_wall_seconds - attributed_host_seconds),
                "pytorch_allocated_bytes": allocated_bytes,
                "pytorch_reserved_bytes": reserved_bytes,
            }
        )

    def _flush_measurements(self) -> None:
        if not self._measurement_enabled:
            return
        if self._measurement_segment is None:
            raise RuntimeError("enabled measurement has no active evidence segment")
        self._measurement_segment["end_counters"] = dict(self.counters)
        self._measurement_segment["complete"] = self._measurement_completed
        payload = {
            "schema_version": 3,
            "checkpoint_identity": self.checkpoint_identity,
            "measurement_evidence_id": self._measurement_evidence_id,
            "complete": self._measurement_completed,
            "segments": self._measurement_segments,
            "checkpoint_boundaries": self._measurement_checkpoint_boundaries,
        }
        self._measurement_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self._measurement_path.name}.",
            suffix=".tmp",
            dir=str(self._measurement_path.parent),
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self._measurement_path)
            _fsync_directory(self._measurement_path.parent)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def _initialize_measurements(self) -> None:
        self._measurement_rows.clear()
        self._measurement_segments.clear()
        self._measurement_checkpoint_boundaries.clear()
        self._measurement_segment = None
        self._active_measurement_checkpoint_boundary = None
        self._measurement_completed = False
        if not self._measurement_enabled:
            return

        prior_path: Path | None = None
        prior_status = "fresh"
        if self._resumed_from is not None:
            prior_config = self._resume_measurement_config or {}
            if bool(prior_config.get("enabled", False)):
                if self._resume_measurement_evidence_id is None:
                    raise ValueError("resume checkpoint has no measurement evidence identity")
                self._measurement_evidence_id = self._resume_measurement_evidence_id
                prior_path = _resolve_measurement_output_path(
                    self._resumed_from.path.parent,
                    prior_config.get("output_path"),
                )
                segments, boundaries = self._load_measurement_evidence(prior_path)
                self._measurement_segments.extend(segments)
                self._measurement_checkpoint_boundaries.extend(boundaries)
                prior_status = "verified"
            else:
                prior_status = "disabled"

        if self._measurement_path.exists() and self._measurement_path != prior_path:
            raise ValueError(
                "measurement output already exists without matching resume evidence: "
                f"{self._measurement_path}"
            )

        start_counters = dict(self.counters)
        resumed_from = None
        if self._resumed_from is not None:
            resumed_from = {
                "path": str(self._resumed_from.path.resolve()),
                "counters": start_counters,
                "prior_measurement": {
                    "status": prior_status,
                    "path": str(prior_path) if prior_path is not None else None,
                },
            }
        self._measurement_segment = {
            "segment_index": len(self._measurement_segments),
            "start_counters": start_counters,
            "end_counters": start_counters,
            "resumed_from": resumed_from,
            "parent_boundary_id": (
                self._resume_measurement_checkpoint_boundary["boundary_id"]
                if prior_status == "verified"
                and self._resume_measurement_checkpoint_boundary is not None
                else None
            ),
            "measurement": {
                "warmup_optimizer_steps": self._measurement_warmup_steps,
                "cuda_events": self._measurement_cuda_events,
                "device": str(self.device),
                "output_path": str(self._measurement_path),
            },
            "complete": False,
            "rows": self._measurement_rows,
        }
        self._measurement_segments.append(self._measurement_segment)

    def _load_measurement_evidence(
        self, path: Path
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(
                f"resume checkpoint requires prior measurement evidence at {path}"
            ) from error
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"prior measurement evidence is unreadable: {path}") from error
        if not isinstance(payload, dict) or payload.get("schema_version") != 3:
            raise ValueError("prior measurement evidence has an unsupported schema")
        if payload.get("checkpoint_identity") != self.checkpoint_identity:
            raise ValueError("prior measurement evidence checkpoint identity does not match")
        if payload.get("measurement_evidence_id") != self._resume_measurement_evidence_id:
            raise ValueError("prior measurement evidence chain identity does not match")
        raw_segments = payload.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise ValueError("prior measurement evidence has no segments")

        segments: list[dict[str, Any]] = []
        for expected_index, segment in enumerate(raw_segments):
            if (
                not isinstance(segment, dict)
                or set(segment) != _MEASUREMENT_SEGMENT_KEYS
                or segment.get("segment_index") != expected_index
            ):
                raise ValueError("prior measurement evidence has invalid segment ordering")
            start = self._validated_measurement_counters(
                segment.get("start_counters"), f"segment {expected_index} start"
            )
            end = self._validated_measurement_counters(
                segment.get("end_counters"), f"segment {expected_index} end"
            )
            if (
                end["optimizer_step"] < start["optimizer_step"]
                or end["target_tokens"] < start["target_tokens"]
                or end["elapsed_seconds"] < start["elapsed_seconds"]
            ):
                raise ValueError("prior measurement evidence segment counters move backwards")
            if not isinstance(segment.get("complete"), bool):
                raise ValueError("prior measurement evidence segment completion is invalid")
            measurement = segment.get("measurement")
            if (
                not isinstance(measurement, dict)
                or set(measurement)
                != {"warmup_optimizer_steps", "cuda_events", "device", "output_path"}
                or isinstance(measurement["warmup_optimizer_steps"], bool)
                or not isinstance(measurement["warmup_optimizer_steps"], int)
                or measurement["warmup_optimizer_steps"] < 0
                or not isinstance(measurement["cuda_events"], bool)
                or not isinstance(measurement["device"], str)
                or not measurement["device"]
                or not isinstance(measurement["output_path"], str)
                or not measurement["output_path"]
            ):
                raise ValueError("prior measurement evidence segment settings are invalid")
            self._validate_measurement_segment_resume(segment, start, expected_index)
            rows = segment.get("rows")
            if not isinstance(rows, list):
                raise ValueError("prior measurement evidence segment rows are invalid")
            previous_step = start["optimizer_step"]
            previous_tokens = start["target_tokens"]
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("prior measurement evidence contains an invalid row")
                step = row.get("optimizer_step")
                tokens = row.get("target_tokens")
                if (
                    isinstance(step, bool)
                    or not isinstance(step, int)
                    or isinstance(tokens, bool)
                    or not isinstance(tokens, int)
                    or step < previous_step
                    or tokens < previous_tokens
                    or step > end["optimizer_step"]
                    or tokens > end["target_tokens"]
                ):
                    raise ValueError("prior measurement evidence row counters are invalid")
                previous_step = step
                previous_tokens = tokens
            segments.append(segment)

        raw_boundaries = payload.get("checkpoint_boundaries")
        if not isinstance(raw_boundaries, list):
            raise ValueError("prior measurement checkpoint boundaries are invalid")
        boundaries = self._validated_measurement_checkpoint_boundaries(
            raw_boundaries,
            segments,
        )
        self._validate_measurement_segment_lineage(segments, boundaries)

        complete = payload.get("complete")
        if not isinstance(complete, bool) or complete != segments[-1]["complete"]:
            raise ValueError("prior measurement evidence completion state is inconsistent")
        assert self._resumed_from is not None
        selected = self._resume_measurement_checkpoint_boundary
        if not isinstance(selected, dict):
            raise ValueError("resume checkpoint has no measurement boundary binding")
        selected_binding = self._validated_measurement_checkpoint_binding(
            selected,
            label="selected checkpoint",
        )
        if selected_binding["kind"] != self._resumed_from.payload["kind"]:
            raise ValueError("selected checkpoint kind differs from its measurement boundary")
        if selected_binding["counters"] != self._validated_measurement_counters(
            self.counters,
            "selected checkpoint",
        ):
            raise ValueError("selected checkpoint counters differ from its measurement boundary")
        matching = [
            boundary
            for boundary in boundaries
            if {key: boundary[key] for key in _MEASUREMENT_BOUNDARY_KEYS} == selected_binding
        ]
        if len(matching) != 1 or matching[0]["status"] not in {"pending", "committed"}:
            raise ValueError(
                "prior measurement evidence does not contain an accepted selected resume boundary"
            )
        return segments, boundaries

    def _validate_measurement_segment_resume(
        self,
        segment: dict[str, Any],
        start: dict[str, int | float],
        segment_index: int,
    ) -> None:
        resumed_from = segment["resumed_from"]
        parent_boundary_id = segment["parent_boundary_id"]
        if resumed_from is None:
            if segment_index != 0 or parent_boundary_id is not None:
                raise ValueError("prior measurement segment has invalid fresh lineage")
            return
        if not isinstance(resumed_from, dict) or set(resumed_from) != {
            "path",
            "counters",
            "prior_measurement",
        }:
            raise ValueError("prior measurement segment resume metadata is invalid")
        if not isinstance(resumed_from["path"], str) or not resumed_from["path"]:
            raise ValueError("prior measurement segment resume path is invalid")
        if (
            self._validated_measurement_counters(
                resumed_from["counters"], f"segment {segment_index} resume"
            )
            != start
        ):
            raise ValueError("prior measurement segment resume counters differ from its start")
        prior = resumed_from["prior_measurement"]
        if not isinstance(prior, dict) or set(prior) != {"status", "path"}:
            raise ValueError("prior measurement segment resume evidence metadata is invalid")
        status = prior["status"]
        prior_path = prior["path"]
        if status == "verified":
            if (
                not isinstance(parent_boundary_id, str)
                or not parent_boundary_id
                or not isinstance(prior_path, str)
                or not prior_path
            ):
                raise ValueError("verified measurement resume has invalid parent evidence")
        elif status == "disabled":
            if parent_boundary_id is not None or prior_path is not None:
                raise ValueError("disabled measurement resume has unexpected parent evidence")
        else:
            raise ValueError("prior measurement segment resume status is invalid")

    def _validate_measurement_segment_lineage(
        self,
        segments: list[dict[str, Any]],
        boundaries: list[dict[str, Any]],
    ) -> None:
        boundaries_by_id = {boundary["boundary_id"]: boundary for boundary in boundaries}
        for segment in segments:
            parent_boundary_id = segment["parent_boundary_id"]
            if parent_boundary_id is None:
                continue
            parent = boundaries_by_id.get(parent_boundary_id)
            if (
                parent is None
                or parent["status"] not in {"pending", "committed"}
                or parent["segment_index"] >= segment["segment_index"]
                or self._validated_measurement_counters(
                    parent["counters"],
                    f"segment {segment['segment_index']} parent",
                )
                != self._validated_measurement_counters(
                    segment["start_counters"],
                    f"segment {segment['segment_index']} start",
                )
            ):
                raise ValueError("prior measurement segment parent lineage is invalid")

    def _validated_measurement_checkpoint_boundaries(
        self,
        raw_boundaries: list[Any],
        segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        boundaries: list[dict[str, Any]] = []
        boundary_ids: set[str] = set()
        for expected_index, boundary in enumerate(raw_boundaries):
            if not isinstance(boundary, dict) or set(boundary) != (
                _MEASUREMENT_BOUNDARY_KEYS | {"status", "checkpoint_path"}
            ):
                raise ValueError("prior measurement checkpoint boundary is invalid")
            binding = self._validated_measurement_checkpoint_binding(
                {key: boundary[key] for key in _MEASUREMENT_BOUNDARY_KEYS},
                label=f"checkpoint boundary {expected_index}",
            )
            if binding["boundary_index"] != expected_index:
                raise ValueError("prior measurement checkpoint boundary ordering is invalid")
            boundary_id = binding["boundary_id"]
            if boundary_id in boundary_ids:
                raise ValueError("prior measurement checkpoint boundary IDs are not unique")
            boundary_ids.add(boundary_id)
            segment_index = binding["segment_index"]
            if segment_index >= len(segments):
                raise ValueError("prior measurement checkpoint boundary segment is invalid")
            segment = segments[segment_index]
            counters = binding["counters"]
            start = self._validated_measurement_counters(
                segment["start_counters"], f"segment {segment_index} start"
            )
            end = self._validated_measurement_counters(
                segment["end_counters"], f"segment {segment_index} end"
            )
            if any(
                not start[key] <= counters[key] <= end[key]
                for key in ("optimizer_step", "target_tokens", "elapsed_seconds")
            ):
                raise ValueError("prior measurement checkpoint boundary counters are out of range")
            status = boundary["status"]
            checkpoint_path = boundary["checkpoint_path"]
            if status not in {"pending", "committed", "failed"}:
                raise ValueError("prior measurement checkpoint boundary status is invalid")
            if status == "committed":
                if not isinstance(checkpoint_path, str) or not checkpoint_path:
                    raise ValueError("committed measurement checkpoint boundary has no path")
            elif checkpoint_path is not None:
                raise ValueError("uncommitted measurement checkpoint boundary has a path")
            boundaries.append(boundary)
        return boundaries

    def _validated_measurement_checkpoint_binding(
        self,
        value: Any,
        *,
        label: str,
        expected_evidence_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != _MEASUREMENT_BOUNDARY_KEYS:
            raise ValueError(f"measurement evidence {label} boundary binding is invalid")
        boundary_index = value["boundary_index"]
        boundary_id = value["boundary_id"]
        evidence_id = value["evidence_id"]
        expected_evidence_id = (
            self._resume_measurement_evidence_id
            if expected_evidence_id is None
            else expected_evidence_id
        )
        segment_index = value["segment_index"]
        kind = value["kind"]
        if (
            isinstance(boundary_index, bool)
            or not isinstance(boundary_index, int)
            or boundary_index < 0
            or not isinstance(boundary_id, str)
            or not boundary_id
            or evidence_id != expected_evidence_id
            or isinstance(segment_index, bool)
            or not isinstance(segment_index, int)
            or segment_index < 0
            or kind not in _CHECKPOINT_KINDS
        ):
            raise ValueError(f"measurement evidence {label} boundary binding is invalid")
        return {
            "boundary_index": boundary_index,
            "boundary_id": boundary_id,
            "evidence_id": evidence_id,
            "segment_index": segment_index,
            "kind": kind,
            "counters": self._validated_measurement_counters(value["counters"], label),
        }

    @staticmethod
    def _validated_measurement_counters(value: Any, label: str) -> dict[str, int | float]:
        if not isinstance(value, dict):
            raise ValueError(f"prior measurement evidence {label} counters are invalid")
        optimizer_step = value.get("optimizer_step")
        target_tokens = value.get("target_tokens")
        elapsed_seconds = value.get("elapsed_seconds")
        if (
            isinstance(optimizer_step, bool)
            or not isinstance(optimizer_step, int)
            or optimizer_step < 0
            or isinstance(target_tokens, bool)
            or not isinstance(target_tokens, int)
            or target_tokens < 0
            or isinstance(elapsed_seconds, bool)
            or not isinstance(elapsed_seconds, (int, float))
            or not math.isfinite(float(elapsed_seconds))
            or elapsed_seconds < 0
        ):
            raise ValueError(f"prior measurement evidence {label} counters are invalid")
        return {
            "optimizer_step": optimizer_step,
            "target_tokens": target_tokens,
            "elapsed_seconds": float(elapsed_seconds),
        }

    def _event_trigger(self, prefix: str, epoch_end: bool) -> str:
        if epoch_end:
            return "epoch"
        step_cadence = self._training_value(f"{prefix}_every_n_steps")
        if step_cadence is not None and self.optimizer_step % int(step_cadence) == 0:
            return "optimizer_step"
        return "target_tokens"

    def _save_checkpoint(self) -> Path:
        return self._save_checkpoint_with_measurement_boundary(
            "recovery", self.checkpoints.save_recovery
        )

    def _save_best_checkpoint(self) -> Path:
        return self._save_checkpoint_with_measurement_boundary("best", self.checkpoints.save_best)

    def _save_final_checkpoint(self) -> Path:
        return self._save_checkpoint_with_measurement_boundary("final", self.checkpoints.save_final)

    def _save_milestone_checkpoint(self) -> Path:
        return self._save_checkpoint_with_measurement_boundary(
            "milestone", self.checkpoints.save_milestone
        )

    def _save_checkpoint_with_measurement_boundary(
        self,
        kind: str,
        save: Callable[[dict[str, Any]], Path],
    ) -> Path:
        if not self._measurement_enabled:
            return save(self._checkpoint_state())
        if self._measurement_segment is None or self._measurement_evidence_id is None:
            raise RuntimeError("measurement checkpoint save has no active evidence segment")
        binding = {
            "boundary_index": len(self._measurement_checkpoint_boundaries),
            "boundary_id": uuid.uuid4().hex,
            "evidence_id": self._measurement_evidence_id,
            "segment_index": int(self._measurement_segment["segment_index"]),
            "kind": kind,
            "counters": dict(self.counters),
        }
        boundary = {
            **binding,
            "counters": dict(binding["counters"]),
            "status": "pending",
            "checkpoint_path": None,
        }
        self._measurement_checkpoint_boundaries.append(boundary)
        self._active_measurement_checkpoint_boundary = binding
        self._flush_measurements()
        try:
            checkpoint_path = save(self._checkpoint_state())
        except BaseException as error:
            boundary["status"] = "failed"
            self._active_measurement_checkpoint_boundary = None
            try:
                self._flush_measurements()
            except BaseException as flush_error:
                error.add_note(
                    f"measurement boundary could not be durably marked failed: {flush_error!r}"
                )
            raise
        self._active_measurement_checkpoint_boundary = None
        boundary["status"] = "committed"
        boundary["checkpoint_path"] = str(checkpoint_path.resolve())
        self._flush_measurements()
        return checkpoint_path

    def _checkpoint_measurement_metrics(self) -> dict[str, float | int]:
        measurement = self.checkpoints.last_write_measurement
        if measurement is None:
            return {}
        return {
            "checkpoint/size_bytes": measurement.size_bytes,
            "checkpoint/write_seconds": measurement.write_seconds,
            "checkpoint/verification_seconds": measurement.verification_seconds,
            "checkpoint/pause_seconds": measurement.pause_seconds,
            "checkpoint/write_bytes_per_second": measurement.write_bytes_per_second,
        }

    def _checkpoint_state(self) -> dict[str, Any]:
        scheduler_state = self.scheduler.state_dict() if self.scheduler is not None else None
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": scheduler_state,
            "precision": {"mode": self.precision, "grad_scaler": None},
            "counters": self.counters,
            "event_state": {
                "last_validation_step": self._last_validation_step,
                "last_checkpoint_step": self._last_checkpoint_step,
                "last_milestone_step": self._last_milestone_step,
                "last_log_step": self._last_log_step,
                "last_token_event_boundary": dict(self._last_token_event_boundary),
                "best_validation_loss": self._best_validation_loss,
                "best_checkpoint_step": self._best_checkpoint_step,
            },
            "rng": capture_rng_state(),
            "stream_cursor": self._stream_cursor_state(),
            "resolved_config": OmegaConf.to_container(self.cfg, resolve=True),
            "run_identity": dict(self.checkpoint_identity),
            "measurement_evidence": {
                "enabled": self._measurement_enabled,
                "evidence_id": self._measurement_evidence_id,
                "checkpoint_boundary": (
                    {
                        **self._active_measurement_checkpoint_boundary,
                        "counters": dict(self._active_measurement_checkpoint_boundary["counters"]),
                    }
                    if self._active_measurement_checkpoint_boundary is not None
                    else None
                ),
            },
        }

    def _restore_checkpoint(self, resume_path: str | Path) -> None:
        resumed = self.checkpoints.load_resume(resume_path)
        state = resumed.payload["state"]
        require_exact_stream_resume_state(state)
        required = {
            "model",
            "optimizer",
            "scheduler",
            "precision",
            "counters",
            "event_state",
            "rng",
            "stream_cursor",
            "resolved_config",
            "run_identity",
            "measurement_evidence",
        }
        missing = required.difference(state)
        if missing:
            raise CheckpointCompatibilityError(
                f"checkpoint {resumed.path} is missing full-state entries {sorted(missing)}"
            )
        precision = state["precision"]
        if not isinstance(precision, dict) or precision.get("mode") != self.precision:
            raise CheckpointCompatibilityError("checkpoint precision mode differs from this run")
        if state["run_identity"] != self.checkpoint_identity:
            raise CheckpointCompatibilityError(
                "checkpoint run/data/tokenizer identity differs from this run"
            )
        resolved_config = state["resolved_config"]
        if not isinstance(resolved_config, dict):
            raise CheckpointCompatibilityError("checkpoint resolved config is invalid")
        resume_measurement_config = resolved_config.get("measurement", {}) or {}
        if not isinstance(resume_measurement_config, dict):
            raise CheckpointCompatibilityError("checkpoint measurement config is invalid")
        measurement_evidence = state["measurement_evidence"]
        if not isinstance(measurement_evidence, dict) or set(measurement_evidence) != {
            "enabled",
            "evidence_id",
            "checkpoint_boundary",
        }:
            raise CheckpointCompatibilityError("checkpoint measurement evidence state is invalid")
        prior_measurement_enabled = measurement_evidence.get("enabled")
        prior_measurement_id = measurement_evidence.get("evidence_id")
        prior_measurement_boundary = measurement_evidence.get("checkpoint_boundary")
        if (
            not isinstance(prior_measurement_enabled, bool)
            or prior_measurement_enabled != bool(resume_measurement_config.get("enabled", False))
            or (
                prior_measurement_enabled
                and (not isinstance(prior_measurement_id, str) or not prior_measurement_id)
            )
            or (not prior_measurement_enabled and prior_measurement_id is not None)
            or (prior_measurement_enabled and not isinstance(prior_measurement_boundary, dict))
            or (not prior_measurement_enabled and prior_measurement_boundary is not None)
        ):
            raise CheckpointCompatibilityError("checkpoint measurement evidence state is invalid")
        counters = state["counters"]
        if not isinstance(counters, dict):
            raise CheckpointCompatibilityError("checkpoint counters are invalid")
        optimizer_step = counters.get("optimizer_step")
        target_tokens = counters.get("target_tokens")
        elapsed_seconds = counters.get("elapsed_seconds")
        if (
            isinstance(optimizer_step, bool)
            or not isinstance(optimizer_step, int)
            or optimizer_step < 0
            or isinstance(target_tokens, bool)
            or not isinstance(target_tokens, int)
            or target_tokens < 0
            or not isinstance(elapsed_seconds, (int, float))
            or elapsed_seconds < 0
        ):
            raise CheckpointCompatibilityError("checkpoint counters are invalid")
        if prior_measurement_enabled:
            try:
                checkpoint_boundary = self._validated_measurement_checkpoint_binding(
                    prior_measurement_boundary,
                    label="checkpoint payload",
                    expected_evidence_id=prior_measurement_id,
                )
            except ValueError as error:
                raise CheckpointCompatibilityError(
                    "checkpoint measurement boundary binding is invalid"
                ) from error
            if checkpoint_boundary["kind"] != resumed.payload["kind"]:
                raise CheckpointCompatibilityError(
                    "checkpoint kind differs from its measurement boundary binding"
                )
            if checkpoint_boundary["counters"] != self._validated_measurement_counters(
                counters,
                "checkpoint payload",
            ):
                raise CheckpointCompatibilityError(
                    "checkpoint counters differ from its measurement boundary binding"
                )
            prior_measurement_boundary = checkpoint_boundary
        self._load_stream_cursor(dict(state["stream_cursor"]))
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is None:
            if state["scheduler"] is not None:
                raise CheckpointCompatibilityError(
                    "checkpoint requires a scheduler but this run disables it"
                )
        else:
            if state["scheduler"] is None:
                raise CheckpointCompatibilityError("checkpoint has no scheduler state for this run")
            self.scheduler.load_state_dict(state["scheduler"])
        self.optimizer_step = optimizer_step
        self.target_tokens = target_tokens
        self.elapsed_seconds = float(elapsed_seconds)
        event_state = state["event_state"]
        if not isinstance(event_state, dict):
            raise CheckpointCompatibilityError("checkpoint event state is invalid")
        self._last_validation_step = event_state.get("last_validation_step")
        self._last_checkpoint_step = event_state.get("last_checkpoint_step")
        self._last_milestone_step = event_state.get("last_milestone_step")
        self._last_log_step = event_state.get("last_log_step")
        token_boundaries = event_state.get("last_token_event_boundary", {})
        if not isinstance(token_boundaries, dict):
            raise CheckpointCompatibilityError("checkpoint token-event state is invalid")
        self._last_token_event_boundary = {
            str(key): int(value) for key, value in token_boundaries.items()
        }
        best = event_state.get("best_validation_loss")
        self._best_validation_loss = float(best) if best is not None else None
        best_step = event_state.get("best_checkpoint_step")
        if best is not None and best_step is None:
            raise CheckpointCompatibilityError(
                "checkpoint with a best validation score is missing its checkpoint step"
            )
        if best_step is not None and (
            isinstance(best_step, bool) or not isinstance(best_step, int) or best_step < 0
        ):
            raise CheckpointCompatibilityError("checkpoint best-checkpoint step is invalid")
        self._best_checkpoint_step = best_step
        self._best_checkpoint_path = self._recover_best_checkpoint(
            resumed,
            best_validation_loss=self._best_validation_loss,
            best_checkpoint_step=best_step,
        )
        restore_rng_state(state["rng"])
        self._resume_measurement_config = dict(resume_measurement_config)
        self._resume_measurement_evidence_id = prior_measurement_id
        self._resume_measurement_checkpoint_boundary = prior_measurement_boundary
        self._resumed_from = resumed
        if resumed.rejected_paths:
            logger.warning(
                "Recovered from {} after rejecting corrupt newer checkpoints: {}",
                resumed.path,
                ", ".join(str(path) for path in resumed.rejected_paths),
            )
        else:
            logger.info("Resuming verified full state from {}", resumed.path)

    def _recover_best_checkpoint(
        self,
        resumed: ResumeCheckpoint,
        *,
        best_validation_loss: float | None,
        best_checkpoint_step: int | None,
    ) -> Path | None:
        """Recover a verified retained best checkpoint without changing run state."""

        current_path = self.checkpoint_dir / "best.pt"
        resumed_path = resumed.path.parent / "best.pt"
        self._expected_best_checkpoint_path = (
            resumed_path if best_checkpoint_step is not None else current_path
        )
        candidates = [current_path]
        if resumed_path != current_path:
            candidates.append(resumed_path)
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                loaded = load_checkpoint_for_generation(candidate)
                payload = loaded.payload
                state = payload["state"]
                event_state = state.get("event_state", {})
                if (
                    payload["kind"] != "best"
                    or dict(payload["identity"]) != self.checkpoint_identity
                    or state["counters"]["optimizer_step"] != best_checkpoint_step
                    or not isinstance(event_state, dict)
                    or event_state.get("best_checkpoint_step") != best_checkpoint_step
                    or event_state.get("best_validation_loss") != best_validation_loss
                ):
                    raise CheckpointCompatibilityError(
                        "retained best checkpoint does not match restored best state"
                    )
            except Exception as error:
                logger.warning(
                    "Ignoring unusable retained best checkpoint {}: {}",
                    candidate,
                    error,
                )
                continue
            self._expected_best_checkpoint_path = candidate
            return candidate
        return None

    def _stream_cursor_state(self) -> dict[str, Any] | None:
        dataset = getattr(self.train_loader, "dataset", None)
        state_dict = getattr(dataset, "state_dict", None)
        if not callable(state_dict):
            return None
        return state_dict()

    def _load_stream_cursor(self, cursor: dict[str, Any]) -> None:
        dataset = getattr(self.train_loader, "dataset", None)
        load_state_dict = getattr(dataset, "load_state_dict", None)
        if not callable(load_state_dict):
            raise CheckpointCompatibilityError(
                "checkpoint contains a stream cursor but this train loader cannot restore one"
            )
        load_state_dict(cursor)

    def _get_scheduler_interval(self) -> str:
        if self.scheduler is None:
            return "epoch"
        scheduler_cfg = self.cfg.get("training", {}).get("scheduler", {})
        interval = scheduler_cfg.get("interval", "epoch")
        if interval not in {"epoch", "step"}:
            raise ValueError("training.scheduler.interval must be either 'epoch' or 'step'.")
        if interval == "step" and isinstance(self.scheduler, ReduceLROnPlateau):
            raise ValueError("ReduceLROnPlateau only supports training.scheduler.interval='epoch'.")
        return interval

    def _step_scheduler(self, metric: float | None = None) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metric is None:
                raise ValueError("ReduceLROnPlateau requires a metric when stepping.")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def _training_value(self, key: str, default: Any = None) -> Any:
        training = self.cfg.get("training", {})
        if key in training:
            return training.get(key)
        return default

    def _cadence(self, key: str) -> int | None:
        value = self._training_value(key)
        if value is None:
            cadence = self._training_value("cadence", {}) or {}
            value = cadence.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"training.{key} must be positive when configured")
        return value

    def _positive_budget(self, key: str) -> int | float | None:
        value = self._training_value(key)
        if value is None:
            return None
        if key in {"max_steps", "max_tokens"}:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"training.{key} must be a positive integer when configured")
            if value <= 0:
                raise ValueError(f"training.{key} must be positive when configured")
            return value
        value = float(value)
        if value <= 0:
            raise ValueError(f"training.{key} must be positive when configured")
        return value

    def _positive_training_integer(self, key: str, *, default: int) -> int:
        value = self._training_value(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"training.{key} must be a positive integer")
        return value

    def _max_grad_norm(self) -> float | None:
        value = self._training_value("max_grad_norm")
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("training.max_grad_norm must be a positive finite number or null")
        return value

    def _configured_effective_tokens(self) -> int:
        return (
            int(self._training_value("batch_size", 1))
            * int(self._training_value("sequence_length", 1))
            * self.gradient_accumulation_steps
        )

    def _event_due(self, step_key: str, token_key: str, epoch_end: bool) -> bool:
        step_cadence = self._cadence(step_key)
        token_cadence = self._cadence(token_key)
        due_step = step_cadence is not None and self.optimizer_step % step_cadence == 0
        due_token = False
        if token_cadence is not None:
            boundary = self._last_token_event_boundary.get(token_key, 0)
            if self.target_tokens >= boundary + token_cadence:
                due_token = True
                self._last_token_event_boundary[token_key] = (
                    self.target_tokens // token_cadence
                ) * token_cadence
        return (
            due_step or due_token or (step_cadence is None and token_cadence is None and epoch_end)
        )

    def _gradients_are_finite(self) -> bool:
        return all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in self.model.parameters()
        )

    def _scale_gradients(self, scale: float) -> None:
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)

    def _global_gradient_norm(self) -> torch.Tensor:
        norms = [
            torch.linalg.vector_norm(parameter.grad.detach())
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        if not norms:
            raise RuntimeError("training update produced no gradients")
        return torch.linalg.vector_norm(torch.stack(norms))

    @staticmethod
    def _close_train_iterator(iterator) -> None:
        """Close an interrupted iterable-dataset generator before process exit.

        PyTorch's single-process ``DataLoader`` does not expose a public close
        method, but retains an iterable dataset's generator in its fetcher. A
        token/step budget can stop before that generator naturally exhausts;
        closing it lets ``StreamingTokenDataset`` leave its ``StreamLoader``
        context and join the non-daemon prefetch worker.
        """

        fetcher = getattr(iterator, "_dataset_fetcher", None)
        dataset_iterator = getattr(fetcher, "dataset_iter", None)
        close = getattr(dataset_iterator, "close", None)
        if callable(close):
            close()

    def _parameters_are_finite(self) -> bool:
        return all(torch.isfinite(parameter).all().item() for parameter in self.model.parameters())

    def _numeric_failure_message(self, kind: str, batch_index: int | None) -> str:
        return (
            f"non-finite {kind} at optimizer_step={self.optimizer_step}, "
            f"target_tokens={self.target_tokens}, batch_index={batch_index}"
        )

    def _record_numeric_failure(self, kind: str, batch_index: int | None) -> None:
        message = self._numeric_failure_message(kind, batch_index)
        logger.error(message)
        preceding_checkpoint = (
            self.checkpoint_dir / f"recovery-step-{self._last_checkpoint_step:012d}.pt"
            if self._last_checkpoint_step is not None
            else None
        )
        self._record_metrics(
            {
                "event": f"nonfinite_{kind}",
                "optimizer_step": self.optimizer_step,
                "target_tokens": self.target_tokens,
                "batch_index": batch_index,
                "elapsed_seconds": self.elapsed_seconds,
                "error": message,
                "preceding_checkpoint_step": self._last_checkpoint_step,
                "preceding_checkpoint": (
                    str(preceding_checkpoint)
                    if preceding_checkpoint is not None and preceding_checkpoint.exists()
                    else None
                ),
            },
            send_to_wandb=False,
        )
        self.wandb.log(
            {
                "event": "nonfinite",
                "optimizer_step": self.optimizer_step,
                "target_tokens": self.target_tokens,
                "elapsed_seconds": self.elapsed_seconds,
                "optimizer/nonfinite_count": 1,
                "stability/nonfinite_kind": kind,
                "stability/batch_index": batch_index,
            }
        )

    def _remaining_tokens(self, pending_tokens: int = 0) -> int | None:
        if self.max_tokens is None:
            return None
        return max(0, int(self.max_tokens) - self.target_tokens - pending_tokens)

    def _budget_reached(self) -> bool:
        if self.max_steps is not None and self.optimizer_step >= self.max_steps:
            return True
        if self.max_tokens is not None and self.target_tokens >= self.max_tokens:
            return True
        if self.max_time is not None and self.elapsed_seconds >= self.max_time:
            return True
        return False

    def _update_elapsed(self) -> None:
        if self._start_time is not None:
            self.elapsed_seconds = max(0.0, time.monotonic() - self._start_time)

    @property
    def counters(self) -> dict[str, int | float]:
        """Return a serializable snapshot of the authoritative work units."""

        return {
            "optimizer_step": self.optimizer_step,
            "target_tokens": self.target_tokens,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")
