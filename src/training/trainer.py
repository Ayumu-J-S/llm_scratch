"""Small, deterministic training loop with explicit work counters.

The trainer deliberately owns the definition of a step and a target token.  A
batch can therefore be partial (padding or a token budget boundary) without
changing the reported objective: NLL is accumulated as a token-weighted sum.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import wandb
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
    require_exact_stream_resume_state,
    restore_rng_state,
)
from training.optimization import autocast_context, get_learning_rate


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
        self.run = None
        self._last_validation_step: int | None = None
        self._last_checkpoint_step: int | None = None
        self._last_milestone_step: int | None = None
        self._last_log_step: int | None = None
        self._last_token_event_boundary: dict[str, int] = {}
        self._start_time: float | None = None
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
        self._measurement_completed = False
        self._measurement_evidence_id = uuid.uuid4().hex if self._measurement_enabled else None
        self._resume_measurement_config: dict[str, Any] | None = None
        self._resume_measurement_evidence_id: str | None = None

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
        # run-local evidence, not append-only checkpoint state. Validate and
        # initialize measurement evidence before external logging or training.
        self.metrics.clear()
        self.run = None
        self._initialize_measurements()
        self.run = self._init_wandb()
        if self._resumed_from is None:
            self._reset_local_metrics()
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
                    send_to_wandb=True,
                )
                self._run_events(epoch_end=True, train_loss=train_loss)
                if self.scheduler is not None and self.scheduler_interval == "epoch":
                    # Metric-free schedulers advance at the epoch boundary;
                    # ReduceLROnPlateau alone requires a validation metric.
                    validation_loss = self._latest_validation_loss
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        if validation_loss is None:
                            validation_result = self._evaluate()
                            self._update_elapsed()
                            validation_loss = validation_result.nll
                            self._latest_validation_loss = validation_loss
                            self._record_validation_metrics(validation_result)
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
            if self.run is not None:
                self.run.finish()
                self.run = None

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
    ) -> None:
        step = self.optimizer_step
        if step < 1:
            return
        self._latest_validation_loss = getattr(self, "_latest_validation_loss", None)

        should_validate = self._event_due(
            "validation_every_n_steps", "validation_every_n_tokens", epoch_end
        )
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
                checkpoint_started = time.perf_counter()
                best_path = self._save_best_checkpoint()
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
            self._record_validation_metrics(validation_result)
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

        should_log = self._event_due("log_every_n_steps", "log_every_n_tokens", epoch_end)
        if should_log and self._last_log_step != step:
            self._last_log_step = step
            values = {
                "event": "log",
                "optimizer_step": step,
                "target_tokens": self.target_tokens,
                "elapsed_seconds": self.elapsed_seconds,
                "optimizer/lr": get_learning_rate(self.optimizer),
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
        if milestone_due and self.run is not None:
            assert milestone_path is not None
            self._log_model_artifact(
                run=self.run,
                checkpoint_path=milestone_path,
                epoch_number=step,
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
        self._record_metrics(
            {
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
            },
            send_to_wandb=False,
        )

    def _record_validation_metrics(self, validation_result: EvaluationResult) -> None:
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
        self._record_metrics(
            values,
            send_to_wandb=True,
            preserve_none=(f"{namespace}/perplexity",),
        )

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
        if send_to_wandb and self.run is not None:
            self.run.log(record)

    def _reset_local_metrics(self) -> None:
        """Atomically start a fresh local evidence stream after run init."""

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self.checkpoint_dir / "metrics.jsonl"
        temporary_path = metrics_path.with_name(f".{metrics_path.name}.tmp")
        temporary_path.write_text("", encoding="utf-8")
        temporary_path.replace(metrics_path)

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
            "schema_version": 2,
            "checkpoint_identity": self.checkpoint_identity,
            "measurement_evidence_id": self._measurement_evidence_id,
            "complete": self._measurement_completed,
            "segments": self._measurement_segments,
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
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            temporary_path.replace(self._measurement_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def _initialize_measurements(self) -> None:
        self._measurement_rows.clear()
        self._measurement_segments.clear()
        self._measurement_segment = None
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
                self._measurement_segments.extend(self._load_measurement_segments(prior_path))
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

    def _load_measurement_segments(self, path: Path) -> list[dict[str, Any]]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(
                f"resume checkpoint requires prior measurement evidence at {path}"
            ) from error
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"prior measurement evidence is unreadable: {path}") from error
        if not isinstance(payload, dict) or payload.get("schema_version") != 2:
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
            if not isinstance(segment, dict) or segment.get("segment_index") != expected_index:
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
            if not isinstance(measurement, dict):
                raise ValueError("prior measurement evidence segment settings are invalid")
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

        complete = payload.get("complete")
        if not isinstance(complete, bool) or complete != segments[-1]["complete"]:
            raise ValueError("prior measurement evidence completion state is inconsistent")
        assert self._resumed_from is not None
        resume_step = self.optimizer_step
        resume_tokens = self.target_tokens
        checkpoint_events = {"checkpoint", "final_checkpoint", "milestone"}
        boundary_found = any(
            row.get("optimizer_step") == resume_step
            and row.get("target_tokens") == resume_tokens
            and (
                row.get("event") in checkpoint_events
                or (row.get("event") == "validation" and row.get("best_checkpoint_written") is True)
            )
            for segment in segments
            for row in segment["rows"]
        )
        if not boundary_found:
            raise ValueError(
                "prior measurement evidence does not contain the selected resume boundary"
            )
        return segments

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

    def _init_wandb(self):
        wandb_cfg = self.cfg.get("wandb")
        if wandb_cfg is None or not wandb_cfg.get("enabled", False):
            return None
        run = wandb.init(
            project=wandb_cfg.get("project"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("name"),
            mode=wandb_cfg.get("mode", "online"),
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )
        run.watch(self.model)
        return run

    def _save_checkpoint(self) -> Path:
        return self.checkpoints.save_recovery(self._checkpoint_state())

    def _save_best_checkpoint(self) -> Path:
        return self.checkpoints.save_best(self._checkpoint_state())

    def _save_final_checkpoint(self) -> Path:
        return self.checkpoints.save_final(self._checkpoint_state())

    def _save_milestone_checkpoint(self) -> Path:
        return self.checkpoints.save_milestone(self._checkpoint_state())

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
            },
            "rng": capture_rng_state(),
            "stream_cursor": self._stream_cursor_state(),
            "resolved_config": OmegaConf.to_container(self.cfg, resolve=True),
            "run_identity": dict(self.checkpoint_identity),
            "measurement_evidence": {
                "enabled": self._measurement_enabled,
                "evidence_id": self._measurement_evidence_id,
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
        if not isinstance(measurement_evidence, dict):
            raise CheckpointCompatibilityError("checkpoint measurement evidence state is invalid")
        prior_measurement_enabled = measurement_evidence.get("enabled")
        prior_measurement_id = measurement_evidence.get("evidence_id")
        if (
            not isinstance(prior_measurement_enabled, bool)
            or prior_measurement_enabled != bool(resume_measurement_config.get("enabled", False))
            or (
                prior_measurement_enabled
                and (not isinstance(prior_measurement_id, str) or not prior_measurement_id)
            )
            or (not prior_measurement_enabled and prior_measurement_id is not None)
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
        restore_rng_state(state["rng"])
        self._resume_measurement_config = dict(resume_measurement_config)
        self._resume_measurement_evidence_id = prior_measurement_id
        self._resumed_from = resumed
        if resumed.rejected_paths:
            logger.warning(
                "Recovered from {} after rejecting corrupt newer checkpoints: {}",
                resumed.path,
                ", ".join(str(path) for path in resumed.rejected_paths),
            )
        else:
            logger.info("Resuming verified full state from {}", resumed.path)

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

    def _log_model_artifact(self, *, run, checkpoint_path: Path, epoch_number: int) -> None:
        artifact = wandb.Artifact(
            name=self._model_artifact_name(), type="model", metadata={"step": epoch_number}
        )
        artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)
        run.log_artifact(artifact, aliases=[f"step-{epoch_number}", "latest"])

    def _model_artifact_name(self) -> str:
        model_name = self.model.__class__.__name__.strip()
        if not model_name:
            return "model"
        return re.sub(r"(?<!^)(?=[A-Z])", "-", model_name).lower()

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
