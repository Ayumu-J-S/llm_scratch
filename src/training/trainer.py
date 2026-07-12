"""Small, deterministic training loop with explicit work counters.

The trainer deliberately owns the definition of a step and a target token.  A
batch can therefore be partial (padding or a token budget boundary) without
changing the reported objective: NLL is accumulated as a token-weighted sum.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import wandb
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from training.optimization import get_learning_rate


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
        validation_loader,
        checkpoint_dir: Path,
        cfg: DictConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.validation_loader = validation_loader
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
        self._last_log_step: int | None = None
        self._last_token_event_boundary: dict[str, int] = {}
        self._start_time: float | None = None

        self.max_steps = self._positive_budget("max_steps")
        self.max_tokens = self._positive_budget("max_tokens")
        self.max_time = self._positive_budget("max_time")
        self.epochs = int(self._training_value("epochs", 1))
        if self.epochs < 1:
            raise ValueError("training.epochs must be positive")

    @property
    def train_step(self) -> int:
        """Compatibility readout for the authoritative optimizer step."""

        return self.optimizer_step

    def fit(self) -> list[dict[str, Any]]:
        """Run training and return the local metric records."""

        # A checkpoint directory can be reused by a later run. Metrics are
        # run-local evidence, not append-only checkpoint state. Initialize W&B
        # first; if that fails, preserve the previous evidence for diagnosis.
        self.metrics.clear()
        self.run = None
        self.run = self._init_wandb()
        self._reset_local_metrics()
        self._start_time = time.monotonic()
        saw_batch = False

        logger.info("Training...")
        try:
            for epoch_index in range(self.epochs):
                self.model.train()
                self._latest_validation_loss = None
                epoch_loss_sum = 0.0
                epoch_tokens = 0
                epoch_saw_batch = False
                iterator = tqdm(
                    self.train_loader,
                    desc=f"epoch {epoch_index + 1}/{self.epochs}",
                    leave=False,
                )
                for batch_index, batch in enumerate(iterator, start=1):
                    self._update_elapsed()
                    if self._budget_reached():
                        break
                    loss_sum, token_count = self._train_batch(batch, batch_index=batch_index)
                    if token_count == 0:
                        break
                    epoch_saw_batch = saw_batch = True
                    epoch_loss_sum += loss_sum
                    epoch_tokens += token_count
                    self._update_elapsed()
                    self._record_step_metrics(loss_sum / token_count, token_count)
                    self._run_events(epoch_end=False)
                    if self._budget_reached():
                        break

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
                            validation_loss = self._evaluate()
                            self._record_validation_metrics(validation_loss)
                        self._step_scheduler(validation_loss)
                    else:
                        self._step_scheduler()

                if self._budget_reached():
                    break

            if not saw_batch:
                raise ValueError("training loader is empty; no optimizer steps were taken")
            self._update_elapsed()
            return list(self.metrics)
        finally:
            if self.run is not None:
                self.run.finish()
                self.run = None

    def _train_batch(
        self,
        batch: dict[str, torch.Tensor],
        *,
        batch_index: int | None = None,
    ) -> tuple[float, int]:
        input_batch = batch["inputs"].to(self.device)
        label_batch = batch["labels"].to(self.device)
        ignore_index = int(self._training_value("ignore_index", -100))

        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(input_batch)
        flat_labels = label_batch.reshape(-1)
        flat_losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            flat_labels,
            reduction="none",
            ignore_index=ignore_index,
        )
        valid_indices = torch.nonzero(flat_labels != ignore_index, as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            raise ValueError("training batch contains zero target tokens")

        remaining = self._remaining_tokens()
        if remaining is not None:
            valid_indices = valid_indices[:remaining]
            if valid_indices.numel() == 0:
                return 0.0, 0
        selected_losses = flat_losses.index_select(0, valid_indices)
        token_count = int(valid_indices.numel())
        loss = selected_losses.sum() / token_count
        if not torch.isfinite(loss):
            self._record_numeric_failure("loss", batch_index)
            raise FloatingPointError(self._numeric_failure_message("loss", batch_index))
        loss.backward()
        if not self._gradients_are_finite():
            self._record_numeric_failure("gradients", batch_index)
            raise FloatingPointError(self._numeric_failure_message("gradients", batch_index))
        self.optimizer.step()
        if not self._parameters_are_finite():
            self._record_numeric_failure("parameters", batch_index)
            raise FloatingPointError(self._numeric_failure_message("parameters", batch_index))
        self.optimizer_step += 1
        self.target_tokens += token_count
        if self.scheduler is not None and self.scheduler_interval == "step":
            # A scheduler observes the update only after optimizer.step and
            # after the authoritative step counter advances.
            self._step_scheduler()
        return float(selected_losses.detach().sum().item()), token_count

    def _evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        ignore_index = int(self._training_value("ignore_index", -100))
        try:
            with torch.no_grad():
                for batch_index, batch in enumerate(self.validation_loader, start=1):
                    input_batch = batch["inputs"].to(self.device)
                    label_batch = batch["labels"].to(self.device)
                    logits = self.model(input_batch)
                    flat_labels = label_batch.reshape(-1)
                    losses = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        flat_labels,
                        reduction="none",
                        ignore_index=ignore_index,
                    )
                    if not torch.isfinite(losses).all():
                        self._record_numeric_failure("validation", batch_index)
                        raise FloatingPointError(
                            self._numeric_failure_message("validation", batch_index)
                        )
                    valid = flat_labels != ignore_index
                    count = int(valid.sum().item())
                    if count:
                        total_loss += float(losses[valid].sum().item())
                        total_tokens += count
        finally:
            self.model.train()
        if total_tokens == 0:
            raise ValueError("validation loader is empty or contains zero target tokens")
        result = total_loss / total_tokens
        if not math.isfinite(result):
            self._record_numeric_failure("validation", None)
            raise FloatingPointError(self._numeric_failure_message("validation", None))
        return result

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
            validation_loss = self._evaluate()
            self._latest_validation_loss = validation_loss
            self._record_validation_metrics(validation_loss)
            self._last_validation_step = step

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
            self._record_metrics(values, send_to_wandb=True)

        should_save = self._event_due(
            "checkpoint_every_n_steps", "checkpoint_every_n_tokens", epoch_end
        )
        checkpoint_path: Path | None = None
        if should_save and self._last_checkpoint_step != step:
            checkpoint_path = self._save_checkpoint()
            self._last_checkpoint_step = step
            self._record_metrics(
                {
                    "event": "checkpoint",
                    "optimizer_step": step,
                    "target_tokens": self.target_tokens,
                    "elapsed_seconds": self.elapsed_seconds,
                },
                send_to_wandb=False,
            )

        # Milestones are independent of both logging and recovery saves.  A
        # milestone needs a state file for W&B, so create one only when that
        # event is due and checkpoint cadence did not already create it.
        milestone_due = self._event_due(
            "milestone_every_n_steps", "milestone_every_n_tokens", False
        )
        if milestone_due:
            self._record_metrics(
                {
                    "event": "milestone",
                    "optimizer_step": step,
                    "target_tokens": self.target_tokens,
                    "elapsed_seconds": self.elapsed_seconds,
                },
                send_to_wandb=False,
            )
        if milestone_due and self.run is not None:
            if checkpoint_path is None:
                checkpoint_path = self._save_checkpoint()
            self._log_model_artifact(
                run=self.run,
                checkpoint_path=checkpoint_path,
                epoch_number=step,
            )

    def _record_step_metrics(self, loss: float, token_count: int) -> None:
        self._record_metrics(
            {
                "event": "step",
                "optimizer_step": self.optimizer_step,
                "target_tokens": self.target_tokens,
                "elapsed_seconds": self.elapsed_seconds,
                "train/loss_step": loss,
                "train/target_tokens_step": token_count,
            },
            send_to_wandb=False,
        )

    def _record_validation_metrics(self, validation_loss: float) -> None:
        self._record_metrics(
            {
                "event": "validation",
                "optimizer_step": self.optimizer_step,
                "target_tokens": self.target_tokens,
                "elapsed_seconds": self.elapsed_seconds,
                "validation/loss": validation_loss,
                "validation/perplexity": _perplexity(validation_loss),
            },
            send_to_wandb=True,
        )

    def _record_metrics(self, values: dict[str, Any], *, send_to_wandb: bool) -> None:
        record = {key: value for key, value in values.items() if value is not None}
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
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / "model_last.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

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
        return due_step or due_token or (
            step_cadence is None and token_cadence is None and epoch_end
        )

    def _gradients_are_finite(self) -> bool:
        return all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in self.model.parameters()
        )

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
        preceding_checkpoint = self.checkpoint_dir / "model_last.pth"
        self._record_metrics(
            {
                "event": f"nonfinite_{kind}",
                "optimizer_step": self.optimizer_step,
                "target_tokens": self.target_tokens,
                "batch_index": batch_index,
                "elapsed_seconds": self.elapsed_seconds,
                "error": message,
                "preceding_checkpoint_step": self._last_checkpoint_step,
                "preceding_checkpoint": str(preceding_checkpoint)
                if preceding_checkpoint.exists()
                else None,
            },
            send_to_wandb=False,
        )

    def _remaining_tokens(self) -> int | None:
        if self.max_tokens is None:
            return None
        return max(0, int(self.max_tokens) - self.target_tokens)

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
