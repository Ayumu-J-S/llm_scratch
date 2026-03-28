import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from training.optimization import get_learning_rate


class Trainer:
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
        self.checkpoint_dir = checkpoint_dir
        self.cfg = cfg
        self.device = device
        self.log_model_every_n_epoch = self.cfg.wandb.log_model_every_n_epoch
        self.scheduler_interval = self._get_scheduler_interval()

    def fit(self) -> None:
        run = self._init_wandb()

        print("Training...", flush=True)
        try:
            for epoch_index in range(self.cfg.training.epochs):
                epoch_number = epoch_index + 1
                train_loss = self._train_epoch(epoch_index)
                validation_loss = self._evaluate()
                if self.scheduler is not None and self.scheduler_interval == "epoch":
                    self._step_scheduler(validation_loss)
                train_perplexity = math.exp(train_loss)
                validation_perplexity = math.exp(validation_loss)
                learning_rate = get_learning_rate(self.optimizer)

                metrics = {
                    "epoch": epoch_number,
                    "train/loss": train_loss,
                    "train/perplexity": train_perplexity,
                    "validation/loss": validation_loss,
                    "validation/perplexity": validation_perplexity,
                    "optimizer/lr": learning_rate,
                }

                print(
                    f"Epoch {epoch_number}: "
                    f"train_loss={train_loss:.6f} "
                    f"val_loss={validation_loss:.6f} "
                    f"train_ppl={train_perplexity:.3f} "
                    f"val_ppl={validation_perplexity:.3f} "
                    f"lr={learning_rate:.6g}",
                    flush=True,
                )

                if run is not None:
                    run.log(metrics)

                checkpoint_path = self._save_checkpoint()
                if run is not None and self._should_log_model_artifact(epoch_number):
                    self._log_model_artifact(
                        run=run,
                        checkpoint_path=checkpoint_path,
                        epoch_number=epoch_number,
                    )
        finally:
            if run is not None:
                run.finish()

    def _train_epoch(self, epoch_index: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(
            self.train_loader,
            desc=f"epoch {epoch_index + 1}/{self.cfg.training.epochs}",
            leave=False,
        ):
            input_batch = batch["inputs"].to(self.device)
            label_batch = batch["labels"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(input_batch)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                label_batch.reshape(-1),
            )
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None and self.scheduler_interval == "step":
                self._step_scheduler(loss.item())

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.validation_loader:
                input_batch = batch["inputs"].to(self.device)
                label_batch = batch["labels"].to(self.device)
                logits = self.model(input_batch)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    label_batch.reshape(-1),
                )
                total_loss += loss.item()
                num_batches += 1

        self.model.train()
        return total_loss / num_batches

    def _init_wandb(self):
        if not self.cfg.wandb.enabled:
            return None

        run = wandb.init(
            project=self.cfg.wandb.project,
            entity=self.cfg.wandb.entity,
            name=self.cfg.wandb.name,
            mode=self.cfg.wandb.mode,
            config=OmegaConf.to_container(self.cfg, resolve=True),
        )

        run.define_metric("epoch")
        run.define_metric("train/loss", step_metric="epoch")
        run.define_metric("train/perplexity", step_metric="epoch")
        run.define_metric("validation/loss", step_metric="epoch")
        run.define_metric("validation/perplexity", step_metric="epoch")
        run.define_metric("optimizer/lr", step_metric="epoch")

        run_url = getattr(run, "url", None)
        if run_url:
            print(f"W&B run URL: {run_url}", flush=True)
        else:
            print(f"W&B run directory: {run.dir}", flush=True)

        return run

    def _save_checkpoint(self) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / "model_last.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _should_log_model_artifact(self, epoch_number: int) -> bool:
        if epoch_number == self.cfg.training.epochs:
            return True
        if self.log_model_every_n_epoch is None:
            return False
        return epoch_number % self.log_model_every_n_epoch == 0

    def _log_model_artifact(
        self,
        *,
        run,
        checkpoint_path: Path,
        epoch_number: int,
    ) -> None:
        artifact = wandb.Artifact(
            name=self._model_artifact_name(),
            type="model",
            metadata={"epoch": epoch_number},
        )
        artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)
        aliases = [f"epoch-{epoch_number}", "latest"]
        if epoch_number == self.cfg.training.epochs:
            aliases.append("final")
        run.log_artifact(artifact, aliases=aliases)
        print(
            f"Logged model artifact for epoch {epoch_number}: "
            f"{checkpoint_path.name} ({', '.join(aliases)})",
            flush=True,
        )

    def _model_artifact_name(self) -> str:
        model_name = self.model.__class__.__name__.strip()
        if not model_name:
            return "model"
        normalized_name = re.sub(r"(?<!^)(?=[A-Z])", "-", model_name).lower()
        return normalized_name

    def _get_scheduler_interval(self) -> str:
        if self.scheduler is None:
            return "epoch"

        interval = self.cfg.training.scheduler.get("interval", "epoch")
        if interval not in {"epoch", "step"}:
            raise ValueError(
                "training.scheduler.interval must be either 'epoch' or 'step'."
            )
        if interval == "step" and isinstance(self.scheduler, ReduceLROnPlateau):
            raise ValueError(
                "ReduceLROnPlateau only supports training.scheduler.interval='epoch'."
            )
        return interval

    def _step_scheduler(self, metric: float | None = None) -> None:
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metric is None:
                raise ValueError(
                    "ReduceLROnPlateau requires a metric when stepping."
                )
            self.scheduler.step(metric)
            return

        self.scheduler.step()
