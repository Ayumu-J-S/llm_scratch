import math
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        *,
        model,
        optimizer,
        train_loader,
        validation_loader,
        checkpoint_dir: Path,
        cfg: DictConfig,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.checkpoint_dir = checkpoint_dir
        self.cfg = cfg
        self.device = device

    def fit(self) -> None:
        run = self._init_wandb()

        print("Training...", flush=True)
        try:
            for epoch_index in range(self.cfg.training.epochs):
                train_loss = self._train_epoch(epoch_index)
                validation_loss = self._evaluate()
                train_perplexity = math.exp(train_loss)
                validation_perplexity = math.exp(validation_loss)

                metrics = {
                    "epoch": epoch_index + 1,
                    "train/loss": train_loss,
                    "train/perplexity": train_perplexity,
                    "validation/loss": validation_loss,
                    "validation/perplexity": validation_perplexity,
                }

                print(
                    f"Epoch {epoch_index + 1}: "
                    f"train_loss={train_loss:.6f} "
                    f"val_loss={validation_loss:.6f} "
                    f"train_ppl={train_perplexity:.3f} "
                    f"val_ppl={validation_perplexity:.3f}",
                    flush=True,
                )

                if run is not None:
                    run.log(metrics)

                self._save_checkpoint()
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

        run_url = getattr(run, "url", None)
        if run_url:
            print(f"W&B run URL: {run_url}", flush=True)
        else:
            print(f"W&B run directory: {run.dir}", flush=True)

        return run

    def _save_checkpoint(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.checkpoint_dir / "model_last.pth")
