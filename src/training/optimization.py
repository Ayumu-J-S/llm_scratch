from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


SCHEDULER_META_KEYS = {"enabled", "interval"}


class WarmupCosineScheduler:
    """Warm up then decay learning rates by completed optimizer updates.

    The scheduler intentionally has a small, explicit contract instead of a
    chain of generic scheduler wrappers: ``step()`` is called once after each
    successful optimizer update and changes the rate for the next update.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        warmup_steps: int,
        decay_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> None:
        if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int):
            raise ValueError("warmup_steps must be a non-negative integer")
        if isinstance(decay_steps, bool) or not isinstance(decay_steps, int):
            raise ValueError("decay_steps must be a positive integer")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if decay_steps < 1 or decay_steps < warmup_steps:
            raise ValueError("decay_steps must be positive and at least warmup_steps")
        if not math.isfinite(float(min_lr_ratio)) or not 0.0 <= float(min_lr_ratio) <= 1.0:
            raise ValueError("min_lr_ratio must be finite and between zero and one")

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.optimizer_steps = 0
        self._last_lr: list[float] = []
        self._set_learning_rates(self._scale_for_update(0))

    def _scale_for_update(self, update_index: int) -> float:
        if self.warmup_steps:
            if update_index < self.warmup_steps:
                return float(update_index + 1) / float(self.warmup_steps)
            if self.decay_steps == self.warmup_steps:
                return self.min_lr_ratio
            decay_progress = (update_index - self.warmup_steps) / (
                self.decay_steps - self.warmup_steps
            )
        else:
            decay_progress = update_index / self.decay_steps

        progress = min(max(decay_progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def _set_learning_rates(self, scale: float) -> None:
        self._last_lr = []
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups, strict=True):
            learning_rate = base_lr * scale
            group["lr"] = learning_rate
            self._last_lr.append(learning_rate)

    def step(self) -> None:
        self.optimizer_steps += 1
        self._set_learning_rates(self._scale_for_update(self.optimizer_steps))

    def get_last_lr(self) -> list[float]:
        return list(self._last_lr)

    def state_dict(self) -> dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": list(self.base_lrs),
            "optimizer_steps": self.optimizer_steps,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        expected = {
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": self.base_lrs,
        }
        observed = {key: state_dict[key] for key in expected}
        if observed != expected:
            raise ValueError("scheduler state does not match the configured schedule")
        optimizer_steps = state_dict.get("optimizer_steps")
        if isinstance(optimizer_steps, bool) or not isinstance(optimizer_steps, int):
            raise ValueError("scheduler optimizer_steps must be an integer")
        if optimizer_steps < 0:
            raise ValueError("scheduler optimizer_steps must be non-negative")
        self.optimizer_steps = optimizer_steps
        self._set_learning_rates(self._scale_for_update(self.optimizer_steps))


def autocast_context(device: torch.device, precision: str):
    """Return the configured autocast context, refusing implicit CPU BF16."""

    if precision == "fp32":
        return nullcontext()
    if precision != "bf16":
        raise ValueError("training.precision must be either 'fp32' or 'bf16'")
    if device.type != "cuda":
        raise ValueError("training.precision=bf16 requires runtime.device=cuda; use fp32 on CPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("training.precision=bf16 was requested but CUDA BF16 is unsupported")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def build_optimizer(model, optimizer_cfg: DictConfig):
    optimizer_config = OmegaConf.to_container(optimizer_cfg, resolve=True)
    return instantiate(optimizer_config, params=model.parameters())


def build_scheduler(optimizer, scheduler_cfg: DictConfig | None):
    if scheduler_cfg is None or not scheduler_cfg.get("enabled", True):
        return None

    scheduler_config = OmegaConf.to_container(scheduler_cfg, resolve=True)
    for key in SCHEDULER_META_KEYS:
        scheduler_config.pop(key, None)

    return instantiate(scheduler_config, optimizer=optimizer)


def get_learning_rate(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]
