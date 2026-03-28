from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


SCHEDULER_META_KEYS = {"enabled", "interval"}


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
