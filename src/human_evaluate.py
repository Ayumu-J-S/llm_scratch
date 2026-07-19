"""Hydra entrypoint for HUMAN-001 blinded continuation evaluation."""

from __future__ import annotations

import json

import hydra
from omegaconf import DictConfig, OmegaConf

from human_evaluation.workflow import run_from_config


@hydra.main(version_base=None, config_path="../config", config_name="human_evaluation")
def main(cfg: DictConfig) -> None:
    """Run one key, preparation, or score-import action."""

    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError("human evaluation requires a mapping Hydra configuration")
    print(json.dumps(run_from_config(resolved), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
