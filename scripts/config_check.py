"""Canonical Hydra profile composition and preflight command."""

import hydra

from train import run_config_check


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg):
    run_config_check(cfg)


if __name__ == "__main__":
    main()
