"""Installed console entrypoints for canonical Hydra workflows."""

from pathlib import Path
import sys

import hydra


def _compose():
    config_dir = Path.cwd() / "config"
    if not config_dir.is_dir():
        config_dir = Path(__file__).resolve().parent.parent / "config"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return hydra.compose(config_name="train", overrides=sys.argv[1:])


def train() -> None:
    from train import main

    main.__wrapped__(_compose())


def config_check() -> None:
    from train import run_config_check

    run_config_check(_compose())
