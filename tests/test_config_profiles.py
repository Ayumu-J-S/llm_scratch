from pathlib import Path

import hydra
import pytest
from omegaconf import OmegaConf

from runtime.config import ConfigPreflightError, validate_training_config


CONFIG_DIR = Path(__file__).parents[1] / "config"


def compose(*overrides: str):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return hydra.compose(config_name="train", overrides=list(overrides))


def test_canonical_profiles_compose_with_real_root_sections():
    smoke = compose("profile=smoke_overfit")
    stream = compose("profile=pretrain_streaming")
    evaluation = compose("profile=evaluation")

    assert smoke.profile.name == "smoke_overfit"
    assert smoke.data.mode == "memorization_smoke"
    assert stream.profile.name == "pretrain_streaming"
    assert stream.data.mode == "streaming"
    assert evaluation.profile.name == "evaluation"
    assert evaluation.profile.task == "evaluate_checkpoint"


def test_streaming_profile_has_distinct_manifest_selections():
    config = compose("profile=pretrain_streaming")
    validate_training_config(config)
    train = config.data.streaming.train.sources[0]
    validation = config.data.streaming.validation.sources[0]
    assert train.selection == "train"
    assert validation.selection == "validation"


def test_preflight_rejects_empty_sources():
    config = compose("profile=pretrain_streaming")
    config.data.streaming.train.sources = []
    with pytest.raises(ConfigPreflightError, match="at least one source"):
        validate_training_config(config)


def test_preflight_rejects_identical_real_train_and_validation():
    config = compose("profile=pretrain_streaming")
    source = OmegaConf.to_container(config.data.streaming.train.sources[0], resolve=True)
    config.data.streaming.validation.sources = [source]
    with pytest.raises(ConfigPreflightError, match="must be distinct"):
        validate_training_config(config)


def test_preflight_rejects_unknown_critical_keys():
    config = compose("profile=pretrain_streaming")
    OmegaConf.set_struct(config, False)
    config.data.streaming.train.sorces = config.data.streaming.train.sources
    with pytest.raises(ConfigPreflightError, match="unknown critical"):
        validate_training_config(config)
