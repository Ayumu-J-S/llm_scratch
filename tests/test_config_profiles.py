from pathlib import Path

import hydra
import pytest
from omegaconf import OmegaConf

import train as train_module
from runtime.config import ConfigPreflightError, validate_training_config


CONFIG_DIR = Path(__file__).parents[1] / "config"


def compose(*overrides: str):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return hydra.compose(config_name="train", overrides=list(overrides))


def test_canonical_profiles_compose_with_real_root_sections():
    smoke = compose("profile=smoke_overfit")
    stream = compose("profile=pretrain_streaming")
    gate = compose("profile=gate_overfit")
    evaluation = compose("profile=evaluation")

    assert smoke.profile.name == "smoke_overfit"
    assert smoke.data.mode == "memorization_smoke"
    assert stream.profile.name == "pretrain_streaming"
    assert stream.data.mode == "streaming"
    assert gate.profile.name == "gate_overfit"
    assert gate.profile.purpose == "memorization_gate"
    assert gate.data.mode == "streaming"
    assert evaluation.profile.name == "evaluation"
    assert evaluation.profile.task == "evaluate_checkpoint"


def test_streaming_profile_has_distinct_manifest_selections():
    config = compose("profile=pretrain_streaming")
    validate_training_config(config)
    assert config.data.streaming.repeat is True
    train = config.data.streaming.train.sources[0]
    validation = config.data.streaming.validation.sources[0]
    assert train.selection == "train"
    assert validation.selection == "validation"


def test_stability_smoke_exposes_the_bf16_update_recipe():
    config = compose("profile=stability_smoke")
    validate_training_config(config)

    assert config.runtime.device == "cuda"
    assert config.training.max_steps == 100
    assert config.training.precision == "bf16"
    assert config.training.gradient_accumulation_steps == 4
    assert config.training.max_grad_norm == 10.0
    assert config.training.optimizer._target_ == "torch.optim.AdamW"
    assert config.training.optimizer.betas == [0.9, 0.95]
    assert config.training.scheduler.interval == "step"
    assert config.training.scheduler.warmup_steps == 10
    assert config.training.scheduler.decay_steps == 100


def test_gate_overfit_uses_versioned_distinct_fixture_manifests_without_validation_events():
    config = compose("profile=gate_overfit")
    validate_training_config(config)

    assert config.runtime.device == "cuda"
    assert config.training.max_steps == 200
    assert config.training.validation_every_n_steps == 1000
    assert config.training.checkpoint_every_n_steps == 100
    assert config.wandb.enabled is False
    assert config.data.streaming.train.sources[0].selection == "all"
    assert config.data.streaming.validation.sources[0].selection == "all"
    assert (
        config.data.streaming.train.sources[0].expected_fingerprint
        != config.data.streaming.validation.sources[0].expected_fingerprint
    )


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


def test_evaluation_profile_is_rejected_by_training_preflight(monkeypatch):
    config = compose("profile=evaluation")
    tokenizer_touched = False

    def fail_if_tokenizer_is_loaded(*args, **kwargs):
        nonlocal tokenizer_touched
        tokenizer_touched = True
        raise AssertionError("evaluation profile must stop before tokenizer initialization")

    monkeypatch.setattr(train_module.CanonicalTokenizer, "from_config", fail_if_tokenizer_is_loaded)
    with pytest.raises(ConfigPreflightError, match="composition-only"):
        train_module.main.__wrapped__(config)
    assert not tokenizer_touched


@pytest.mark.parametrize(
    ("profile_name", "mode", "purpose"),
    [
        ("smoke_overfit", "streaming", "pretraining"),
        ("pretrain_streaming", "memorization_smoke", "memorization_smoke"),
    ],
)
def test_preflight_rejects_profile_mode_mismatch(profile_name, mode, purpose):
    config = compose(f"profile={profile_name}")
    config.data.mode = mode
    config.profile.purpose = purpose
    with pytest.raises(ConfigPreflightError, match="must use data.mode"):
        validate_training_config(config)
