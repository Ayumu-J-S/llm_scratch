from pathlib import Path

import hydra

from dgx.runtime_image import RUNTIME_SPEC_LABEL, runtime_image_spec_sha256


ROOT = Path(__file__).resolve().parent.parent


def test_committed_runtime_spec_matches_exact_build_inputs():
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        config = hydra.compose(config_name="dgx")
    assert config.image.expected_runtime_spec_sha256 == runtime_image_spec_sha256(ROOT)


def test_dependency_image_has_no_source_or_config_self_reference():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY . /workspace" not in dockerfile
    assert f'{RUNTIME_SPEC_LABEL}="${{RUNTIME_SPEC_SHA256}}"' in dockerfile
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "--build-arg RUNTIME_SPEC_SHA256=" in makefile
    assert "--volume $(CURDIR):$(CURDIR):ro" in makefile
