from pathlib import Path

import pytest

from human_evaluation.schema import EvaluationSchemaError, load_prompt_set


PROMPT_SET_PATH = Path("evaluation/human/prompts-v1.json")


def test_versioned_prompt_set_has_four_japanese_and_four_english_prompts():
    prompt_set = load_prompt_set(PROMPT_SET_PATH)

    assert prompt_set.version == "HUMAN-001-v1"
    assert len(prompt_set.prompts) == 8
    assert [prompt.language for prompt in prompt_set.prompts].count("ja") == 4
    assert [prompt.language for prompt in prompt_set.prompts].count("en") == 4


def test_prompt_schema_rejects_duplicate_text(tmp_path: Path):
    duplicate = PROMPT_SET_PATH.read_text(encoding="utf-8").replace(
        "The research team began by checking whether",
        "At the edge of the old harbor, the morning fog",
    )
    path = tmp_path / "prompts.json"
    path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(EvaluationSchemaError, match="unique"):
        load_prompt_set(path)
