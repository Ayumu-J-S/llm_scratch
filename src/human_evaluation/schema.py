"""Strict, versioned schemas for the HUMAN-001 evaluation workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROMPT_SCHEMA_VERSION = "human-evaluation-prompts-v1"
PUBLIC_BUNDLE_SCHEMA_VERSION = "human-evaluation-bundle-v1"
PRIVATE_MAPPING_SCHEMA_VERSION = "human-evaluation-private-v1"
SCORE_SCHEMA_VERSION = "human-evaluation-scores-v1"
RESULT_SCHEMA_VERSION = "human-evaluation-result-v1"
PROTOCOL_VERSION = "HUMAN-001-protocol-v1"
EVALUATOR_REVISION = "checkpoint-sampler-v1"

PROMPT_SET_SIZE = 8
PROMPTS_PER_LANGUAGE = 4
SUPPORTED_LANGUAGES = frozenset({"ja", "en"})

MAX_NEW_TOKENS = 64
TEMPERATURE = 0.8
TOP_K = 40
MINIMUM_RELATIVE_TARGET_TOKEN_GAP = 0.25

RUBRIC = {
    "scale": {"minimum": 1, "maximum": 5},
    "dimensions": [
        {
            "id": "fluency",
            "question": "Is the continuation grammatical and readable in the prompt language?",
            "anchors": {"1": "unreadable", "3": "mixed", "5": "fluent"},
        },
        {
            "id": "coherence",
            "question": "Does the continuation form a coherent continuation of the prompt?",
            "anchors": {"1": "unrelated", "3": "partly coherent", "5": "coherent"},
        },
        {
            "id": "naturalness",
            "question": "Does the continuation read like natural Japanese or English prose?",
            "anchors": {"1": "unnatural", "3": "mixed", "5": "natural"},
        },
    ],
    "preference": {
        "values": ["A", "B", "tie"],
        "question": "Which continuation is better overall for this base-model prompt?",
    },
    "instruction": (
        "Score continuation quality, not instruction following or chat behavior. "
        "Use the full 1-5 scale and choose A, B, or tie independently for every item."
    ),
}


class EvaluationSchemaError(ValueError):
    """A HUMAN-001 input or output does not satisfy its public contract."""


@dataclass(frozen=True)
class Prompt:
    """One versioned base-model continuation prompt."""

    id: str
    language: str
    text: str


@dataclass(frozen=True)
class PromptSet:
    """The complete balanced HUMAN-001 prompt set."""

    version: str
    prompts: tuple[Prompt, ...]


def load_prompt_set(path: str | Path) -> PromptSet:
    """Load exactly eight distinct prompts, balanced across Japanese and English."""

    try:
        payload_bytes = Path(path).read_bytes()
    except OSError as error:
        raise EvaluationSchemaError(f"cannot read prompt set: {error}") from error
    return load_prompt_set_bytes(payload_bytes)


def load_prompt_set_bytes(payload_bytes: bytes) -> PromptSet:
    """Parse one captured prompt-set byte buffer."""

    try:
        payload = json.loads(payload_bytes.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationSchemaError(f"cannot read prompt set: {error}") from error
    if not isinstance(payload, dict):
        raise EvaluationSchemaError("prompt set must be an object")
    _exact_fields(payload, {"schema_version", "prompt_set_version", "prompts"}, "prompt set")
    if payload["schema_version"] != PROMPT_SCHEMA_VERSION:
        raise EvaluationSchemaError("prompt set schema_version is unsupported")
    version = _nonempty_string(payload["prompt_set_version"], "prompt_set_version")
    raw_prompts = payload["prompts"]
    if not isinstance(raw_prompts, list) or len(raw_prompts) != PROMPT_SET_SIZE:
        raise EvaluationSchemaError(f"prompt set must contain exactly {PROMPT_SET_SIZE} prompts")

    prompts: list[Prompt] = []
    ids: set[str] = set()
    texts: set[str] = set()
    language_counts = {language: 0 for language in SUPPORTED_LANGUAGES}
    for index, raw_prompt in enumerate(raw_prompts):
        if not isinstance(raw_prompt, dict):
            raise EvaluationSchemaError(f"prompt {index} must be an object")
        _exact_fields(raw_prompt, {"id", "language", "text"}, f"prompt {index}")
        prompt_id = _nonempty_string(raw_prompt["id"], f"prompt {index}.id")
        language = _nonempty_string(raw_prompt["language"], f"prompt {index}.language")
        text = _nonempty_string(raw_prompt["text"], f"prompt {index}.text")
        if language not in SUPPORTED_LANGUAGES:
            raise EvaluationSchemaError(f"prompt {prompt_id!r} has unsupported language")
        if prompt_id in ids or text in texts:
            raise EvaluationSchemaError("prompt IDs and texts must be unique")
        ids.add(prompt_id)
        texts.add(text)
        language_counts[language] += 1
        prompts.append(Prompt(id=prompt_id, language=language, text=text))
    if any(count != PROMPTS_PER_LANGUAGE for count in language_counts.values()):
        raise EvaluationSchemaError(
            f"prompt set must contain {PROMPTS_PER_LANGUAGE} prompts per language"
        )
    return PromptSet(version=version, prompts=tuple(prompts))


def _exact_fields(payload: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise EvaluationSchemaError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationSchemaError(f"{label} must be a non-empty string")
    return value
