"""Pinned BENCH-001 suite data and protocol definitions.

The canonical CLI never takes a registry path from Hydra.  Both development
and reserved-final evaluation use the repository-owned registry and its
compiled expected fingerprint; only the separately guarded final entrypoint
can request the final source entries.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from data.identity import canonical_fingerprint, canonical_json_bytes
from data.stream_loader.cache import BoundedShardCache, download_url_to_path


BenchmarkAccess = Literal["dev", "final"]

SUITE_ID = "BENCH-001-suite-v1"
FINAL_ACKNOWLEDGEMENT = SUITE_ID
CANONICAL_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data/benchmarks/suite-v1.json"
CANONICAL_REGISTRY_FINGERPRINT = "39e658f55b445b5390a01390523b077018a1337c257ad82f0a167085636b7bd2"
JCOMMONSENSEQA_PROMPT_REVISION = "BENCH-001-jcommonsenseqa-zero-shot-v1"
JCOMMONSENSEQA_SCORER_REVISION = "BENCH-001-conditional-logprob-v2"
GSM8K_PROMPT_REVISION = "BENCH-001-gsm8k-zero-shot-v1"
GSM8K_SCORER_REVISION = "openai-gsm8k-ANS_RE-v1"
GSM8K_MAX_NEW_TOKENS = 128
PROTOCOL_MINIMUM_CONTEXT_LENGTH = GSM8K_MAX_NEW_TOKENS + 1
GENERATED_TOKEN_TRACE_REVISION = "canonical-json-token-ids-sha256-v1"
SUBSET_SELECTOR_REVISION = "sha256-example-id-v1"
INVALID_GSM8K_ANSWER = "[invalid]"
_GSM8K_ANSWER = re.compile(r"#### (\-?[0-9\.\,]+)")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")

_PROMPT_SPECS = {
    "jcommonsenseqa": {
        "revision": JCOMMONSENSEQA_PROMPT_REVISION,
        "prefix": "質問: {question}\n選択肢:\n",
        "choice": "{A-E}. {choice}",
        "suffix": "\n答え:\n",
        "few_shot_examples": 0,
    },
    "gsm8k": {
        "revision": GSM8K_PROMPT_REVISION,
        "template": "Question: {question}\nAnswer:\n",
        "few_shot_examples": 0,
    },
}
_SCORER_SPECS = {
    "jcommonsenseqa": {
        "revision": JCOMMONSENSEQA_SCORER_REVISION,
        "candidate": "choice text",
        "tokenization": "exact prompt-plus-choice string encoded once",
        "target_span": "offset-masked choice suffix; boundary-crossing tokens rejected",
        "tie_break": "lowest choice index",
        "primary": "sum_log_probability / continuation_token_count",
        "secondary": "sum_log_probability",
    },
    "gsm8k": {
        "revision": GSM8K_SCORER_REVISION,
        "answer_regex": r"#### (\-?[0-9\.\,]+)",
        "normalization": "strip then remove commas",
        "comparison": "exact string equality",
        "generated_token_trace": {
            "revision": GENERATED_TOKEN_TRACE_REVISION,
            "encoding": "canonical JSON array of integer token IDs",
            "retention": "SHA-256 only; raw IDs excluded",
        },
    },
}


class BenchmarkSuiteError(ValueError):
    """The pinned suite registry or one of its source artifacts is invalid."""


@dataclass(frozen=True)
class BenchmarkExample:
    task: str
    example_id: str
    record: dict[str, Any]


@dataclass(frozen=True)
class LoadedTask:
    name: str
    examples: tuple[BenchmarkExample, ...]
    source_identity: dict[str, Any]


@dataclass(frozen=True)
class LoadedSuite:
    suite_id: str
    suite_fingerprint: str
    access: BenchmarkAccess
    protocol: dict[str, Any]
    protocol_sha256: str
    tasks: tuple[LoadedTask, ...]

    @property
    def example_count(self) -> int:
        return sum(len(task.examples) for task in self.tasks)

    def identity(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "suite_fingerprint": self.suite_fingerprint,
            "access": self.access,
            "protocol": self.protocol,
            "protocol_sha256": self.protocol_sha256,
            "component_hashes": protocol_component_hashes(),
            "tasks": {
                task.name: task.source_identity
                for task in sorted(self.tasks, key=lambda item: item.name)
            },
        }


def protocol_component_hashes() -> dict[str, dict[str, str]]:
    """Return explicit prompt and scorer hashes attached to every result."""

    return {
        task: {
            "prompt_sha256": hashlib.sha256(canonical_json_bytes(_PROMPT_SPECS[task])).hexdigest(),
            "scorer_sha256": hashlib.sha256(canonical_json_bytes(_SCORER_SPECS[task])).hexdigest(),
        }
        for task in ("jcommonsenseqa", "gsm8k")
    }


def load_suite(
    registry_path: str | Path,
    *,
    expected_fingerprint: str,
    access: BenchmarkAccess,
    cache: BoundedShardCache,
    timeout_seconds: float,
) -> LoadedSuite:
    """Load and checksum-verify only the source partition authorized by ``access``."""

    if access not in {"dev", "final"}:
        raise BenchmarkSuiteError("benchmark access must be dev or final")
    registry_file = Path(registry_path).resolve()
    registry = _load_json_object(registry_file)
    _validate_registry(registry, expected_fingerprint=expected_fingerprint)
    protocol = dict(registry["protocol"])
    tasks: list[LoadedTask] = []
    for task_name in ("jcommonsenseqa", "gsm8k"):
        task_config = _mapping(registry["tasks"][task_name], f"tasks.{task_name}")
        source = _mapping(task_config[access], f"tasks.{task_name}.{access}")
        payload = _source_bytes(
            source,
            registry_dir=registry_file.parent,
            cache=cache,
            timeout_seconds=timeout_seconds,
        )
        records = _parse_records(payload, task_name=task_name)
        if len(records) != source["expected_records"]:
            raise BenchmarkSuiteError(
                f"{task_name} {access} record count differs from its pinned registry identity"
            )
        examples = _examples(task_name, records)
        selector = "official-order-v1"
        if access == "dev":
            examples = _deterministic_subset(
                examples,
                size=int(registry["dev_subset"]["size"]),
            )
            selector = SUBSET_SELECTOR_REVISION
        selected_identity = [
            {
                "example_id": example.example_id,
                "record_sha256": hashlib.sha256(canonical_json_bytes(example.record)).hexdigest(),
            }
            for example in examples
        ]
        source_identity = {
            "repository": task_config["repository"],
            "revision": task_config["revision"],
            "revision_url": task_config["revision_url"],
            "license": task_config["license"],
            "source": dict(source),
            "subset_selector": selector,
            "selected_examples": len(examples),
            "selected_examples_sha256": hashlib.sha256(
                canonical_json_bytes(selected_identity)
            ).hexdigest(),
        }
        if access == "dev":
            expected_selected = registry["dev_subset"]["selected_examples_sha256"][task_name]
            if source_identity["selected_examples_sha256"] != expected_selected:
                raise BenchmarkSuiteError(
                    f"{task_name} selected development examples differ from the registry identity"
                )
        tasks.append(
            LoadedTask(name=task_name, examples=tuple(examples), source_identity=source_identity)
        )
    return LoadedSuite(
        suite_id=str(registry["suite_id"]),
        suite_fingerprint=str(registry["suite_fingerprint"]),
        access=access,
        protocol=protocol,
        protocol_sha256=hashlib.sha256(canonical_json_bytes(protocol)).hexdigest(),
        tasks=tuple(tasks),
    )


def canonical_external_dev_identity() -> dict[str, Any]:
    """Return the repository-owned protocol and partition for external records."""

    registry = _load_json_object(CANONICAL_REGISTRY_PATH)
    _validate_registry(
        registry,
        expected_fingerprint=CANONICAL_REGISTRY_FINGERPRINT,
    )
    subset = registry["dev_subset"]
    protocol = registry["protocol"]
    return {
        "suite_id": registry["suite_id"],
        "suite_fingerprint": registry["suite_fingerprint"],
        "access": "dev",
        "protocol_sha256": hashlib.sha256(canonical_json_bytes(protocol)).hexdigest(),
        "minimum_context_length": PROTOCOL_MINIMUM_CONTEXT_LENGTH,
        "subset_selector": subset["selector"],
        "tasks": {
            task_name: {
                "source_sha256": registry["tasks"][task_name]["dev"]["sha256"],
                "selected_examples": subset["size"],
                "selected_examples_sha256": subset["selected_examples_sha256"][task_name],
            }
            for task_name in ("jcommonsenseqa", "gsm8k")
        },
    }


def format_jcommonsenseqa(example: BenchmarkExample) -> tuple[str, tuple[str, ...], int]:
    record = example.record
    choices = tuple(str(record[f"choice{index}"]) for index in range(5))
    prompt = (
        "質問: "
        + str(record["question"])
        + "\n選択肢:\n"
        + "\n".join(f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices))
    )
    prompt += "\n答え:\n"
    return prompt, choices, int(record["label"])


def format_gsm8k(example: BenchmarkExample) -> tuple[str, str]:
    return f"Question: {example.record['question']}\nAnswer:\n", extract_gsm8k_answer(
        str(example.record["answer"])
    )


def extract_gsm8k_answer(text: str) -> str:
    """Apply the regex and comma normalization published with GSM8K."""

    match = _GSM8K_ANSWER.search(text)
    if match is None:
        return INVALID_GSM8K_ANSWER
    return match.group(1).strip().replace(",", "")


def contamination_probes(example: BenchmarkExample) -> tuple[tuple[str, str], ...]:
    """Return benchmark-owned text fields without constructing synthetic prose."""

    if example.task == "jcommonsenseqa":
        prompt, _, _ = format_jcommonsenseqa(example)
        return tuple(
            [
                ("question", str(example.record["question"])),
                ("prompt", prompt),
                (
                    "canonical_record",
                    json.dumps(
                        example.record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            ]
            + [(f"choice{index}", str(example.record[f"choice{index}"])) for index in range(5)]
        )
    if example.task == "gsm8k":
        return (
            ("question", str(example.record["question"])),
            ("reference_answer", str(example.record["answer"])),
            (
                "question_answer",
                f"{example.record['question']}\n{example.record['answer']}",
            ),
            (
                "canonical_record",
                json.dumps(
                    example.record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        )
    raise BenchmarkSuiteError(f"unsupported benchmark task: {example.task}")


def _deterministic_subset(examples: list[BenchmarkExample], *, size: int) -> list[BenchmarkExample]:
    if size < 1 or size > len(examples):
        raise BenchmarkSuiteError("development subset size is outside the source record count")
    ranked = sorted(
        examples,
        key=lambda example: (
            hashlib.sha256(
                canonical_json_bytes({"task": example.task, "example_id": example.example_id})
            ).hexdigest(),
            example.example_id,
        ),
    )
    return ranked[:size]


def _examples(task_name: str, records: list[dict[str, Any]]) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if task_name == "jcommonsenseqa":
            required = {"q_id", "question", "label", *(f"choice{i}" for i in range(5))}
            if set(record) != required:
                raise BenchmarkSuiteError("JCommonsenseQA record fields differ from v1.3")
            q_id = record["q_id"]
            if isinstance(q_id, bool) or not isinstance(q_id, int):
                raise BenchmarkSuiteError("JCommonsenseQA q_id must be an integer")
            label = record["label"]
            if isinstance(label, bool) or not isinstance(label, int) or label not in range(5):
                raise BenchmarkSuiteError("JCommonsenseQA label must be an integer in 0..4")
            for field in ["question", *(f"choice{i}" for i in range(5))]:
                if not isinstance(record[field], str) or not record[field]:
                    raise BenchmarkSuiteError(f"JCommonsenseQA {field} must be non-empty text")
            example_id = str(q_id)
        elif task_name == "gsm8k":
            if set(record) != {"question", "answer"}:
                raise BenchmarkSuiteError("GSM8K record fields differ from the pinned format")
            if not all(isinstance(record[field], str) and record[field] for field in record):
                raise BenchmarkSuiteError("GSM8K question and answer must be non-empty text")
            if extract_gsm8k_answer(record["answer"]) == INVALID_GSM8K_ANSWER:
                raise BenchmarkSuiteError("GSM8K reference answer is missing the official marker")
            example_id = str(index)
        else:
            raise BenchmarkSuiteError(f"unsupported benchmark task: {task_name}")
        if example_id in seen:
            raise BenchmarkSuiteError(f"duplicate {task_name} example ID: {example_id}")
        seen.add(example_id)
        examples.append(BenchmarkExample(task=task_name, example_id=example_id, record=record))
    return examples


def _parse_records(payload: bytes, *, task_name: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise BenchmarkSuiteError(f"{task_name} source is not valid UTF-8") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise BenchmarkSuiteError(
                f"{task_name} source line {line_number} is not valid JSON"
            ) from error
        if not isinstance(record, dict):
            raise BenchmarkSuiteError(f"{task_name} source records must be JSON objects")
        records.append(record)
    if not records:
        raise BenchmarkSuiteError(f"{task_name} source contains no records")
    return records


def _source_bytes(
    source: Mapping[str, Any],
    *,
    registry_dir: Path,
    cache: BoundedShardCache,
    timeout_seconds: float,
) -> bytes:
    if "url" in source:
        url = str(source["url"])

        def downloader(destination: Path) -> None:
            download_url_to_path(url, destination, timeout_seconds)

        key = url
    else:
        source_path = (registry_dir / str(source["path"])).resolve()
        key = source_path.as_uri()

        def downloader(destination: Path) -> None:
            destination.write_bytes(source_path.read_bytes())

    with cache.acquire(
        key,
        downloader,
        expected_sha256=str(source["sha256"]),
        expected_size_bytes=int(source["size_bytes"]),
    ) as path:
        return path.read_bytes()


def _validate_registry(registry: dict[str, Any], *, expected_fingerprint: str) -> None:
    expected_fields = {
        "schema_version",
        "suite_id",
        "dev_subset",
        "protocol",
        "tasks",
        "suite_fingerprint",
    }
    if set(registry) != expected_fields or registry.get("schema_version") != 1:
        raise BenchmarkSuiteError("benchmark registry must use the exact v1 schema")
    if registry.get("suite_id") != SUITE_ID:
        raise BenchmarkSuiteError(f"benchmark suite_id must be {SUITE_ID}")
    _require_sha256(expected_fingerprint, "expected_fingerprint")
    stored = registry.get("suite_fingerprint")
    _require_sha256(stored, "suite_fingerprint")
    payload = dict(registry)
    del payload["suite_fingerprint"]
    actual = canonical_fingerprint(payload)
    if stored != actual or expected_fingerprint != actual:
        raise BenchmarkSuiteError(
            "benchmark registry fingerprint differs from the compiled suite identity"
        )
    subset = _mapping(registry["dev_subset"], "dev_subset")
    if (
        set(subset) != {"size", "selector", "selected_examples_sha256"}
        or subset.get("selector") != SUBSET_SELECTOR_REVISION
    ):
        raise BenchmarkSuiteError("benchmark development subset policy is invalid")
    if isinstance(subset.get("size"), bool) or not isinstance(subset.get("size"), int):
        raise BenchmarkSuiteError("benchmark development subset size must be an integer")
    selected_examples = _mapping(
        subset["selected_examples_sha256"],
        "dev_subset.selected_examples_sha256",
    )
    if set(selected_examples) != {"jcommonsenseqa", "gsm8k"}:
        raise BenchmarkSuiteError("benchmark development selected-example identities are invalid")
    for task_name, value in selected_examples.items():
        _require_sha256(value, f"dev_subset.selected_examples_sha256.{task_name}")
    protocol = _mapping(registry["protocol"], "protocol")
    expected_protocol = {
        "few_shot_examples": 0,
        "jcommonsenseqa": {
            "prompt_revision": JCOMMONSENSEQA_PROMPT_REVISION,
            "scorer_revision": JCOMMONSENSEQA_SCORER_REVISION,
            "primary_metric": "length_normalized_accuracy",
        },
        "gsm8k": {
            "prompt_revision": GSM8K_PROMPT_REVISION,
            "scorer_revision": GSM8K_SCORER_REVISION,
            "decoding": {"method": "greedy", "max_new_tokens": GSM8K_MAX_NEW_TOKENS},
            "primary_metric": "exact_match",
        },
    }
    if protocol != expected_protocol:
        raise BenchmarkSuiteError("benchmark prompt, decoding, or scorer protocol is invalid")
    tasks = _mapping(registry["tasks"], "tasks")
    if set(tasks) != {"jcommonsenseqa", "gsm8k"}:
        raise BenchmarkSuiteError("benchmark registry must contain exactly two v1 tasks")
    for task_name, task_value in tasks.items():
        task = _mapping(task_value, f"tasks.{task_name}")
        if set(task) != {"repository", "revision", "revision_url", "license", "dev", "final"}:
            raise BenchmarkSuiteError(f"tasks.{task_name} fields differ from the v1 schema")
        if not isinstance(task["revision"], str) or _REVISION.fullmatch(task["revision"]) is None:
            raise BenchmarkSuiteError(f"tasks.{task_name}.revision must be a Git commit")
        for access in ("dev", "final"):
            source = _mapping(task[access], f"tasks.{task_name}.{access}")
            if set(source) not in (
                {"url", "sha256", "size_bytes", "split", "expected_records"},
                {"path", "sha256", "size_bytes", "split", "expected_records"},
            ):
                raise BenchmarkSuiteError(
                    f"tasks.{task_name}.{access} source fields differ from v1"
                )
            _require_sha256(source["sha256"], f"tasks.{task_name}.{access}.sha256")
            for field in ("size_bytes", "expected_records"):
                value = source[field]
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise BenchmarkSuiteError(
                        f"tasks.{task_name}.{access}.{field} must be positive"
                    )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkSuiteError(f"cannot read benchmark registry: {path}") from error
    if not isinstance(value, dict):
        raise BenchmarkSuiteError("benchmark registry must be a JSON object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkSuiteError(f"{label} must be a mapping")
    return value


def _require_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BenchmarkSuiteError(f"{label} must be a lowercase SHA-256 digest")
