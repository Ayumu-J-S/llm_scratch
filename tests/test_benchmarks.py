from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import unicodedata
import weakref
from pathlib import Path
from typing import Any

import hydra
import pytest
import torch
from omegaconf import OmegaConf

import benchmark as benchmark_cli
import benchmarks.contamination as contamination_scans
import benchmarks.external as external_records
import benchmarks.runner as benchmark_runner
from benchmarks.contamination import (
    SHINGLE_CODEPOINTS,
    ContaminationScanError,
    _canonical_json_objects,
    _JsonTraversalLimitExceeded,
    _JsonTraversalLimits,
    _ShingleMatcher,
    _build_probe_index,
    _document_matches,
    scan_checkpoint_training_data,
)
from benchmarks.external import ExternalComparisonError, write_external_comparison
from benchmarks.runner import (
    BenchmarkContaminationError,
    _apply_evaluation_determinism_policy,
    _write_json_atomic,
    run_benchmark,
)
from benchmarks.scoring import (
    BenchmarkScoringError,
    conditional_log_probability,
    score_suite,
)
from benchmarks.suite import (
    FINAL_ACKNOWLEDGEMENT,
    GSM8K_PROMPT_REVISION,
    GSM8K_SCORER_REVISION,
    JCOMMONSENSEQA_PROMPT_REVISION,
    JCOMMONSENSEQA_SCORER_REVISION,
    SUBSET_SELECTOR_REVISION,
    BenchmarkExample,
    LoadedSuite,
    LoadedTask,
    load_suite,
    protocol_component_hashes,
)
from data.identity import canonical_fingerprint
from data.stream_loader.cache import BoundedShardCache
from data.stream_loader.loader import RawDocument
from generation.sampler import CheckpointSampler
from models.simple_decoder_transformer import SimpleDecoderTransformer
from runtime.config import ConfigPreflightError, validate_benchmark_config
from tokenizer.canonical import CanonicalTokenizer
from training.checkpoint import (
    CheckpointManager,
    build_checkpoint_identity,
    load_checkpoint_for_generation,
)


ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "config"
TOKENIZER_CONFIG = {
    "manifest_path": "assets/tokenizers/llm-jp-v1/manifest.json",
    "expected_fingerprint": "12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b",
}
PROTOCOL = {
    "contamination": {
        "probe_revision": "source-record-and-canonical-json-object-v1",
    },
    "few_shot_examples": 0,
    "jcommonsenseqa": {
        "prompt_revision": JCOMMONSENSEQA_PROMPT_REVISION,
        "scorer_revision": JCOMMONSENSEQA_SCORER_REVISION,
        "primary_metric": "length_normalized_accuracy",
    },
    "gsm8k": {
        "prompt_revision": GSM8K_PROMPT_REVISION,
        "scorer_revision": GSM8K_SCORER_REVISION,
        "decoding": {"method": "greedy", "max_new_tokens": 128},
        "primary_metric": "exact_match",
    },
}


def compose(*overrides: str):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return hydra.compose(config_name="train", overrides=list(overrides))


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> tuple[str, int]:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    ).encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _registry(
    tmp_path: Path,
    *,
    question: str = "A benchmark-only Japanese question?",
) -> tuple[Path, str]:
    dev_j = [
        {
            "q_id": 1,
            "question": question,
            "choice0": "alpha",
            "choice1": "beta",
            "choice2": "gamma",
            "choice3": "delta",
            "choice4": "epsilon",
            "label": 0,
        }
    ]
    dev_gsm = [{"question": "What is one plus one?", "answer": "Add them.\n#### 2"}]
    j_sha, j_size = _write_jsonl(tmp_path / "j-dev.jsonl", dev_j)
    gsm_sha, gsm_size = _write_jsonl(tmp_path / "gsm-dev.jsonl", dev_gsm)
    missing_sha = "0" * 64
    registry = {
        "schema_version": 1,
        "suite_id": FINAL_ACKNOWLEDGEMENT,
        "dev_subset": {
            "size": 1,
            "selector": SUBSET_SELECTOR_REVISION,
            "selected_examples_sha256": {
                "jcommonsenseqa": canonical_fingerprint(
                    [
                        {
                            "example_id": "1",
                            "record_sha256": canonical_fingerprint(dev_j[0]),
                            "source_record_sha256": hashlib.sha256(
                                json.dumps(
                                    dev_j[0],
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            ).hexdigest(),
                        }
                    ]
                ),
                "gsm8k": canonical_fingerprint(
                    [
                        {
                            "example_id": "0",
                            "record_sha256": canonical_fingerprint(dev_gsm[0]),
                            "source_record_sha256": hashlib.sha256(
                                json.dumps(
                                    dev_gsm[0],
                                    ensure_ascii=False,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            ).hexdigest(),
                        }
                    ]
                ),
            },
        },
        "protocol": PROTOCOL,
        "tasks": {
            "jcommonsenseqa": {
                "repository": "https://github.com/yahoojapan/JGLUE",
                "revision": "6f071c09316baae89c3d083a90985b4b1cb9968c",
                "revision_url": "https://github.com/yahoojapan/JGLUE/commit/6f071c09316baae89c3d083a90985b4b1cb9968c",
                "license": "CC-BY-SA-4.0",
                "dev": {
                    "path": "j-dev.jsonl",
                    "sha256": j_sha,
                    "size_bytes": j_size,
                    "split": "validation",
                    "expected_records": 1,
                },
                "final": {
                    "path": "missing-j-final.jsonl",
                    "sha256": missing_sha,
                    "size_bytes": 1,
                    "split": "test",
                    "expected_records": 1,
                },
            },
            "gsm8k": {
                "repository": "https://github.com/openai/grade-school-math",
                "revision": "3101c7d5072418e28b9008a6636bde82a006892c",
                "revision_url": "https://github.com/openai/grade-school-math/commit/3101c7d5072418e28b9008a6636bde82a006892c",
                "license": "MIT",
                "dev": {
                    "path": "gsm-dev.jsonl",
                    "sha256": gsm_sha,
                    "size_bytes": gsm_size,
                    "split": "train",
                    "expected_records": 1,
                },
                "final": {
                    "path": "missing-gsm-final.jsonl",
                    "sha256": missing_sha,
                    "size_bytes": 1,
                    "split": "test",
                    "expected_records": 1,
                },
            },
        },
    }
    registry["suite_fingerprint"] = canonical_fingerprint(registry)
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")
    return path, str(registry["suite_fingerprint"])


def _pretraining_checkpoint(
    tmp_path: Path,
    *,
    sequence_length: int = 256,
    precision: str = "fp32",
) -> tuple[Path, dict[str, Any]]:
    config = compose("profile=pretrain_streaming")
    config.runtime.device = "cpu"
    config.training.precision = precision
    config.training.sequence_length = sequence_length
    config.training.batch_size = 1
    config.model.embed_size = 8
    config.model.num_heads = 2
    config.model.num_layers = 1
    config.model.dropout = 0.0
    config.data.streaming.cache.dir = str(tmp_path / "training-cache")
    config.data.streaming.cache.max_size_bytes = 10_000_000
    config.data.streaming.cache.min_free_bytes = 0
    fixture_manifest = ROOT / "tests/fixtures/data_manifests/bilingual.manifest.json"
    source = {
        "name": "fixture_train",
        "type": "manifest",
        "manifest_path": str(fixture_manifest),
        "expected_fingerprint": "47cca88c4a5595e27eb5d60d99918fb77c30b23f7c0ae98024153f25e14ffc19",
        "selection": "train",
        "ratio": 1.0,
    }
    validation = {**source, "name": "fixture_validation", "selection": "validation"}
    config.data.streaming.train.sources = [source]
    config.data.streaming.validation.sources = [validation]
    plain = OmegaConf.to_container(config, resolve=True)
    assert isinstance(plain, dict)
    tokenizer = CanonicalTokenizer.from_config(TOKENIZER_CONFIG)
    identity = build_checkpoint_identity(plain)
    identity["tokenizer_fingerprint"] = tokenizer.fingerprint
    model = SimpleDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_size=8,
        num_heads=2,
        max_len=sequence_length,
        num_layers=1,
        dropout=0.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpoint = CheckpointManager(
        tmp_path / "checkpoints", keep_last_n=1, identity=identity
    ).save_final(
        {
            "model": model.state_dict(),
            "counters": {"optimizer_step": 4, "target_tokens": 512, "elapsed_seconds": 1.0},
            "resolved_config": plain,
            "run_identity": identity,
        }
    )
    return checkpoint, plain


def _benchmark_config(tmp_path: Path, checkpoint: Path, output: str = "result.json"):
    config = compose(
        "profile=benchmark",
        f"benchmark.checkpoint_path={checkpoint}",
        "benchmark.device=cpu",
        f"benchmark.output_root={tmp_path / 'benchmark-results'}",
        f"benchmark.output_path={output}",
        f"benchmark.cache.dir={tmp_path / 'benchmark-cache'}",
        "benchmark.cache.max_size_bytes=10000000",
        "benchmark.cache.min_free_bytes=0",
        "benchmark.wandb.mode=disabled",
    )
    validate_benchmark_config(config)
    return config


def test_dev_suite_is_deterministic_and_never_opens_reserved_sources(tmp_path: Path):
    registry, fingerprint = _registry(tmp_path)
    cache = BoundedShardCache(tmp_path / "cache", max_size_bytes=10_000_000)

    first = load_suite(
        registry,
        expected_fingerprint=fingerprint,
        access="dev",
        cache=cache,
        timeout_seconds=1.0,
    )
    second = load_suite(
        registry,
        expected_fingerprint=fingerprint,
        access="dev",
        cache=cache,
        timeout_seconds=1.0,
    )

    assert first.identity() == second.identity()
    assert first.example_count == 2
    assert {task.source_identity["subset_selector"] for task in first.tasks} == {
        SUBSET_SELECTOR_REVISION
    }
    for hashes in first.identity()["component_hashes"].values():
        assert len(hashes["prompt_sha256"]) == 64
        assert len(hashes["scorer_sha256"]) == 64
    assert not (tmp_path / "missing-j-final.jsonl").exists()
    assert not (tmp_path / "missing-gsm-final.jsonl").exists()


def test_final_access_cannot_be_granted_by_hydra(monkeypatch, tmp_path: Path):
    config = compose("profile=benchmark", "benchmark.checkpoint_path=/tmp/checkpoint.pt")
    assert "access" not in config.benchmark
    assert "registry_path" not in config.benchmark
    with pytest.raises(ConfigPreflightError, match="unknown critical"):
        injected = compose(
            "profile=benchmark",
            "benchmark.checkpoint_path=/tmp/checkpoint.pt",
            "+benchmark.access=final",
        )
        validate_benchmark_config(injected)
    with pytest.raises(PermissionError, match="BENCHMARK_FINAL_ACK"):
        run_benchmark(config, access="final")

    monkeypatch.delenv("BENCHMARK_FINAL_ACK", raising=False)
    monkeypatch.setattr(
        benchmark_cli,
        "_final_hydra",
        lambda: pytest.fail("final Hydra composition must not start without acknowledgement"),
    )
    with pytest.raises(SystemExit, match="BENCHMARK_FINAL_ACK"):
        benchmark_cli.final_main()


@pytest.mark.parametrize(
    ("entrypoint", "requires_final_ack"),
    [
        ("llm-scratch-benchmark", False),
        ("llm-scratch-benchmark-final", True),
    ],
)
def test_installed_benchmark_entrypoints_resolve_canonical_hydra_config(
    tmp_path: Path,
    entrypoint: str,
    requires_final_ack: bool,
):
    executable = Path(sys.executable).with_name(entrypoint)
    assert executable.is_file()
    environment = dict(os.environ)
    if requires_final_ack:
        environment["BENCHMARK_FINAL_ACK"] = FINAL_ACKNOWLEDGEMENT

    completed = subprocess.run(
        [str(executable), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Configuration groups" in completed.stdout
    assert "profile: benchmark" in completed.stdout


def test_injected_contamination_reports_source_and_document_id(monkeypatch, tmp_path: Path):
    registry, fingerprint = _registry(tmp_path, question="小さなモデルを一から学習する。")
    checkpoint, checkpoint_config = _pretraining_checkpoint(tmp_path)
    suite = load_suite(
        registry,
        expected_fingerprint=fingerprint,
        access="dev",
        cache=BoundedShardCache(tmp_path / "benchmark-cache", max_size_bytes=10_000_000),
        timeout_seconds=1.0,
    )

    evidence = scan_checkpoint_training_data(
        checkpoint_config,
        suite,
        fallback_cache=BoundedShardCache(tmp_path / "fallback", max_size_bytes=10_000_000),
    )

    assert evidence["scan_complete"] is True
    assert evidence["contaminated"] is True
    match = next(
        item
        for item in evidence["matches"]
        if item["task"] == "jcommonsenseqa" and item["match_type"] in {"exact", "normalized"}
    )
    assert match["source"] == "fixture_train"
    assert match["training_document_id"] == (
        "816b305c1bb974cb2894e518fe74205acc577d53913eacd01016660c9d3104a0"
    )
    assert match["training_upstream_id"] == "ja-002"
    index_files = list((tmp_path / "training-cache/contamination-scans").glob("*.json"))
    assert len(index_files) == 1
    index = json.loads(index_files[0].read_text(encoding="utf-8"))
    assert index["index_identity_sha256"] == evidence["scan_index_identity_sha256"]
    assert index["index_identity"]["normalization_revision"] == (
        "normalize-text-identity-nfc-strip-plus-json-object-v21"
    )
    assert index["index_identity"]["json_object_normalization_revision"] == (
        "bounded-all-object-string-input-projection-json-nfc-sha256-v20"
    )
    assert index["index_identity"]["matcher_revision"] == (
        "collision-verified-rolling-hash-codepoint-v1"
    )
    assert index["index_identity"]["producer"]["dependency_lock_sha256"]
    assert index["index_identity"]["producer"]["runtime"]["packages"]["pyarrow"]

    producer_source = index["index_identity"]["producer"]["source"]
    producer_paths = {entry["path"] for entry in producer_source["files"]}
    assert producer_source["scope_revision"] == "src-python-pyproject-lock-v1"
    assert {"pyproject.toml", "uv.lock", "src/benchmarks/contamination.py"} <= producer_paths
    assert all(
        path in {"pyproject.toml", "uv.lock"} or (path.startswith("src/") and path.endswith(".py"))
        for path in producer_paths
    )
    assert not any(
        namespace in path
        for path in producer_paths
        for namespace in ("outputs/", "artifacts/", "checkpoints/", "wandb/")
    )
    assert index["index_identity"]["suite"] == suite.identity()
    indexed_source = index["index_identity"]["training_sources"][0]
    assert indexed_source["name"] == "fixture_train"
    assert indexed_source["manifest_fingerprint"] == (
        "47cca88c4a5595e27eb5d60d99918fb77c30b23f7c0ae98024153f25e14ffc19"
    )
    assert indexed_source["selection"] == "train"
    assert len(indexed_source["manifest_sha256"]) == 64
    monkeypatch.setattr(
        "benchmarks.contamination.ManifestTextSource",
        lambda *_args, **_kwargs: pytest.fail(
            "a completed manifest/suite-bound contamination index must avoid a corpus rescan"
        ),
    )
    reused = scan_checkpoint_training_data(
        checkpoint_config,
        suite,
        fallback_cache=BoundedShardCache(tmp_path / "fallback-2", max_size_bytes=10_000_000),
    )
    assert reused == evidence
    index["report"]["scanned_documents"] += 1
    index_files[0].write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ContaminationScanError, match="artifact checksum"):
        scan_checkpoint_training_data(
            checkpoint_config,
            suite,
            fallback_cache=BoundedShardCache(tmp_path / "fallback-3", max_size_bytes=10_000_000),
        )
    changed_producer = copy.deepcopy(index["index_identity"]["producer"])
    changed_producer["runtime"]["packages"]["pyarrow"] += "-changed"
    monkeypatch.setattr(contamination_scans, "_producer_identity", lambda: changed_producer)

    def require_rescan(*_args, **_kwargs):
        raise ContaminationScanError("changed producer identity requires a corpus rescan")

    monkeypatch.setattr(contamination_scans, "_scan_training_sources", require_rescan)
    with pytest.raises(ContaminationScanError, match="changed producer identity"):
        scan_checkpoint_training_data(
            checkpoint_config,
            suite,
            fallback_cache=BoundedShardCache(tmp_path / "fallback-4", max_size_bytes=10_000_000),
        )
    assert checkpoint.is_file()


def test_scan_rejects_producer_identity_change_before_cache_publication(
    monkeypatch,
    tmp_path: Path,
):
    registry, fingerprint = _registry(tmp_path)
    _, checkpoint_config = _pretraining_checkpoint(tmp_path)
    suite = load_suite(
        registry,
        expected_fingerprint=fingerprint,
        access="dev",
        cache=BoundedShardCache(tmp_path / "benchmark-cache", max_size_bytes=10_000_000),
        timeout_seconds=1.0,
    )
    before = contamination_scans._producer_identity()
    after = copy.deepcopy(before)
    after["runtime"]["packages"]["pyarrow"] += "-changed-during-scan"
    observed = iter((before, after))
    monkeypatch.setattr(contamination_scans, "_producer_identity", lambda: next(observed))

    with pytest.raises(ContaminationScanError, match="identity changed during execution"):
        scan_checkpoint_training_data(
            checkpoint_config,
            suite,
            fallback_cache=BoundedShardCache(tmp_path / "fallback", max_size_bytes=10_000_000),
        )

    assert not list((tmp_path / "training-cache/contamination-scans").glob("*.json"))


def test_relative_benchmark_cache_uses_repository_root(monkeypatch, tmp_path: Path):
    repository = tmp_path / "repository"
    caller = tmp_path / "caller"
    repository.mkdir()
    caller.mkdir()
    monkeypatch.setattr(benchmark_runner, "ROOT_DIR", repository)
    monkeypatch.chdir(caller)
    config = compose("profile=benchmark", "benchmark.checkpoint_path=/tmp/checkpoint.pt")

    cache = benchmark_runner._benchmark_cache(config.benchmark.cache)

    assert cache.cache_dir == (repository / "outputs/benchmark_cache").resolve()
    assert not (caller / "outputs/benchmark_cache").exists()


def test_all_selected_source_records_match_verbatim_and_reordered_json():
    jcommonsenseqa_examples = []
    gsm8k_examples = []
    for index in range(128):
        jcommonsenseqa_record = {
            "q_id": 10_000 + index,
            "question": f"が短い質問{index}",
            "choice0": "甲",
            "choice1": "乙",
            "choice2": "丙",
            "choice3": "丁",
            "choice4": "戊",
            "label": index % 5,
        }
        gsm8k_record = {
            "question": f"What is {index} plus one?",
            "answer": f"Add one.\n#### {index + 1}",
        }
        jcommonsenseqa_examples.append(
            BenchmarkExample(
                task="jcommonsenseqa",
                example_id=str(jcommonsenseqa_record["q_id"]),
                record=jcommonsenseqa_record,
                source_record=json.dumps(jcommonsenseqa_record, ensure_ascii=False),
            )
        )
        gsm8k_examples.append(
            BenchmarkExample(
                task="gsm8k",
                example_id=str(index),
                record=gsm8k_record,
                source_record=json.dumps(gsm8k_record, ensure_ascii=False),
            )
        )
    suite = LoadedSuite(
        suite_id=FINAL_ACKNOWLEDGEMENT,
        suite_fingerprint="fixture",
        access="dev",
        protocol=PROTOCOL,
        protocol_sha256="fixture",
        tasks=(
            LoadedTask(
                name="jcommonsenseqa",
                examples=tuple(jcommonsenseqa_examples),
                source_identity={},
            ),
            LoadedTask(name="gsm8k", examples=tuple(gsm8k_examples), source_identity={}),
        ),
    )
    probe_index = _build_probe_index(suite)
    decoded_wrapper_counts = {
        "double_serialized": {"jcommonsenseqa": 0, "gsm8k": 0},
        "nested_object_array": {"jcommonsenseqa": 0, "gsm8k": 0},
        "object_key": {"jcommonsenseqa": 0, "gsm8k": 0},
        "metadata_enriched": {"jcommonsenseqa": 0, "gsm8k": 0},
        "metadata_collision": {"jcommonsenseqa": 0, "gsm8k": 0},
        "deep_metadata_enriched": {"jcommonsenseqa": 0, "gsm8k": 0},
        "unlabeled_input": {"jcommonsenseqa": 0, "gsm8k": 0},
        "duplicate_key_input": {"jcommonsenseqa": 0, "gsm8k": 0},
        "quoted_prose": {"jcommonsenseqa": 0, "gsm8k": 0},
        "unterminated_quote_object": {"jcommonsenseqa": 0, "gsm8k": 0},
        "escaped_unterminated_quote_object": {"jcommonsenseqa": 0, "gsm8k": 0},
        "escaped_unterminated_quote_inside_object": {
            "jcommonsenseqa": 0,
            "gsm8k": 0,
        },
        "recursive_escaped_unterminated_quote_inside_object": {
            "jcommonsenseqa": 0,
            "gsm8k": 0,
        },
        "newline_unterminated_quote_object": {"jcommonsenseqa": 0, "gsm8k": 0},
        "newline_escaped_unterminated_quote_object": {
            "jcommonsenseqa": 0,
            "gsm8k": 0,
        },
        "recursive_newline_escaped_unterminated_quote_object": {
            "jcommonsenseqa": 0,
            "gsm8k": 0,
        },
        "recursive_escaped_unterminated_quote_object": {"jcommonsenseqa": 0, "gsm8k": 0},
    }

    for task in suite.tasks:
        for example in task.examples:
            exact = _document_matches(
                example.source_record,
                source_name="fixture_train",
                document_id=f"{task.name}-{example.example_id}-raw",
                upstream_id=None,
                probe_index=probe_index,
            )
            reordered_record = {
                key: unicodedata.normalize("NFD", value) if isinstance(value, str) else value
                for key, value in reversed(list(example.record.items()))
            }
            reordered = json.dumps(reordered_record, ensure_ascii=True, indent=2)
            embedded = (
                "\ufefftraining metadata\r\n"
                + '{"payload":'
                + reordered.replace("\n", "\r\n")
                + ',"origin":"training"}'
                + "\r\ntrailing metadata"
            )
            json_string_record = json.dumps(reordered, ensure_ascii=True)
            escaped_unmatched_quote = "\\" + '"'
            input_only_record = {
                key: value
                for key, value in reordered_record.items()
                if key not in ({"label", "q_id"} if task.name == "jcommonsenseqa" else {"answer"})
            }
            duplicate_key_input = (
                "{"
                + ",".join(
                    [
                        *(
                            f"{json.dumps(key)}:{json.dumps(value, ensure_ascii=True)}"
                            for key, value in input_only_record.items()
                        ),
                        *(
                            f"{json.dumps(key)}:{json.dumps(f'benign-{key}')}"
                            for key in input_only_record
                        ),
                    ]
                )
                + "}"
            )
            decoded_wrappers = {
                "double_serialized": json_string_record,
                "nested_object_array": json.dumps(
                    {"outer": [{"payloads": [reordered]}]},
                    ensure_ascii=True,
                ),
                "object_key": json.dumps({reordered: 0}, ensure_ascii=True),
                "metadata_enriched": json.dumps(
                    {**reordered_record, "source": {"dataset": "benchmark-import"}},
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                ),
                "metadata_collision": json.dumps(
                    {
                        **reordered_record,
                        "source": {
                            "é": "first",
                            unicodedata.normalize("NFD", "é"): "second",
                        },
                    },
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                ),
                "deep_metadata_enriched": (
                    '{"wrapper":' * 40
                    + json.dumps(
                        {**reordered_record, "source": {"dataset": "benchmark-import"}},
                        ensure_ascii=True,
                        indent=2,
                        sort_keys=True,
                    )
                    + "}" * 40
                ),
                "unlabeled_input": json.dumps(
                    input_only_record,
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                ),
                "duplicate_key_input": duplicate_key_input,
                "quoted_prose": f"training payload {json_string_record} end",
                "unterminated_quote_object": f'ordinary malformed " prefix {reordered} trailing',
                "escaped_unterminated_quote_object": (
                    f"ordinary malformed {escaped_unmatched_quote} prefix {reordered} trailing"
                ),
                "escaped_unterminated_quote_inside_object": (
                    '{"meta":"unterminated '
                    + escaped_unmatched_quote
                    + " prefix "
                    + json.dumps(input_only_record, ensure_ascii=True, indent=2)
                    + " trailing}"
                ),
                "recursive_escaped_unterminated_quote_inside_object": json.dumps(
                    {
                        "payload": (
                            '{"meta":"unterminated '
                            + escaped_unmatched_quote
                            + " prefix "
                            + json.dumps(input_only_record, ensure_ascii=True, indent=2)
                            + " trailing}"
                        )
                    },
                    ensure_ascii=True,
                ),
                "newline_unterminated_quote_object": (
                    'ordinary malformed\n" prefix '
                    + json.dumps(input_only_record, ensure_ascii=True, indent=2)
                    + " trailing"
                ),
                "newline_escaped_unterminated_quote_object": (
                    f"ordinary malformed\n{escaped_unmatched_quote} prefix "
                    + json.dumps(input_only_record, ensure_ascii=True, indent=2)
                    + " trailing"
                ),
                "recursive_newline_escaped_unterminated_quote_object": json.dumps(
                    {
                        "payload": (
                            f"ordinary malformed\n{escaped_unmatched_quote} prefix "
                            + json.dumps(input_only_record, ensure_ascii=True, indent=2)
                            + " trailing"
                        )
                    },
                    ensure_ascii=True,
                ),
                "recursive_escaped_unterminated_quote_object": json.dumps(
                    {
                        "payload": (
                            f"ordinary malformed {escaped_unmatched_quote} prefix "
                            f"{reordered} trailing"
                        )
                    },
                    ensure_ascii=True,
                ),
            }
            structured = _document_matches(
                embedded,
                source_name="fixture_train",
                document_id=f"{task.name}-{example.example_id}-reordered",
                upstream_id=None,
                probe_index=probe_index,
            )
            assert any(
                match["task"] == task.name
                and match["benchmark_example_id"] == example.example_id
                and match["benchmark_field"] == "source_record"
                and match["match_type"] == "exact"
                for match in exact
            )
            assert any(
                match["task"] == task.name
                and match["benchmark_example_id"] == example.example_id
                and match["benchmark_field"] == "canonical_record"
                and match["match_type"] == "structured_json"
                for match in structured
            )
            for wrapper_name, wrapper_text in decoded_wrappers.items():
                wrapped_matches = _document_matches(
                    wrapper_text,
                    source_name="fixture_train",
                    document_id=f"{task.name}-{example.example_id}-{wrapper_name}",
                    upstream_id=None,
                    probe_index=probe_index,
                )
                if any(
                    match["task"] == task.name
                    and match["benchmark_example_id"] == example.example_id
                    and match["benchmark_field"] == "canonical_record"
                    and match["match_type"] == "structured_json"
                    for match in wrapped_matches
                ):
                    decoded_wrapper_counts[wrapper_name][task.name] += 1

    assert decoded_wrapper_counts == {
        "double_serialized": {"jcommonsenseqa": 128, "gsm8k": 128},
        "nested_object_array": {"jcommonsenseqa": 128, "gsm8k": 128},
        "object_key": {"jcommonsenseqa": 128, "gsm8k": 128},
        "metadata_enriched": {"jcommonsenseqa": 128, "gsm8k": 128},
        "metadata_collision": {"jcommonsenseqa": 128, "gsm8k": 128},
        "deep_metadata_enriched": {"jcommonsenseqa": 128, "gsm8k": 128},
        "unlabeled_input": {"jcommonsenseqa": 128, "gsm8k": 128},
        "duplicate_key_input": {"jcommonsenseqa": 128, "gsm8k": 128},
        "quoted_prose": {"jcommonsenseqa": 128, "gsm8k": 128},
        "unterminated_quote_object": {"jcommonsenseqa": 128, "gsm8k": 128},
        "escaped_unterminated_quote_object": {"jcommonsenseqa": 128, "gsm8k": 128},
        "escaped_unterminated_quote_inside_object": {
            "jcommonsenseqa": 128,
            "gsm8k": 128,
        },
        "recursive_escaped_unterminated_quote_inside_object": {
            "jcommonsenseqa": 128,
            "gsm8k": 128,
        },
        "newline_unterminated_quote_object": {"jcommonsenseqa": 128, "gsm8k": 128},
        "newline_escaped_unterminated_quote_object": {
            "jcommonsenseqa": 128,
            "gsm8k": 128,
        },
        "recursive_newline_escaped_unterminated_quote_object": {
            "jcommonsenseqa": 128,
            "gsm8k": 128,
        },
        "recursive_escaped_unterminated_quote_object": {
            "jcommonsenseqa": 128,
            "gsm8k": 128,
        },
    }


def test_decoded_json_traversal_enforces_each_exact_boundary():
    record = {"key": ["value"]}
    serialized = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    candidate_bytes = len(serialized.encode("utf-8"))

    def identities(**overrides: int) -> list[str]:
        limits = {
            "total_bytes": candidate_bytes,
            "nodes": 4,
            "depth": 2,
            "decoded_strings": 2,
            **overrides,
        }
        return list(_canonical_json_objects(serialized, limits=_JsonTraversalLimits(**limits)))

    assert identities() == [serialized]
    with pytest.raises(_JsonTraversalLimitExceeded, match="total_bytes"):
        identities(total_bytes=candidate_bytes - 1)
    with pytest.raises(_JsonTraversalLimitExceeded, match="nodes"):
        identities(nodes=3)
    assert identities(depth=1) == []
    with pytest.raises(_JsonTraversalLimitExceeded, match="decoded_strings"):
        identities(decoded_strings=1)

    nested_record = '{"key":"value"}'
    recursively_encoded = json.dumps(json.dumps(nested_record))
    permissive = _JsonTraversalLimits(
        total_bytes=1_000,
        nodes=16,
        depth=3,
        decoded_strings=8,
    )
    shallow = _JsonTraversalLimits(
        total_bytes=1_000,
        nodes=16,
        depth=2,
        decoded_strings=8,
    )
    assert list(_canonical_json_objects(recursively_encoded, limits=permissive)) == [nested_record]
    with pytest.raises(_JsonTraversalLimitExceeded, match="depth"):
        list(_canonical_json_objects(recursively_encoded, limits=shallow))


def test_decoded_json_rejects_recursive_normalized_key_collisions():
    nfc_key = "é"
    nfd_key = unicodedata.normalize("NFD", nfc_key)
    colliding_record = json.dumps(
        {nfc_key: "first", nfd_key: "second"},
        ensure_ascii=True,
    )

    assert list(_canonical_json_objects(json.dumps(colliding_record))) == []


def test_hostile_json_depth_is_a_non_match_and_later_record_still_scans():
    hostile_array = '{"payload":' + "[" * 1_200 + '"x"' + "]" * 1_200 + "}"
    hostile_object = '{"payload":' * 1_200 + '"x"' + "}" * 1_200
    valid_record = {"key": "later"}
    serialized = json.dumps(
        valid_record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    for hostile in (hostile_array, hostile_object):
        assert list(_canonical_json_objects(hostile + "\n" + serialized))[-1] == serialized

    malformed_string = '"unterminated { [ \\" brace noise\n'
    assert list(_canonical_json_objects(malformed_string + serialized))[-1] == serialized


def _structured_adversarial_record(example: BenchmarkExample) -> str:
    reordered = {
        key: unicodedata.normalize("NFD", value) if isinstance(value, str) else value
        for key, value in reversed(list(example.record.items()))
    }
    return json.dumps(reordered, ensure_ascii=True, indent=2)


def _metadata_enriched_adversarial_record(example: BenchmarkExample) -> str:
    reordered = {
        key: unicodedata.normalize("NFD", value) if isinstance(value, str) else value
        for key, value in reversed(list(example.record.items()))
    }
    return json.dumps(
        {**reordered, "source": {"dataset": "benchmark-import", "split": "train"}},
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )


def _decoded_string_budget_document(example: BenchmarkExample, *, prefixes: int) -> str:
    return ('{"n":"x"}\n' * prefixes) + _structured_adversarial_record(example)


def _deep_serialized_wrapper(target: str, wrapper_name: str, *, depth: int = 40) -> str:
    if wrapper_name == "object":
        return '{"wrapper":' * depth + target + "}" * depth
    serialized = json.dumps(target, ensure_ascii=True)
    if wrapper_name == "array_serialized":
        return "[" * depth + serialized + "]" * depth
    if wrapper_name == "mixed_serialized":
        return '[{"wrapper":' * depth + serialized + "}]" * depth
    raise ValueError(f"unknown deep serialized wrapper: {wrapper_name}")


def test_selected_record_survives_overdepth_object_array_and_mixed_wrappers(tmp_path: Path):
    registry, fingerprint = _registry(tmp_path)
    suite = load_suite(
        registry,
        expected_fingerprint=fingerprint,
        access="dev",
        cache=BoundedShardCache(tmp_path / "benchmark-cache", max_size_bytes=10_000_000),
        timeout_seconds=1.0,
    )
    probe_index = _build_probe_index(suite)
    example = next(task for task in suite.tasks if task.name == "jcommonsenseqa").examples[0]
    target = _structured_adversarial_record(example)
    wrappers = {
        "object": '{"wrapper":' * 40 + target + "}" * 40,
        "array": "[" * 40 + target + "]" * 40,
        "mixed": '{"wrapper":[' * 40 + target + "]}" * 40,
        "array_serialized": _deep_serialized_wrapper(target, "array_serialized"),
        "mixed_serialized": _deep_serialized_wrapper(target, "mixed_serialized"),
    }

    for wrapper_name, text in wrappers.items():
        matches = _document_matches(
            text,
            source_name="fixture_train",
            document_id=f"deep-{wrapper_name}",
            upstream_id=None,
            probe_index=probe_index,
        )
        assert any(
            match["benchmark_example_id"] == example.example_id
            and match["benchmark_field"] == "canonical_record"
            and match["match_type"] == "structured_json"
            for match in matches
        )


def test_metadata_enriched_record_survives_overdepth_object_wrapper(tmp_path: Path):
    registry, fingerprint = _registry(tmp_path)
    suite = load_suite(
        registry,
        expected_fingerprint=fingerprint,
        access="dev",
        cache=BoundedShardCache(tmp_path / "benchmark-cache", max_size_bytes=10_000_000),
        timeout_seconds=1.0,
    )
    probe_index = _build_probe_index(suite)
    example = next(task for task in suite.tasks if task.name == "jcommonsenseqa").examples[0]
    target = _metadata_enriched_adversarial_record(example)
    text = '{"wrapper":' * 40 + target + "}" * 40

    matches = _document_matches(
        text,
        source_name="fixture_train",
        document_id="deep-metadata-enriched",
        upstream_id=None,
        probe_index=probe_index,
    )

    assert any(
        match["benchmark_example_id"] == example.example_id
        and match["benchmark_field"] == "canonical_record"
        and match["match_type"] == "structured_json"
        for match in matches
    )


@pytest.mark.parametrize("wrapper_name", ["object", "array_serialized", "mixed_serialized"])
def test_full_scan_cannot_cache_clean_evidence_for_overdepth_wrapped_record(
    monkeypatch,
    tmp_path: Path,
    wrapper_name: str,
):
    registry, fingerprint = _registry(tmp_path)
    _, checkpoint_config = _pretraining_checkpoint(tmp_path)
    suite = load_suite(
        registry,
        expected_fingerprint=fingerprint,
        access="dev",
        cache=BoundedShardCache(tmp_path / "benchmark-cache", max_size_bytes=10_000_000),
        timeout_seconds=1.0,
    )
    example = next(task for task in suite.tasks if task.name == "jcommonsenseqa").examples[0]
    target = _structured_adversarial_record(example)
    text = _deep_serialized_wrapper(target, wrapper_name)
    document_id = "e" * 64

    monkeypatch.setattr(
        contamination_scans,
        "ManifestTextSource",
        lambda *_args, **_kwargs: iter(
            [
                RawDocument(
                    text=text,
                    metadata={
                        "document_id": document_id,
                        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "upstream_id": wrapper_name,
                    },
                )
            ]
        ),
    )
    evidence = scan_checkpoint_training_data(
        checkpoint_config,
        suite,
        fallback_cache=BoundedShardCache(tmp_path / "fallback", max_size_bytes=10_000_000),
    )

    assert evidence["scan_complete"] is True
    assert evidence["contaminated"] is True
    assert any(match["training_document_id"] == document_id for match in evidence["matches"])
    index_files = list((tmp_path / "training-cache/contamination-scans").glob("*.json"))
    assert len(index_files) == 1
    cached = json.loads(index_files[0].read_text(encoding="utf-8"))
    assert cached["report"]["contaminated"] is True
    assert cached["report"]["matches"]


def test_full_scan_detects_serialized_record_inside_malformed_balanced_object(
    monkeypatch,
    tmp_path: Path,
):
    registry, fingerprint = _registry(tmp_path)
    _, checkpoint_config = _pretraining_checkpoint(tmp_path)
    suite = load_suite(
        registry,
        expected_fingerprint=fingerprint,
        access="dev",
        cache=BoundedShardCache(tmp_path / "benchmark-cache", max_size_bytes=10_000_000),
        timeout_seconds=1.0,
    )
    example = next(task for task in suite.tasks if task.name == "jcommonsenseqa").examples[0]
    serialized_record = json.dumps(_structured_adversarial_record(example), ensure_ascii=True)
    text = '{"payload":' + serialized_record + ", trailing}"
    document_id = "f" * 64

    monkeypatch.setattr(
        contamination_scans,
        "ManifestTextSource",
        lambda *_args, **_kwargs: iter(
            [
                RawDocument(
                    text=text,
                    metadata={
                        "document_id": document_id,
                        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "upstream_id": "malformed-balanced-wrapper",
                    },
                )
            ]
        ),
    )

    evidence = scan_checkpoint_training_data(
        checkpoint_config,
        suite,
        fallback_cache=BoundedShardCache(tmp_path / "fallback", max_size_bytes=10_000_000),
    )

    assert evidence["scan_complete"] is True
    assert evidence["contaminated"] is True
    assert any(match["training_document_id"] == document_id for match in evidence["matches"])
    index_files = list((tmp_path / "training-cache/contamination-scans").glob("*.json"))
    assert len(index_files) == 1
    cached = json.loads(index_files[0].read_text(encoding="utf-8"))
    assert cached["report"]["contaminated"] is True
    assert cached["report"]["matches"]


def test_full_scan_detects_serialized_record_used_as_valid_json_object_key(
    monkeypatch,
    tmp_path: Path,
):
    registry, fingerprint = _registry(tmp_path)
    _, checkpoint_config = _pretraining_checkpoint(tmp_path)
    suite = load_suite(
        registry,
        expected_fingerprint=fingerprint,
        access="dev",
        cache=BoundedShardCache(tmp_path / "benchmark-cache", max_size_bytes=10_000_000),
        timeout_seconds=1.0,
    )
    example = next(task for task in suite.tasks if task.name == "jcommonsenseqa").examples[0]
    serialized_record = _structured_adversarial_record(example)
    text = json.dumps({serialized_record: 0}, ensure_ascii=True)
    document_id = "a" * 64

    monkeypatch.setattr(
        contamination_scans,
        "ManifestTextSource",
        lambda *_args, **_kwargs: iter(
            [
                RawDocument(
                    text=text,
                    metadata={
                        "document_id": document_id,
                        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "upstream_id": "valid-object-key-wrapper",
                    },
                )
            ]
        ),
    )

    evidence = scan_checkpoint_training_data(
        checkpoint_config,
        suite,
        fallback_cache=BoundedShardCache(tmp_path / "fallback", max_size_bytes=10_000_000),
    )

    assert evidence["scan_complete"] is True
    assert evidence["contaminated"] is True
    assert any(match["training_document_id"] == document_id for match in evidence["matches"])
    index_files = list((tmp_path / "training-cache/contamination-scans").glob("*.json"))
    assert len(index_files) == 1
    cached = json.loads(index_files[0].read_text(encoding="utf-8"))
    assert cached["report"]["contaminated"] is True
    assert cached["report"]["matches"]


@pytest.mark.parametrize(
    "wrapper_name",
    [
        "unterminated_quote",
        "escaped_unterminated_quote",
        "escaped_unterminated_quote_inside_object",
        "newline_unterminated_quote",
        "newline_escaped_unterminated_quote",
        "recursive_newline_escaped_unterminated_quote",
        "recursive_escaped_unterminated_quote_inside_object",
        "recursive_escaped_unterminated_quote",
    ],
)
def test_full_scan_detects_reordered_record_after_unterminated_quote(
    monkeypatch,
    tmp_path: Path,
    wrapper_name: str,
):
    registry, fingerprint = _registry(tmp_path)
    _, checkpoint_config = _pretraining_checkpoint(tmp_path)
    suite = load_suite(
        registry,
        expected_fingerprint=fingerprint,
        access="dev",
        cache=BoundedShardCache(tmp_path / "benchmark-cache", max_size_bytes=10_000_000),
        timeout_seconds=1.0,
    )
    example = next(task for task in suite.tasks if task.name == "jcommonsenseqa").examples[0]
    target = _structured_adversarial_record(example)
    if "inside_object" in wrapper_name or "newline" in wrapper_name:
        input_only = {
            key: unicodedata.normalize("NFD", value) if isinstance(value, str) else value
            for key, value in reversed(list(example.record.items()))
            if key not in {"label", "q_id"}
        }
        target = json.dumps(input_only, ensure_ascii=True, indent=2)
    prefix = "\\" + '"' if "escaped" in wrapper_name else '"'
    if "inside_object" in wrapper_name:
        text = f'{{"meta":"unterminated {prefix} prefix {target} trailing}}'
    elif "newline" in wrapper_name:
        text = f"ordinary malformed\n{prefix} prefix {target} trailing"
    else:
        text = f"ordinary malformed {prefix} prefix {target} trailing"
    if wrapper_name.startswith("recursive"):
        text = json.dumps({"payload": text}, ensure_ascii=True)
    document_id = "b" * 64

    monkeypatch.setattr(
        contamination_scans,
        "ManifestTextSource",
        lambda *_args, **_kwargs: iter(
            [
                RawDocument(
                    text=text,
                    metadata={
                        "document_id": document_id,
                        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "upstream_id": wrapper_name,
                    },
                )
            ]
        ),
    )

    evidence = scan_checkpoint_training_data(
        checkpoint_config,
        suite,
        fallback_cache=BoundedShardCache(tmp_path / "fallback", max_size_bytes=10_000_000),
    )

    assert evidence["scan_complete"] is True
    assert evidence["contaminated"] is True
    assert any(match["training_document_id"] == document_id for match in evidence["matches"])
    index_files = list((tmp_path / "training-cache/contamination-scans").glob("*.json"))
    assert len(index_files) == 1
    cached = json.loads(index_files[0].read_text(encoding="utf-8"))
    assert cached["report"]["contaminated"] is True
    assert cached["report"]["matches"]


def test_full_scan_detects_metadata_enriched_selected_record(
    monkeypatch,
    tmp_path: Path,
):
    registry, fingerprint = _registry(tmp_path)
    _, checkpoint_config = _pretraining_checkpoint(tmp_path)
    suite = load_suite(
        registry,
        expected_fingerprint=fingerprint,
        access="dev",
        cache=BoundedShardCache(tmp_path / "benchmark-cache", max_size_bytes=10_000_000),
        timeout_seconds=1.0,
    )
    example = next(task for task in suite.tasks if task.name == "jcommonsenseqa").examples[0]
    enriched = {
        **{
            key: unicodedata.normalize("NFD", value) if isinstance(value, str) else value
            for key, value in reversed(list(example.record.items()))
        },
        "source": {"dataset": "benchmark-import", "split": "train"},
    }
    text = json.dumps(enriched, ensure_ascii=True, indent=2, sort_keys=True)
    document_id = "c" * 64

    monkeypatch.setattr(
        contamination_scans,
        "ManifestTextSource",
        lambda *_args, **_kwargs: iter(
            [
                RawDocument(
                    text=text,
                    metadata={
                        "document_id": document_id,
                        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "upstream_id": "metadata-enriched-record",
                    },
                )
            ]
        ),
    )

    evidence = scan_checkpoint_training_data(
        checkpoint_config,
        suite,
        fallback_cache=BoundedShardCache(tmp_path / "fallback", max_size_bytes=10_000_000),
    )

    assert evidence["scan_complete"] is True
    assert evidence["contaminated"] is True
    assert any(
        match["training_document_id"] == document_id
        and match["benchmark_example_id"] == example.example_id
        and match["benchmark_field"] == "canonical_record"
        and match["match_type"] == "structured_json"
        for match in evidence["matches"]
    )
    index_files = list((tmp_path / "training-cache/contamination-scans").glob("*.json"))
    assert len(index_files) == 1
    cached = json.loads(index_files[0].read_text(encoding="utf-8"))
    assert cached["report"]["contaminated"] is True
    assert cached["report"]["matches"]


@pytest.mark.parametrize("duplicate_keys", [False, True])
def test_full_scan_cannot_cache_clean_evidence_for_unlabeled_selected_input(
    monkeypatch,
    tmp_path: Path,
    duplicate_keys: bool,
):
    registry, fingerprint = _registry(tmp_path)
    _, checkpoint_config = _pretraining_checkpoint(tmp_path)
    suite = load_suite(
        registry,
        expected_fingerprint=fingerprint,
        access="dev",
        cache=BoundedShardCache(tmp_path / "benchmark-cache", max_size_bytes=10_000_000),
        timeout_seconds=1.0,
    )
    example = next(task for task in suite.tasks if task.name == "jcommonsenseqa").examples[0]
    unlabeled = {
        key: unicodedata.normalize("NFD", value) if isinstance(value, str) else value
        for key, value in reversed(list(example.record.items()))
        if key not in {"label", "q_id"}
    }
    if duplicate_keys:
        text = (
            "{"
            + ",".join(
                [
                    *(
                        f"{json.dumps(key)}:{json.dumps(value, ensure_ascii=True)}"
                        for key, value in unlabeled.items()
                    ),
                    *(f"{json.dumps(key)}:{json.dumps(f'benign-{key}')}" for key in unlabeled),
                ]
            )
            + "}"
        )
    else:
        text = json.dumps(unlabeled, ensure_ascii=True, indent=2, sort_keys=True)
    document_id = "d" * 64

    monkeypatch.setattr(
        contamination_scans,
        "ManifestTextSource",
        lambda *_args, **_kwargs: iter(
            [
                RawDocument(
                    text=text,
                    metadata={
                        "document_id": document_id,
                        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "upstream_id": (
                            "duplicate-key-benchmark-input"
                            if duplicate_keys
                            else "unlabeled-benchmark-input"
                        ),
                    },
                )
            ]
        ),
    )

    evidence = scan_checkpoint_training_data(
        checkpoint_config,
        suite,
        fallback_cache=BoundedShardCache(tmp_path / "fallback", max_size_bytes=10_000_000),
    )

    assert evidence["scan_complete"] is True
    assert evidence["contaminated"] is True
    assert any(
        match["training_document_id"] == document_id
        and match["benchmark_example_id"] == example.example_id
        and match["benchmark_field"] == "canonical_record"
        and match["match_type"] == "structured_json"
        for match in evidence["matches"]
    )
    index_files = list((tmp_path / "training-cache/contamination-scans").glob("*.json"))
    assert len(index_files) == 1
    cached = json.loads(index_files[0].read_text(encoding="utf-8"))
    assert cached["report"]["contaminated"] is True
    assert cached["report"]["matches"]


def test_many_leaf_json_strings_still_exhaust_the_shared_document_budget():
    text = "[" + ",".join(json.dumps("x") for _ in range(8_193)) + "]"

    with pytest.raises(_JsonTraversalLimitExceeded, match="decoded_strings"):
        list(_canonical_json_objects(text))


def test_duplicate_key_projection_combinations_fail_closed_when_unbounded(tmp_path: Path):
    registry, fingerprint = _registry(tmp_path)
    suite = load_suite(
        registry,
        expected_fingerprint=fingerprint,
        access="dev",
        cache=BoundedShardCache(tmp_path / "benchmark-cache", max_size_bytes=10_000_000),
        timeout_seconds=1.0,
    )
    probe_index = _build_probe_index(suite)
    text = "{" + ",".join(f'"question":"candidate-{index}"' for index in range(32_769)) + "}"

    with pytest.raises(ContaminationScanError, match="nodes limit exhausted"):
        _document_matches(
            text,
            source_name="fixture_train",
            document_id="duplicate-key-budget",
            upstream_id=None,
            probe_index=probe_index,
        )


def test_default_decoded_string_budget_exhaustion_fails_closed(tmp_path: Path):
    registry, fingerprint = _registry(tmp_path)
    suite = load_suite(
        registry,
        expected_fingerprint=fingerprint,
        access="dev",
        cache=BoundedShardCache(tmp_path / "benchmark-cache", max_size_bytes=10_000_000),
        timeout_seconds=1.0,
    )
    probe_index = _build_probe_index(suite)
    example = next(task for task in suite.tasks if task.name == "jcommonsenseqa").examples[0]
    exhausted = _decoded_string_budget_document(example, prefixes=4_095)

    with pytest.raises(
        ContaminationScanError,
        match="decoded_strings limit exhausted.*fixture_train.*budget-document",
    ):
        _document_matches(
            exhausted,
            source_name="fixture_train",
            document_id="budget-document",
            upstream_id=None,
            probe_index=probe_index,
        )

    bounded = _document_matches(
        _decoded_string_budget_document(example, prefixes=4_000),
        source_name="fixture_train",
        document_id="bounded-document",
        upstream_id=None,
        probe_index=probe_index,
    )
    assert any(
        match["benchmark_example_id"] == example.example_id
        and match["benchmark_field"] == "canonical_record"
        and match["match_type"] == "structured_json"
        for match in bounded
    )


def test_scan_budget_exhaustion_writes_no_complete_or_stale_index(
    monkeypatch,
    tmp_path: Path,
):
    registry, fingerprint = _registry(tmp_path)
    _, checkpoint_config = _pretraining_checkpoint(tmp_path)
    suite = load_suite(
        registry,
        expected_fingerprint=fingerprint,
        access="dev",
        cache=BoundedShardCache(tmp_path / "benchmark-cache", max_size_bytes=10_000_000),
        timeout_seconds=1.0,
    )
    example = next(task for task in suite.tasks if task.name == "jcommonsenseqa").examples[0]
    documents = [_decoded_string_budget_document(example, prefixes=4_095)]

    def manifest_source(*_args, **_kwargs):
        text = documents[0]
        return iter(
            [
                RawDocument(
                    text=text,
                    metadata={
                        "document_id": "f" * 64,
                        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "upstream_id": "budget-upstream",
                    },
                )
            ]
        )

    monkeypatch.setattr(contamination_scans, "ManifestTextSource", manifest_source)
    fallback = BoundedShardCache(tmp_path / "fallback", max_size_bytes=10_000_000)
    with pytest.raises(
        ContaminationScanError,
        match=f"decoded_strings limit exhausted.*fixture_train.*{'f' * 64}",
    ):
        scan_checkpoint_training_data(
            checkpoint_config,
            suite,
            fallback_cache=fallback,
        )

    index_dir = tmp_path / "training-cache/contamination-scans"
    assert list(index_dir.glob("*.json")) == []

    documents[0] = _decoded_string_budget_document(example, prefixes=4_000)
    evidence = scan_checkpoint_training_data(
        checkpoint_config,
        suite,
        fallback_cache=fallback,
    )
    assert evidence["scan_complete"] is True
    assert evidence["contaminated"] is True
    assert list(index_dir.glob("*.json"))


def test_shingle_matcher_is_linear_and_does_not_materialize_corpus_windows():
    reference = {
        "task": "fixture",
        "benchmark_example_id": "example",
        "benchmark_field": "question",
    }
    first = "a" * (SHINGLE_CODEPOINTS - 1) + "b"
    second = "b" + "a" * (SHINGLE_CODEPOINTS - 1)
    matcher = _ShingleMatcher({first: [reference], second: [reference]})
    corpus = "z" * 500_000 + first + "z" * 500_000
    operations: list[int] = []
    verifications: list[int] = []

    matches = matcher.references_in(
        corpus,
        operation_counter=operations,
        verification_counter=verifications,
    )

    assert matches == [reference]
    assert operations == [len(corpus)]
    assert verifications == [1]
    assert matcher.pattern_count == 2
    assert matcher.stored_codepoints == len(first) + len(second)


def test_shingle_matcher_deduplicates_reference_identity_within_one_document():
    shared = {
        "task": "jcommonsenseqa",
        "benchmark_example_id": "shared",
        "benchmark_field": "question",
    }
    distinct = {
        "task": "jcommonsenseqa",
        "benchmark_example_id": "distinct",
        "benchmark_field": "choice0",
    }
    document = "".join(chr(0x4E00 + index) for index in range(SHINGLE_CODEPOINTS + 100))
    patterns = {document[index : index + SHINGLE_CODEPOINTS]: [shared] for index in range(101)}
    patterns[document[:SHINGLE_CODEPOINTS]] = [shared, distinct]

    assert _ShingleMatcher(patterns).references_in(document) == [shared, distinct]


def test_runner_retains_blocked_evidence_and_never_scores_contamination(
    monkeypatch, tmp_path: Path
):
    registry, fingerprint = _registry(tmp_path, question="小さなモデルを一から学習する。")
    checkpoint, _ = _pretraining_checkpoint(tmp_path)
    config = _benchmark_config(tmp_path, checkpoint)
    monkeypatch.setattr(
        "benchmarks.runner.score_suite",
        lambda *_args, **_kwargs: pytest.fail("contaminated benchmark must not score"),
    )

    with pytest.raises(BenchmarkContaminationError) as captured:
        run_benchmark(
            config,
            registry_path=registry,
            expected_registry_fingerprint=fingerprint,
        )

    report = json.loads(captured.value.output_path.read_text(encoding="utf-8"))
    assert report["status"] == "blocked_contamination"
    assert report["tasks"] == {}
    assert report["contamination"]["scan_complete"] is True
    assert report["contamination"]["matches"]


def test_runner_rejects_short_context_before_training_corpus_scan(monkeypatch, tmp_path: Path):
    registry, fingerprint = _registry(tmp_path)
    checkpoint, _ = _pretraining_checkpoint(tmp_path, sequence_length=8)
    config = _benchmark_config(tmp_path, checkpoint)
    monkeypatch.setattr(
        "benchmarks.runner.scan_checkpoint_training_data",
        lambda *_args, **_kwargs: pytest.fail(
            "context incompatibility must fail before the complete contamination scan"
        ),
    )

    with pytest.raises(BenchmarkScoringError, match="checkpoint context is incompatible"):
        run_benchmark(
            config,
            registry_path=registry,
            expected_registry_fingerprint=fingerprint,
        )


def test_runner_rejects_unsupported_cuda_bf16_before_training_scan(monkeypatch, tmp_path: Path):
    registry, fingerprint = _registry(tmp_path)
    checkpoint, _ = _pretraining_checkpoint(tmp_path, precision="bf16")
    config = _benchmark_config(tmp_path, checkpoint)
    config.benchmark.device = "cuda"
    monkeypatch.setattr("runtime.config.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("runtime.config.torch.cuda.is_bf16_supported", lambda: False)
    monkeypatch.setattr(
        "benchmarks.runner.scan_checkpoint_training_data",
        lambda *_args, **_kwargs: pytest.fail(
            "unsupported CUDA BF16 must fail before the complete contamination scan"
        ),
    )

    with pytest.raises(ConfigPreflightError, match="does not support BF16"):
        run_benchmark(
            config,
            registry_path=registry,
            expected_registry_fingerprint=fingerprint,
        )


def test_fixed_benchmark_determinism_policy_is_applied_and_observable(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    policy = _apply_evaluation_determinism_policy()

    assert policy == {
        "revision": "strict-math-sdpa-v1",
        "seed": 0,
        "deterministic_algorithms": {"enabled": True, "warn_only": False},
        "cublas_workspace_config": ":4096:8",
        "scaled_dot_product_attention": {
            "math": True,
            "flash": False,
            "memory_efficient": False,
            "cudnn": False,
        },
        "cudnn": {"deterministic": True, "benchmark": False},
        "float32": {
            "matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
    }
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cuda.math_sdp_enabled()
    assert not torch.backends.cuda.flash_sdp_enabled()
    assert not torch.backends.cuda.mem_efficient_sdp_enabled()
    assert not torch.backends.cuda.cudnn_sdp_enabled()


def test_benchmark_determinism_fails_if_cuda_was_initialized_under_another_policy(
    monkeypatch,
):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    with pytest.raises(RuntimeError, match="before CUDA initialization"):
        _apply_evaluation_determinism_policy()


def test_runner_applies_determinism_before_loading_checkpoint(monkeypatch, tmp_path: Path):
    checkpoint, _ = _pretraining_checkpoint(tmp_path)
    config = _benchmark_config(tmp_path, checkpoint)
    policy_applied = False

    def tracked_policy():
        nonlocal policy_applied
        policy_applied = True
        return _apply_evaluation_determinism_policy()

    def blocked_load(_path: Path):
        assert policy_applied
        raise RuntimeError("checkpoint load observed deterministic policy")

    monkeypatch.setattr("benchmarks.runner._apply_evaluation_determinism_policy", tracked_policy)
    monkeypatch.setattr("benchmarks.runner.load_checkpoint_for_generation", blocked_load)

    with pytest.raises(RuntimeError, match="observed deterministic policy"):
        run_benchmark(config)


def test_runner_rejects_sibling_checkpoint_output_before_suite_loading(monkeypatch, tmp_path: Path):
    checkpoint, _ = _pretraining_checkpoint(tmp_path)
    milestone = checkpoint.with_name("milestone-step-000000000004.pt")
    shutil.copy2(checkpoint, milestone)
    checkpoint_bytes = checkpoint.read_bytes()
    milestone_bytes = milestone.read_bytes()
    config = _benchmark_config(tmp_path, milestone)
    config.benchmark.output_path = str(checkpoint)
    monkeypatch.setattr(
        "benchmarks.runner.load_suite",
        lambda *_args, **_kwargs: pytest.fail(
            "checkpoint namespace collision must fail before suite loading"
        ),
    )

    with pytest.raises(ValueError, match="outside checkpoint namespaces"):
        run_benchmark(config)

    assert checkpoint.read_bytes() == checkpoint_bytes
    assert milestone.read_bytes() == milestone_bytes


def test_runner_rejects_checkpoint_symlink_and_hardlink_aliases(tmp_path: Path):
    checkpoint, _ = _pretraining_checkpoint(tmp_path)
    checkpoint_bytes = checkpoint.read_bytes()
    alias_directory = tmp_path / "checkpoint-alias"
    alias_directory.symlink_to(checkpoint.parent, target_is_directory=True)
    symlink_config = _benchmark_config(tmp_path, checkpoint)
    symlink_config.benchmark.output_path = str(alias_directory / "result.json")

    with pytest.raises(ValueError, match="outside checkpoint namespaces"):
        run_benchmark(symlink_config)

    hardlink = tmp_path / "checkpoint-hardlink.json"
    hardlink.hardlink_to(checkpoint)
    hardlink_config = _benchmark_config(tmp_path, checkpoint)
    hardlink_config.benchmark.output_path = str(hardlink)
    with pytest.raises(ValueError, match="must not be a hardlink"):
        run_benchmark(hardlink_config)

    assert checkpoint.read_bytes() == checkpoint_bytes


def test_runner_accepts_output_beside_repository_when_checkpoint_parent_is_broad(
    monkeypatch, tmp_path: Path
):
    checkpoint, _ = _pretraining_checkpoint(tmp_path)
    copied_checkpoint = tmp_path / "milestone.pt"
    shutil.copy2(checkpoint, copied_checkpoint)
    output = tmp_path / "repository" / "runs" / "benchmark.json"
    output.parent.mkdir(parents=True)
    config = _benchmark_config(tmp_path, copied_checkpoint)
    config.benchmark.output_root = str(output.parent)
    config.benchmark.output_path = output.name
    monkeypatch.setattr(
        "benchmarks.runner.load_suite",
        lambda *_args, **_kwargs: pytest.fail(
            "valid output path must reach suite loading after checkpoint preflight"
        ),
    )

    with pytest.raises(pytest.fail.Exception, match="valid output path"):
        run_benchmark(config)


def test_runner_rejects_repository_input_and_output_root_escape_before_suite_loading(
    monkeypatch, tmp_path: Path
):
    checkpoint, _ = _pretraining_checkpoint(tmp_path)
    registry = ROOT / "data/benchmarks/suite-v1.json"
    registry_bytes = registry.read_bytes()
    config = _benchmark_config(tmp_path, checkpoint)
    config.benchmark.output_path = str(registry)
    monkeypatch.setattr(
        "benchmarks.runner.load_suite",
        lambda *_args, **_kwargs: pytest.fail("unsafe output must fail before suite loading"),
    )

    with pytest.raises(ValueError, match="must not overwrite an existing file"):
        run_benchmark(config)
    assert registry.read_bytes() == registry_bytes

    config.benchmark.output_path = str(tmp_path / "outside" / "new.json")
    with pytest.raises(ValueError, match="inside its configured output root"):
        run_benchmark(config)

    config.benchmark.output_root = str(ROOT / "data/benchmark-results")
    config.benchmark.output_path = "new.json"
    with pytest.raises(ValueError, match="repository-local benchmark output roots"):
        run_benchmark(config)


def test_atomic_benchmark_result_publish_never_replaces_existing_output(tmp_path: Path):
    output = tmp_path / "benchmark-results/result.json"
    _write_json_atomic(output, {"result": "first"})
    first_bytes = output.read_bytes()

    with pytest.raises(ValueError, match="will not be replaced"):
        _write_json_atomic(output, {"result": "second"})
    assert output.read_bytes() == first_bytes


def test_runner_releases_loaded_checkpoint_before_suite_and_training_scan(
    monkeypatch, tmp_path: Path
):
    checkpoint, _ = _pretraining_checkpoint(tmp_path)
    config = _benchmark_config(tmp_path, checkpoint)
    loaded_reference: weakref.ReferenceType[Any] | None = None
    optimizer_tensor_reference: weakref.ReferenceType[torch.Tensor] | None = None

    def tracked_load(path: Path):
        nonlocal loaded_reference, optimizer_tensor_reference
        loaded = load_checkpoint_for_generation(path)
        optimizer_tensor = torch.ones(1)
        loaded.payload["state"]["optimizer"] = {"retained_probe": optimizer_tensor}
        loaded_reference = weakref.ref(loaded)
        optimizer_tensor_reference = weakref.ref(optimizer_tensor)
        return loaded

    def assert_released(*_args, **_kwargs):
        gc.collect()
        assert loaded_reference is not None
        assert optimizer_tensor_reference is not None
        assert loaded_reference() is None
        assert optimizer_tensor_reference() is None
        raise RuntimeError("checkpoint payload released")

    monkeypatch.setattr("benchmarks.runner.load_checkpoint_for_generation", tracked_load)
    monkeypatch.setattr("benchmarks.runner.load_suite", assert_released)

    with pytest.raises(RuntimeError, match="checkpoint payload released"):
        run_benchmark(config)


def test_fixture_checkpoint_scoring_and_result_identity_are_deterministic(
    monkeypatch, tmp_path: Path
):
    registry, fingerprint = _registry(tmp_path)
    checkpoint, _ = _pretraining_checkpoint(tmp_path)
    suite = load_suite(
        registry,
        expected_fingerprint=fingerprint,
        access="dev",
        cache=BoundedShardCache(tmp_path / "suite-cache", max_size_bytes=10_000_000),
        timeout_seconds=1.0,
    )
    sampler = CheckpointSampler.from_checkpoint(checkpoint, device="cpu")

    first_scores = score_suite(sampler, suite)
    second_scores = score_suite(sampler, suite)
    assert first_scores == second_scores
    assert first_scores["jcommonsenseqa"] == {
        "primary_metric": "length_normalized_accuracy",
        "correct": 1,
        "total": 1,
        "length_normalized_accuracy": 1.0,
        "raw_log_probability_accuracy": 1.0,
        "prediction_trace": [
            {
                "example_id": "1",
                "prediction": 0,
                "raw_prediction": 0,
                "correct": True,
                "score_sha256": (
                    "e9490f58d16f2462a9668afc8b364b7310feee2849d971757910ccc4b623cf2f"
                ),
            }
        ],
        "prediction_trace_sha256": (
            "da8e6d9f41fced7c5aaff8df454c238d9f5e5b9f1512a3a21d84909c4d364529"
        ),
    }
    assert first_scores["gsm8k"] == {
        "primary_metric": "exact_match",
        "correct": 0,
        "total": 1,
        "exact_match": 0.0,
        "valid_answer_format": 0,
        "valid_answer_format_rate": 0.0,
        "prediction_trace": [
            {
                "example_id": "0",
                "correct": False,
                "valid_answer_format": False,
                "generated_token_count": 128,
                "generated_token_ids_hash_revision": "canonical-json-token-ids-sha256-v1",
                "generated_token_ids_sha256": (
                    "1b912946d3fd584649a7924a6d699d2bd0ee0da2ebc320bd53135fa9b1ec21a0"
                ),
                "stop_reason": "max_new_tokens",
                "completion_sha256": (
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                ),
            }
        ],
        "prediction_trace_sha256": (
            "f2ade7738dfbfb6af7bb08e549d8dd7fe64d59df50423e700cde2255961a69a0"
        ),
    }

    monkeypatch.setattr("benchmarks.runner.score_suite", lambda *_args: copy.deepcopy(first_scores))
    first_config = _benchmark_config(tmp_path, checkpoint, "first.json")
    second_config = _benchmark_config(tmp_path, checkpoint, "second.json")
    first_path = run_benchmark(
        first_config,
        registry_path=registry,
        expected_registry_fingerprint=fingerprint,
    )
    second_path = run_benchmark(
        second_config,
        registry_path=registry,
        expected_registry_fingerprint=fingerprint,
    )
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    assert first == second
    assert first["evaluation_identity_sha256"] == second["evaluation_identity_sha256"]
    assert first["evaluation_identity"]["determinism_policy"] == {
        "revision": "strict-math-sdpa-v1",
        "seed": 0,
        "deterministic_algorithms": {"enabled": True, "warn_only": False},
        "cublas_workspace_config": ":4096:8",
        "scaled_dot_product_attention": {
            "math": True,
            "flash": False,
            "memory_efficient": False,
            "cudnn": False,
        },
        "cudnn": {"deterministic": True, "benchmark": False},
        "float32": {
            "matmul_precision": "highest",
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
    }
    assert set(first["evaluation_identity"]["evaluator"]) == {
        "git",
        "lock_sha256",
        "environment",
    }
    assert len(first["evaluation_identity"]["evaluator"]["git"]["sha"]) == 40
    serialized = json.dumps(first, ensure_ascii=False).lower()
    assert '"prompt"' not in serialized
    assert '"completion"' not in serialized


def test_runner_derives_exclusive_identity_bound_default_result_path(monkeypatch, tmp_path: Path):
    registry, fingerprint = _registry(tmp_path)
    checkpoint, _ = _pretraining_checkpoint(tmp_path)
    config = _benchmark_config(tmp_path, checkpoint)
    config.benchmark.output_path = None
    monkeypatch.setattr("benchmarks.runner.score_suite", lambda *_args: {})

    output_path = run_benchmark(
        config,
        registry_path=registry,
        expected_registry_fingerprint=fingerprint,
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.parent == Path(str(config.benchmark.output_root))
    assert output_path.name == f"dev-{result['evaluation_identity_sha256']}.json"
    assert result["evaluation_identity"]["suite"]["access"] == "dev"
    assert len(result["evaluation_identity"]["checkpoint"]["physical"]["sha256"]) == 64
    monkeypatch.setattr(
        "benchmarks.runner.scan_checkpoint_training_data",
        lambda *_args, **_kwargs: pytest.fail(
            "an existing identity-bound output must fail before a repeated training scan"
        ),
    )
    with pytest.raises(ValueError, match="must not overwrite an existing file"):
        run_benchmark(
            config,
            registry_path=registry,
            expected_registry_fingerprint=fingerprint,
        )


def test_jcommonsenseqa_scores_exact_joint_tokenization(monkeypatch, tmp_path: Path):
    checkpoint, _ = _pretraining_checkpoint(tmp_path)
    sampler = CheckpointSampler.from_checkpoint(checkpoint, device="cpu")
    prompt = "質問: 生き物を一つ選んでください。\n答え:\n"
    continuation = "動物"
    joint_ids, offsets = sampler.tokenizer.encode_with_offsets(prompt + continuation)
    separately_encoded = sampler.tokenizer.encode(prompt) + sampler.tokenizer.encode(continuation)
    assert joint_ids != separately_encoded
    continuation_start = next(
        index for index, (start, end) in enumerate(offsets) if start >= len(prompt) and end > start
    )
    observed_inputs: list[list[int]] = []
    original_forward = sampler.model.forward

    def capture_forward(input_ids: torch.Tensor, *args, **kwargs):
        observed_inputs.append(input_ids[0].tolist())
        return original_forward(input_ids, *args, **kwargs)

    monkeypatch.setattr(sampler.model, "forward", capture_forward)
    _, token_count = conditional_log_probability(
        sampler,
        prompt=prompt,
        continuation=continuation,
        precision="fp32",
    )

    assert observed_inputs == [joint_ids[:-1]]
    assert token_count == len(joint_ids[continuation_start:])


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_jcommonsenseqa_rejects_nonfinite_raw_logits(monkeypatch, tmp_path: Path, invalid: float):
    checkpoint, _ = _pretraining_checkpoint(tmp_path)
    sampler = CheckpointSampler.from_checkpoint(checkpoint, device="cpu")
    prompt = "質問: 生き物を一つ選んでください。\n答え:\n"
    continuation = "動物"
    joint_ids, offsets = sampler.tokenizer.encode_with_offsets(prompt + continuation)
    continuation_start = next(
        index for index, (start, _end) in enumerate(offsets) if start >= len(prompt)
    )
    continuation_ids = set(joint_ids[continuation_start:])
    non_target_id = next(
        token_id
        for token_id in range(sampler.tokenizer.vocab_size)
        if token_id not in continuation_ids
    )
    original_forward = sampler.model.forward

    def nonfinite_forward(input_ids: torch.Tensor, *args, **kwargs):
        logits = original_forward(input_ids, *args, **kwargs)
        logits[0, -1, non_target_id] = invalid
        return logits

    monkeypatch.setattr(sampler.model, "forward", nonfinite_forward)
    with pytest.raises(BenchmarkScoringError, match="non-finite logits"):
        conditional_log_probability(
            sampler,
            prompt=prompt,
            continuation=continuation,
            precision="fp32",
        )


def test_jcommonsenseqa_rejects_nonfinite_normalized_log_probabilities(monkeypatch, tmp_path: Path):
    checkpoint, _ = _pretraining_checkpoint(tmp_path)
    sampler = CheckpointSampler.from_checkpoint(checkpoint, device="cpu")
    prompt = "質問: 生き物を一つ選んでください。\n答え:\n"
    continuation = "動物"
    joint_ids, offsets = sampler.tokenizer.encode_with_offsets(prompt + continuation)
    continuation_start = next(
        index for index, (start, _end) in enumerate(offsets) if start >= len(prompt)
    )
    target_id = joint_ids[-1]
    continuation_ids = set(joint_ids[continuation_start:])
    non_target_id = next(
        token_id
        for token_id in range(sampler.tokenizer.vocab_size)
        if token_id not in continuation_ids
    )
    original_forward = sampler.model.forward

    def extreme_finite_forward(input_ids: torch.Tensor, *args, **kwargs):
        logits = original_forward(input_ids, *args, **kwargs)
        finite_max = torch.finfo(logits.dtype).max
        logits[0, -1, target_id] = -finite_max
        logits[0, -1, non_target_id] = finite_max
        return logits

    monkeypatch.setattr(sampler.model, "forward", extreme_finite_forward)
    with pytest.raises(BenchmarkScoringError, match="non-finite log probabilities"):
        conditional_log_probability(
            sampler,
            prompt=prompt,
            continuation=continuation,
            precision="fp32",
        )


def test_gsm8k_generation_receives_checkpoint_precision(monkeypatch, tmp_path: Path):
    registry, fingerprint = _registry(tmp_path)
    checkpoint, _ = _pretraining_checkpoint(tmp_path)
    suite = load_suite(
        registry,
        expected_fingerprint=fingerprint,
        access="dev",
        cache=BoundedShardCache(tmp_path / "suite-cache", max_size_bytes=10_000_000),
        timeout_seconds=1.0,
    )
    sampler = CheckpointSampler.from_checkpoint(checkpoint, device="cpu")
    original_generate = sampler.generate
    observed_precisions: list[str] = []

    def generate(prompt: str, *, max_new_tokens: int, precision: str):
        observed_precisions.append(precision)
        return original_generate(
            prompt,
            max_new_tokens=max_new_tokens,
            precision=precision,
        )

    monkeypatch.setattr(sampler, "generate", generate)
    score_suite(sampler, suite)

    assert observed_precisions == ["fp32"]


def test_external_baseline_record_is_aggregate_only_and_isolated(monkeypatch, tmp_path: Path):
    comparison_root = tmp_path / "outputs/external-comparisons"
    monkeypatch.setattr(external_records, "EXTERNAL_COMPARISON_ROOT", comparison_root)
    payload = {
        "subject": {
            "name": "small-open-baseline",
            "parameter_count": 1_000_000,
            "training_compute": "disclosed by upstream",
            "tokenizer": "upstream tokenizer",
            "context_length": 2048,
            "data_access": "upstream disclosure",
            "protocol_context_preflight": {
                "protocol_sha256": (
                    "d56ffdbdf0862929f40e51b1fe748b58826b8bb95532f11e1af4e8a9a7972377"
                ),
                "component_hashes": protocol_component_hashes(),
                "passed": True,
                "no_truncation": True,
                "required_context_length": 300,
                "task_required_context_lengths": {
                    "jcommonsenseqa": 120,
                    "gsm8k": 300,
                },
            },
        },
        "tasks": {
            "jcommonsenseqa": {
                "primary_metric": "length_normalized_accuracy",
                "value": 0.5,
                "correct": 64,
                "total": 128,
            },
            "gsm8k": {
                "primary_metric": "exact_match",
                "value": 0.0,
                "correct": 0,
                "total": 128,
            },
        },
    }
    output = write_external_comparison(
        payload,
        output_path="external.json",
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["kind"] == "external_baseline_comparison"
    assert record["isolation"]["repository_checkpoint_runner"] is False
    assert record["isolation"]["eligible_for_training_data_or_targets"] is False
    assert record["suite"]["access"] == "dev"
    assert record["suite"]["minimum_context_length"] == 129
    assert record["suite"]["protocol_sha256"] == (
        "d56ffdbdf0862929f40e51b1fe748b58826b8bb95532f11e1af4e8a9a7972377"
    )
    assert record["suite"]["component_hashes"] == protocol_component_hashes()
    assert record["suite"]["tasks"]["jcommonsenseqa"]["selected_examples_sha256"] == (
        "37e39dca6ce5108fe720dda6e0246f7c8ef858e22961229540d2023faeabe0bd"
    )
    with pytest.raises(ExternalComparisonError, match="must not overwrite an existing file"):
        write_external_comparison(payload, output_path="external.json")

    missing_name = copy.deepcopy(payload)
    missing_name["subject"]["name"] = " \t\n"
    with pytest.raises(ExternalComparisonError, match="external subject name"):
        write_external_comparison(missing_name, output_path="whitespace-name.json")

    for field in ("training_compute", "tokenizer", "data_access"):
        missing_disclosure = copy.deepcopy(payload)
        missing_disclosure["subject"][field] = " \t\n"
        with pytest.raises(ExternalComparisonError, match=f"external {field} disclosure"):
            write_external_comparison(
                missing_disclosure,
                output_path=f"whitespace-{field}.json",
            )

    contaminated = copy.deepcopy(payload)
    contaminated["tasks"]["gsm8k"]["completions"] = ["raw output"]
    with pytest.raises(ExternalComparisonError, match="aggregate-only"):
        write_external_comparison(
            contaminated,
            output_path="rejected.json",
        )

    wrong_partition = copy.deepcopy(payload)
    wrong_partition["tasks"]["gsm8k"]["total"] = 2
    with pytest.raises(ExternalComparisonError, match="pinned development partition"):
        write_external_comparison(
            wrong_partition,
            output_path="wrong-partition.json",
        )

    impossible_context = copy.deepcopy(payload)
    impossible_context["subject"]["context_length"] = 1
    with pytest.raises(ExternalComparisonError, match="fixed protocol minimum"):
        write_external_comparison(impossible_context, output_path="impossible-context.json")

    failed_preflight = copy.deepcopy(payload)
    failed_preflight["subject"]["protocol_context_preflight"]["passed"] = False
    with pytest.raises(ExternalComparisonError, match="complete no-truncation"):
        write_external_comparison(failed_preflight, output_path="failed-preflight.json")

    wrong_protocol = copy.deepcopy(payload)
    wrong_protocol["subject"]["protocol_context_preflight"]["protocol_sha256"] = "0" * 64
    with pytest.raises(ExternalComparisonError, match="compiled benchmark protocol"):
        write_external_comparison(wrong_protocol, output_path="wrong-protocol.json")

    wrong_component = copy.deepcopy(payload)
    wrong_component["subject"]["protocol_context_preflight"]["component_hashes"]["gsm8k"][
        "scorer_sha256"
    ] = "0" * 64
    with pytest.raises(ExternalComparisonError, match="prompt and scorer components"):
        write_external_comparison(wrong_component, output_path="wrong-component.json")

    truncated_context = copy.deepcopy(payload)
    truncated_context["subject"]["protocol_context_preflight"]["required_context_length"] = 3000
    truncated_context["subject"]["protocol_context_preflight"]["task_required_context_lengths"][
        "gsm8k"
    ] = 3000
    with pytest.raises(ExternalComparisonError, match="below its protocol preflight"):
        write_external_comparison(truncated_context, output_path="truncated-context.json")

    checkpoint_dir = tmp_path / "artifacts/checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "final.pt"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(ExternalComparisonError, match="outputs/external-comparisons"):
        write_external_comparison(payload, output_path=checkpoint)
    comparison_root.mkdir(parents=True, exist_ok=True)
    hardlink = comparison_root / "checkpoint-hardlink.json"
    hardlink.hardlink_to(checkpoint)
    with pytest.raises(ExternalComparisonError, match="share an inode"):
        write_external_comparison(payload, output_path=hardlink)
    assert checkpoint.read_bytes() == b"checkpoint"


def test_wandb_receives_only_compact_aggregate_rows(monkeypatch, tmp_path: Path):
    import benchmarks.runner as runner

    class Run:
        def __init__(self):
            self.summary = {}
            self.logged = []
            self.finished = False

        def log(self, payload):
            self.logged.append(payload)

        def finish(self):
            self.finished = True

    run = Run()
    init_kwargs = {}

    def fake_init(**kwargs):
        init_kwargs.update(kwargs)
        return run

    monkeypatch.setattr(runner.wandb, "init", fake_init)
    monkeypatch.setattr(runner.wandb, "Settings", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        runner.wandb,
        "Table",
        lambda *, columns, data: {"columns": columns, "data": data},
    )
    result = {
        "evaluation_identity_sha256": "e" * 64,
        "evaluation_identity": {
            "checkpoint": {
                "kind": "final",
                "logical": {},
                "physical": {"sha256": "c" * 64},
            },
            "tokenizer_fingerprint": "t" * 64,
            "suite": {
                "access": "dev",
                "protocol_sha256": "p" * 64,
                "tasks": {},
            },
        },
        "contamination": {
            "scan_complete": True,
            "scanned_documents": 12,
            "match_counts": {
                "exact": 0,
                "normalized": 0,
                "structured_json": 0,
                "shingle_48": 0,
            },
        },
        "tasks": {
            "jcommonsenseqa": {
                "primary_metric": "length_normalized_accuracy",
                "length_normalized_accuracy": 0.5,
                "correct": 64,
                "total": 128,
                "prediction_trace": [{"example_id": "secret-id", "correct": True}],
            },
            "gsm8k": {
                "primary_metric": "exact_match",
                "exact_match": 0.25,
                "correct": 32,
                "total": 128,
                "prediction_trace": [{"example_id": "secret-id", "completion_sha256": "x" * 64}],
            },
        },
    }
    config = _benchmark_config(tmp_path, tmp_path / "unused.pt")
    config.benchmark.wandb.mode = "offline"

    runner._maybe_log_wandb(
        config.benchmark,
        result,
        local_result_identity={"path": "/local/result.json", "sha256": "r" * 64},
    )

    serialized = json.dumps(
        {"init": init_kwargs, "summary": run.summary, "logged": run.logged},
        sort_keys=True,
    )
    assert "secret-id" not in serialized
    assert "prediction_trace" not in serialized
    assert "completion_sha256" not in serialized
    assert len(run.logged[0]["benchmark/results"]["data"]) == 2
    assert run.finished is True


def test_wandb_failure_cannot_fail_committed_benchmark_result(monkeypatch, tmp_path: Path):
    import benchmarks.runner as runner

    secret = "OPS_FAKE_BENCHMARK_WANDB_SECRET"
    evidence_path = tmp_path / "benchmark-wandb-events.jsonl"

    def fail_init(**_kwargs):
        raise RuntimeError(f"tracking unavailable password={secret}")

    monkeypatch.setattr(runner.wandb, "init", fail_init)
    monkeypatch.setattr(runner.wandb, "Settings", lambda **kwargs: kwargs)
    monkeypatch.setenv("LLM_SCRATCH_WANDB_EVIDENCE_PATH", str(evidence_path))
    config = _benchmark_config(tmp_path, tmp_path / "unused.pt")
    config.benchmark.wandb.mode = "offline"

    runner._maybe_log_wandb(
        config.benchmark,
        {
            "evaluation_identity_sha256": "e" * 64,
            "evaluation_identity": {"suite": {}, "checkpoint": {}, "tokenizer_fingerprint": "t"},
            "contamination": {},
            "tasks": {},
        },
        local_result_identity={"path": "/local/result.json", "sha256": "r" * 64},
    )

    evidence = evidence_path.read_text(encoding="utf-8")
    assert secret not in evidence
    assert '"action": "init"' in evidence
    assert '"outcome": "failed"' in evidence
