from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import hydra
import pytest
import torch
from omegaconf import OmegaConf

import benchmark as benchmark_cli
from benchmarks.contamination import scan_checkpoint_training_data
from benchmarks.external import ExternalComparisonError, write_external_comparison
from benchmarks.runner import BenchmarkContaminationError, run_benchmark
from benchmarks.scoring import BenchmarkScoringError, score_suite
from benchmarks.suite import (
    FINAL_ACKNOWLEDGEMENT,
    GSM8K_PROMPT_REVISION,
    GSM8K_SCORER_REVISION,
    JCOMMONSENSEQA_PROMPT_REVISION,
    JCOMMONSENSEQA_SCORER_REVISION,
    SUBSET_SELECTOR_REVISION,
    load_suite,
)
from data.identity import canonical_fingerprint
from data.stream_loader.cache import BoundedShardCache
from generation.sampler import CheckpointSampler
from models.simple_decoder_transformer import SimpleDecoderTransformer
from runtime.config import ConfigPreflightError, validate_benchmark_config
from tokenizer.canonical import CanonicalTokenizer
from training.checkpoint import CheckpointManager, build_checkpoint_identity


ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "config"
TOKENIZER_CONFIG = {
    "manifest_path": "assets/tokenizers/llm-jp-v1/manifest.json",
    "expected_fingerprint": "12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b",
}
PROTOCOL = {
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
                        }
                    ]
                ),
                "gsm8k": canonical_fingerprint(
                    [
                        {
                            "example_id": "0",
                            "record_sha256": canonical_fingerprint(dev_gsm[0]),
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
) -> tuple[Path, dict[str, Any]]:
    config = compose("profile=pretrain_streaming")
    config.runtime.device = "cpu"
    config.training.precision = "fp32"
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
        f"benchmark.output_path={tmp_path / output}",
        f"benchmark.cache.dir={tmp_path / 'benchmark-cache'}",
        "benchmark.cache.max_size_bytes=10000000",
        "benchmark.cache.min_free_bytes=0",
        "benchmark.wandb.enabled=false",
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


def test_injected_contamination_reports_source_and_document_id(tmp_path: Path):
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
    assert checkpoint.is_file()


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
                    "bc6b31d024d7fae41498cfb575440cf3c60a8afdc62144229bf289154c2ee3fc"
                ),
            }
        ],
        "prediction_trace_sha256": (
            "310663e4b3769e0578aba4e1b958c4b7fed4db6f5f2e70f0b63c990e08c8dc26"
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
                "stop_reason": "max_new_tokens",
                "completion_sha256": (
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                ),
            }
        ],
        "prediction_trace_sha256": (
            "1324e33caac696f2ca5f233879d69616399761449e055b71485005c2b2b6aae4"
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
    assert set(first["evaluation_identity"]["evaluator"]) == {
        "git",
        "lock_sha256",
        "environment",
    }
    assert len(first["evaluation_identity"]["evaluator"]["git"]["sha"]) == 40
    serialized = json.dumps(first, ensure_ascii=False).lower()
    assert '"prompt"' not in serialized
    assert '"completion"' not in serialized


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


def test_external_baseline_record_is_aggregate_only_and_isolated(tmp_path: Path):
    payload = {
        "subject": {
            "name": "small-open-baseline",
            "parameter_count": 1_000_000,
            "training_compute": "disclosed by upstream",
            "tokenizer": "upstream tokenizer",
            "context_length": 2048,
            "data_access": "upstream disclosure",
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
        output_path=tmp_path / "external.json",
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["kind"] == "external_baseline_comparison"
    assert record["isolation"]["repository_checkpoint_runner"] is False
    assert record["isolation"]["eligible_for_training_data_or_targets"] is False
    assert record["suite"]["access"] == "dev"
    assert record["suite"]["protocol_sha256"] == (
        "169058462fd2ea3c3dddddce6486d6f1fceebaf1836f228bada6b5c5c8e602e2"
    )
    assert record["suite"]["tasks"]["jcommonsenseqa"]["selected_examples_sha256"] == (
        "fa5ce35310f98b171da7db6afeff222381161f1987a99d70e7ede9b77a283b0e"
    )

    contaminated = copy.deepcopy(payload)
    contaminated["tasks"]["gsm8k"]["completions"] = ["raw output"]
    with pytest.raises(ExternalComparisonError, match="aggregate-only"):
        write_external_comparison(
            contaminated,
            output_path=tmp_path / "rejected.json",
        )

    wrong_partition = copy.deepcopy(payload)
    wrong_partition["tasks"]["gsm8k"]["total"] = 2
    with pytest.raises(ExternalComparisonError, match="pinned development partition"):
        write_external_comparison(
            wrong_partition,
            output_path=tmp_path / "wrong-partition.json",
        )


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
            "match_counts": {"exact": 0, "normalized": 0, "shingle_48": 0},
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
    config.benchmark.wandb.enabled = True

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
