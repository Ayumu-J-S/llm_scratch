from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from data.preflight import (
    DataPreflightError,
    assert_cache_released,
    assert_cold_cache_is_empty,
    assert_loader_accounting,
    assert_observed_disjointness,
    audit_raw_stream,
    cache_telemetry,
    config_fingerprint,
    data_fingerprints,
    disk_forecast,
    merge_loader_quality,
    merge_quota_truncation,
    process_resource_report,
    reproduction_payload,
    render_markdown,
    summarize_packing,
    summarize_repeated_reports,
    write_reports,
)
from tokenizer.canonical import CanonicalTokenizer


TOKENIZER_CONFIG = {
    "manifest_path": "assets/tokenizers/llm-jp-v1/manifest.json",
    "expected_fingerprint": "12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b",
}


class _Tokenizer:
    fingerprint = "f" * 64

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def count_byte_fallback_tokens(self, token_ids: list[int]) -> int:
        return sum(token_id == 255 for token_id in token_ids)


def _sample(source: str, text: str, content_hash: str, **metadata):
    return {
        "source": source,
        "text": text,
        "metadata": {
            "content_sha256": content_hash,
            "document_id": metadata.pop("document_id", content_hash),
            **metadata,
        },
    }


def test_audit_reports_scripts_languages_fallbacks_duplicates_and_no_raw_text():
    samples = [
        _sample("en", "hello", "a" * 64),
        _sample("ja", "日本語", "b" * 64, fallback_id=True, truncated=True),
        _sample("en", "duplicate text", "a" * 64),
        _sample("ja", "mixed text 日本", "c" * 64),
    ]

    result = audit_raw_stream(samples, _Tokenizer(), max_documents=4, add_eos=True)
    merge_loader_quality(
        result.report,
        rejection_counts={
            "en": {"empty": 2, "control_character": 1},
            "ja": {"invalid_unicode": 1, "wrong_script": 3},
        },
        fallback_counts={"ja": 1},
        truncated_counts={"ja": 1},
        raw_row_counts={"en": 9, "ja": 7},
        missing_data_counts={"en": {"null:text": 2}},
    )

    assert result.report["documents"] == 4
    assert result.report["sources"]["en"]["quality"]["duplicates"] == 1
    assert result.report["sources"]["ja"]["quality"]["fallback_document_ids"] == 1
    assert result.report["sources"]["ja"]["quality"]["truncated"] == 1
    assert result.report["sources"]["ja"]["languages"] == {
        "en": 0,
        "ja": 1,
        "mixed": 1,
        "other": 0,
    }
    assert result.report["sources"]["en"]["quality"]["rejections"]["empty"] == 2
    assert result.report["sources"]["en"]["quality"]["source_rows_read"] == 9
    assert result.report["sources"]["en"]["quality"]["missing_data"] == {"null:text": 2}
    assert result.report["sources"]["en"]["rates"]["source_rows_per_second"] > 0
    serialized = json.dumps(result.report, sort_keys=True, ensure_ascii=False)
    assert "hello" not in serialized
    assert "日本語" not in serialized
    assert len(result.content_hashes) == 3
    assert len(result.document_ids) == 3
    assert result.report["timing"]["elapsed_seconds"] > 0
    assert result.report["sources"]["en"]["rates"]["canonical_tokens_per_second"] > 0


def test_audit_requires_document_ids_and_reports_both_overlap_types():
    missing_id = {
        "source": "en",
        "text": "missing",
        "metadata": {"content_sha256": "a" * 64},
    }
    with pytest.raises(DataPreflightError, match="metadata.document_id"):
        audit_raw_stream([missing_id], _Tokenizer(), max_documents=1, add_eos=False)

    train = audit_raw_stream(
        [_sample("en", "same", "a" * 64, document_id="b" * 64)],
        _Tokenizer(),
        max_documents=1,
        add_eos=False,
    )
    validation = audit_raw_stream(
        [_sample("en", "same", "a" * 64, document_id="b" * 64)],
        _Tokenizer(),
        max_documents=1,
        add_eos=False,
    )
    with pytest.raises(DataPreflightError, match="document_ids=1, normalized_content=1"):
        assert_observed_disjointness(train, validation)

    disjoint = audit_raw_stream(
        [_sample("en", "different", "c" * 64, document_id="d" * 64)],
        _Tokenizer(),
        max_documents=1,
        add_eos=False,
    )
    assert assert_observed_disjointness(train, disjoint) == {
        "observed_document_id_overlap": 0,
        "observed_normalized_content_overlap": 0,
    }


def test_audit_is_bounded_and_fails_closed_on_tokenization_error():
    class BrokenTokenizer(_Tokenizer):
        def encode(self, text: str) -> list[int]:
            raise ValueError("reserved token")

    with pytest.raises(DataPreflightError, match="canonical tokenization failed"):
        audit_raw_stream(
            [_sample("en", "bad", "d" * 64)],
            BrokenTokenizer(),
            max_documents=1,
            add_eos=True,
        )

    result = audit_raw_stream(
        [_sample("en", str(index), f"{index:064x}") for index in range(10)],
        _Tokenizer(),
        max_documents=3,
        add_eos=False,
    )
    assert result.report["documents"] == 3
    assert result.report["bounded"] is True


def test_packing_reconciles_targets_and_enforces_ratio_tolerance():
    samples = [
        {
            "window_token_count": 5,
            "target_token_count": 4,
            "source_target_counts": {"en": 2, "ja": 2},
        },
        {
            "window_token_count": 5,
            "target_token_count": 4,
            "source_target_counts": {"en": 2, "ja": 2},
        },
    ]
    report = summarize_packing(
        samples,
        expected_ratios={"en": 0.5, "ja": 0.5},
        tolerance=0.01,
    )
    assert report["trained_targets"] == 8
    assert report["accounting_reconciled"] is True
    assert report["ratios_within_tolerance"] is True

    loader = SimpleNamespace(
        trained_target_counts={"en": 4, "ja": 4},
        packed_token_counts={"window_token_count": 10, "target_token_count": 8},
    )
    assert_loader_accounting(loader, report)
    loader.quota_truncated_fragment_counts = {"en": 1, "ja": 0}
    loader.quota_removed_token_counts = {"en": 3, "ja": 0}
    merge_quota_truncation(loader, report)
    assert report["quota_truncation"] == {
        "fragments_by_source": {"en": 1, "ja": 0},
        "removed_tokens_by_source": {"en": 3, "ja": 0},
        "total_fragments": 1,
        "total_removed_tokens": 3,
        "policy": "trained-target source quota; distinct from byte-policy document truncation",
    }
    assert report["rates"]["target_tokens_per_second"] > 0
    loader.trained_target_counts["en"] = 3
    with pytest.raises(DataPreflightError, match="trained-target counters"):
        assert_loader_accounting(loader, report)

    skewed = summarize_packing(
        [
            {
                "window_token_count": 5,
                "target_token_count": 4,
                "source_target_counts": {"en": 4},
            }
        ],
        expected_ratios={"en": 0.5, "ja": 0.5},
        tolerance=0.01,
    )
    assert skewed["ratios_within_tolerance"] is False


def test_packing_rejects_inconsistent_or_undeclared_source_accounting():
    with pytest.raises(DataPreflightError, match="do not reconcile"):
        summarize_packing(
            [
                {
                    "window_token_count": 4,
                    "target_token_count": 3,
                    "source_target_counts": {"en": 2},
                }
            ],
            expected_ratios={"en": 1.0},
            tolerance=0.0,
        )
    with pytest.raises(DataPreflightError, match="undeclared"):
        summarize_packing(
            [
                {
                    "window_token_count": 2,
                    "target_token_count": 1,
                    "source_target_counts": {"other": 1},
                }
            ],
            expected_ratios={"en": 1.0},
            tolerance=0.0,
        )


def test_cold_cache_requires_existing_empty_directory_and_never_deletes(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(DataPreflightError, match="existing empty"):
        assert_cold_cache_is_empty(missing)

    cache = tmp_path / "cache"
    cache.mkdir()
    assert_cold_cache_is_empty(cache)
    sentinel = cache / "do-not-delete"
    sentinel.write_text("retain", encoding="utf-8")
    with pytest.raises(DataPreflightError, match="refuses to delete"):
        assert_cold_cache_is_empty(cache)
    assert sentinel.read_text(encoding="utf-8") == "retain"


def test_cache_telemetry_reports_retries_bytes_and_requires_all_leases_released():
    loader = SimpleNamespace(
        cache=SimpleNamespace(
            telemetry={
                "hits": 2,
                "misses": 1,
                "downloads": 1,
                "downloaded_bytes": 123,
                "retries": 1,
                "evictions": 0,
                "corruptions": 0,
                "wait_timeouts": 0,
                "active_leases": 0,
                "size_bytes": 123,
                "free_bytes": 999,
            }
        )
    )
    telemetry = cache_telemetry([loader])
    assert telemetry["downloaded_bytes"] == 123
    assert telemetry["retries"] == 1
    assert_cache_released(telemetry)
    telemetry["active_leases"] = 1
    with pytest.raises(DataPreflightError, match="unreleased cache leases"):
        assert_cache_released(telemetry)


def test_disk_forecast_reserves_full_cache_largest_temp_and_os_checkpoint_headroom(
    monkeypatch, tmp_path
):
    usage = SimpleNamespace(total=1_000, used=200, free=800)
    monkeypatch.setattr("data.preflight.shutil.disk_usage", lambda _: usage)
    safe = disk_forecast(
        tmp_path,
        cache_cap_bytes=400,
        current_managed_cache_bytes=100,
        reserved_os_checkpoint_bytes=200,
        largest_shard_bytes=100,
        cache_cap_limit_bytes=500,
    )
    assert safe["projected_free_at_full_cache_with_largest_temp_bytes"] == 400
    assert safe["headroom_admission_passed"] is True

    unsafe = disk_forecast(
        tmp_path,
        cache_cap_bytes=600,
        current_managed_cache_bytes=0,
        reserved_os_checkpoint_bytes=200,
        largest_shard_bytes=100,
        cache_cap_limit_bytes=500,
    )
    assert unsafe["headroom_admission_passed"] is False
    assert unsafe["cache_cap_within_limit"] is False


def test_data_fingerprints_reject_conflicting_source_identity():
    first = SimpleNamespace(
        resolved_manifests={
            "en": SimpleNamespace(
                dataset_fingerprint="a" * 64,
                manifest_fingerprint="b" * 64,
                selection="train",
            )
        }
    )
    second = SimpleNamespace(
        resolved_manifests={
            "en": SimpleNamespace(
                dataset_fingerprint="a" * 64,
                manifest_fingerprint="c" * 64,
                selection="train",
            )
        }
    )
    assert data_fingerprints([first])["en:train"]["selection"] == "train"
    with pytest.raises(DataPreflightError, match="conflicting manifest identity"):
        data_fingerprints([first, second])


def test_reports_are_stably_serialized_and_markdown_contains_no_raw_text(tmp_path):
    report = {
        "schema_version": 1,
        "mode": "warm",
        "fingerprints": {
            "code": {"git_commit": "a" * 40, "dirty": False},
            "config": "b" * 64,
            "data": {},
            "tokenizer": "c" * 64,
        },
        "splits": {
            "train": {
                "sources": {
                    "en": {
                        "documents": 1,
                        "utf8_bytes": 5,
                        "canonical_tokens": 1,
                        "eos_tokens": 1,
                        "quality": {"duplicates": 0, "rejections": {}},
                    }
                }
            }
        },
        "packing": {
            "windows": 1,
            "trained_targets": 1,
            "accounting_reconciled": True,
            "ratios_within_tolerance": True,
            "expected_ratios": {"en": 1.0},
            "realized_ratios": {"en": 1.0},
            "ratio_deviations": {"en": 0.0},
        },
        "cache": {"hits": 1},
        "disk": {
            "cache_cap_within_limit": True,
            "headroom_admission_passed": True,
            "reserved_os_checkpoint_bytes": 200,
            "largest_shard_temp_bytes": 100,
        },
    }
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    write_reports(report, json_path, markdown_path)
    first_json = json_path.read_bytes()
    write_reports(report, json_path, markdown_path)
    assert json_path.read_bytes() == first_json
    assert json.loads(first_json)["schema_version"] == 1
    assert render_markdown(report) == markdown_path.read_text(encoding="utf-8")
    assert "raw corpus" in markdown_path.read_text(encoding="utf-8")
    assert config_fingerprint({"b": 2, "a": 1}) == config_fingerprint({"a": 1, "b": 2})


def test_reproduction_payload_embeds_complete_recomputable_config_and_exact_argv():
    config = {"profile": "pretrain_streaming", "preflight": {"report_stem": "run-1"}}
    payload = reproduction_payload(
        config,
        ["scripts/preflight_data.py", "profile=x", "--cold"],
        effective_argv=["scripts/preflight_data.py", "profile=x", "+preflight.cold=true"],
    )
    assert payload["resolved_hydra_config"] == config
    assert payload["argv"] == ["scripts/preflight_data.py", "profile=x", "--cold"]
    assert payload["effective_argv"][-1] == "+preflight.cold=true"
    assert payload["config_sha256"] == config_fingerprint(payload["resolved_hydra_config"])
    assert "profile=x" in payload["python_command"]
    with pytest.raises(DataPreflightError, match="secret-bearing"):
        reproduction_payload({"api_key": "do-not-record"}, ["preflight"])


def test_process_resource_report_retains_rss_fault_swap_and_rates():
    start = {
        "monotonic_ns": 1,
        "current_rss_bytes": 10,
        "peak_rss_bytes": 20,
        "swap_bytes": 0,
        "minor_page_faults": 3,
        "major_page_faults": 1,
    }
    end = {
        "monotonic_ns": 2,
        "current_rss_bytes": 15,
        "peak_rss_bytes": 30,
        "swap_bytes": 2,
        "minor_page_faults": 7,
        "major_page_faults": 2,
    }
    report = process_resource_report(start, end, elapsed_seconds=2.0)
    assert report["current_rss_bytes"] == 15
    assert report["peak_rss_growth_bytes"] == 10
    assert report["swap_growth_bytes"] == 2
    assert report["minor_page_faults_per_second"] == 2.0
    assert report["major_page_faults_per_second"] == 0.5


def _repeat_report(value: float, stem: str = "run") -> dict:
    config = {"preflight": {"report_stem": stem, "output_dir": "/tmp/out"}, "fixed": 1}
    return {
        "mode": "warm",
        "limits": {"max_documents_per_split": 4, "max_target_tokens": 8},
        "fingerprints": {
            "code": {"git_commit": "a" * 40},
            "config": config_fingerprint(config),
            "data": {"en:train": {"manifest": "b" * 64}},
            "tokenizer": "c" * 64,
        },
        "reproduction": {"resolved_hydra_config": config},
        "measurement": {
            "whole_run_elapsed_seconds": value,
            "conditions": {
                "platform": "linux",
                "machine": "aarch64",
                "python": "3.12",
                "scope": "loader_only",
            },
            "process_resources": {
                "peak_rss_bytes": 100 + value,
                "current_rss_bytes": 90 + value,
                "minor_page_faults_per_second": value,
                "major_page_faults_per_second": 0.0,
            },
        },
        "packing": {"rates": {"target_tokens_per_second": 100 / value}},
        "cache": {"downloaded_bytes_per_second": 0.0},
        "splits": {
            "train": {
                "sources": {
                    "en": {
                        "rates": {
                            "documents_per_second": 10 / value,
                            "utf8_bytes_per_second": 20 / value,
                            "canonical_tokens_per_second": 30 / value,
                            "source_rows_per_second": 40 / value,
                        }
                    }
                }
            }
        },
    }


def test_repeated_summary_retains_observations_median_spread_and_comparability():
    summary = summarize_repeated_reports(
        [_repeat_report(1.0, "one"), _repeat_report(3.0, "two"), _repeat_report(2.0, "three")]
    )
    elapsed = summary["metrics"]["whole_run_elapsed_seconds"]
    assert elapsed == {
        "observations": [1.0, 3.0, 2.0],
        "median": 2.0,
        "min": 1.0,
        "max": 3.0,
        "spread": 2.0,
    }
    incompatible = _repeat_report(2.0)
    incompatible["mode"] = "cold"
    with pytest.raises(DataPreflightError, match="not comparable"):
        summarize_repeated_reports([_repeat_report(1.0), incompatible])


def test_canonical_tokenizer_counts_only_exact_serialized_byte_fallback_pieces():
    tokenizer = CanonicalTokenizer.from_config(TOKENIZER_CONFIG)
    fallback_id = tokenizer._tokenizer.token_to_id("<0xFF>")
    lowercase_literal_id = tokenizer._tokenizer.token_to_id("<0xff>")
    assert fallback_id is not None
    ids = [fallback_id]
    if lowercase_literal_id is not None:
        ids.append(lowercase_literal_id)
    assert tokenizer.count_byte_fallback_tokens(ids) == 1
    with pytest.raises(TypeError, match="integers"):
        tokenizer.count_byte_fallback_tokens([True])
