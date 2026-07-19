from __future__ import annotations

import json
from pathlib import Path

import human_evaluation.contamination as scans
from data.stream_loader.cache import BoundedShardCache
from human_evaluation.contamination import scan_checkpoint_training_prompts
from human_evaluation.schema import Prompt


ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "tests/fixtures/data_manifests/bilingual.manifest.json"
MANIFEST_FINGERPRINT = "47cca88c4a5595e27eb5d60d99918fb77c30b23f7c0ae98024153f25e14ffc19"


def _checkpoint_config(tmp_path: Path) -> dict:
    return {
        "profile": {"purpose": "pretraining"},
        "data": {
            "mode": "streaming",
            "streaming": {
                "cache": {
                    "dir": str(tmp_path / "training-cache"),
                    "max_size_bytes": 10_000_000,
                    "min_free_bytes": 0,
                    "wait_timeout_seconds": 1.0,
                },
                "retry": {
                    "max_attempts": 1,
                    "initial_delay_seconds": 0.0,
                    "max_delay_seconds": 0.0,
                    "multiplier": 1.0,
                },
                "train": {
                    "sources": [
                        {
                            "name": "fixture_train",
                            "type": "manifest",
                            "manifest_path": str(MANIFEST),
                            "expected_fingerprint": MANIFEST_FINGERPRINT,
                            "selection": "train",
                            "timeout_seconds": 1.0,
                        }
                    ]
                },
            },
        },
    }


def _checkpoint_identity() -> dict:
    return {
        "config_sha256": "a" * 64,
        "run_lineage_id": "run-11111111111111111111111111111111",
        "data_fingerprints": [MANIFEST_FINGERPRINT],
    }


def _evaluated_checkpoints() -> list[dict]:
    return [
        {"slot": "earlier", "sha256": "c" * 64, "optimizer_step": 10, "target_tokens": 100},
        {"slot": "later", "sha256": "d" * 64, "optimizer_step": 20, "target_tokens": 200},
    ]


def test_complete_prompt_scan_finds_exact_and_normalized_occurrences_without_text(
    tmp_path: Path, monkeypatch
):
    exact = "We train a small model from random initialization."
    normalized_only = f"  {exact}  "
    prompts = (
        Prompt(id="en-exact", language="en", text=exact),
        Prompt(id="en-normalized", language="en", text=normalized_only),
    )
    fallback = BoundedShardCache(tmp_path / "fallback", max_size_bytes=10_000_000)
    report = scan_checkpoint_training_prompts(
        _checkpoint_config(tmp_path),
        _checkpoint_identity(),
        prompts,
        prompt_set_version="fixture-v1",
        prompt_set_sha256="b" * 64,
        evaluated_checkpoints=_evaluated_checkpoints(),
        fallback_cache=fallback,
        repository_root=ROOT,
    )

    assert report["scan_complete"] is True
    assert report["contaminated"] is True
    assert report["match_counts"] == {"exact": 1, "normalized": 2}
    assert report["scanned_documents"] == report["training_sources"][0]["documents"]
    assert report["scanned_documents"] > 1
    assert {match["training_document_id"] for match in report["matches"]} == {
        "07adfc7271fb15054164c9ec16a4c0b6c455b0ff04a6db5fa225215d05c65c81"
    }
    serialized = json.dumps(report, ensure_ascii=False)
    assert exact not in serialized
    assert normalized_only not in serialized
    assert report["identity"]["checkpoint"] == _checkpoint_identity()
    assert report["identity"]["evaluated_checkpoints"] == _evaluated_checkpoints()
    assert report["minimum_free_bytes"] == scans.MINIMUM_FREE_BYTES

    original_source = scans.ManifestTextSource
    source_calls = 0

    def counted_source(*args, **kwargs):
        nonlocal source_calls
        source_calls += 1
        return original_source(*args, **kwargs)

    monkeypatch.setattr(scans, "ManifestTextSource", counted_source)
    reused = scan_checkpoint_training_prompts(
        _checkpoint_config(tmp_path),
        _checkpoint_identity(),
        prompts,
        prompt_set_version="fixture-v1",
        prompt_set_sha256="b" * 64,
        evaluated_checkpoints=_evaluated_checkpoints(),
        fallback_cache=fallback,
        repository_root=ROOT,
    )
    assert reused == report
    assert source_calls == 0

    changed_pair = _evaluated_checkpoints()
    changed_pair[1]["sha256"] = "e" * 64
    changed = scan_checkpoint_training_prompts(
        _checkpoint_config(tmp_path),
        _checkpoint_identity(),
        prompts,
        prompt_set_version="fixture-v1",
        prompt_set_sha256="b" * 64,
        evaluated_checkpoints=changed_pair,
        fallback_cache=fallback,
        repository_root=ROOT,
    )
    assert source_calls == 1
    assert changed["identity"]["evaluated_checkpoints"] == changed_pair


def test_complete_prompt_scan_records_clean_checkpoint_owned_manifests(tmp_path: Path):
    prompt = Prompt(
        id="en-clean",
        language="en",
        text="This HUMAN-only prompt does not occur in the fixture corpus.",
    )
    report = scan_checkpoint_training_prompts(
        _checkpoint_config(tmp_path),
        _checkpoint_identity(),
        (prompt,),
        prompt_set_version="fixture-clean-v1",
        prompt_set_sha256="c" * 64,
        evaluated_checkpoints=_evaluated_checkpoints(),
        fallback_cache=BoundedShardCache(tmp_path / "fallback-clean", max_size_bytes=10_000_000),
        repository_root=ROOT,
    )

    assert report["scan_complete"] is True
    assert report["contaminated"] is False
    assert report["matches"] == []
    assert report["match_counts"] == {"exact": 0, "normalized": 0}
    assert report["training_sources"][0]["manifest_fingerprint"] == MANIFEST_FINGERPRINT
    assert report["scanned_document_order_sha256"]
