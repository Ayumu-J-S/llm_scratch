import hashlib
import json
from collections import Counter
from pathlib import Path

from tokenizer.comparison import select_winner


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/tokenizer_comparison/v1"
REPORT_DIR = ROOT / "reports/tokenizers/TOK-001"


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_frozen_corpus_checksum_count_and_strata():
    corpus_path = FIXTURE / "corpus.jsonl"
    metadata = read_json(FIXTURE / "metadata.json")
    raw = corpus_path.read_bytes()
    documents = [json.loads(line) for line in raw.decode("utf-8").splitlines()]

    assert hashlib.sha256(raw).hexdigest() == metadata["sha256"]
    assert len(documents) == metadata["document_count"] == 160
    assert len({document["id"] for document in documents}) == 160
    assert Counter(document["stratum"] for document in documents) == metadata["strata"]
    assert set(metadata["strata"]) == {
        "japanese",
        "english",
        "mixed",
        "code_symbols",
        "emoji_unicode",
        "whitespace_normalization",
        "short",
        "long",
    }


def test_malformed_fixture_covers_bytes_and_surrogate_codepoints():
    recipes = read_json(FIXTURE / "malformed.json")["recipes"]

    assert len(recipes) == 10
    assert Counter(recipe["kind"] for recipe in recipes) == {
        "utf8_bytes": 5,
        "codepoints": 5,
    }
    assert all(
        any(0xD800 <= value <= 0xDFFF for value in recipe["values"])
        for recipe in recipes
        if recipe["kind"] == "codepoints"
    )


def test_candidate_lock_uses_exact_revisions_and_allowlisted_non_weight_files():
    lock = read_json(REPORT_DIR / "candidates.lock.json")
    candidates = {candidate["id"]: candidate for candidate in lock["candidates"]}

    assert {
        candidate_id: candidate["revision"] for candidate_id, candidate in candidates.items()
    } == {
        "llm-jp-v1": "c3134b3a958b56d443c1484a3d640502637cfbd2",
        "rinna-bilingual": "803fb7671ac30766ffc6d21139d809b549ee26a3",
        "llm-jp-v3": "cd3823f4c1fcbb0ad2e2af46036ab1b0ca13192a",
        "qwen3-control": "c1899de289a04d12100db370d81485cdf75e47ca",
    }
    assert not candidates["rinna-bilingual"]["eligible_to_compare"]
    assert candidates["rinna-bilingual"]["files"] == []
    allowed_names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "LICENSE",
        "LICENSE.tokenizer-source",
    }
    for candidate in candidates.values():
        assert len(candidate["revision"]) == 40
        assert all(character in "0123456789abcdef" for character in candidate["revision"])
        for file_record in candidate["files"]:
            assert file_record["local_filename"] in allowed_names
            assert len(file_record["sha256"]) == 64
            assert file_record["size_bytes"] > 0
            assert not any(
                forbidden in file_record["local_filename"].lower()
                for forbidden in ("model.safetensors", "pytorch_model", "generation_config")
            )


def make_result(candidate_id, vocab_size, japanese, english, throughput, *, passed=True):
    return {
        "id": candidate_id,
        "hard_gate_pass": passed,
        "identity": {"vocab_size": vocab_size},
        "compression": {
            "by_stratum": {
                "japanese": {"tokens_per_utf8_byte": japanese},
                "english": {"tokens_per_utf8_byte": english},
            }
        },
        "cost": {"vocabulary_parameters": vocab_size * 769},
        "performance": {
            "summary": {
                "utf8_bytes_per_second": {
                    "median": throughput,
                    "min": throughput * 0.95,
                }
            }
        },
    }


def test_selection_rule_rejects_hard_gate_failure_and_ties_to_smaller_vocab():
    smaller = make_result("small", 50_000, 0.20, 0.20, 1_000_000)
    larger = make_result("large", 100_000, 0.19, 0.19, 1_000_000)
    failed = make_result("failed", 10_000, 0.01, 0.01, 10_000_000, passed=False)

    selection = select_winner([larger, failed, smaller])

    assert selection["winner"] == "small"
    assert not selection["trace"][0]["promoted"]


def test_selection_rule_promotes_material_bilingual_compression_gain():
    smaller = make_result("small", 50_000, 0.20, 0.20, 1_000_000)
    larger = make_result("large", 100_000, 0.17, 0.18, 960_000)

    selection = select_winner([smaller, larger])

    assert selection["winner"] == "large"
    assert selection["trace"][0]["promoted"]


def test_committed_report_schema_and_selection_are_reproducible():
    report = read_json(REPORT_DIR / "comparison.json")
    measured = [
        candidate for candidate in report["candidates"] if candidate["status"] == "measured"
    ]
    by_id = {candidate["id"]: candidate for candidate in measured}

    assert report["schema_version"] == 1
    assert report["fixture"]["sha256"] == read_json(FIXTURE / "metadata.json")["sha256"]
    assert report["selection"] == select_winner(report["candidates"])
    assert report["selection"]["winner"] == "llm-jp-v1"
    assert by_id["llm-jp-v1"]["hard_gate_pass"]
    assert by_id["llm-jp-v3"]["hard_gate_pass"]
    assert not by_id["qwen3-control"]["hard_gate_pass"]
    assert not by_id["qwen3-control"]["hard_gates"]["exact_unicode_round_trip"]
    for candidate in measured:
        passes = candidate["performance"]["passes"]
        assert len(passes) == 5
        assert {item["ids_sha256"] for item in passes} == {candidate["ids_sha256"]}
        assert candidate["malformed"]["all_rejected"]
        assert candidate["malformed"]["all_deterministic"]
        assert candidate["cost"]["batch_logit_elements"] == (
            64 * 64 * candidate["identity"]["vocab_size"]
        )
