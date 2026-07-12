"""Reproduce the TOK-001 frozen tokenizer comparison without loading model weights."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = ROOT / "reports/tokenizers/TOK-001/candidates.lock.json"
DEFAULT_CORPUS = ROOT / "tests/fixtures/tokenizer_comparison/v1/corpus.jsonl"
DEFAULT_OUTPUT = ROOT / "reports/tokenizers/TOK-001/comparison.json"
DEFAULT_CACHE = Path.home() / ".cache/llm-scratch/tokenizers/TOK-001"
BYTE_FALLBACK_PATTERN = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile(values: list[float] | list[int], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a quantile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def rounded(value: float) -> float:
    return round(value, 6)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_fixture(corpus_path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    metadata = load_json(corpus_path.with_name("metadata.json"))
    actual_hash = sha256_file(corpus_path)
    if actual_hash != metadata["sha256"]:
        raise ValueError(
            f"frozen corpus checksum mismatch: expected {metadata['sha256']}, got {actual_hash}"
        )
    with corpus_path.open(encoding="utf-8") as handle:
        documents = [json.loads(line) for line in handle if line.strip()]
    if len(documents) != metadata["document_count"]:
        raise ValueError("frozen corpus document count does not match metadata")
    counts = Counter(document["stratum"] for document in documents)
    if dict(counts) != metadata["strata"]:
        raise ValueError(f"frozen corpus strata mismatch: {dict(counts)}")
    if len({document["id"] for document in documents}) != len(documents):
        raise ValueError("frozen corpus document IDs are not unique")
    return documents, metadata


def candidate_directory(cache_root: Path, candidate: dict[str, Any]) -> Path:
    return cache_root / candidate["slug"] / candidate["revision"]


def fetch_allowlisted_files(lock: dict[str, Any], cache_root: Path) -> None:
    for candidate in lock["candidates"]:
        if not candidate["eligible_to_compare"]:
            continue
        directory = candidate_directory(cache_root, candidate)
        directory.mkdir(parents=True, exist_ok=True)
        for file_record in candidate["files"]:
            destination = directory / file_record["local_filename"]
            if (
                destination.exists()
                and destination.stat().st_size == file_record["size_bytes"]
                and sha256_file(destination) == file_record["sha256"]
            ):
                continue
            temporary = destination.with_suffix(destination.suffix + ".download")
            with urllib.request.urlopen(file_record["url"], timeout=120) as response:
                with temporary.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            if temporary.stat().st_size != file_record["size_bytes"]:
                temporary.unlink()
                raise ValueError(f"download size mismatch for {candidate['id']}/{destination.name}")
            actual_hash = sha256_file(temporary)
            if actual_hash != file_record["sha256"]:
                temporary.unlink()
                raise ValueError(
                    f"download checksum mismatch for {candidate['id']}/{destination.name}"
                )
            temporary.replace(destination)


def verify_candidate_files(
    candidate: dict[str, Any], cache_root: Path
) -> tuple[Path, list[dict[str, Any]]]:
    directory = candidate_directory(cache_root, candidate)
    verified: list[dict[str, Any]] = []
    tokenizer_path: Path | None = None
    for file_record in candidate["files"]:
        path = directory / file_record["local_filename"]
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned file: {path}")
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path)
        if actual_size != file_record["size_bytes"] or actual_hash != file_record["sha256"]:
            raise ValueError(
                f"pinned file mismatch for {path}: size={actual_size}, sha256={actual_hash}"
            )
        verified.append(
            {
                "local_filename": file_record["local_filename"],
                "role": file_record["role"],
                "size_bytes": actual_size,
                "sha256": actual_hash,
                "local_path": str(path),
            }
        )
        if file_record["role"] == "tokenizer":
            tokenizer_path = path
    if tokenizer_path is None:
        raise ValueError(f"candidate {candidate['id']} has no allowlisted tokenizer file")
    return tokenizer_path, verified


def read_vmhwm_kib() -> int | None:
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return None
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return None


def component_types(component: Any) -> list[str]:
    types = []
    if isinstance(component, dict):
        if isinstance(component.get("type"), str):
            types.append(component["type"])
        for value in component.values():
            types.extend(component_types(value))
    elif isinstance(component, list):
        for value in component:
            types.extend(component_types(value))
    return list(dict.fromkeys(types))


def tokenizer_identity(tokenizer: Any, directory: Path) -> dict[str, Any]:
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    serialized = json.loads(tokenizer.to_str())
    decoder = tokenizer.get_added_tokens_decoder()
    special_tokens = [
        {"id": int(token_id), "token": str(token), "special": bool(token.special)}
        for token_id, token in sorted(decoder.items())
    ]
    config_path = directory / "tokenizer_config.json"
    semantic_tokens: dict[str, Any] = {}
    if config_path.exists():
        config = load_json(config_path)
        for key in ("unk_token", "bos_token", "eos_token", "pad_token"):
            token = config.get(key)
            if isinstance(token, dict):
                token = token.get("content")
            semantic_tokens[key] = {
                "token": token,
                "id": None if token is None else tokenizer.token_to_id(token),
            }
    else:
        # Qwen's tokenizer_config carries a chat template and is deliberately not fetched.
        for key, token in {
            "unk_token": "<unk>",
            "bos_token": None,
            "eos_token": "<|endoftext|>",
            "pad_token": None,
        }.items():
            semantic_tokens[key] = {
                "token": token,
                "id": None if token is None else tokenizer.token_to_id(token),
            }
    return {
        "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
        "max_vocab_id": max(vocab.values()),
        "min_vocab_id": min(vocab.values()),
        "special_tokens": special_tokens,
        "semantic_tokens": semantic_tokens,
        "pipeline": {
            "model_type": serialized["model"]["type"],
            "model_byte_fallback": serialized["model"].get("byte_fallback"),
            "normalizer_types": component_types(serialized.get("normalizer")),
            "pre_tokenizer_types": component_types(serialized.get("pre_tokenizer")),
            "decoder_types": component_types(serialized.get("decoder")),
        },
    }


def malformed_behavior(tokenizer: Any, malformed_path: Path) -> dict[str, Any]:
    recipes = load_json(malformed_path)["recipes"]
    outcomes: list[dict[str, Any]] = []
    for recipe in recipes:
        signatures = []
        for _ in range(2):
            try:
                if recipe["kind"] == "utf8_bytes":
                    text = bytes.fromhex(recipe["hex"]).decode("utf-8", errors="strict")
                else:
                    text = "".join(chr(value) for value in recipe["values"])
                tokenizer.encode(text, add_special_tokens=False)
                signature = {"outcome": "accepted"}
            except Exception as error:  # the exception type is the evidence under test
                signature = {
                    "outcome": "rejected",
                    "exception": type(error).__name__,
                    "message": str(error).encode("ascii", "backslashreplace").decode("ascii"),
                }
            signatures.append(signature)
        outcomes.append(
            {
                "id": recipe["id"],
                "kind": recipe["kind"],
                "deterministic": signatures[0] == signatures[1],
                **signatures[0],
            }
        )
    return {
        "policy": load_json(malformed_path)["policy"],
        "all_rejected": all(item["outcome"] == "rejected" for item in outcomes),
        "all_deterministic": all(item["deterministic"] for item in outcomes),
        "outcomes": outcomes,
    }


def cost_estimate(vocab_size: int, measurement: dict[str, Any]) -> dict[str, int]:
    embed_size = measurement["embed_size"]
    batch_tokens = measurement["batch_size"] * measurement["sequence_length"]
    embedding_parameters = vocab_size * embed_size
    lm_head_parameters = vocab_size * embed_size + vocab_size
    vocabulary_parameters = embedding_parameters + lm_head_parameters
    logits_elements = batch_tokens * vocab_size
    return {
        "embedding_parameters": embedding_parameters,
        "lm_head_parameters_including_bias": lm_head_parameters,
        "vocabulary_parameters": vocabulary_parameters,
        "fp32_model_bytes": vocabulary_parameters * 4,
        "fp32_gradient_bytes": vocabulary_parameters * 4,
        "adam_moment_bytes": vocabulary_parameters * 8,
        "fp32_training_state_bytes": vocabulary_parameters * 16,
        "model_plus_adam_checkpoint_bytes": vocabulary_parameters * 12,
        "batch_logit_elements": logits_elements,
        "batch_logits_bf16_bytes": logits_elements * 2,
        "batch_logits_fp32_bytes": logits_elements * 4,
        "lm_head_multiply_accumulates_per_batch": batch_tokens * embed_size * vocab_size,
    }


def encode_validation(tokenizer: Any, documents: list[dict[str, str]]) -> dict[str, Any]:
    by_stratum: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"tokens": 0, "codepoints": 0, "utf8_bytes": 0, "document_tokens": []}
    )
    encodings: list[list[int]] = []
    round_trip_failures = []
    unknown_count = 0
    fallback_count = 0
    fallback_documents = 0
    exceptions = []
    identity = tokenizer_identity(tokenizer, Path("."))
    unknown_ids = {
        item["id"] for item in identity["special_tokens"] if "unk" in item["token"].lower()
    }
    common_unk = tokenizer.token_to_id("<unk>")
    if common_unk is not None:
        unknown_ids.add(common_unk)
    max_encoded_id = -1
    for document in documents:
        text = document["text"]
        try:
            encoding = tokenizer.encode(text, add_special_tokens=False)
            ids = [int(token_id) for token_id in encoding.ids]
            encodings.append(ids)
            max_encoded_id = max(max_encoded_id, max(ids, default=-1))
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
            if decoded != text:
                round_trip_failures.append(
                    {
                        "document_id": document["id"],
                        "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "decoded_sha256": hashlib.sha256(decoded.encode("utf-8")).hexdigest(),
                        "input_codepoints": [f"U+{ord(char):04X}" for char in text],
                        "decoded_codepoints": [f"U+{ord(char):04X}" for char in decoded],
                    }
                )
            unknown_count += sum(token_id in unknown_ids for token_id in ids)
            byte_fallbacks = sum(
                bool(BYTE_FALLBACK_PATTERN.match(token)) for token in encoding.tokens
            )
            fallback_count += byte_fallbacks
            fallback_documents += byte_fallbacks > 0
            record = by_stratum[document["stratum"]]
            record["tokens"] += len(ids)
            record["codepoints"] += len(text)
            record["utf8_bytes"] += len(text.encode("utf-8"))
            record["document_tokens"].append(len(ids))
        except Exception as error:
            encodings.append([])
            exceptions.append(
                {
                    "document_id": document["id"],
                    "exception": type(error).__name__,
                    "message": str(error),
                }
            )
    strata = {}
    for name, values in by_stratum.items():
        strata[name] = {
            "documents": len(values["document_tokens"]),
            "tokens": values["tokens"],
            "codepoints": values["codepoints"],
            "utf8_bytes": values["utf8_bytes"],
            "tokens_per_codepoint": rounded(values["tokens"] / values["codepoints"]),
            "tokens_per_utf8_byte": rounded(values["tokens"] / values["utf8_bytes"]),
            "document_tokens_p50": rounded(quantile(values["document_tokens"], 0.50)),
            "document_tokens_p95": rounded(quantile(values["document_tokens"], 0.95)),
            "document_tokens_p99": rounded(quantile(values["document_tokens"], 0.99)),
        }
    total_tokens = sum(len(ids) for ids in encodings)
    total_codepoints = sum(len(document["text"]) for document in documents)
    total_bytes = sum(len(document["text"].encode("utf-8")) for document in documents)
    ids_payload = json.dumps(encodings, separators=(",", ":")).encode()
    return {
        "encodings": encodings,
        "ids_sha256": hashlib.sha256(ids_payload).hexdigest(),
        "compression": {
            "overall": {
                "tokens": total_tokens,
                "codepoints": total_codepoints,
                "utf8_bytes": total_bytes,
                "tokens_per_codepoint": rounded(total_tokens / total_codepoints),
                "tokens_per_utf8_byte": rounded(total_tokens / total_bytes),
            },
            "by_stratum": strata,
        },
        "round_trip": {
            "exact_successes": len(documents) - len(round_trip_failures) - len(exceptions),
            "failures": len(round_trip_failures),
            "failure_examples": round_trip_failures[:10],
        },
        "fallback": {
            "definition": "Tokens whose serialized spelling matches <0xHH>.",
            "explicit_byte_fallback_tokens": fallback_count,
            "documents_with_explicit_byte_fallback": fallback_documents,
            "unknown_token_ids": sorted(unknown_ids),
            "unknown_tokens": unknown_count,
        },
        "exceptions": exceptions,
        "max_encoded_id": max_encoded_id,
    }


def measure_passes(
    tokenizer: Any, documents: list[dict[str, str]], expected_ids_hash: str, passes: int
) -> dict[str, Any]:
    # One complete warm pass is deliberately outside the measured window.
    for document in documents:
        tokenizer.encode(document["text"], add_special_tokens=False)
    pass_results = []
    all_latencies_us: list[float] = []
    for index in range(passes):
        encoded_ids = []
        latencies_us = []
        start = time.perf_counter_ns()
        for document in documents:
            before = time.perf_counter_ns()
            encoding = tokenizer.encode(document["text"], add_special_tokens=False)
            after = time.perf_counter_ns()
            encoded_ids.append([int(token_id) for token_id in encoding.ids])
            latencies_us.append((after - before) / 1000)
        elapsed_seconds = (time.perf_counter_ns() - start) / 1_000_000_000
        ids_hash = hashlib.sha256(
            json.dumps(encoded_ids, separators=(",", ":")).encode()
        ).hexdigest()
        if ids_hash != expected_ids_hash:
            raise ValueError("token IDs changed between the validation and measured passes")
        tokens = sum(len(ids) for ids in encoded_ids)
        codepoints = sum(len(document["text"]) for document in documents)
        utf8_bytes = sum(len(document["text"].encode("utf-8")) for document in documents)
        pass_results.append(
            {
                "pass": index + 1,
                "elapsed_seconds": rounded(elapsed_seconds),
                "documents_per_second": rounded(len(documents) / elapsed_seconds),
                "codepoints_per_second": rounded(codepoints / elapsed_seconds),
                "utf8_bytes_per_second": rounded(utf8_bytes / elapsed_seconds),
                "tokens_per_second": rounded(tokens / elapsed_seconds),
                "document_latency_us_p50": rounded(quantile(latencies_us, 0.50)),
                "document_latency_us_p95": rounded(quantile(latencies_us, 0.95)),
                "document_latency_us_p99": rounded(quantile(latencies_us, 0.99)),
                "ids_sha256": ids_hash,
            }
        )
        all_latencies_us.extend(latencies_us)
    summary = {}
    for field in (
        "documents_per_second",
        "codepoints_per_second",
        "utf8_bytes_per_second",
        "tokens_per_second",
    ):
        values = [result[field] for result in pass_results]
        summary[field] = {
            "median": rounded(statistics.median(values)),
            "min": rounded(min(values)),
            "max": rounded(max(values)),
            "relative_range": rounded((max(values) - min(values)) / statistics.median(values)),
        }
    summary["document_latency_us"] = {
        "p50": rounded(quantile(all_latencies_us, 0.50)),
        "p95": rounded(quantile(all_latencies_us, 0.95)),
        "p99": rounded(quantile(all_latencies_us, 0.99)),
    }
    return {
        "warmup_passes": 1,
        "measured_passes": passes,
        "passes": pass_results,
        "summary": summary,
    }


def fresh_process_probe(
    candidate_id: str, lock_path: Path, corpus_path: Path, cache_root: Path
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--probe-candidate",
        candidate_id,
        "--lock",
        str(lock_path),
        "--corpus",
        str(corpus_path),
        "--cache-root",
        str(cache_root),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def run_probe(args: argparse.Namespace) -> None:
    from tokenizers import Tokenizer

    lock = load_json(args.lock)
    candidate = next(item for item in lock["candidates"] if item["id"] == args.probe_candidate)
    tokenizer_path, _ = verify_candidate_files(candidate, args.cache_root)
    documents, _ = load_fixture(args.corpus)
    before_load_kib = read_vmhwm_kib()
    start = time.perf_counter_ns()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    load_seconds = (time.perf_counter_ns() - start) / 1_000_000_000
    after_load_kib = read_vmhwm_kib()
    start = time.perf_counter_ns()
    tokens = 0
    for document in documents:
        tokens += len(tokenizer.encode(document["text"], add_special_tokens=False).ids)
    encode_seconds = (time.perf_counter_ns() - start) / 1_000_000_000
    print(
        json.dumps(
            {
                "load_seconds": rounded(load_seconds),
                "encode_corpus_seconds": rounded(encode_seconds),
                "tokens": tokens,
                "vmhwm_kib_before_load": before_load_kib,
                "vmhwm_kib_after_load": after_load_kib,
                "vmhwm_kib_after_corpus": read_vmhwm_kib(),
            }
        )
    )


def pareto_frontier(results: list[dict[str, Any]]) -> list[str]:
    eligible = [result for result in results if result.get("hard_gate_pass")]
    frontier = []
    for candidate in eligible:
        candidate_metrics = decision_metrics(candidate)
        dominated = False
        for other in eligible:
            if other["id"] == candidate["id"]:
                continue
            other_metrics = decision_metrics(other)
            no_worse = (
                other_metrics["japanese"] <= candidate_metrics["japanese"]
                and other_metrics["english"] <= candidate_metrics["english"]
                and other_metrics["vocab_parameters"] <= candidate_metrics["vocab_parameters"]
                and other_metrics["bytes_per_second"] >= candidate_metrics["bytes_per_second"]
            )
            strictly_better = other_metrics != candidate_metrics
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate["id"])
    return sorted(frontier)


def decision_metrics(result: dict[str, Any]) -> dict[str, float]:
    strata = result["compression"]["by_stratum"]
    return {
        "japanese": strata["japanese"]["tokens_per_utf8_byte"],
        "english": strata["english"]["tokens_per_utf8_byte"],
        "vocab_parameters": result["cost"]["vocabulary_parameters"],
        "bytes_per_second": result["performance"]["summary"]["utf8_bytes_per_second"]["median"],
    }


def select_winner(results: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = sorted(
        (result for result in results if result.get("hard_gate_pass")),
        key=lambda result: result["identity"]["vocab_size"],
    )
    if not eligible:
        return {"winner": None, "rationale": "No candidate passed every hard gate.", "trace": []}
    baseline = eligible[0]
    selected = baseline
    trace = []
    baseline_metrics = decision_metrics(baseline)
    baseline_min_throughput = baseline["performance"]["summary"]["utf8_bytes_per_second"]["min"]
    for candidate in eligible[1:]:
        metrics = decision_metrics(candidate)
        japanese_gain = 1 - metrics["japanese"] / baseline_metrics["japanese"]
        english_gain = 1 - metrics["english"] / baseline_metrics["english"]
        throughput_within_spread = metrics["bytes_per_second"] >= baseline_min_throughput
        promoted = (
            japanese_gain >= 0.05
            and english_gain >= 0.05
            and max(japanese_gain, english_gain) >= 0.10
            and throughput_within_spread
        )
        trace.append(
            {
                "candidate": candidate["id"],
                "relative_to": baseline["id"],
                "japanese_tokens_per_byte_improvement": rounded(japanese_gain),
                "english_tokens_per_byte_improvement": rounded(english_gain),
                "median_bytes_per_second_within_baseline_range": throughput_within_spread,
                "promoted": promoted,
            }
        )
        if promoted:
            current_score = sum(
                item.get("japanese_tokens_per_byte_improvement", 0)
                + item.get("english_tokens_per_byte_improvement", 0)
                for item in trace
                if item["candidate"] == selected["id"]
            )
            if japanese_gain + english_gain > current_score:
                selected = candidate
    if selected["id"] == baseline["id"]:
        rationale = (
            f"{baseline['id']} is the smallest eligible vocabulary; no larger candidate met "
            "the predeclared bilingual compression and throughput promotion thresholds."
        )
    else:
        rationale = (
            f"{selected['id']} cleared every predeclared larger-vocabulary promotion threshold "
            f"relative to {baseline['id']}."
        )
    return {"winner": selected["id"], "rationale": rationale, "trace": trace}


def format_bytes(value: int) -> str:
    return f"{value / (1024**2):,.2f} MiB"


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# TOK-001 frozen tokenizer comparison",
        "",
        f"Generated from corpus `{report['fixture']['sha256']}` ({report['fixture']['document_count']} documents) and candidate lock `{report['candidate_lock_sha256']}`.",
        "No model weights, model configuration, generation configuration, or chat template was downloaded or loaded.",
        "",
        "## Decision",
        "",
        f"**Selected: `{report['selection']['winner']}`.** {report['selection']['rationale']}",
        "",
        "The hard gates precede the Pareto comparison. A larger vocabulary can replace the smallest eligible candidate only under the promotion rule frozen in `candidates.lock.json`; individual timing passes remain in `comparison.json`.",
        "",
        "## Candidate disposition",
        "",
        "| Candidate | Revision/source | License evidence | Status | Hard gates |",
        "| --- | --- | --- | --- | --- |",
    ]
    lock_by_id = {
        candidate["id"]: candidate for candidate in report["candidate_lock"]["candidates"]
    }
    exclusions = []
    for result in report["candidates"]:
        candidate = lock_by_id[result["id"]]
        license_url = candidate["license"].get("license_file_url") or candidate["license"].get(
            "card_evidence_url"
        )
        status = result["status"]
        gates = "N/A" if status == "excluded" else ("PASS" if result["hard_gate_pass"] else "FAIL")
        lines.append(
            f"| `{result['id']}` | [`{candidate['revision']}`]({candidate['revision_url']}) | [official evidence]({license_url}) | {status} | {gates} |"
        )
        if status == "excluded":
            exclusions.append(f"`{result['id']}` exclusion: {result['exclusion']}")
    lines.extend(["", *exclusions, ""])
    lines.extend(
        [
            "## Compression and sequence-length tail",
            "",
            "| Candidate | Stratum | tok/codepoint | tok/UTF-8 byte | doc tokens p50 | p95 | p99 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    measured = [result for result in report["candidates"] if result["status"] == "measured"]
    for result in measured:
        for stratum, values in result["compression"]["by_stratum"].items():
            lines.append(
                f"| `{result['id']}` | {stratum} | {values['tokens_per_codepoint']:.4f} | {values['tokens_per_utf8_byte']:.4f} | {values['document_tokens_p50']:.1f} | {values['document_tokens_p95']:.1f} | {values['document_tokens_p99']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## Runtime (one warmup, five warm passes)",
            "",
            "| Candidate | docs/s median [min,max] | codepoints/s median | bytes/s median | tokens/s median | latency µs p50/p95/p99 | load s | fresh VmHWM after corpus |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in measured:
        summary = result["performance"]["summary"]
        docs = summary["documents_per_second"]
        latency = summary["document_latency_us"]
        probe = result["fresh_process"]
        lines.append(
            f"| `{result['id']}` | {docs['median']:,.1f} [{docs['min']:,.1f},{docs['max']:,.1f}] | {summary['codepoints_per_second']['median']:,.0f} | {summary['utf8_bytes_per_second']['median']:,.0f} | {summary['tokens_per_second']['median']:,.0f} | {latency['p50']:.1f}/{latency['p95']:.1f}/{latency['p99']:.1f} | {probe['load_seconds']:.4f} | {probe['vmhwm_kib_after_corpus'] / 1024:,.1f} MiB |"
        )
    lines.extend(
        [
            "",
            "## Vocabulary-driven cost (embed 384, batch 64 × sequence 64)",
            "",
            "The conventional model has untied token embeddings and a biased LM head. Training state is estimated as FP32 parameter + gradient + two Adam moments (16 bytes/parameter); checkpoint is model + two Adam moments (12 bytes/parameter). Activation and non-vocabulary model costs are intentionally excluded.",
            "",
            "| Candidate | Vocab | vocab params | FP32 training state | model+Adam checkpoint | BF16/FP32 logits | LM-head MACs/batch | tokenizer files |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in measured:
        cost = result["cost"]
        lines.append(
            f"| `{result['id']}` | {result['identity']['vocab_size']:,} | {cost['vocabulary_parameters']:,} | {format_bytes(cost['fp32_training_state_bytes'])} | {format_bytes(cost['model_plus_adam_checkpoint_bytes'])} | {format_bytes(cost['batch_logits_bf16_bytes'])} / {format_bytes(cost['batch_logits_fp32_bytes'])} | {cost['lm_head_multiply_accumulates_per_batch']:,} | {format_bytes(result['tokenizer_artifact_size_bytes'])} |"
        )
    lines.extend(["", "## Correctness gates", ""])
    for result in measured:
        failed = [name for name, passed in result["hard_gates"].items() if not passed]
        semantic = result["identity"]["semantic_tokens"]
        pipeline = result["identity"]["pipeline"]
        lines.extend(
            [
                f"### `{result['id']}` — {'PASS' if result['hard_gate_pass'] else 'FAIL'}",
                "",
                f"- Token IDs SHA-256: `{result['ids_sha256']}`; vocab/max encoded/max vocab ID: {result['identity']['vocab_size']:,}/{result['max_encoded_id']:,}/{result['identity']['max_vocab_id']:,}.",
                f"- PAD/EOS/BOS/UNK: `{semantic['pad_token']}`, `{semantic['eos_token']}`, `{semantic['bos_token']}`, `{semantic['unk_token']}`.",
                f"- Pipeline: model `{pipeline['model_type']}` (byte_fallback={pipeline['model_byte_fallback']}), normalizers `{pipeline['normalizer_types']}`, pre-tokenizers `{pipeline['pre_tokenizer_types']}`, decoders `{pipeline['decoder_types']}`.",
                f"- Exact round trips: {result['round_trip']['exact_successes']}/{report['fixture']['document_count']}; unknown tokens: {result['fallback']['unknown_tokens']}; explicit `<0xHH>` fallback tokens: {result['fallback']['explicit_byte_fallback_tokens']}; corpus exceptions: {len(result['exceptions'])}.",
                f"- Malformed recipes: {len(result['malformed']['outcomes'])}; all rejected deterministically: {result['malformed']['all_rejected'] and result['malformed']['all_deterministic']}.",
                f"- Failed gates: {', '.join(failed) if failed else 'none'}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Reproduction",
            "",
            "```bash",
            "uv run python src/tokenizer/comparison.py --fetch --fetch-only",
            "uv run python src/tokenizer/comparison.py",
            "uv run pytest tests/test_tokenizer_comparison.py -q",
            "```",
            "",
            f"Environment: `{report['environment']['platform']}`, Python `{report['environment']['python']}`, tokenizers `{report['environment']['tokenizers']}`, machine `{report['environment']['machine']}`. Cache root: `{report['cache_root']}`.",
            "",
            "The comparison is CPU R1 evidence. It does not claim DGX R2 model-step throughput or a final integration verdict; winner packaging and the real streamed-batch/model check remain in TOK-001 phase 2.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    from tokenizers import Tokenizer

    lock = load_json(args.lock)
    documents, fixture_metadata = load_fixture(args.corpus)
    results = []
    for candidate in lock["candidates"]:
        if not candidate["eligible_to_compare"]:
            results.append(
                {"id": candidate["id"], "status": "excluded", "exclusion": candidate["exclusion"]}
            )
            continue
        tokenizer_path, verified_files = verify_candidate_files(candidate, args.cache_root)
        directory = tokenizer_path.parent
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        identity = tokenizer_identity(tokenizer, directory)
        validation = encode_validation(tokenizer, documents)
        malformed = malformed_behavior(tokenizer, args.corpus.with_name("malformed.json"))
        performance = measure_passes(
            tokenizer, documents, validation["ids_sha256"], lock["measurement"]["measured_passes"]
        )
        fresh_process = fresh_process_probe(
            candidate["id"], args.lock, args.corpus, args.cache_root
        )
        hard_gates = {
            "redistribution_evidence": bool(candidate["license"]["redistribution_evidence"]),
            "artifact_integrity": True,
            "offline_load": True,
            "exact_unicode_round_trip": validation["round_trip"]["failures"] == 0,
            "zero_unknown_tokens": validation["fallback"]["unknown_tokens"] == 0,
            "zero_corpus_exceptions": not validation["exceptions"],
            "deterministic_malformed_rejection": malformed["all_rejected"]
            and malformed["all_deterministic"],
            "id_range_agreement": identity["min_vocab_id"] == 0
            and identity["max_vocab_id"] == identity["vocab_size"] - 1
            and validation["max_encoded_id"] < identity["vocab_size"],
        }
        result = {
            "id": candidate["id"],
            "status": "measured",
            "revision": candidate["revision"],
            "verified_files": verified_files,
            "tokenizer_artifact_size_bytes": sum(
                file["size_bytes"]
                for file in verified_files
                if file["role"].startswith("tokenizer")
            ),
            "identity": identity,
            "ids_sha256": validation["ids_sha256"],
            "compression": validation["compression"],
            "round_trip": validation["round_trip"],
            "fallback": validation["fallback"],
            "exceptions": validation["exceptions"],
            "max_encoded_id": validation["max_encoded_id"],
            "malformed": malformed,
            "performance": performance,
            "fresh_process": fresh_process,
            "cost": cost_estimate(identity["vocab_size"], lock["measurement"]),
            "hard_gates": hard_gates,
            "hard_gate_pass": all(hard_gates.values()),
        }
        results.append(result)
    selection = select_winner(results)
    return {
        "schema_version": 1,
        "ticket": "TOK-001",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_lock_sha256": sha256_file(args.lock),
        "candidate_lock": lock,
        "fixture": fixture_metadata,
        "cache_root": str(args.cache_root),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "tokenizers": importlib.metadata.version("tokenizers"),
            "hostname": platform.node(),
            "cpu_count": os.cpu_count(),
        },
        "measurement_definition": {
            **lock["measurement"],
            "quantile_method": "linear interpolation at (n - 1) * q",
            "timer": "time.perf_counter_ns",
            "tokenization": "tokenizers.Tokenizer.encode(text, add_special_tokens=False)",
        },
        "candidates": results,
        "pareto_frontier": pareto_frontier(results),
        "selection": selection,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--fetch", action="store_true", help="fetch only exact allowlisted files")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--probe-candidate", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.probe_candidate:
        run_probe(args)
        return
    lock = load_json(args.lock)
    if args.fetch:
        fetch_allowlisted_files(lock, args.cache_root)
    if args.fetch_only:
        return
    report = run_comparison(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(report, args.output.with_suffix(".md"))
    print(json.dumps(report["selection"], indent=2))


if __name__ == "__main__":
    main()
