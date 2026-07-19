"""Small, fixed BENCH-001 checkpoint scorers."""

from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn.functional as F

from benchmarks.suite import (
    INVALID_GSM8K_ANSWER,
    BenchmarkExample,
    LoadedSuite,
    extract_gsm8k_answer,
    format_gsm8k,
    format_jcommonsenseqa,
)
from data.identity import canonical_json_bytes
from generation.sampler import CheckpointSampler
from training.optimization import autocast_context


class BenchmarkScoringError(ValueError):
    """A fixed benchmark protocol cannot be evaluated by this checkpoint."""


def score_suite(sampler: CheckpointSampler, suite: LoadedSuite) -> dict[str, Any]:
    precision = str(sampler.resolved_config.get("training", {}).get("precision", "fp32"))
    results: dict[str, Any] = {}
    for task in suite.tasks:
        if task.name == "jcommonsenseqa":
            results[task.name] = _score_jcommonsenseqa(
                sampler, task.examples, precision=precision
            )
        elif task.name == "gsm8k":
            max_new_tokens = int(suite.protocol["gsm8k"]["decoding"]["max_new_tokens"])
            results[task.name] = _score_gsm8k(
                sampler,
                task.examples,
                max_new_tokens=max_new_tokens,
            )
        else:
            raise BenchmarkScoringError(f"unsupported task: {task.name}")
    return results


def _score_jcommonsenseqa(
    sampler: CheckpointSampler,
    examples: tuple[BenchmarkExample, ...],
    *,
    precision: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    length_correct = 0
    raw_correct = 0
    for example in examples:
        prompt, choices, label = format_jcommonsenseqa(example)
        raw_scores: list[float] = []
        normalized_scores: list[float] = []
        token_counts: list[int] = []
        for choice in choices:
            raw_score, token_count = conditional_log_probability(
                sampler,
                prompt=prompt,
                continuation=choice,
                precision=precision,
            )
            raw_scores.append(raw_score)
            token_counts.append(token_count)
            normalized_scores.append(raw_score / token_count)
        normalized_prediction = _argmax_first(normalized_scores)
        raw_prediction = _argmax_first(raw_scores)
        length_correct += int(normalized_prediction == label)
        raw_correct += int(raw_prediction == label)
        score_identity = {
            "raw_log_probabilities": raw_scores,
            "continuation_token_counts": token_counts,
            "length_normalized_log_probabilities": normalized_scores,
        }
        rows.append(
            {
                "example_id": example.example_id,
                "prediction": normalized_prediction,
                "raw_prediction": raw_prediction,
                "correct": normalized_prediction == label,
                "score_sha256": hashlib.sha256(canonical_json_bytes(score_identity)).hexdigest(),
            }
        )
    total = len(rows)
    return {
        "primary_metric": "length_normalized_accuracy",
        "correct": length_correct,
        "total": total,
        "length_normalized_accuracy": length_correct / total,
        "raw_log_probability_accuracy": raw_correct / total,
        "prediction_trace": rows,
        "prediction_trace_sha256": hashlib.sha256(canonical_json_bytes(rows)).hexdigest(),
    }


def conditional_log_probability(
    sampler: CheckpointSampler,
    *,
    prompt: str,
    continuation: str,
    precision: str,
) -> tuple[float, int]:
    """Score continuation tokens after an explicitly tokenized prompt boundary."""

    prompt_ids = sampler.tokenizer.encode(prompt)
    continuation_ids = sampler.tokenizer.encode(continuation)
    if not prompt_ids:
        raise BenchmarkScoringError("benchmark prompt encoded to zero tokens")
    if not continuation_ids:
        raise BenchmarkScoringError("benchmark continuation encoded to zero tokens")
    if len(prompt_ids) + len(continuation_ids) > sampler.model.max_len:
        raise BenchmarkScoringError(
            "benchmark prompt plus continuation exceeds checkpoint context; "
            "the fixed protocol does not truncate"
        )
    input_ids = prompt_ids + continuation_ids[:-1]
    inputs = torch.tensor([input_ids], dtype=torch.long, device=sampler.device)
    with torch.inference_mode(), autocast_context(sampler.device, precision):
        logits = sampler.model(inputs)[0]
        start = len(prompt_ids) - 1
        end = start + len(continuation_ids)
        continuation_logits = logits[start:end]
        targets = torch.tensor(continuation_ids, dtype=torch.long, device=sampler.device)
        log_probabilities = F.log_softmax(continuation_logits.float(), dim=-1)
        score = log_probabilities.gather(1, targets.unsqueeze(1)).sum()
    return float(score.item()), len(continuation_ids)


def _score_gsm8k(
    sampler: CheckpointSampler,
    examples: tuple[BenchmarkExample, ...],
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    correct = 0
    valid_format = 0
    for example in examples:
        prompt, reference = format_gsm8k(example)
        result = sampler.generate(prompt, max_new_tokens=max_new_tokens)
        prediction = extract_gsm8k_answer(result.completion)
        is_valid = prediction != INVALID_GSM8K_ANSWER
        is_correct = is_valid and prediction == reference
        valid_format += int(is_valid)
        correct += int(is_correct)
        rows.append(
            {
                "example_id": example.example_id,
                "correct": is_correct,
                "valid_answer_format": is_valid,
                "generated_token_count": len(result.generated_token_ids),
                "stop_reason": result.stop_reason,
                "completion_sha256": hashlib.sha256(
                    result.completion.encode("utf-8", errors="strict")
                ).hexdigest(),
            }
        )
    total = len(rows)
    return {
        "primary_metric": "exact_match",
        "correct": correct,
        "total": total,
        "exact_match": correct / total,
        "valid_answer_format": valid_format,
        "valid_answer_format_rate": valid_format / total,
        "prediction_trace": rows,
        "prediction_trace_sha256": hashlib.sha256(canonical_json_bytes(rows)).hexdigest(),
    }


def _argmax_first(values: list[float]) -> int:
    if not values:
        raise BenchmarkScoringError("cannot select from an empty score vector")
    return max(range(len(values)), key=values.__getitem__)
