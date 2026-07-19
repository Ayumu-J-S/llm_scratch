"""Small, fixed BENCH-001 checkpoint scorers."""

from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn.functional as F

from benchmarks.suite import (
    GENERATED_TOKEN_TRACE_REVISION,
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
    validate_suite_context(sampler, suite)
    results: dict[str, Any] = {}
    for task in suite.tasks:
        if task.name == "jcommonsenseqa":
            results[task.name] = _score_jcommonsenseqa(sampler, task.examples, precision=precision)
        elif task.name == "gsm8k":
            max_new_tokens = int(suite.protocol["gsm8k"]["decoding"]["max_new_tokens"])
            results[task.name] = _score_gsm8k(
                sampler,
                task.examples,
                max_new_tokens=max_new_tokens,
                precision=precision,
            )
        else:
            raise BenchmarkScoringError(f"unsupported task: {task.name}")
    return results


def validate_suite_context(sampler: CheckpointSampler, suite: LoadedSuite) -> dict[str, Any]:
    """Fail cheaply unless the checkpoint can execute the complete fixed protocol.

    GSM8K requires room for the entire declared generation cap so scores do not
    silently become context-length-dependent.  The runner invokes this before
    the complete training-corpus contamination scan; ``score_suite`` invokes it
    again as a defensive boundary for direct callers.
    """

    task_requirements: dict[str, int] = {}
    for task in suite.tasks:
        if task.name == "jcommonsenseqa":
            required = 0
            for example in task.examples:
                prompt, choices, _ = format_jcommonsenseqa(example)
                for choice in choices:
                    joint_ids, _, _ = _joint_continuation_tokens(
                        sampler,
                        prompt=prompt,
                        continuation=choice,
                    )
                    required = max(required, len(joint_ids))
        elif task.name == "gsm8k":
            max_new_tokens = int(suite.protocol["gsm8k"]["decoding"]["max_new_tokens"])
            required = 0
            for example in task.examples:
                prompt, _ = format_gsm8k(example)
                prompt_tokens = sampler.tokenizer.encode(prompt)
                if not prompt_tokens:
                    raise BenchmarkScoringError("benchmark prompt encoded to zero tokens")
                required = max(required, len(prompt_tokens) + max_new_tokens)
        else:
            raise BenchmarkScoringError(f"unsupported task: {task.name}")
        task_requirements[task.name] = required

    required_context = max(task_requirements.values(), default=0)
    if sampler.model.max_len < required_context:
        limiting_task = max(task_requirements, key=task_requirements.__getitem__)
        raise BenchmarkScoringError(
            "checkpoint context is incompatible with the fixed benchmark protocol: "
            f"{limiting_task} requires {task_requirements[limiting_task]} tokens, "
            f"checkpoint provides {sampler.model.max_len}"
        )
    return {
        "checkpoint_context_length": sampler.model.max_len,
        "required_context_length": required_context,
        "task_required_context_lengths": task_requirements,
    }


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
    """Score the exact joint prompt-plus-continuation tokenization at its source span."""

    joint_ids, continuation_start, continuation_ids = _joint_continuation_tokens(
        sampler,
        prompt=prompt,
        continuation=continuation,
    )
    if len(joint_ids) > sampler.model.max_len:
        raise BenchmarkScoringError(
            "benchmark prompt plus continuation exceeds checkpoint context; "
            "the fixed protocol does not truncate"
        )
    input_ids = joint_ids[:-1]
    inputs = torch.tensor([input_ids], dtype=torch.long, device=sampler.device)
    with torch.inference_mode(), autocast_context(sampler.device, precision):
        logits = sampler.model(inputs)[0]
        if not bool(torch.isfinite(logits).all().item()):
            raise BenchmarkScoringError("benchmark choice scoring produced non-finite logits")
        start = continuation_start - 1
        end = len(joint_ids) - 1
        continuation_logits = logits[start:end]
        targets = torch.tensor(continuation_ids, dtype=torch.long, device=sampler.device)
        log_probabilities = F.log_softmax(continuation_logits.float(), dim=-1)
        if not bool(torch.isfinite(log_probabilities).all().item()):
            raise BenchmarkScoringError(
                "benchmark choice scoring produced non-finite log probabilities"
            )
        score = log_probabilities.gather(1, targets.unsqueeze(1)).sum()
        if not bool(torch.isfinite(score).item()):
            raise BenchmarkScoringError("benchmark choice scoring produced a non-finite score")
    return float(score.item()), len(continuation_ids)


def _joint_continuation_tokens(
    sampler: CheckpointSampler,
    *,
    prompt: str,
    continuation: str,
) -> tuple[list[int], int, list[int]]:
    """Tokenize the declared string once and locate its exact continuation span."""

    joint_text = prompt + continuation
    joint_ids, offsets = sampler.tokenizer.encode_with_offsets(joint_text)
    boundary = len(prompt)
    continuation_start: int | None = None
    for index, (start, end) in enumerate(offsets):
        if start < boundary < end:
            raise BenchmarkScoringError(
                "canonical tokenization crosses the declared continuation boundary"
            )
        if end > boundary:
            if start < boundary:
                raise BenchmarkScoringError(
                    "canonical tokenization crosses the declared continuation boundary"
                )
            continuation_start = index
            break
    if continuation_start is None:
        raise BenchmarkScoringError("benchmark continuation encoded to zero scoreable tokens")
    if continuation_start < 1:
        raise BenchmarkScoringError("benchmark prompt encoded to zero scoreable tokens")
    continuation_offsets = offsets[continuation_start:]
    if (
        continuation_offsets[0][0] != boundary
        or continuation_offsets[-1][1] != len(joint_text)
        or any(start < boundary or end <= boundary for start, end in continuation_offsets)
    ):
        raise BenchmarkScoringError(
            "canonical token offsets do not form the complete continuation suffix"
        )
    continuation_ids = joint_ids[continuation_start:]
    return joint_ids, continuation_start, continuation_ids


def _score_gsm8k(
    sampler: CheckpointSampler,
    examples: tuple[BenchmarkExample, ...],
    *,
    max_new_tokens: int,
    precision: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    correct = 0
    valid_format = 0
    for example in examples:
        prompt, reference = format_gsm8k(example)
        result = sampler.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            precision=precision,
        )
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
                "generated_token_ids_hash_revision": GENERATED_TOKEN_TRACE_REVISION,
                "generated_token_ids_sha256": hashlib.sha256(
                    canonical_json_bytes(list(result.generated_token_ids))
                ).hexdigest(),
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
