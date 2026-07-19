"""Prepare, blind, and unblind the small HUMAN-001 continuation study."""

from __future__ import annotations

import copy
import hashlib
import hmac
import itertools
import json
import os
import secrets
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from data.identity import canonical_json_bytes
from generation.sampler import CheckpointSampler
from human_evaluation.schema import (
    MAX_NEW_TOKENS,
    MINIMUM_RELATIVE_TARGET_TOKEN_GAP,
    PRIVATE_MAPPING_SCHEMA_VERSION,
    PUBLIC_BUNDLE_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    RUBRIC,
    SCORE_SCHEMA_VERSION,
    TEMPERATURE,
    TOP_K,
    EvaluationSchemaError,
    PromptSet,
    load_prompt_set,
)
from training.checkpoint import LoadedCheckpoint, load_checkpoint_for_generation


_FORBIDDEN_OUTPUT_PARTS = frozenset(
    {
        "artifacts",
        "benchmark_cache",
        "cache",
        "checkpoints",
        "data",
        "manifests",
        "runs",
        "stream_loader_cache",
        "training",
    }
)
_CHECKPOINT_SHARED_IDENTITY_FIELDS = (
    "schema_version",
    "experiment_id",
    "git_sha",
    "lock_sha256",
    "config_sha256",
    "model_config",
    "tokenizer_fingerprint",
    "data_fingerprints",
)
_PUBLIC_FIELDS = {
    "schema_version",
    "study_id",
    "prompt_set_version",
    "evaluation_type",
    "sampling",
    "rubric",
    "items",
}
_PUBLIC_ITEM_FIELDS = {"item_id", "language", "prompt", "candidates"}
_SCORE_FIELDS = {"schema_version", "study_id", "reviewer_id", "ratings"}
_RATING_FIELDS = {"item_id", "candidate_a", "candidate_b", "preference", "comment"}
_DIMENSIONS = tuple(dimension["id"] for dimension in RUBRIC["dimensions"])


class HumanEvaluationError(EvaluationSchemaError):
    """The study cannot be prepared or unblinded without violating its contract."""


def create_blinding_key(path: str | Path) -> Path:
    """Create a new 256-bit blinding key with owner-only permissions."""

    key_path = Path(path).expanduser().resolve()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise HumanEvaluationError(
            f"refusing to replace existing blinding key: {key_path}"
        ) from error
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(secrets.token_bytes(32))
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        key_path.unlink(missing_ok=True)
        raise
    return key_path


def prepare_evaluation(
    *,
    prompt_set_path: str | Path,
    workspace_dir: str | Path,
    blinding_key_path: str | Path,
    earlier_checkpoint: str | Path,
    later_checkpoint: str | Path,
    generation_seed: int,
    device: str = "cpu",
) -> dict[str, Path]:
    """Generate a balanced blinded bundle and authenticated private mapping."""

    seed = _nonnegative_int(generation_seed, "generation_seed")
    if device not in {"cpu", "cuda"}:
        raise HumanEvaluationError("device must be explicitly cpu or cuda")
    workspace, public_path, mapping_path = _study_paths(workspace_dir)
    key_path, key = _read_key(blinding_key_path)
    _require_key_outside_workspace(key_path, workspace)
    prompt_path = Path(prompt_set_path).resolve()
    prompt_set = load_prompt_set(prompt_path)
    _require_prompt_asset_isolated(prompt_path)

    candidates = _load_checkpoint_candidates(earlier_checkpoint, later_checkpoint)
    study_id = _blind_id(
        key,
        "study",
        {
            "prompt_set_version": prompt_set.version,
            "prompt_set_sha256": _sha256_file(prompt_path),
            "generation_seed": seed,
            "checkpoint_pair": [
                {
                    "slot": slot,
                    "sha256": candidates[slot]["sha256"],
                    "target_tokens": candidates[slot]["target_tokens"],
                    "optimizer_step": candidates[slot]["optimizer_step"],
                    "experiment_id": candidates[slot]["experiment_id"],
                }
                for slot in ("earlier", "later")
            ],
        },
    )
    assignments = _balanced_assignments(prompt_set, key=key, study_id=study_id)

    generation_seeds = {
        prompt.id: _derived_seed(key, seed, prompt.id) for prompt in prompt_set.prompts
    }
    completions_by_slot: dict[str, dict[str, str]] = {}
    # One sampler at a time avoids doubling model residency on the target UMA machine.
    for slot in ("earlier", "later"):
        sampler = CheckpointSampler.from_checkpoint(candidates[slot]["path"], device=device)
        completions_by_slot[slot] = {
            prompt.id: sampler.generate(
                prompt.text,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_k=TOP_K,
                seed=generation_seeds[prompt.id],
            ).completion
            for prompt in prompt_set.prompts
        }
        del sampler

    randomized_prompts = sorted(
        prompt_set.prompts,
        key=lambda prompt: hmac.new(
            key, f"item-order:{study_id}:{prompt.id}".encode(), hashlib.sha256
        ).digest(),
    )
    public_items: list[dict[str, Any]] = []
    private_assignments: list[dict[str, Any]] = []
    for prompt in randomized_prompts:
        item_id = _blind_id(key, "item", {"study_id": study_id, "prompt_id": prompt.id})
        generation_item_seed = generation_seeds[prompt.id]
        completions: dict[str, str] = {}
        private_candidates: dict[str, str] = {}
        for label, slot in assignments[prompt.id].items():
            completions[label] = completions_by_slot[slot][prompt.id]
            private_candidates[label] = slot
        public_items.append(
            {
                "item_id": item_id,
                "language": prompt.language,
                "prompt": prompt.text,
                "candidates": {"A": completions["A"], "B": completions["B"]},
            }
        )
        private_assignments.append(
            {
                "item_id": item_id,
                "prompt_id": prompt.id,
                "generation_seed": generation_item_seed,
                "candidates": private_candidates,
            }
        )

    public_bundle = {
        "schema_version": PUBLIC_BUNDLE_SCHEMA_VERSION,
        "study_id": study_id,
        "prompt_set_version": prompt_set.version,
        "evaluation_type": "base-model-continuation",
        "sampling": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
        },
        "rubric": copy.deepcopy(RUBRIC),
        "items": public_items,
    }
    _validate_public_bundle(public_bundle)
    public_bytes = _json_bytes(public_bundle)
    _reject_private_identity_leak(public_bytes, candidates)
    private_payload = {
        "study_id": study_id,
        "prompt_set": {
            "version": prompt_set.version,
            "path": str(prompt_path),
            "sha256": _sha256_file(prompt_path),
        },
        "generation": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
            "seed": seed,
            "device": device,
        },
        "checkpoints": [candidates["earlier"], candidates["later"]],
        "assignments": private_assignments,
        "public_bundle_sha256": hashlib.sha256(public_bytes).hexdigest(),
    }
    private_mapping = {
        "schema_version": PRIVATE_MAPPING_SCHEMA_VERSION,
        "payload": private_payload,
        "authentication": {
            "algorithm": "HMAC-SHA256",
            "tag": hmac.new(key, canonical_json_bytes(private_payload), hashlib.sha256).hexdigest(),
        },
    }
    _prepare_workspace_directories(workspace)
    _write_new_or_identical(public_path, public_bytes, mode=0o644)
    _write_new_or_identical(mapping_path, _json_bytes(private_mapping), mode=0o600)
    return {"public_bundle": public_path, "private_mapping": mapping_path}


def import_scores(
    *,
    workspace_dir: str | Path,
    blinding_key_path: str | Path,
    score_paths: Sequence[str | Path],
) -> Path:
    """Validate human score files, compute agreement, and unblind privately."""

    workspace, public_path, mapping_path = _study_paths(workspace_dir)
    _require_workspace_directories(workspace)
    key_path, key = _read_key(blinding_key_path)
    _require_key_outside_workspace(key_path, workspace)
    if len(score_paths) < 2:
        raise HumanEvaluationError("at least two human score files are required")
    for score_path in score_paths:
        _require_review_path(Path(score_path), workspace)

    _require_public_file(public_path, "public bundle")
    public_bundle = _read_json_object(public_path, "public bundle")
    _validate_public_bundle(public_bundle)
    _require_private_file(mapping_path, "private mapping")
    mapping = _read_json_object(mapping_path, "private mapping")
    private_payload = _authenticate_private_mapping(mapping, key)
    _validate_private_payload(private_payload, public_bundle)
    expected_public_hash = private_payload.get("public_bundle_sha256")
    if expected_public_hash != hashlib.sha256(_json_bytes(public_bundle)).hexdigest():
        raise HumanEvaluationError("public bundle does not match the authenticated private mapping")
    if private_payload.get("study_id") != public_bundle["study_id"]:
        raise HumanEvaluationError("public and private study IDs differ")

    item_ids = [item["item_id"] for item in public_bundle["items"]]
    scores = [
        _load_score_file(path, study_id=public_bundle["study_id"], item_ids=item_ids)
        for path in score_paths
    ]
    normalized_reviewer_ids = [score["reviewer_id"].casefold() for score in scores]
    if len(set(normalized_reviewer_ids)) != len(normalized_reviewer_ids):
        raise HumanEvaluationError("score files must come from distinct human reviewers")

    assignments = {
        assignment["item_id"]: assignment for assignment in private_payload["assignments"]
    }
    checkpoint_by_slot = {
        checkpoint["slot"]: checkpoint for checkpoint in private_payload["checkpoints"]
    }
    unblinded: list[dict[str, Any]] = []
    for score in scores:
        for rating in score["ratings"]:
            assignment = assignments[rating["item_id"]]
            a_slot = assignment["candidates"]["A"]
            b_slot = assignment["candidates"]["B"]
            preference = rating["preference"]
            unblinded.append(
                {
                    "reviewer_id": score["reviewer_id"],
                    "item_id": rating["item_id"],
                    "prompt_id": assignment["prompt_id"],
                    "ratings": {
                        a_slot: rating["candidate_a"],
                        b_slot: rating["candidate_b"],
                    },
                    "preference": "tie"
                    if preference == "tie"
                    else assignment["candidates"][preference],
                    "comment": rating["comment"],
                }
            )

    score_files = [
        {"path": str(Path(path).resolve()), "sha256": _sha256_file(path)} for path in score_paths
    ]
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "study_id": public_bundle["study_id"],
        "reviewers": [score["reviewer_id"] for score in scores],
        "score_files": score_files,
        "checkpoints": list(checkpoint_by_slot.values()),
        "agreement": _agreement(scores),
        "checkpoint_summary": _checkpoint_summary(unblinded),
        "unblinded_ratings": unblinded,
        "research_integrity": {
            "training_reuse_permitted": False,
            "contains_human_scores": True,
            "namespace": "human-evaluation/private",
        },
    }
    result_identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "study_id": public_bundle["study_id"],
                "score_sha256": [record["sha256"] for record in score_files],
            }
        )
    ).hexdigest()[:16]
    result_path = workspace / "private" / f"result-{result_identity}.json"
    _write_new_or_identical(result_path, _json_bytes(result), mode=0o600)
    return result_path


def run_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch one Hydra-composed HUMAN-001 action."""

    action = config.get("action")
    key_path = _required_path(config.get("blinding_key_path"), "blinding_key_path")
    if action == "create_key":
        return {"blinding_key": str(create_blinding_key(key_path))}
    workspace = _required_path(config.get("workspace_dir"), "workspace_dir")
    if action == "prepare":
        checkpoints = _required_mapping(config.get("checkpoints"), "checkpoints")
        generation = _required_mapping(config.get("generation"), "generation")
        expected_generation = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
        }
        for field, expected in expected_generation.items():
            if generation.get(field) != expected:
                raise HumanEvaluationError(
                    f"generation.{field} is fixed at {expected!r} for HUMAN-001"
                )
        device = _nonempty_string(config.get("device"), "device")
        if device not in {"cpu", "cuda"}:
            raise HumanEvaluationError("device must be explicitly cpu or cuda")
        paths = prepare_evaluation(
            prompt_set_path=_required_path(config.get("prompt_set_path"), "prompt_set_path"),
            workspace_dir=workspace,
            blinding_key_path=key_path,
            earlier_checkpoint=_required_path(checkpoints.get("earlier"), "checkpoints.earlier"),
            later_checkpoint=_required_path(checkpoints.get("later"), "checkpoints.later"),
            generation_seed=_nonnegative_int(generation.get("seed"), "generation.seed"),
            device=device,
        )
        return {name: str(path) for name, path in paths.items()}
    if action == "import_scores":
        scores = _required_mapping(config.get("scores"), "scores")
        paths = scores.get("paths")
        if not isinstance(paths, list):
            raise HumanEvaluationError("scores.paths must be a list")
        result = import_scores(
            workspace_dir=workspace,
            blinding_key_path=key_path,
            score_paths=paths,
        )
        return {"private_result": str(result)}
    raise HumanEvaluationError("action must be create_key, prepare, or import_scores")


def _load_checkpoint_candidates(
    earlier_path: str | Path,
    later_path: str | Path,
    *,
    loader: Callable[[str | Path], LoadedCheckpoint] | None = None,
) -> dict[str, dict[str, Any]]:
    if loader is None:
        loader = load_checkpoint_for_generation
    candidates: dict[str, dict[str, Any]] = {}
    identities: dict[str, Mapping[str, Any]] = {}
    for slot, path in (("earlier", earlier_path), ("later", later_path)):
        checkpoint = loader(path)
        payload = checkpoint.payload
        state = _required_mapping(payload.get("state"), f"{slot} checkpoint state")
        counters = _required_mapping(state.get("counters"), f"{slot} checkpoint counters")
        identity = _required_mapping(payload.get("identity"), f"{slot} checkpoint identity")
        identities[slot] = copy.deepcopy(dict(identity))
        for field in ("experiment_id", "git_sha", "lock_sha256"):
            _nonempty_string(identity.get(field), f"{slot} checkpoint identity.{field}")
        candidates[slot] = {
            "slot": slot,
            "path": str(Path(checkpoint.physical_identity["path"]).resolve()),
            "sha256": _sha256_string(
                checkpoint.physical_identity.get("sha256"), f"{slot} checkpoint sha256"
            ),
            "size_bytes": _positive_int(
                checkpoint.physical_identity.get("size_bytes"), f"{slot} checkpoint size_bytes"
            ),
            "kind": _nonempty_string(payload.get("kind"), f"{slot} checkpoint kind"),
            "experiment_id": identity["experiment_id"],
            "git_sha": identity["git_sha"],
            "lock_sha256": identity["lock_sha256"],
            "config_sha256": _sha256_string(
                identity.get("config_sha256"), f"{slot} checkpoint config_sha256"
            ),
            "tokenizer_fingerprint": _sha256_string(
                identity.get("tokenizer_fingerprint"),
                f"{slot} checkpoint tokenizer_fingerprint",
            ),
            "optimizer_step": _nonnegative_int(
                counters.get("optimizer_step"), f"{slot} optimizer_step"
            ),
            "target_tokens": _positive_int(counters.get("target_tokens"), f"{slot} target_tokens"),
        }
        del checkpoint, payload, state, counters, identity
    if candidates["earlier"]["sha256"] == candidates["later"]["sha256"]:
        raise HumanEvaluationError("the two checkpoints must be physically distinct")
    for field in _CHECKPOINT_SHARED_IDENTITY_FIELDS:
        if identities["earlier"].get(field) != identities["later"].get(field):
            raise HumanEvaluationError(
                f"checkpoints must share one run/config identity; {field} differs"
            )
    earlier_tokens = candidates["earlier"]["target_tokens"]
    later_tokens = candidates["later"]["target_tokens"]
    if later_tokens <= earlier_tokens:
        raise HumanEvaluationError("later checkpoint must have a larger target-token counter")
    gap_fraction = (later_tokens - earlier_tokens) / later_tokens
    if gap_fraction < MINIMUM_RELATIVE_TARGET_TOKEN_GAP:
        raise HumanEvaluationError(
            "checkpoints must be separated by at least 25% of the later target-token count"
        )
    if candidates["later"]["optimizer_step"] <= candidates["earlier"]["optimizer_step"]:
        raise HumanEvaluationError("later checkpoint must have a larger optimizer-step counter")
    return candidates


def _balanced_assignments(
    prompt_set: PromptSet, *, key: bytes, study_id: str
) -> dict[str, dict[str, str]]:
    earlier_as_a: set[str] = set()
    for language in ("ja", "en"):
        ranked = sorted(
            (prompt for prompt in prompt_set.prompts if prompt.language == language),
            key=lambda prompt: hmac.new(
                key, f"order:{study_id}:{prompt.id}".encode(), hashlib.sha256
            ).digest(),
        )
        earlier_as_a.update(prompt.id for prompt in ranked[: len(ranked) // 2])
    return {
        prompt.id: (
            {"A": "earlier", "B": "later"}
            if prompt.id in earlier_as_a
            else {"A": "later", "B": "earlier"}
        )
        for prompt in prompt_set.prompts
    }


def _validate_public_bundle(bundle: dict[str, Any]) -> None:
    _exact_fields(bundle, _PUBLIC_FIELDS, "public bundle")
    if bundle["schema_version"] != PUBLIC_BUNDLE_SCHEMA_VERSION:
        raise HumanEvaluationError("public bundle schema_version is unsupported")
    _nonempty_string(bundle["study_id"], "study_id")
    _nonempty_string(bundle["prompt_set_version"], "prompt_set_version")
    if bundle["evaluation_type"] != "base-model-continuation":
        raise HumanEvaluationError("evaluation_type must be base-model-continuation")
    if bundle["sampling"] != {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
    }:
        raise HumanEvaluationError("public bundle sampling contract differs from HUMAN-001")
    if bundle["rubric"] != RUBRIC:
        raise HumanEvaluationError("public bundle rubric differs from HUMAN-001")
    items = bundle["items"]
    if not isinstance(items, list) or len(items) != 8:
        raise HumanEvaluationError("public bundle must contain exactly eight items")
    item_ids: set[str] = set()
    language_counts = {"ja": 0, "en": 0}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise HumanEvaluationError(f"public item {index} must be an object")
        _exact_fields(item, _PUBLIC_ITEM_FIELDS, f"public item {index}")
        item_id = _nonempty_string(item["item_id"], f"public item {index}.item_id")
        if item_id in item_ids:
            raise HumanEvaluationError("public item IDs must be unique")
        item_ids.add(item_id)
        language = item["language"]
        if language not in language_counts:
            raise HumanEvaluationError("public item language must be ja or en")
        language_counts[language] += 1
        _nonempty_string(item["prompt"], f"public item {index}.prompt")
        candidates = item["candidates"]
        if not isinstance(candidates, dict):
            raise HumanEvaluationError("public item candidates must be an object")
        _exact_fields(candidates, {"A", "B"}, f"public item {index}.candidates")
        for label in ("A", "B"):
            if not isinstance(candidates[label], str):
                raise HumanEvaluationError("continuations must be strings")
    if language_counts != {"ja": 4, "en": 4}:
        raise HumanEvaluationError("public bundle must retain the 4/4 language balance")


def _load_score_file(path: str | Path, *, study_id: str, item_ids: Sequence[str]) -> dict[str, Any]:
    score = _read_json_object(path, "score file")
    _exact_fields(score, _SCORE_FIELDS, "score file")
    if score["schema_version"] != SCORE_SCHEMA_VERSION:
        raise HumanEvaluationError("score schema_version is unsupported")
    if score["study_id"] != study_id:
        raise HumanEvaluationError("score file study_id differs from the public bundle")
    reviewer_id = _nonempty_string(score["reviewer_id"], "reviewer_id")
    if reviewer_id != reviewer_id.strip():
        raise HumanEvaluationError("reviewer_id must not have surrounding whitespace")
    ratings = score["ratings"]
    if not isinstance(ratings, list) or len(ratings) != len(item_ids):
        raise HumanEvaluationError("each reviewer must rate every public item exactly once")
    rating_ids: list[str] = []
    for index, rating in enumerate(ratings):
        if not isinstance(rating, dict):
            raise HumanEvaluationError(f"rating {index} must be an object")
        _exact_fields(rating, _RATING_FIELDS, f"rating {index}")
        item_id = _nonempty_string(rating["item_id"], f"rating {index}.item_id")
        rating_ids.append(item_id)
        _validate_candidate_score(rating["candidate_a"], f"rating {index}.candidate_a")
        _validate_candidate_score(rating["candidate_b"], f"rating {index}.candidate_b")
        if rating["preference"] not in {"A", "B", "tie"}:
            raise HumanEvaluationError("rating preference must be A, B, or tie")
        if not isinstance(rating["comment"], str):
            raise HumanEvaluationError("rating comment must be a string")
    if len(set(rating_ids)) != len(rating_ids) or set(rating_ids) != set(item_ids):
        raise HumanEvaluationError("score ratings must match every public item exactly once")
    return score


def _validate_candidate_score(score: Any, label: str) -> None:
    if not isinstance(score, dict):
        raise HumanEvaluationError(f"{label} must be an object")
    _exact_fields(score, set(_DIMENSIONS), label)
    for dimension, value in score.items():
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
            raise HumanEvaluationError(f"{label}.{dimension} must be an integer from 1 to 5")


def _agreement(scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    preferences: list[bool] = []
    exact_ratings: list[bool] = []
    absolute_differences: list[int] = []
    for first, second in itertools.combinations(scores, 2):
        first_by_id = {rating["item_id"]: rating for rating in first["ratings"]}
        second_by_id = {rating["item_id"]: rating for rating in second["ratings"]}
        for item_id, first_rating in first_by_id.items():
            second_rating = second_by_id[item_id]
            preferences.append(first_rating["preference"] == second_rating["preference"])
            for candidate_field in ("candidate_a", "candidate_b"):
                for dimension in _DIMENSIONS:
                    first_value = first_rating[candidate_field][dimension]
                    second_value = second_rating[candidate_field][dimension]
                    exact_ratings.append(first_value == second_value)
                    absolute_differences.append(abs(first_value - second_value))
    return {
        "reviewer_count": len(scores),
        "reviewer_pair_count": len(scores) * (len(scores) - 1) // 2,
        "preference_comparisons": len(preferences),
        "preference_exact_fraction": sum(preferences) / len(preferences),
        "rating_comparisons": len(exact_ratings),
        "rating_exact_fraction": sum(exact_ratings) / len(exact_ratings),
        "rating_mean_absolute_difference": sum(absolute_differences) / len(absolute_differences),
    }


def _checkpoint_summary(unblinded: Sequence[dict[str, Any]]) -> dict[str, Any]:
    accumulators = {
        slot: {
            "rating_count": 0,
            "dimension_sums": {dimension: 0 for dimension in _DIMENSIONS},
            "preferred": 0,
            "ties": 0,
        }
        for slot in ("earlier", "later")
    }
    for rating in unblinded:
        for slot, values in rating["ratings"].items():
            accumulator = accumulators[slot]
            accumulator["rating_count"] += 1
            for dimension in _DIMENSIONS:
                accumulator["dimension_sums"][dimension] += values[dimension]
        preference = rating["preference"]
        if preference == "tie":
            for accumulator in accumulators.values():
                accumulator["ties"] += 1
        else:
            accumulators[preference]["preferred"] += 1
    result: dict[str, Any] = {}
    for slot, accumulator in accumulators.items():
        denominator = accumulator["rating_count"]
        result[slot] = {
            "rating_count": denominator,
            "dimension_means": {
                dimension: accumulator["dimension_sums"][dimension] / denominator
                for dimension in _DIMENSIONS
            },
            "preferred": accumulator["preferred"],
            "ties": accumulator["ties"],
        }
    return result


def _authenticate_private_mapping(mapping: dict[str, Any], key: bytes) -> dict[str, Any]:
    _exact_fields(mapping, {"schema_version", "payload", "authentication"}, "private mapping")
    if mapping["schema_version"] != PRIVATE_MAPPING_SCHEMA_VERSION:
        raise HumanEvaluationError("private mapping schema_version is unsupported")
    payload = mapping["payload"]
    authentication = mapping["authentication"]
    if not isinstance(payload, dict) or not isinstance(authentication, dict):
        raise HumanEvaluationError("private mapping payload/authentication is invalid")
    _exact_fields(authentication, {"algorithm", "tag"}, "private mapping authentication")
    if authentication["algorithm"] != "HMAC-SHA256":
        raise HumanEvaluationError("private mapping authentication algorithm is unsupported")
    tag = authentication["tag"]
    if not isinstance(tag, str):
        raise HumanEvaluationError("private mapping authentication tag must be a string")
    expected = hmac.new(key, canonical_json_bytes(payload), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(tag, expected):
        raise HumanEvaluationError("private mapping authentication failed")
    return payload


def _validate_private_payload(payload: dict[str, Any], public_bundle: dict[str, Any]) -> None:
    _exact_fields(
        payload,
        {
            "study_id",
            "prompt_set",
            "generation",
            "checkpoints",
            "assignments",
            "public_bundle_sha256",
        },
        "private mapping payload",
    )
    if payload["study_id"] != public_bundle["study_id"]:
        raise HumanEvaluationError("public and private study IDs differ")
    prompt_set = payload["prompt_set"]
    if not isinstance(prompt_set, dict):
        raise HumanEvaluationError("private prompt_set must be an object")
    _exact_fields(prompt_set, {"version", "path", "sha256"}, "private prompt_set")
    if prompt_set["version"] != public_bundle["prompt_set_version"]:
        raise HumanEvaluationError("public and private prompt set versions differ")
    _nonempty_string(prompt_set["path"], "private prompt_set.path")
    _sha256_string(prompt_set["sha256"], "private prompt_set.sha256")

    generation = payload["generation"]
    if not isinstance(generation, dict):
        raise HumanEvaluationError("private generation contract must be an object")
    _exact_fields(
        generation,
        {"max_new_tokens", "temperature", "top_k", "seed", "device"},
        "private generation contract",
    )
    if {
        field: generation[field] for field in ("max_new_tokens", "temperature", "top_k")
    } != public_bundle["sampling"]:
        raise HumanEvaluationError("public and private generation contracts differ")
    _nonnegative_int(generation["seed"], "private generation.seed")
    if generation["device"] not in {"cpu", "cuda"}:
        raise HumanEvaluationError("private generation.device must be cpu or cuda")

    checkpoints = payload["checkpoints"]
    if not isinstance(checkpoints, list) or len(checkpoints) != 2:
        raise HumanEvaluationError("private mapping must contain exactly two checkpoints")
    expected_checkpoint_fields = {
        "slot",
        "path",
        "sha256",
        "size_bytes",
        "kind",
        "experiment_id",
        "git_sha",
        "lock_sha256",
        "config_sha256",
        "tokenizer_fingerprint",
        "optimizer_step",
        "target_tokens",
    }
    slots: set[str] = set()
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict):
            raise HumanEvaluationError("private checkpoint must be an object")
        _exact_fields(checkpoint, expected_checkpoint_fields, "private checkpoint")
        slot = checkpoint["slot"]
        if slot not in {"earlier", "later"}:
            raise HumanEvaluationError("private checkpoint slot must be earlier or later")
        slots.add(slot)
        _nonempty_string(checkpoint["path"], f"private checkpoint {slot}.path")
        for field in ("sha256", "lock_sha256", "config_sha256", "tokenizer_fingerprint"):
            _sha256_string(checkpoint[field], f"private checkpoint {slot}.{field}")
        _positive_int(checkpoint["size_bytes"], f"private checkpoint {slot}.size_bytes")
        _nonempty_string(checkpoint["kind"], f"private checkpoint {slot}.kind")
        _nonempty_string(checkpoint["experiment_id"], f"private checkpoint {slot}.experiment_id")
        _nonempty_string(checkpoint["git_sha"], f"private checkpoint {slot}.git_sha")
        _nonnegative_int(checkpoint["optimizer_step"], f"private checkpoint {slot}.optimizer_step")
        _positive_int(checkpoint["target_tokens"], f"private checkpoint {slot}.target_tokens")
    if slots != {"earlier", "later"}:
        raise HumanEvaluationError("private mapping must contain earlier and later checkpoints")

    assignments = payload["assignments"]
    public_item_ids = {item["item_id"] for item in public_bundle["items"]}
    if not isinstance(assignments, list) or len(assignments) != len(public_item_ids):
        raise HumanEvaluationError("private assignments must cover every public item")
    assignment_ids: set[str] = set()
    prompt_ids: set[str] = set()
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise HumanEvaluationError("private assignment must be an object")
        _exact_fields(
            assignment,
            {"item_id", "prompt_id", "generation_seed", "candidates"},
            "private assignment",
        )
        assignment_ids.add(_nonempty_string(assignment["item_id"], "assignment.item_id"))
        prompt_ids.add(_nonempty_string(assignment["prompt_id"], "assignment.prompt_id"))
        _nonnegative_int(assignment["generation_seed"], "assignment.generation_seed")
        candidates = assignment["candidates"]
        if not isinstance(candidates, dict):
            raise HumanEvaluationError("assignment.candidates must be an object")
        _exact_fields(candidates, {"A", "B"}, "assignment.candidates")
        if set(candidates.values()) != {"earlier", "later"}:
            raise HumanEvaluationError("assignment must map A/B to different checkpoints")
    if assignment_ids != public_item_ids or len(prompt_ids) != len(assignments):
        raise HumanEvaluationError("private assignments do not map one-to-one to public items")
    _sha256_string(payload["public_bundle_sha256"], "private public_bundle_sha256")


def _reject_private_identity_leak(
    public_bytes: bytes, candidates: Mapping[str, Mapping[str, Any]]
) -> None:
    public_text = public_bytes.decode("utf-8")
    private_values: set[str] = set()
    for candidate in candidates.values():
        for field in (
            "path",
            "sha256",
            "experiment_id",
            "git_sha",
            "lock_sha256",
            "config_sha256",
            "tokenizer_fingerprint",
        ):
            value = candidate.get(field)
            if isinstance(value, str) and value:
                private_values.add(value)
    leaked = sorted(value for value in private_values if value in public_text)
    if leaked:
        raise HumanEvaluationError("generated public bundle contains a private checkpoint identity")


def _study_paths(workspace_dir: str | Path) -> tuple[Path, Path, Path]:
    raw_workspace = Path(workspace_dir).expanduser().absolute()
    if raw_workspace.is_symlink():
        raise HumanEvaluationError("human-evaluation workspace must not be a symlink")
    workspace = raw_workspace.resolve()
    _require_isolated_workspace(workspace)
    return workspace, workspace / "public" / "bundle.json", workspace / "private" / "mapping.json"


def _prepare_workspace_directories(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
    public_dir = workspace / "public"
    private_dir = workspace / "private"
    public_dir.mkdir(exist_ok=True, mode=0o755)
    private_dir.mkdir(exist_ok=True, mode=0o700)
    for path, mode in ((workspace, 0o700), (public_dir, 0o755), (private_dir, 0o700)):
        if path.is_symlink() or not path.is_dir():
            raise HumanEvaluationError(f"study namespace must be a real directory: {path}")
        os.chmod(path, mode)


def _require_workspace_directories(workspace: Path) -> None:
    for path, mode in (
        (workspace, 0o700),
        (workspace / "public", 0o755),
        (workspace / "private", 0o700),
    ):
        if path.is_symlink() or not path.is_dir():
            raise HumanEvaluationError(f"study namespace must be a real directory: {path}")
        if stat.S_IMODE(path.stat().st_mode) != mode:
            raise HumanEvaluationError(f"study directory has wrong permissions: {path}")


def _require_isolated_workspace(workspace: Path) -> None:
    lowered = {part.casefold() for part in workspace.parts}
    if "human-evaluation" not in lowered:
        raise HumanEvaluationError(
            "workspace path must include a dedicated human-evaluation directory"
        )
    overlap = lowered.intersection(_FORBIDDEN_OUTPUT_PARTS)
    if overlap:
        raise HumanEvaluationError(
            f"human-evaluation workspace overlaps forbidden training/cache namespace: {sorted(overlap)}"
        )


def _require_prompt_asset_isolated(path: Path) -> None:
    parts = [part.casefold() for part in path.parts]
    if len(parts) < 3 or parts[-3:-1] != ["evaluation", "human"]:
        raise HumanEvaluationError(
            "prompt set must live under evaluation/human, outside training data"
        )
    if any(part in _FORBIDDEN_OUTPUT_PARTS for part in parts[:-3]):
        raise HumanEvaluationError("prompt set path overlaps a training/cache namespace")


def _require_review_path(path: Path, workspace: Path) -> None:
    raw_path = path.expanduser().absolute()
    if raw_path.is_symlink():
        raise HumanEvaluationError("score file must be a regular non-symlink file")
    resolved = raw_path.resolve()
    raw_review_root = workspace / "reviews"
    if raw_review_root.is_symlink():
        raise HumanEvaluationError("study reviews directory must not be a symlink")
    review_root = raw_review_root.resolve()
    if not resolved.is_relative_to(review_root):
        raise HumanEvaluationError("score files must live under the study's reviews directory")
    if not review_root.is_dir() or stat.S_IMODE(review_root.stat().st_mode) != 0o700:
        raise HumanEvaluationError("study reviews directory permissions must be exactly 0700")
    _require_private_file(resolved, "score file")


def _require_private_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise HumanEvaluationError(f"cannot read {label}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise HumanEvaluationError(f"{label} must be a regular non-symlink file")
    if metadata.st_nlink != 1:
        raise HumanEvaluationError(f"{label} must not be hardlinked")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise HumanEvaluationError(f"{label} permissions must be exactly 0600")


def _require_public_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise HumanEvaluationError(f"cannot read {label}: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise HumanEvaluationError(f"{label} must be a regular non-symlink file")
    if metadata.st_nlink != 1:
        raise HumanEvaluationError(f"{label} must not be hardlinked")
    if stat.S_IMODE(metadata.st_mode) != 0o644:
        raise HumanEvaluationError(f"{label} permissions must be exactly 0644")


def _read_key(path: str | Path) -> tuple[Path, bytes]:
    key_path = Path(path).expanduser().absolute()
    try:
        metadata = key_path.lstat()
    except OSError as error:
        raise HumanEvaluationError(f"cannot read blinding key: {key_path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise HumanEvaluationError("blinding key must be a regular non-symlink file")
    if metadata.st_nlink != 1:
        raise HumanEvaluationError("blinding key must not be hardlinked")
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise HumanEvaluationError("blinding key permissions must be exactly 0600")
    key_path = key_path.resolve()
    key = key_path.read_bytes()
    if len(key) < 32:
        raise HumanEvaluationError("blinding key must contain at least 32 bytes")
    return key_path, key


def _require_key_outside_workspace(key_path: Path, workspace: Path) -> None:
    if key_path == workspace or key_path.is_relative_to(workspace):
        raise HumanEvaluationError(
            "blinding key must be stored outside the evaluation workspace/export"
        )


def _blind_id(key: bytes, namespace: str, value: Mapping[str, Any]) -> str:
    digest = hmac.new(
        key,
        namespace.encode() + b":" + canonical_json_bytes(value),
        hashlib.sha256,
    ).hexdigest()
    return f"{namespace}-{digest[:16]}"


def _derived_seed(key: bytes, master_seed: int, prompt_id: str) -> int:
    digest = hmac.new(
        key, f"generation:{master_seed}:{prompt_id}".encode(), hashlib.sha256
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _write_new_or_identical(path: Path, payload: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise HumanEvaluationError(f"study evidence must be a regular file: {path}")
        if metadata.st_nlink != 1:
            raise HumanEvaluationError(f"study evidence must not be hardlinked: {path}")
        if path.read_bytes() != payload:
            raise HumanEvaluationError(f"refusing to replace non-identical study evidence: {path}")
        if stat.S_IMODE(metadata.st_mode) != mode:
            raise HumanEvaluationError(f"existing study evidence has wrong permissions: {path}")
        return
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json_object(path: str | Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HumanEvaluationError(f"cannot read {label}: {error}") from error
    if not isinstance(payload, dict):
        raise HumanEvaluationError(f"{label} must be an object")
    return payload


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise HumanEvaluationError(f"{label} must be a SHA-256 hex digest")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise HumanEvaluationError(f"{label} must be a SHA-256 hex digest") from error
    return value


def _exact_fields(payload: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise HumanEvaluationError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _required_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HumanEvaluationError(f"{label} must be a mapping")
    return value


def _required_path(value: Any, label: str) -> str:
    return _nonempty_string(value, label)


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HumanEvaluationError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HumanEvaluationError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HumanEvaluationError(f"{label} must be a non-negative integer")
    return value
