from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from human_evaluation import workflow
from human_evaluation.schema import SCORE_SCHEMA_VERSION, load_prompt_set
from human_evaluation.workflow import (
    HumanEvaluationError,
    _load_checkpoint_candidates,
    create_blinding_key,
    import_scores,
    prepare_evaluation,
)
from training.checkpoint import LoadedCheckpoint


PROMPT_SET_PATH = Path("evaluation/human/prompts-v1.json").resolve()
SHARED_IDENTITY = {
    "schema_version": 1,
    "experiment_id": "RUN-001-synthetic-human-fixture",
    "git_sha": "1" * 40,
    "lock_sha256": "2" * 64,
    "config_sha256": "3" * 64,
    "model_config": {"embed_size": 8, "num_heads": 2, "num_layers": 1, "dropout": 0.0},
    "tokenizer_fingerprint": "4" * 64,
    "data_fingerprints": ["5" * 64, "6" * 64],
}


class FakeSampler:
    completion_variant = ""

    def __init__(self, slot: str, physical_identity: dict | None = None):
        self.slot = slot
        self.physical_checkpoint_identity = physical_identity

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, device: str):
        assert device in {"cpu", "cuda"}
        return cls("earlier" if "earlier" in str(path) else "later")

    @classmethod
    def from_loaded_checkpoint(cls, path: str | Path, loaded: LoadedCheckpoint, *, device: str):
        assert loaded.physical_identity["path"] == str(Path(path).resolve())
        assert device in {"cpu", "cuda"}
        return cls(
            "earlier" if "earlier" in str(path) else "later",
            dict(loaded.physical_identity),
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        seed: int,
        precision: str,
    ):
        assert (max_new_tokens, temperature, top_k) == (64, 0.8, 40)
        assert isinstance(seed, int) and seed >= 0
        assert precision == "fp32"
        text = "続きの文章です。" if prompt.startswith("東") else " generated continuation."
        suffix = " First." if self.slot == "earlier" else " Second."
        return SimpleNamespace(completion=text + suffix + self.completion_variant)


def _loaded(slot: str, *, target_tokens: int, optimizer_step: int) -> LoadedCheckpoint:
    identity = copy.deepcopy(SHARED_IDENTITY)
    return LoadedCheckpoint(
        payload={
            "kind": "milestone",
            "identity": identity,
            "state": {
                "counters": {
                    "optimizer_step": optimizer_step,
                    "target_tokens": target_tokens,
                },
                "resolved_config": {"training": {"precision": "fp32"}},
            },
        },
        physical_identity={
            "path": f"/synthetic/{slot}.pt",
            "sha256": ("a" if slot == "earlier" else "b") * 64,
            "size_bytes": 1024,
        },
    )


@pytest.fixture
def prepared_study(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "human-evaluation" / "study"
    key = create_blinding_key(tmp_path / "secret" / "human.key")
    key.write_bytes(b"HUMAN-001 deterministic fixture key")
    key.chmod(0o600)

    def fake_loader(path: str | Path):
        slot = "earlier" if "earlier" in str(path) else "later"
        return _loaded(
            slot,
            target_tokens=1_000_000 if slot == "earlier" else 1_500_000,
            optimizer_step=100 if slot == "earlier" else 150,
        )

    monkeypatch.setattr(workflow, "load_checkpoint_for_generation", fake_loader)
    monkeypatch.setattr(workflow, "CheckpointSampler", FakeSampler)
    paths = prepare_evaluation(
        prompt_set_path=PROMPT_SET_PATH,
        workspace_dir=workspace,
        blinding_key_path=key,
        earlier_checkpoint="/synthetic/earlier.pt",
        later_checkpoint="/synthetic/later.pt",
        generation_seed=20260719,
    )
    return workspace, key, paths


def _score(bundle: dict, reviewer_id: str, *, offset: int = 0) -> dict:
    ratings = []
    for index, item in enumerate(bundle["items"]):
        value = 3 + ((index + offset) % 2)
        ratings.append(
            {
                "item_id": item["item_id"],
                "candidate_a": {
                    "fluency": value,
                    "coherence": value,
                    "naturalness": value,
                },
                "candidate_b": {
                    "fluency": value,
                    "coherence": value,
                    "naturalness": value,
                },
                "preference": ("A", "B", "tie")[(index + offset) % 3],
                "comment": "",
            }
        )
    return {
        "schema_version": SCORE_SCHEMA_VERSION,
        "study_id": bundle["study_id"],
        "bundle_id": bundle["bundle_id"],
        "reviewer_id": reviewer_id,
        "ratings": ratings,
    }


def _write_scores(workspace: Path, *scores: dict) -> list[Path]:
    review_dir = workspace / "reviews"
    review_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    review_dir.chmod(0o700)
    paths = []
    for index, score in enumerate(scores, start=1):
        path = review_dir / f"review-{index}.json"
        path.write_text(json.dumps(score, ensure_ascii=False), encoding="utf-8")
        path.chmod(0o600)
        paths.append(path)
    return paths


def test_prepare_is_reproducible_balanced_and_public_bundle_has_no_identity_leak(
    prepared_study, tmp_path: Path, monkeypatch
):
    workspace, key, paths = prepared_study
    public = json.loads(paths["public_bundle"].read_text(encoding="utf-8"))
    private = json.loads(paths["private_mapping"].read_text(encoding="utf-8"))

    assert [item["language"] for item in public["items"]].count("ja") == 4
    assert [item["language"] for item in public["items"]].count("en") == 4
    assert set(public) == {
        "schema_version",
        "study_id",
        "bundle_id",
        "prompt_set_version",
        "evaluation_type",
        "protocol",
        "sampling",
        "rubric",
        "items",
    }
    serialized_public = paths["public_bundle"].read_text(encoding="utf-8")
    for private_value in (
        "/synthetic/earlier.pt",
        "/synthetic/later.pt",
        "a" * 64,
        "b" * 64,
        "1000000",
        "1500000",
        "RUN-001-synthetic-human-fixture",
    ):
        assert private_value not in serialized_public
    forbidden_key_fragments = {
        "checkpoint",
        "path",
        "sha",
        "hash",
        "token_count",
        "target_tokens",
        "optimizer_step",
        "seed",
        "mapping",
        "order",
    }
    public_keys = set()

    def collect_keys(value):
        if isinstance(value, dict):
            public_keys.update(key.casefold() for key in value)
            for child in value.values():
                collect_keys(child)
        elif isinstance(value, list):
            for child in value:
                collect_keys(child)

    collect_keys(public)
    assert not any(fragment in key for fragment in forbidden_key_fragments for key in public_keys)

    assignments = private["payload"]["assignments"]
    prompt_ids = [prompt.id for prompt in load_prompt_set(PROMPT_SET_PATH).prompts]
    expected_prompt_order = sorted(
        prompt_ids,
        key=lambda prompt_id: hmac.new(
            key.read_bytes(),
            f"item-order:{public['study_id']}:{prompt_id}".encode(),
            hashlib.sha256,
        ).digest(),
    )
    assert [assignment["prompt_id"] for assignment in assignments] == expected_prompt_order
    assert sum(item["candidates"]["A"] == "earlier" for item in assignments) == 4
    assert sum(item["candidates"]["B"] == "earlier" for item in assignments) == 4
    assert (
        sum(
            item["prompt_id"].startswith("ja-") and item["candidates"]["A"] == "earlier"
            for item in assignments
        )
        == 2
    )
    assert (
        sum(
            item["prompt_id"].startswith("en-") and item["candidates"]["A"] == "earlier"
            for item in assignments
        )
        == 2
    )
    assert stat.S_IMODE(paths["private_mapping"].stat().st_mode) == 0o600
    assert stat.S_IMODE((workspace / "private").stat().st_mode) == 0o700
    assert stat.S_IMODE((workspace / "public").stat().st_mode) == 0o755

    second_workspace = tmp_path / "human-evaluation" / "second-study"

    def fake_loader(path: str | Path):
        slot = "earlier" if "earlier" in str(path) else "later"
        return _loaded(
            slot,
            target_tokens=1_000_000 if slot == "earlier" else 1_500_000,
            optimizer_step=100 if slot == "earlier" else 150,
        )

    monkeypatch.setattr(workflow, "load_checkpoint_for_generation", fake_loader)
    monkeypatch.setattr(workflow, "CheckpointSampler", FakeSampler)
    second = prepare_evaluation(
        prompt_set_path=PROMPT_SET_PATH,
        workspace_dir=second_workspace,
        blinding_key_path=key,
        earlier_checkpoint="/synthetic/earlier.pt",
        later_checkpoint="/synthetic/later.pt",
        generation_seed=20260719,
    )
    assert second["public_bundle"].read_bytes() == paths["public_bundle"].read_bytes()

    def changed_pair_loader(path: str | Path):
        loaded = fake_loader(path)
        if "later" in str(path):
            loaded.physical_identity["sha256"] = "c" * 64
        return loaded

    monkeypatch.setattr(workflow, "load_checkpoint_for_generation", changed_pair_loader)
    third = prepare_evaluation(
        prompt_set_path=PROMPT_SET_PATH,
        workspace_dir=tmp_path / "human-evaluation" / "third-study",
        blinding_key_path=key,
        earlier_checkpoint="/synthetic/earlier.pt",
        later_checkpoint="/synthetic/later.pt",
        generation_seed=20260719,
    )
    third_public = json.loads(third["public_bundle"].read_text(encoding="utf-8"))
    assert third_public["study_id"] != public["study_id"]


def test_score_import_round_trip_agreement_and_exact_unblinding(prepared_study):
    workspace, key, paths = prepared_study
    public = json.loads(paths["public_bundle"].read_text(encoding="utf-8"))
    score_paths = _write_scores(
        workspace,
        _score(public, "reviewer-one"),
        _score(public, "reviewer-two", offset=1),
    )

    result_path = import_scores(
        workspace_dir=workspace,
        blinding_key_path=key,
        score_paths=score_paths,
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["reviewers"] == ["reviewer-one", "reviewer-two"]
    assert result["bundle_id"] == public["bundle_id"]
    assert result["public_bundle_sha256"] == private_bundle_hash(paths["private_mapping"])
    assert result["agreement"]["reviewer_count"] == 2
    assert result["agreement"]["preference_comparisons"] == 8
    assert result["agreement"]["rating_comparisons"] == 48
    assert 0 <= result["agreement"]["preference_exact_fraction"] <= 1
    assert set(result["checkpoint_summary"]) == {"earlier", "later"}
    assert len(result["unblinded_ratings"]) == 16
    assert all(
        set(rating["ratings"]) == {"earlier", "later"} for rating in result["unblinded_ratings"]
    )
    assert result["research_integrity"]["training_reuse_permitted"] is False
    assert stat.S_IMODE(result_path.stat().st_mode) == 0o600


def test_score_import_requires_two_distinct_complete_human_reviewers(prepared_study):
    workspace, key, paths = prepared_study
    public = json.loads(paths["public_bundle"].read_text(encoding="utf-8"))
    one_path = _write_scores(workspace, _score(public, "same-reviewer"))[0]

    with pytest.raises(HumanEvaluationError, match="at least two"):
        import_scores(
            workspace_dir=workspace,
            blinding_key_path=key,
            score_paths=[one_path],
        )

    duplicate_paths = _write_scores(
        workspace,
        _score(public, "Same-Reviewer"),
        _score(public, "same-reviewer"),
    )
    with pytest.raises(HumanEvaluationError, match="distinct"):
        import_scores(
            workspace_dir=workspace,
            blinding_key_path=key,
            score_paths=duplicate_paths,
        )

    incomplete = _score(public, "other-reviewer")
    incomplete["ratings"].pop()
    incomplete_paths = _write_scores(workspace, _score(public, "first-reviewer"), incomplete)
    with pytest.raises(HumanEvaluationError, match="every public item"):
        import_scores(
            workspace_dir=workspace,
            blinding_key_path=key,
            score_paths=incomplete_paths,
        )


def private_bundle_hash(mapping_path: Path) -> str:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    return mapping["payload"]["public_bundle_sha256"]


def test_scores_cannot_cross_accept_a_distinct_exact_bundle_with_the_same_study(
    tmp_path: Path, monkeypatch
):
    key = create_blinding_key(tmp_path / "secret" / "bundle.key")
    key.write_bytes(b"HUMAN-001 exact bundle fixture key")
    key.chmod(0o600)

    def fake_loader(path: str | Path):
        slot = "earlier" if "earlier" in str(path) else "later"
        return _loaded(
            slot,
            target_tokens=1_000_000 if slot == "earlier" else 1_500_000,
            optimizer_step=100 if slot == "earlier" else 150,
        )

    class VariantSampler(FakeSampler):
        completion_variant = " evaluator-variant"

    monkeypatch.setattr(workflow, "load_checkpoint_for_generation", fake_loader)
    monkeypatch.setattr(workflow, "CheckpointSampler", FakeSampler)
    first_workspace = tmp_path / "human-evaluation" / "bundle-first"
    first_paths = prepare_evaluation(
        prompt_set_path=PROMPT_SET_PATH,
        workspace_dir=first_workspace,
        blinding_key_path=key,
        earlier_checkpoint="/synthetic/earlier.pt",
        later_checkpoint="/synthetic/later.pt",
        generation_seed=20260719,
        device="cpu",
    )
    first_public = json.loads(first_paths["public_bundle"].read_text(encoding="utf-8"))

    monkeypatch.setattr(workflow, "CheckpointSampler", VariantSampler)
    second_workspace = tmp_path / "human-evaluation" / "bundle-second"
    second_paths = prepare_evaluation(
        prompt_set_path=PROMPT_SET_PATH,
        workspace_dir=second_workspace,
        blinding_key_path=key,
        earlier_checkpoint="/synthetic/earlier.pt",
        later_checkpoint="/synthetic/later.pt",
        generation_seed=20260719,
        device="cpu",
    )
    second_public = json.loads(second_paths["public_bundle"].read_text(encoding="utf-8"))
    assert first_public["study_id"] == second_public["study_id"]
    assert [item["item_id"] for item in first_public["items"]] == [
        item["item_id"] for item in second_public["items"]
    ]
    assert first_public["bundle_id"] != second_public["bundle_id"]

    copied_scores = _write_scores(
        second_workspace,
        _score(first_public, "first"),
        _score(first_public, "second"),
    )
    with pytest.raises(HumanEvaluationError, match="bundle_id differs"):
        import_scores(
            workspace_dir=second_workspace,
            blinding_key_path=key,
            score_paths=copied_scores,
        )


def test_public_and_private_tampering_are_rejected(prepared_study):
    workspace, key, paths = prepared_study
    original_public = paths["public_bundle"].read_text(encoding="utf-8")
    public = json.loads(original_public)
    score_paths = _write_scores(workspace, _score(public, "first"), _score(public, "second"))
    public["items"][0]["candidates"]["A"] += " tampered"
    paths["public_bundle"].write_text(json.dumps(public), encoding="utf-8")
    with pytest.raises(HumanEvaluationError, match="does not match"):
        import_scores(
            workspace_dir=workspace,
            blinding_key_path=key,
            score_paths=score_paths,
        )

    paths["public_bundle"].write_text(original_public, encoding="utf-8")
    private = json.loads(paths["private_mapping"].read_text(encoding="utf-8"))
    private["payload"]["assignments"][0]["candidates"]["A"] = "tampered-slot"
    paths["private_mapping"].write_text(json.dumps(private), encoding="utf-8")
    with pytest.raises(HumanEvaluationError, match="authentication failed"):
        import_scores(
            workspace_dir=workspace,
            blinding_key_path=key,
            score_paths=score_paths,
        )


def test_import_rejects_wrong_private_and_public_modes_or_hardlinks(prepared_study):
    workspace, key, paths = prepared_study
    public = json.loads(paths["public_bundle"].read_text(encoding="utf-8"))
    score_paths = _write_scores(workspace, _score(public, "first"), _score(public, "second"))

    def run_import():
        return import_scores(
            workspace_dir=workspace,
            blinding_key_path=key,
            score_paths=score_paths,
        )

    for path, bad_mode, expected in (
        (workspace, 0o755, "directory has wrong permissions"),
        (workspace / "private", 0o755, "directory has wrong permissions"),
        (workspace / "public", 0o700, "directory has wrong permissions"),
        (workspace / "reviews", 0o755, "reviews directory permissions"),
        (paths["public_bundle"], 0o600, "public bundle permissions"),
        (paths["private_mapping"], 0o644, "private mapping permissions"),
        (score_paths[0], 0o644, "score file permissions"),
        (key, 0o644, "blinding key permissions"),
    ):
        original_mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(bad_mode)
        with pytest.raises(HumanEvaluationError, match=expected):
            run_import()
        path.chmod(original_mode)

    for path, expected in (
        (paths["public_bundle"], "public bundle must not be hardlinked"),
        (paths["private_mapping"], "private mapping must not be hardlinked"),
        (score_paths[0], "score file must not be hardlinked"),
        (key, "blinding key must not be hardlinked"),
    ):
        hardlink = path.parent / f"{path.name}.hardlink"
        os.link(path, hardlink)
        with pytest.raises(HumanEvaluationError, match=expected):
            run_import()
        hardlink.unlink()


def test_import_rejects_workspace_public_private_review_score_and_key_symlinks(prepared_study):
    workspace, key, paths = prepared_study
    public = json.loads(paths["public_bundle"].read_text(encoding="utf-8"))
    score_paths = _write_scores(workspace, _score(public, "first"), _score(public, "second"))

    workspace_alias = workspace.parent / "workspace-alias"
    workspace_alias.symlink_to(workspace, target_is_directory=True)
    with pytest.raises(HumanEvaluationError, match="workspace must not be a symlink"):
        import_scores(
            workspace_dir=workspace_alias,
            blinding_key_path=key,
            score_paths=score_paths,
        )
    workspace_alias.unlink()

    key_alias = key.parent / "key-alias"
    key_alias.symlink_to(key)
    with pytest.raises(HumanEvaluationError, match="blinding key must be a regular non-symlink"):
        import_scores(
            workspace_dir=workspace,
            blinding_key_path=key_alias,
            score_paths=score_paths,
        )
    key_alias.unlink()

    for directory_name in ("public", "private", "reviews"):
        directory = workspace / directory_name
        real_directory = workspace / f"{directory_name}-real"
        directory.rename(real_directory)
        directory.symlink_to(real_directory, target_is_directory=True)
        with pytest.raises(HumanEvaluationError, match="symlink|real directory"):
            import_scores(
                workspace_dir=workspace,
                blinding_key_path=key,
                score_paths=[
                    workspace / "reviews" / score_paths[0].name,
                    workspace / "reviews" / score_paths[1].name,
                ],
            )
        directory.unlink()
        real_directory.rename(directory)

    for path, expected in (
        (paths["public_bundle"], "public bundle must be a regular non-symlink"),
        (paths["private_mapping"], "private mapping must be a regular non-symlink"),
        (score_paths[0], "score file must be a regular non-symlink"),
    ):
        real_path = path.with_name(f"{path.name}.real")
        path.rename(real_path)
        path.symlink_to(real_path)
        with pytest.raises(HumanEvaluationError, match=expected):
            import_scores(
                workspace_dir=workspace,
                blinding_key_path=key,
                score_paths=score_paths,
            )
        path.unlink()
        real_path.rename(path)


def test_checkpoint_pair_requires_same_run_and_at_least_25_percent_token_separation():
    def too_close(path: str | Path):
        slot = "earlier" if "earlier" in str(path) else "later"
        return _loaded(
            slot,
            target_tokens=800 if slot == "earlier" else 1_000,
            optimizer_step=80 if slot == "earlier" else 100,
        )

    with pytest.raises(HumanEvaluationError, match="at least 25%"):
        _load_checkpoint_candidates("earlier", "later", loader=too_close)

    def different_run(path: str | Path):
        slot = "earlier" if "earlier" in str(path) else "later"
        loaded = _loaded(
            slot,
            target_tokens=700 if slot == "earlier" else 1_000,
            optimizer_step=70 if slot == "earlier" else 100,
        )
        if slot == "later":
            loaded.payload["identity"]["experiment_id"] = "another-run"
        return loaded

    with pytest.raises(HumanEvaluationError, match="experiment_id differs"):
        _load_checkpoint_candidates("earlier", "later", loader=different_run)


def test_prepare_rejects_checkpoint_bytes_changed_after_pair_validation(
    tmp_path: Path, monkeypatch
):
    workspace = tmp_path / "human-evaluation" / "changed-checkpoint"
    key = create_blinding_key(tmp_path / "secret" / "changed-checkpoint.key")
    calls = {"earlier": 0, "later": 0}

    def changing_loader(path: str | Path):
        slot = "earlier" if "earlier" in str(path) else "later"
        calls[slot] += 1
        loaded = _loaded(
            slot,
            target_tokens=1_000_000 if slot == "earlier" else 1_500_000,
            optimizer_step=100 if slot == "earlier" else 150,
        )
        if slot == "later" and calls[slot] == 2:
            loaded.physical_identity["sha256"] = "c" * 64
        return loaded

    monkeypatch.setattr(workflow, "load_checkpoint_for_generation", changing_loader)
    monkeypatch.setattr(workflow, "CheckpointSampler", FakeSampler)
    with pytest.raises(HumanEvaluationError, match="changed between pair validation"):
        prepare_evaluation(
            prompt_set_path=PROMPT_SET_PATH,
            workspace_dir=workspace,
            blinding_key_path=key,
            earlier_checkpoint="/synthetic/earlier.pt",
            later_checkpoint="/synthetic/later.pt",
            generation_seed=20260719,
            device="cpu",
        )
    assert not workspace.exists()


def test_prepare_rejects_sampler_physical_identity_mismatch(tmp_path: Path, monkeypatch):
    workspace = tmp_path / "human-evaluation" / "sampler-mismatch"
    key = create_blinding_key(tmp_path / "secret" / "sampler-mismatch.key")

    def fake_loader(path: str | Path):
        slot = "earlier" if "earlier" in str(path) else "later"
        return _loaded(
            slot,
            target_tokens=1_000_000 if slot == "earlier" else 1_500_000,
            optimizer_step=100 if slot == "earlier" else 150,
        )

    class MismatchedSampler(FakeSampler):
        @classmethod
        def from_loaded_checkpoint(cls, path: str | Path, loaded: LoadedCheckpoint, *, device: str):
            sampler = super().from_loaded_checkpoint(path, loaded, device=device)
            sampler.physical_checkpoint_identity["sha256"] = "d" * 64
            return sampler

    monkeypatch.setattr(workflow, "load_checkpoint_for_generation", fake_loader)
    monkeypatch.setattr(workflow, "CheckpointSampler", MismatchedSampler)
    with pytest.raises(HumanEvaluationError, match="sampler physical identity differs"):
        prepare_evaluation(
            prompt_set_path=PROMPT_SET_PATH,
            workspace_dir=workspace,
            blinding_key_path=key,
            earlier_checkpoint="/synthetic/earlier.pt",
            later_checkpoint="/synthetic/later.pt",
            generation_seed=20260719,
            device="cpu",
        )
    assert not workspace.exists()


def test_key_permissions_and_training_namespace_isolation(tmp_path: Path):
    key = create_blinding_key(tmp_path / "secret" / "key")
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    with pytest.raises(HumanEvaluationError, match="replace existing"):
        create_blinding_key(key)
    key.chmod(0o644)
    with pytest.raises(HumanEvaluationError, match="0600"):
        prepare_evaluation(
            prompt_set_path=PROMPT_SET_PATH,
            workspace_dir=tmp_path / "human-evaluation" / "study",
            blinding_key_path=key,
            earlier_checkpoint="earlier",
            later_checkpoint="later",
            generation_seed=1,
        )

    safe_key = create_blinding_key(tmp_path / "secret" / "safe-key")
    for unsafe_workspace in (
        tmp_path / "runs" / "human-evaluation" / "study",
        tmp_path / "data" / "human-evaluation" / "study",
        tmp_path / "cache" / "human-evaluation" / "study",
        tmp_path / "human-evaluation" / "checkpoints" / "study",
    ):
        with pytest.raises(HumanEvaluationError, match="overlaps"):
            prepare_evaluation(
                prompt_set_path=PROMPT_SET_PATH,
                workspace_dir=unsafe_workspace,
                blinding_key_path=safe_key,
                earlier_checkpoint="earlier",
                later_checkpoint="later",
                generation_seed=1,
            )

    inside_key = create_blinding_key(tmp_path / "human-evaluation" / "study" / "key")
    with pytest.raises(HumanEvaluationError, match="outside"):
        prepare_evaluation(
            prompt_set_path=PROMPT_SET_PATH,
            workspace_dir=tmp_path / "human-evaluation" / "study",
            blinding_key_path=inside_key,
            earlier_checkpoint="earlier",
            later_checkpoint="later",
            generation_seed=1,
        )


def test_blinding_key_must_be_outside_repository(tmp_path: Path, monkeypatch):
    repository = tmp_path / "repository"
    monkeypatch.setattr(workflow, "_REPOSITORY_ROOT", repository.resolve())
    prompt_path = repository / "evaluation" / "human" / "prompts-v1.json"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_bytes(PROMPT_SET_PATH.read_bytes())
    key = create_blinding_key(repository / ".secrets" / "HUMAN-001.key")

    with pytest.raises(HumanEvaluationError, match="outside the repository"):
        prepare_evaluation(
            prompt_set_path=prompt_path,
            workspace_dir=tmp_path / "human-evaluation" / "study",
            blinding_key_path=key,
            earlier_checkpoint="earlier",
            later_checkpoint="later",
            generation_seed=1,
        )

    create_path = repository / ".secrets" / "new-HUMAN-001.key"
    with pytest.raises(HumanEvaluationError, match="outside the repository"):
        workflow.run_from_config(
            {
                "action": "create_key",
                "prompt_set_path": str(prompt_path),
                "blinding_key_path": str(create_path),
            }
        )
    assert not create_path.exists()

    external_key = create_blinding_key(tmp_path / "secret" / "external-HUMAN-001.key")
    external_key.write_bytes(key.read_bytes())
    external_key.chmod(0o600)

    def fake_loader(path: str | Path):
        slot = "earlier" if "earlier" in str(path) else "later"
        return _loaded(
            slot,
            target_tokens=1_000_000 if slot == "earlier" else 1_500_000,
            optimizer_step=100 if slot == "earlier" else 150,
        )

    monkeypatch.setattr(workflow, "load_checkpoint_for_generation", fake_loader)
    monkeypatch.setattr(workflow, "CheckpointSampler", FakeSampler)
    workspace = tmp_path / "human-evaluation" / "import-key-location"
    paths = prepare_evaluation(
        prompt_set_path=prompt_path,
        workspace_dir=workspace,
        blinding_key_path=external_key,
        earlier_checkpoint="/synthetic/earlier.pt",
        later_checkpoint="/synthetic/later.pt",
        generation_seed=20260719,
        device="cpu",
    )
    public = json.loads(paths["public_bundle"].read_text(encoding="utf-8"))
    score_paths = _write_scores(
        workspace,
        _score(public, "reviewer-one"),
        _score(public, "reviewer-two"),
    )
    with pytest.raises(HumanEvaluationError, match="outside the repository"):
        import_scores(
            workspace_dir=workspace,
            blinding_key_path=key,
            score_paths=score_paths,
        )

    alternate_prompt = tmp_path / "alternate-protocol" / "evaluation" / "human" / "prompts-v1.json"
    alternate_prompt.parent.mkdir(parents=True)
    alternate_prompt.write_bytes(PROMPT_SET_PATH.read_bytes())
    with pytest.raises(HumanEvaluationError, match="within the evaluator repository"):
        prepare_evaluation(
            prompt_set_path=alternate_prompt,
            workspace_dir=tmp_path / "human-evaluation" / "alternate-prepare",
            blinding_key_path=key,
            earlier_checkpoint="earlier",
            later_checkpoint="later",
            generation_seed=1,
        )

    alternate_create_path = tmp_path / "secret" / "alternate-create.key"
    with pytest.raises(HumanEvaluationError, match="within the evaluator repository"):
        workflow.run_from_config(
            {
                "action": "create_key",
                "prompt_set_path": str(alternate_prompt),
                "blinding_key_path": str(alternate_create_path),
            }
        )
    assert not alternate_create_path.exists()

    mapping = json.loads(paths["private_mapping"].read_text(encoding="utf-8"))
    mapping["payload"]["prompt_set"]["path"] = str(alternate_prompt.resolve())
    mapping["payload"]["prompt_set"]["sha256"] = hashlib.sha256(
        alternate_prompt.read_bytes()
    ).hexdigest()
    mapping["authentication"]["tag"] = hmac.new(
        key.read_bytes(),
        workflow.canonical_json_bytes(mapping["payload"]),
        hashlib.sha256,
    ).hexdigest()
    paths["private_mapping"].write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
    paths["private_mapping"].chmod(0o600)
    with pytest.raises(HumanEvaluationError, match="within the evaluator repository"):
        import_scores(
            workspace_dir=workspace,
            blinding_key_path=key,
            score_paths=score_paths,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("max_new_tokens", 63), ("temperature", 0.7), ("top_k", 39)),
)
def test_hydra_dispatch_rejects_sampling_contract_overrides(field: str, value):
    generation = {"max_new_tokens": 64, "temperature": 0.8, "top_k": 40, "seed": 1}
    generation[field] = value
    with pytest.raises(HumanEvaluationError, match=f"generation.{field} is fixed"):
        workflow.run_from_config(
            {
                "action": "prepare",
                "blinding_key_path": "/secret/key",
                "workspace_dir": "/tmp/human-evaluation/study",
                "prompt_set_path": str(PROMPT_SET_PATH),
                "checkpoints": {"earlier": "/early", "later": "/late"},
                "generation": generation,
                "device": "cpu",
            }
        )


def test_hydra_dispatch_rejects_implicit_or_unknown_device():
    with pytest.raises(HumanEvaluationError, match="explicitly cpu or cuda"):
        workflow.run_from_config(
            {
                "action": "prepare",
                "blinding_key_path": "/secret/key",
                "workspace_dir": "/tmp/human-evaluation/study",
                "prompt_set_path": str(PROMPT_SET_PATH),
                "checkpoints": {"earlier": "/early", "later": "/late"},
                "generation": {
                    "max_new_tokens": 64,
                    "temperature": 0.8,
                    "top_k": 40,
                    "seed": 1,
                },
                "device": "auto",
            }
        )
