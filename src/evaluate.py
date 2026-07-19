"""Standalone Hydra entrypoint for checkpoint-owned validation evaluation."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from data.identity import canonical_json_bytes
from evaluation.scoring import CausalLMScorer, EvaluationResult
from models.simple_decoder_transformer import SimpleDecoderTransformer
from runtime.config import (
    validate_evaluation_checkpoint_runtime,
    validate_evaluation_config,
    validate_training_config,
)
from runtime.device import select_device
from runtime.environment import collect_environment
from runtime.reproducibility import collect_git_identity, sha256_bytes, sha256_file
from tokenizer.canonical import CanonicalTokenizer
from train import build_validation_loader_factory, build_streaming_dataloader
from train import validate_streaming_dataloaders
from training.checkpoint import (
    build_logical_checkpoint_identity,
    checkpoint_config_sha256,
    configured_manifest_fingerprints,
    load_checkpoint_for_generation,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


def evaluate_checkpoint(cfg: DictConfig) -> Path:
    """Evaluate one verified checkpoint and atomically write a compact JSON result."""

    validate_evaluation_config(cfg)
    evaluation_cfg = cfg.get("evaluation")
    assert evaluation_cfg is not None
    checkpoint_path = Path(str(evaluation_cfg.checkpoint_path))
    if not checkpoint_path.is_absolute():
        checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
    output_path = _output_path(evaluation_cfg).resolve()
    _reject_output_checkpoint_collision(checkpoint_path, output_path)
    evaluator_run_identity = _evaluator_run_identity(cfg)
    loaded = load_checkpoint_for_generation(checkpoint_path)
    payload = loaded.payload
    state = payload["state"]
    resolved_config = state.get("resolved_config")
    if not isinstance(resolved_config, Mapping):
        raise ValueError("checkpoint is missing its resolved Hydra configuration")
    checkpoint_cfg = OmegaConf.create(resolved_config)
    validate_training_config(checkpoint_cfg)
    if checkpoint_cfg.data.mode != "streaming" or checkpoint_cfg.profile.purpose != "pretraining":
        raise ValueError("held-out evaluation requires a pretraining streaming checkpoint")
    train_split = checkpoint_cfg.data.streaming.train
    validation_split = checkpoint_cfg.data.streaming.validation
    train_sources = train_split.get("sources", train_split.get("datasets", []))
    validation_sources = validation_split.get(
        "sources", validation_split.get("datasets", [])
    )
    if any(source.selection != "train" for source in train_sources) or any(
        source.selection != "validation" for source in validation_sources
    ):
        raise ValueError(
            "held-out evaluation requires explicit train and validation manifest selections"
        )

    validate_evaluation_checkpoint_runtime(cfg, checkpoint_cfg)

    device = select_device(str(evaluation_cfg.device))
    tokenizer = CanonicalTokenizer.from_config(checkpoint_cfg.tokenizer)
    identity = payload["identity"]
    model_config = OmegaConf.to_container(checkpoint_cfg.model, resolve=True)
    if not isinstance(model_config, dict) or identity.get("model_config") != model_config:
        raise ValueError(
            "checkpoint model identity does not match its resolved model configuration"
        )
    if identity.get("tokenizer_fingerprint") != tokenizer.fingerprint:
        raise ValueError(
            "checkpoint tokenizer identity does not match the canonical tokenizer artifact"
        )
    validation_factory = build_validation_loader_factory(checkpoint_cfg, device=device)
    validation_loader = validation_factory()
    if checkpoint_cfg.data.mode == "streaming":
        train_loader = build_streaming_dataloader(checkpoint_cfg, "train", device=device)
        validate_streaming_dataloaders(train_loader, validation_loader)

    model = SimpleDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_size=checkpoint_cfg.model.embed_size,
        num_heads=checkpoint_cfg.model.num_heads,
        max_len=checkpoint_cfg.training.sequence_length,
        num_layers=checkpoint_cfg.model.num_layers,
        dropout=checkpoint_cfg.model.dropout,
        pad_token_id=tokenizer.pad_token_id,
    ).to(device)
    model.load_state_dict(state["model"], strict=True)

    checkpoint_identity = build_logical_checkpoint_identity(
        payload["identity"], state.get("counters", {})
    )
    scorer = CausalLMScorer(
        device=device,
        precision=str(checkpoint_cfg.training.get("precision", "fp32")),
        ignore_index=int(checkpoint_cfg.training.get("ignore_index", -100)),
    )
    result = scorer.score(
        model,
        validation_factory,
        namespace="validation",
        logical_checkpoint_identity=checkpoint_identity,
        physical_checkpoint_identity=loaded.physical_identity,
        configured_data_fingerprints=configured_manifest_fingerprints(checkpoint_cfg),
    )
    output = {
        "schema_version": 1,
        "evaluator_run": evaluator_run_identity,
        "evaluation": {
            "kind": result.namespace,
            "scorer_revision": result.scorer_revision,
            "device": str(device),
            "precision": str(checkpoint_cfg.training.get("precision", "fp32")),
            "checkpoint_config_sha256": checkpoint_config_sha256(checkpoint_cfg),
            "tokenizer_fingerprint": tokenizer.fingerprint,
        },
        "checkpoint": {
            "kind": payload["kind"],
            "logical": checkpoint_identity,
            "physical": loaded.physical_identity,
        },
        "result": result.as_dict(),
    }
    _write_json_atomic(output_path, output)
    local_result_identity = {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "size_bytes": output_path.stat().st_size,
    }
    _maybe_log_wandb(
        evaluation_cfg,
        result,
        checkpoint_kind=str(payload["kind"]),
        local_result_identity=local_result_identity,
    )
    return output_path


def _output_path(evaluation_cfg: DictConfig) -> Path:
    path = Path(str(evaluation_cfg.get("output_path", "evaluation.json")))
    if path.is_absolute():
        return path
    try:
        output_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:
        output_dir = Path.cwd()
    return output_dir / path


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _reject_output_checkpoint_collision(checkpoint_path: Path, output_path: Path) -> None:
    """Reject paths that could overwrite the checkpoint used for evaluation."""

    checkpoint = checkpoint_path.resolve()
    output = output_path.resolve()
    if checkpoint == output:
        raise ValueError("evaluation output path must not be the checkpoint path")
    try:
        if output.exists() and os.path.samefile(checkpoint, output):
            raise ValueError("evaluation output and checkpoint must not share an inode")
    except FileNotFoundError:
        return


def _evaluator_run_identity(cfg: DictConfig) -> dict[str, Any]:
    resolved_config = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved_config, dict):
        raise ValueError("evaluation requires a mapping Hydra configuration")
    environment = collect_environment()
    return {
        "git": collect_git_identity(ROOT_DIR),
        "resolved_config": resolved_config,
        "resolved_config_sha256": sha256_bytes(canonical_json_bytes(resolved_config)),
        "lock": {
            "path": str(ROOT_DIR / "uv.lock"),
            "sha256": sha256_file(ROOT_DIR / "uv.lock"),
        },
        "environment": {
            "os": environment["os"],
            "os_release": environment["os_release"],
            "architecture": environment["architecture"],
            "python": environment["python"],
            "torch": environment["torch"],
            "cuda": environment["cuda"],
            "container_image": environment["container_image"],
        },
    }


def _maybe_log_wandb(
    evaluation_cfg: DictConfig,
    result: EvaluationResult,
    *,
    checkpoint_kind: str,
    local_result_identity: Mapping[str, Any],
) -> None:
    wandb_cfg = evaluation_cfg.get("wandb", {}) or {}
    if not wandb_cfg.get("enabled", False):
        return
    run = wandb.init(
        project=wandb_cfg.get("project"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name"),
        mode=wandb_cfg.get("mode", "online"),
    )
    try:
        run.summary.update(
            {
                "evaluation/namespace": result.namespace,
                "evaluation/nll": result.nll,
                "evaluation/perplexity": result.perplexity,
                "evaluation/target_tokens": result.target_tokens,
                "evaluation/evaluated_windows": result.evaluated_windows,
                "evaluation/evaluated_window_sha256": result.evaluated_window_sha256,
                "evaluation/evaluated_token_sha256": result.evaluated_token_sha256,
                "evaluation/pause_seconds": result.pause_seconds,
                "evaluation/scorer_identity": {"revision": result.scorer_revision},
                "evaluation/checkpoint_identity": {
                    "kind": checkpoint_kind,
                    "logical": result.logical_checkpoint_identity,
                    "physical": result.physical_checkpoint_identity,
                },
                "evaluation/manifest_identity": result.manifest_identity,
                "evaluation/local_result_identity": dict(local_result_identity),
                "evaluation/by_corpus": {
                    name: score.as_dict() for name, score in sorted(result.by_corpus.items())
                },
            }
        )
    finally:
        run.finish()


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    """Run with ``profile=evaluation evaluation.checkpoint_path=...``."""

    evaluate_checkpoint(cfg)


if __name__ == "__main__":
    main()
