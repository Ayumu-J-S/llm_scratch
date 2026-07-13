"""Standalone Hydra entrypoint for checkpoint-owned validation evaluation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import hydra
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from evaluation.scoring import CausalLMScorer, manifest_identities
from models.simple_decoder_transformer import SimpleDecoderTransformer
from runtime.config import validate_evaluation_config, validate_training_config
from runtime.device import select_device
from tokenizer.canonical import CanonicalTokenizer
from train import build_validation_loader_factory, build_streaming_dataloader
from train import validate_streaming_dataloaders
from training.checkpoint import (
    checkpoint_file_identity,
    load_checkpoint_for_generation,
)
from data.identity import canonical_fingerprint


def evaluate_checkpoint(cfg: DictConfig) -> Path:
    """Evaluate one verified checkpoint and atomically write a compact JSON result."""

    validate_evaluation_config(cfg)
    evaluation_cfg = cfg.get("evaluation")
    assert evaluation_cfg is not None
    checkpoint_path = Path(str(evaluation_cfg.checkpoint_path))
    if not checkpoint_path.is_absolute():
        checkpoint_path = (Path.cwd() / checkpoint_path).resolve()
    payload = load_checkpoint_for_generation(checkpoint_path)
    state = payload["state"]
    resolved_config = state.get("resolved_config")
    if not isinstance(resolved_config, Mapping):
        raise ValueError("checkpoint is missing its resolved Hydra configuration")
    checkpoint_cfg = OmegaConf.create(resolved_config)
    validate_training_config(checkpoint_cfg)
    if checkpoint_cfg.data.mode != "streaming" or checkpoint_cfg.profile.purpose != "pretraining":
        raise ValueError("held-out evaluation requires a pretraining streaming checkpoint")
    if any(
        source.selection != "train" for source in checkpoint_cfg.data.streaming.train.sources
    ) or any(
        source.selection != "validation"
        for source in checkpoint_cfg.data.streaming.validation.sources
    ):
        raise ValueError(
            "held-out evaluation requires explicit train and validation manifest selections"
        )

    device = select_device(str(evaluation_cfg.get("device", "cpu")))
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

    checkpoint_identity = {
        "checkpoint_identity": payload["identity"],
        "kind": payload["kind"],
        "counters": dict(state.get("counters", {})),
    }
    physical_identity = checkpoint_file_identity(checkpoint_path)
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
        physical_checkpoint_identity=physical_identity,
        manifest_identity=manifest_identities(
            getattr(getattr(validation_loader, "dataset", None), "resolved_manifests", None)
        ),
    )
    output = {
        "schema_version": 1,
        "evaluation": {
            "kind": result.namespace,
            "scorer_revision": result.scorer_revision,
            "device": str(device),
            "precision": str(checkpoint_cfg.training.get("precision", "fp32")),
            "checkpoint_config_sha256": canonical_fingerprint(
                OmegaConf.to_container(checkpoint_cfg, resolve=True)
            ),
            "tokenizer_fingerprint": tokenizer.fingerprint,
        },
        "checkpoint": {
            "logical": checkpoint_identity,
            "physical": physical_identity,
        },
        "result": result.as_dict(),
    }
    output_path = _output_path(evaluation_cfg)
    _write_json_atomic(output_path, output)
    _maybe_log_wandb(evaluation_cfg, result)
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


def _maybe_log_wandb(evaluation_cfg: DictConfig, result) -> None:
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
