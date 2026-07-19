import os
from contextlib import contextmanager
from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from data.manifests import ResolvedManifest, preflight_manifest, validate_disjoint_manifests
from data.streaming_dataset import create_streaming_token_dataloader
from data.text_dataset import create_autoregressive_dataloader
from models.simple_decoder_transformer import SimpleDecoderTransformer
from runtime.device import select_device
from runtime.config import ConfigPreflightError, validate_training_config
from runtime.reproducibility import (
    dataloader_generator,
    dataloader_worker_init_fn,
    seed_everything,
    write_run_manifest,
)
from training.checkpoint import (
    CheckpointManager,
    build_checkpoint_identity,
    load_run_lineage_from_resume,
    require_exact_stream_resume_state,
)
from training.optimization import build_optimizer, build_scheduler
from training.trainer import Trainer
from tokenizer.canonical import CanonicalTokenizer
from utils.model import get_parameter_counts


ROOT_DIR = Path(__file__).resolve().parent.parent


def _run_directory() -> Path:
    """Return Hydra's output directory without requiring an active Hydra job."""

    try:
        output_dir = HydraConfig.get().runtime.output_dir
    except Exception:  # HydraConfig is unavailable for direct/importable calls.
        output_dir = ROOT_DIR / "runs" / "manual"
    return Path(output_dir)


def save_resolved_config(cfg: DictConfig, *, run_dir: Path | None = None) -> Path:
    """Persist the exact resolved Hydra config in the run directory."""

    run_dir = _run_directory() if run_dir is None else Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    destination = run_dir / "resolved_config.yaml"
    OmegaConf.save(config=cfg, f=str(destination), resolve=True)
    return destination


def validate_profile_batches(cfg: DictConfig) -> None:
    """Build one real train and validation batch during profile preflight."""

    if cfg.data.mode != "streaming":
        return
    train_loader = build_streaming_dataloader(cfg, "train")
    validation_loader = build_streaming_dataloader(cfg, "validation")
    validate_streaming_dataloaders(train_loader, validation_loader)
    for name, loader in (("train", train_loader), ("validation", validation_loader)):
        try:
            next(iter(loader))
        except StopIteration as error:
            raise ConfigPreflightError(
                f"streaming {name} profile produced no complete batch"
            ) from error


def validate_profile_manifests(cfg: DictConfig) -> None:
    """Resolve immutable metadata without opening or downloading source shards."""

    if cfg.data.mode != "streaming":
        return
    train_loader = build_streaming_dataloader(cfg, "train")
    validation_loader = build_streaming_dataloader(cfg, "validation")
    validate_streaming_dataloaders(train_loader, validation_loader)


def log_sample_batch(tokenizer, batch) -> None:
    inputs = batch["inputs"]
    labels = batch["labels"]
    sample_index = min(20, len(inputs) - 1)
    logger.info("Training samples: {} in preview batch", len(inputs))
    logger.info(
        "Example decoder input: {!r}",
        tokenizer.decode(inputs[sample_index].tolist(), skip_special_tokens=False),
    )
    logger.info(
        "Example label: {!r}",
        tokenizer.decode(labels[sample_index].tolist(), skip_special_tokens=False),
    )


def load_token_ids(input_path: str, tokenizer, split_name: str) -> list[int]:
    split_label = split_name.capitalize()
    logger.info("Loading {} corpus from: {}", split_name, input_path)
    text = (ROOT_DIR / input_path).read_text(encoding="utf-8")
    logger.info("Tokenizing {} corpus...", split_name)
    token_ids = tokenizer.encode(text)
    logger.info("{} corpus length: {} tokens", split_label, len(token_ids))
    return token_ids


def build_tokenizer_config(cfg: DictConfig) -> dict[str, str]:
    config = OmegaConf.to_container(cfg.tokenizer, resolve=True)
    if not isinstance(config, dict):
        raise TypeError("tokenizer config must resolve to a mapping")
    return config


def to_plain_config(config: DictConfig) -> dict:
    container = OmegaConf.to_container(config, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("config must resolve to a mapping")
    return container


def streaming_split_config(streaming_cfg: DictConfig, split_name: str) -> dict:
    config = to_plain_config(streaming_cfg)
    split_key = "train" if split_name == "train" else "validation"
    split_config = config.get(split_key)
    if split_config is None:
        raise ValueError(f"data.streaming.{split_key} is required for streaming mode")
    if not isinstance(split_config, dict):
        raise TypeError(f"data.streaming.{split_key} must be a mapping")

    common_config = {
        key: value for key, value in config.items() if key not in {"train", "validation"}
    }
    return {**common_config, **split_config}


def resolve_memorization_smoke(data_cfg: DictConfig) -> ResolvedManifest:
    if data_cfg.get("mode") != "memorization_smoke":
        raise ValueError("default same-corpus training requires data.mode=memorization_smoke")
    smoke_cfg = data_cfg.get("memorization")
    if smoke_cfg is None:
        raise ValueError("data.memorization is required for memorization_smoke")
    return preflight_manifest(
        smoke_cfg.manifest_path,
        expected_fingerprint=smoke_cfg.expected_fingerprint,
        selection="all",
        access="training",
        allow_reserved_benchmark=False,
    )


def resolved_manifest_token_ids(manifest: ResolvedManifest, tokenizer) -> list[int]:
    token_ids: list[int] = []
    for document in manifest.documents:
        token_ids.extend(tokenizer.encode(document.text))
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            token_ids.append(int(eos_token_id))
    return token_ids


def build_streaming_dataloader(
    cfg: DictConfig,
    split_name: str,
    *,
    device: torch.device | None = None,
    data_root: Path | None = None,
):
    reproducibility_cfg = cfg.get("reproducibility", {})
    base_seed = int(reproducibility_cfg.get("seed", 0))
    stream_config = streaming_split_config(cfg.data.streaming, split_name)
    if data_root is not None:
        _bind_streaming_loader_paths(stream_config, root=data_root)
    if stream_config.get("require_manifests") is False:
        raise ValueError("streaming training cannot set require_manifests=false")
    stream_config["require_manifests"] = True
    stream_config["tokenizer"] = build_tokenizer_config(cfg)
    stream_config["seed"] = base_seed + (0 if split_name == "train" else 1)
    # Validation needs source attribution for token-weighted per-corpus scores.
    # The factory creates a fresh deterministic stream for every score pass.
    if split_name == "validation":
        stream_config["preserve_metadata"] = True
    return create_streaming_token_dataloader(
        config=stream_config,
        sequence_length=cfg.training.sequence_length,
        batch_size=cfg.training.batch_size,
        drop_last=split_name == "train",
        num_workers=0,
        pin_memory=(device or torch.device("cpu")).type == "cuda",
        generator=dataloader_generator(base_seed, stream=split_name),
        worker_init_fn=dataloader_worker_init_fn,
    )


def _bind_streaming_loader_paths(config: dict, *, root: Path) -> None:
    """Bind checkpoint-owned relative loader paths to one recorded data root."""

    root = root.resolve()
    for field in ("sources", "datasets"):
        entries = config.get(field, [])
        if not isinstance(entries, list):
            continue
        for source in entries:
            if not isinstance(source, dict):
                continue
            if source.get("type", source.get("source", "hf")) != "manifest":
                continue
            manifest_path = Path(str(source["manifest_path"]))
            if not manifest_path.is_absolute():
                manifest_path = root / manifest_path
            source["manifest_path"] = str(manifest_path.resolve())
    cache = config.get("cache")
    if not isinstance(cache, dict) or cache.get("dir") is None:
        return
    cache_dir = Path(str(cache["dir"]))
    if not cache_dir.is_absolute():
        cache_dir = root / cache_dir
    cache["dir"] = str(cache_dir.resolve())


def build_validation_loader_factory(
    cfg: DictConfig,
    *,
    tokenizer=None,
    device: torch.device | None = None,
    data_root: Path | None = None,
):
    """Return a fresh fixed-window validation loader for each scoring event."""

    data_mode = cfg.data.get("mode")
    if data_mode == "streaming":
        return lambda: build_streaming_dataloader(
            cfg,
            "validation",
            device=device,
            data_root=data_root,
        )
    if data_mode != "memorization_smoke":
        raise ValueError("data.mode must be either 'memorization_smoke' or 'streaming'")
    if tokenizer is None:
        tokenizer = CanonicalTokenizer.from_config(cfg.tokenizer)
    smoke_manifest = resolve_memorization_smoke(cfg.data)
    token_ids = resolved_manifest_token_ids(smoke_manifest, tokenizer)

    def factory():
        loader = create_autoregressive_dataloader(
            token_ids=list(token_ids),
            seq_len=cfg.training.sequence_length,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            generator=dataloader_generator(int(cfg.reproducibility.seed), stream="validation"),
            worker_init_fn=dataloader_worker_init_fn,
        )
        # Map-style memorization smoke has no stream dataset object, but its
        # manifest remains part of the evaluation identity.
        loader.dataset.resolved_manifests = {smoke_manifest.name: smoke_manifest}
        loader.dataset.config = {
            "datasets": [
                {
                    "name": smoke_manifest.name,
                    "type": "manifest",
                    "expected_fingerprint": smoke_manifest.manifest_fingerprint,
                    "selection": smoke_manifest.selection,
                }
            ]
        }
        return loader

    return factory


def validate_streaming_dataloaders(train_loader, validation_loader) -> None:
    validate_disjoint_manifests(
        train_loader.dataset.resolved_manifests,
        validation_loader.dataset.resolved_manifests,
    )


def preview_streaming_batch(cfg: DictConfig, *, device: torch.device):
    """Build the human preview from an isolated stream, never the train cursor."""

    preview_loader = build_streaming_dataloader(cfg, "train", device=device)
    iterator = iter(preview_loader)
    try:
        return next(iterator)
    except StopIteration as error:
        raise ConfigPreflightError("streaming train profile produced no preview batch") from error
    finally:
        Trainer._close_train_iterator(iterator)


def log_loader_size(name: str, loader) -> None:
    try:
        size = len(loader.dataset)
    except TypeError:
        logger.info("{} windows: streaming/unknown", name)
        return
    logger.info("{} windows: {}", name, size)


@contextmanager
def _exclusive_run_preparation(run_dir: Path):
    """Serialize construction so timestamp-colliding launches cannot share a run."""

    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".run-preparation.lock"
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ConfigPreflightError(
            f"run directory is already being prepared by another process: {run_dir}"
        ) from error
    try:
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def prepare_trainer(cfg: DictConfig, *, run_dir: Path | None = None) -> Trainer:
    """Assemble the canonical training path without beginning optimizer updates.

    The normal Hydra entrypoint and bounded experiment scripts share this one
    construction path.  ``run_dir`` is an operational output location, not a
    Hydra/model/data setting, so it never changes checkpoint compatibility.
    """

    run_dir = _run_directory() if run_dir is None else Path(run_dir)
    validate_training_config(cfg)
    with _exclusive_run_preparation(run_dir):
        return _prepare_trainer_locked(cfg, run_dir=run_dir)


def _prepare_trainer_locked(cfg: DictConfig, *, run_dir: Path) -> Trainer:
    checkpoint_dir = run_dir / Path(cfg.artifacts.checkpoints_dir)
    resume_path = cfg.artifacts.get("resume_path")
    manifest_path = run_dir / "run_manifest.json"
    if resume_path is None and manifest_path.exists():
        raise ConfigPreflightError(
            "refusing a fresh launch into an occupied run directory; "
            f"existing run manifest: {manifest_path}"
        )
    inherited_run_lineage = None
    if resume_path is not None:
        inherited_run_lineage = load_run_lineage_from_resume(
            resume_path, checkpoint_dir=checkpoint_dir
        )
    seed_everything(
        int(cfg.reproducibility.seed),
        deterministic=bool(cfg.reproducibility.get("deterministic", True)),
    )
    device = select_device(cfg.runtime.device)
    logger.info("Using device: {}", device)
    resolved_config_path = save_resolved_config(cfg, run_dir=run_dir)
    logger.info("Resolved Hydra config: {}", resolved_config_path)
    tokenizer_config = build_tokenizer_config(cfg)
    run_manifest_path = write_run_manifest(
        cfg=to_plain_config(cfg),
        run_dir=run_dir,
        root_dir=ROOT_DIR,
        resolved_config_path=resolved_config_path,
        tokenizer_manifest_path=ROOT_DIR / tokenizer_config["manifest_path"],
        tokenizer_expected_fingerprint=tokenizer_config.get("expected_fingerprint"),
        run_lineage_id=inherited_run_lineage,
    )
    logger.info("Run manifest: {}", run_manifest_path)

    data_mode = cfg.data.get("mode")
    logger.info("Loading tokenizer artifact...")
    tokenizer = CanonicalTokenizer.from_config(cfg.tokenizer)
    logger.info("Tokenizer vocab size: {}", tokenizer.vocab_size)
    logger.info("Tokenizer fingerprint: {}", tokenizer.fingerprint)

    checkpoint_identity = build_checkpoint_identity(
        cfg,
        run_manifest_path=run_manifest_path,
    )
    if resume_path is not None:
        # Read and compare the full checkpoint header before opening the train
        # stream. Trainer performs the same verified load immediately before
        # mutation, after model/optimizer construction.
        resume_checkpoint = CheckpointManager(
            checkpoint_dir,
            keep_last_n=int(cfg.artifacts.keep_last_n),
            identity=checkpoint_identity,
        ).load_resume(resume_path)
        require_exact_stream_resume_state(resume_checkpoint.payload["state"])
        logger.info("Resume checkpoint compatibility preflight passed: {}", resume_path)

    smoke_manifest = None
    if data_mode == "memorization_smoke":
        smoke_manifest = resolve_memorization_smoke(cfg.data)

    if data_mode == "memorization_smoke":
        assert smoke_manifest is not None
        train_token_ids = resolved_manifest_token_ids(smoke_manifest, tokenizer)
        logger.info(
            "Explicit memorization smoke uses manifest {} for both train and validation",
            smoke_manifest.manifest_fingerprint,
        )

        logger.info("Building decoder-only autoregressive dataloader...")
        train_loader = create_autoregressive_dataloader(
            token_ids=train_token_ids,
            seq_len=cfg.training.sequence_length,
            batch_size=cfg.training.batch_size,
            shuffle=cfg.training.shuffle,
            generator=dataloader_generator(int(cfg.reproducibility.seed), stream="train"),
            worker_init_fn=dataloader_worker_init_fn,
        )
        validation_loader_factory = build_validation_loader_factory(
            cfg, tokenizer=tokenizer, device=device
        )
        validation_loader = validation_loader_factory()
    elif data_mode == "streaming":
        logger.info("Building streaming causal-LM dataloaders...")
        train_loader = build_streaming_dataloader(cfg, "train", device=device)
        validation_loader = build_streaming_dataloader(cfg, "validation", device=device)
        validate_streaming_dataloaders(train_loader, validation_loader)
        validation_loader_factory = build_validation_loader_factory(cfg, device=device)
    else:
        raise ValueError("data.mode must be either 'memorization_smoke' or 'streaming'")

    if data_mode == "streaming":
        preview_batch = preview_streaming_batch(cfg, device=device)
    else:
        preview_batch = next(iter(train_loader))
    log_sample_batch(tokenizer, preview_batch)
    log_loader_size("Training", train_loader)
    log_loader_size("Validation", validation_loader)

    logger.info("Building model...")
    model = SimpleDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_size=cfg.model.embed_size,
        num_heads=cfg.model.num_heads,
        max_len=cfg.training.sequence_length,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.to(device)
    model.train()
    parameter_counts = get_parameter_counts(model)
    logger.info(
        "Model parameters: total={:,} trainable={:,} frozen={:,}",
        parameter_counts.total,
        parameter_counts.trainable,
        parameter_counts.non_trainable,
    )

    optimizer = build_optimizer(model, cfg.training.optimizer)
    logger.info("Using optimizer: {}", optimizer.__class__.__name__)

    scheduler = build_scheduler(optimizer, cfg.training.get("scheduler"))
    if scheduler is None:
        logger.info("Learning rate scheduler: disabled")
    else:
        scheduler_interval = cfg.training.scheduler.get("interval", "epoch")
        logger.info(
            "Learning rate scheduler: {} ({})",
            scheduler.__class__.__name__,
            scheduler_interval,
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        validation_loader_factory=validation_loader_factory,
        checkpoint_dir=checkpoint_dir,
        cfg=cfg,
        device=device,
        checkpoint_identity=checkpoint_identity,
        resume_path=resume_path,
    )
    return trainer


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    """Run the canonical Hydra training path."""

    prepare_trainer(cfg).fit()


def run_config_check(cfg: DictConfig) -> Path:
    """Preflight config and immutable metadata without consuming remote data."""
    validate_training_config(cfg)
    validate_profile_manifests(cfg)
    resolved_config_path = save_resolved_config(cfg)
    logger.info("Config preflight passed: {}", resolved_config_path)
    return resolved_config_path


@hydra.main(version_base=None, config_path="../config", config_name="train")
def config_check(cfg: DictConfig) -> None:
    """Compose and metadata-preflight one canonical profile."""

    run_config_check(cfg)


if __name__ == "__main__":
    main()
