from pathlib import Path

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from data.streaming_dataset import create_streaming_token_dataloader
from data.text_dataset import create_autoregressive_dataloader
from models.simple_decoder_transformer import SimpleDecoderTransformer
from training.optimization import build_optimizer, build_scheduler
from training.trainer import Trainer
from tokenizer.artifacts import load_text, load_tokenizer
from utils.model import get_parameter_counts


ROOT_DIR = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_token_ids(input_path: str, tokenizer, split_name: str):
    split_label = split_name.capitalize()
    logger.info("Loading {} corpus from: {}", split_name, input_path)
    text = load_text(input_path)
    logger.info("Tokenizing {} corpus...", split_name)
    token_ids = tokenizer.encode(text)
    logger.info("{} corpus length: {} tokens", split_label, len(token_ids))
    return token_ids


def build_tokenizer_config(cfg: DictConfig) -> dict[str, str]:
    tokenizer_path = ROOT_DIR / cfg.artifacts.tokenizers_dir / cfg.artifacts.tokenizer_filename
    return {"kind": "bpe", "path": str(tokenizer_path)}


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
        key: value
        for key, value in config.items()
        if key not in {"train", "validation"}
    }
    return {**common_config, **split_config}


def build_streaming_dataloader(cfg: DictConfig, split_name: str):
    stream_config = streaming_split_config(cfg.data.streaming, split_name)
    stream_config["tokenizer"] = build_tokenizer_config(cfg)
    return create_streaming_token_dataloader(
        config=stream_config,
        sequence_length=cfg.training.sequence_length,
        batch_size=cfg.training.batch_size,
        drop_last=split_name == "train",
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )


def log_loader_size(name: str, loader) -> None:
    try:
        size = len(loader.dataset)
    except TypeError:
        logger.info("{} windows: streaming/unknown", name)
        return
    logger.info("{} windows: {}", name, size)


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    logger.info("Using device: {}", DEVICE)

    logger.info("Loading tokenizer artifact...")
    tokenizer = load_tokenizer(
        cfg.artifacts.tokenizers_dir,
        cfg.artifacts.tokenizer_filename,
    )
    logger.info("Tokenizer vocab size: {}", tokenizer.vocab_size)

    data_mode = cfg.data.get("mode", "local_text")
    if data_mode == "local_text":
        train_token_ids = load_token_ids(
            cfg.data.train,
            tokenizer,
            "training",
        )
        validation_token_ids = load_token_ids(
            cfg.data.val,
            tokenizer,
            "validation",
        )

        if cfg.data.val == cfg.data.train:
            logger.info(
                "Validation uses the same corpus as training. "
                "This run is measuring memorization/overfitting.",
            )

        logger.info("Building decoder-only autoregressive dataloader...")
        train_loader = create_autoregressive_dataloader(
            token_ids=train_token_ids,
            seq_len=cfg.training.sequence_length,
            batch_size=cfg.training.batch_size,
            shuffle=cfg.training.shuffle,
        )
        validation_loader = create_autoregressive_dataloader(
            token_ids=validation_token_ids,
            seq_len=cfg.training.sequence_length,
            batch_size=cfg.training.batch_size,
            shuffle=False,
        )
    elif data_mode == "streaming":
        logger.info("Building streaming causal-LM dataloaders...")
        train_loader = build_streaming_dataloader(cfg, "train")
        validation_loader = build_streaming_dataloader(cfg, "validation")
    else:
        raise ValueError("data.mode must be either 'local_text' or 'streaming'")

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
    model.to(DEVICE)
    model.train()
    parameter_counts = get_parameter_counts(model)
    logger.info(
        "Model parameters: "
        "total={:,} "
        "trainable={:,} "
        "frozen={:,}",
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

    checkpoint_dir = ROOT_DIR / cfg.artifacts.checkpoints_dir
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        validation_loader=validation_loader,
        checkpoint_dir=checkpoint_dir,
        cfg=cfg,
        device=DEVICE,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
