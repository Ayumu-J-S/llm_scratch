from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, IterableDataset

from data.stream_loader import StreamLoader
from data.stream_loader.loader import preflight_stream_manifests
from tokenizer.canonical import CanonicalTokenizer


class StreamingTokenDataset(IterableDataset):
    def __init__(
        self,
        config: Mapping[str, Any] | DictConfig,
        sequence_length: int,
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")

        self.sequence_length = int(sequence_length)
        self.window_length = self.sequence_length + 1
        self.config = _stream_loader_config(config, window_length=self.window_length)
        CanonicalTokenizer.from_config(self.config.get("tokenizer"))
        self.resolved_manifests = preflight_stream_manifests(self.config)

    def __iter__(self):
        with StreamLoader(
            self.config,
            resolved_manifests=self.resolved_manifests,
        ) as loader:
            for sample in loader:
                input_ids = torch.as_tensor(sample["input_ids"], dtype=torch.long)
                if input_ids.numel() != self.window_length:
                    raise ValueError(
                        "StreamingTokenDataset expected packed windows of "
                        f"{self.window_length} tokens, got {input_ids.numel()}"
                    )
                yield {"input_ids": input_ids}


def causal_lm_collate_fn(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([sample["input_ids"] for sample in samples]).long()
    if input_ids.size(1) < 2:
        raise ValueError("causal LM batches require at least two tokens per sample")

    return {
        "inputs": input_ids[:, :-1].contiguous(),
        "labels": input_ids[:, 1:].contiguous(),
    }


def create_streaming_token_dataloader(
    *,
    config: Mapping[str, Any] | DictConfig,
    sequence_length: int,
    batch_size: int,
    drop_last: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = StreamingTokenDataset(
        config=config,
        sequence_length=sequence_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=causal_lm_collate_fn,
    )


def _stream_loader_config(
    config: Mapping[str, Any] | DictConfig,
    *,
    window_length: int,
) -> dict[str, Any]:
    config_dict = _to_plain_dict(config)
    if "sources" in config_dict and "datasets" not in config_dict:
        config_dict["datasets"] = config_dict.pop("sources")

    config_dict["output_mode"] = "packed_sequences"
    config_dict["sequence_length"] = window_length
    config_dict["drop_remainder"] = True
    return config_dict


def _to_plain_dict(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        container = OmegaConf.to_container(config, resolve=True)
        if not isinstance(container, dict):
            raise TypeError("streaming config must be a mapping")
        return copy.deepcopy(container)
    if isinstance(config, Mapping):
        return copy.deepcopy(dict(config))
    raise TypeError("streaming config must be a mapping")
