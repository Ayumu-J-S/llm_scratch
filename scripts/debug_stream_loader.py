from __future__ import annotations

import argparse
import copy
import gc
import sys
import time
from collections.abc import Mapping
from pathlib import Path

from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.streaming_dataset import create_streaming_token_dataloader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview streaming causal-LM batches.")
    parser.add_argument(
        "--config",
        default="config/stream_loader.yaml",
        help="Path to config/stream_loader.yaml or config/train.yaml.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "validation"],
        help="Split to preview when using config/train.yaml.",
    )
    parser.add_argument("--limit", type=int, default=1, help="Number of batches to print.")
    parser.add_argument("--batch-size", type=int, default=2, help="Debug batch size.")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Training sequence length. Defaults to config value, then 64.",
    )
    parser.add_argument(
        "--preview-tokens",
        type=int,
        default=32,
        help="Number of tokens to print from the first row.",
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Use the config prefetch setting. The default preview path disables prefetch.",
    )
    parser.add_argument(
        "--max-doc-chars",
        type=int,
        default=50_000,
        help="Cap each document before tokenization for preview speed. Use 0 for full documents.",
    )
    parser.add_argument(
        "--shutdown-grace-seconds",
        type=float,
        default=5.0,
        help="Grace period after closing HF streams before interpreter shutdown.",
    )
    args = parser.parse_args()

    config, config_sequence_length = load_debug_config(Path(args.config), split=args.split)
    sequence_length = args.sequence_length or config_sequence_length or 64
    validate_args(args, sequence_length=sequence_length)
    apply_debug_overrides(
        config,
        prefetch_enabled=args.prefetch,
        max_doc_chars=args.max_doc_chars,
    )

    uses_hf_streams = any(is_hf_source(dataset) for dataset in config.get("datasets", []))
    loader = create_streaming_token_dataloader(
        config=config,
        sequence_length=sequence_length,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    iterator = iter(loader)

    try:
        for index in range(args.limit):
            batch = next(iterator)
            print_batch(index, batch, preview_tokens=args.preview_tokens)
    except StopIteration:
        pass
    finally:
        del iterator
        del loader
        gc.collect()
        if uses_hf_streams and args.shutdown_grace_seconds > 0:
            time.sleep(args.shutdown_grace_seconds)


def load_debug_config(path: Path, *, split: str) -> tuple[dict, int | None]:
    raw_config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw_config, Mapping):
        raise TypeError("debug config must be a mapping")

    config = copy.deepcopy(dict(raw_config))
    data_config = config.get("data")
    if isinstance(data_config, Mapping) and isinstance(data_config.get("streaming"), Mapping):
        return load_train_streaming_config(config, split=split)

    if "sources" in config and "datasets" not in config:
        config["datasets"] = config.pop("sources")
    sequence_length = config.get("sequence_length")
    return config, None if sequence_length is None else int(sequence_length)


def load_train_streaming_config(config: Mapping, *, split: str) -> tuple[dict, int | None]:
    data_config = as_mapping(config.get("data"), name="data")
    streaming_config = as_mapping(data_config.get("streaming"), name="data.streaming")
    split_config = as_mapping(
        streaming_config.get(split),
        name=f"data.streaming.{split}",
    )
    common_config = {
        key: value
        for key, value in streaming_config.items()
        if key not in {"train", "validation"}
    }
    debug_config = {**common_config, **dict(split_config)}
    if "sources" in debug_config and "datasets" not in debug_config:
        debug_config["datasets"] = debug_config.pop("sources")

    artifacts_config = as_optional_mapping(config.get("artifacts"))
    if artifacts_config is not None and "tokenizer" not in debug_config:
        tokenizer_path = (
            ROOT_DIR
            / str(artifacts_config["tokenizers_dir"])
            / str(artifacts_config["tokenizer_filename"])
        )
        debug_config["tokenizer"] = {"kind": "bpe", "path": str(tokenizer_path)}

    training_config = as_optional_mapping(config.get("training"))
    sequence_length = None
    if training_config is not None and "sequence_length" in training_config:
        sequence_length = int(training_config["sequence_length"])
    return debug_config, sequence_length


def as_mapping(value, *, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be configured")
    return value


def as_optional_mapping(value) -> Mapping | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("expected mapping config")
    return value


def validate_args(args, *, sequence_length: int) -> None:
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if sequence_length < 1:
        raise ValueError("--sequence-length must be positive")
    if args.preview_tokens < 1:
        raise ValueError("--preview-tokens must be positive")
    if args.max_doc_chars < 0:
        raise ValueError("--max-doc-chars must be non-negative")
    if args.shutdown_grace_seconds < 0:
        raise ValueError("--shutdown-grace-seconds must be non-negative")


def apply_debug_overrides(
    config: dict,
    *,
    prefetch_enabled: bool,
    max_doc_chars: int,
) -> None:
    if not prefetch_enabled:
        config.setdefault("prefetch", {})["enabled"] = False
    if max_doc_chars:
        for dataset in config.get("datasets", []):
            dataset["max_text_chars"] = max_doc_chars
        print(
            f"debug: capping each document at {max_doc_chars} chars "
            "before tokenization; pass --max-doc-chars 0 to disable",
            file=sys.stderr,
        )


def is_hf_source(dataset: Mapping) -> bool:
    return dataset.get("type", dataset.get("source", "hf")) == "hf"


def print_batch(index: int, batch: dict, *, preview_tokens: int) -> None:
    inputs = batch["inputs"]
    labels = batch["labels"]
    first_inputs = inputs[0, :preview_tokens].tolist()
    first_labels = labels[0, :preview_tokens].tolist()
    shifted = bool(inputs[0, 1:].equal(labels[0, :-1])) if inputs.size(1) > 1 else True

    print(f"batch={index} batch_size={inputs.size(0)} sequence_length={inputs.size(1)}")
    print(f"inputs.shape={tuple(inputs.shape)} labels.shape={tuple(labels.shape)}")
    print(f"inputs[0,:{preview_tokens}]={first_inputs}")
    print(f"labels[0,:{preview_tokens}]={first_labels}")
    print(f"shift_check={shifted}")


if __name__ == "__main__":
    main()
