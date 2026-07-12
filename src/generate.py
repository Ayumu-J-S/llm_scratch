"""CLI for bounded checkpoint-based base-model continuations."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from generation.sampler import CheckpointSampler, SamplingError


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Generate a labeled base-model continuation from a full-state checkpoint."
    )
    command.add_argument("--checkpoint", required=True, help="verified full-state checkpoint path")
    command.add_argument("--prompt", required=True, help="text prompt to continue")
    command.add_argument("--max-new-tokens", required=True, type=int)
    command.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="positive value enables seeded sampling; omit for greedy decoding",
    )
    command.add_argument("--top-k", type=int, default=None)
    command.add_argument("--seed", type=int, default=None)
    command.add_argument("--device", default="cpu", help="generation device, default: cpu")
    command.add_argument("--json", action="store_true", help="emit result metadata as JSON")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        sampler = CheckpointSampler.from_checkpoint(args.checkpoint, device=args.device)
        result = sampler.generate(
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )
    except (OSError, RuntimeError, SamplingError) as error:
        parser().error(str(error))
    metadata = result.metadata()
    if args.json:
        print(json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    else:
        print("label: base-model-continuation")
        print(f"checkpoint: {metadata['checkpoint_path']} ({metadata['checkpoint_kind']})")
        print(f"stop_reason: {metadata['stop_reason']}")
        print("completion:")
        print(result.completion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
