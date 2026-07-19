"""One explicit command surface for bounded experiment operation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from operations.artifacts import AttemptError


ROOT_DIR = Path(__file__).resolve().parent.parent
_ACTIONS = (
    "config-check",
    "preflight",
    "smoke",
    "train",
    "resume",
    "eval",
    "benchmark",
    "status",
    "handoff",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-scratch-ops",
        description=("Operate one bounded attempt. Put Hydra overrides after a literal '--'."),
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in _ACTIONS:
        command = subparsers.add_parser(action)
        command.add_argument("--run-root", type=Path, required=True)
        command.add_argument("--run-id", required=True)
        if action not in {"status", "handoff"}:
            command.add_argument("--attempt-id", required=True)
            command.add_argument("--executor", choices=("host", "container"), required=True)
            command.add_argument("--device", choices=("cpu", "cuda"), required=True)
            command.add_argument("--image")
            command.add_argument("--retry-from")
            command.add_argument("--experiment-record", type=Path)
            if action in {"preflight", "train"}:
                command.add_argument("--profile", required=True)
        else:
            command.add_argument("--attempt-id", required=True)
        if action == "resume":
            command.add_argument("--checkpoint", type=Path, required=True)
        if action in {"eval", "benchmark"}:
            command.add_argument("--checkpoint", type=Path, required=True)
    return parser


def _split_overrides(argv: Sequence[str]) -> tuple[list[str], list[str], bool]:
    values = list(argv)
    if "--" not in values:
        return values, [], False
    index = values.index("--")
    return values[:index], values[index + 1 :], True


def main(argv: Sequence[str] | None = None) -> int:
    arguments, overrides, had_separator = _split_overrides(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(arguments)
    if args.action in {"status", "handoff"}:
        if had_separator or overrides:
            raise AttemptError(f"{args.action} does not accept Hydra overrides")
    elif not had_separator:
        raise AttemptError("Hydra-capable commands require a literal '--' separator")
    if getattr(args, "executor", None) == "container" and not args.image:
        raise AttemptError("container execution requires --image")
    if getattr(args, "executor", None) == "host" and getattr(args, "image", None):
        raise AttemptError("--image is valid only with --executor container")

    # Command implementations are added behind this stable parser surface.
    from operations.runner import dispatch

    return dispatch(args, overrides, root_dir=ROOT_DIR)


if __name__ == "__main__":
    raise SystemExit(main())
