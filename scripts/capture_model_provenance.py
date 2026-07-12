#!/usr/bin/env python3
"""Capture privacy-safe Codex model provenance without guessing runtime state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UNAVAILABLE = "not exposed by runtime"


def _git(repo: Path, *args: str) -> str | None:
    try:
        value = subprocess.check_output(["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None
    value = value.strip()
    return value or None


def _cli_version() -> str:
    try:
        value = subprocess.check_output(["codex", "--version"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        return UNAVAILABLE
    return value.strip() or UNAVAILABLE


def _field(value: str, source: str, *, unavailable_reason: str | None = None) -> dict[str, str]:
    result = {"value": value, "source": source, "status": "unavailable" if value == UNAVAILABLE else "observed"}
    if unavailable_reason:
        result["unavailable_reason"] = unavailable_reason
    return result


def capture(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(args.repo).resolve()
    branch = _git(repo, "branch", "--show-current") or UNAVAILABLE
    commit = _git(repo, "rev-parse", "HEAD") or UNAVAILABLE
    actual_id = args.actual_exact_model or UNAVAILABLE
    actual_mode = args.actual_reasoning_mode or UNAVAILABLE
    actual_id_reason = None if args.actual_exact_model else "The active runtime did not expose an exact deployment/model identifier to the caller."
    actual_mode_reason = None if args.actual_reasoning_mode else "The active runtime did not expose a reasoning mode to the caller."
    requested = {
        "model": _field(args.requested_model or UNAVAILABLE, "explicit invocation/config default" if args.requested_model else "not supplied"),
        "reasoning_mode": _field(args.requested_reasoning_mode or UNAVAILABLE, "explicit invocation/config default" if args.requested_reasoning_mode else "not supplied"),
    }
    actual = {
        "product": _field(args.actual_product or UNAVAILABLE, "active runtime display" if args.actual_product else "not exposed by runtime", unavailable_reason=None if args.actual_product else "The runtime did not expose product metadata to the caller."),
        "displayed_model_family": _field(args.actual_model_family or UNAVAILABLE, "active runtime display" if args.actual_model_family else "not exposed by runtime", unavailable_reason=None if args.actual_model_family else "The runtime did not expose a model family to the caller."),
        "exact_model_identifier": _field(actual_id, "active runtime display" if args.actual_exact_model else "not exposed by runtime", unavailable_reason=actual_id_reason),
        "reasoning_mode": _field(actual_mode, "active runtime display" if args.actual_reasoning_mode else "not exposed by runtime", unavailable_reason=actual_mode_reason),
    }
    thread = "not recorded (privacy)"
    if args.include_thread_id and os.environ.get("CODEX_THREAD_ID"):
        thread = hashlib.sha256(os.environ["CODEX_THREAD_ID"].encode()).hexdigest()[:16]
    return {
        "schema_version": "1.0",
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "phase": args.phase,
        "role": args.role,
        "task_path": args.task_path or UNAVAILABLE,
        "requested": requested,
        "actual": actual,
        "environment": {"codex_cli_version": _cli_version(), "branch": branch, "commit": commit, "thread_id": thread},
        "privacy": {"raw_thread_id_recorded": False, "prompts_recorded": False, "hidden_chain_of_thought_recorded": False, "token_counts_recorded": False, "secrets_recorded": False},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--phase", default="implementation")
    parser.add_argument("--role", default="agent")
    parser.add_argument("--task-path")
    parser.add_argument("--requested-model")
    parser.add_argument("--requested-reasoning-mode")
    parser.add_argument("--actual-product")
    parser.add_argument("--actual-model-family")
    parser.add_argument("--actual-exact-model")
    parser.add_argument("--actual-reasoning-mode")
    parser.add_argument("--include-thread-id", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = capture(args)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
