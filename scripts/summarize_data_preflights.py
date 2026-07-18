"""Summarize comparable DATA-004 preflight observations without rerunning work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data.preflight import summarize_repeated_reports


def summarize_paths(paths: list[Path]) -> dict[str, Any]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return summarize_repeated_reports(reports)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retain median and spread across comparable DATA-004 reports."
    )
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = summarize_paths(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
