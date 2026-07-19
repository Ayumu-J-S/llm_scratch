"""Record one aggregate-only external baseline comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from benchmarks.external import ExternalComparisonError, write_external_comparison


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write an isolated aggregate-only same-protocol external comparison."
    )
    parser.add_argument("--input", required=True, help="aggregate comparison JSON")
    parser.add_argument(
        "--output",
        required=True,
        help="JSON name/path beneath outputs/external-comparisons",
    )
    args = parser.parse_args(argv)
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ExternalComparisonError("external comparison input must be a JSON object")
        write_external_comparison(
            payload,
            output_path=args.output,
        )
    except (OSError, json.JSONDecodeError, ExternalComparisonError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
