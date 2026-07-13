#!/usr/bin/env python
"""Summarize local W&B records without syncing or mutating retained evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any

import wandb
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_watch_histograms(items: Any) -> list[str]:
    groups: dict[str, dict[str, Any]] = {}
    for item in items:
        nested = list(item.nested_key)
        if len(nested) != 2 or not nested[0].startswith("gradients/"):
            continue
        try:
            groups.setdefault(nested[0], {})[nested[1]] = json.loads(item.value_json)
        except json.JSONDecodeError:
            continue

    valid = []
    for name, fields in groups.items():
        values = fields.get("values")
        bins = fields.get("bins")
        numeric_values = isinstance(values, list) and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            for value in values
        )
        numeric_bins = isinstance(bins, list) and all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in bins
        )
        if (
            fields.get("_type") == "histogram"
            and numeric_values
            and values
            and sum(values) > 0
            and numeric_bins
            and len(bins) == len(values) + 1
            and all(left <= right for left, right in zip(bins, bins[1:]))
        ):
            valid.append(name)
    return sorted(valid)


def inspect_file(source: Path) -> dict[str, Any]:
    record_types: Counter[str] = Counter()
    watch_records = 0
    watch_histograms = 0
    watch_series: set[str] = set()
    watch_steps: list[int] = []
    history_records = 0
    with tempfile.TemporaryDirectory(prefix="wb001-wandb-scan-") as directory:
        copy = Path(directory) / source.name
        shutil.copyfile(source, copy)
        store = DataStore()
        store.open_for_scan(str(copy))
        while (data := store.scan_data()) is not None:
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            record_type = record.WhichOneof("record_type") or "unknown"
            record_types[record_type] += 1
            if record_type != "history":
                continue
            history_records += 1
            series = _valid_watch_histograms(record.history.item)
            if series:
                watch_records += 1
                watch_histograms += len(series)
                watch_series.update(series)
                if record.history.HasField("step"):
                    watch_steps.append(record.history.step.num)
        store.close()
    return {
        "path": str(source),
        "sha256": _sha256(source),
        "size_bytes": source.stat().st_size,
        "record_types": dict(sorted(record_types.items())),
        "history_records": history_records,
        "watch_histogram_records": watch_records,
        "watch_histograms": watch_histograms,
        "watch_histogram_series": sorted(watch_series),
        "watch_history_steps": watch_steps,
    }


def inspect_root(root: Path) -> dict[str, Any]:
    files = [inspect_file(path) for path in sorted(root.rglob("*.wandb"))]
    return {
        "schema_version": 1,
        "wandb_version": wandb.__version__,
        "file_count": len(files),
        "watch_histogram_records": sum(item["watch_histogram_records"] for item in files),
        "watch_histograms": sum(item["watch_histograms"] for item in files),
        "watch_histogram_series": sorted(
            {series for item in files for series in item["watch_histogram_series"]}
        ),
        "watch_history_steps": sorted(
            {step for item in files for step in item["watch_history_steps"]}
        ),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_root(args.root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
