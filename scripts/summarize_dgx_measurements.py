#!/usr/bin/env python3
"""Gate repeated DGX-001 evidence and select one baseline profile."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from dgx.planning import summarize_evidence


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@hydra.main(version_base=None, config_path="../config", config_name="dgx")
def main(config: DictConfig) -> None:
    if not config.output_root:
        raise RuntimeError("output_root is required")
    root = Path(str(config.output_root)).resolve()
    summary = summarize_evidence(root, config)
    destination = root / "dgx-summary.json"
    _atomic_json(destination, summary)
    print(destination)
    if summary["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
