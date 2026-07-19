"""Hydra entrypoints for development and guarded reserved-final benchmarks."""

from __future__ import annotations

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from benchmarks.runner import run_benchmark
from benchmarks.suite import FINAL_ACKNOWLEDGEMENT


CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "config")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train")
def main(cfg: DictConfig) -> None:
    """Run the deterministic development subsets; final access is impossible here."""

    run_benchmark(cfg, access="dev")


@hydra.main(version_base=None, config_path=CONFIG_DIR, config_name="train")
def _final_hydra(cfg: DictConfig) -> None:
    acknowledgement = os.environ.get("BENCHMARK_FINAL_ACK")
    run_benchmark(
        cfg,
        access="final",
        final_acknowledgement=acknowledgement,
    )


def final_main() -> None:
    """Require a non-Hydra acknowledgement before composing the final run."""

    if os.environ.get("BENCHMARK_FINAL_ACK") != FINAL_ACKNOWLEDGEMENT:
        raise SystemExit(
            f"reserved final benchmark requires BENCHMARK_FINAL_ACK={FINAL_ACKNOWLEDGEMENT}"
        )
    _final_hydra()


if __name__ == "__main__":
    main()
