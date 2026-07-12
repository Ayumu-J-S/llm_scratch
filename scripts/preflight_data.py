"""Hydra-driven, bounded DATA-004 aggregate data preflight."""

from __future__ import annotations

import copy
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from data.manifests import validate_disjoint_manifests
from data.preflight import (
    REPORT_SCHEMA_VERSION,
    DataPreflightError,
    assert_cache_released,
    assert_cold_cache_is_empty,
    assert_loader_accounting,
    audit_raw_stream,
    cache_telemetry,
    code_fingerprint,
    config_fingerprint,
    data_fingerprints,
    disk_forecast,
    merge_loader_quality,
    summarize_packing,
    write_reports,
)
from data.stream_loader.loader import StreamLoader
from runtime.config import validate_training_config
from tokenizer.canonical import CanonicalTokenizer


ROOT = Path(__file__).resolve().parents[1]


def _plain(value: Any) -> dict[str, Any]:
    result = OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else value
    if not isinstance(result, Mapping):
        raise TypeError("configuration must resolve to a mapping")
    return copy.deepcopy(dict(result))


def _preflight_options(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("preflight", {})
    if not isinstance(raw, Mapping):
        raise DataPreflightError("preflight must be a mapping")
    allowed = {
        "cold",
        "max_documents_per_split",
        "ratio_tolerance",
        "output_dir",
        "report_stem",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise DataPreflightError(f"unknown preflight option(s): {unknown}")
    max_documents = int(raw.get("max_documents_per_split", 4096))
    ratio_tolerance = float(raw.get("ratio_tolerance", 0.01))
    report_stem = str(raw.get("report_stem", "data_preflight"))
    if max_documents < 1:
        raise DataPreflightError("preflight.max_documents_per_split must be positive")
    if not 0.0 <= ratio_tolerance <= 1.0:
        raise DataPreflightError("preflight.ratio_tolerance must be between zero and one")
    if not report_stem or Path(report_stem).name != report_stem:
        raise DataPreflightError("preflight.report_stem must be a plain file stem")
    return {
        "cold": bool(raw.get("cold", False)),
        "max_documents_per_split": max_documents,
        "ratio_tolerance": ratio_tolerance,
        "output_dir": raw.get("output_dir"),
        "report_stem": report_stem,
    }


def _split_config(config: Mapping[str, Any], split: str) -> dict[str, Any]:
    streaming = config["data"]["streaming"]
    common = {key: value for key, value in streaming.items() if key not in {"train", "validation"}}
    result = {**common, **streaming[split]}
    if "sources" in result and "datasets" not in result:
        result["datasets"] = result.pop("sources")
    result["tokenizer"] = config["tokenizer"]
    result["seed"] = int(config["reproducibility"]["seed"]) + int(split == "validation")
    return result


def _raw_config(config: Mapping[str, Any], split: str) -> dict[str, Any]:
    result = _split_config(config, split)
    result.update(
        {
            "output_mode": "raw_text",
            "max_tokens": "max",
            "mixture_basis": "tokenizer_tokens",
            "preserve_metadata": True,
            "repeat": False,
        }
    )
    result.pop("max_target_tokens", None)
    return result


def _packing_config(config: Mapping[str, Any]) -> dict[str, Any]:
    result = _split_config(config, "train")
    if result.get("mixture_basis") != "trained_targets":
        raise DataPreflightError("train preflight requires mixture_basis=trained_targets")
    if result.get("max_target_tokens") is None:
        raise DataPreflightError("train preflight requires max_target_tokens")
    result.update(
        {
            "output_mode": "packed_sequences",
            "sequence_length": int(config["training"]["sequence_length"]) + 1,
            "drop_remainder": True,
            "preserve_metadata": True,
        }
    )
    return result


def _cache_config(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    cache = config["data"]["streaming"].get("cache", {})
    if not isinstance(cache, Mapping) or not cache.get("dir"):
        raise DataPreflightError("DATA-004 preflight requires data.streaming.cache.dir")
    path = Path(str(cache["dir"]))
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve(), dict(cache)


def _largest_shard(loaders: list[StreamLoader]) -> int:
    sizes: list[int] = []
    for loader in loaders:
        for manifest in loader.resolved_manifests.values():
            if manifest.source is None:
                continue
            sizes.extend(int(item["size_bytes"]) for item in manifest.source["data_files"])
    return max(sizes, default=0)


def _quality(loader: StreamLoader, audit: dict[str, Any]) -> None:
    merge_loader_quality(
        audit,
        rejection_counts=loader.rejection_counts,
        fallback_counts=loader.fallback_counts,
        truncated_counts=loader.truncated_counts,
    )


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    resolved = _plain(cfg)
    options = _preflight_options(resolved)
    training_config = {key: value for key, value in resolved.items() if key != "preflight"}
    validate_training_config(training_config)
    if training_config["data"]["mode"] != "streaming":
        raise DataPreflightError("DATA-004 preflight requires a streaming profile")

    cache_dir, cache_config = _cache_config(training_config)
    if options["cold"]:
        assert_cold_cache_is_empty(cache_dir)

    tokenizer = CanonicalTokenizer.from_config(training_config["tokenizer"])
    train_audit_loader = StreamLoader(_raw_config(training_config, "train"))
    validation_audit_loader = StreamLoader(_raw_config(training_config, "validation"))
    validate_disjoint_manifests(
        train_audit_loader.resolved_manifests,
        validation_audit_loader.resolved_manifests,
    )

    train_audit = audit_raw_stream(
        train_audit_loader,
        tokenizer,
        max_documents=options["max_documents_per_split"],
        add_eos=bool(train_audit_loader.add_eos),
    )
    validation_audit = audit_raw_stream(
        validation_audit_loader,
        tokenizer,
        max_documents=options["max_documents_per_split"],
        add_eos=bool(validation_audit_loader.add_eos),
    )
    _quality(train_audit_loader, train_audit.report)
    _quality(validation_audit_loader, validation_audit.report)
    overlap = train_audit.content_hashes & validation_audit.content_hashes
    if overlap:
        raise DataPreflightError(
            f"observed train/validation normalized-content overlap: {len(overlap)}"
        )

    packing_loader = StreamLoader(_packing_config(training_config))
    ratios = {
        str(source["name"]): float(source["ratio"]) for source in packing_loader.dataset_configs
    }
    packing = summarize_packing(
        packing_loader,
        expected_ratios=ratios,
        tolerance=options["ratio_tolerance"],
    )
    assert_loader_accounting(packing_loader, packing)
    if not packing["ratios_within_tolerance"]:
        raise DataPreflightError("realized trained-target ratios exceed declared tolerance")

    loaders = [train_audit_loader, validation_audit_loader, packing_loader]
    largest_shard = _largest_shard(loaders)
    telemetry = cache_telemetry(loaders)
    assert_cache_released(telemetry)
    disk = disk_forecast(
        cache_dir,
        cache_cap_bytes=int(cache_config["max_size_bytes"]),
        current_managed_cache_bytes=int(telemetry["size_bytes"]),
        reserved_os_checkpoint_bytes=int(cache_config.get("min_free_bytes", 0)),
        largest_shard_bytes=largest_shard,
    )
    if not disk["headroom_admission_passed"]:
        raise DataPreflightError("cache cap or filesystem headroom fails DATA-004 admission")

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "mode": "cold" if options["cold"] else "warm",
        "limits": {
            "max_documents_per_split": options["max_documents_per_split"],
            "max_target_tokens": int(_packing_config(training_config)["max_target_tokens"]),
        },
        "fingerprints": {
            "code": code_fingerprint(ROOT),
            "config": config_fingerprint(resolved),
            "data": data_fingerprints(loaders),
            "tokenizer": tokenizer.fingerprint,
        },
        "integrity": {
            "manifest_disjointness_validated": True,
            "observed_normalized_content_overlap": 0,
            "raw_text_retained": False,
        },
        "splits": {
            "train": train_audit.report,
            "validation": validation_audit.report,
        },
        "packing": packing,
        "cache": telemetry,
        "disk": disk,
    }
    output_dir_value = options["output_dir"]
    output_dir = (
        Path(str(output_dir_value))
        if output_dir_value
        else Path(HydraConfig.get().runtime.output_dir)
    )
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    stem = options["report_stem"]
    write_reports(report, output_dir / f"{stem}.json", output_dir / f"{stem}.md")
    print(output_dir / f"{stem}.json")


def _translate_cold_flag(arguments: list[str]) -> list[str]:
    """Keep the operator-friendly --cold spelling while storing it in Hydra."""

    if "--cold" not in arguments:
        return arguments
    if any(argument.endswith("preflight.cold=true") for argument in arguments):
        raise DataPreflightError("--cold and preflight.cold=true are redundant")
    result = [argument for argument in arguments if argument != "--cold"]
    result.append("+preflight.cold=true")
    return result


if __name__ == "__main__":
    sys.argv = _translate_cold_flag(sys.argv)
    main()
