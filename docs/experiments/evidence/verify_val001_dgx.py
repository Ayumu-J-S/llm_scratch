#!/usr/bin/env python3
"""Build and verify durable VAL-001 trajectory/parity evidence projections."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


RUN_ORDER = ["1-off", "1-on", "2-on", "2-off", "3-off", "3-on"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source(path: Path) -> dict[str, Any]:
    return {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}


def _artifact_inventory(path: Path, run_root: Path) -> dict[str, Any]:
    entries = {}
    for line in path.read_text().splitlines():
        digest, artifact = line.split(maxsplit=1)
        artifact_path = Path(artifact)
        entries[str(artifact_path.relative_to(run_root))] = digest
    return {
        **_source(path),
        "entries": entries,
        "entries_sha256": _canonical_json_sha256(entries),
    }


def _checkpoint_projection(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload["state"]["model"]
    canonical = hashlib.sha256()
    tensors = []
    for name, tensor in sorted(state.items()):
        cpu_tensor = tensor.detach().cpu().contiguous()
        data = cpu_tensor.numpy().tobytes()
        dtype = str(cpu_tensor.dtype)
        shape = list(cpu_tensor.shape)
        canonical.update(name.encode("utf-8"))
        canonical.update(dtype.encode("ascii"))
        canonical.update(str(tuple(shape)).encode("ascii"))
        canonical.update(data)
        tensors.append(
            {
                "name": name,
                "dtype": dtype,
                "shape": shape,
                "nbytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return {
        "source": _source(path),
        "canonical_model_digest_sha256": canonical.hexdigest(),
        "tensor_manifest": tensors,
        "tensor_manifest_sha256": _canonical_json_sha256(tensors),
    }


def build_record(root: Path, summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text())
    runs = {}
    for run in RUN_ORDER:
        run_root = root / run
        metrics_path = run_root / "checkpoints" / "metrics.jsonl"
        metric_rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
        trajectory = [
            {key: value for key, value in row.items() if key != "elapsed_seconds"}
            for row in metric_rows
            if row.get("event") == "step"
        ]
        validations = [
            {key: value for key, value in row.items() if key != "elapsed_seconds"}
            for row in metric_rows
            if row.get("event") == "validation"
        ]
        projection = {
            "sources": {
                "artifact_hash_inventory": _artifact_inventory(
                    run_root / "artifact-sha256.txt", run_root
                ),
                "metrics": _source(metrics_path),
            },
            "step_trajectory": trajectory,
            "step_trajectory_sha256": _canonical_json_sha256(trajectory),
            "validation_metrics": validations,
            "validation_metrics_sha256": _canonical_json_sha256(validations),
            "final_checkpoint": _checkpoint_projection(run_root / "checkpoints" / "final.pt"),
        }
        runs[run] = projection

    standalone_path = root / "1-on" / "standalone.json"
    standalone = json.loads(standalone_path.read_text())
    return {
        "schema_version": 1,
        "ticket": "VAL-001",
        "attempt": summary["attempt"],
        "measured_commit": summary["measured_commit"],
        "purpose": (
            "Durable, compact projections of the raw metrics and checkpoints needed "
            "to independently recompute every committed training-trajectory, final-model, "
            "cross-run validation-score/identity, and standalone-score/identity parity claim."
        ),
        "source_summary": summary_path.name,
        "derivation": {
            "step_trajectory": (
                "Every metrics.jsonl event=step row in file order, with only the "
                "wall-clock-dependent elapsed_seconds field removed."
            ),
            "validation_metrics": (
                "Every metrics.jsonl event=validation row in file order, with only the "
                "wall-clock-dependent elapsed_seconds field removed."
            ),
            "final_checkpoint": (
                "For every sorted model-state tensor: name, dtype, shape, byte count, and "
                "SHA-256 of contiguous CPU tensor bytes. The canonical digest additionally "
                "hashes name, dtype, tuple(shape), and raw bytes in that order."
            ),
            "standalone": "The complete parsed standalone JSON payload, without omission.",
        },
        "runs": runs,
        "standalone": {
            "source": _source(standalone_path),
            "payload": standalone,
            "payload_sha256": _canonical_json_sha256(standalone),
        },
        "limitations": [
            "The 600 MB checkpoint containers and raw corpus/token contents are not copied into Git.",
            "Raw-file SHA-256 and byte sizes bind each projection to the separately retained raw capture; per-tensor hashes make pairwise final-model equality reviewable without committing 7.2 GB of duplicate checkpoints.",
            "This record supports equality/parity recomputation. Performance, resource, and pause summaries remain in the compact attempt summary and their separately hashed raw time series.",
        ],
    }


def _validation_payload(row: dict[str, Any]) -> dict[str, Any]:
    by_corpus = row["validation/by_corpus"]
    nll_sum = sum(value["nll_sum"] for value in by_corpus.values())
    return {
        "aggregate": {
            "nll": row["validation/loss"],
            "nll_sum": nll_sum,
            "perplexity": row["validation/perplexity"],
            "perplexity_overflow": False,
            "target_tokens": row["validation/target_tokens"],
        },
        "by_corpus": by_corpus,
        "evaluated_token_sha256": row["validation/evaluated_token_sha256"],
        "evaluated_window_sha256": row["validation/evaluated_window_sha256"],
        "evaluated_windows": row["validation/evaluated_windows"],
        "logical_checkpoint_identity": row["validation/logical_checkpoint_identity"],
        "manifest_identity": row["validation/manifest_identity"],
        "training_time_optimizer_step": row["optimizer_step"],
    }


def verify_record(record_path: Path, summary_path: Path) -> dict[str, Any]:
    record = json.loads(record_path.read_text())
    summary = json.loads(summary_path.read_text())
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    check("ticket", record["ticket"] == summary["ticket"] == "VAL-001")
    check("attempt", record["attempt"] == summary["attempt"])
    check("measured_commit", record["measured_commit"] == summary["measured_commit"])
    recomputation = summary["recomputation"]
    verifier_path = Path(__file__).resolve()
    check("record_locator", recomputation["record"] == record_path.name)
    check(
        "record_file_hash",
        recomputation["record_sha256"] == _sha256_file(record_path),
    )
    check("verifier_locator", recomputation["verifier"] == verifier_path.name)
    check(
        "verifier_file_hash",
        recomputation["verifier_sha256"] == _sha256_file(verifier_path),
    )

    summary_runs = {run["id"]: run for run in summary["runs"]}
    summary_artifact_paths = {
        "container": "container.csv",
        "gpu": "gpu.csv",
        "host": "host-vmstat.txt",
        "measurement": "measurement.json",
        "metrics": "checkpoints/metrics.jsonl",
        "resolved_config": "hydra/resolved_config.yaml",
        "run_manifest": "hydra/run_manifest.json",
        "stderr": "stderr.log",
        "stdout": "stdout.log",
    }
    for run in RUN_ORDER:
        source = record["runs"][run]
        inventory = source["sources"]["artifact_hash_inventory"]
        check(
            f"{run}:artifact_inventory_projection_integrity",
            inventory["entries_sha256"] == _canonical_json_sha256(inventory["entries"]),
        )
        check(
            f"{run}:summary_artifact_hash_links",
            all(
                inventory["entries"][path] == summary_runs[run]["artifacts"][summary_name]
                for summary_name, path in summary_artifact_paths.items()
            ),
        )
        check(
            f"{run}:metrics_raw_hash_link",
            source["sources"]["metrics"]["sha256"]
            == inventory["entries"]["checkpoints/metrics.jsonl"],
        )
        check(
            f"{run}:final_checkpoint_raw_hash_link",
            source["final_checkpoint"]["source"]["sha256"]
            == inventory["entries"]["checkpoints/final.pt"],
        )
        check(
            f"{run}:trajectory_projection_integrity",
            source["step_trajectory_sha256"] == _canonical_json_sha256(source["step_trajectory"]),
        )
        check(
            f"{run}:validation_projection_integrity",
            source["validation_metrics_sha256"]
            == _canonical_json_sha256(source["validation_metrics"]),
        )
        tensor_manifest = source["final_checkpoint"]["tensor_manifest"]
        check(
            f"{run}:tensor_projection_integrity",
            source["final_checkpoint"]["tensor_manifest_sha256"]
            == _canonical_json_sha256(tensor_manifest),
        )
        trajectory = source["step_trajectory"]
        check(
            f"{run}:work",
            len(trajectory) == summary_runs[run]["work"]["steps"]
            and trajectory[-1]["target_tokens"] == summary_runs[run]["work"]["targets"],
        )

    for pair in summary["pairs"]:
        off = record["runs"][pair["off"]]
        on = record["runs"][pair["on"]]
        trajectory_exact = off["step_trajectory"] == on["step_trajectory"]
        tensor_exact = (
            off["final_checkpoint"]["tensor_manifest"] == on["final_checkpoint"]["tensor_manifest"]
        )
        digest_exact = (
            off["final_checkpoint"]["canonical_model_digest_sha256"]
            == on["final_checkpoint"]["canonical_model_digest_sha256"]
        )
        check(
            f"pair_{pair['id']}:trajectory_exact",
            trajectory_exact == pair["trajectory_exact"],
        )
        check(f"pair_{pair['id']}:tensor_manifest_exact", tensor_exact)
        check(
            f"pair_{pair['id']}:canonical_digest_exact",
            digest_exact == pair["final_model_digest_exact"]
            and off["final_checkpoint"]["canonical_model_digest_sha256"]
            == pair["final_model_digest"],
        )

    validation_rows = {
        run: {row["optimizer_step"]: row for row in record["runs"][run]["validation_metrics"]}
        for run in RUN_ORDER
    }
    on_rows = [validation_rows[f"{pair}-on"][step] for pair in (1, 2, 3) for step in (25, 50)]
    first = on_rows[0]
    identity_exact = all(
        row["validation/manifest_identity"] == first["validation/manifest_identity"]
        and row["validation/evaluated_window_sha256"] == first["validation/evaluated_window_sha256"]
        and row["validation/evaluated_token_sha256"] == first["validation/evaluated_token_sha256"]
        for row in on_rows
    )
    score_exact = all(
        validation_rows[f"{pair}-on"][step]["validation/loss"]
        == validation_rows["1-on"][step]["validation/loss"]
        and validation_rows[f"{pair}-on"][step]["validation/by_corpus"]
        == validation_rows["1-on"][step]["validation/by_corpus"]
        for pair in (1, 2, 3)
        for step in (25, 50)
    )
    check(
        "cross_run_validation_identity_exact",
        identity_exact == summary["aggregate"]["validation_identity_exact"],
    )
    check(
        "cross_run_validation_score_exact_by_step",
        score_exact == summary["aggregate"]["validation_score_exact_by_step"],
    )

    standalone = record["standalone"]
    check(
        "standalone_raw_hash_link",
        standalone["source"]["sha256"] == summary["standalone_parity"]["output_sha256"],
    )
    check(
        "standalone_projection_integrity",
        standalone["payload_sha256"] == _canonical_json_sha256(standalone["payload"]),
    )
    training_payload = _validation_payload(validation_rows["1-on"][50])
    standalone_result = standalone["payload"]["result"]
    standalone_exact = all(
        training_payload[key] == standalone_result[key]
        for key in (
            "aggregate",
            "by_corpus",
            "evaluated_token_sha256",
            "evaluated_window_sha256",
            "evaluated_windows",
            "logical_checkpoint_identity",
            "manifest_identity",
        )
    )
    check(
        "standalone_score_identity_exact",
        standalone_exact == summary["standalone_parity"]["exact"],
    )
    if "matched_payload" in summary["standalone_parity"]:
        check(
            "standalone_matched_payload",
            training_payload == summary["standalone_parity"]["matched_payload"],
        )

    failures = [item["name"] for item in checks if not item["passed"]]
    return {
        "record": record_path.name,
        "summary": summary_path.name,
        "checks": len(checks),
        "passed": not failures,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("raw_root", type=Path)
    build.add_argument("summary", type=Path)
    build.add_argument("output", type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("record", type=Path)
    verify.add_argument("summary", type=Path)
    args = parser.parse_args()

    if args.command == "build":
        payload = build_record(args.raw_root, args.summary)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        result = verify_record(args.record, args.summary)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
