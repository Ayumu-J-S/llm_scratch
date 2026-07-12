#!/usr/bin/env python3
"""Run the bounded GATE-001 bilingual memorization and resume proof.

This command intentionally reuses :func:`train.prepare_trainer`, the same
assembly path as ``src/train.py``.  It performs two independent same-seed
200-update executions, then proves the same trajectory through a deliberate
interruption after the verified step-100 recovery checkpoint followed by normal
``latest`` resume.  It is a fixed-fixture *memorization* proof, never held-out
validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from data.identity import canonical_json_bytes
from train import ROOT_DIR, build_streaming_dataloader, prepare_trainer


MAX_STEPS = 200
INTERRUPT_STEP = 100
LOSS_THRESHOLD = 0.20
# These fixed prefixes leave the trained suffix unambiguous while retaining
# enough context for a human-readable JP/EN continuation check.
JAPANESE_PROMPT = "日本語の合図: 桜は"
JAPANESE_EXPECTED_SUFFIX = "春に咲きます。"
ENGLISH_PROMPT = "English cue: small models"
ENGLISH_EXPECTED_SUFFIX = "memorize fixed text."


class GateProofError(RuntimeError):
    """The fixed GATE-001 decision rule was not satisfied."""


class VerifiedInterruption(RuntimeError):
    """A deliberate test interruption after a verified recovery checkpoint."""


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Run the bounded GATE-001 bilingual memorization/resume proof."
    )
    command.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="new local evidence directory; never overwritten implicitly",
    )
    command.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
        help="explicit device override; canonical target smoke uses cuda",
    )
    return command


def compose_config(*, device: str) -> DictConfig:
    """Compose the versioned gate profile through the normal Hydra source."""

    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT_DIR / "config")):
        cfg = hydra.compose(config_name="train", overrides=["profile=gate_overfit"])
    cfg.runtime.device = device
    return cfg


def _copy_config(cfg: DictConfig, *, resume_path: str | None = None) -> DictConfig:
    copied = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    copied.artifacts.resume_path = resume_path
    return copied


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GateProofError(f"expected JSON object: {path}")
    return value


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise GateProofError(f"non-object metrics record in {path}")
        records.append(value)
    return records


def _model_digest(checkpoint_path: Path) -> str:
    """Return a stable tensor-content digest without serializing pickled state."""

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("state", {})
    model = state.get("model") if isinstance(state, dict) else None
    if not isinstance(model, dict):
        raise GateProofError(f"checkpoint has no model state: {checkpoint_path}")
    digest = hashlib.sha256()
    for name in sorted(model):
        tensor = model[name]
        if not isinstance(tensor, torch.Tensor):
            raise GateProofError(f"checkpoint model entry is not a tensor: {name}")
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _checkpoint_summary(checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise GateProofError(f"invalid checkpoint payload: {checkpoint_path}")
    identity = payload.get("identity")
    state = payload.get("state")
    if not isinstance(identity, dict) or not isinstance(state, dict):
        raise GateProofError(f"checkpoint missing identity/state: {checkpoint_path}")
    counters = state.get("counters")
    if not isinstance(counters, dict):
        raise GateProofError(f"checkpoint missing counters: {checkpoint_path}")
    return {
        "path": str(checkpoint_path.resolve()),
        "sha256": _sha256_file(checkpoint_path),
        "identity_sha256": hashlib.sha256(canonical_json_bytes(identity)).hexdigest(),
        "model_sha256": _model_digest(checkpoint_path),
        "optimizer_step": counters.get("optimizer_step"),
        "target_tokens": counters.get("target_tokens"),
    }


def _sample_with_gen_cli(checkpoint_path: Path, *, prompt: str, device: str) -> dict[str, Any]:
    """Use the GEN-001 CLI, rather than an in-process duplicate sampler path."""

    result = subprocess.run(
        [
            sys.executable,
            "src/generate.py",
            "--checkpoint",
            str(checkpoint_path),
            "--prompt",
            prompt,
            "--max-new-tokens",
            "8",
            "--device",
            device,
            "--json",
        ],
        cwd=ROOT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise GateProofError(
            f"GEN-001 sampling failed for prompt {prompt!r}: {result.stderr.strip()}"
        )
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise GateProofError("GEN-001 did not emit a JSON object")
    return value


def _run_uninterrupted(cfg: DictConfig, *, run_dir: Path) -> None:
    trainer = prepare_trainer(_copy_config(cfg), run_dir=run_dir)
    trainer.fit()


def _batches_per_train_pass(cfg: DictConfig, *, device: str) -> int:
    """Count one finite fixture pass before choosing the resume boundary."""

    loader = build_streaming_dataloader(cfg, "train", device=torch.device(device))
    count = sum(1 for _ in loader)
    if count < 2:
        raise GateProofError("GATE-001 fixture must contain more than one train batch per pass")
    return count


def _run_interrupted_then_resumed(cfg: DictConfig, *, run_dir: Path) -> None:
    """Stop only after normal event handling wrote the verified recovery file."""

    interrupted = prepare_trainer(_copy_config(cfg), run_dir=run_dir)
    original_events = interrupted._run_events

    def stop_after_recovery(*, epoch_end: bool, train_loss: float | None = None) -> None:
        original_events(epoch_end=epoch_end, train_loss=train_loss)
        recovery = interrupted.checkpoint_dir / f"recovery-step-{INTERRUPT_STEP:012d}.pt"
        if (
            not epoch_end
            and interrupted.optimizer_step == INTERRUPT_STEP
            and recovery.is_file()
        ):
            raise VerifiedInterruption(
                f"deliberate GATE-001 interruption after verified recovery {recovery.name}"
            )

    interrupted._run_events = stop_after_recovery  # type: ignore[method-assign]
    try:
        interrupted.fit()
    except VerifiedInterruption:
        pass
    else:
        raise GateProofError("deliberate interruption did not occur at the verified recovery")

    resumed = prepare_trainer(_copy_config(cfg, resume_path="latest"), run_dir=run_dir)
    resumed.fit()


def _run_summary(run_dir: Path, *, device: str) -> dict[str, Any]:
    checkpoint_dir = run_dir / "checkpoints"
    final_checkpoint = checkpoint_dir / "final.pt"
    if not final_checkpoint.is_file():
        raise GateProofError(f"final checkpoint is missing: {final_checkpoint}")
    metrics = _read_metrics(checkpoint_dir / "metrics.jsonl")
    steps = [record for record in metrics if record.get("event") == "step"]
    trace = [record.get("train/loss_step") for record in steps]
    if any(not isinstance(value, (int, float)) for value in trace):
        raise GateProofError("step loss trace is incomplete")
    if len(trace) != MAX_STEPS:
        raise GateProofError(f"expected {MAX_STEPS} step metrics, observed {len(trace)}")
    if not all(torch.isfinite(torch.tensor(value)) for value in trace):
        raise GateProofError("non-finite loss appears in the step trace")

    checkpoint = _checkpoint_summary(final_checkpoint)
    samples = {
        "japanese": _sample_with_gen_cli(final_checkpoint, prompt=JAPANESE_PROMPT, device=device),
        "english": _sample_with_gen_cli(final_checkpoint, prompt=ENGLISH_PROMPT, device=device),
    }
    return {
        "run_dir": str(run_dir.resolve()),
        "run_manifest": _read_json(run_dir / "run_manifest.json"),
        "resolved_config_path": str((run_dir / "resolved_config.yaml").resolve()),
        "checkpoint": checkpoint,
        "metrics_path": str((checkpoint_dir / "metrics.jsonl").resolve()),
        "optimizer_steps": [record.get("optimizer_step") for record in steps],
        "target_tokens": [record.get("target_tokens") for record in steps],
        "loss_trace": trace,
        "final_loss": float(trace[-1]),
        "samples": samples,
    }


def _compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Compare every decision-relevant result against one fixed reference."""

    same_steps = reference["optimizer_steps"] == candidate["optimizer_steps"]
    same_targets = reference["target_tokens"] == candidate["target_tokens"]
    same_trace = reference["loss_trace"] == candidate["loss_trace"]
    same_identity = (
        reference["checkpoint"]["identity_sha256"]
        == candidate["checkpoint"]["identity_sha256"]
    )
    same_model = (
        reference["checkpoint"]["model_sha256"] == candidate["checkpoint"]["model_sha256"]
    )
    same_samples = all(
        reference["samples"][language]["completion"]
        == candidate["samples"][language]["completion"]
        and reference["samples"][language]["generated_token_ids"]
        == candidate["samples"][language]["generated_token_ids"]
        for language in ("japanese", "english")
    )
    expected_japanese = JAPANESE_EXPECTED_SUFFIX in reference["samples"]["japanese"]["completion"]
    expected_english = ENGLISH_EXPECTED_SUFFIX in reference["samples"]["english"]["completion"]
    final_loss_passes = (
        reference["final_loss"] <= LOSS_THRESHOLD and candidate["final_loss"] <= LOSS_THRESHOLD
    )
    checks = {
        "same_optimizer_steps": same_steps,
        "same_target_tokens": same_targets,
        "same_loss_trace": same_trace,
        "same_checkpoint_identity": same_identity,
        "same_final_model": same_model,
        "same_samples": same_samples,
        "reference_japanese_suffix_recognizable": expected_japanese,
        "reference_english_suffix_recognizable": expected_english,
        "final_loss_within_predeclared_threshold": final_loss_passes,
    }
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise GateProofError("GATE-001 decision rule failed: " + ", ".join(failed))
    return checks


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_proof(*, output_dir: Path, device: str) -> dict[str, Any]:
    """Execute the full predeclared proof and retain one local JSON record."""

    if output_dir.exists():
        raise GateProofError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    cfg = compose_config(device=device)
    if int(cfg.training.max_steps) != MAX_STEPS:
        raise GateProofError("profile max_steps no longer matches the predeclared GATE-001 budget")
    if int(cfg.training.checkpoint_every_n_steps) != INTERRUPT_STEP:
        raise GateProofError("profile checkpoint cadence no longer matches the resume point")
    validation_cadence = cfg.training.validation_every_n_steps
    if validation_cadence is None or int(validation_cadence) <= MAX_STEPS:
        raise GateProofError(
            "GATE-001 must not score its auxiliary source within the memorization budget"
        )
    batches_per_pass = _batches_per_train_pass(cfg, device=device)
    if INTERRUPT_STEP % batches_per_pass == 0:
        raise GateProofError(
            "GATE-001 interruption would land on a terminal stream batch; "
            "use a fixed fixture whose pass length leaves a real resume suffix"
        )
    if int(cfg.training.epochs) * batches_per_pass < MAX_STEPS:
        raise GateProofError("profile does not provide enough finite stream passes for the step budget")

    try:
        reference_dir = output_dir / "reference"
        repeat_dir = output_dir / "repeat"
        resumed_dir = output_dir / "interrupted-resumed"
        _run_uninterrupted(cfg, run_dir=reference_dir)
        _run_uninterrupted(cfg, run_dir=repeat_dir)
        _run_interrupted_then_resumed(cfg, run_dir=resumed_dir)
        reference = _run_summary(reference_dir, device=device)
        repeat = _run_summary(repeat_dir, device=device)
        resumed = _run_summary(resumed_dir, device=device)
        comparisons = {
            "independent_same_seed_repeat": _compare(reference, repeat),
            "interrupted_resume": _compare(reference, resumed),
        }
        record = {
            "schema_version": 1,
            "ticket": "GATE-001",
            "conclusion_label": "fixed-fixture memorization proof; not held-out validation or generalization",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "command": " ".join(["uv", "run", "python", *sys.argv]),
            "predeclared": {
                "max_steps": MAX_STEPS,
                "interruption_step": INTERRUPT_STEP,
                "batches_per_train_pass": batches_per_pass,
                "final_loss_threshold": LOSS_THRESHOLD,
                "stop_conditions": [
                    "non-finite loss or gradients",
                    "loss threshold missed by 200 updates",
                    "any compared counter, trace, identity, or sample diverges",
                    "recognizable fixed Japanese or English suffix missing",
                ],
            },
            "fixture": {
                "train_manifest": str(
                    (ROOT_DIR / "tests/fixtures/gate_overfit/v1/train.manifest.json").resolve()
                ),
                "train_manifest_sha256": _sha256_file(
                    ROOT_DIR / "tests/fixtures/gate_overfit/v1/train.manifest.json"
                ),
                "auxiliary_manifest": str(
                    (ROOT_DIR / "tests/fixtures/gate_overfit/v1/auxiliary.manifest.json").resolve()
                ),
                "auxiliary_manifest_sha256": _sha256_file(
                    ROOT_DIR / "tests/fixtures/gate_overfit/v1/auxiliary.manifest.json"
                ),
            },
            "reference": reference,
            "repeat": repeat,
            "interrupted_resumed": resumed,
            "comparisons": comparisons,
            "verdict": "PASS",
        }
        (output_dir / "gate_record.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return record
    except Exception as error:
        # Retain partial evidence for the required failed-attempt handoff.
        failure = {
            "ticket": "GATE-001",
            "verdict": "FAILURE_RETAINED",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "command": " ".join(["uv", "run", "python", *sys.argv]),
            "error": f"{type(error).__name__}: {error}",
        }
        (output_dir / "failure_record.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    record = run_proof(output_dir=output_dir, device=args.device)
    print(
        json.dumps(
            {
                "verdict": record["verdict"],
                "record": str((output_dir / "gate_record.json").resolve()),
                "final_loss": record["reference"]["final_loss"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
