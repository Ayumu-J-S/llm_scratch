"""Minimal, reproducible sampling from repository full-state checkpoints."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from models.simple_decoder_transformer import SimpleDecoderTransformer
from tokenizer.canonical import CanonicalTokenizer
from training.checkpoint import load_checkpoint_for_generation


BASE_MODEL_CONTINUATION_LABEL = "base-model-continuation"


class SamplingError(ValueError):
    """The checkpoint or generation request cannot produce a safe continuation."""


@dataclass(frozen=True)
class GenerationResult:
    """A labeled base-model continuation plus enough identity for comparison."""

    label: str
    checkpoint_path: str
    checkpoint_kind: str
    checkpoint_optimizer_step: int
    tokenizer_fingerprint: str
    prompt: str
    prompt_token_count: int
    completion: str
    generated_token_ids: tuple[int, ...]
    max_new_tokens_requested: int
    max_new_tokens_allowed_by_context: int
    temperature: float | None
    top_k: int | None
    seed: int | None
    stop_reason: str

    def metadata(self) -> dict[str, Any]:
        """Return serializable metadata without obscuring this as chat output."""

        return asdict(self)


class CheckpointSampler:
    """Reconstruct and sample a base model from one full-state checkpoint."""

    def __init__(
        self,
        *,
        model: SimpleDecoderTransformer,
        tokenizer: CanonicalTokenizer,
        checkpoint_path: Path,
        checkpoint_kind: str,
        checkpoint_optimizer_step: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_path = checkpoint_path
        self.checkpoint_kind = checkpoint_kind
        self.checkpoint_optimizer_step = checkpoint_optimizer_step
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> CheckpointSampler:
        """Reconstruct model and tokenizer solely from a full-state checkpoint."""

        path = Path(checkpoint_path).resolve()
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise SamplingError("CUDA was requested for generation but is unavailable")
        payload = load_checkpoint_for_generation(path)
        state = _mapping(payload["state"], "checkpoint state")
        config = _mapping(state["resolved_config"], "checkpoint resolved_config")
        model_config = _mapping(config.get("model"), "checkpoint resolved_config.model")
        training_config = _mapping(config.get("training"), "checkpoint resolved_config.training")
        tokenizer_config = _mapping(config.get("tokenizer"), "checkpoint resolved_config.tokenizer")
        tokenizer = CanonicalTokenizer.from_config(tokenizer_config)
        identity = _mapping(payload["identity"], "checkpoint identity")
        expected_fingerprint = identity.get("tokenizer_fingerprint")
        if expected_fingerprint != tokenizer.fingerprint:
            raise SamplingError(
                "checkpoint tokenizer fingerprint does not match the canonical tokenizer artifact"
            )

        model = SimpleDecoderTransformer(
            vocab_size=tokenizer.vocab_size,
            embed_size=_positive_int(model_config.get("embed_size"), "model.embed_size"),
            num_heads=_positive_int(model_config.get("num_heads"), "model.num_heads"),
            max_len=_positive_int(
                training_config.get("sequence_length"), "training.sequence_length"
            ),
            num_layers=_positive_int(model_config.get("num_layers"), "model.num_layers"),
            dropout=_dropout(model_config.get("dropout")),
            pad_token_id=tokenizer.pad_token_id,
        )
        try:
            model.load_state_dict(state["model"], strict=True)
        except (RuntimeError, TypeError) as error:
            raise SamplingError(
                "checkpoint model weights do not match its checkpoint-owned model/tokenizer config"
            ) from error
        model.to(resolved_device)
        model.eval()
        counters = _mapping(state["counters"], "checkpoint counters")
        return cls(
            model=model,
            tokenizer=tokenizer,
            checkpoint_path=path,
            checkpoint_kind=str(payload["kind"]),
            checkpoint_optimizer_step=_positive_or_zero_int(
                counters.get("optimizer_step"), "checkpoint optimizer_step"
            ),
            device=resolved_device,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        """Produce one bounded base-model continuation.

        Omitting ``temperature`` selects deterministic greedy decoding.  A
        positive temperature selects sampling and requires a seed; optional
        top-k filtering is applied before the seeded multinomial draw.
        """

        if not isinstance(prompt, str):
            raise SamplingError("prompt must be a string")
        requested = _positive_int(max_new_tokens, "max_new_tokens")
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            raise SamplingError("prompt must encode to at least one token")
        if len(prompt_ids) > self.model.max_len:
            raise SamplingError(
                f"prompt has {len(prompt_ids)} tokens but checkpoint context is {self.model.max_len}"
            )
        sampling = temperature is not None
        if sampling:
            temperature_value = _positive_finite_float(temperature, "temperature")
            if seed is None:
                raise SamplingError("seed is required for temperature/top-k sampling")
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise SamplingError("seed must be a non-negative integer")
            if top_k is not None:
                top_k = _positive_int(top_k, "top_k")
                if top_k > self.tokenizer.vocab_size:
                    raise SamplingError(
                        f"top_k must be no greater than tokenizer vocab size {self.tokenizer.vocab_size}"
                    )
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
        else:
            if top_k is not None or seed is not None:
                raise SamplingError("top_k and seed require a positive temperature")
            temperature_value = None
            generator = None

        remaining_context = self.model.max_len - len(prompt_ids)
        allowed = min(requested, remaining_context)
        generated_ids: list[int] = []
        if prompt_ids[-1] == self.tokenizer.eos_token_id:
            stop_reason = "prompt_eos"
        elif allowed == 0:
            stop_reason = "context_limit"
        else:
            stop_reason = "max_new_tokens"
            token_ids = list(prompt_ids)
            with torch.inference_mode():
                for _ in range(allowed):
                    tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)
                    logits = self.model(tokens)[0, -1]
                    if temperature_value is None:
                        token_id = int(torch.argmax(logits).item())
                    else:
                        token_id = _sample_token(
                            logits,
                            temperature=temperature_value,
                            top_k=top_k,
                            generator=generator,
                        )
                    generated_ids.append(token_id)
                    token_ids.append(token_id)
                    if token_id == self.tokenizer.eos_token_id:
                        stop_reason = "eos"
                        break
                else:
                    if allowed < requested:
                        stop_reason = "context_limit"

        return GenerationResult(
            label=BASE_MODEL_CONTINUATION_LABEL,
            checkpoint_path=str(self.checkpoint_path),
            checkpoint_kind=self.checkpoint_kind,
            checkpoint_optimizer_step=self.checkpoint_optimizer_step,
            tokenizer_fingerprint=self.tokenizer.fingerprint,
            prompt=prompt,
            prompt_token_count=len(prompt_ids),
            completion=self.tokenizer.decode(generated_ids, skip_special_tokens=True),
            generated_token_ids=tuple(generated_ids),
            max_new_tokens_requested=requested,
            max_new_tokens_allowed_by_context=allowed,
            temperature=temperature_value,
            top_k=top_k,
            seed=seed,
            stop_reason=stop_reason,
        )


def _sample_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int | None,
    generator: torch.Generator,
) -> int:
    scaled = logits / temperature
    if top_k is not None:
        values, indices = torch.topk(scaled, top_k)
        probabilities = torch.softmax(values, dim=-1)
        selected = torch.multinomial(probabilities, 1, generator=generator)
        return int(indices[selected].item())
    probabilities = torch.softmax(scaled, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SamplingError(f"{label} must be a mapping")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SamplingError(f"{label} must be a positive integer")
    return value


def _positive_or_zero_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SamplingError(f"{label} must be a non-negative integer")
    return value


def _positive_finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise SamplingError(f"{label} must be a positive finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise SamplingError(f"{label} must be a positive finite number") from error
    if not math.isfinite(numeric) or numeric <= 0:
        raise SamplingError(f"{label} must be a positive finite number")
    return numeric


def _dropout(value: Any) -> float:
    if isinstance(value, bool):
        raise SamplingError("model.dropout must be a finite number in [0, 1)")
    try:
        dropout = float(value)
    except (TypeError, ValueError) as error:
        raise SamplingError("model.dropout must be a finite number in [0, 1)") from error
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise SamplingError("model.dropout must be a finite number in [0, 1)")
    return dropout
