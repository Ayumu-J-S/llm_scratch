from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from data.identity import normalized_content_sha256


@dataclass(frozen=True)
class DocumentPolicy:
    """Deterministic, model-free policy applied before split assignment."""

    version: int = 1
    language: str = "any"
    max_utf8_bytes: int | None = None
    reject_controls: bool = True
    reject_wrong_script: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or self.version != 1:
            raise ValueError("document policy version must be 1")
        if self.language not in {"any", "en", "ja"}:
            raise ValueError("document policy language must be any, en, or ja")
        if self.max_utf8_bytes is not None and (
            isinstance(self.max_utf8_bytes, bool)
            or not isinstance(self.max_utf8_bytes, int)
            or self.max_utf8_bytes < 1
        ):
            raise ValueError("max_utf8_bytes must be positive when set")
        if not isinstance(self.reject_controls, bool) or not isinstance(
            self.reject_wrong_script, bool
        ):
            raise ValueError("document policy reject flags must be booleans")


@dataclass(frozen=True)
class QualityResult:
    accepted: bool
    text: str | None
    content_sha256: str | None
    reason: str | None
    truncated: bool
    latin_count: int
    japanese_count: int
    other_letter_count: int


@dataclass
class QualityTracker:
    """Bounded-preflight accounting; production streaming need not retain hashes."""

    seen_content: set[str] = field(default_factory=set)
    counts: dict[str, int] = field(default_factory=dict)

    def observe(self, result: QualityResult) -> str | None:
        reason = result.reason
        if result.accepted and result.content_sha256 is not None:
            if result.content_sha256 in self.seen_content:
                reason = "duplicate"
            else:
                self.seen_content.add(result.content_sha256)
        key = reason or "accepted"
        self.counts[key] = self.counts.get(key, 0) + 1
        if result.truncated:
            self.counts["truncated"] = self.counts.get("truncated", 0) + 1
        return reason


def apply_document_policy(text: object, policy: DocumentPolicy) -> QualityResult:
    if not isinstance(text, str):
        return _rejected("non_string")
    try:
        # Normalize line endings before NFC so identity and byte truncation are stable.
        normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
        normalized.encode("utf-8", errors="strict")
    except (UnicodeEncodeError, UnicodeError):
        return _rejected("invalid_unicode")
    normalized = normalized.strip()
    if not normalized:
        return _rejected("empty")
    if policy.reject_controls and any(
        unicodedata.category(character) in {"Cc", "Cs"} and character not in {"\n", "\t"}
        for character in normalized
    ):
        return _rejected("control_character")

    truncated = False
    if policy.max_utf8_bytes is not None:
        encoded = normalized.encode("utf-8")
        if len(encoded) > policy.max_utf8_bytes:
            encoded = encoded[: policy.max_utf8_bytes]
            while True:
                try:
                    normalized = encoded.decode("utf-8")
                    break
                except UnicodeDecodeError as error:
                    encoded = encoded[: error.start]
            normalized = normalized.rstrip()
            truncated = True
            if not normalized:
                return _rejected("empty_after_truncation", truncated=True)

    latin, japanese, other = _script_counts(normalized)
    if policy.reject_wrong_script and policy.language == "ja" and japanese == 0:
        return _rejected("wrong_script", truncated, latin, japanese, other)
    if policy.reject_wrong_script and policy.language == "en" and latin == 0 and japanese > 0:
        return _rejected("wrong_script", truncated, latin, japanese, other)
    return QualityResult(
        accepted=True,
        text=normalized,
        content_sha256=normalized_content_sha256(normalized),
        reason=None,
        truncated=truncated,
        latin_count=latin,
        japanese_count=japanese,
        other_letter_count=other,
    )


def _script_counts(text: str) -> tuple[int, int, int]:
    latin = japanese = other = 0
    for character in text:
        codepoint = ord(character)
        if (
            0x3040 <= codepoint <= 0x30FF
            or 0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        ):
            japanese += 1
        elif "LATIN" in unicodedata.name(character, ""):
            latin += 1
        elif unicodedata.category(character).startswith("L"):
            other += 1
    return latin, japanese, other


def _rejected(
    reason: str,
    truncated: bool = False,
    latin: int = 0,
    japanese: int = 0,
    other: int = 0,
) -> QualityResult:
    return QualityResult(False, None, None, reason, truncated, latin, japanese, other)
