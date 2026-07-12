from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_fingerprint(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def normalize_text_identity(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("document text must be a string")
    # Reject surrogate-containing strings rather than allowing platform-dependent
    # error handlers to change document identity.
    text.encode("utf-8", errors="strict")
    if text.startswith("\ufeff"):
        text = text[1:]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return unicodedata.normalize("NFC", text).strip()


def normalized_content_sha256(text: str) -> str:
    return sha256_bytes(normalize_text_identity(text).encode("utf-8"))


def stable_document_id(
    *,
    source_identity: str,
    content_sha256: str,
    upstream_id: str | int | None,
) -> str:
    if not source_identity:
        raise ValueError("source_identity must be non-empty")
    _require_sha256(content_sha256, "content_sha256")
    if upstream_id is None:
        identity = {"content_sha256": content_sha256, "source_identity": source_identity}
    else:
        upstream_id = str(upstream_id)
        if not upstream_id:
            raise ValueError("upstream_id must be non-empty when provided")
        identity = {"source_identity": source_identity, "upstream_id": upstream_id}
    return canonical_fingerprint(identity)


def _require_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
