from __future__ import annotations

import hashlib
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from enum import Enum
from typing import Iterable, Mapping

from data.identity import canonical_fingerprint


UINT64_SPACE = 1 << 64


class DataPurpose(str, Enum):
    PRETRAINING = "pretraining"
    MEMORIZATION_SMOKE = "memorization_smoke"
    BENCHMARK_DEV = "benchmark_dev"
    BENCHMARK_RESERVED = "benchmark_reserved"


def validation_threshold(validation_fraction: str) -> int:
    if not isinstance(validation_fraction, str):
        raise TypeError("validation_fraction must be a decimal string")
    try:
        fraction = Decimal(validation_fraction)
    except InvalidOperation as error:
        raise ValueError("validation_fraction must be a decimal string") from error
    if not fraction.is_finite() or not Decimal("0") < fraction < Decimal("1"):
        raise ValueError("validation_fraction must be between zero and one")
    return int((fraction * UINT64_SPACE).to_integral_value(rounding=ROUND_FLOOR))


def assign_split(*, content_sha256: str, salt: str, validation_fraction: str) -> str:
    if not salt:
        raise ValueError("split salt must be non-empty")
    if len(content_sha256) != 64:
        raise ValueError("content_sha256 must be a SHA-256 hex digest")
    score = int.from_bytes(
        hashlib.sha256(f"{salt}\0{content_sha256}".encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )
    return "validation" if score < validation_threshold(validation_fraction) else "train"


def split_fingerprint(documents: Iterable[Mapping[str, object]], split: str) -> str:
    identities = sorted(
        (
            {
                "content_sha256": str(document["content_sha256"]),
                "document_id": str(document["document_id"]),
            }
            for document in documents
            if document["split"] == split
        ),
        key=lambda item: (item["document_id"], item["content_sha256"]),
    )
    return canonical_fingerprint({"documents": identities, "split": split})


def dataset_fingerprint(documents: Iterable[Mapping[str, object]]) -> str:
    identities = sorted(
        (
            {
                "content_sha256": str(document["content_sha256"]),
                "document_id": str(document["document_id"]),
                "split": str(document["split"]),
            }
            for document in documents
        ),
        key=lambda item: (item["document_id"], item["content_sha256"], item["split"]),
    )
    return canonical_fingerprint({"documents": identities})
