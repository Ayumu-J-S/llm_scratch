from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


ROOT_DIR = Path(__file__).resolve().parents[2]
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_CONFIG_KEYS = {"manifest_path", "expected_fingerprint"}
_SPECIAL_ROLES = {
    "unk",
    "bos",
    "additional_eos",
    "mask",
    "pad",
    "cls",
    "sep",
    "eos_eod",
}


class CanonicalTokenizer:
    """Validated, offline-only access to the repository's canonical tokenizer."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        *,
        manifest_path: Path,
        manifest: Mapping[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self.manifest_path = manifest_path
        self.fingerprint = str(manifest["fingerprint"])
        runtime = _mapping(manifest["runtime"], "manifest.runtime")
        special_tokens = _mapping(manifest["special_tokens"], "manifest.special_tokens")
        self._reserved_special_tokens = {
            _special_id(special_tokens, role): (
                role,
                str(_mapping(special_tokens[role], f"manifest.special_tokens.{role}")["token"]),
            )
            for role in sorted(_SPECIAL_ROLES)
        }
        self.vocab_size = _integer(runtime["vocab_size"], "manifest.runtime.vocab_size")
        self.max_token_id = _integer(runtime["max_token_id"], "manifest.runtime.max_token_id")
        self.unk_token_id = _special_id(special_tokens, "unk")
        self.bos_token_id = _special_id(special_tokens, "bos")
        self.pad_token_id = _special_id(special_tokens, "pad")
        self.eos_token_id = _special_id(special_tokens, "eos_eod")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> CanonicalTokenizer:
        plain_config = _plain_mapping(config)
        unexpected = set(plain_config) - _CONFIG_KEYS
        missing = _CONFIG_KEYS - set(plain_config)
        if missing or unexpected:
            raise ValueError(
                "canonical tokenizer config must contain exactly manifest_path and "
                f"expected_fingerprint; missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

        manifest_value = plain_config["manifest_path"]
        if not isinstance(manifest_value, str) or not manifest_value:
            raise TypeError("tokenizer.manifest_path must be a non-empty string")
        manifest_path = Path(manifest_value)
        if not manifest_path.is_absolute():
            manifest_path = ROOT_DIR / manifest_path
        manifest_path = manifest_path.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"canonical tokenizer manifest not found: {manifest_path}")

        expected_fingerprint = plain_config["expected_fingerprint"]
        _require_sha256(expected_fingerprint, "tokenizer.expected_fingerprint")
        manifest = _load_json(manifest_path, "canonical tokenizer manifest")
        _validate_manifest_fingerprint(manifest, expected_fingerprint)
        tokenizer = _validate_manifest_and_load_tokenizer(manifest_path, manifest)
        return cls(tokenizer, manifest_path=manifest_path, manifest=manifest)

    def encode(self, text: str) -> list[int]:
        _validate_text(text)
        token_ids = [
            int(token_id) for token_id in self._tokenizer.encode(text, add_special_tokens=False).ids
        ]
        self._validate_ids(token_ids)
        for token_id in token_ids:
            reserved = self._reserved_special_tokens.get(token_id)
            if reserved is not None:
                role, token = reserved
                raise ValueError(
                    "raw text encoded to reserved canonical special token: "
                    f"role={role}, token={token!r}, id={token_id}"
                )
        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        if not isinstance(skip_special_tokens, bool):
            raise TypeError("skip_special_tokens must be a bool")
        ids = list(token_ids)
        self._validate_ids(ids)
        text = self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        _validate_text(text)
        return text

    def assert_fingerprint(self, expected_fingerprint: str) -> None:
        _require_sha256(expected_fingerprint, "expected_fingerprint")
        if expected_fingerprint != self.fingerprint:
            raise ValueError(
                "canonical tokenizer fingerprint mismatch: "
                f"expected {expected_fingerprint}, got {self.fingerprint}"
            )

    def _validate_ids(self, token_ids: list[int]) -> None:
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token_ids must contain only integers")
            if token_id < 0 or token_id > self.max_token_id:
                raise ValueError(
                    f"token id {token_id} is outside canonical range 0..{self.max_token_id}"
                )


def _plain_mapping(config: Any) -> dict[str, Any]:
    try:
        from omegaconf import DictConfig, OmegaConf
    except ImportError:
        pass
    else:
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)

    if not isinstance(config, Mapping):
        raise TypeError("canonical tokenizer config must be a serializable mapping")
    plain = dict(config)
    try:
        json.dumps(plain)
    except (TypeError, ValueError) as error:
        raise TypeError("canonical tokenizer config must be JSON serializable") from error
    return plain


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        text = path.read_bytes().decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not valid UTF-8: {path}") from error
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def _validate_manifest_fingerprint(manifest: Mapping[str, Any], expected_fingerprint: str) -> None:
    stored_fingerprint = manifest.get("fingerprint")
    _require_sha256(stored_fingerprint, "manifest.fingerprint")
    fingerprint_payload = dict(manifest)
    del fingerprint_payload["fingerprint"]
    encoded = json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    actual_fingerprint = hashlib.sha256(encoded).hexdigest()
    if stored_fingerprint != actual_fingerprint:
        raise ValueError(
            "canonical tokenizer manifest fingerprint is invalid: "
            f"stored {stored_fingerprint}, computed {actual_fingerprint}"
        )
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "canonical tokenizer fingerprint mismatch: "
            f"expected {expected_fingerprint}, got {actual_fingerprint}"
        )


def _validate_manifest_and_load_tokenizer(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> Tokenizer:
    if manifest.get("schema_version") != 1:
        raise ValueError("canonical tokenizer manifest.schema_version must be 1")
    if not isinstance(manifest.get("canonical_id"), str) or not manifest["canonical_id"]:
        raise TypeError("manifest.canonical_id must be a non-empty string")

    upstream = _mapping(manifest.get("upstream"), "manifest.upstream")
    for field in (
        "repository",
        "revision_url",
        "tokenizer_source_repository",
        "tokenizer_source_url",
    ):
        if not isinstance(upstream.get(field), str) or not upstream[field]:
            raise TypeError(f"manifest.upstream.{field} must be a non-empty string")
    for field in ("revision", "tokenizer_source_revision"):
        revision = upstream.get(field)
        if not isinstance(revision, str) or _REVISION_PATTERN.fullmatch(revision) is None:
            raise ValueError(f"manifest.upstream.{field} must be a lowercase 40-hex revision")

    license_config = _mapping(manifest.get("license"), "manifest.license")
    if license_config.get("spdx") != "Apache-2.0":
        raise ValueError("manifest.license.spdx must be Apache-2.0")
    if not isinstance(license_config.get("url"), str) or not license_config["url"]:
        raise TypeError("manifest.license.url must be a non-empty string")

    files = _mapping(manifest.get("files"), "manifest.files")
    if set(files) != {"tokenizer", "license"}:
        raise ValueError("manifest.files must contain exactly tokenizer and license")
    validated_paths = {
        role: _validate_file(manifest_path.parent, role, files[role])
        for role in ("tokenizer", "license")
    }

    tokenizer_json = _load_json(validated_paths["tokenizer"], "tokenizer artifact")
    runtime = _mapping(manifest.get("runtime"), "manifest.runtime")
    if runtime.get("library") != "tokenizers":
        raise ValueError("manifest.runtime.library must be tokenizers")
    if runtime.get("encode_add_special_tokens") is not False:
        raise ValueError("canonical encode must disable implicit special tokens")
    if runtime.get("decode_skip_special_tokens") is not False:
        raise ValueError("canonical decode must preserve special tokens by default")
    if tokenizer_json.get("normalizer") != runtime.get("normalization"):
        raise ValueError("tokenizer normalization does not match the manifest")
    model_config = _mapping(tokenizer_json.get("model"), "tokenizer.model")
    if model_config.get("byte_fallback") is not runtime.get("byte_fallback"):
        raise ValueError("tokenizer byte_fallback does not match the manifest")

    tokenizer = Tokenizer.from_file(str(validated_paths["tokenizer"]))
    vocabulary = tokenizer.get_vocab(with_added_tokens=True)
    vocab_size = _integer(runtime.get("vocab_size"), "manifest.runtime.vocab_size")
    max_token_id = _integer(runtime.get("max_token_id"), "manifest.runtime.max_token_id")
    if tokenizer.get_vocab_size(with_added_tokens=True) != vocab_size:
        raise ValueError("tokenizer vocabulary size does not match the manifest")
    if not vocabulary or max(vocabulary.values()) != max_token_id:
        raise ValueError("tokenizer maximum token ID does not match the manifest")
    if set(vocabulary.values()) != set(range(vocab_size)):
        raise ValueError("canonical tokenizer IDs must be contiguous from zero")

    special_tokens = _mapping(manifest.get("special_tokens"), "manifest.special_tokens")
    if set(special_tokens) != _SPECIAL_ROLES:
        raise ValueError(f"manifest.special_tokens roles must be {sorted(_SPECIAL_ROLES)}")
    for role in sorted(_SPECIAL_ROLES):
        token_config = _mapping(special_tokens[role], f"manifest.special_tokens.{role}")
        if set(token_config) != {"token", "id"}:
            raise ValueError(f"manifest.special_tokens.{role} must contain token and id")
        token = token_config["token"]
        token_id = _integer(token_config["id"], f"manifest.special_tokens.{role}.id")
        if not isinstance(token, str) or not token:
            raise TypeError(f"manifest.special_tokens.{role}.token must be a string")
        if tokenizer.token_to_id(token) != token_id or tokenizer.id_to_token(token_id) != token:
            raise ValueError(f"special token {role} does not match the tokenizer artifact")

    probes = manifest.get("probes")
    if not isinstance(probes, list) or not probes:
        raise ValueError("manifest.probes must be a non-empty list")
    for index, probe_value in enumerate(probes):
        probe = _mapping(probe_value, f"manifest.probes[{index}]")
        if set(probe) != {"text", "ids"}:
            raise ValueError(f"manifest.probes[{index}] must contain text and ids")
        text = probe["text"]
        _validate_text(text)
        expected_ids = probe["ids"]
        if not isinstance(expected_ids, list) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in expected_ids
        ):
            raise TypeError(f"manifest.probes[{index}].ids must be integer IDs")
        actual_ids = tokenizer.encode(text, add_special_tokens=False).ids
        if actual_ids != expected_ids:
            raise ValueError(f"tokenizer probe {index} IDs do not match the manifest")
        decoded = tokenizer.decode(actual_ids, skip_special_tokens=False)
        if decoded != text:
            raise ValueError(f"tokenizer probe {index} does not round-trip exactly")
    return tokenizer


def _validate_file(base_path: Path, role: str, value: Any) -> Path:
    file_config = _mapping(value, f"manifest.files.{role}")
    if set(file_config) != {"path", "size_bytes", "sha256"}:
        raise ValueError(f"manifest.files.{role} must contain path, size_bytes, and sha256")
    relative_path = file_config["path"]
    if not isinstance(relative_path, str) or not relative_path:
        raise TypeError(f"manifest.files.{role}.path must be a non-empty string")
    relative = Path(relative_path)
    if relative.is_absolute() or relative.parent != Path("."):
        raise ValueError(f"manifest.files.{role}.path must name a file beside the manifest")
    path = (base_path / relative).resolve()
    if path.parent != base_path.resolve():
        raise ValueError(f"manifest.files.{role}.path escapes the install directory")
    if not path.is_file():
        raise FileNotFoundError(f"canonical tokenizer {role} file not found: {path}")
    size_bytes = _integer(file_config["size_bytes"], f"manifest.files.{role}.size_bytes")
    if path.stat().st_size != size_bytes:
        raise ValueError(f"canonical tokenizer {role} file size does not match the manifest")
    expected_sha256 = file_config["sha256"]
    _require_sha256(expected_sha256, f"manifest.files.{role}.sha256")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"canonical tokenizer {role} SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _special_id(special_tokens: Mapping[str, Any], role: str) -> int:
    token = _mapping(special_tokens[role], f"manifest.special_tokens.{role}")
    return _integer(token["id"], f"manifest.special_tokens.{role}.id")


def _require_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")


def _validate_text(text: Any) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise ValueError(
            "text must be valid UTF-8 Unicode without surrogate code points"
        ) from error
