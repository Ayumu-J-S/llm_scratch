from __future__ import annotations

import copy
import hashlib
import json
import re
import runpy
import shutil
import socket
from pathlib import Path

import hydra
import pytest
import torch
import torch.nn.functional as functional
from omegaconf import OmegaConf

import train as train_module
from data.stream_loader import StreamLoader, StreamLoaderError
from data.streaming_dataset import create_streaming_token_dataloader
from models.simple_decoder_transformer import SimpleDecoderTransformer
from tokenizer.canonical import CanonicalTokenizer


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets/tokenizers/llm-jp-v1"
FINGERPRINT = "12ccbc02d53338d1f5f506f2fec6e483fc08beea56cc1c04539d26e3025f484b"
CONFIG = {
    "manifest_path": "assets/tokenizers/llm-jp-v1/manifest.json",
    "expected_fingerprint": FINGERPRINT,
}
LOAD_DEBUG_CONFIG = runpy.run_path(str(ROOT / "scripts/debug_stream_loader.py"))[
    "load_debug_config"
]
PROBES = {
    "日本語とEnglish。": [31, 26702, 15291, 38106, 25590],
    "Hello, world!": [18228, 10020, 10147, 10000],
    "記号: []{}<> /\\ +-=_": [
        31,
        31974,
        10011,
        31,
        10004,
        10003,
        10009,
        10008,
        10006,
        10012,
        31,
        10016,
        10002,
        31,
        10001,
        10015,
        10014,
        10022,
    ],
    "絵文字🙂🌸🚀": [31, 27072, 26865, 273, 192, 186, 163, 49867, 273, 192, 187, 161],
}
SPECIAL_TOKENS = json.loads((ASSET_DIR / "manifest.json").read_text(encoding="utf-8"))[
    "special_tokens"
]
RESERVED_SPECIAL_IDS = {config["id"] for config in SPECIAL_TOKENS.values()}


def test_canonical_identity_special_tokens_and_offline_probes(monkeypatch):
    def reject_network(*args, **kwargs):
        raise AssertionError("canonical tokenizer runtime must not access the network")

    monkeypatch.setattr(socket, "socket", reject_network)
    tokenizer = CanonicalTokenizer.from_config(CONFIG)

    assert tokenizer.fingerprint == FINGERPRINT
    assert tokenizer.vocab_size == 50_570
    assert tokenizer.max_token_id == 50_569
    assert tokenizer.unk_token_id == 0
    assert tokenizer.bos_token_id == 1
    assert tokenizer.pad_token_id == 4
    assert tokenizer.eos_token_id == 7
    for text, expected_ids in PROBES.items():
        assert tokenizer.encode(text) == expected_ids
        assert tokenizer.decode(expected_ids) == text
    assert tokenizer.decode([tokenizer.eos_token_id]) == "<EOD|LLM-jp>"
    assert tokenizer.decode([tokenizer.eos_token_id], skip_special_tokens=True) == ""
    tokenizer.assert_fingerprint(FINGERPRINT)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        tokenizer.assert_fingerprint("0" * 64)


def test_canonical_tokenizer_rejects_invalid_utf8_and_ids():
    tokenizer = CanonicalTokenizer.from_config(CONFIG)

    with pytest.raises(ValueError, match="valid UTF-8"):
        tokenizer.encode("broken\ud800")
    with pytest.raises(TypeError, match="integers"):
        tokenizer.decode([1.5])
    with pytest.raises(ValueError, match="outside canonical range"):
        tokenizer.decode([tokenizer.vocab_size])


@pytest.mark.parametrize(
    ("role", "special_token", "special_id"),
    [(role, config["token"], config["id"]) for role, config in SPECIAL_TOKENS.items()],
)
def test_raw_text_rejects_every_reserved_special_token(role, special_token, special_id):
    tokenizer = CanonicalTokenizer.from_config(CONFIG)
    text = f"ordinary prefix {special_token} ordinary suffix"

    with pytest.raises(
        ValueError,
        match=re.escape(f"role={role}, token={special_token!r}, id={special_id}"),
    ):
        tokenizer.encode(text)


@pytest.mark.parametrize(
    ("role", "text"),
    [
        ("pad", "<pad|LLM-jp> suffix"),
        ("pad", "prefix <pad|LLM-jp>"),
        ("pad", "<pad|LLM-jp><pad|LLM-jp>"),
        ("eos_eod", "<EOD|LLM-jp> suffix"),
        ("eos_eod", "prefix <EOD|LLM-jp>"),
        ("eos_eod", "<EOD|LLM-jp><EOD|LLM-jp>"),
    ],
)
def test_raw_pad_and_eod_are_rejected_at_boundaries_and_when_repeated(role, text):
    tokenizer = CanonicalTokenizer.from_config(CONFIG)

    with pytest.raises(ValueError, match=rf"role={role}, .*id="):
        tokenizer.encode(text)


@pytest.mark.parametrize(
    "text",
    [
        "自然な日本語の文章です。",
        "A clean English sentence.",
        "日本語 and English in one clean document 🙂",
        "絵文字🙂🌸🚀と記号 []{}<> /\\ +=_",
    ],
)
def test_clean_multilingual_text_never_emits_reserved_special_ids(text):
    tokenizer = CanonicalTokenizer.from_config(CONFIG)

    token_ids = tokenizer.encode(text)

    assert RESERVED_SPECIAL_IDS.isdisjoint(token_ids)


def test_frozen_corpus_retains_selected_wrapper_id_digest_without_reserved_ids():
    tokenizer = CanonicalTokenizer.from_config(CONFIG)
    documents = [
        json.loads(line)
        for line in (ROOT / "tests/fixtures/tokenizer_comparison/v1/corpus.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    encodings = [tokenizer.encode(document["text"]) for document in documents]
    digest = hashlib.sha256(json.dumps(encodings, separators=(",", ":")).encode()).hexdigest()

    assert len(encodings) == 160
    assert digest == "3c1078f72957170fd3c7ac94c9d3313b367f3bf243562a693588810f07dfe907"
    assert all(RESERVED_SPECIAL_IDS.isdisjoint(token_ids) for token_ids in encodings)


@pytest.mark.parametrize(
    ("special_token", "role"),
    [("<pad|LLM-jp>", "pad"), ("<EOD|LLM-jp>", "eos_eod")],
)
def test_local_text_rejects_raw_pad_and_eod_before_batch_construction(
    tmp_path, special_token, role
):
    input_path = tmp_path / "contaminated.txt"
    input_path.write_text(f"clean prefix {special_token} clean suffix", encoding="utf-8")
    tokenizer = CanonicalTokenizer.from_config(CONFIG)

    with pytest.raises(ValueError, match=rf"role={role}, .*id="):
        train_module.load_token_ids(str(input_path), tokenizer, "training")


@pytest.mark.parametrize("process_prefetch", [False, True])
@pytest.mark.parametrize(
    ("special_token", "role"),
    [("<pad|LLM-jp>", "pad"), ("<EOD|LLM-jp>", "eos_eod")],
)
def test_stream_loader_rejects_raw_pad_and_eod_before_emitting_samples(
    process_prefetch, special_token, role
):
    config = {
        "tokenizer": dict(CONFIG),
        "output_mode": "raw_text",
        "max_tokens": "max",
        "add_eos": True,
        "prefetch": {
            "enabled": process_prefetch,
            "mode": "process",
            "buffer_size": 1,
        },
        "datasets": [
            {
                "name": "contaminated",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": f"clean {special_token} raw"}],
            }
        ],
    }

    expected_error = StreamLoaderError if process_prefetch else ValueError
    with pytest.raises(expected_error, match=rf"role={role}, .*id="):
        next(iter(StreamLoader(config)))


def test_loader_appends_one_eod_to_clean_text_without_padding():
    tokenizer = CanonicalTokenizer.from_config(CONFIG)
    text = "日本語 and English 🙂"
    config = {
        "tokenizer": dict(CONFIG),
        "output_mode": "tokenized_docs",
        "max_tokens": "max",
        "add_eos": True,
        "datasets": [
            {
                "name": "clean",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": text}],
            }
        ],
    }

    sample = next(iter(StreamLoader(config)))
    token_ids = sample["input_ids"].tolist()

    assert token_ids == tokenizer.encode(text) + [tokenizer.eos_token_id]
    assert token_ids.count(tokenizer.eos_token_id) == 1
    assert tokenizer.pad_token_id not in token_ids


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda manifest: manifest["upstream"].pop("revision"), "upstream.revision"),
        (
            lambda manifest: manifest["upstream"].update(revision="main"),
            "upstream.revision",
        ),
        (
            lambda manifest: manifest["upstream"].update(tokenizer_source_revision="main"),
            "upstream.tokenizer_source_revision",
        ),
        (
            lambda manifest: manifest["special_tokens"]["eos_eod"].update(id=6),
            "unique token strings and IDs",
        ),
        (lambda manifest: manifest["runtime"].update(vocab_size=50_569), "vocabulary size"),
        (lambda manifest: manifest["probes"][0]["ids"].append(1), "probe 0 IDs"),
    ],
)
def test_manifest_semantic_mutations_are_rejected(tmp_path, mutation, message):
    install_dir = _copy_install(tmp_path)
    manifest = _read_manifest(install_dir)
    mutation(manifest)
    config = _write_refingerprinted_manifest(install_dir, manifest)

    with pytest.raises((TypeError, ValueError), match=message):
        CanonicalTokenizer.from_config(config)


def test_manifest_and_expected_fingerprint_mutations_are_rejected(tmp_path):
    install_dir = _copy_install(tmp_path)
    manifest_path = install_dir / "manifest.json"
    manifest = _read_manifest(install_dir)
    manifest["upstream"]["revision"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest fingerprint is invalid"):
        CanonicalTokenizer.from_config(
            {"manifest_path": str(manifest_path), "expected_fingerprint": FINGERPRINT}
        )

    manifest = json.loads((ASSET_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest["fingerprint"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest fingerprint is invalid"):
        CanonicalTokenizer.from_config(
            {"manifest_path": str(manifest_path), "expected_fingerprint": FINGERPRINT}
        )

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        CanonicalTokenizer.from_config(
            {
                "manifest_path": str(ASSET_DIR / "manifest.json"),
                "expected_fingerprint": "0" * 64,
            }
        )


def test_refingerprinted_manifest_rejects_special_role_alias(tmp_path):
    install_dir = _copy_install(tmp_path)
    manifest = _read_manifest(install_dir)
    manifest["special_tokens"]["additional_eos"] = copy.deepcopy(
        manifest["special_tokens"]["eos_eod"]
    )
    config = _write_refingerprinted_manifest(install_dir, manifest)

    with pytest.raises(ValueError, match="eight unique token strings and IDs"):
        CanonicalTokenizer.from_config(config)


def test_refingerprinted_manifest_rejects_ordinary_vocabulary_substitution(tmp_path):
    install_dir = _copy_install(tmp_path)
    manifest = _read_manifest(install_dir)
    tokenizer_json = _read_tokenizer_json(install_dir)
    ordinary_token = tokenizer_json["model"]["vocab"][8][0]
    manifest["special_tokens"]["additional_eos"] = {"token": ordinary_token, "id": 8}
    config = _write_refingerprinted_manifest(install_dir, manifest)

    with pytest.raises(ValueError, match="must exactly match tokenizer artifact"):
        CanonicalTokenizer.from_config(config)


def test_refingerprinted_artifact_rejects_missing_special_added_token(tmp_path):
    install_dir = _copy_install(tmp_path)
    tokenizer_json = _read_tokenizer_json(install_dir)
    tokenizer_json["added_tokens"][7]["special"] = False
    config = _write_tokenizer_and_refingerprinted_manifest(install_dir, tokenizer_json)

    with pytest.raises(ValueError, match=r"artifact_special_count=7"):
        CanonicalTokenizer.from_config(config)


def test_refingerprinted_artifact_rejects_extra_special_added_token(tmp_path):
    install_dir = _copy_install(tmp_path)
    tokenizer_json = _read_tokenizer_json(install_dir)
    tokenizer_json["added_tokens"].append(
        {
            "id": 8,
            "content": tokenizer_json["model"]["vocab"][8][0],
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    )
    config = _write_tokenizer_and_refingerprinted_manifest(install_dir, tokenizer_json)

    with pytest.raises(ValueError, match=r"artifact_special_count=9"):
        CanonicalTokenizer.from_config(config)


def test_missing_moved_and_mutated_artifact_bytes_are_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        CanonicalTokenizer.from_config(
            {
                "manifest_path": str(tmp_path / "missing/manifest.json"),
                "expected_fingerprint": FINGERPRINT,
            }
        )

    install_dir = _copy_install(tmp_path)
    tokenizer_path = install_dir / "tokenizer.json"
    moved_path = install_dir / "tokenizer.moved.json"
    tokenizer_path.rename(moved_path)
    config = {
        "manifest_path": str(install_dir / "manifest.json"),
        "expected_fingerprint": FINGERPRINT,
    }
    with pytest.raises(FileNotFoundError, match="tokenizer file not found"):
        CanonicalTokenizer.from_config(config)

    moved_path.rename(tokenizer_path)
    tokenizer_bytes = bytearray(tokenizer_path.read_bytes())
    tokenizer_bytes[100] ^= 1
    tokenizer_path.write_bytes(tokenizer_bytes)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        CanonicalTokenizer.from_config(config)


def test_process_child_revalidates_identity_before_source_access(tmp_path):
    install_dir = _copy_install(tmp_path)
    config = {
        "tokenizer": {
            "manifest_path": str(install_dir / "manifest.json"),
            "expected_fingerprint": FINGERPRINT,
        },
        "output_mode": "raw_text",
        "max_tokens": 1,
        "add_eos": False,
        "prefetch": {"enabled": True, "mode": "process", "buffer_size": 1},
        "datasets": [
            {
                "name": "must_not_open",
                "type": "jsonl",
                "path": str(tmp_path / "missing-source.jsonl"),
                "ratio": 1.0,
            }
        ],
    }
    loader = StreamLoader(config)
    tokenizer_path = install_dir / "tokenizer.json"
    tokenizer_path.write_bytes(tokenizer_path.read_bytes() + b"\n")

    with pytest.raises(StreamLoaderError, match="tokenizer file size"):
        list(loader)


@pytest.mark.parametrize(
    "tokenizer_config",
    [
        {"kind": "qwen", "name": "Qwen/Qwen3-0.6B"},
        {"manifest_path": CONFIG["manifest_path"]},
        object(),
    ],
)
def test_stream_loader_rejects_noncanonical_tokenizer_inputs(tokenizer_config):
    config = {
        "tokenizer": tokenizer_config,
        "output_mode": "raw_text",
        "max_tokens": "max",
        "datasets": [
            {
                "name": "unused",
                "type": "memory",
                "ratio": 1.0,
                "documents": [{"text": "unused"}],
            }
        ],
    }

    with pytest.raises((TypeError, ValueError), match="canonical tokenizer config"):
        StreamLoader(config)


def test_identity_failure_precedes_data_source_access(tmp_path):
    calls = {"source": 0}

    def records():
        calls["source"] += 1
        yield {"text": "must not be read"}

    config = {
        "tokenizer": {
            "manifest_path": str(tmp_path / "missing.json"),
            "expected_fingerprint": FINGERPRINT,
        },
        "output_mode": "raw_text",
        "max_tokens": "max",
        "datasets": [
            {
                "name": "guarded",
                "type": "iterable",
                "ratio": 1.0,
                "iterable": records,
            }
        ],
    }

    with pytest.raises(FileNotFoundError, match="manifest not found"):
        StreamLoader(config)
    assert calls["source"] == 0


def test_train_stream_and_debug_hydra_configs_have_identical_ids(tmp_path):
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        train_config = hydra.compose(config_name="train")
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "config")):
        stream_config = hydra.compose(config_name="stream_loader")
    debug_config, sequence_length = LOAD_DEBUG_CONFIG(
        ROOT / "config/stream_loader.yaml", split="train"
    )

    train_tokenizer_config = train_module.build_tokenizer_config(train_config)
    stream_tokenizer_config = OmegaConf.to_container(stream_config.tokenizer, resolve=True)
    assert train_tokenizer_config == stream_tokenizer_config == debug_config["tokenizer"]
    assert sequence_length == 4096

    text = "設定経路 parity / 同一ID"
    tokenizers = [
        CanonicalTokenizer.from_config(train_tokenizer_config),
        CanonicalTokenizer.from_config(stream_tokenizer_config),
        CanonicalTokenizer.from_config(debug_config["tokenizer"]),
    ]
    expected_ids = tokenizers[0].encode(text)
    local_path = tmp_path / "local.txt"
    local_path.write_text(text, encoding="utf-8")
    local_ids = train_module.load_token_ids(str(local_path), tokenizers[0], "training")
    assert [tokenizer.encode(text) for tokenizer in tokenizers] == [expected_ids] * 3
    assert local_ids == expected_ids


def test_process_streamed_jsonl_batch_runs_through_model_with_finite_loss(tmp_path):
    jsonl_path = tmp_path / "bilingual.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps({"text": "日本語"}, ensure_ascii=False) + "\n" for _ in range(4)),
        encoding="utf-8",
    )
    config = {
        "tokenizer": dict(CONFIG),
        "max_tokens": 10,
        "add_eos": True,
        "prefetch": {"enabled": True, "mode": "process", "buffer_size": 2},
        "datasets": [
            {
                "name": "offline_jsonl",
                "type": "jsonl",
                "path": str(jsonl_path),
                "ratio": 1.0,
            }
        ],
    }
    loader = create_streaming_token_dataloader(
        config=config,
        sequence_length=4,
        batch_size=2,
        drop_last=True,
    )
    batch = next(iter(loader))
    tokenizer = CanonicalTokenizer.from_config(CONFIG)
    model = SimpleDecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_size=16,
        num_heads=4,
        max_len=4,
        num_layers=1,
        dropout=0.0,
        pad_token_id=tokenizer.pad_token_id,
    )

    logits = model(batch["inputs"])
    loss = functional.cross_entropy(logits.flatten(0, 1), batch["labels"].flatten())

    assert batch["inputs"].shape == (2, 4)
    assert logits.shape == (2, 4, tokenizer.vocab_size)
    assert torch.isfinite(loss)
    assert int(batch["inputs"].max()) < tokenizer.vocab_size
    assert torch.any(batch["labels"] == tokenizer.eos_token_id)
    assert not torch.any(batch["inputs"] == tokenizer.pad_token_id)
    assert not torch.any(batch["labels"] == tokenizer.pad_token_id)
    assert model.pad_token_id == tokenizer.pad_token_id
    assert model.embedding.token.embedding.padding_idx == tokenizer.pad_token_id
    assert not torch.any(model.make_padding_mask(batch["inputs"]))

    tokens_with_pad = batch["inputs"].clone()
    tokens_with_pad[0, 0] = tokenizer.pad_token_id
    expected_mask = torch.zeros_like(tokens_with_pad, dtype=torch.bool)
    expected_mask[0, 0] = True
    assert torch.equal(model.make_padding_mask(tokens_with_pad), expected_mask)


def _copy_install(tmp_path: Path) -> Path:
    install_dir = tmp_path / "llm-jp-v1"
    shutil.copytree(ASSET_DIR, install_dir)
    return install_dir


def _read_manifest(install_dir: Path) -> dict:
    return json.loads((install_dir / "manifest.json").read_text(encoding="utf-8"))


def _read_tokenizer_json(install_dir: Path) -> dict:
    return json.loads((install_dir / "tokenizer.json").read_text(encoding="utf-8"))


def _write_tokenizer_and_refingerprinted_manifest(
    install_dir: Path, tokenizer_json: dict
) -> dict[str, str]:
    tokenizer_path = install_dir / "tokenizer.json"
    tokenizer_path.write_text(json.dumps(tokenizer_json, ensure_ascii=False), encoding="utf-8")
    manifest = _read_manifest(install_dir)
    manifest["files"]["tokenizer"]["size_bytes"] = tokenizer_path.stat().st_size
    manifest["files"]["tokenizer"]["sha256"] = hashlib.sha256(
        tokenizer_path.read_bytes()
    ).hexdigest()
    return _write_refingerprinted_manifest(install_dir, manifest)


def _write_refingerprinted_manifest(install_dir: Path, manifest: dict) -> dict[str, str]:
    manifest = copy.deepcopy(manifest)
    manifest.pop("fingerprint", None)
    encoded = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()
    manifest["fingerprint"] = fingerprint
    manifest_path = install_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest_path": str(manifest_path), "expected_fingerprint": fingerprint}
