# LLM-jp v1 tokenizer source

This directory vendors tokenizer-only files selected by the TOK-001 frozen
comparison. It contains no model weights, model configuration, or chat template.

- Model repository: `llm-jp/llm-jp-13b-v1.0`
- Pinned model-repository revision:
  `c3134b3a958b56d443c1484a3d640502637cfbd2`
- Tokenizer source repository: `llm-jp/llm-jp-tokenizer`
- Pinned tokenizer-source revision:
  `132f21625417ed0f3dc6484bf0bc1fb6a433acdd`
- Tokenizer artifact:
  `https://huggingface.co/llm-jp/llm-jp-13b-v1.0/resolve/c3134b3a958b56d443c1484a3d640502637cfbd2/tokenizer.json`
- License source:
  `https://raw.githubusercontent.com/llm-jp/llm-jp-tokenizer/132f21625417ed0f3dc6484bf0bc1fb6a433acdd/LICENSE`
- SPDX license identifier: `Apache-2.0`

`manifest.json` is the machine-checked identity contract. Runtime loading is
offline and verifies the manifest fingerprint, pinned revisions, file sizes and
SHA-256 hashes, vocabulary, special-token semantics, tokenizer pipeline, and
deterministic probes before any data source is opened.
