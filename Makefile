.PHONY: help sync activate train train-tokenizer

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make sync            - Create/update the local uv environment' \
		'  make activate  - Print the command to activate the local virtual environment' \
		'  make train-tokenizer  - Train and save the tokenizer artifact inside uv' \
		'  make train            - Run the Hydra model training entrypoint inside uv'

sync:
	uv sync

activate:
	@test -f .venv/bin/activate || { printf '%s\n' 'No .venv yet; run: make sync' >&2; exit 1; }; \
		. .venv/bin/activate && exec $${SHELL:-/bin/sh} -i

train-tokenizer:
	uv run python src/train_tokenizer.py

train:
	uv run python src/train.py
