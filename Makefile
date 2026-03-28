.PHONY: help sync activate test train train-tokenizer

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make sync      - Create/update the local uv environment with dev dependencies' \
		'  make activate  - Print the command to activate the local virtual environment' \
		'  make test             - Run the unit tests inside uv' \
		'  make train-tokenizer  - Train and save the tokenizer artifact inside uv' \
		'  make train            - Run the Hydra model training entrypoint inside uv'

sync:
	uv sync --dev

activate:
	@test -f .venv/bin/activate || { printf '%s\n' 'No .venv yet; run: make sync' >&2; exit 1; }; \
		. .venv/bin/activate && exec $${SHELL:-/bin/sh} -i

test:
	uv run python -m pytest -q

train-tokenizer:
	uv run python src/train_tokenizer.py

train:
	uv run python src/train.py
