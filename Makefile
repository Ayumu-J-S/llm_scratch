.PHONY: help sync activate train train-tokenizer runtime-lock diagnose \
	dgx-build dgx-diagnose dgx-smoke test-cpu

DGX_IMAGE := llm-scratch:env-001
DGX_RUN_FLAGS := --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make sync            - Create/update the local uv environment' \
		'  make activate  - Print the command to activate the local virtual environment' \
		'  make train-tokenizer  - Train and save the tokenizer artifact inside uv' \
		'  make train            - Run the CUDA-default Hydra training entrypoint inside uv' \
		'  make runtime-lock     - Regenerate the non-Torch container dependency overlay' \
		'  make diagnose         - Report the current host development environment' \
		'  make dgx-build        - Build the pinned ARM64 DGX Spark image without cache' \
		'  make dgx-diagnose     - Require CUDA and BF16 in the DGX Spark image' \
		'  make dgx-smoke        - Run exactly ten BF16 CUDA optimizer steps' \
		'  make test-cpu         - Run the explicit CPU development test suite'

sync:
	uv sync --locked --group dev

activate:
	@test -f .venv/bin/activate || { printf '%s\n' 'No .venv yet; run: make sync' >&2; exit 1; }; \
		. .venv/bin/activate && exec $${SHELL:-/bin/sh} -i

train-tokenizer:
	uv run python src/train_tokenizer.py

train:
	uv run python src/train.py

runtime-lock:
	uv export --quiet --locked --no-default-groups --no-dev --no-emit-project --prune torch --no-header --output-file requirements/runtime.txt
	uv run python scripts/check_runtime_requirements.py requirements/runtime.txt

diagnose:
	PYTHONPATH=src uv run python scripts/diagnose_environment.py

dgx-build:
	docker build --platform linux/arm64 --no-cache -t $(DGX_IMAGE) .

dgx-diagnose:
	docker run $(DGX_RUN_FLAGS) $(DGX_IMAGE) python scripts/diagnose_environment.py --require-cuda --require-bf16

dgx-smoke:
	docker run $(DGX_RUN_FLAGS) $(DGX_IMAGE) python scripts/cuda_smoke.py

test-cpu:
	uv run pytest -q
