.PHONY: help sync activate train smoke pretrain-streaming config-check \
	runtime-lock diagnose dgx-build dgx-diagnose dgx-smoke test-cpu \
	ci-sync ci-lint ci-test ci-config ci-lock ci-offline-smoke ci-cpu

DGX_IMAGE := llm-scratch:env-001
DGX_RUN_FLAGS := --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make sync            - Create/update the local uv environment' \
		'  make activate  - Print the command to activate the local virtual environment' \
		'  make train            - Run the default smoke_overfit Hydra entrypoint inside uv' \
		'  make smoke            - Run the one-epoch CPU smoke_overfit profile' \
		'  make pretrain-streaming - Run the manifest-backed streaming profile' \
		'  make config-check PROFILE=<name> - Compose and preflight a Hydra profile' \
		'  make runtime-lock     - Regenerate the non-Torch container dependency overlay' \
		'  make diagnose         - Report the current host development environment' \
		'  make dgx-build        - Build the pinned ARM64 DGX Spark image without cache' \
		'  make dgx-diagnose     - Require CUDA and BF16 in the DGX Spark image' \
		'  make dgx-smoke        - Run exactly ten BF16 CUDA optimizer steps' \
		'  make test-cpu         - Run the explicit CPU development test suite' \
		'  make ci-cpu           - Run the network-free pull-request quality gate'

sync:
	uv sync --locked --group dev

activate:
	@test -f .venv/bin/activate || { printf '%s\n' 'No .venv yet; run: make sync' >&2; exit 1; }; \
		. .venv/bin/activate && exec $${SHELL:-/bin/sh} -i

train:
	uv run python src/train.py

smoke:
	uv run python src/train.py profile=smoke_overfit runtime.device=cpu training.epochs=1 training.batch_size=2 wandb.mode=disabled

pretrain-streaming:
	uv run python src/train.py profile=pretrain_streaming

config-check:
	uv run python scripts/config_check.py profile=$${PROFILE:-smoke_overfit}

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

# CI-001: sync is the only target permitted to obtain packages. Every following
# target is explicitly offline, uses the already-created environment, and must
# fail rather than let uv resolve or download anything on demand.
ci-sync:
	uv sync --locked --no-default-groups --group dev

ci-lint:
	UV_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled WANDB_DISABLED=true uv run --no-sync ruff check .

ci-test:
	UV_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled WANDB_DISABLED=true uv run --no-sync pytest -q

ci-config:
	UV_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled WANDB_DISABLED=true uv run --no-sync python scripts/config_check.py profile=smoke_overfit

ci-lock:
	UV_OFFLINE=1 uv run --no-sync python scripts/check_lock_drift.py

ci-offline-smoke:
	UV_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=disabled WANDB_DISABLED=true uv run --no-sync python scripts/offline_smoke.py

ci-cpu: ci-sync ci-lint ci-test ci-config ci-lock ci-offline-smoke
