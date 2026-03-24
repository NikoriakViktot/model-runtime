# Makefile — model-runtime
#
# Targets are organised in three groups:
#   lite-*   LITE stack (CPU-only, no GPU required)
#   dev-*    Full dev stack (requires NVIDIA GPU + driver)
#   test-*   Test runners
#
# Quick-start (LITE):
#   make lite-build   # build cpu_runtime + gateway + mrm images
#   make lite-model   # download a tiny GGUF model (~1 GB)
#   make lite-up      # start the stack
#
# Quick-start (Dev / GPU):
#   make dev-up       # start full GPU stack

COMPOSE_LITE   := docker compose -f docker-compose.lite.yml
COMPOSE_GPU    := $(COMPOSE_LITE) -f docker-compose.gpu.yml
COMPOSE_DEV    := docker compose -f docker-compose.dev.yml

# Host port defaults (match docker-compose.lite.yml defaults)
LITE_GATEWAY_PORT  ?= 8181
LITE_MRM_PORT      ?= 8011
LITE_CPU_PORT      ?= 8091

# GGUF model defaults
GGUF_DIR       ?= ./hf_cache/gguf
GGUF_MODEL     ?= $(GGUF_DIR)/model.gguf
HF_REPO        ?= bartowski/Qwen2.5-1.5B-Instruct-GGUF
HF_FILE        ?= Qwen2.5-1.5B-Instruct-Q4_K_M.gguf

.DEFAULT_GOAL := help

# ──────────────────────────────────────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo ""
	@echo "  model-runtime Makefile"
	@echo ""
	@echo "  LITE stack (CPU-only, no GPU required)"
	@echo "  ─────────────────────────────────────"
	@echo "  make lite-preflight   Check prerequisites before first start"
	@echo "  make lite-build       Build all LITE service images"
	@echo "  make lite-model       Download default GGUF model (~1 GB)"
	@echo "  make lite-up          Start LITE stack (detached)"
	@echo "  make lite-up-ui       Start LITE stack + Streamlit UI"
	@echo "  make lite-up-gpu      Start LITE stack with GPU overlay"
	@echo "  make lite-down        Stop and remove LITE containers"
	@echo "  make lite-logs        Follow all LITE container logs"
	@echo "  make lite-status      Show LITE container status + ports"
	@echo "  make lite-test        Run smoke tests against running LITE stack"
	@echo "  make lite-shell-gw    Open shell in lite_gateway container"
	@echo "  make lite-shell-mrm   Open shell in lite_mrm container"
	@echo ""
	@echo "  Full dev stack (requires NVIDIA GPU)"
	@echo "  ────────────────────────────────────"
	@echo "  make dev-up           Start full dev stack (detached)"
	@echo "  make dev-down         Stop and remove dev containers"
	@echo "  make dev-logs         Follow all dev container logs"
	@echo "  make dev-status       Show dev container status"
	@echo ""
	@echo "  Build"
	@echo "  ─────"
	@echo "  make build-cpu        Build cpu_runtime image"
	@echo "  make build-gateway    Build gateway image"
	@echo "  make build-mrm        Build model_runtime_manager image"
	@echo "  make build-all        Build all images"
	@echo ""
	@echo "  Tests"
	@echo "  ─────"
	@echo "  make test             Run all unit tests (docker)"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-chaos       Run chaos tests (requires running stack)"
	@echo ""
	@echo "  Utilities"
	@echo "  ─────────"
	@echo "  make clean            Remove stopped containers + dangling images"
	@echo "  make prune            Full docker system prune (ask for confirm)"
	@echo ""

# ──────────────────────────────────────────────────────────────────────────────
# LITE stack
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: lite-preflight
lite-preflight:
	@bash scripts/preflight.sh

.PHONY: lite-build
lite-build: build-cpu build-gateway build-mrm
	@echo "✓ All LITE images built."

.PHONY: lite-model
lite-model:
	@echo "→ Downloading GGUF model from HuggingFace..."
	@mkdir -p $(GGUF_DIR)
	@if [ -f "$(GGUF_MODEL)" ]; then \
		echo "  Model already exists at $(GGUF_MODEL). Delete it to re-download."; \
	else \
		pip install -q huggingface_hub[cli] && \
		huggingface-cli download $(HF_REPO) $(HF_FILE) \
			--local-dir $(GGUF_DIR) \
			--local-dir-use-symlinks False && \
		mv $(GGUF_DIR)/$(HF_FILE) $(GGUF_MODEL) && \
		echo "✓ Model saved to $(GGUF_MODEL)"; \
	fi

.PHONY: lite-up
lite-up: lite-preflight
	$(COMPOSE_LITE) up -d
	@echo ""
	@echo "  LITE stack is up!"
	@echo "  Gateway  → http://localhost:$(LITE_GATEWAY_PORT)"
	@echo "  MRM      → http://localhost:$(LITE_MRM_PORT)"
	@echo "  CPU RT   → http://localhost:$(LITE_CPU_PORT)"
	@echo ""
	@echo "  Run 'make lite-logs' to follow logs."
	@echo "  Run 'make lite-status' to check health."

.PHONY: lite-up-ui
lite-up-ui: lite-preflight
	$(COMPOSE_LITE) --profile ui up -d
	@echo "  Frontend → http://localhost:$${LITE_FRONTEND_PORT:-8192}"

.PHONY: lite-up-gpu
lite-up-gpu: lite-preflight
	$(COMPOSE_GPU) up -d
	@echo "  GPU overlay active — CPU fallback disabled."

.PHONY: lite-down
lite-down:
	$(COMPOSE_LITE) --profile ui down

.PHONY: lite-logs
lite-logs:
	$(COMPOSE_LITE) --profile ui logs -f

.PHONY: lite-status
lite-status:
	@echo "=== Container status ==="
	@$(COMPOSE_LITE) ps
	@echo ""
	@echo "=== Health probes ==="
	@echo -n "  cpu_runtime : "; \
		docker inspect --format='{{.State.Health.Status}}' lite_cpu_runtime 2>/dev/null || echo "not running"
	@echo -n "  mrm         : "; \
		docker inspect --format='{{.State.Health.Status}}' lite_mrm 2>/dev/null || echo "not running"
	@echo -n "  gateway     : "; \
		docker inspect --format='{{.State.Health.Status}}' lite_gateway 2>/dev/null || echo "not running"
	@echo ""
	@echo "=== Port mapping ==="
	@echo "  Gateway  → http://localhost:$(LITE_GATEWAY_PORT)"
	@echo "  MRM      → http://localhost:$(LITE_MRM_PORT)"
	@echo "  CPU RT   → http://localhost:$(LITE_CPU_PORT)"

.PHONY: lite-test
lite-test:
	@bash scripts/test_lite.sh

.PHONY: lite-shell-gw
lite-shell-gw:
	docker exec -it lite_gateway /bin/bash

.PHONY: lite-shell-mrm
lite-shell-mrm:
	docker exec -it lite_mrm /bin/bash

# ──────────────────────────────────────────────────────────────────────────────
# Dev / GPU stack
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: dev-up
dev-up:
	$(COMPOSE_DEV) up -d

.PHONY: dev-down
dev-down:
	$(COMPOSE_DEV) down

.PHONY: dev-logs
dev-logs:
	$(COMPOSE_DEV) logs -f

.PHONY: dev-status
dev-status:
	$(COMPOSE_DEV) ps

# ──────────────────────────────────────────────────────────────────────────────
# Build individual images
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: build-cpu
build-cpu:
	docker build -t model-runtime-cpu:latest ./cpu_runtime
	@echo "✓ model-runtime-cpu:latest"

.PHONY: build-gateway
build-gateway:
	docker build -t model-runtime-gateway:latest ./gateway
	@echo "✓ model-runtime-gateway:latest"

.PHONY: build-mrm
build-mrm:
	docker build -t model-runtime-mrm:latest ./model_runtime_manager
	@echo "✓ model-runtime-mrm:latest"

.PHONY: build-all
build-all: build-cpu build-gateway build-mrm
	@echo "✓ All images built."

# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: test
test: test-unit

.PHONY: test-unit
test-unit:
	@echo "→ Running unit tests in docker..."
	docker run --rm \
		-v "$(PWD)":/project \
		-w /project \
		python:3.12-slim \
		bash -c " \
			pip install -q \
				fastapi httpx pydantic pydantic-settings pytest pytest-asyncio \
				structlog prometheus-client docker redis pyyaml anyio \
			&& python -m pytest tests/unit/ -v --tb=short \
		"

.PHONY: test-chaos
test-chaos:
	@echo "→ Running chaos tests (LITE stack must be running)..."
	docker run --rm \
		-v "$(PWD)":/project \
		-w /project \
		--network ai_net_lite \
		python:3.12-slim \
		bash -c " \
			pip install -q pytest pytest-asyncio httpx \
			&& python -m pytest tests/chaos/ -v --tb=short \
		"

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	docker container prune -f
	docker image prune -f
	@echo "✓ Stopped containers and dangling images removed."

.PHONY: prune
prune:
	@echo "WARNING: This will remove ALL unused Docker resources."
	@read -p "Continue? [y/N] " ans && [ "$$ans" = "y" ]
	docker system prune -f
