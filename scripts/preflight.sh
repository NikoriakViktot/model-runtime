#!/usr/bin/env bash
# scripts/preflight.sh
#
# Preflight checks for the LITE stack.
# Exits 1 (with clear instructions) if any required condition is not met.
#
# Checks:
#   1. Docker daemon is running
#   2. cpu_runtime image is built
#   3. model.gguf exists (path from GGUF_MODELS_PATH env var)
#   4. Required host ports are free
#   5. docker-compose binary is available
#
# Usage:
#   bash scripts/preflight.sh
#   make lite-preflight

set -euo pipefail

# ── Colours ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'; RESET='\033[0m'
ok()   { echo -e "${GREEN}  ✓${RESET}  $*"; }
warn() { echo -e "${YELLOW}  ⚠${RESET}  $*"; }
fail() { echo -e "${RED}  ✗${RESET}  $*"; }

ERRORS=0

echo ""
echo "  model-runtime LITE — preflight checks"
echo "  ──────────────────────────────────────"

# ── 1. Docker daemon ────────────────────────────────────────────────────────
if docker info >/dev/null 2>&1; then
    ok "Docker daemon is running"
else
    fail "Docker daemon is not running (or current user lacks permission)"
    echo "     Fix: start Docker Desktop / systemctl start docker"
    ERRORS=$((ERRORS + 1))
fi

# ── 2. docker compose available ─────────────────────────────────────────────
if docker compose version >/dev/null 2>&1; then
    ok "docker compose available"
elif command -v docker-compose >/dev/null 2>&1; then
    warn "Found docker-compose v1 (legacy). Recommend upgrading to Docker Compose v2."
    warn "Makefile targets use 'docker compose' (v2 syntax)."
else
    fail "docker compose not found"
    echo "     Fix: install Docker Desktop or 'apt install docker-compose-plugin'"
    ERRORS=$((ERRORS + 1))
fi

# ── 3. cpu_runtime image ─────────────────────────────────────────────────────
if docker image inspect model-runtime-cpu:latest >/dev/null 2>&1; then
    ok "model-runtime-cpu:latest image exists"
else
    fail "model-runtime-cpu:latest image not found"
    echo "     Fix: run  make lite-build  (or: docker build -t model-runtime-cpu:latest ./cpu_runtime)"
    ERRORS=$((ERRORS + 1))
fi

# ── 4. GGUF model file ────────────────────────────────────────────────────────
GGUF_MODELS_PATH="${GGUF_MODELS_PATH:-./hf_cache/gguf}"
MODEL_FILE="${GGUF_MODELS_PATH}/model.gguf"

if [ -f "${MODEL_FILE}" ]; then
    SIZE=$(du -sh "${MODEL_FILE}" 2>/dev/null | cut -f1)
    ok "GGUF model found: ${MODEL_FILE} (${SIZE})"
else
    fail "GGUF model not found at: ${MODEL_FILE}"
    echo "     The LITE stack cannot start without a GGUF model."
    echo ""
    echo "     Quick fix — download Qwen2.5-1.5B-Instruct Q4 (~1 GB):"
    echo "       make lite-model"
    echo ""
    echo "     Manual download:"
    echo "       mkdir -p ${GGUF_MODELS_PATH}"
    echo "       pip install huggingface_hub[cli]"
    echo "       huggingface-cli download bartowski/Qwen2.5-1.5B-Instruct-GGUF \\"
    echo "         Qwen2.5-1.5B-Instruct-Q4_K_M.gguf \\"
    echo "         --local-dir ${GGUF_MODELS_PATH} --local-dir-use-symlinks False"
    echo "       mv ${GGUF_MODELS_PATH}/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf ${MODEL_FILE}"
    ERRORS=$((ERRORS + 1))
fi

# ── 5. Port availability ──────────────────────────────────────────────────────
check_port() {
    local port="$1"
    local label="$2"
    if lsof -iTCP:"${port}" -sTCP:LISTEN -n -P >/dev/null 2>&1 || \
       ss -tlnp "sport = :${port}" 2>/dev/null | grep -q ":${port}"; then
        local owner
        owner=$(lsof -iTCP:"${port}" -sTCP:LISTEN -n -P 2>/dev/null | awk 'NR==2{print $1}' || echo "unknown")
        fail "Port ${port} (${label}) is already in use by: ${owner}"
        echo "     Fix: stop the process using port ${port}, or override:"
        echo "       export LITE_$(echo "${label}" | tr '[:lower:]' '[:upper:]' | tr ' ' '_')_PORT=<free-port>"
        ERRORS=$((ERRORS + 1))
    else
        ok "Port ${port} (${label}) is free"
    fi
}

GATEWAY_PORT="${LITE_GATEWAY_PORT:-8181}"
MRM_PORT="${LITE_MRM_PORT:-8011}"
CPU_PORT="${LITE_CPU_PORT:-8091}"

check_port "${GATEWAY_PORT}" "gateway"
check_port "${MRM_PORT}"     "mrm"
check_port "${CPU_PORT}"     "cpu_runtime"

# ── 6. Disk space (warn only) ──────────────────────────────────────────────
FREE_GB=$(df -BG . 2>/dev/null | awk 'NR==2{print $4}' | tr -d 'G' || echo 0)
if [ "${FREE_GB}" -ge 5 ] 2>/dev/null; then
    ok "Disk space: ${FREE_GB}G free"
else
    warn "Low disk space: ${FREE_GB}G free (recommend ≥ 5 GB for model + images)"
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
if [ "${ERRORS}" -eq 0 ]; then
    echo -e "${GREEN}  All checks passed. Ready to start.${RESET}"
    echo ""
    echo "  Run:  make lite-up"
    echo ""
    exit 0
else
    echo -e "${RED}  ${ERRORS} check(s) failed. Fix the issues above before starting.${RESET}"
    echo ""
    exit 1
fi
