#!/usr/bin/env bash
# scripts/test_lite.sh
#
# End-to-end validation for the LITE stack.
# Checks: services alive → model loaded → inference working.
#
# Usage:
#   ./scripts/test_lite.sh
#   MODEL_ALIAS=my-model ./scripts/test_lite.sh

set -euo pipefail

GATEWAY="${GATEWAY_URL:-http://localhost:8080}"
MRM="${MRM_URL:-http://model_runtime_manager:8010}"
CPU_RUNTIME="${CPU_RUNTIME_URL:-http://localhost:8090}"
MODEL="${MODEL_ALIAS:-cpu-model}"
TIMEOUT=10

GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[1;33m"
NC="\033[0m"

pass() { echo -e "${GREEN}✅ $1${NC}"; }
fail() { echo -e "${RED}❌ $1${NC}"; FAILURES=$((FAILURES + 1)); }
info() { echo -e "${YELLOW}ℹ  $1${NC}"; }

FAILURES=0

echo ""
echo "═══════════════════════════════════════════════"
echo "  LITE Stack — End-to-End Validation"
echo "═══════════════════════════════════════════════"
echo ""

# ── 1. Service health checks ─────────────────────────────────────────
info "1. Service health checks"

check_health() {
    local name="$1"
    local url="$2"
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$TIMEOUT" "$url" 2>/dev/null || echo "000")
    if [[ "$http_code" == "200" ]]; then
        pass "$name is healthy ($url)"
    else
        fail "$name returned HTTP $http_code ($url)"
    fi
}

check_health "Gateway"     "$GATEWAY/health"
check_health "MRM"         "http://localhost:8010/health"
check_health "CPU Runtime" "$CPU_RUNTIME/health"

echo ""
# ── 2. CPU Runtime: model loaded ─────────────────────────────────────
info "2. CPU Runtime: model loaded"

models_resp=$(curl -s --max-time "$TIMEOUT" "$CPU_RUNTIME/v1/models" 2>/dev/null || echo "{}")
if echo "$models_resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('data'), 'no models'" 2>/dev/null; then
    model_id=$(echo "$models_resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])")
    pass "CPU runtime serving model: $model_id"
else
    fail "CPU runtime /v1/models returned: $models_resp"
fi

echo ""
# ── 3. Gateway: model list ────────────────────────────────────────────
info "3. Gateway /v1/models"

gw_models=$(curl -s --max-time "$TIMEOUT" "$GATEWAY/v1/models" 2>/dev/null || echo "{}")
if echo "$gw_models" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'data' in d" 2>/dev/null; then
    pass "Gateway /v1/models OK"
else
    info "Gateway /v1/models returned: $gw_models (may be empty before first ensure)"
fi

echo ""
# ── 4. MRM ensure (register model with CPU fallback) ─────────────────
info "4. MRM ensure: requesting model '$MODEL'"

ensure_resp=$(curl -s --max-time 30 -X POST "http://localhost:8010/models/ensure" \
    -H "Content-Type: application/json" \
    -d "{\"base_model\": \"$MODEL\"}" 2>/dev/null || echo "{}")

if echo "$ensure_resp" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('state') == 'READY'" 2>/dev/null; then
    api_base=$(echo "$ensure_resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('api_base',''))")
    runtime_type=$(echo "$ensure_resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('runtime_type','?'))")
    fallback=$(echo "$ensure_resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('fallback',False))")
    pass "MRM ensure → READY (runtime=$runtime_type, api_base=$api_base)"
    [[ "$fallback" == "True" ]] && info "CPU fallback active (no GPU detected)"
else
    fail "MRM ensure returned: $ensure_resp"
fi

echo ""
# ── 5. Direct CPU Runtime inference ──────────────────────────────────
info "5. Direct inference via CPU Runtime"

cpu_resp=$(curl -s --max-time 60 -X POST "$CPU_RUNTIME/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly: OK\"}],
      \"max_tokens\": 10,
      \"temperature\": 0.0
    }" 2>/dev/null || echo "{}")

if echo "$cpu_resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
choices = d.get('choices', [])
assert choices, 'no choices'
content = choices[0]['message']['content']
assert content.strip(), 'empty response'
print(f'Response: {content.strip()[:80]}')
" 2>/dev/null; then
    pass "CPU Runtime inference works"
else
    fail "CPU Runtime inference failed: $cpu_resp"
fi

echo ""
# ── 6. Gateway inference (end-to-end) ────────────────────────────────
info "6. End-to-end inference via Gateway"

gw_resp=$(curl -s --max-time 60 -X POST "$GATEWAY/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Reply with exactly: LITE_OK\"}],
      \"max_tokens\": 10,
      \"temperature\": 0.0,
      \"runtime_preference\": \"cpu\"
    }" 2>/dev/null || echo "{}")

if echo "$gw_resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
choices = d.get('choices', [])
assert choices, 'no choices in: ' + json.dumps(d)
content = choices[0]['message']['content']
assert content.strip(), 'empty response'
print(f'Response: {content.strip()[:80]}')
" 2>/dev/null; then
    pass "Gateway end-to-end inference works"
else
    fail "Gateway inference failed: $gw_resp"
fi

echo ""
# ── 7. GPU absent — no crash ──────────────────────────────────────────
info "7. Verify system survives without GPU"

gpu_resp=$(curl -s --max-time "$TIMEOUT" "http://localhost:8010/gpu/metrics" 2>/dev/null || echo "{}")
if echo "$gpu_resp" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
    pass "MRM /gpu/metrics returns valid JSON (ok=false is expected without GPU)"
else
    fail "MRM /gpu/metrics returned invalid JSON: $gpu_resp"
fi

echo ""
# ── Summary ───────────────────────────────────────────────────────────
echo "═══════════════════════════════════════════════"
if [[ "$FAILURES" -eq 0 ]]; then
    echo -e "${GREEN}All checks passed. LITE stack is healthy.${NC}"
else
    echo -e "${RED}$FAILURES check(s) failed. Review output above.${NC}"
    exit 1
fi
echo "═══════════════════════════════════════════════"
echo ""

echo "Memory usage estimate:"
docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}" \
    lite_redis lite_cpu_runtime lite_mrm lite_gateway 2>/dev/null || true
