"""
tests/slo/test_slo.py

Service Level Objective (SLO) validation tests.

These tests enforce production-grade reliability guarantees.  Unlike
correctness tests (which verify *what* the system does), SLO tests verify
*how well* it does it — latency bounds, error rate budgets, and resource
efficiency under load.

SLOs UNDER TEST
---------------
1. Scheduler warm-path p99 < 50 ms
   The cache-hit path (placement already in Redis) must complete in under
   50 ms.  Violation → blocking the event loop or slow Redis access.

2. Router decision overhead p99 < 1 ms
   Pure Python routing logic over 1 000 calls must cost under 1 ms p99.
   Violation → quadratic algorithm, GIL contention, or accidental I/O.

3. Zero error rate under healthy upstream
   With a perfectly healthy vLLM mock, the gateway error rate must be 0%.
   Violation → spurious errors in the proxy or routing layer.

4. Error rate matches upstream failure rate (±10 pp)
   Under a fixed 25% upstream failure rate, gateway errors must be 25 ± 10%.
   Violation → errors being swallowed (under-counted) or phantom errors.

5. Inflight counter accuracy
   After N sequential requests, inflight must return to 0.
   Violation → counter leak that would eventually trigger false backpressure.

6. SLO status detector — reports violations correctly
   slo_status() must report violations for bad instances and not for good ones.
   This is the runtime invariant detector itself under test.

7. Performance regression gate
   Warm-path latency measured in this run is written to a baseline file.
   If a previous baseline exists, the current p99 must not exceed it by > 3×.
   Violation → catastrophic regression (e.g. accidental synchronous Redis call).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest
import respx

from gateway.services.router import InstanceInfo, LeastLoadedStrategy, ModelRouter
from tests.utils.fault_injection import LatencyTracker

pytestmark = pytest.mark.slo

NODE_URL = "http://node-1:8020"
BASELINE_PATH = Path("test-results/slo_baseline.json")


# ---------------------------------------------------------------------------
# 1. Scheduler warm-path p99 latency
# ---------------------------------------------------------------------------


@pytest.mark.slow
async def test_slo_scheduler_warm_path_p99_under_50ms(scheduler):
    """
    SLO: ensure() warm-path (cache hit) p99 must be under 50 ms.

    Cold-start places the model once; all subsequent calls hit the Redis
    placement cache.  With in-process fakeredis this should be <1 ms per
    call; 50 ms is the SLO guard against event-loop blockage.
    """
    from tests.slo.conftest import register_node, node_ensure_payload, MODEL

    await register_node(scheduler, "node-1", NODE_URL)
    tracker = LatencyTracker()
    N = 200

    async def mock_ensure(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.005)  # 5 ms simulated cold-start
        return httpx.Response(200, json=node_ensure_payload())

    async with respx.MockRouter() as mock:
        mock.post(f"{NODE_URL}/local/ensure").mock(side_effect=mock_ensure)
        # Cold start (not measured)
        await scheduler.ensure(MODEL)

    # Warm-path measurements (no mock needed — placement is cached)
    async with respx.MockRouter(assert_all_called=False):
        for _ in range(N):
            async with tracker.measure():
                await scheduler.ensure(MODEL)

    p99 = tracker.p99_ms
    assert p99 < 50.0, (
        f"SLO VIOLATION: scheduler warm-path p99={p99:.2f}ms exceeds 50ms threshold. "
        f"p50={tracker.p50_ms:.2f}ms  max={tracker.max_ms:.2f}ms"
    )

    # Write to performance baseline for regression detection (test 7)
    _write_baseline_metric("scheduler_warm_p99_ms", p99)


# ---------------------------------------------------------------------------
# 2. Router decision overhead p99 < 1 ms
# ---------------------------------------------------------------------------


def test_slo_router_decision_p99_under_1ms():
    """
    SLO: 1 000 routing decisions over a 5-instance pool must complete p99 < 1 ms.

    The routing logic is pure Python with no I/O.  Any regression here
    indicates a quadratic algorithm or accidental blocking call.
    """
    router = ModelRouter(strategy=LeastLoadedStrategy())
    instances = [
        InstanceInfo(api_base=f"http://inst-{i}:8000/v1", load=float(i) / 10)
        for i in range(5)
    ]

    samples: list[float] = []
    N = 1_000

    for _ in range(N):
        t0 = time.perf_counter()
        router.choose_instance(instances)
        samples.append((time.perf_counter() - t0) * 1_000)

    samples.sort()
    p99 = samples[int(N * 0.99)]
    p50 = samples[N // 2]

    assert p99 < 1.0, (
        f"SLO VIOLATION: router decision p99={p99:.3f}ms exceeds 1ms threshold. "
        f"p50={p50:.3f}ms"
    )

    _write_baseline_metric("router_decision_p99_ms", p99)


# ---------------------------------------------------------------------------
# 3. Zero error rate under healthy upstream
# ---------------------------------------------------------------------------


async def test_slo_zero_errors_under_healthy_upstream(gateway_client):
    """
    SLO: error rate must be exactly 0% when the upstream is healthy.

    Any non-zero error rate here indicates a bug in the gateway's routing,
    proxy, or metrics layer — not an upstream problem.
    """
    from tests.slo.conftest import (
        MRM_URL, VLLM_API_BASE, MRM_ENSURE_RESPONSE, VLLM_CHAT_RESPONSE, CHAT_REQUEST
    )

    N = 20

    async with respx.MockRouter() as mock:
        mock.post(f"{MRM_URL}/models/ensure").mock(
            return_value=httpx.Response(200, json=MRM_ENSURE_RESPONSE)
        )
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=VLLM_CHAT_RESPONSE)
        )

        responses = [
            await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)
            for _ in range(N)
        ]

    assert all(r.status_code == 200 for r in responses), (
        f"SLO VIOLATION: expected 0% error rate under healthy upstream. "
        f"Got: {[r.status_code for r in responses if r.status_code != 200]}"
    )

    metrics = (await gateway_client.get("/v1/router/metrics")).json()
    instance_m = metrics["instances"].get(VLLM_API_BASE, {})
    assert instance_m.get("errors", 0) == 0, (
        f"SLO VIOLATION: router metrics must record 0 errors. "
        f"Got errors={instance_m.get('errors')}"
    )

    # SLO status must report no violations after a healthy run
    slo = metrics.get("slo", {})
    assert slo.get("slo_ok", True), (
        f"SLO VIOLATION: slo_status reported violations after clean run: "
        f"{slo.get('violations')}"
    )


# ---------------------------------------------------------------------------
# 4. Error rate matches upstream failure rate (±10 pp)
# ---------------------------------------------------------------------------


async def test_slo_error_rate_matches_upstream_failure_rate(gateway_client):
    """
    SLO: the gateway must accurately reflect upstream error rates.

    Under a fixed 25% upstream failure rate, the gateway error rate measured
    via metrics must be within ±10 percentage points of 25%.

    Under-counting errors hides reliability problems.
    Over-counting suggests phantom errors from the proxy layer.
    """
    from tests.slo.conftest import (
        MRM_URL, VLLM_API_BASE, MRM_ENSURE_RESPONSE, VLLM_CHAT_RESPONSE, CHAT_REQUEST
    )

    N = 40
    target_failure_rate = 0.25
    call_count = 0

    async def fixed_failure_rate(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        # Deterministic: every 4th request fails
        if call_count % 4 == 0:
            return httpx.Response(500, json={"detail": "injected failure"})
        return httpx.Response(200, json=VLLM_CHAT_RESPONSE)

    async with respx.MockRouter() as mock:
        mock.post(f"{MRM_URL}/models/ensure").mock(
            return_value=httpx.Response(200, json=MRM_ENSURE_RESPONSE)
        )
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(side_effect=fixed_failure_rate)

        for _ in range(N):
            await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    metrics = (await gateway_client.get("/v1/router/metrics")).json()
    instance_m = metrics["instances"].get(VLLM_API_BASE, {})

    recorded = instance_m.get("requests", 0)
    recorded_errors = instance_m.get("errors", 0)

    assert recorded == N, (
        f"SLO VIOLATION: metrics must record all {N} requests; got {recorded}"
    )

    measured_rate = recorded_errors / recorded
    tolerance = 0.10
    assert abs(measured_rate - target_failure_rate) <= tolerance, (
        f"SLO VIOLATION: measured error rate {measured_rate:.1%} deviates from "
        f"expected {target_failure_rate:.1%} by more than {tolerance:.0%}. "
        f"Errors={recorded_errors}/{recorded}"
    )


# ---------------------------------------------------------------------------
# 5. Inflight counter accuracy — no leak
# ---------------------------------------------------------------------------


@pytest.mark.invariant
async def test_slo_inflight_counter_returns_to_zero_after_requests(gateway_client):
    """
    SLO: after all requests complete, the in-flight counter must be 0.

    A non-zero counter after completion means track_inflight() has a counter
    leak.  Over time this causes false backpressure: the router accumulates
    phantom load and stops routing to the instance.
    """
    from tests.slo.conftest import (
        MRM_URL, VLLM_API_BASE, MRM_ENSURE_RESPONSE, VLLM_CHAT_RESPONSE, CHAT_REQUEST
    )

    N = 15

    async with respx.MockRouter() as mock:
        mock.post(f"{MRM_URL}/models/ensure").mock(
            return_value=httpx.Response(200, json=MRM_ENSURE_RESPONSE)
        )
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=VLLM_CHAT_RESPONSE)
        )

        for _ in range(N):
            await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    metrics = (await gateway_client.get("/v1/router/metrics")).json()
    instance_m = metrics["instances"].get(VLLM_API_BASE, {})
    inflight = instance_m.get("inflight", -1)

    assert inflight == 0, (
        f"SLO VIOLATION: in-flight counter must be 0 after all requests complete; "
        f"got inflight={inflight}.  Counter leak in track_inflight()."
    )


# ---------------------------------------------------------------------------
# 6. SLO status detector correctness
# ---------------------------------------------------------------------------


@pytest.mark.invariant
def test_slo_status_detector_reports_violations():
    """
    SLO: slo_status() must correctly identify instances violating thresholds.

    This tests the detector itself — if the detector is broken, silent
    degradation in production will go unnoticed.
    """
    router = ModelRouter(strategy=LeastLoadedStrategy())

    # Instance with high error rate (> 5%)
    for _ in range(10):
        router.record("http://bad-errors", latency_ms=100.0, error=True)

    # Instance with high latency (> 5 000 ms)
    for _ in range(10):
        router.record("http://bad-latency", latency_ms=6_000.0)

    # Healthy instance
    for _ in range(10):
        router.record("http://healthy", latency_ms=50.0)

    status = router.slo_status()

    assert not status["slo_ok"], "slo_ok must be False when violations exist"

    violation_instances = {v["instance"] for v in status["violations"]}
    assert "http://bad-errors" in violation_instances, (
        "High error-rate instance must appear in violations"
    )
    assert "http://bad-latency" in violation_instances, (
        "High latency instance must appear in violations"
    )
    assert "http://healthy" not in violation_instances, (
        "Healthy instance must NOT appear in violations"
    )


@pytest.mark.invariant
def test_slo_status_ok_under_healthy_conditions():
    """SLO: slo_status() must report no violations for a healthy instance."""
    router = ModelRouter(strategy=LeastLoadedStrategy())
    for _ in range(20):
        router.record("http://good", latency_ms=100.0)

    status = router.slo_status()
    assert status["slo_ok"], (
        f"slo_ok must be True for a healthy instance. Violations: {status['violations']}"
    )
    assert status["violations"] == []


# ---------------------------------------------------------------------------
# 7. Performance regression gate
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_slo_performance_no_regression():
    """
    SLO: current routing p99 must not exceed the stored baseline by more than 3×.

    On first run (no baseline) this test always passes and writes the baseline.
    On subsequent runs it detects catastrophic regressions — e.g., an
    accidental blocking call that inflates latency from 0.01 ms to 50 ms.

    Baseline is stored at test-results/slo_baseline.json.
    """
    router = ModelRouter(strategy=LeastLoadedStrategy())
    instances = [
        InstanceInfo(api_base=f"http://inst-{i}:8000/v1", load=float(i) / 10)
        for i in range(5)
    ]

    N = 2_000
    samples: list[float] = []

    for _ in range(N):
        t0 = time.perf_counter()
        router.choose_instance(instances)
        samples.append((time.perf_counter() - t0) * 1_000)

    samples.sort()
    current_p99 = samples[int(N * 0.99)]

    baseline = _read_baseline()
    baseline_p99 = baseline.get("router_decision_p99_ms")

    if baseline_p99 is not None:
        regression_threshold = baseline_p99 * 3.0
        assert current_p99 <= regression_threshold, (
            f"PERFORMANCE REGRESSION: router decision p99={current_p99:.3f}ms "
            f"exceeds 3× baseline ({baseline_p99:.3f}ms → threshold {regression_threshold:.3f}ms). "
            f"A blocking call or quadratic algorithm may have been introduced."
        )

    # Always update the baseline with the current measurement
    _write_baseline_metric("router_decision_p99_ms", current_p99)


# ---------------------------------------------------------------------------
# 8. Global SLO aggregation via /v1/slo endpoint
# ---------------------------------------------------------------------------


async def test_slo_global_endpoint_aggregates_all_instances(gateway_client):
    """
    SLO: the /v1/slo endpoint must return a fleet-wide latency summary
    that covers all requests sent through the gateway in this session.

    This validates that GlobalSLO is wired into the request path and that
    the aggregated p50/p95/p99 figures are plausible.
    """
    from tests.slo.conftest import (
        MRM_URL, VLLM_API_BASE, MRM_ENSURE_RESPONSE, VLLM_CHAT_RESPONSE, CHAT_REQUEST
    )

    N = 20

    async with respx.MockRouter() as mock:
        mock.post(f"{MRM_URL}/models/ensure").mock(
            return_value=httpx.Response(200, json=MRM_ENSURE_RESPONSE)
        )
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=VLLM_CHAT_RESPONSE)
        )
        for _ in range(N):
            await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    resp = await gateway_client.get("/v1/slo")
    assert resp.status_code == 200

    snap = resp.json()
    assert snap["samples"] == N, (
        f"Global SLO must count all {N} requests; got samples={snap['samples']}"
    )
    assert snap["error_rate"] == 0.0, (
        "Error rate must be 0.0 under healthy upstream"
    )
    assert snap["p50_ms"] is not None, "p50_ms must be present after N requests"
    assert snap["p99_ms"] is not None, "p99_ms must be present after N requests"
    assert snap["p50_ms"] >= 0.0
    assert snap["p99_ms"] >= snap["p50_ms"], "p99 must be >= p50"


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------


def _read_baseline() -> dict:
    """Read the performance baseline JSON, or return empty dict if absent."""
    try:
        return json.loads(BASELINE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_baseline_metric(key: str, value: float) -> None:
    """Upsert a single metric into the baseline JSON file."""
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    baseline = _read_baseline()
    baseline[key] = round(value, 6)
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2))
