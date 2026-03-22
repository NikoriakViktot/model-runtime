"""
tests/resilience/test_resilience.py

Recovery and stability tests — validate system behavior during and after
failure events such as node loss, placement invalidation, and error storms.

INVARIANTS UNDER TEST
---------------------
1. Node loss during inference  → Gateway returns error for that request;
                                  next request triggers re-placement on a live node.
2. Scheduler re-places model   → after a dead node is evicted, ensure() succeeds
                                  on a different node.
3. Error metrics update        → Gateway records errors in per-instance metrics
                                  visible via GET /v1/router/metrics.
4. Load-based routing          → LeastLoaded strategy routes away from busy instances.
5. System stable after errors  → N failures followed by recovery → stable state.
6. Placement survives node loss → New ensure() after node death goes to survivor.

DOCUMENTED LIMITATIONS (xfail)
------------------------------
These tests document behaviors that are NOT yet implemented but are important
for production resilience.  They are marked xfail so CI tracks them without
blocking release.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

pytestmark = pytest.mark.resilience

from gateway.services.router import InstanceInfo, LeastLoadedStrategy, ModelRouter
from tests.resilience.conftest import (
    CHAT_REQUEST,
    MRM_URL,
    VLLM_API_BASE,
    VLLM_CHAT_RESPONSE,
    MRM_ENSURE_RESPONSE,
    MODEL,
    register_node,
    node_ensure_payload,
)
from tests.utils.fault_injection import (
    MidStreamFailure,
    simulate_failure,
)

# Re-import SSE_CHUNKS from conftest constants
from tests.resilience.conftest import SSE_CHUNKS

NODE_1 = "http://node-1:8020"
NODE_2 = "http://node-2:8020"


def _mock_mrm_ensure(mock: respx.MockRouter, response: dict = None) -> None:
    mock.post(f"{MRM_URL}/models/ensure").mock(
        return_value=httpx.Response(200, json=response or MRM_ENSURE_RESPONSE)
    )


# ---------------------------------------------------------------------------
# 1. Gateway is healthy after a failed request
# ---------------------------------------------------------------------------


async def test_gateway_stable_after_vllm_crash(gateway_client):
    """
    INVARIANT: after vLLM crashes (connection refused), the Gateway returns
    one 502 for the failing request and then serves the next request normally.

    A single instance failure must not poison the Gateway process.
    """
    call_count = 0

    async def crash_then_recover(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ConnectError("Connection refused", request=request)
        return httpx.Response(200, json=VLLM_CHAT_RESPONSE)

    async with respx.MockRouter() as mock:
        _mock_mrm_ensure(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(side_effect=crash_then_recover)

        r1 = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)
        assert r1.status_code == 502

        r2 = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)
        assert r2.status_code == 200, (
            "Gateway must serve requests normally after recovering from a vLLM crash"
        )


# ---------------------------------------------------------------------------
# 2. Scheduler re-places after node death
# ---------------------------------------------------------------------------


async def test_scheduler_re_places_after_node_death(scheduler, fake_redis):
    """
    INVARIANT: when the node holding a model placement dies (TTL expires),
    the next ensure() call places the model on a different live node.

    This is the distributed failover invariant: the cluster self-heals
    without operator intervention.
    """
    # Set up two nodes
    await register_node(scheduler, "node-1", NODE_1, free_mb=20_000)

    async with respx.MockRouter() as mock:
        mock.post(f"{NODE_1}/local/ensure").mock(
            return_value=httpx.Response(
                200, json=node_ensure_payload(api_base="http://node-1:8000/v1")
            )
        )
        r1 = await scheduler.ensure(MODEL)

    assert r1.instances[0].node_id == "node-1"

    # Kill node-1 (simulate TTL expiry)
    await fake_redis.delete(f"scheduler:node:node-1")
    await fake_redis.srem("scheduler:nodes", "node-1")

    # Register node-2 as the survivor
    await register_node(scheduler, "node-2", NODE_2, free_mb=20_000)

    async with respx.MockRouter() as mock:
        mock.post(f"{NODE_2}/local/ensure").mock(
            return_value=httpx.Response(
                200, json=node_ensure_payload(api_base="http://node-2:8000/v1")
            )
        )
        r2 = await scheduler.ensure(MODEL)

    assert r2.instances[0].node_id == "node-2", (
        "Scheduler must place the model on node-2 after node-1 dies"
    )
    assert r2.api_base != r1.api_base, (
        "The api_base must change after failover to a different node"
    )


# ---------------------------------------------------------------------------
# 3. Error metrics are recorded
# ---------------------------------------------------------------------------


async def test_gateway_records_error_metrics_after_vllm_failure(gateway_client):
    """
    INVARIANT: each vLLM failure increments the error counter in the
    Gateway's per-instance metrics.

    Operators monitor GET /v1/router/metrics to detect degraded instances
    and trigger manual intervention or automated replacement.
    """
    async with respx.MockRouter() as mock:
        _mock_mrm_ensure(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            side_effect=simulate_failure(500, "GPU OOM")
        )

        for _ in range(3):
            await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    metrics_resp = await gateway_client.get("/v1/router/metrics")
    assert metrics_resp.status_code == 200
    metrics = metrics_resp.json()

    instance_metrics = metrics.get("instances", {}).get(VLLM_API_BASE, {})
    assert instance_metrics.get("requests", 0) >= 3, (
        "All 3 requests must be recorded in metrics"
    )
    assert instance_metrics.get("errors", 0) >= 3, (
        "All 3 errors must be recorded in metrics"
    )


async def test_error_rate_reflects_upstream_reliability(gateway_client):
    """
    INVARIANT: the error rate visible in metrics accurately reflects the
    actual upstream failure rate.  This is a fundamental observability guarantee.
    """
    total_requests = 10
    expected_errors = 5

    call_count = 0

    async def alternating(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count % 2 == 0:
            return httpx.Response(500, json={"detail": "error"})
        return httpx.Response(200, json=VLLM_CHAT_RESPONSE)

    async with respx.MockRouter() as mock:
        _mock_mrm_ensure(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(side_effect=alternating)

        for _ in range(total_requests):
            await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    metrics = (await gateway_client.get("/v1/router/metrics")).json()
    instance_metrics = metrics.get("instances", {}).get(VLLM_API_BASE, {})

    recorded_requests = instance_metrics.get("requests", 0)
    recorded_errors = instance_metrics.get("errors", 0)

    assert recorded_requests == total_requests, (
        f"Expected {total_requests} recorded requests, got {recorded_requests}"
    )
    assert recorded_errors == expected_errors, (
        f"Expected {expected_errors} errors recorded, got {recorded_errors}"
    )


# ---------------------------------------------------------------------------
# 4. Load-based routing
# ---------------------------------------------------------------------------


@pytest.mark.invariant
def test_least_loaded_routes_all_traffic_away_from_saturated_instance():
    """
    INVARIANT: when one instance reports load=1.0 (fully saturated) and
    another reports load=0.0 (idle), ALL traffic must go to the idle instance.

    This is the fundamental correctness guarantee of the LeastLoaded strategy.
    """
    router = ModelRouter(strategy=LeastLoadedStrategy())
    instances = [
        InstanceInfo(api_base="http://busy", load=1.0),
        InstanceInfo(api_base="http://idle", load=0.0),
    ]

    chosen = [router.choose_instance(instances).api_base for _ in range(50)]
    assert all(c == "http://idle" for c in chosen), (
        "LeastLoaded must never route to a saturated (load=1.0) instance "
        "when an idle (load=0.0) instance is available"
    )


@pytest.mark.invariant
def test_routing_distribution_proportional_to_load_gap():
    """
    INVARIANT: the routing strategy consistently avoids the high-load instance.
    With load=0.9 vs load=0.1, the low-load instance should receive all traffic.
    """
    router = ModelRouter(strategy=LeastLoadedStrategy())
    instances = [
        InstanceInfo(api_base="http://heavy", load=0.9),
        InstanceInfo(api_base="http://light", load=0.1),
    ]

    N = 100
    chosen = [router.choose_instance(instances).api_base for _ in range(N)]
    heavy_count = chosen.count("http://heavy")
    light_count = chosen.count("http://light")

    assert light_count == N, (
        f"LeastLoaded must route all {N} requests to the lighter instance. "
        f"Got: heavy={heavy_count}, light={light_count}"
    )


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Latency-based routing not implemented. "
        "Currently, the router uses the static 'load' field from MRM, not observed latency. "
        "To fix: implement a LatencyAwareStrategy that reads from model_router.get_metrics() "
        "and feeds back into instance selection."
    ),
)
def test_router_deprioritises_consistently_slow_instance():
    """
    FUTURE INVARIANT: after recording high latency for an instance, the
    router should select it less often than a fast instance.

    STATUS: Not implemented. The router ignores recorded latency when
    selecting instances. See gateway/services/router.py.
    """
    router = ModelRouter(strategy=LeastLoadedStrategy())
    instances = [
        InstanceInfo(api_base="http://slow", load=0.0),
        InstanceInfo(api_base="http://fast", load=0.0),
    ]

    # Record high latency for http://slow
    for _ in range(20):
        router.record("http://slow", latency_ms=5_000.0)
        router.record("http://fast", latency_ms=50.0)

    # After recording, the router should prefer http://fast
    N = 100
    chosen = [router.choose_instance(instances).api_base for _ in range(N)]
    slow_count = chosen.count("http://slow")

    # This assertion will fail until latency-aware routing is implemented
    assert slow_count < N * 0.2, (
        f"Slow instance received {slow_count}/{N} requests — "
        f"latency-aware routing should reduce this below 20%"
    )


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Backpressure not implemented. "
        "Currently the router has no concept of in-flight request count. "
        "To fix: track active requests per instance and deprioritise overloaded ones."
    ),
)
async def test_backpressure_deprioritises_instance_with_many_inflight_requests():
    """
    FUTURE INVARIANT: an instance with many in-flight requests should receive
    fewer new requests than a less-busy instance.

    STATUS: Not implemented. The router does not track in-flight request count.
    """
    router = ModelRouter(strategy=LeastLoadedStrategy())
    instances = [
        InstanceInfo(api_base="http://busy", load=0.0),  # load=0.0 but many inflight
        InstanceInfo(api_base="http://free", load=0.0),
    ]

    # Simulate 100 in-flight requests on http://busy (not yet implemented)
    # This would require a mechanism to track inflight count
    raise NotImplementedError("Backpressure tracking not implemented")


# ---------------------------------------------------------------------------
# 5. System stability after error storm
# ---------------------------------------------------------------------------


async def test_system_stable_after_error_storm_and_recovery(gateway_client):
    """
    INVARIANT: after a storm of 20 consecutive errors followed by recovery,
    the Gateway serves requests correctly and metrics reflect the full history.
    """
    total = 25
    failure_count = 20
    effect = simulate_n_failures_then_recover(VLLM_CHAT_RESPONSE, fail_count=failure_count)

    statuses: list[int] = []

    async with respx.MockRouter() as mock:
        _mock_mrm_ensure(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(side_effect=effect)

        for _ in range(total):
            r = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)
            statuses.append(r.status_code)

    failure_statuses = statuses[:failure_count]
    recovery_statuses = statuses[failure_count:]

    assert all(s == 502 for s in failure_statuses), (
        "All requests during the error storm must return 502"
    )
    assert all(s == 200 for s in recovery_statuses), (
        "All requests after recovery must return 200"
    )

    # Metrics must capture the full history
    metrics = (await gateway_client.get("/v1/router/metrics")).json()
    instance_m = metrics.get("instances", {}).get(VLLM_API_BASE, {})
    assert instance_m.get("requests", 0) == total
    assert instance_m.get("errors", 0) == failure_count


# ---------------------------------------------------------------------------
# 6. Concurrent placement stability during node oscillation
# ---------------------------------------------------------------------------


async def test_placement_stable_during_node_registration_churn(scheduler, fake_redis):
    """
    INVARIANT: even while nodes are rapidly joining and leaving the cluster,
    a model that is already placed must keep its placement on a live node.

    Simulates a node rolling restart where a new node comes up before the
    old one finishes going down.
    """
    await register_node(scheduler, "node-stable", "http://stable:8020", free_mb=20_000)
    await register_node(scheduler, "node-temp", "http://temp:8020", free_mb=16_000)

    async with respx.MockRouter(assert_all_called=False) as mock:
        mock.post("http://stable:8020/local/ensure").mock(
            return_value=httpx.Response(
                200, json=node_ensure_payload(api_base="http://stable:8000/v1")
            )
        )
        mock.post("http://temp:8020/local/ensure").mock(
            return_value=httpx.Response(
                200, json=node_ensure_payload(api_base="http://temp:8000/v1")
            )
        )

        initial = await scheduler.ensure(MODEL)

    placed_node = initial.instances[0].node_id

    # Kill the non-placement node (temp)
    if placed_node == "node-stable":
        await fake_redis.delete("scheduler:node:node-temp")
        await fake_redis.srem("scheduler:nodes", "node-temp")
    else:
        await fake_redis.delete("scheduler:node:node-stable")
        await fake_redis.srem("scheduler:nodes", "node-stable")

    # Re-ensure should still work (the placed node is still alive)
    async with respx.MockRouter(assert_all_called=False) as mock:
        mock.post("http://stable:8020/local/ensure").mock(
            return_value=httpx.Response(
                200, json=node_ensure_payload(api_base="http://stable:8000/v1")
            )
        )
        mock.post("http://temp:8020/local/ensure").mock(
            return_value=httpx.Response(
                200, json=node_ensure_payload(api_base="http://temp:8000/v1")
            )
        )
        follow_up = await scheduler.ensure(MODEL)

    assert follow_up.api_base == initial.api_base, (
        "Placement must remain stable when the placed node is still alive"
    )


# ---------------------------------------------------------------------------
# Import fix — re-export SSE_CHUNKS properly
# ---------------------------------------------------------------------------
# (Re-import at top of module via conftest constants)
from tests.utils.fault_injection import simulate_n_failures_then_recover  # noqa: F401, E402
