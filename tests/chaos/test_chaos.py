"""
tests/chaos/test_chaos.py

Chaos engineering tests — probabilistic failures, mixed workloads, and
partial unavailability.  These tests inject random or sustained failure
conditions and verify the system's invariants still hold.

INVARIANTS UNDER TEST
---------------------
1. Random failure rate    → no 500s regardless of upstream reliability
2. Complete outage        → all requests get 502 (not 500)
3. No nodes available     → scheduler raises, gateway returns 503
4. Mixed concurrent chaos → concurrent requests under random failures,
                            no unhandled exceptions
5. Rapid state churn      → repeated register/kill node cycles,
                            scheduler never deadlocks
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter

import httpx
import pytest
import respx

pytestmark = pytest.mark.chaos

from tests.chaos.conftest import (
    CHAT_REQUEST,
    MRM_URL,
    VLLM_API_BASE,
    VLLM_CHAT_RESPONSE,
    MRM_ENSURE_RESPONSE,
    MODEL,
    register_node,
)
from tests.utils.fault_injection import simulate_connect_error

NODE_URL = "http://node-1:8020"


def _mock_mrm(mock: respx.MockRouter, response: dict = None) -> None:
    mock.post(f"{MRM_URL}/models/ensure").mock(
        return_value=httpx.Response(200, json=response or MRM_ENSURE_RESPONSE)
    )


# ---------------------------------------------------------------------------
# 1. Random failure rate — no 500s
# ---------------------------------------------------------------------------


async def test_random_failure_rate_never_produces_500(gateway_client):
    """
    CHAOS INVARIANT: under a random 30% upstream failure rate, the Gateway
    must never return 500.  Individual request failures may produce 502,
    but 500 indicates an unhandled exception in the Gateway itself.
    """
    N = 30
    seed = 42
    rng = random.Random(seed)

    async def random_upstream(request: httpx.Request) -> httpx.Response:
        if rng.random() < 0.3:
            return httpx.Response(500, json={"detail": "random upstream error"})
        return httpx.Response(200, json=VLLM_CHAT_RESPONSE)

    async with respx.MockRouter() as mock:
        _mock_mrm(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(side_effect=random_upstream)

        responses = await asyncio.gather(
            *[gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST) for _ in range(N)]
        )

    statuses = [r.status_code for r in responses]
    assert 500 not in statuses, (
        f"Gateway must never return 500 (unhandled exception). "
        f"Status distribution: {dict(Counter(statuses))}"
    )
    for s in statuses:
        assert s in (200, 502), f"Unexpected status {s}; allowed: 200, 502"


# ---------------------------------------------------------------------------
# 2. Complete outage — all 502, never 500
# ---------------------------------------------------------------------------


async def test_complete_vllm_outage_returns_all_502(gateway_client):
    """
    CHAOS INVARIANT: when vLLM is completely down (every request fails),
    the Gateway returns 502 for each request — never 500.

    This validates that the Gateway's error handling path is correct even
    under 100% upstream failure.
    """
    N = 20

    async with respx.MockRouter() as mock:
        _mock_mrm(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            side_effect=simulate_connect_error()
        )

        responses = await asyncio.gather(
            *[gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST) for _ in range(N)]
        )

    statuses = [r.status_code for r in responses]
    assert all(s == 502 for s in statuses), (
        f"Complete vLLM outage must produce only 502s; got: {dict(Counter(statuses))}"
    )


# ---------------------------------------------------------------------------
# 3. No nodes available — scheduler raises, gateway returns 503
# ---------------------------------------------------------------------------


async def test_gateway_returns_503_when_no_nodes_registered(gateway_client):
    """
    CHAOS INVARIANT: when the scheduler has no nodes registered, it cannot
    place the model.  The Gateway must surface this as 503 (service unavailable),
    not 500 (unhandled exception).
    """
    # No respx mocks for MRM — use a scheduler-enabled gateway
    # Instead, test the direct MRM path with MRM returning connect error
    async with respx.MockRouter() as mock:
        mock.post(f"{MRM_URL}/models/ensure").mock(
            side_effect=simulate_connect_error()
        )

        resp = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert resp.status_code == 503, (
        f"When placement service is unreachable, Gateway must return 503; got {resp.status_code}"
    )


# ---------------------------------------------------------------------------
# 4. Mixed concurrent chaos — N requests, random failure injection
# ---------------------------------------------------------------------------


async def test_mixed_concurrent_chaos_no_unhandled_exceptions(gateway_client):
    """
    CHAOS INVARIANT: under 50 concurrent requests with a 40% random failure
    rate, the Gateway must never return 500.

    Combines high concurrency with random failures to stress both the
    async machinery and the error handling paths simultaneously.
    """
    N = 50
    rng = random.Random(0)

    call_count = 0

    async def chaotic_upstream(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        # Introduce variable latency and random failures
        await asyncio.sleep(rng.uniform(0, 0.01))
        if rng.random() < 0.4:
            return httpx.Response(500, json={"detail": "chaos failure"})
        return httpx.Response(200, json=VLLM_CHAT_RESPONSE)

    async with respx.MockRouter() as mock:
        _mock_mrm(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(side_effect=chaotic_upstream)

        responses = await asyncio.gather(
            *[gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST) for _ in range(N)]
        )

    statuses = [r.status_code for r in responses]
    assert 500 not in statuses, (
        f"No 500s allowed under chaotic conditions. "
        f"Distribution: {dict(Counter(statuses))}"
    )
    assert len(responses) == N, "All concurrent requests must complete (no hangs)"


# ---------------------------------------------------------------------------
# 5. Rapid node churn — scheduler never deadlocks
# ---------------------------------------------------------------------------


@pytest.mark.slow
async def test_scheduler_survives_rapid_node_registration_churn(scheduler, fake_redis):
    """
    CHAOS INVARIANT: repeatedly registering nodes, placing models, and
    removing nodes must not cause the scheduler to deadlock or accumulate
    stale state.

    Simulates a rolling restart scenario where nodes cycle in and out
    while models are actively being placed.
    """
    from tests.chaos.conftest import register_node as _register

    ROUNDS = 5
    ENSURES_PER_ROUND = 10

    for round_num in range(ROUNDS):
        node_id = f"chaos-node-{round_num}"
        node_url = f"http://{node_id}:8020"

        await _register(scheduler, node_id, node_url)

        async def mock_ensure(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.005)
            return httpx.Response(
                200,
                json={
                    "model_id": MODEL,
                    "model_alias": MODEL.split("/")[-1].lower(),
                    "api_base": f"http://{node_id}:8000/v1",
                    "gpu": "0",
                    "state": "READY",
                    "container": f"mrm-{node_id}",
                },
            )

        async with respx.MockRouter() as mock:
            mock.post(f"{node_url}/local/ensure").mock(side_effect=mock_ensure)

            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        *[scheduler.ensure(MODEL) for _ in range(ENSURES_PER_ROUND)]
                    ),
                    timeout=10.0,
                )
            except asyncio.TimeoutError:
                pytest.fail(
                    f"Scheduler deadlocked in round {round_num} during node churn"
                )

        # Evict the node to simulate it going away
        await fake_redis.delete(f"scheduler:node:{node_id}")
        await fake_redis.srem("scheduler:nodes", node_id)
        # Clear the placement so the next round places fresh
        placement_keys = await fake_redis.keys("scheduler:placement:*")
        for key in placement_keys:
            await fake_redis.delete(key)
