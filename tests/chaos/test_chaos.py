"""
tests/chaos/test_chaos.py

Chaos engineering tests — probabilistic failures, mixed workloads, and
partial unavailability.  These tests inject random or sustained failure
conditions and verify the system's invariants still hold.

INVARIANTS UNDER TEST
---------------------
1. Random failure rate    → no 500s; error rate in metrics matches injection
2. Complete outage        → all 502 with proper error body structure
3. No nodes available     → 503 (not 500 or 200)
4. Mixed concurrent chaos → no 500s, no hangs, response bodies not corrupted
5. Rapid state churn      → scheduler never deadlocks, placement count stable
6. Metrics no silent drops → every request is counted, no phantom errors
7. Node eviction invariant → placement count never exceeds model count
"""

from __future__ import annotations

import asyncio
import json
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
# 1. Random failure rate — no 500s + error rate accuracy
# ---------------------------------------------------------------------------


async def test_random_failure_rate_never_produces_500(gateway_client):
    """
    CHAOS INVARIANT: under a random 30% upstream failure rate:
    - The Gateway must never return 500 (unhandled exception)
    - The error rate recorded in metrics must be within ±15pp of the injection rate
    """
    N = 30
    seed = 42
    rng = random.Random(seed)
    injected_errors = 0

    async def random_upstream(request: httpx.Request) -> httpx.Response:
        nonlocal injected_errors
        if rng.random() < 0.3:
            injected_errors += 1
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

    # Verify error rate tracking accuracy
    metrics = (await gateway_client.get("/v1/router/metrics")).json()
    instance_m = metrics["instances"].get(VLLM_API_BASE, {})
    recorded_errors = instance_m.get("errors", 0)
    recorded_requests = instance_m.get("requests", 0)

    assert recorded_requests == N, (
        f"Metrics must record all {N} requests; got {recorded_requests}"
    )
    assert recorded_errors == injected_errors, (
        f"Metrics errors ({recorded_errors}) must match injected errors ({injected_errors})"
    )


# ---------------------------------------------------------------------------
# 2. Complete outage — all 502 with proper error body
# ---------------------------------------------------------------------------


async def test_complete_vllm_outage_returns_all_502(gateway_client):
    """
    CHAOS INVARIANT: when vLLM is completely down:
    - Every response must be 502 (not 500 or 200)
    - Every 502 response must contain a structured error body with 'detail'
    - Metrics must record all failures
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

    # Every error response must have a structured body (not empty)
    for r in responses:
        body = r.json()
        assert "detail" in body, (
            f"502 response must contain 'detail' field; got: {body!r}"
        )

    # Metrics must record all failures — no silent drops
    metrics = (await gateway_client.get("/v1/router/metrics")).json()
    instance_m = metrics["instances"].get(VLLM_API_BASE, {})
    assert instance_m.get("requests", 0) == N, (
        f"All {N} failures must be recorded in metrics"
    )
    assert instance_m.get("errors", 0) == N, (
        f"All {N} requests must be recorded as errors"
    )


# ---------------------------------------------------------------------------
# 3. No nodes available — scheduler raises, gateway returns 503
# ---------------------------------------------------------------------------


async def test_gateway_returns_503_when_no_nodes_registered(gateway_client):
    """
    CHAOS INVARIANT: when the placement service is unreachable:
    - Response must be 503 (not 500 or 502)
    - Response must have a structured error body
    """
    async with respx.MockRouter() as mock:
        mock.post(f"{MRM_URL}/models/ensure").mock(
            side_effect=simulate_connect_error()
        )

        resp = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert resp.status_code == 503, (
        f"When placement service is unreachable, Gateway must return 503; got {resp.status_code}"
    )
    body = resp.json()
    assert "detail" in body, f"503 response must contain 'detail' field; got: {body!r}"


# ---------------------------------------------------------------------------
# 4. Mixed concurrent chaos — correctness under load + failure
# ---------------------------------------------------------------------------


async def test_mixed_concurrent_chaos_no_unhandled_exceptions(gateway_client):
    """
    CHAOS INVARIANT: under 50 concurrent requests with 40% random failure:
    - No 500s (unhandled exceptions)
    - All responses complete (no hangs)
    - Response bodies are structurally correct (200→choices, 502→detail)
    """
    N = 50
    rng = random.Random(0)

    async def chaotic_upstream(request: httpx.Request) -> httpx.Response:
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

    assert len(responses) == N, "All concurrent requests must complete (no hangs)"

    statuses = [r.status_code for r in responses]
    assert 500 not in statuses, (
        f"No 500s allowed under chaotic conditions. "
        f"Distribution: {dict(Counter(statuses))}"
    )

    # Response body integrity: every response must have the correct structure
    for r in responses:
        body = r.json()
        if r.status_code == 200:
            assert "choices" in body, (
                f"Successful response missing 'choices' field: {body!r}"
            )
        elif r.status_code == 502:
            assert "detail" in body, (
                f"Error response missing 'detail' field: {body!r}"
            )


# ---------------------------------------------------------------------------
# 5. Rapid node churn — scheduler never deadlocks + placement count stable
# ---------------------------------------------------------------------------


@pytest.mark.slow
async def test_scheduler_survives_rapid_node_registration_churn(scheduler, fake_redis):
    """
    CHAOS INVARIANT: repeated node register/kill cycles must not cause:
    - Scheduler deadlock
    - Placement count > 1 (no duplicate records)
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

        # INVARIANT: exactly one placement record must exist per model
        placement_keys = await fake_redis.keys("scheduler:placement:*")
        assert len(placement_keys) == 1, (
            f"Round {round_num}: expected exactly 1 placement record; "
            f"got {len(placement_keys)}"
        )

        # Evict the node and clear placement for next round
        await fake_redis.delete(f"scheduler:node:{node_id}")
        await fake_redis.srem("scheduler:nodes", node_id)
        for key in placement_keys:
            await fake_redis.delete(key)


# ---------------------------------------------------------------------------
# 6. Metrics no silent drops — every request is counted
# ---------------------------------------------------------------------------


async def test_chaos_metrics_no_silent_drops(gateway_client):
    """
    CHAOS INVARIANT: under partial failures, the router's metrics must
    reflect every request — successes and errors.  No request may be silently
    dropped from the count.

    This validates the completeness of the observability layer under chaos.
    """
    N = 30
    injected_errors = 0
    call_counter = 0

    async def partial_failure(request: httpx.Request) -> httpx.Response:
        nonlocal injected_errors, call_counter
        call_counter += 1
        # Deterministic 1-in-3 failure pattern
        if call_counter % 3 == 0:
            injected_errors += 1
            return httpx.Response(500, json={"detail": "injected"})
        return httpx.Response(200, json=VLLM_CHAT_RESPONSE)

    async with respx.MockRouter() as mock:
        _mock_mrm(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(side_effect=partial_failure)

        # Sequential to keep call_counter deterministic
        for _ in range(N):
            await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    metrics = (await gateway_client.get("/v1/router/metrics")).json()
    instance_m = metrics["instances"].get(VLLM_API_BASE, {})

    assert instance_m.get("requests", 0) == N, (
        f"CHAOS INVARIANT: all {N} requests must be counted; "
        f"got {instance_m.get('requests', 0)}"
    )
    assert instance_m.get("errors", 0) == injected_errors, (
        f"CHAOS INVARIANT: errors must match injected count {injected_errors}; "
        f"got {instance_m.get('errors', 0)}"
    )


# ---------------------------------------------------------------------------
# 7. Node eviction invariant — placement count bounded by model count
# ---------------------------------------------------------------------------


async def test_chaos_node_eviction_preserves_placement_bound(scheduler, fake_redis):
    """
    CHAOS INVARIANT: even under rapid node eviction and re-registration,
    the number of placement records must never exceed the number of distinct
    models being placed.

    Placement accumulation would cause the scheduler to attempt to use
    dead instances, causing silent failures that bypasses the failover logic.
    """
    MODELS = [
        "meta-llama/Llama-2-7b-hf",
        "Qwen/Qwen2-7B",
    ]
    ROUNDS = 3

    for round_num in range(ROUNDS):
        node_id = f"evict-node-{round_num}"
        node_url = f"http://{node_id}:8020"
        await register_node(scheduler, node_id, node_url)

        async def mock_ensure(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            m = body.get("model", MODELS[0])
            return httpx.Response(
                200,
                json={
                    "model_id": m,
                    "model_alias": m.split("/")[-1].lower(),
                    "api_base": f"http://{node_id}:8000/v1",
                    "gpu": "0",
                    "state": "READY",
                    "container": f"mrm-{node_id}",
                },
            )

        async with respx.MockRouter() as mock:
            mock.post(f"{node_url}/local/ensure").mock(side_effect=mock_ensure)
            for m in MODELS:
                await scheduler.ensure(m)

        placement_keys = await fake_redis.keys("scheduler:placement:*")
        assert len(placement_keys) <= len(MODELS), (
            f"Round {round_num}: placement count {len(placement_keys)} exceeds "
            f"model count {len(MODELS)} — accumulation detected"
        )

        # Evict node, keep placements (they'll be invalidated on next ensure)
        await fake_redis.delete(f"scheduler:node:{node_id}")
        await fake_redis.srem("scheduler:nodes", node_id)
