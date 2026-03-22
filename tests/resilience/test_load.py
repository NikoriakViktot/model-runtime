"""
tests/resilience/test_load.py

Load and concurrency tests — validate system behavior under sustained
and burst concurrent load.

These are asyncio concurrency tests, not network load tests.  They stress
the Python event loop, asyncio locks, and per-instance metrics — not raw
throughput.  For HTTP throughput benchmarking, use locust or k6.

INVARIANTS UNDER TEST
---------------------
1. 100 concurrent ensures    → exactly 1 Node Agent call, 1 placement
2. 200 concurrent ensures    → still exactly 1 placement (high-stress lock test)
3. 50 concurrent requests    → all complete, no 500s (no unhandled exceptions)
4. Response isolation        → concurrent responses are not mixed up
5. Multi-model concurrency   → N models × M concurrent ensures → N placements
6. Latency profile           → p99 within an acceptable bound under modest load
7. No deadlock               → ensure() never hangs indefinitely
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter

import httpx
import pytest
import respx

pytestmark = pytest.mark.concurrency

from tests.resilience.conftest import (
    CHAT_REQUEST,
    MRM_URL,
    MODEL,
    VLLM_API_BASE,
    VLLM_CHAT_RESPONSE,
    MRM_ENSURE_RESPONSE,
    MODEL_ALIAS,
    register_node,
    node_ensure_payload,
)
from tests.utils.fault_injection import ConcurrencyCounter, LatencyTracker

NODE_URL = "http://node-1:8020"

# Separate models for multi-model concurrency tests
MODEL_A = "meta-llama/Llama-2-7b-hf"
MODEL_B = "Qwen/Qwen2-7B"
MODEL_C = "mistralai/Mistral-7B-v0.1"


def _mock_mrm(mock: respx.MockRouter, response: dict = None) -> None:
    mock.post(f"{MRM_URL}/models/ensure").mock(
        return_value=httpx.Response(200, json=response or MRM_ENSURE_RESPONSE)
    )


# ---------------------------------------------------------------------------
# 1. 100 concurrent ensures → single placement, single Node Agent call
# ---------------------------------------------------------------------------


async def test_100_concurrent_ensures_produce_single_placement(scheduler, fake_redis):
    """
    INVARIANT: 100 concurrent ensure() calls for the same model must result in
    exactly one placement in Redis and exactly one Node Agent invocation.

    Validates the per-model asyncio.Lock in the Scheduler.
    """
    await register_node(scheduler, "node-1", NODE_URL)
    counter = ConcurrencyCounter()

    async def mock_ensure(request: httpx.Request) -> httpx.Response:
        async with counter:
            await asyncio.sleep(0.05)  # simulate cold-start
        return httpx.Response(200, json=node_ensure_payload())

    N = 100
    async with respx.MockRouter() as mock:
        mock.post(f"{NODE_URL}/local/ensure").mock(side_effect=mock_ensure)
        results = await asyncio.gather(*[scheduler.ensure(MODEL) for _ in range(N)])

    api_bases = {r.api_base for r in results}
    assert len(api_bases) == 1, (
        f"All {N} ensure() calls must return the same api_base; got {len(api_bases)} distinct values"
    )

    placement_keys = await fake_redis.keys("scheduler:placement:*")
    assert len(placement_keys) == 1, (
        f"Exactly one placement must exist in Redis; got {len(placement_keys)}"
    )

    assert counter.total == 1, (
        f"Node Agent must be called exactly once; got {counter.total} calls"
    )
    assert counter.peak == 1, (
        f"Peak concurrency into Node Agent must be 1 (lock prevents parallelism); "
        f"got peak={counter.peak}"
    )


# ---------------------------------------------------------------------------
# 2. 200 concurrent ensures — high-stress lock test
# ---------------------------------------------------------------------------


@pytest.mark.slow
async def test_200_concurrent_ensures_produce_single_placement(scheduler, fake_redis):
    """
    INVARIANT: the per-model lock must hold under extreme concurrency.

    200 coroutines, each pausing inside the mock to maximise lock contention.
    This catches race conditions that only manifest at higher concurrency.
    """
    await register_node(scheduler, "node-1", NODE_URL)
    node_agent_call_count = 0

    async def mock_ensure(request: httpx.Request) -> httpx.Response:
        nonlocal node_agent_call_count
        await asyncio.sleep(0.01)  # yield to maximise contention
        node_agent_call_count += 1
        return httpx.Response(200, json=node_ensure_payload())

    N = 200
    async with respx.MockRouter() as mock:
        mock.post(f"{NODE_URL}/local/ensure").mock(side_effect=mock_ensure)
        results = await asyncio.gather(*[scheduler.ensure(MODEL) for _ in range(N)])

    assert node_agent_call_count == 1, (
        f"Expected exactly 1 Node Agent call regardless of concurrency; got {node_agent_call_count}"
    )

    api_bases = {r.api_base for r in results}
    assert len(api_bases) == 1, (
        f"All {N} results must have the same api_base; got {len(api_bases)}"
    )


# ---------------------------------------------------------------------------
# 3. 50 concurrent gateway requests — no 500s
# ---------------------------------------------------------------------------


async def test_50_concurrent_gateway_requests_produce_no_500s(gateway_client):
    """
    INVARIANT: under 50 concurrent requests, the Gateway must never return
    500 (Internal Server Error).

    500 indicates an unhandled exception in the Gateway itself.  Individual
    upstream errors may produce 502/503, but those are expected and acceptable.
    """
    N = 50

    async with respx.MockRouter() as mock:
        _mock_mrm(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=VLLM_CHAT_RESPONSE)
        )

        responses = await asyncio.gather(
            *[gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST) for _ in range(N)]
        )

    statuses = [r.status_code for r in responses]
    status_counts = Counter(statuses)

    assert status_counts.get(500, 0) == 0, (
        f"No 500s allowed under concurrent load. Status distribution: {dict(status_counts)}"
    )
    assert status_counts.get(200, 0) == N, (
        f"All {N} concurrent requests must succeed. Status distribution: {dict(status_counts)}"
    )


# ---------------------------------------------------------------------------
# 4. Response isolation — concurrent responses must not mix up
# ---------------------------------------------------------------------------


async def test_concurrent_responses_are_not_mixed_up(gateway_client):
    """
    INVARIANT: each concurrent request receives its own response body.
    Response content must not be shared or mixed between concurrent requests.

    This catches potential buffer sharing bugs in the proxy layer.
    """
    N = 30

    async def echo_response(request: httpx.Request) -> httpx.Response:
        """Return a response that echoes the unique request ID back."""
        body = json.loads(request.content)
        # Extract the unique tag embedded in the message content
        tag = body["messages"][0]["content"]
        return httpx.Response(
            200,
            json={
                "id": f"cmpl-{tag}",
                "object": "chat.completion",
                "model": MODEL_ALIAS,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"echo:{tag}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    async def make_request(idx: int) -> tuple[int, str]:
        tag = f"req-{idx:04d}"
        resp = await gateway_client.post(
            "/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": tag}]},
        )
        assert resp.status_code == 200
        body = resp.json()
        return idx, body["choices"][0]["message"]["content"]

    async with respx.MockRouter() as mock:
        _mock_mrm(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(side_effect=echo_response)

        results: list[tuple[int, str]] = await asyncio.gather(
            *[make_request(i) for i in range(N)]
        )

    for idx, content in results:
        expected_tag = f"req-{idx:04d}"
        assert expected_tag in content, (
            f"Request {idx} received wrong content: '{content}'. "
            f"Expected content echoing tag '{expected_tag}'. "
            f"This indicates response body mixing under concurrency."
        )


# ---------------------------------------------------------------------------
# 5. Multi-model concurrency — N models × M concurrent ensures
# ---------------------------------------------------------------------------


async def test_multi_model_concurrency_produces_independent_placements(
    scheduler, fake_redis
):
    """
    INVARIANT: concurrent ensures for different models produce independent
    placements.  Each model gets exactly one placement, and placements
    do not interfere with each other.
    """
    await register_node(scheduler, "node-1", NODE_URL)
    models = [MODEL_A, MODEL_B, MODEL_C]
    M = 20  # concurrent ensures per model

    per_model_calls: dict[str, int] = {m: 0 for m in models}

    async def mock_ensure(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model_id = body.get("model", MODEL_A)
        per_model_calls[model_id] = per_model_calls.get(model_id, 0) + 1
        await asyncio.sleep(0.02)
        return httpx.Response(200, json=node_ensure_payload(model_id=model_id))

    async with respx.MockRouter() as mock:
        mock.post(f"{NODE_URL}/local/ensure").mock(side_effect=mock_ensure)

        all_tasks = [
            scheduler.ensure(model)
            for model in models
            for _ in range(M)
        ]
        results = await asyncio.gather(*all_tasks)

    # Each model must have exactly one placement
    placement_keys = await fake_redis.keys("scheduler:placement:*")
    assert len(placement_keys) == len(models), (
        f"Expected {len(models)} placements (one per model); got {len(placement_keys)}"
    )

    # Each model must route to the same api_base across all M results
    for model in models:
        model_results = [r for r in results if r.model_id == model]
        api_bases = {r.api_base for r in model_results}
        assert len(api_bases) == 1, (
            f"Model '{model}': all {M} ensures must return the same api_base; "
            f"got {len(api_bases)} distinct values: {api_bases}"
        )


# ---------------------------------------------------------------------------
# 6. Latency profile under modest load
# ---------------------------------------------------------------------------


@pytest.mark.slow
async def test_ensure_latency_profile_under_moderate_load(scheduler):
    """
    INVARIANT: ensure() latency must not degrade severely under 50 concurrent
    calls.  This is a heuristic bound — not a hard SLA.

    A placement served from cache (fast path, no lock) should complete in
    well under 100ms.  The cold-start (first call) may take longer.
    """
    await register_node(scheduler, "node-1", NODE_URL)
    tracker = LatencyTracker()
    N = 50

    async def mock_ensure(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)  # 50ms simulated cold-start
        return httpx.Response(200, json=node_ensure_payload())

    async with respx.MockRouter() as mock:
        mock.post(f"{NODE_URL}/local/ensure").mock(side_effect=mock_ensure)

        # First call (cold-start)
        async with tracker.measure():
            await scheduler.ensure(MODEL)

    # Warm calls (served from Redis cache)
    async with respx.MockRouter(assert_all_called=False):
        for _ in range(N):
            async with tracker.measure():
                await scheduler.ensure(MODEL)

    # The cold-start accounts for most of the max latency.
    # Warm-path calls should be very fast (fakeredis is in-process).
    warm_samples = tracker._samples[1:]  # exclude cold start
    if warm_samples:
        warm_p99 = sorted(warm_samples)[int(len(warm_samples) * 0.99)]
        assert warm_p99 < 500, (  # 500ms is extremely generous for in-process fakeredis
            f"Warm-path p99 latency is {warm_p99:.1f}ms — "
            f"something is blocking the event loop unexpectedly"
        )


# ---------------------------------------------------------------------------
# 7. Deadlock detection — ensure() must not hang indefinitely
# ---------------------------------------------------------------------------


@pytest.mark.slow
async def test_ensure_completes_within_deadline_under_concurrency(scheduler):
    """
    INVARIANT: ensure() must never deadlock.  Under any concurrency level,
    all calls must complete within a reasonable timeout.

    A deadlock would manifest as asyncio.wait_for() raising TimeoutError.
    """
    await register_node(scheduler, "node-1", NODE_URL)

    async def mock_ensure(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.01)
        return httpx.Response(200, json=node_ensure_payload())

    N = 50

    async with respx.MockRouter() as mock:
        mock.post(f"{NODE_URL}/local/ensure").mock(side_effect=mock_ensure)

        try:
            await asyncio.wait_for(
                asyncio.gather(*[scheduler.ensure(MODEL) for _ in range(N)]),
                timeout=30.0,  # 30s is extremely generous; deadlock would hang forever
            )
        except asyncio.TimeoutError:
            pytest.fail(
                f"ensure() deadlocked under {N} concurrent calls. "
                f"The per-model asyncio.Lock may have a starvation issue."
            )


# ---------------------------------------------------------------------------
# 8. Sustained load — no state leakage across request batches
# ---------------------------------------------------------------------------


async def test_sustained_load_no_placement_accumulation(scheduler, fake_redis):
    """
    INVARIANT: repeated ensure() calls for the same model never accumulate
    extra placement records.  Redis must always have exactly one placement
    per model regardless of how many requests have been served.
    """
    await register_node(scheduler, "node-1", NODE_URL)

    async def mock_ensure(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=node_ensure_payload())

    BATCHES = 5
    BATCH_SIZE = 20

    async with respx.MockRouter(assert_all_called=False) as mock:
        mock.post(f"{NODE_URL}/local/ensure").mock(side_effect=mock_ensure)

        for batch in range(BATCHES):
            await asyncio.gather(*[scheduler.ensure(MODEL) for _ in range(BATCH_SIZE)])

            # After every batch: still exactly one placement
            keys = await fake_redis.keys("scheduler:placement:*")
            assert len(keys) == 1, (
                f"After batch {batch + 1}: expected 1 placement, found {len(keys)}"
            )
