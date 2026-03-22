"""
tests/integration/test_scheduler_invariants.py

Integration tests for the Scheduler's core behavioral invariants.

These tests run the real Scheduler + real placement/registry logic against
a fake Redis.  Node Agent HTTP calls are intercepted by respx.

INVARIANTS UNDER TEST
---------------------
1. Idempotency      ensure(model) × N  →  same api_base, Node Agent called once
2. Single placement one model          →  exactly one Redis placement record
3. Concurrency      N parallel ensures →  one Node Agent call, all same api_base
4. Failover         dead node          →  next ensure goes to a live node
5. No cross-contamination  modelA placement does not affect modelB
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

pytestmark = pytest.mark.invariant

from tests.integration.conftest import node_agent_ensure_payload, register_node

NODE_1_URL = "http://node-1:8020"
NODE_2_URL = "http://node-2:8020"
MODEL = "meta-llama/Llama-2-7b-hf"
MODEL_B = "Qwen/Qwen2-7B"


# ---------------------------------------------------------------------------
# 1. Idempotency
# ---------------------------------------------------------------------------


async def test_ensure_is_idempotent(scheduler):
    """
    INVARIANT: ensure(model) called N times always returns the same api_base.

    The Node Agent must be called exactly once (cold start); every subsequent
    call is served from the Redis placement cache.
    """
    await register_node(scheduler, "node-1", NODE_1_URL)

    async with respx.MockRouter(assert_all_called=False) as mock:
        mock.post(f"{NODE_1_URL}/local/ensure").mock(
            return_value=httpx.Response(200, json=node_agent_ensure_payload(MODEL))
        )

        results = [await scheduler.ensure(MODEL) for _ in range(5)]
        node_agent_calls = mock.calls.call_count

    api_bases = {r.api_base for r in results}
    assert len(api_bases) == 1, "All ensure calls must return the same api_base"
    assert node_agent_calls == 1, (
        f"Node Agent must be called exactly once (cold start); got {node_agent_calls}"
    )


# ---------------------------------------------------------------------------
# 2. Single placement record
# ---------------------------------------------------------------------------


async def test_single_placement_record_in_redis(scheduler, fake_redis):
    """
    INVARIANT: after N ensure() calls for the same model, Redis contains
    exactly one placement record with exactly one instance.
    """
    await register_node(scheduler, "node-1", NODE_1_URL)

    async with respx.MockRouter(assert_all_called=False) as mock:
        mock.post(f"{NODE_1_URL}/local/ensure").mock(
            return_value=httpx.Response(200, json=node_agent_ensure_payload(MODEL))
        )
        for _ in range(3):
            await scheduler.ensure(MODEL)

    placement_keys = await fake_redis.keys("scheduler:placement:*")
    assert len(placement_keys) == 1, "Exactly one placement record per model"

    placement = json.loads(await fake_redis.get(placement_keys[0]))
    assert len(placement["instances"]) == 1, "One instance per placement (unless scaled)"


# ---------------------------------------------------------------------------
# 3. Concurrency — the thundering herd
# ---------------------------------------------------------------------------


async def test_concurrent_ensures_produce_single_placement(scheduler):
    """
    INVARIANT: N concurrent ensure() calls for the same model result in
    exactly one Node Agent invocation and all callers receive the same api_base.

    This validates the per-model asyncio.Lock in the Scheduler.
    """
    await register_node(scheduler, "node-1", NODE_1_URL)

    node_agent_call_count = 0

    async def slow_ensure(request: httpx.Request) -> httpx.Response:
        nonlocal node_agent_call_count
        # Simulate cold-start delay — gives the event loop time to switch
        # between the waiting coroutines, maximising lock contention.
        await asyncio.sleep(0.05)
        node_agent_call_count += 1
        return httpx.Response(200, json=node_agent_ensure_payload(MODEL))

    N = 40

    async with respx.MockRouter() as mock:
        mock.post(f"{NODE_1_URL}/local/ensure").mock(side_effect=slow_ensure)
        results = await asyncio.gather(*[scheduler.ensure(MODEL) for _ in range(N)])

    api_bases = {r.api_base for r in results}
    assert len(api_bases) == 1, f"All {N} callers must receive the same api_base"
    assert node_agent_call_count == 1, (
        f"Node Agent must be called exactly once despite {N} concurrent ensures; "
        f"got {node_agent_call_count}"
    )


# ---------------------------------------------------------------------------
# 4. Failover — dead node triggers re-placement
# ---------------------------------------------------------------------------


async def test_failover_re_places_model_on_live_node(scheduler, fake_redis):
    """
    INVARIANT: when the node hosting a model's placement dies (its TTL
    expires in Redis), the next ensure() call selects a different live node.
    """
    await register_node(scheduler, "node-1", NODE_1_URL, free_mb=20_000)

    # Phase 1: initial placement on node-1
    async with respx.MockRouter() as mock:
        mock.post(f"{NODE_1_URL}/local/ensure").mock(
            return_value=httpx.Response(
                200,
                json=node_agent_ensure_payload(MODEL, api_base="http://node-1:8000/v1"),
            )
        )
        r1 = await scheduler.ensure(MODEL)

    assert r1.instances[0].node_id == "node-1"
    assert "node-1:8000" in r1.api_base

    # Phase 2: kill node-1 (simulate TTL expiry)
    await fake_redis.delete(f"scheduler:node:node-1")
    await fake_redis.srem("scheduler:nodes", "node-1")

    # Phase 3: register node-2 as a replacement
    await register_node(scheduler, "node-2", NODE_2_URL, free_mb=20_000)

    # Phase 4: re-ensure must go to node-2
    async with respx.MockRouter() as mock:
        mock.post(f"{NODE_2_URL}/local/ensure").mock(
            return_value=httpx.Response(
                200,
                json=node_agent_ensure_payload(MODEL, api_base="http://node-2:8000/v1"),
            )
        )
        r2 = await scheduler.ensure(MODEL)

    assert r2.instances[0].node_id == "node-2", (
        "After node-1 dies, the model must be re-placed on node-2"
    )
    assert "node-2:8000" in r2.api_base
    assert r2.api_base != r1.api_base, "api_base must change after failover"


# ---------------------------------------------------------------------------
# 5. No cross-contamination between models
# ---------------------------------------------------------------------------


async def test_two_models_get_independent_placements(scheduler, fake_redis):
    """
    INVARIANT: placing modelA has no effect on modelB's placement.
    Each model is tracked independently in Redis.
    """
    await register_node(scheduler, "node-1", NODE_1_URL)

    async with respx.MockRouter(assert_all_called=False) as mock:
        mock.post(f"{NODE_1_URL}/local/ensure").mock(
            side_effect=lambda req: httpx.Response(
                200,
                json=node_agent_ensure_payload(
                    req.content.decode(),  # just needs to not crash
                    api_base="http://node-1:8000/v1",
                ),
            )
        )
        rA = await scheduler.ensure(MODEL)
        rB = await scheduler.ensure(MODEL_B)

    # Both placed
    keys = await fake_redis.keys("scheduler:placement:*")
    assert len(keys) == 2, "Each model must have its own placement record"

    # Placements are independent objects
    assert rA.model_id != rB.model_id


# ---------------------------------------------------------------------------
# 6. Stop removes placement from Redis
# ---------------------------------------------------------------------------


async def test_stop_removes_placement(scheduler, fake_redis):
    """
    INVARIANT: after stop(model), the placement is removed from Redis and
    a subsequent ensure() triggers a new cold-start placement.
    """
    await register_node(scheduler, "node-1", NODE_1_URL)

    call_count = 0

    async def counting_ensure(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(200, json=node_agent_ensure_payload(MODEL))

    async with respx.MockRouter(assert_all_called=False) as mock:
        mock.post(f"{NODE_1_URL}/local/ensure").mock(side_effect=counting_ensure)
        mock.post(f"{NODE_1_URL}/local/stop").mock(return_value=httpx.Response(204))

        await scheduler.ensure(MODEL)
        assert call_count == 1

        await scheduler.stop(MODEL)

        # Placement must be gone from Redis
        assert await fake_redis.get(f"scheduler:placement:{MODEL}") is None

        # Re-ensure triggers another cold start
        await scheduler.ensure(MODEL)
        assert call_count == 2, "After stop, re-ensure must call the Node Agent again"


# ---------------------------------------------------------------------------
# 7. No healthy nodes → clear error
# ---------------------------------------------------------------------------


async def test_ensure_raises_when_no_nodes_are_available(scheduler):
    """
    INVARIANT: if no healthy nodes are registered, ensure() raises RuntimeError.
    The scheduler must never return a response with an empty api_base.
    """
    # No nodes registered
    with pytest.raises(RuntimeError, match="No healthy nodes"):
        await scheduler.ensure(MODEL)
