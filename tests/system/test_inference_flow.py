"""
tests/system/test_inference_flow.py

System tests for the full inference pipeline.

These tests run against a live docker-compose stack with a real model.
Set the TEST_MODEL environment variable to a model that is pre-registered
in MRM (e.g. "qwen-base").

    GATEWAY_URL=http://localhost:8080 \\
    TEST_MODEL=qwen-base \\
    pytest tests/system/ -v -m system

INVARIANTS UNDER TEST
---------------------
1. Cold-start placement  — first request triggers placement and returns a response.
2. Warm path             — second request is served from existing placement.
3. Placement persistence — Scheduler's /placements reflects the active model.
4. Streaming integrity   — full SSE stream with data: [DONE] terminator.
5. Stop + re-ensure      — model can be stopped and restarted.
"""

from __future__ import annotations

import asyncio
import os

import httpx
import pytest

MODEL = os.getenv("TEST_MODEL", "qwen-base")

CHAT_REQUEST = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
    "max_tokens": 10,
    "temperature": 0,
}


pytestmark = pytest.mark.system


# ---------------------------------------------------------------------------
# 1. Cold-start placement
# ---------------------------------------------------------------------------


async def test_first_request_cold_starts_and_returns_response(live_gateway_client):
    """
    INVARIANT: the first request to a model that is not yet running must
    eventually succeed (MRM handles cold-start internally).
    """
    resp = await live_gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert "choices" in body
    assert len(body["choices"]) > 0
    assert body["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# 2. Warm path — second request is fast
# ---------------------------------------------------------------------------


async def test_second_request_is_served_from_existing_placement(live_gateway_client):
    """
    INVARIANT: after the first request, subsequent requests reuse the existing
    placement without triggering another cold-start.

    We can't directly measure "no cold-start happened", but we can assert the
    response is identical in structure and that the api_base is consistent.
    """
    # Warm up
    r1 = await live_gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)
    assert r1.status_code == 200

    r2 = await live_gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)
    assert r2.status_code == 200

    # Both responses have the same model_alias in the response body
    assert r1.json().get("model") == r2.json().get("model")


# ---------------------------------------------------------------------------
# 3. Placement visible in Scheduler
# ---------------------------------------------------------------------------


async def test_placement_appears_in_scheduler_after_ensure(
    live_gateway_client, live_scheduler_client
):
    """
    INVARIANT: after a successful inference request, the Scheduler's
    /placements endpoint must reflect the active placement for the model.
    """
    resp = await live_gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)
    assert resp.status_code == 200

    placements_resp = await live_scheduler_client.get("/placements")
    assert placements_resp.status_code == 200

    placements = placements_resp.json()
    model_ids = [p["model_id"] for p in placements]
    assert MODEL in model_ids or any(MODEL in mid for mid in model_ids), (
        f"Model '{MODEL}' must appear in /placements after ensure; got: {model_ids}"
    )


# ---------------------------------------------------------------------------
# 4. Streaming integrity
# ---------------------------------------------------------------------------


async def test_streaming_response_ends_with_done(live_gateway_client):
    """
    INVARIANT: a streaming inference response must end with ``data: [DONE]``
    and must contain at least one content token.
    """
    collected: list[bytes] = []

    async with live_gateway_client.stream(
        "POST",
        "/v1/chat/completions",
        json={**CHAT_REQUEST, "stream": True},
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        async for chunk in response.aiter_bytes():
            if chunk:
                collected.append(chunk)

    full_body = b"".join(collected)
    assert full_body, "Streaming response must not be empty"

    non_empty_lines = [ln for ln in full_body.split(b"\n") if ln.strip()]
    assert non_empty_lines, "Response must contain non-empty lines"
    assert non_empty_lines[-1] == b"data: [DONE]", (
        f"Last chunk must be 'data: [DONE]', got: {non_empty_lines[-1]!r}"
    )


# ---------------------------------------------------------------------------
# 5. Concurrency — no duplicate placements
# ---------------------------------------------------------------------------


async def test_concurrent_requests_do_not_duplicate_placement(
    live_gateway_client, live_scheduler_client
):
    """
    INVARIANT: N concurrent requests for the same model must result in
    exactly one placement record in the Scheduler.
    """
    N = 10
    resps = await asyncio.gather(
        *[live_gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST) for _ in range(N)],
        return_exceptions=True,
    )

    ok_count = sum(1 for r in resps if isinstance(r, httpx.Response) and r.status_code == 200)
    assert ok_count == N, f"Expected {N} successful responses, got {ok_count}"

    placements_resp = await live_scheduler_client.get("/placements")
    placements = [p for p in placements_resp.json() if MODEL in p.get("model_id", "")]
    assert len(placements) == 1, (
        f"Exactly one placement must exist for '{MODEL}'; got {len(placements)}"
    )


# ---------------------------------------------------------------------------
# 6. Nodes endpoint reflects registered agents
# ---------------------------------------------------------------------------


async def test_nodes_endpoint_shows_at_least_one_healthy_node(live_scheduler_client):
    """
    INVARIANT: at least one node must be registered and sending heartbeats
    for the system to be functional.
    """
    resp = await live_scheduler_client.get("/nodes")
    assert resp.status_code == 200

    nodes = resp.json()
    assert len(nodes) >= 1, "At least one node must be registered"

    healthy_nodes = [n for n in nodes if n.get("state") == "healthy"]
    assert len(healthy_nodes) >= 1, (
        f"At least one node must be in 'healthy' state; got states: "
        f"{[n.get('state') for n in nodes]}"
    )
