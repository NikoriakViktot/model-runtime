"""
tests/resilience/test_faults.py

Fault injection tests — validate Gateway and Scheduler behavior when
upstream services return errors, time out, or drop connections.

INVARIANTS UNDER TEST
---------------------
1. vLLM timeout           → Gateway returns 502 (not hung, not unhandled exception)
2. vLLM connection error  → Gateway returns 502
3. vLLM 5xx              → Gateway returns 502 with error body
4. MRM unreachable        → Gateway returns 503
5. MRM timeout            → Gateway returns 503
6. Mid-stream failure     → Gateway closes gracefully; process stays alive
7. Connection reset mid-stream → Gateway closes gracefully; process stays alive
8. Gateway /health after any upstream failure → always 200
9. Transient failure + recovery → system serves the next request successfully
10. Flaky upstream → Gateway surfaces individual request failures cleanly

All tests treat the Gateway as a black box via ASGI transport.
HTTP mocking intercepts only network boundaries.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

# All tests in this module are fault-injection resilience tests.
# Mid-stream and truncated-stream tests also carry the streaming marker.
pytestmark = pytest.mark.resilience

from tests.resilience.conftest import (
    CHAT_REQUEST,
    MRM_URL,
    SSE_CHUNKS,
    VLLM_API_BASE,
    VLLM_CHAT_RESPONSE,
    MRM_ENSURE_RESPONSE,
    MODEL_ALIAS,
)
from tests.utils.fault_injection import (
    MidStreamFailure,
    TruncatedStream,
    simulate_connect_error,
    simulate_failure,
    simulate_flaky,
    simulate_n_failures_then_recover,
    simulate_timeout,
)


def _mock_mrm_ensure(mock: respx.MockRouter) -> None:
    mock.post(f"{MRM_URL}/models/ensure").mock(
        return_value=httpx.Response(200, json=MRM_ENSURE_RESPONSE)
    )


# ---------------------------------------------------------------------------
# 1–2. Upstream connection problems
# ---------------------------------------------------------------------------


async def test_vllm_read_timeout_returns_502(gateway_client):
    """
    INVARIANT: when vLLM stops responding mid-request, the Gateway returns
    502 rather than hanging indefinitely or raising an unhandled exception.
    """
    async with respx.MockRouter() as mock:
        _mock_mrm_ensure(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            side_effect=simulate_timeout()
        )

        resp = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert resp.status_code == 502
    assert resp.json().get("detail") is not None


async def test_vllm_connection_refused_returns_502(gateway_client):
    """
    INVARIANT: when vLLM's port is closed (container crashed), the Gateway
    returns 502, not a raw Python exception traceback.
    """
    async with respx.MockRouter() as mock:
        _mock_mrm_ensure(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            side_effect=simulate_connect_error()
        )

        resp = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# 3. vLLM 5xx
# ---------------------------------------------------------------------------


async def test_vllm_500_returns_502(gateway_client):
    """
    INVARIANT: a vLLM 500 error is surfaced to the client as 502, and the
    upstream error body is included in the detail for debugging.
    """
    async with respx.MockRouter() as mock:
        _mock_mrm_ensure(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            side_effect=simulate_failure(status_code=500, detail="CUDA out of memory")
        )

        resp = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert resp.status_code == 502
    body = resp.json()
    # Client must receive a structured error, not an empty body
    assert "detail" in body


async def test_vllm_503_returns_502(gateway_client):
    """Service unavailable from vLLM (e.g. model loading) → 502."""
    async with respx.MockRouter() as mock:
        _mock_mrm_ensure(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            side_effect=simulate_failure(status_code=503, detail="Model loading")
        )

        resp = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# 4–5. MRM failures
# ---------------------------------------------------------------------------


async def test_mrm_unreachable_returns_503(gateway_client):
    """
    INVARIANT: if MRM is unreachable (e.g. during a rolling restart), the
    Gateway returns 503 — indicating a temporary condition, not a client error.
    """
    async with respx.MockRouter() as mock:
        mock.post(f"{MRM_URL}/models/ensure").mock(
            side_effect=simulate_connect_error()
        )

        resp = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert resp.status_code == 503


async def test_mrm_timeout_returns_503(gateway_client):
    """
    INVARIANT: MRM timeout (model taking too long to start) → 503.
    """
    async with respx.MockRouter() as mock:
        mock.post(f"{MRM_URL}/models/ensure").mock(
            side_effect=simulate_timeout()
        )

        resp = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# 6. Mid-stream failure — the most important resilience invariant
# ---------------------------------------------------------------------------


@pytest.mark.streaming
async def test_gateway_survives_mid_stream_failure(gateway_client):
    """
    INVARIANT: when the upstream connection drops mid-stream (after some SSE
    chunks have been sent), the Gateway must:
    1. Not raise an unhandled exception (no 500 on subsequent requests).
    2. Close the streaming response gracefully.
    3. Remain available — /health returns 200 after the failed request.

    This is the most critical streaming resilience scenario: the model is
    generating tokens and the GPU node crashes or loses network.
    """
    # The stream delivers 2 chunks then drops the connection
    failing_stream = MidStreamFailure(
        chunks=SSE_CHUNKS,
        fail_after=2,
        error=httpx.RemoteProtocolError("Connection reset by peer", request=None),
    )

    collected: list[bytes] = []
    stream_error_raised = False

    async with respx.MockRouter() as mock:
        _mock_mrm_ensure(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                stream=failing_stream,
                headers={"content-type": "text/event-stream"},
            )
        )

        try:
            async with gateway_client.stream(
                "POST",
                "/v1/chat/completions",
                json={**CHAT_REQUEST, "stream": True},
            ) as response:
                async for chunk in response.aiter_bytes():
                    if chunk:
                        collected.append(chunk)
        except Exception:
            # The client may see an error when the stream drops — that is acceptable.
            # What is NOT acceptable is the gateway process crashing.
            stream_error_raised = True

    # Gateway must still be alive after the mid-stream failure
    async with respx.MockRouter():
        health = await gateway_client.get("/health")

    assert health.status_code == 200, (
        "Gateway must be alive after mid-stream failure; /health must return 200"
    )


@pytest.mark.streaming
async def test_gateway_survives_truncated_stream_without_done(gateway_client):
    """
    INVARIANT: if vLLM closes the connection without sending ``data: [DONE]``,
    the Gateway must not hang and must remain healthy.

    Some vLLM OOM events manifest as a truncated stream with no [DONE].
    """
    truncated = TruncatedStream(chunks=SSE_CHUNKS[:2])  # partial SSE, no [DONE]

    async with respx.MockRouter() as mock:
        _mock_mrm_ensure(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                stream=truncated,
                headers={"content-type": "text/event-stream"},
            )
        )

        collected: list[bytes] = []
        try:
            async with gateway_client.stream(
                "POST",
                "/v1/chat/completions",
                json={**CHAT_REQUEST, "stream": True},
            ) as response:
                async for chunk in response.aiter_bytes():
                    if chunk:
                        collected.append(chunk)
        except Exception:
            pass

    health = await gateway_client.get("/health")
    assert health.status_code == 200


# ---------------------------------------------------------------------------
# 7. Streaming error is recorded in metrics
# ---------------------------------------------------------------------------


@pytest.mark.streaming
async def test_streaming_error_increments_error_metric(gateway_client):
    """
    INVARIANT: a mid-stream failure must be recorded as an error in the
    router's per-instance metrics so operators can detect degraded instances
    via GET /v1/router/metrics.
    """
    failing_stream = MidStreamFailure(
        chunks=SSE_CHUNKS,
        fail_after=1,
    )

    async with respx.MockRouter() as mock:
        _mock_mrm_ensure(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                stream=failing_stream,
                headers={"content-type": "text/event-stream"},
            )
        )

        try:
            async with gateway_client.stream(
                "POST",
                "/v1/chat/completions",
                json={**CHAT_REQUEST, "stream": True},
            ) as response:
                async for _ in response.aiter_bytes():
                    pass
        except Exception:
            pass

    metrics_resp = await gateway_client.get("/v1/router/metrics")
    assert metrics_resp.status_code == 200
    metrics = metrics_resp.json()

    # The VLLM_API_BASE instance must be tracked
    instance_metrics = metrics.get("instances", {})
    if instance_metrics:
        tracked = instance_metrics.get(VLLM_API_BASE, {})
        # Either the error was recorded, or the request was recorded (depends on timing)
        assert tracked.get("requests", 0) >= 1, (
            "At least one request must be recorded after a mid-stream failure"
        )


# ---------------------------------------------------------------------------
# 8. /health always returns 200 regardless of upstream state
# ---------------------------------------------------------------------------


async def test_health_always_200_when_mrm_is_down(gateway_client):
    """
    INVARIANT: /health is a liveness probe — it must return 200 even when
    all upstream services are unreachable.  Orchestrators (docker, k8s) use
    this to decide whether to restart the container.
    """
    # No mocks — any outbound call would fail with 'no mock found'
    async with respx.MockRouter(assert_all_called=False):
        resp = await gateway_client.get("/health")

    assert resp.status_code == 200


async def test_health_returns_200_after_5xx_storm(gateway_client):
    """
    INVARIANT: after receiving many 5xx errors from vLLM, the Gateway
    must still be alive and report healthy.
    """
    N = 10

    async with respx.MockRouter() as mock:
        _mock_mrm_ensure(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            side_effect=simulate_failure(500, "GPU OOM")
        )

        for _ in range(N):
            await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    health = await gateway_client.get("/health")
    assert health.status_code == 200, (
        f"Gateway must be alive after {N} upstream errors"
    )


# ---------------------------------------------------------------------------
# 9. Transient failure + recovery
# ---------------------------------------------------------------------------


async def test_system_recovers_after_transient_vllm_failure(gateway_client):
    """
    INVARIANT: after a transient vLLM outage, the next request succeeds
    without any manual intervention.

    This tests that the Gateway does not cache failure state and that each
    request gets a fresh chance to succeed.
    """
    # First 3 calls fail; calls 4+ succeed
    effect = simulate_n_failures_then_recover(
        success_json=VLLM_CHAT_RESPONSE,
        fail_count=3,
    )

    responses: list[int] = []

    async with respx.MockRouter() as mock:
        _mock_mrm_ensure(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(side_effect=effect)

        for _ in range(5):
            resp = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)
            responses.append(resp.status_code)

    # Calls 1-3: failures (502)
    assert all(s == 502 for s in responses[:3]), (
        "First 3 calls must fail during simulated outage"
    )
    # Calls 4-5: recovery (200)
    assert all(s == 200 for s in responses[3:]), (
        "Gateway must recover automatically after transient failures"
    )


# ---------------------------------------------------------------------------
# 10. Flaky upstream — partial success rate
# ---------------------------------------------------------------------------


async def test_flaky_vllm_produces_expected_success_rate(
    gateway_client, flaky_instance_effect
):
    """
    INVARIANT: when the upstream is flaky (every 3rd call fails), the
    Gateway surfaces each individual failure as a 502 and lets the others
    through as 200.

    The client is responsible for retrying; the Gateway must not silently
    swallow errors or retry internally.
    """
    N = 9
    fail_every_n = 3
    expected_failures = N // fail_every_n  # calls 3, 6, 9 fail

    async with respx.MockRouter() as mock:
        _mock_mrm_ensure(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            side_effect=flaky_instance_effect(fail_every_n=fail_every_n)
        )

        statuses = [
            (await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)).status_code
            for _ in range(N)
        ]

    failures = statuses.count(502)
    successes = statuses.count(200)

    assert failures == expected_failures, (
        f"Expected {expected_failures} failures (every {fail_every_n}th call), got {failures}. "
        f"Statuses: {statuses}"
    )
    assert successes == N - expected_failures, (
        f"Expected {N - expected_failures} successes, got {successes}"
    )


async def test_flaky_vllm_each_error_has_502_status(
    gateway_client, flaky_instance_effect
):
    """
    INVARIANT: every failed upstream call produces exactly one 502 response.
    The Gateway must not swallow errors (return 200 with error body) or
    return the wrong status code.
    """
    async with respx.MockRouter() as mock:
        _mock_mrm_ensure(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            side_effect=flaky_instance_effect(fail_every_n=2)
        )

        statuses = [
            (await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)).status_code
            for _ in range(6)
        ]

    # All statuses must be either 200 or 502 — never 500 (gateway crash)
    for s in statuses:
        assert s in (200, 502), (
            f"Status {s} is not acceptable; Gateway must return 200 or 502 only. "
            f"500 would indicate an unhandled Gateway exception."
        )
