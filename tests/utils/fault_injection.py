"""
tests/utils/fault_injection.py

Fault injection primitives for resilience testing.

These are plain callables and classes — not pytest fixtures — so they can
be used in any context (integration tests, load tests, property tests).

Usage example
-------------
    async with respx.MockRouter() as mock:
        mock.post(VLLM_URL).mock(side_effect=simulate_timeout())
        resp = await gateway_client.post("/v1/chat/completions", json=body)
        assert resp.status_code == 502

    async with respx.MockRouter() as mock:
        mock.post(VLLM_URL).mock(
            return_value=httpx.Response(
                200,
                stream=MidStreamFailure(SSE_CHUNKS, fail_after=2),
                headers={"content-type": "text/event-stream"},
            )
        )
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Streaming fault: partial SSE then connection reset
# ---------------------------------------------------------------------------


class MidStreamFailure(httpx.AsyncByteStream):
    """
    An httpx AsyncByteStream that yields ``fail_after`` chunks normally,
    then raises the given error to simulate a mid-stream connection reset.

    Used to test Gateway resilience when the upstream connection drops
    while a streaming response is in progress.

    Example::

        stream = MidStreamFailure(
            chunks=[chunk_1, chunk_2, chunk_3],
            fail_after=2,                          # yields chunk_1, chunk_2, then fails
            error=httpx.RemoteProtocolError("Connection reset by peer"),
        )
        respx.post(url).mock(
            return_value=httpx.Response(200, stream=stream,
                                        headers={"content-type": "text/event-stream"})
        )
    """

    def __init__(
        self,
        chunks: list[bytes],
        fail_after: int,
        error: Exception | None = None,
    ) -> None:
        self._chunks = chunks
        self._fail_after = fail_after
        self._error = error or httpx.RemoteProtocolError(
            "Connection reset by peer", request=None
        )

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for i, chunk in enumerate(self._chunks):
            if i >= self._fail_after:
                raise self._error
            await asyncio.sleep(0)  # give the event loop a chance to switch
            yield chunk

    async def aclose(self) -> None:
        pass


class TruncatedStream(httpx.AsyncByteStream):
    """
    Yields all chunks but never sends the ``data: [DONE]`` terminator.
    Simulates a server that closes the connection without finishing the SSE stream.
    """

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            await asyncio.sleep(0)
            yield chunk
        # Intentionally omits data: [DONE]

    async def aclose(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Side-effect factories for respx
# ---------------------------------------------------------------------------


def simulate_latency(
    delay_sec: float,
    response_json: dict[str, Any],
    status_code: int = 200,
) -> Callable[[httpx.Request], httpx.Response]:
    """
    Returns a respx side_effect that introduces artificial latency before
    returning a successful response.

    Use this to simulate a slow instance and verify that the routing strategy
    deprioritises it when load information is available.
    """

    async def _side_effect(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(delay_sec)
        return httpx.Response(status_code, json=response_json)

    return _side_effect


def simulate_failure(
    status_code: int = 500,
    detail: str = "Internal Server Error",
) -> Callable[[httpx.Request], httpx.Response]:
    """Returns a respx side_effect that always returns an error response."""

    async def _side_effect(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"detail": detail, "error": detail})

    return _side_effect


def simulate_timeout() -> Callable[[httpx.Request], httpx.Response]:
    """Returns a respx side_effect that raises httpx.ReadTimeout."""

    async def _side_effect(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Upstream read timeout", request=request)

    return _side_effect


def simulate_connect_error() -> Callable[[httpx.Request], httpx.Response]:
    """Returns a respx side_effect that raises httpx.ConnectError."""

    async def _side_effect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused", request=request)

    return _side_effect


def simulate_flaky(
    success_json: dict[str, Any],
    fail_every_n: int = 2,
    fail_status: int = 500,
) -> Callable[[httpx.Request], httpx.Response]:
    """
    Returns a respx side_effect that fails every N-th call and succeeds otherwise.

    Useful for testing retry logic and partial-failure scenarios where some
    requests succeed even when the upstream is degraded.

    Example: fail_every_n=3 → calls 1, 2 succeed; call 3 fails; 4, 5 succeed; 6 fails; ...
    """
    call_count = 0

    async def _side_effect(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count % fail_every_n == 0:
            return httpx.Response(fail_status, json={"detail": "Simulated flaky failure"})
        return httpx.Response(200, json=success_json)

    return _side_effect


def simulate_n_failures_then_recover(
    success_json: dict[str, Any],
    fail_count: int,
    fail_status: int = 500,
) -> Callable[[httpx.Request], httpx.Response]:
    """
    Returns a respx side_effect that fails the first ``fail_count`` calls,
    then succeeds for all subsequent calls.

    Simulates a transient outage followed by recovery.
    """
    call_count = 0

    async def _side_effect(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count <= fail_count:
            return httpx.Response(fail_status, json={"detail": f"Failure {call_count}/{fail_count}"})
        return httpx.Response(200, json=success_json)

    return _side_effect


# ---------------------------------------------------------------------------
# Concurrency tracking
# ---------------------------------------------------------------------------


class ConcurrencyCounter:
    """
    Tracks peak concurrent calls in async side effects.

    Use this to verify that the system does not issue more concurrent
    upstream calls than expected (e.g. the scheduler lock ensures only
    one ensure() call goes to the Node Agent at a time).

    Example::

        counter = ConcurrencyCounter()

        async def mock_ensure(request):
            async with counter:
                await asyncio.sleep(0.05)
                return httpx.Response(200, json={...})

        ...

        assert counter.peak == 1, "Only one concurrent Node Agent call expected"
    """

    def __init__(self) -> None:
        self.current: int = 0
        self.peak: int = 0
        self.total: int = 0
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "ConcurrencyCounter":
        async with self._lock:
            self.current += 1
            self.total += 1
            if self.current > self.peak:
                self.peak = self.current
        return self

    async def __aexit__(self, *_: object) -> None:
        async with self._lock:
            self.current -= 1


# ---------------------------------------------------------------------------
# Latency tracker
# ---------------------------------------------------------------------------


class LatencyTracker:
    """
    Records end-to-end latency for a batch of async calls.

    Use in load tests to assert p50/p99 bounds.

    Example::

        tracker = LatencyTracker()
        async with tracker.measure():
            result = await scheduler.ensure(MODEL)
        print(tracker.p99_ms)
    """

    def __init__(self) -> None:
        self._samples: list[float] = []

    class _Measure:
        def __init__(self, tracker: "LatencyTracker") -> None:
            self._tracker = tracker
            self._t0: float = 0.0

        async def __aenter__(self) -> "_Measure":
            self._t0 = time.perf_counter()
            return self

        async def __aexit__(self, *_: object) -> None:
            elapsed_ms = (time.perf_counter() - self._t0) * 1_000
            self._tracker._samples.append(elapsed_ms)

    def measure(self) -> "_Measure":
        return self._Measure(self)

    def _sorted(self) -> list[float]:
        return sorted(self._samples)

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def p50_ms(self) -> float:
        s = self._sorted()
        return s[len(s) // 2] if s else 0.0

    @property
    def p95_ms(self) -> float:
        s = self._sorted()
        return s[int(len(s) * 0.95)] if s else 0.0

    @property
    def p99_ms(self) -> float:
        s = self._sorted()
        return s[int(len(s) * 0.99)] if s else 0.0

    @property
    def max_ms(self) -> float:
        return max(self._samples) if self._samples else 0.0
