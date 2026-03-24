"""
tests/unit/test_timeout.py

Tests for generation timeout protection on the cpu_runtime inference endpoint:
  - Unary: asyncio.TimeoutError from eng.generate() → HTTP 504
  - Streaming: TimeoutError inside the generator → SSE error frame + counter decremented
  - Active request counter decremented on timeout in all paths
  - Gateway proxy: proxy.setup() accepts separate connect / read timeouts
"""
from __future__ import annotations

import asyncio
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client_with_timeout_engine(timeout_in_generate: bool = True):
    """
    Return a TestClient whose engine.generate() raises asyncio.TimeoutError.
    If timeout_in_generate=False, engine.generate() succeeds (for counter tests).
    """
    import cpu_runtime.inference as inf_mod
    from cpu_runtime.config import settings

    mock_engine = MagicMock()
    if timeout_in_generate:
        mock_engine.generate = AsyncMock(side_effect=asyncio.TimeoutError())
    else:
        mock_engine.generate = AsyncMock(return_value=MagicMock(
            text="ok",
            prompt_tokens=1,
            completion_tokens=1,
            finish_reason="stop",
            model=settings.model_alias,
        ))
    inf_mod.engine = mock_engine
    inf_mod.load_state = "loaded"

    from cpu_runtime.app import app
    return TestClient(app, raise_server_exceptions=False), mock_engine


# ---------------------------------------------------------------------------
# Unary timeout
# ---------------------------------------------------------------------------

class TestUnaryTimeout:

    def test_504_when_generate_times_out(self):
        """asyncio.TimeoutError from eng.generate() must return HTTP 504."""
        client, _ = _client_with_timeout_engine(timeout_in_generate=True)
        resp = client.post("/v1/chat/completions", json={
            "model": "cpu-model",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 504
        assert "timed out" in resp.json()["detail"].lower()

    def test_active_counter_decremented_on_timeout(self):
        """After a timeout the active request counter must return to its pre-request value."""
        import cpu_runtime.routes.chat as chat_mod

        client, _ = _client_with_timeout_engine(timeout_in_generate=True)
        before = chat_mod._active_requests

        client.post("/v1/chat/completions", json={
            "model": "cpu-model",
            "messages": [{"role": "user", "content": "hi"}],
        })

        assert chat_mod._active_requests == before, (
            f"Counter leaked: before={before}, after={chat_mod._active_requests}"
        )

    def test_active_counter_decremented_on_success(self):
        """Sanity: counter is also decremented on a successful call."""
        import cpu_runtime.routes.chat as chat_mod

        client, _ = _client_with_timeout_engine(timeout_in_generate=False)
        before = chat_mod._active_requests

        client.post("/v1/chat/completions", json={
            "model": "cpu-model",
            "messages": [{"role": "user", "content": "hi"}],
        })

        assert chat_mod._active_requests == before


# ---------------------------------------------------------------------------
# Streaming timeout (TimeoutError propagated through SSE generator)
# ---------------------------------------------------------------------------

class TestStreamingTimeout:

    def test_stream_error_frame_on_timeout(self):
        """
        When the stream generator raises asyncio.TimeoutError the response
        must include an SSE error frame rather than silently dropping the connection.
        """
        import cpu_runtime.inference as inf_mod

        async def _timeout_stream(*args, **kwargs):
            yield b"data: {}\n\n"
            raise asyncio.TimeoutError("stream timed out")

        mock_engine = MagicMock()
        mock_engine.stream = _timeout_stream
        inf_mod.engine = mock_engine
        inf_mod.load_state = "loaded"

        from cpu_runtime.app import app
        client = TestClient(app, raise_server_exceptions=False)
        with client.stream("POST", "/v1/chat/completions", json={
            "model": "cpu-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }) as response:
            raw = b"".join(response.iter_bytes())

        # An error SSE frame must be present
        assert b"error" in raw.lower()

    def test_stream_counter_decremented_after_timeout(self):
        """Active counter must be decremented after a streaming timeout."""
        import cpu_runtime.inference as inf_mod
        import cpu_runtime.routes.chat as chat_mod

        async def _timeout_stream(*args, **kwargs):
            raise asyncio.TimeoutError("stream timed out")
            yield  # make it an async generator

        mock_engine = MagicMock()
        mock_engine.stream = _timeout_stream
        inf_mod.engine = mock_engine
        inf_mod.load_state = "loaded"

        before = chat_mod._active_requests

        from cpu_runtime.app import app
        client = TestClient(app, raise_server_exceptions=False)
        with client.stream("POST", "/v1/chat/completions", json={
            "model": "cpu-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }) as response:
            list(response.iter_bytes())

        assert chat_mod._active_requests == before


# ---------------------------------------------------------------------------
# Gateway proxy timeout configuration
# ---------------------------------------------------------------------------

class TestProxyTimeoutConfig:
    """ProxyService.setup() must use separate connect / read timeouts."""

    @pytest.mark.asyncio
    async def test_proxy_uses_split_timeouts(self):
        """setup() must configure httpx.Timeout with distinct connect and read values."""
        import httpx
        from gateway.services.proxy import ProxyService

        svc = ProxyService()
        await svc.setup(connect_timeout=3.0, read_timeout=120.0)

        t = svc._http.timeout
        assert t.connect == 3.0, f"connect timeout mismatch: {t.connect}"
        assert t.read == 120.0, f"read timeout mismatch: {t.read}"

        await svc.teardown()

    @pytest.mark.asyncio
    async def test_proxy_falls_back_to_timeout_arg(self):
        """When read_timeout is not given, the legacy 'timeout' arg is used for read."""
        from gateway.services.proxy import ProxyService

        svc = ProxyService()
        await svc.setup(timeout=60.0, connect_timeout=4.0)

        t = svc._http.timeout
        assert t.read == 60.0
        assert t.connect == 4.0

        await svc.teardown()
