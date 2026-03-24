"""
tests/unit/test_tracing.py

Tests for end-to-end request_id propagation.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock


def _setup_loaded():
    import cpu_runtime.inference as inf_mod
    from cpu_runtime.config import settings
    mock_engine = MagicMock()
    mock_engine.generate = AsyncMock(return_value=MagicMock(
        text="hi", prompt_tokens=1, completion_tokens=1,
        finish_reason="stop", model=settings.model_alias,
    ))
    inf_mod.engine = mock_engine
    inf_mod.load_state = "loaded"


class TestRequestIdInCpuRuntime:

    def test_x_request_id_in_response_headers(self):
        """Every response from cpu_runtime must include X-Request-ID."""
        _setup_loaded()
        from cpu_runtime.app import app
        client = TestClient(app)
        resp = client.get("/health")
        assert "x-request-id" in resp.headers

    def test_forwarded_request_id_echoed(self):
        """A client-provided X-Request-ID must be echoed back in the response."""
        _setup_loaded()
        from cpu_runtime.app import app
        client = TestClient(app)
        resp = client.get("/health", headers={"x-request-id": "test-abc-123"})
        assert resp.headers.get("x-request-id") == "test-abc-123"

    def test_generated_request_id_when_none_provided(self):
        """When no X-Request-ID is sent, cpu_runtime generates one."""
        _setup_loaded()
        from cpu_runtime.app import app
        client = TestClient(app)
        resp = client.get("/health")
        rid = resp.headers.get("x-request-id", "")
        assert rid != ""
        assert len(rid) >= 8  # at least 8 hex chars

    def test_request_id_consistent_within_response(self):
        """The same request_id must be in the response header."""
        _setup_loaded()
        from cpu_runtime.app import app
        client = TestClient(app)
        resp = client.post("/v1/chat/completions", json={
            "model": "cpu-model",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert "x-request-id" in resp.headers
        rid = resp.headers["x-request-id"]
        assert len(rid) >= 8

    def test_request_id_in_ready_endpoint(self):
        """All endpoints including /ready must propagate X-Request-ID."""
        _setup_loaded()
        from cpu_runtime.app import app
        client = TestClient(app)
        resp = client.get("/ready")
        assert "x-request-id" in resp.headers


class TestRequestIdPropagationToGateway:
    """Gateway must propagate X-Request-ID in responses."""

    def test_gateway_returns_request_id_header(self):
        """The gateway observability_middleware must set X-Request-ID."""
        from gateway.main import app as gw_app
        client = TestClient(gw_app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert "x-request-id" in resp.headers

    def test_gateway_echoes_client_request_id(self):
        """A client-provided X-Request-ID must be echoed in the response."""
        from gateway.main import app as gw_app
        client = TestClient(gw_app, raise_server_exceptions=False)
        resp = client.get("/health", headers={"x-request-id": "client-id-xyz"})
        assert resp.headers.get("x-request-id") == "client-id-xyz"
