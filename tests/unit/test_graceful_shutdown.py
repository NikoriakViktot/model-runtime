"""
tests/unit/test_graceful_shutdown.py

Tests for graceful shutdown behaviour in cpu_runtime.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock


def _setup_loaded_engine():
    import cpu_runtime.inference as inf_mod
    from cpu_runtime.config import settings
    mock_engine = MagicMock()
    mock_engine.generate = AsyncMock(return_value=MagicMock(
        text="hi", prompt_tokens=1, completion_tokens=1,
        finish_reason="stop", model=settings.model_alias,
    ))
    inf_mod.engine = mock_engine
    inf_mod.load_state = "loaded"
    return mock_engine


class TestShutdownRejection:

    def test_503_when_shutting_down(self):
        """New requests must get 503 when state.shutting_down is True."""
        import cpu_runtime.state as _state
        _setup_loaded_engine()
        orig = _state.shutting_down
        try:
            _state.shutting_down = True
            from cpu_runtime.app import app
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/v1/chat/completions", json={
                "model": "cpu-model",
                "messages": [{"role": "user", "content": "hi"}],
            })
            assert resp.status_code == 503
            assert "shutting down" in resp.json()["detail"].lower()
        finally:
            _state.shutting_down = orig

    def test_health_still_200_when_shutting_down(self):
        """/health must remain 200 during shutdown (liveness ≠ readiness)."""
        import cpu_runtime.state as _state
        _setup_loaded_engine()
        orig = _state.shutting_down
        try:
            _state.shutting_down = True
            from cpu_runtime.app import app
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/health")
            assert resp.status_code == 200
        finally:
            _state.shutting_down = orig

    def test_counter_not_incremented_on_shutdown_rejection(self):
        """Active request counter must NOT be incremented for rejected requests."""
        import cpu_runtime.state as _state
        import cpu_runtime.routes.chat as chat_mod
        _setup_loaded_engine()
        orig_state = _state.shutting_down
        before = chat_mod._active_requests
        try:
            _state.shutting_down = True
            from cpu_runtime.app import app
            client = TestClient(app, raise_server_exceptions=False)
            client.post("/v1/chat/completions", json={
                "model": "cpu-model",
                "messages": [{"role": "user", "content": "hi"}],
            })
            assert chat_mod._active_requests == before
        finally:
            _state.shutting_down = orig_state
