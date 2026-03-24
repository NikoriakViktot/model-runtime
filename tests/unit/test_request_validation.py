"""
tests/unit/test_request_validation.py

Tests for request-body guardrails on the cpu_runtime inference endpoint:
  - Prompt too large → HTTP 413
  - max_tokens clamped to settings.max_total_tokens (never rejected)
  - Missing messages → HTTP 422
  - Queue full → HTTP 429
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch


def _client_with_loaded_engine():
    """Return a TestClient with engine in loaded state."""
    import cpu_runtime.inference as inf_mod
    from cpu_runtime.config import settings

    mock_engine = MagicMock()
    mock_engine.generate = AsyncMock(return_value=MagicMock(
        text="hi",
        prompt_tokens=3,
        completion_tokens=1,
        finish_reason="stop",
        model=settings.model_alias,
    ))
    inf_mod.engine = mock_engine
    inf_mod.load_state = "loaded"

    from cpu_runtime.app import app
    return TestClient(app, raise_server_exceptions=False), mock_engine


class TestPromptSizeGuard:
    """Requests with oversized prompts must be rejected with 413."""

    def test_413_when_prompt_exceeds_limit(self):
        from cpu_runtime.config import settings
        client, _ = _client_with_loaded_engine()

        # Build a prompt that slightly exceeds the limit
        big_content = "x" * (settings.max_prompt_chars + 1)
        resp = client.post("/v1/chat/completions", json={
            "model": "cpu-model",
            "messages": [{"role": "user", "content": big_content}],
        })
        assert resp.status_code == 413
        assert "too large" in resp.json()["detail"].lower()

    def test_200_when_prompt_exactly_at_limit(self):
        """Prompt exactly at the limit must succeed."""
        from cpu_runtime.config import settings
        client, _ = _client_with_loaded_engine()

        exact_content = "x" * settings.max_prompt_chars
        resp = client.post("/v1/chat/completions", json={
            "model": "cpu-model",
            "messages": [{"role": "user", "content": exact_content}],
        })
        assert resp.status_code == 200

    def test_413_multi_message_total_exceeds_limit(self):
        """Total across all messages must be checked."""
        from cpu_runtime.config import settings
        client, _ = _client_with_loaded_engine()

        half = settings.max_prompt_chars // 2 + 1
        resp = client.post("/v1/chat/completions", json={
            "model": "cpu-model",
            "messages": [
                {"role": "system", "content": "x" * half},
                {"role": "user", "content": "x" * half},
            ],
        })
        assert resp.status_code == 413


class TestMaxTokensClamping:
    """max_tokens beyond the limit must be clamped, not rejected."""

    def test_max_tokens_clamped_not_rejected(self):
        from cpu_runtime.config import settings
        client, mock_engine = _client_with_loaded_engine()

        oversized = settings.max_total_tokens * 10
        resp = client.post("/v1/chat/completions", json={
            "model": "cpu-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": oversized,
        })
        # Request should succeed (clamped silently)
        assert resp.status_code == 200

        # The engine was called with clamped max_tokens
        call_args = mock_engine.generate.call_args
        gen_req = call_args[0][0]
        assert gen_req.max_tokens <= settings.max_total_tokens

    def test_max_tokens_default_used_when_omitted(self):
        from cpu_runtime.config import settings
        client, mock_engine = _client_with_loaded_engine()

        resp = client.post("/v1/chat/completions", json={
            "model": "cpu-model",
            "messages": [{"role": "user", "content": "hi"}],
        })
        assert resp.status_code == 200
        gen_req = mock_engine.generate.call_args[0][0]
        assert gen_req.max_tokens == min(
            settings.max_tokens_default, settings.max_total_tokens
        )


class TestMissingMessages:
    """Missing or empty messages field must return 422."""

    def test_422_when_messages_missing(self):
        client, _ = _client_with_loaded_engine()
        resp = client.post("/v1/chat/completions", json={"model": "cpu-model"})
        assert resp.status_code == 422

    def test_422_when_messages_empty(self):
        client, _ = _client_with_loaded_engine()
        resp = client.post("/v1/chat/completions", json={
            "model": "cpu-model",
            "messages": [],
        })
        assert resp.status_code == 422

    def test_422_when_body_not_json(self):
        client, _ = _client_with_loaded_engine()
        resp = client.post(
            "/v1/chat/completions",
            content=b"not json at all",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422


class TestQueueOverflow:
    """When the queue is at capacity, new requests must get HTTP 429."""

    def test_429_when_queue_full(self):
        from cpu_runtime import routes as _routes_pkg
        import cpu_runtime.routes.chat as chat_mod

        client, _ = _client_with_loaded_engine()

        from cpu_runtime.config import settings
        orig = chat_mod._active_requests
        try:
            chat_mod._active_requests = settings.max_queue_depth  # simulate full queue
            resp = client.post("/v1/chat/completions", json={
                "model": "cpu-model",
                "messages": [{"role": "user", "content": "hi"}],
            })
            assert resp.status_code == 429
            assert "Retry-After" in resp.headers
        finally:
            chat_mod._active_requests = orig
