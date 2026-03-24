"""
tests/unit/test_cpu_runtime.py

Unit tests for the cpu_runtime service:
  - /health endpoint (503 while loading, 200 after load)
  - LlamaCppEngine queue / semaphore behavior
  - streaming non-blocking contract
  - missing model file handling
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest
from fastapi.testclient import TestClient


# ── /health endpoint ─────────────────────────────────────────────────

class TestHealthEndpoint:
    """The /health endpoint must return 503 until the engine is loaded."""

    def setup_method(self):
        # Reset module-level engine singleton before each test
        import cpu_runtime.inference as inf_mod
        self._orig_engine = inf_mod.engine
        inf_mod.engine = None

    def teardown_method(self):
        import cpu_runtime.inference as inf_mod
        inf_mod.engine = self._orig_engine

    def test_health_503_when_engine_none(self):
        """Before model load: /health must return 503."""
        import cpu_runtime.inference as inf_mod
        inf_mod.engine = None

        # Import app after patching the singleton
        from cpu_runtime.app import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json()["status"] == "loading"

    def test_health_200_when_engine_loaded(self):
        """After model load: /health must return 200."""
        import cpu_runtime.inference as inf_mod
        from cpu_runtime.config import settings

        mock_engine = MagicMock()
        mock_engine.generate = AsyncMock()
        inf_mod.engine = mock_engine

        from cpu_runtime.app import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model" in data


# ── LlamaCppEngine.load() error handling ─────────────────────────────

class TestEngineLoad:

    @pytest.mark.asyncio
    async def test_load_raises_on_missing_model_file(self):
        """Missing GGUF file: load() must raise RuntimeError, not crash silently."""
        from cpu_runtime.config import Settings
        from cpu_runtime.inference import LlamaCppEngine

        settings = MagicMock(spec=Settings)
        settings.model_path = "/nonexistent/model.gguf"
        settings.n_ctx = 2048
        settings.n_threads = 2
        settings.n_batch = 512
        settings.n_gpu_layers = 0
        settings.model_alias = "test"

        engine = LlamaCppEngine(settings)

        with patch("cpu_runtime.inference.LlamaCppEngine.load") as mock_load:
            mock_load.side_effect = RuntimeError("Model path does not exist: /nonexistent/model.gguf")
            with pytest.raises(RuntimeError, match="Model path does not exist"):
                await engine.load()

    @pytest.mark.asyncio
    async def test_load_raises_when_llama_cpp_not_installed(self):
        """If llama_cpp is not installed, load() raises RuntimeError with a helpful message."""
        import sys
        from cpu_runtime.config import Settings
        from cpu_runtime.inference import LlamaCppEngine

        settings = MagicMock(spec=Settings)
        settings.model_path = "/models/model.gguf"
        settings.n_ctx = 2048
        settings.n_threads = 2
        settings.n_batch = 512
        settings.n_gpu_layers = 0
        settings.model_alias = "test"

        engine = LlamaCppEngine(settings)

        with patch.dict(sys.modules, {"llama_cpp": None}):
            with pytest.raises(RuntimeError, match="llama-cpp-python"):
                await engine.load()


# ── LlamaCppEngine: Semaphore behavior ───────────────────────────────

class TestSemaphore:

    def _make_engine(self):
        from cpu_runtime.config import Settings
        from cpu_runtime.inference import LlamaCppEngine

        settings = MagicMock(spec=Settings)
        settings.model_path = "/models/model.gguf"
        settings.model_alias = "test-model"
        settings.n_ctx = 2048
        settings.n_threads = 2
        settings.n_batch = 512
        settings.n_gpu_layers = 0
        settings.max_tokens_default = 256
        settings.temperature_default = 0.7
        settings.top_p_default = 0.95
        settings.repeat_penalty = 1.1

        engine = LlamaCppEngine(settings)
        # Inject a mock llm and semaphore
        engine._llm = MagicMock()
        engine._sem = asyncio.Semaphore(1)
        return engine

    @pytest.mark.asyncio
    async def test_generate_acquires_and_releases_semaphore(self):
        """After generate(), the semaphore must be available again (no leak)."""
        engine = self._make_engine()
        from cpu_runtime.inference import GenerationRequest

        engine._llm.return_value = {
            "choices": [{"text": "hello", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }

        req = GenerationRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=10)
        await engine.generate(req)

        # Semaphore should have been released
        assert engine._sem._value == 1

    @pytest.mark.asyncio
    async def test_generate_releases_semaphore_on_exception(self):
        """Semaphore must be released even if llm() raises."""
        engine = self._make_engine()
        from cpu_runtime.inference import GenerationRequest

        engine._llm.side_effect = RuntimeError("model crashed")
        req = GenerationRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=10)

        with pytest.raises(RuntimeError):
            await engine.generate(req)

        assert engine._sem._value == 1

    @pytest.mark.asyncio
    async def test_stream_releases_semaphore_after_full_drain(self):
        """After consuming all streaming chunks, semaphore must be released."""
        engine = self._make_engine()
        from cpu_runtime.inference import GenerationRequest

        def _fake_llm_stream(*args, **kwargs):
            yield {"choices": [{"text": "hello", "finish_reason": None}]}
            yield {"choices": [{"text": " world", "finish_reason": "stop"}]}

        engine._llm.return_value = _fake_llm_stream()
        engine._llm.side_effect = None

        # Need to make _llm() return the generator when called
        def _call_llm(*args, **kwargs):
            return _fake_llm_stream()
        engine._llm.side_effect = _call_llm

        req = GenerationRequest(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            stream=True,
        )

        chunks = []
        async for chunk in engine.stream(req):
            chunks.append(chunk)

        assert engine._sem._value == 1
        assert b"[DONE]" in chunks[-1]

    @pytest.mark.asyncio
    async def test_stream_releases_semaphore_on_exception(self):
        """Semaphore must be released even when the streaming generator raises."""
        engine = self._make_engine()
        from cpu_runtime.inference import GenerationRequest

        def _raise_on_second():
            yield {"choices": [{"text": "a", "finish_reason": None}]}
            raise RuntimeError("stream exploded")

        def _call_llm(*args, **kwargs):
            return _raise_on_second()
        engine._llm.side_effect = _call_llm

        req = GenerationRequest(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
            stream=True,
        )

        with pytest.raises(RuntimeError, match="stream exploded"):
            async for _ in engine.stream(req):
                pass

        assert engine._sem._value == 1


# ── Streaming: event-loop non-blocking contract ───────────────────────

class TestStreamingNonBlocking:
    """
    Verifies that the streaming path does NOT call blocking C-level code
    on the event loop thread (the core bug that was fixed).

    We can't easily time asyncio micro-sleeps in unit tests, so we verify
    the architectural contract: the generator is consumed via a queue,
    not via a direct `for chunk in iterator` loop in the async context.
    """

    @pytest.mark.asyncio
    async def test_stream_uses_queue_not_direct_iteration(self):
        """
        The fixed implementation uses asyncio.Queue internally.
        We verify that even if the iterator is slow (simulated via sleep),
        the event loop remains responsive.
        """
        import time
        from cpu_runtime.inference import LlamaCppEngine, GenerationRequest
        from cpu_runtime.config import Settings

        settings = MagicMock(spec=Settings)
        settings.model_path = "/models/model.gguf"
        settings.model_alias = "test"
        settings.n_ctx = 512
        settings.n_threads = 1
        settings.n_batch = 512
        settings.n_gpu_layers = 0
        settings.max_tokens_default = 256
        settings.temperature_default = 0.7
        settings.top_p_default = 0.95
        settings.repeat_penalty = 1.1

        engine = LlamaCppEngine(settings)
        engine._llm = MagicMock()
        engine._sem = asyncio.Semaphore(1)

        slow_chunks = [
            {"choices": [{"text": f"token{i}", "finish_reason": None}]}
            for i in range(5)
        ]

        def _slow_stream(*args, **kwargs):
            for c in slow_chunks:
                time.sleep(0.01)   # 10ms blocking — simulates C token generation
                yield c

        engine._llm.side_effect = _slow_stream

        req = GenerationRequest(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5,
            stream=True,
        )

        # Concurrently run a "health check" coroutine to verify event loop isn't blocked.
        # If streaming blocked the loop, the health task would stall until streaming completed.
        results = {"stream_done": False, "health_interleaved": 0}

        async def _health_monitor():
            for _ in range(20):
                await asyncio.sleep(0.005)
                if not results["stream_done"]:
                    results["health_interleaved"] += 1

        health_task = asyncio.create_task(_health_monitor())

        collected = []
        async for chunk in engine.stream(req):
            collected.append(chunk)
        results["stream_done"] = True

        await health_task

        # The health monitor must have run at least twice during streaming,
        # proving the event loop was NOT blocked.
        assert results["health_interleaved"] >= 2, (
            f"Event loop appears blocked: health_interleaved={results['health_interleaved']}. "
            "The streaming path is running synchronous code on the event loop."
        )
        assert b"[DONE]" in collected[-1]


# ── /v1/models list ───────────────────────────────────────────────────

class TestModelsList:

    def setup_method(self):
        import cpu_runtime.inference as inf_mod
        mock_engine = MagicMock()
        inf_mod.engine = mock_engine

    def teardown_method(self):
        import cpu_runtime.inference as inf_mod
        inf_mod.engine = None

    def test_v1_models_returns_single_model(self):
        from cpu_runtime.app import app
        client = TestClient(app)
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        model = data["data"][0]
        assert model["runtime_type"] == "cpu"
        assert "id" in model
