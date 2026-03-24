"""
tests/unit/test_load_shedding.py

Tests for adaptive load shedding in cpu_runtime:
  - LoadShedder unit tests (latency + RAM)
  - Integration: RAM low → HTTP 503
  - Integration: dynamic concurrency reduction → HTTP 429 earlier
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# LoadShedder unit tests
# ---------------------------------------------------------------------------

class TestLoadShedderLatency:

    def test_avg_latency_zero_with_no_data(self):
        from cpu_runtime.load_shedder import LoadShedder
        s = LoadShedder()
        assert s.avg_latency_ms == 0.0

    def test_avg_latency_computed_correctly(self):
        from cpu_runtime.load_shedder import LoadShedder
        s = LoadShedder()
        s.record_latency(1000.0)
        s.record_latency(3000.0)
        assert s.avg_latency_ms == 2000.0

    def test_window_is_bounded(self):
        from cpu_runtime.load_shedder import LoadShedder
        s = LoadShedder(window_size=3)
        for i in range(10):
            s.record_latency(float(i * 1000))
        # Only last 3 values: 7000, 8000, 9000 → avg = 8000
        assert s.avg_latency_ms == pytest.approx(8000.0)

    def test_effective_max_no_shedding_below_threshold(self):
        from cpu_runtime.load_shedder import LoadShedder
        s = LoadShedder()
        s.record_latency(2000.0)  # below 5000ms threshold
        assert s.effective_max_requests(base=8, threshold_ms=5000.0) == 8

    def test_effective_max_no_shedding_at_threshold(self):
        from cpu_runtime.load_shedder import LoadShedder
        s = LoadShedder()
        s.record_latency(5000.0)  # exactly at threshold
        assert s.effective_max_requests(base=8, threshold_ms=5000.0) == 8

    def test_effective_max_shedding_above_threshold(self):
        from cpu_runtime.load_shedder import LoadShedder
        s = LoadShedder()
        for _ in range(5):
            s.record_latency(10000.0)  # 2× threshold → half capacity
        result = s.effective_max_requests(base=8, threshold_ms=5000.0)
        assert result == 4

    def test_effective_max_floor_at_one(self):
        from cpu_runtime.load_shedder import LoadShedder
        s = LoadShedder()
        for _ in range(5):
            s.record_latency(100000.0)  # extreme latency
        assert s.effective_max_requests(base=8, threshold_ms=5000.0) == 1

    def test_effective_max_no_data_returns_base(self):
        from cpu_runtime.load_shedder import LoadShedder
        s = LoadShedder()
        assert s.effective_max_requests(base=16, threshold_ms=5000.0) == 16

    def test_disabled_threshold_returns_base(self):
        from cpu_runtime.load_shedder import LoadShedder
        s = LoadShedder()
        for _ in range(5):
            s.record_latency(99999.0)
        # threshold_ms=0 means disabled
        assert s.effective_max_requests(base=8, threshold_ms=0.0) == 8


class TestLoadShedderRAM:

    def test_ram_cached(self):
        """Second call within TTL must return cached value without re-reading."""
        from cpu_runtime.load_shedder import LoadShedder
        s = LoadShedder(ram_check_interval_sec=60.0)
        with patch("cpu_runtime.load_shedder._read_free_ram_mb", return_value=1024) as mock_read:
            s.check_ram()
            s.check_ram()
        assert mock_read.call_count == 1  # only one /proc read

    def test_ram_cache_expires(self):
        """After TTL, a fresh read is performed."""
        from cpu_runtime.load_shedder import LoadShedder
        import time
        s = LoadShedder(ram_check_interval_sec=0.01)  # 10ms TTL
        with patch("cpu_runtime.load_shedder._read_free_ram_mb", return_value=512) as mock_read:
            s.check_ram()
            time.sleep(0.02)  # expire the cache
            s.check_ram()
        assert mock_read.call_count == 2

    def test_returns_minus_one_when_unreadable(self):
        from cpu_runtime.load_shedder import LoadShedder
        s = LoadShedder()
        with patch("cpu_runtime.load_shedder._read_free_ram_mb", return_value=-1):
            assert s.check_ram() == -1


# ---------------------------------------------------------------------------
# Integration: RAM-based shedding
# ---------------------------------------------------------------------------

class TestRAMSheddingIntegration:

    def _client(self):
        import cpu_runtime.inference as inf_mod
        from cpu_runtime.config import settings
        mock_engine = MagicMock()
        mock_engine.generate = AsyncMock(return_value=MagicMock(
            text="ok", prompt_tokens=1, completion_tokens=1,
            finish_reason="stop", model=settings.model_alias,
        ))
        inf_mod.engine = mock_engine
        inf_mod.load_state = "loaded"
        from cpu_runtime.app import app
        return TestClient(app, raise_server_exceptions=False)

    def test_503_when_ram_critically_low(self):
        """When free RAM < min_free_ram_mb, requests get 503."""
        import cpu_runtime.load_shedder as ls_mod
        from cpu_runtime.config import settings

        client = self._client()
        with patch.object(ls_mod.shedder, "check_ram", return_value=1):
            resp = client.post("/v1/chat/completions", json={
                "model": "cpu-model",
                "messages": [{"role": "user", "content": "hi"}],
            })
        assert resp.status_code == 503
        assert "memory" in resp.json()["detail"].lower()

    def test_200_when_ram_above_threshold(self):
        """When free RAM is ample, requests proceed normally."""
        import cpu_runtime.load_shedder as ls_mod
        from cpu_runtime.config import settings

        client = self._client()
        with patch.object(ls_mod.shedder, "check_ram", return_value=settings.min_free_ram_mb + 1024):
            resp = client.post("/v1/chat/completions", json={
                "model": "cpu-model",
                "messages": [{"role": "user", "content": "hi"}],
            })
        assert resp.status_code == 200

    def test_200_when_ram_unreadable(self):
        """When /proc/meminfo is unavailable (-1), no shedding occurs."""
        import cpu_runtime.load_shedder as ls_mod

        client = self._client()
        with patch.object(ls_mod.shedder, "check_ram", return_value=-1):
            resp = client.post("/v1/chat/completions", json={
                "model": "cpu-model",
                "messages": [{"role": "user", "content": "hi"}],
            })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Integration: Dynamic concurrency shedding
# ---------------------------------------------------------------------------

class TestDynamicConcurrencyShedding:

    def test_effective_ceiling_used_for_429(self):
        """When dynamic concurrency is enabled and latency spikes,
        active_requests < max_queue_depth can still trigger 429."""
        import cpu_runtime.routes.chat as chat_mod
        import cpu_runtime.load_shedder as ls_mod
        import cpu_runtime.inference as inf_mod
        from cpu_runtime.config import settings

        inf_mod.engine = MagicMock()
        inf_mod.load_state = "loaded"

        orig = chat_mod._active_requests
        try:
            # Simulate 1 active request with an effective ceiling of 1
            chat_mod._active_requests = 1
            with patch.object(
                ls_mod.shedder,
                "effective_max_requests",
                return_value=1,  # ceiling = 1
            ):
                from cpu_runtime.app import app
                client = TestClient(app, raise_server_exceptions=False)
                resp = client.post("/v1/chat/completions", json={
                    "model": "cpu-model",
                    "messages": [{"role": "user", "content": "hi"}],
                })
            assert resp.status_code == 429
        finally:
            chat_mod._active_requests = orig
