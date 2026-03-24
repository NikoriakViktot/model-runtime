"""
tests/unit/test_retry.py

Tests for the gateway retry-with-budget logic.
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestRetryOnTransientErrors:

    @pytest.mark.asyncio
    async def test_retries_on_503(self):
        """A 503 UpstreamError should trigger exactly one retry."""
        from gateway.services.retry import call_with_retry
        from gateway.services.proxy import UpstreamError

        calls = []

        async def fn():
            calls.append(1)
            if len(calls) == 1:
                raise UpstreamError("unavailable", status_code=503)
            return {"ok": True}

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await call_with_retry(fn, max_retries=1)

        assert len(calls) == 2
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_retries_on_504(self):
        """A 504 UpstreamError should trigger exactly one retry."""
        from gateway.services.retry import call_with_retry
        from gateway.services.proxy import UpstreamError

        calls = []

        async def fn():
            calls.append(1)
            if len(calls) == 1:
                raise UpstreamError("timeout", status_code=504)
            return {"ok": True}

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await call_with_retry(fn, max_retries=1)

        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_429(self):
        """A 429 must NOT be retried — it's a backpressure signal."""
        from gateway.services.retry import call_with_retry
        from gateway.services.proxy import UpstreamError

        calls = []

        async def fn():
            calls.append(1)
            raise UpstreamError("too many requests", status_code=429)

        with pytest.raises(UpstreamError) as exc_info:
            await call_with_retry(fn, max_retries=1)

        assert len(calls) == 1
        assert exc_info.value.status_code == 429

    @pytest.mark.asyncio
    async def test_no_retry_on_400(self):
        """A 400 client error must NOT be retried."""
        from gateway.services.retry import call_with_retry
        from gateway.services.proxy import UpstreamError

        calls = []

        async def fn():
            calls.append(1)
            raise UpstreamError("bad request", status_code=400)

        with pytest.raises(UpstreamError):
            await call_with_retry(fn, max_retries=1)

        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_no_retry_on_404(self):
        """A 404 must NOT be retried."""
        from gateway.services.retry import call_with_retry
        from gateway.services.proxy import UpstreamError

        calls = []

        async def fn():
            calls.append(1)
            raise UpstreamError("not found", status_code=404)

        with pytest.raises(UpstreamError):
            await call_with_retry(fn, max_retries=1)

        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_max_retries_respected(self):
        """With max_retries=1, total call count = 2 regardless of outcome."""
        from gateway.services.retry import call_with_retry
        from gateway.services.proxy import UpstreamError

        calls = []

        async def fn():
            calls.append(1)
            raise UpstreamError("always fails", status_code=503)

        with pytest.raises(UpstreamError):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await call_with_retry(fn, max_retries=1)

        assert len(calls) == 2  # 1 original + 1 retry

    @pytest.mark.asyncio
    async def test_max_retries_zero_means_no_retry(self):
        """max_retries=0 means one attempt only."""
        from gateway.services.retry import call_with_retry
        from gateway.services.proxy import UpstreamError

        calls = []

        async def fn():
            calls.append(1)
            raise UpstreamError("fail", status_code=503)

        with pytest.raises(UpstreamError):
            await call_with_retry(fn, max_retries=0)

        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_jitter_applied_between_retries(self):
        """asyncio.sleep must be called with a value in [jitter_min, jitter_max]."""
        from gateway.services.retry import call_with_retry
        from gateway.services.proxy import UpstreamError

        sleep_args = []
        orig_sleep = asyncio.sleep

        async def _mock_sleep(secs):
            sleep_args.append(secs)

        calls = []

        async def fn():
            calls.append(1)
            if len(calls) == 1:
                raise UpstreamError("fail", status_code=503)
            return "ok"

        with patch("asyncio.sleep", side_effect=_mock_sleep):
            await call_with_retry(
                fn,
                max_retries=1,
                jitter_min_ms=50,
                jitter_max_ms=150,
            )

        assert len(sleep_args) == 1
        assert 0.05 <= sleep_args[0] <= 0.15

    @pytest.mark.asyncio
    async def test_on_retry_callback_called(self):
        """on_retry callback must be called for each failed attempt."""
        from gateway.services.retry import call_with_retry
        from gateway.services.proxy import UpstreamError

        cb_calls = []

        def on_retry(attempt, exc):
            cb_calls.append((attempt, exc))

        calls = []

        async def fn():
            calls.append(1)
            if len(calls) == 1:
                raise UpstreamError("fail", status_code=503)
            return "ok"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await call_with_retry(fn, max_retries=1, on_retry=on_retry)

        assert len(cb_calls) == 1
        attempt, exc = cb_calls[0]
        assert attempt == 1


class TestRetryWithConnectionErrors:

    @pytest.mark.asyncio
    async def test_retries_on_connect_error(self):
        """httpx.ConnectError should be retried."""
        import httpx
        from gateway.services.retry import call_with_retry

        calls = []

        async def fn():
            calls.append(1)
            if len(calls) == 1:
                raise httpx.ConnectError("connection refused")
            return "ok"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await call_with_retry(fn, max_retries=1)

        assert len(calls) == 2
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self):
        """httpx.TimeoutException should be retried."""
        import httpx
        from gateway.services.retry import call_with_retry

        calls = []

        async def fn():
            calls.append(1)
            if len(calls) == 1:
                raise httpx.ReadTimeout("read timeout")
            return "ok"

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await call_with_retry(fn, max_retries=1)

        assert len(calls) == 2
