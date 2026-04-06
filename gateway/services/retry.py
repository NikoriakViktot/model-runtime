"""
gateway/services/retry.py

Retry-with-budget for upstream proxy calls.

Contract
--------
- Max 1 retry per request (max_retries=1 default).
- Retryable errors:   HTTP 503, 504, connection errors, timeouts.
- Non-retryable:      HTTP 4xx (including 429 — respects backpressure).
- Jitter:             uniform random delay in [jitter_min_ms, jitter_max_ms].
- No recursive calls: uses a plain loop to enforce the retry budget.
- on_retry callback:  optional; called with (attempt, exc) so callers can
                      record circuit-breaker signals for each failed attempt.
"""
from __future__ import annotations

import asyncio
import random
from typing import Any, Awaitable, Callable

import structlog

log = structlog.get_logger(__name__)

# HTTP status codes that are safe to retry.
_RETRYABLE_STATUS: frozenset[int] = frozenset({503, 504})


async def call_with_retry(
    fn: Callable[[], Awaitable[Any]],
    *,
    max_retries: int = 1,
    jitter_min_ms: int = 50,
    jitter_max_ms: int = 150,
    request_id: str = "",
    on_retry: Callable[[int, Exception], None] | None = None,
) -> Any:
    """
    Call *fn* and retry up to *max_retries* times on retryable failures.

    Args:
        fn:             Zero-argument async callable performing one upstream call.
        max_retries:    Extra attempts beyond the first (default 1 → 2 total).
        jitter_min_ms:  Lower bound of the random inter-retry delay (ms).
        jitter_max_ms:  Upper bound of the random inter-retry delay (ms).
        request_id:     Included in retry log lines for traceability.
        on_retry:       Called just before sleeping; receives (attempt, exc).
                        Use to record circuit-breaker failures per attempt.

    Returns:
        Return value of the first successful *fn* call.

    Raises:
        The last exception when all attempts fail, or immediately for
        non-retryable errors (4xx, 429).
    """
    import httpx

    from gateway.services.proxy import UpstreamError

    for attempt in range(1, max_retries + 2):  # up to max_retries+1 total calls
        try:
            return await fn()

        except (UpstreamError, httpx.ConnectError, httpx.TimeoutException) as exc:
            status: int = getattr(exc, "status_code", 0) or 0

            # Client errors and backpressure signals must never be retried.
            if 400 <= status < 500:
                raise

            # Budget exhausted — re-raise the last error.
            if attempt > max_retries:
                raise

            delay_ms = random.randint(jitter_min_ms, jitter_max_ms)

            if on_retry is not None:
                try:
                    on_retry(attempt, exc)
                except Exception:
                    pass  # never let the callback crash the retry loop

            log.warning(
                "upstream_retry",
                request_id=request_id,
                attempt=attempt,
                max_retries=max_retries,
                delay_ms=delay_ms,
                status_code=status,
                reason=str(exc)[:200],
            )

            await asyncio.sleep(delay_ms / 1000.0)
