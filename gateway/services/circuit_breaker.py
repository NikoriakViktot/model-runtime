"""
gateway/services/circuit_breaker.py

Per-instance circuit breaker for the model router.

States
------
CLOSED    Normal operation.  All requests pass through.
OPEN      Instance is ejected.  Requests are immediately rejected (fail-fast).
          Transitions to HALF_OPEN after ``cooldown_sec`` elapses.
HALF_OPEN Probe state.  Requests are allowed through.  On the next recorded
          result the breaker either closes (success) or reopens (failure).

Transitions
-----------
CLOSED   → OPEN      after ``error_threshold`` consecutive errors
OPEN     → HALF_OPEN after ``cooldown_sec`` elapses (lazy, checked on access)
HALF_OPEN → CLOSED   on record_success()
HALF_OPEN → OPEN     on record_failure()  (cooldown timer resets)

Injectable clock
----------------
Pass ``_clock=<callable returning float>`` to override ``time.monotonic``.
Required for deterministic unit tests that need to control the passage of time
without actually sleeping.

Thread safety
-------------
All state is mutated only from the asyncio event loop (same as ModelRouter),
so no explicit locking is required.
"""

from __future__ import annotations

import time


class CircuitBreaker:
    """
    Tracks consecutive failures for one upstream instance and ejects it when
    a threshold is exceeded.

    Usage::

        cb = CircuitBreaker(error_threshold=5, cooldown_sec=30.0)

        if cb.is_available:
            try:
                result = await call_upstream()
                cb.record_success()
            except UpstreamError:
                cb.record_failure()
        else:
            raise ServiceUnavailable("Circuit is open")
    """

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(
        self,
        error_threshold: int = 5,
        cooldown_sec: float = 30.0,
        _clock=None,
    ) -> None:
        self._error_threshold = error_threshold
        self._cooldown_sec = cooldown_sec
        self._clock = _clock or time.monotonic

        # CLOSED while _opened_at is None.
        # OPEN or HALF_OPEN while _opened_at is set.
        self._opened_at: float | None = None
        self._consecutive_errors: int = 0

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        """
        Current state, computed lazily.

        The OPEN → HALF_OPEN transition happens here — no background task
        is needed.  Checking ``.state`` after the cooldown has elapsed will
        return HALF_OPEN and allow the next probe request through.
        """
        if self._opened_at is None:
            return self.CLOSED
        elapsed = self._clock() - self._opened_at
        if elapsed >= self._cooldown_sec:
            return self.HALF_OPEN
        return self.OPEN

    @property
    def is_available(self) -> bool:
        """True when requests should be forwarded (CLOSED or HALF_OPEN)."""
        return self.state != self.OPEN

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def record_success(self) -> None:
        """
        Record a successful upstream call.

        Resets the consecutive error counter and closes the circuit
        (regardless of the current state — CLOSED, OPEN, or HALF_OPEN).
        """
        self._consecutive_errors = 0
        self._opened_at = None  # → CLOSED

    def record_failure(self) -> None:
        """
        Record a failed upstream call.

        Increments the consecutive error counter.  Opens the circuit once
        the threshold is reached.  In HALF_OPEN state any failure reopens
        the circuit immediately (resetting the cooldown timer).
        """
        self._consecutive_errors += 1
        current = self.state
        if current == self.HALF_OPEN or self._consecutive_errors >= self._error_threshold:
            self._opened_at = self._clock()  # → OPEN (or reopen with fresh timer)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "consecutive_errors": self._consecutive_errors,
            "cooldown_sec": self._cooldown_sec,
        }
