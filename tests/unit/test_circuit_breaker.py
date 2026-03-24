"""
tests/unit/test_circuit_breaker.py

Unit tests for the CircuitBreaker state machine.

INVARIANTS UNDER TEST
---------------------
1. Opens after N consecutive errors
2. Stays CLOSED on scattered errors that never reach threshold
3. Transitions to HALF_OPEN after cooldown elapses
4. Closes after a successful probe in HALF_OPEN state
5. Reopens (resets timer) on failed probe in HALF_OPEN state
6. ModelRouter excludes OPEN instances from routing
7. ModelRouter falls back to full list when ALL instances are OPEN
"""

from __future__ import annotations

import pytest

from gateway.services.circuit_breaker import CircuitBreaker
from gateway.services.router import InstanceInfo, LeastLoadedStrategy, ModelRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cb(threshold: int = 3, cooldown: float = 30.0, clock=None) -> CircuitBreaker:
    """Create a CircuitBreaker with a small threshold for fast tests."""
    return CircuitBreaker(error_threshold=threshold, cooldown_sec=cooldown, _clock=clock)


def _fake_clock(start: float = 0.0):
    """Return a mutable-clock callable and a setter."""
    t = [start]

    def clock() -> float:
        return t[0]

    def advance(seconds: float) -> None:
        t[0] += seconds

    return clock, advance


# ---------------------------------------------------------------------------
# 1. Opens after threshold consecutive errors
# ---------------------------------------------------------------------------


class TestCircuitBreakerOpens:
    def test_starts_closed(self):
        cb = _cb()
        assert cb.state == CircuitBreaker.CLOSED
        assert cb.is_available

    def test_stays_closed_below_threshold(self):
        cb = _cb(threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.CLOSED
        assert cb.is_available

    def test_opens_at_threshold(self):
        cb = _cb(threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN
        assert not cb.is_available

    def test_success_resets_consecutive_counter(self):
        """Scattered errors that never reach threshold do not open the circuit."""
        cb = _cb(threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # resets counter
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        # Only 3 consecutive failures since last success — below threshold
        assert cb.state == CircuitBreaker.CLOSED


# ---------------------------------------------------------------------------
# 2. OPEN → HALF_OPEN after cooldown
# ---------------------------------------------------------------------------


class TestHalfOpen:
    def test_open_before_cooldown(self):
        clock, advance = _fake_clock()
        cb = _cb(threshold=1, cooldown=30.0, clock=clock)
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN

        advance(29.9)
        assert cb.state == CircuitBreaker.OPEN

    def test_half_open_after_cooldown(self):
        clock, advance = _fake_clock()
        cb = _cb(threshold=1, cooldown=30.0, clock=clock)
        cb.record_failure()

        advance(30.0)
        assert cb.state == CircuitBreaker.HALF_OPEN
        assert cb.is_available  # probes are allowed through

    def test_is_available_in_half_open(self):
        clock, advance = _fake_clock()
        cb = _cb(threshold=1, cooldown=10.0, clock=clock)
        cb.record_failure()
        advance(10.0)
        assert cb.is_available


# ---------------------------------------------------------------------------
# 3. HALF_OPEN → CLOSED on success
# ---------------------------------------------------------------------------


class TestProbeSuccess:
    def test_closes_after_successful_probe(self):
        clock, advance = _fake_clock()
        cb = _cb(threshold=1, cooldown=10.0, clock=clock)
        cb.record_failure()
        advance(10.0)
        assert cb.state == CircuitBreaker.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitBreaker.CLOSED
        assert cb.is_available

    def test_success_resets_error_counter(self):
        clock, advance = _fake_clock()
        cb = _cb(threshold=2, cooldown=10.0, clock=clock)
        cb.record_failure()
        cb.record_failure()  # opens
        advance(10.0)        # HALF_OPEN
        cb.record_success()  # closes, resets counter

        # Now we need 2 more failures to reopen
        cb.record_failure()
        assert cb.state == CircuitBreaker.CLOSED  # only 1 consecutive failure
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN    # 2nd → reopen


# ---------------------------------------------------------------------------
# 4. HALF_OPEN → OPEN on failed probe
# ---------------------------------------------------------------------------


class TestProbeFailed:
    def test_reopens_on_failed_probe(self):
        clock, advance = _fake_clock()
        cb = _cb(threshold=1, cooldown=10.0, clock=clock)
        cb.record_failure()  # opens at t=0
        advance(10.0)        # HALF_OPEN
        cb.record_failure()  # probe fails → reopen at t=10
        assert cb.state == CircuitBreaker.OPEN

    def test_failed_probe_resets_cooldown_timer(self):
        """After a failed probe, the cooldown starts fresh."""
        clock, advance = _fake_clock()
        cb = _cb(threshold=1, cooldown=10.0, clock=clock)
        cb.record_failure()  # opens at t=0
        advance(10.0)        # HALF_OPEN
        cb.record_failure()  # reopens at t=10

        advance(5.0)         # only 5s since re-open → still OPEN
        assert cb.state == CircuitBreaker.OPEN

        advance(5.0)         # 10s since re-open → HALF_OPEN again
        assert cb.state == CircuitBreaker.HALF_OPEN


# ---------------------------------------------------------------------------
# 5. ModelRouter excludes OPEN instances
# ---------------------------------------------------------------------------


class TestRouterCircuitIntegration:
    def test_router_skips_open_circuit(self):
        """
        INVARIANT: the router must never route to an OPEN instance
        when a healthy alternative is available.
        """
        router = ModelRouter(
            strategy=LeastLoadedStrategy(),
            circuit_error_threshold=1,
            circuit_cooldown_sec=9999.0,
        )
        instances = [
            InstanceInfo(api_base="http://bad", load=0.0),
            InstanceInfo(api_base="http://good", load=0.0),
        ]

        # Open the circuit for "bad"
        router.record("http://bad", latency_ms=10.0, error=True)

        # All subsequent requests must go to "good"
        chosen = [router.choose_instance(instances).api_base for _ in range(20)]
        assert all(c == "http://good" for c in chosen), (
            "Router must not route to an instance with an OPEN circuit breaker"
        )

    def test_router_falls_back_when_all_circuits_open(self):
        """
        INVARIANT: when every instance has an OPEN circuit, the router must
        fall back to the full list rather than raising.
        """
        router = ModelRouter(
            strategy=LeastLoadedStrategy(),
            circuit_error_threshold=1,
            circuit_cooldown_sec=9999.0,
        )
        instances = [
            InstanceInfo(api_base="http://a", load=0.0),
            InstanceInfo(api_base="http://b", load=0.0),
        ]

        # Open both circuits
        router.record("http://a", latency_ms=10.0, error=True)
        router.record("http://b", latency_ms=10.0, error=True)

        # Must not raise — returns one of the two
        chosen = router.choose_instance(instances)
        assert chosen.api_base in ("http://a", "http://b")

    def test_circuit_closes_after_successful_probe(self):
        """
        INVARIANT: after a failed instance recovers, the circuit closes and
        traffic resumes normally.
        """
        clock, advance = _fake_clock()
        router = ModelRouter(
            strategy=LeastLoadedStrategy(),
            circuit_error_threshold=1,
            circuit_cooldown_sec=10.0,
        )
        # Inject the clock into the circuit breaker for "bad"
        router._circuits["http://bad"] = CircuitBreaker(
            error_threshold=1, cooldown_sec=10.0, _clock=clock
        )

        instances = [
            InstanceInfo(api_base="http://bad", load=0.0),
            InstanceInfo(api_base="http://good", load=0.0),
        ]

        # Open the circuit
        router.record("http://bad", latency_ms=1.0, error=True)
        # Update the circuit with the same clock instance
        router._circuits["http://bad"].record_failure()  # already called by record()

        # Wait out the cooldown → HALF_OPEN
        advance(10.0)
        assert router._circuits["http://bad"].state == CircuitBreaker.HALF_OPEN

        # Successful probe closes the circuit
        router._circuits["http://bad"].record_success()
        assert router._circuits["http://bad"].state == CircuitBreaker.CLOSED
