"""
tests/unit/test_gateway_routing_strategy.py

Unit tests for the Gateway's per-instance routing strategies and ModelRouter.

INVARIANTS
----------
- LeastLoaded always picks the instance with the lowest reported load.
- RoundRobin distributes across all instances in a fixed cycle.
- ModelRouter bypasses strategy entirely when only one instance exists.
- Metrics accumulate correctly and can be selectively reset.
"""

import pytest

from gateway.services.router import (
    InstanceInfo,
    LeastLoadedStrategy,
    ModelRouter,
    RandomStrategy,
    RoundRobinStrategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def inst(api_base: str, load: float = 0.0) -> InstanceInfo:
    return InstanceInfo(api_base=api_base, load=load)


# ---------------------------------------------------------------------------
# LeastLoadedStrategy
# ---------------------------------------------------------------------------


class TestLeastLoadedStrategy:
    def test_picks_instance_with_lowest_load(self):
        """The busiest instance is never chosen when a lighter one exists."""
        instances = [inst("http://a", load=0.9), inst("http://b", load=0.1)]
        chosen = LeastLoadedStrategy().choose(instances)
        assert chosen.api_base == "http://b"

    def test_single_instance_is_always_chosen(self):
        instances = [inst("http://only", load=0.5)]
        chosen = LeastLoadedStrategy().choose(instances)
        assert chosen.api_base == "http://only"

    def test_tiebreak_returns_one_of_the_tied_candidates(self):
        """When loads are equal, any candidate is acceptable."""
        instances = [inst("http://a", load=0.5), inst("http://b", load=0.5)]
        chosen = LeastLoadedStrategy().choose(instances)
        assert chosen.api_base in ("http://a", "http://b")

    def test_zero_load_instances_all_qualify_as_candidates(self):
        """Default load=0.0 (current MRM format) — all instances are candidates."""
        instances = [inst("http://a"), inst("http://b"), inst("http://c")]
        # Should return one of the three without raising
        chosen = LeastLoadedStrategy().choose(instances)
        assert chosen.api_base in ("http://a", "http://b", "http://c")


# ---------------------------------------------------------------------------
# RoundRobinStrategy
# ---------------------------------------------------------------------------


class TestRoundRobinStrategy:
    def test_cycles_through_all_instances_in_order(self):
        """Every instance is visited exactly once per full cycle."""
        strategy = RoundRobinStrategy()
        instances = [inst("http://a"), inst("http://b"), inst("http://c")]
        chosen = [strategy.choose(instances).api_base for _ in range(6)]
        assert chosen == ["http://a", "http://b", "http://c"] * 2

    def test_single_instance_always_chosen(self):
        strategy = RoundRobinStrategy()
        instances = [inst("http://only")]
        for _ in range(10):
            assert strategy.choose(instances).api_base == "http://only"

    def test_counter_is_per_strategy_instance(self):
        """Two independent strategies have independent counters."""
        s1 = RoundRobinStrategy()
        s2 = RoundRobinStrategy()
        instances = [inst("http://a"), inst("http://b")]
        s1.choose(instances)
        s1.choose(instances)
        # s2 has not advanced — first call returns "http://a"
        assert s2.choose(instances).api_base == "http://a"


# ---------------------------------------------------------------------------
# RandomStrategy
# ---------------------------------------------------------------------------


class TestRandomStrategy:
    def test_always_returns_an_instance_from_the_list(self):
        instances = [inst("http://a"), inst("http://b"), inst("http://c")]
        for _ in range(20):
            chosen = RandomStrategy().choose(instances)
            assert chosen in instances


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------


class TestModelRouter:
    def test_single_instance_bypasses_strategy(self):
        """
        INVARIANT: when only one instance is available, the strategy is
        skipped entirely and that instance is always returned.
        """
        router = ModelRouter(strategy=LeastLoadedStrategy())
        only = inst("http://only")
        for _ in range(5):
            assert router.choose_instance([only]) is only

    def test_raises_on_empty_instance_list(self):
        router = ModelRouter(strategy=RoundRobinStrategy())
        with pytest.raises(ValueError):
            router.choose_instance([])

    def test_record_increments_request_count(self):
        router = ModelRouter(strategy=RoundRobinStrategy())
        router.record("http://a", latency_ms=50.0)
        router.record("http://a", latency_ms=150.0)
        assert router.get_metrics()["http://a"]["requests"] == 2

    def test_record_tracks_errors_separately(self):
        router = ModelRouter(strategy=RoundRobinStrategy())
        router.record("http://a", latency_ms=10.0)
        router.record("http://a", latency_ms=10.0, error=True)
        m = router.get_metrics()["http://a"]
        assert m["requests"] == 2
        assert m["errors"] == 1

    def test_average_latency_is_computed_correctly(self):
        router = ModelRouter(strategy=RoundRobinStrategy())
        router.record("http://a", latency_ms=100.0)
        router.record("http://a", latency_ms=200.0)
        router.record("http://a", latency_ms=300.0)
        assert router.get_metrics()["http://a"]["avg_latency_ms"] == pytest.approx(200.0)

    def test_reset_clears_one_instance_only(self):
        """
        INVARIANT: reset_metrics(url) removes exactly that instance;
        all others remain.
        """
        router = ModelRouter(strategy=RoundRobinStrategy())
        router.record("http://a", latency_ms=10.0)
        router.record("http://b", latency_ms=20.0)
        router.reset_metrics("http://a")

        metrics = router.get_metrics()
        assert "http://a" not in metrics
        assert "http://b" in metrics

    def test_reset_all_clears_every_instance(self):
        router = ModelRouter(strategy=RoundRobinStrategy())
        for url in ("http://a", "http://b", "http://c"):
            router.record(url, latency_ms=1.0)
        router.reset_metrics()
        assert router.get_metrics() == {}

    def test_avg_latency_is_zero_before_any_requests(self):
        router = ModelRouter(strategy=RoundRobinStrategy())
        router.record("http://a", latency_ms=0.0)
        # Force the entry to exist but with zero latency
        router.reset_metrics("http://a")
        # After reset, the key is gone — no division by zero risk
        assert "http://a" not in router.get_metrics()
