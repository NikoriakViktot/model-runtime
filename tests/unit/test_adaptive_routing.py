"""
tests/unit/test_adaptive_routing.py

Unit tests for adaptive routing features: EWMA scoring, pool-relative
load balancing, hysteresis, and anomaly detection.

INVARIANTS UNDER TEST
---------------------
1. EWMA smooths transient latency spikes (doesn't over-react)
2. Pool-relative scoring distributes evenly across a uniform pool
3. Pool-relative scoring penalises an outlier vs the pool
4. Hysteresis prevents oscillation when instances are nearly equal
5. Hysteresis allows switching when gap exceeds threshold
6. Anomaly detected on sudden latency spike (vs baseline)
7. No anomaly on steady-state high latency (baseline catches up)
8. Global SLO snapshot aggregates p50/p95/p99 correctly
"""

from __future__ import annotations

import pytest

from gateway.services.router import (
    GlobalSLO,
    InstanceInfo,
    LeastLoadedStrategy,
    ModelRouter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def inst(api_base: str, load: float = 0.0) -> InstanceInfo:
    return InstanceInfo(api_base=api_base, load=load)


# ---------------------------------------------------------------------------
# 1. EWMA smooths transient spikes
# ---------------------------------------------------------------------------


class TestEWMASmoothing:
    def test_ewma_bootstraps_on_first_sample(self):
        router = ModelRouter(strategy=LeastLoadedStrategy())
        router.record("http://a", latency_ms=100.0)
        m = router._get_metrics("http://a")
        assert m.ewma_latency_ms == pytest.approx(100.0)

    def test_ewma_does_not_fully_adopt_single_spike(self):
        """
        INVARIANT: a single 10× spike must not move the EWMA to the spike value.
        With α=0.2, after 20 baseline samples the EWMA ≈ baseline; one spike
        of 10× moves it to ≈ baseline × 1.2, not spike.
        """
        router = ModelRouter(strategy=LeastLoadedStrategy())
        baseline = 100.0
        for _ in range(20):
            router.record("http://a", latency_ms=baseline)

        m = router._get_metrics("http://a")
        ewma_before = m.ewma_latency_ms
        assert ewma_before == pytest.approx(baseline, rel=0.01)

        router.record("http://a", latency_ms=1_000.0)  # spike
        ewma_after = m.ewma_latency_ms
        # Should be 0.2*1000 + 0.8*100 = 280, not 1000
        assert ewma_after < 400.0, (
            f"EWMA should smooth spikes: expected <400ms, got {ewma_after:.1f}ms"
        )


# ---------------------------------------------------------------------------
# 2. Pool-relative scoring: uniform pool
# ---------------------------------------------------------------------------


class TestPoolRelativeUniform:
    def test_equal_latency_instances_share_traffic_evenly(self):
        """
        INVARIANT: when all instances have the same EWMA latency, no instance
        receives a penalty and LeastLoaded distributes via its tie-breaking
        (random).  No single instance should get > 80% of traffic.
        """
        router = ModelRouter(strategy=LeastLoadedStrategy())
        urls = ["http://a", "http://b", "http://c"]

        # Warm up with identical latency
        for url in urls:
            for _ in range(20):
                router.record(url, latency_ms=100.0)

        instances = [inst(url) for url in urls]
        counts: dict[str, int] = {url: 0 for url in urls}
        N = 300

        for _ in range(N):
            chosen = router.choose_instance(instances)
            counts[chosen.api_base] += 1

        for url in urls:
            share = counts[url] / N
            assert share > 0.10, (
                f"{url} received {share:.0%} of traffic — expected even distribution"
            )


# ---------------------------------------------------------------------------
# 3. Pool-relative scoring: outlier is penalised
# ---------------------------------------------------------------------------


class TestPoolRelativeOutlier:
    def test_slow_outlier_receives_less_traffic(self):
        """
        INVARIANT: one instance that is 10× slower than the pool mean
        must receive less than 20% of subsequent routing decisions.
        """
        router = ModelRouter(strategy=LeastLoadedStrategy())

        for _ in range(30):
            router.record("http://fast-1", latency_ms=50.0)
            router.record("http://fast-2", latency_ms=50.0)
            router.record("http://slow", latency_ms=500.0)

        instances = [
            inst("http://fast-1"),
            inst("http://fast-2"),
            inst("http://slow"),
        ]

        N = 200
        counts: dict[str, int] = {"http://fast-1": 0, "http://fast-2": 0, "http://slow": 0}
        for _ in range(N):
            chosen = router.choose_instance(instances)
            counts[chosen.api_base] += 1

        slow_share = counts["http://slow"] / N
        assert slow_share < 0.20, (
            f"Slow outlier must receive <20% of traffic; got {slow_share:.0%}"
        )


# ---------------------------------------------------------------------------
# 4. Hysteresis: small gap does not trigger a switch
# ---------------------------------------------------------------------------


class TestHysteresis:
    def test_hysteresis_prevents_switching_on_small_gap(self):
        """
        INVARIANT: if the current instance's effective load and the best
        alternative differ by less than HYSTERESIS_GAP (0.05), the router
        must stay with the current instance.

        We do this by giving both instances identical history so their effective
        loads are equal, then asking the router 20 times.  Due to hysteresis,
        once a model is pinned it should not oscillate.
        """
        router = ModelRouter(strategy=LeastLoadedStrategy())

        # Both instances have identical latency — no penalty difference
        for _ in range(20):
            router.record("http://a", latency_ms=100.0)
            router.record("http://b", latency_ms=100.0)

        instances = [inst("http://a"), inst("http://b")]

        # First call establishes sticky assignment
        first = router.choose_instance(instances, model_id="m1")
        sticky = first.api_base

        # All subsequent calls with same model_id must stay on the same instance
        for _ in range(19):
            chosen = router.choose_instance(instances, model_id="m1")
            assert chosen.api_base == sticky, (
                f"Hysteresis must prevent switching when loads are equal. "
                f"Expected {sticky}, got {chosen.api_base}"
            )

    def test_hysteresis_allows_switching_on_large_gap(self):
        """
        INVARIANT: when a new instance is significantly better (gap > 0.05),
        the router switches away from the sticky instance.
        """
        router = ModelRouter(strategy=LeastLoadedStrategy())
        instances = [
            inst("http://current", load=0.0),
            inst("http://better", load=0.0),
        ]

        # Pin to "current"
        for _ in range(10):
            router.record("http://current", latency_ms=100.0)
        router.choose_instance(instances, model_id="m1")  # establishes sticky → current

        # Make "current" much worse (high inflight)
        router._get_metrics("http://current").inflight = 80  # +0.32 penalty

        chosen = router.choose_instance(instances, model_id="m1")
        assert chosen.api_base == "http://better", (
            "Router must switch when effective load gap exceeds hysteresis threshold"
        )


# ---------------------------------------------------------------------------
# 5. Anomaly detection
# ---------------------------------------------------------------------------


class TestAnomalyDetection:
    def test_anomaly_detected_on_sudden_latency_spike(self):
        """
        INVARIANT: when EWMA latency exceeds the slow-moving baseline by 3×,
        is_anomalous must return True.
        """
        router = ModelRouter(strategy=LeastLoadedStrategy())

        # Build a stable baseline (slow-moving EWMA ≈ 100ms)
        for _ in range(30):
            router.record("http://a", latency_ms=100.0)

        m = router._get_metrics("http://a")
        baseline_before = m.ewma_latency_baseline_ms
        assert not m.is_anomalous, "No anomaly before spike"

        # Inject a sustained spike: with α_fast=0.2, EWMA jumps quickly;
        # with α_baseline=0.01, the slow baseline barely moves.
        for _ in range(20):
            router.record("http://a", latency_ms=5_000.0)

        # After 20 spikes: fast EWMA ≈4944ms, slow baseline ≈992ms → 4944 > 992×3 → anomalous
        assert m.is_anomalous, (
            f"is_anomalous must be True when EWMA ({m.ewma_latency_ms:.0f}ms) "
            f">> baseline ({m.ewma_latency_baseline_ms:.0f}ms)"
        )

    def test_no_anomaly_on_steady_high_latency(self):
        """
        INVARIANT: when latency is uniformly high from the start, both EMAs
        converge to the same value and no anomaly is triggered.
        """
        router = ModelRouter(strategy=LeastLoadedStrategy())

        for _ in range(50):
            router.record("http://a", latency_ms=2_000.0)

        m = router._get_metrics("http://a")
        # Both EMAs converge to ~2000ms — baseline tracks the same value
        assert not m.is_anomalous, (
            "Steady-state high latency must not trigger anomaly detection"
        )

    def test_no_anomaly_before_10_requests(self):
        router = ModelRouter(strategy=LeastLoadedStrategy())
        for _ in range(9):
            router.record("http://a", latency_ms=5_000.0)
        assert not router._get_metrics("http://a").is_anomalous


# ---------------------------------------------------------------------------
# 6. GlobalSLO aggregation
# ---------------------------------------------------------------------------


class TestGlobalSLO:
    def test_snapshot_empty(self):
        slo = GlobalSLO()
        snap = slo.snapshot()
        assert snap["samples"] == 0
        assert snap["p50_ms"] is None
        assert snap["error_rate"] == 0.0

    def test_snapshot_percentiles(self):
        slo = GlobalSLO()
        for i in range(100):
            slo.record(float(i + 1), error=False)  # 1ms to 100ms

        snap = slo.snapshot()
        assert snap["samples"] == 100
        # p50 ≈ 50ms, p99 ≈ 99ms (index-based)
        assert snap["p50_ms"] == pytest.approx(50.0, abs=2.0)
        assert snap["p99_ms"] == pytest.approx(99.0, abs=2.0)

    def test_snapshot_error_rate(self):
        slo = GlobalSLO()
        for i in range(100):
            slo.record(10.0, error=(i % 4 == 0))  # 25% error rate

        snap = slo.snapshot()
        assert abs(snap["error_rate"] - 0.25) < 0.05

    def test_router_populates_global_slo(self):
        router = ModelRouter(strategy=LeastLoadedStrategy())
        for _ in range(10):
            router.record("http://a", latency_ms=50.0)
        for _ in range(10):
            router.record("http://b", latency_ms=100.0)

        snap = router.global_slo_snapshot()
        assert snap["samples"] == 20

    def test_reset_clears_global_slo(self):
        router = ModelRouter(strategy=LeastLoadedStrategy())
        for _ in range(10):
            router.record("http://a", latency_ms=50.0)
        router.reset_metrics()
        assert router.global_slo_snapshot()["samples"] == 0
