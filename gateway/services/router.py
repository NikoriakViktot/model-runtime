"""
gateway/services/router.py

Self-optimizing instance selection and per-instance metrics for multi-instance
model routing.

When MRM runs a model on multiple GPUs or nodes, it returns a list of
RuntimeInstances.  This module selects one instance per request using a
pluggable strategy augmented with adaptive runtime signals.

Routing pipeline
----------------
For each request ``choose_instance()`` applies the following pipeline:

1. Circuit filter  — exclude OPEN instances (fail-fast on dead upstreams).
                     Falls back to the full list if every instance is OPEN so
                     the system degrades gracefully rather than hard-failing.

2. Pool-relative   — compute each instance's effective load using EWMA latency
   scoring           relative to the pool mean, so the penalty is proportional
                     to how much slower this instance is than its peers.

3. Hysteresis      — per-model sticky routing.  A new instance is selected only
                     when its effective load is lower than the current one by
                     more than HYSTERESIS_GAP (0.05).  Prevents oscillation when
                     two instances are nearly equal.

4. Strategy        — final selection from the scored list (LeastLoaded by default).

Effective load formula
----------------------
    effective_load = reported_load
                   + latency_factor    ← pool-relative EWMA penalty, capped 0.4
                   + inflight_factor   ← inflight / 100, capped 0.4
                   clamped to [0, 1]

When no EWMA data is available (cold instance), falls back to
avg_latency_ms / 10_000 so the formula degrades gracefully.

Circuit breaker
---------------
``CircuitBreaker`` tracks consecutive errors per instance.  After
``circuit_error_threshold`` consecutive errors the instance is ejected
(OPEN).  After ``circuit_cooldown_sec`` it enters HALF_OPEN and the next
request acts as a probe.  A successful probe closes the circuit; a failed
probe reopens it (resetting the cooldown timer).

Global SLO aggregation
----------------------
``GlobalSLO`` maintains a sliding window of the last 1 000 requests
system-wide.  Exposed via ``/v1/slo`` as p50/p95/p99 latency and error rate.
This gives operators a single number for fleet health rather than per-instance
detail.

Anomaly detection
-----------------
``InstanceMetrics.is_anomalous`` detects two patterns:
- Latency spike : EWMA > slow-moving baseline × 3 (sudden degradation)
- Error burst   : recent error rate (last 5 requests) > long-term rate × 2.5

Backward compatibility
----------------------
When MRM returns a single ``api_base`` (current format), ``EnsureResult``
synthesises a one-element ``instances`` list.  All strategies return that
single element — behaviour is identical to before.

Thread safety: all state is touched only from the asyncio event loop, so
plain dict and int operations are safe without locks.
"""

from __future__ import annotations

import logging
import random as _random
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable

from gateway.config import settings
from gateway.observability import CIRCUIT_BREAKER_STATE, INSTANCE_EWMA_LATENCY_MS
from gateway.services.circuit_breaker import CircuitBreaker

_CB_STATE_INT = {"CLOSED": 0, "HALF_OPEN": 1, "OPEN": 2}

logger = logging.getLogger(__name__)

# How much better a candidate must be before hysteresis allows a switch.
_HYSTERESIS_GAP: float = 0.05


# ---------------------------------------------------------------------------
# Instance representation
# ---------------------------------------------------------------------------


@dataclass
class InstanceInfo:
    """
    One live runtime instance for a model.

    ``api_base`` is the full URL including ``/v1``, e.g.
    ``http://vllm_qwen:8000/v1``.  It uniquely identifies the instance and
    is used as the key for metrics.

    ``load`` is a float in [0, 1] reported by MRM.  0.0 means the instance
    is idle; 1.0 means it is at capacity.  Defaults to 0.0 when MRM does
    not report load (current single-instance format).

    INVARIANT: load must be in [0, 1].  Construction with an out-of-range
    value raises ValueError immediately (fail-fast).
    """

    api_base: str
    gpu: str = ""
    load: float = 0.0

    def __post_init__(self) -> None:
        # Runtime invariant — catches bugs in callers that construct InstanceInfo
        # directly with bad load values.  External data should use from_dict(),
        # which clamps the value before construction.
        if not (0.0 <= self.load <= 1.0):
            raise ValueError(
                f"InstanceInfo.load must be in [0, 1]; got {self.load!r} "
                f"for instance {self.api_base!r}"
            )

    @classmethod
    def from_dict(cls, d: dict) -> "InstanceInfo":
        raw_load = float(d.get("load", 0.0))
        return cls(
            api_base=d["api_base"],
            gpu=d.get("gpu", ""),
            # Clamp external data — MRM may report out-of-spec values
            load=max(0.0, min(1.0, raw_load)),
        )


# ---------------------------------------------------------------------------
# Per-instance metrics
# ---------------------------------------------------------------------------

_EWMA_ALPHA: float = 0.2        # fast-moving: tracks current latency
_BASELINE_ALPHA: float = 0.01   # slow-moving: tracks long-term baseline for anomaly detection


@dataclass
class InstanceMetrics:
    """
    Mutable counters and adaptive signals for one instance.

    Raw counters (requests, errors, total_latency_ms, inflight) are updated
    after every request.  EWMA signals (ewma_latency_ms, ewma_latency_baseline_ms)
    are updated via ``update_signals()`` and used by the adaptive scoring formula.
    """

    requests: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    inflight: int = 0
    # EWMA signals — initialised to 0.0 (no data); first sample sets both.
    ewma_latency_ms: float = 0.0
    ewma_latency_baseline_ms: float = 0.0

    def __post_init__(self) -> None:
        # Sliding window of recent errors (1 = error, 0 = ok) for burst detection.
        self._recent_errors: deque[int] = deque(maxlen=20)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.requests if self.requests else 0.0

    def update_signals(self, latency_ms: float, error: bool) -> None:
        """
        Update EWMA latency signals and the recent error window.

        Called by ``ModelRouter.record()`` after every request.
        """
        if self.ewma_latency_ms == 0.0:
            # Bootstrap: first sample initialises both EMAs directly.
            self.ewma_latency_ms = latency_ms
            self.ewma_latency_baseline_ms = latency_ms
        else:
            self.ewma_latency_ms = (
                _EWMA_ALPHA * latency_ms + (1.0 - _EWMA_ALPHA) * self.ewma_latency_ms
            )
            self.ewma_latency_baseline_ms = (
                _BASELINE_ALPHA * latency_ms
                + (1.0 - _BASELINE_ALPHA) * self.ewma_latency_baseline_ms
            )
        self._recent_errors.append(1 if error else 0)

    @property
    def is_anomalous(self) -> bool:
        """
        True when this instance shows signs of sudden degradation.

        Two conditions are checked independently:
        - Latency spike : fast-moving EWMA is ≥ 3× the slow-moving baseline,
                          and the baseline is above 10 ms (avoids false positives
                          on near-zero latency test data).
        - Error burst   : error rate over the last 5 requests exceeds the long-term
                          rate by ≥ 2.5× (and is above 10% to avoid noise).

        Requires at least 10 recorded requests to avoid false positives on
        small samples.
        """
        if self.requests < 10:
            return False

        # Latency spike
        latency_spike = (
            self.ewma_latency_baseline_ms > 10.0
            and self.ewma_latency_ms > self.ewma_latency_baseline_ms * 3.0
        )

        # Error burst
        error_burst = False
        if len(self._recent_errors) >= 5:
            recent = list(self._recent_errors)
            recent_rate = sum(recent[-5:]) / 5.0
            long_rate = sum(recent) / len(recent)
            error_burst = recent_rate > max(long_rate * 2.5, 0.10)

        return latency_spike or error_burst

    def to_dict(self) -> dict:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "inflight": self.inflight,
            "ewma_latency_ms": round(self.ewma_latency_ms, 2),
            "anomalous": self.is_anomalous,
        }


# ---------------------------------------------------------------------------
# Global SLO aggregation
# ---------------------------------------------------------------------------


class GlobalSLO:
    """
    Fleet-wide SLO view: sliding window of the last 1 000 requests.

    Unlike per-instance metrics, GlobalSLO aggregates across all instances
    so operators get a single number for overall fleet health.
    """

    _WINDOW_SIZE = 1_000

    def __init__(self) -> None:
        # Each entry: (latency_ms: float, error: bool)
        self._window: deque[tuple[float, bool]] = deque(maxlen=self._WINDOW_SIZE)

    def record(self, latency_ms: float, error: bool) -> None:
        self._window.append((latency_ms, error))

    def snapshot(self) -> dict:
        """
        Return a point-in-time summary of the current window.

        Returns ``None`` for percentiles when the window is empty.
        """
        if not self._window:
            return {
                "samples": 0,
                "p50_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "error_rate": 0.0,
            }
        items = list(self._window)
        latencies = sorted(x[0] for x in items)
        n = len(latencies)
        errors = sum(1 for x in items if x[1])
        return {
            "samples": n,
            "p50_ms": round(latencies[int(n * 0.50)], 1),
            "p95_ms": round(latencies[int(n * 0.95)], 1),
            "p99_ms": round(latencies[int(n * 0.99)], 1),
            "error_rate": round(errors / n, 4),
        }

    def reset(self) -> None:
        self._window.clear()


# ---------------------------------------------------------------------------
# Strategy protocol + implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class RoutingStrategy(Protocol):
    """
    Selects one instance from a non-empty list.

    Implementations must be safe to call concurrently from the asyncio loop.
    They receive the full list every time; the router never pre-filters.

    NOTE: strategies receive instances whose ``load`` field may have been
    augmented with observed runtime data by ``ModelRouter._effective_load()``.
    """

    def choose(self, instances: list[InstanceInfo]) -> InstanceInfo:
        ...


class LeastLoadedStrategy:
    """
    Pick the instance reporting the lowest ``load``.

    When the router augments loads with latency and inflight data, this
    strategy naturally becomes latency-aware and backpressure-aware without
    any changes.  Equal effective loads fall back to random selection.
    """

    def choose(self, instances: list[InstanceInfo]) -> InstanceInfo:
        min_load = min(i.load for i in instances)
        candidates = [i for i in instances if i.load == min_load]
        return _random.choice(candidates)


class RoundRobinStrategy:
    """
    Cycle through instances in a fixed order.

    The counter is per-strategy instance, so deploying two separate
    ModelRouters will have independent cycles.

    Safe for the asyncio event loop: Python integer increment is atomic
    under the GIL; no explicit lock is required.
    """

    def __init__(self) -> None:
        self._counter = 0

    def choose(self, instances: list[InstanceInfo]) -> InstanceInfo:
        idx = self._counter % len(instances)
        self._counter += 1
        return instances[idx]


class RandomStrategy:
    """Uniform random selection. Stateless."""

    def choose(self, instances: list[InstanceInfo]) -> InstanceInfo:
        return _random.choice(instances)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class ModelRouter:
    """
    Selects a runtime instance for each request and tracks per-instance stats.

    Usage::

        instance = model_router.choose_instance(ensure_result.instances, model_id=model)
        async with model_router.track_inflight(instance.api_base):
            result = await proxy.post(instance.api_base + "/chat/completions", body)
        model_router.record(instance.api_base, latency_ms=42.0, error=False)
    """

    def __init__(
        self,
        strategy: RoutingStrategy,
        circuit_error_threshold: int = 5,
        circuit_cooldown_sec: float = 30.0,
    ) -> None:
        self._strategy = strategy
        self._circuit_error_threshold = circuit_error_threshold
        self._circuit_cooldown_sec = circuit_cooldown_sec

        # api_base → InstanceMetrics; defaultdict so tests can access keys directly.
        # InstanceMetrics.__post_init__ is called when defaultdict creates a new entry.
        self._metrics: dict[str, InstanceMetrics] = defaultdict(InstanceMetrics)
        # api_base → CircuitBreaker; created on first access.
        self._circuits: dict[str, CircuitBreaker] = {}
        # model_id → api_base; used for hysteresis (sticky routing).
        self._sticky: dict[str, str] = {}
        # Fleet-wide SLO window.
        self._global_slo: GlobalSLO = GlobalSLO()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_metrics(self, api_base: str) -> InstanceMetrics:
        return self._metrics[api_base]  # defaultdict creates on first access

    def _get_circuit(self, api_base: str) -> CircuitBreaker:
        if api_base not in self._circuits:
            self._circuits[api_base] = CircuitBreaker(
                error_threshold=self._circuit_error_threshold,
                cooldown_sec=self._circuit_cooldown_sec,
            )
        return self._circuits[api_base]

    # ------------------------------------------------------------------
    # Effective load (adaptive, pool-relative)
    # ------------------------------------------------------------------

    def _effective_load(self, instance: InstanceInfo, pool_mean_latency: float) -> float:
        """
        Combine reported load with pool-relative EWMA latency and in-flight
        count into a single routing score in [0, 1].

        Pool-relative formula:
            latency_factor = max((ewma_latency / pool_mean - 1.0) × 0.2, 0.0)
                             capped at 0.4

        This penalises instances that are slower than the pool average
        proportionally — if the pool mean is 100 ms and this instance is
        200 ms, the penalty is +0.2; if it's 150 ms the penalty is +0.1.
        Instances at or below the pool mean receive no penalty (factor = 0),
        so a uniformly slow pool is treated as healthy rather than all
        instances getting penalised equally.

        Falls back to the absolute formula (avg_latency_ms / 10_000) when
        pool_mean is zero or EWMA is unavailable (cold instance).
        """
        m = self._metrics.get(instance.api_base)
        if m is None or (m.requests == 0 and m.inflight == 0):
            return instance.load

        if pool_mean_latency > 0.0 and m.ewma_latency_ms > 0.0:
            latency_factor = min(
                max((m.ewma_latency_ms / pool_mean_latency - 1.0) * 0.2, 0.0), 0.4
            )
        else:
            latency_factor = min(m.avg_latency_ms / 10_000.0, 0.4)

        inflight_factor = min(m.inflight / 100.0, 0.4)
        return min(instance.load + latency_factor + inflight_factor, 1.0)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def choose_instance(
        self,
        instances: list[InstanceInfo],
        model_id: str | None = None,
    ) -> InstanceInfo:
        """
        Select one instance from ``instances`` using the configured strategy,
        augmented with circuit filtering, adaptive scoring, and hysteresis.

        Args:
            instances: Non-empty list from ``EnsureResult.instances``.
            model_id:  Optional model identifier for hysteresis (sticky routing).
                       When provided, the router avoids switching away from the
                       current instance unless the new one is significantly better.

        Returns:
            The selected ``InstanceInfo`` (with the original reported load,
            not the augmented score).

        Raises:
            ValueError: If ``instances`` is empty (indicates an MRM bug).
        """
        if not instances:
            raise ValueError("choose_instance called with empty instances list")

        # 1. Circuit filter: exclude OPEN circuits.
        available = [i for i in instances if self._get_circuit(i.api_base).is_available]
        if not available:
            # Last resort: all instances are OPEN — use the full list so the
            # system degrades gracefully rather than refusing all requests.
            available = instances

        if len(available) == 1:
            chosen = available[0]
            if model_id:
                self._sticky[model_id] = chosen.api_base
            return chosen

        # 2. Pool-relative scoring: compute mean EWMA latency of live instances.
        ewma_vals = [
            m.ewma_latency_ms
            for url in (i.api_base for i in available)
            if (m := self._metrics.get(url)) and m.ewma_latency_ms > 0.0
        ]
        pool_mean = sum(ewma_vals) / len(ewma_vals) if ewma_vals else 0.0

        scored = [
            InstanceInfo(
                api_base=i.api_base,
                gpu=i.gpu,
                load=self._effective_load(i, pool_mean),
            )
            for i in available
        ]

        chosen_scored = self._strategy.choose(scored)

        # 3. Hysteresis: stick to current instance unless the new one is
        #    significantly better (gap > HYSTERESIS_GAP).
        if model_id:
            current_url = self._sticky.get(model_id)
            if current_url and any(s.api_base == current_url for s in scored):
                current_scored = next(
                    s for s in scored if s.api_base == current_url
                )
                if chosen_scored.api_base != current_url:
                    if not (chosen_scored.load < current_scored.load - _HYSTERESIS_GAP):
                        # Difference too small — stay with the current instance.
                        chosen_scored = current_scored

        # 4. Map back to original InstanceInfo (with un-augmented load).
        chosen = next(i for i in available if i.api_base == chosen_scored.api_base)

        if model_id:
            self._sticky[model_id] = chosen.api_base

        logger.debug(
            "Router chose instance api_base=%s load=%.2f effective=%.2f "
            "(strategy=%s, pool=%d, circuit_filtered=%d)",
            chosen.api_base,
            chosen.load,
            chosen_scored.load,
            type(self._strategy).__name__,
            len(instances),
            len(instances) - len(available),
        )
        return chosen

    # ------------------------------------------------------------------
    # In-flight tracking (backpressure)
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def track_inflight(self, api_base: str) -> AsyncIterator[None]:
        """
        Async context manager that tracks one in-flight request for an instance.

        Increments the inflight counter on entry, decrements on exit (even if
        the request raises).  The router uses this count to apply backpressure
        to instances under heavy concurrent load.

        Example::

            async with model_router.track_inflight(api_base):
                result = await proxy.post(url, body)
        """
        self._get_metrics(api_base).inflight += 1
        try:
            yield
        finally:
            m = self._metrics.get(api_base)
            if m is not None:
                m.inflight = max(0, m.inflight - 1)

    # ------------------------------------------------------------------
    # Metrics recording
    # ------------------------------------------------------------------

    def record(
        self,
        api_base: str,
        *,
        latency_ms: float,
        error: bool = False,
    ) -> None:
        """
        Update counters for the instance that served a request.

        Called after every proxy call (both unary and streaming).
        Updates raw counters, EWMA signals, circuit breaker state, and the
        global SLO window.  Non-blocking — purely in-memory.

        Args:
            api_base:   The instance URL that was used.
            latency_ms: End-to-end latency in milliseconds.
            error:      True if the upstream returned an error or was unreachable.
        """
        m = self._get_metrics(api_base)
        m.requests += 1
        m.total_latency_ms += latency_ms
        if error:
            m.errors += 1
        m.update_signals(latency_ms, error)

        # Circuit breaker
        cb = self._get_circuit(api_base)
        if error:
            cb.record_failure()
        else:
            cb.record_success()

        # Global SLO window
        self._global_slo.record(latency_ms, error)

        # Prometheus per-instance gauges
        CIRCUIT_BREAKER_STATE.labels(instance=api_base).set(
            _CB_STATE_INT.get(cb.state, 0)
        )
        INSTANCE_EWMA_LATENCY_MS.labels(instance=api_base).set(m.ewma_latency_ms)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, dict]:
        """
        Return a snapshot of all per-instance metrics.

        Keys are ``api_base`` strings.  Values contain ``requests``,
        ``errors``, ``avg_latency_ms``, ``total_latency_ms``, ``inflight``,
        ``ewma_latency_ms``, and ``anomalous``.
        """
        return {url: m.to_dict() for url, m in self._metrics.items()}

    def get_circuit_states(self) -> dict[str, dict]:
        """Return a snapshot of all circuit breaker states."""
        return {url: cb.to_dict() for url, cb in self._circuits.items()}

    def anomalies(self) -> list[str]:
        """Return api_base URLs of instances currently showing anomalous signals."""
        return [url for url, m in self._metrics.items() if m.is_anomalous]

    def global_slo_snapshot(self) -> dict:
        """Return a point-in-time fleet-wide SLO summary."""
        return self._global_slo.snapshot()

    def slo_status(self) -> dict:
        """
        Return a health snapshot against SLO thresholds.

        Evaluates every instance with at least 10 requests against:
        - Error rate < 5%
        - Average latency < 5 000 ms

        Returns a dict with ``slo_ok`` (bool) and ``violations`` (list).
        """
        violations: list[dict] = []
        for url, m in self._metrics.items():
            if m.requests < 10:
                continue
            error_rate = m.errors / m.requests
            if error_rate > 0.05:
                violations.append({
                    "instance": url,
                    "type": "high_error_rate",
                    "value": round(error_rate, 4),
                    "threshold": 0.05,
                })
            if m.avg_latency_ms > 5_000.0:
                violations.append({
                    "instance": url,
                    "type": "high_latency_ms",
                    "value": round(m.avg_latency_ms, 1),
                    "threshold": 5_000.0,
                })
        return {"slo_ok": len(violations) == 0, "violations": violations}

    def reset_metrics(self, api_base: str | None = None) -> None:
        """
        Reset metrics for one instance or all instances.

        Args:
            api_base: If given, reset only that instance (metrics + circuit).
                      If ``None``, reset all metrics, circuits, sticky state,
                      and the global SLO window.
        """
        if api_base is not None:
            self._metrics.pop(api_base, None)
            self._circuits.pop(api_base, None)
        else:
            self._metrics.clear()
            self._circuits.clear()
            self._sticky.clear()
            self._global_slo.reset()

    @property
    def strategy_name(self) -> str:
        return type(self._strategy).__name__


# ---------------------------------------------------------------------------
# Strategy factory + module singleton
# ---------------------------------------------------------------------------

_STRATEGIES: dict[str, RoutingStrategy] = {
    "least_loaded": LeastLoadedStrategy(),
    "round_robin": RoundRobinStrategy(),
    "random": RandomStrategy(),
}


def _make_strategy(name: str) -> RoutingStrategy:
    strategy = _STRATEGIES.get(name)
    if strategy is None:
        logger.warning(
            "Unknown routing strategy '%s'. Falling back to 'round_robin'. "
            "Valid options: %s",
            name,
            list(_STRATEGIES),
        )
        return _STRATEGIES["round_robin"]
    logger.info("Routing strategy: %s", name)
    return strategy


#: Module-level singleton.  Import and use directly in routes.
model_router = ModelRouter(
    strategy=_make_strategy(settings.routing_strategy),
    circuit_error_threshold=settings.cpu_cb_failure_threshold,
    circuit_cooldown_sec=settings.cpu_cb_reset_timeout_sec,
)
