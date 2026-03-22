"""
gateway/services/router.py

Instance selection and per-instance metrics for multi-instance model routing.

When MRM runs a model on multiple GPUs or nodes, it returns a list of
RuntimeInstances.  This module selects one instance per request using a
pluggable strategy and tracks metrics so operators can observe the distribution.

Strategies
----------
least_loaded  Pick the instance with the lowest reported load.
              Requires MRM to report ``load`` per instance.
              Falls back to random when all loads are equal.

round_robin   Cycle through instances in order.
              Even distribution regardless of load.
              Default choice when load data is unavailable.

random        Uniform random selection.
              Simple and avoids any coordination state.

Backward compatibility
----------------------
When MRM returns a single ``api_base`` (current format), ``EnsureResult``
synthesises a one-element ``instances`` list.  All strategies return that
single element — behaviour is identical to before.

Metrics
-------
Per-instance counters live in memory for the lifetime of the process.
They are exposed via ``GET /v1/router/metrics`` and optionally logged to
MLflow alongside the per-request inference metrics.

Thread safety: all state is touched only from the asyncio event loop, so
plain dict and int operations are safe without locks.
"""

from __future__ import annotations

import logging
import random as _random
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import cycle
from typing import Protocol, runtime_checkable

from gateway.config import settings

logger = logging.getLogger(__name__)


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
    """

    api_base: str
    gpu: str = ""
    load: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> InstanceInfo:
        return cls(
            api_base=d["api_base"],
            gpu=d.get("gpu", ""),
            load=float(d.get("load", 0.0)),
        )


# ---------------------------------------------------------------------------
# Per-instance metrics
# ---------------------------------------------------------------------------


@dataclass
class InstanceMetrics:
    """Mutable counters for one instance. Updated after every routed request."""

    requests: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.requests if self.requests else 0.0

    def to_dict(self) -> dict:
        return {
            "requests": self.requests,
            "errors": self.errors,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
        }


# ---------------------------------------------------------------------------
# Strategy protocol + implementations
# ---------------------------------------------------------------------------


@runtime_checkable
class RoutingStrategy(Protocol):
    """
    Selects one instance from a non-empty list.

    Implementations must be safe to call concurrently from the asyncio loop.
    They receive the full list every time; the router never pre-filters.
    """

    def choose(self, instances: list[InstanceInfo]) -> InstanceInfo:
        ...


class LeastLoadedStrategy:
    """
    Pick the instance reporting the lowest ``load``.

    When all loads are equal (including the common 0.0 default), fall back
    to random selection to spread traffic across equal candidates.
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

        instance = model_router.choose_instance(ensure_result.instances)
        # ... proxy request to instance.api_base ...
        model_router.record(instance.api_base, latency_ms=42.0, error=False)
    """

    def __init__(self, strategy: RoutingStrategy) -> None:
        self._strategy = strategy
        # api_base → InstanceMetrics; defaultdict auto-creates on first access
        self._metrics: dict[str, InstanceMetrics] = defaultdict(InstanceMetrics)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def choose_instance(self, instances: list[InstanceInfo]) -> InstanceInfo:
        """
        Select one instance from ``instances`` using the configured strategy.

        When only one instance is available the strategy is skipped
        entirely — no overhead, identical behaviour to the pre-router code.

        Args:
            instances: Non-empty list from ``EnsureResult.instances``.

        Returns:
            The selected ``InstanceInfo``.

        Raises:
            ValueError: If ``instances`` is empty (indicates an MRM bug).
        """
        if not instances:
            raise ValueError("choose_instance called with empty instances list")

        if len(instances) == 1:
            return instances[0]

        chosen = self._strategy.choose(instances)
        logger.debug(
            "Router chose instance api_base=%s load=%.2f (strategy=%s, pool=%d)",
            chosen.api_base,
            chosen.load,
            type(self._strategy).__name__,
            len(instances),
        )
        return chosen

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
        Non-blocking — purely in-memory.

        Args:
            api_base:   The instance URL that was used.
            latency_ms: End-to-end latency in milliseconds.
            error:      True if the upstream returned an error or was unreachable.
        """
        m = self._metrics[api_base]
        m.requests += 1
        m.total_latency_ms += latency_ms
        if error:
            m.errors += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, dict]:
        """
        Return a snapshot of all per-instance metrics.

        Keys are ``api_base`` strings.  Values contain ``requests``,
        ``errors``, ``avg_latency_ms``, and ``total_latency_ms``.
        """
        return {url: m.to_dict() for url, m in self._metrics.items()}

    def reset_metrics(self, api_base: str | None = None) -> None:
        """
        Reset metrics for one instance or all instances.

        Args:
            api_base: If given, reset only that instance.
                      If ``None``, reset all.
        """
        if api_base is not None:
            self._metrics.pop(api_base, None)
        else:
            self._metrics.clear()

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
model_router = ModelRouter(strategy=_make_strategy(settings.routing_strategy))
