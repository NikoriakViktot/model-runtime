"""
mrm/services/telemetry.py

Per-model telemetry store backed by Redis.

Redis Key Schema
----------------
  mrm:telem:{model}:stats       HASH — cumulative counters + running sums
    total_requests              int   — inference calls reported
    sum_latency_ms              float — running sum of all reported latencies
    sum_tokens_sec              float — running sum of reported tokens/sec
    oom_count                   int   — OOM failures detected
    load_attempts               int   — total load attempts (success + failure)
    load_successes              int   — successful loads
    load_failures               int   — failed loads

  mrm:telem:{model}:latency_ring  LIST — ring buffer of last 200 latency_ms values
                                         LPUSH + LTRIM; used for p95 calculation

Metrics derived on read
-----------------------
  avg_latency_ms      — sum_latency_ms / total_requests
  p95_latency_ms      — 95th percentile of latency_ring values
  avg_tokens_sec      — sum_tokens_sec / total_requests
  success_rate        — load_successes / load_attempts  (0.0 if no attempts)
  reputation_score    — composite [0, 1]:
                        max(0, success_rate × perf_score − failure_penalty)
                        where:
                          perf_score      = 1 − min(1, avg_latency_ms / 30 000)
                          failure_penalty = min(1, oom_count × 0.15
                                                 + load_failures × 0.10)

Model IDs may contain '/' (HuggingFace repo_ids). Redis keys allow '/' natively.
The *:stats suffix is unambiguous as no model ID ends in ":stats".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MRM.telemetry")

_RING_SIZE = 200   # latency samples kept for p95


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelStats:
    model:            str
    total_requests:   int
    avg_latency_ms:   float
    p95_latency_ms:   float
    avg_tokens_sec:   float
    oom_count:        int
    load_attempts:    int
    load_successes:   int
    load_failures:    int
    success_rate:     float   # [0, 1]
    reputation_score: float   # [0, 1]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Store
# ──────────────────────────────────────────────────────────────────────────────

class TelemetryStore:
    """
    Thread-safe telemetry store backed by a Redis client.

    All write methods use Redis pipelines to minimise round-trips.
    All read methods are synchronous (wrap in asyncio.to_thread when
    calling from async context).
    """

    def __init__(self, redis_client) -> None:
        self._r = redis_client

    # ── Key helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _sk(model: str) -> str:
        return f"mrm:telem:{model}:stats"

    @staticmethod
    def _rk(model: str) -> str:
        return f"mrm:telem:{model}:latency_ring"

    # ── Write methods ────────────────────────────────────────────────────

    def record_inference(
        self,
        model:      str,
        latency_ms: float,
        tokens_sec: float,
    ) -> None:
        """
        Record a completed inference call.

        Args:
            model:      HF repo ID / base_model key.
            latency_ms: End-to-end latency in milliseconds.
            tokens_sec: Output throughput (tokens generated / wall-clock s).
        """
        pipe = self._r.pipeline()
        sk   = self._sk(model)
        rk   = self._rk(model)
        pipe.hincrbyfloat(sk, "total_requests", 1)
        pipe.hincrbyfloat(sk, "sum_latency_ms", latency_ms)
        pipe.hincrbyfloat(sk, "sum_tokens_sec", tokens_sec)
        pipe.lpush(rk, latency_ms)
        pipe.ltrim(rk, 0, _RING_SIZE - 1)
        pipe.execute()
        logger.debug(
            "[TELEM] inference  model=%s  latency=%.1fms  tok/s=%.1f",
            model, latency_ms, tokens_sec,
        )

    def record_oom(self, model: str) -> None:
        """Increment the OOM counter for *model*."""
        self._r.hincrbyfloat(self._sk(model), "oom_count", 1)
        logger.warning("[TELEM] OOM recorded  model=%s", model)

    def record_load_success(self, model: str) -> None:
        """Record a successful model load."""
        pipe = self._r.pipeline()
        pipe.hincrbyfloat(self._sk(model), "load_attempts",  1)
        pipe.hincrbyfloat(self._sk(model), "load_successes", 1)
        pipe.execute()
        logger.info("[TELEM] load_success  model=%s", model)

    def record_load_failure(self, model: str, is_oom: bool = False) -> None:
        """Record a failed model load. Pass is_oom=True to also bump oom_count."""
        pipe = self._r.pipeline()
        pipe.hincrbyfloat(self._sk(model), "load_attempts", 1)
        pipe.hincrbyfloat(self._sk(model), "load_failures", 1)
        if is_oom:
            pipe.hincrbyfloat(self._sk(model), "oom_count", 1)
        pipe.execute()
        logger.warning(
            "[TELEM] load_failure  model=%s  is_oom=%s", model, is_oom
        )

    # ── Read methods ─────────────────────────────────────────────────────

    def get_stats(self, model: str) -> ModelStats:
        """
        Return all telemetry metrics for *model*.

        Returns a zeroed ModelStats when no data has been recorded yet.
        """
        raw  = self._r.hgetall(self._sk(model))
        ring = self._r.lrange(self._rk(model), 0, -1)
        return _build_stats(model, raw, ring)

    def all_models(self) -> List[str]:
        """
        Return all model IDs that have any telemetry data stored.

        Uses Redis KEYS scan — suitable for dashboards / non-critical paths.
        For high-frequency calls prefer maintaining an explicit index set.
        """
        prefix = "mrm:telem:"
        suffix = ":stats"
        keys   = self._r.keys(f"{prefix}*{suffix}")
        result = []
        for k in keys:
            k_str = k.decode() if isinstance(k, bytes) else k
            if k_str.startswith(prefix) and k_str.endswith(suffix):
                result.append(k_str[len(prefix): -len(suffix)])
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_float(raw: dict, key: str) -> float:
    """Safely parse a float from a Redis HGETALL result (handles bytes/str/None)."""
    v = raw.get(key) or raw.get(key.encode() if isinstance(key, str) else key)
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, bytes):
        return float(v.decode())
    return float(v)


def _build_stats(model: str, raw: dict, ring: list) -> ModelStats:
    total_req  = int(_parse_float(raw, "total_requests"))
    sum_lat    = _parse_float(raw, "sum_latency_ms")
    sum_tok    = _parse_float(raw, "sum_tokens_sec")
    oom        = int(_parse_float(raw, "oom_count"))
    attempts   = int(_parse_float(raw, "load_attempts"))
    successes  = int(_parse_float(raw, "load_successes"))
    failures   = int(_parse_float(raw, "load_failures"))

    avg_lat     = (sum_lat / total_req) if total_req > 0 else 0.0
    avg_tok     = (sum_tok / total_req) if total_req > 0 else 0.0
    success_rate = (successes / attempts) if attempts > 0 else 0.0

    # p95 from latency ring buffer
    if ring:
        values = sorted(
            float(v.decode() if isinstance(v, bytes) else v)
            for v in ring
        )
        idx = max(0, int(len(values) * 0.95) - 1)
        p95 = values[idx]
    else:
        p95 = 0.0

    # Reputation formula
    # perf_score: 1.0 at instant response, 0.0 at 30 s avg latency
    perf_score      = 1.0 - min(1.0, avg_lat / 30_000.0)
    # failure_penalty: each OOM costs 0.15, each load failure 0.10, capped at 1.0
    failure_penalty = min(1.0, oom * 0.15 + failures * 0.10)
    reputation      = max(0.0, success_rate * perf_score - failure_penalty)

    return ModelStats(
        model            = model,
        total_requests   = total_req,
        avg_latency_ms   = round(avg_lat, 2),
        p95_latency_ms   = round(p95, 2),
        avg_tokens_sec   = round(avg_tok, 2),
        oom_count        = oom,
        load_attempts    = attempts,
        load_successes   = successes,
        load_failures    = failures,
        success_rate     = round(success_rate, 4),
        reputation_score = round(reputation, 4),
    )
