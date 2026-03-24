"""
cpu_runtime/load_shedder.py

Adaptive load shedding advisor for the CPU inference runtime.

Tracks:
  - Rolling average latency over the last N completed requests.
  - Free RAM in MiB, cached with a configurable TTL.

Both operations are O(1) and I/O-free in the fast path (RAM is cached).
No locks needed — asyncio is single-threaded.

Usage::

    # At module level (singleton):
    shedder = LoadShedder()

    # In the request handler:
    free_mb = shedder.check_ram()
    ceiling = shedder.effective_max_requests(
        base=settings.max_queue_depth,
        threshold_ms=settings.latency_threshold_ms,
    )
    # ... after completion:
    shedder.record_latency(elapsed_ms)
"""
from __future__ import annotations

import time
from collections import deque


def _read_free_ram_mb() -> int:
    """Parse MemAvailable from /proc/meminfo. Returns -1 if unavailable."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


class LoadShedder:
    """
    Lightweight shedding advisor.

    Parameters
    ----------
    window_size:
        Number of recent requests to track for the latency rolling average.
    ram_check_interval_sec:
        How often to re-read /proc/meminfo.  Cached between reads.
    """

    def __init__(
        self,
        window_size: int = 20,
        ram_check_interval_sec: float = 5.0,
    ) -> None:
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._ram_free_mb: int = -1
        self._ram_last_ts: float = 0.0
        self._ram_interval = ram_check_interval_sec

    # ------------------------------------------------------------------
    # Latency tracking
    # ------------------------------------------------------------------

    def record_latency(self, latency_ms: float) -> None:
        """Record the wall-clock duration of one completed request (ms)."""
        self._latencies.append(latency_ms)

    @property
    def avg_latency_ms(self) -> float:
        """Rolling average over the last window_size requests. 0 if empty."""
        if not self._latencies:
            return 0.0
        return sum(self._latencies) / len(self._latencies)

    def effective_max_requests(self, base: int, threshold_ms: float) -> int:
        """
        Return the effective concurrency ceiling.

        When avg_latency_ms > threshold_ms the ceiling is reduced
        proportionally, floored at 1:

            effective = max(1, int(base × threshold_ms / avg_latency_ms))

        Examples:
            avg = 2× threshold  →  base / 2
            avg ≤ threshold     →  base   (no shedding)
            no data yet         →  base   (no shedding)
        """
        avg = self.avg_latency_ms
        if avg <= 0.0 or threshold_ms <= 0.0 or avg <= threshold_ms:
            return base
        return max(1, int(base * (threshold_ms / avg)))

    # ------------------------------------------------------------------
    # RAM tracking
    # ------------------------------------------------------------------

    def check_ram(self) -> int:
        """
        Return free RAM in MiB.  Value is cached for ram_check_interval_sec.
        Returns -1 if /proc/meminfo is unreadable (non-Linux host).
        """
        now = time.monotonic()
        if now - self._ram_last_ts >= self._ram_interval:
            self._ram_free_mb = _read_free_ram_mb()
            self._ram_last_ts = now
        return self._ram_free_mb


# ---------------------------------------------------------------------------
# Module-level singleton — used by routes and app.py
# ---------------------------------------------------------------------------

shedder = LoadShedder()
