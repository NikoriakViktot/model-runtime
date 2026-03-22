"""
node_agent/gpu_reporter.py

Reports per-GPU memory statistics.

Uses pynvml (NVIDIA Management Library Python bindings).  Gracefully degrades
to an empty list on CPU-only nodes or environments where pynvml is not
initialised (e.g. CI, unit tests).

The caller (heartbeat loop) catches all exceptions so a GPU reporting failure
never crashes the agent.
"""

from __future__ import annotations

import logging

from node_agent.models import GpuInfo

logger = logging.getLogger(__name__)

_nvml_initialised = False


def _ensure_nvml() -> bool:
    """Initialise pynvml once. Returns True if NVML is available."""
    global _nvml_initialised
    if _nvml_initialised:
        return True
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_initialised = True
        count = pynvml.nvmlDeviceGetCount()
        logger.info("pynvml initialised — %d GPU(s) detected", count)
        return True
    except Exception as exc:
        logger.debug("pynvml unavailable (%s) — GPU metrics will be empty", exc)
        return False


def get_gpu_info() -> list[GpuInfo]:
    """
    Return per-GPU memory stats.

    Falls back to an empty list if:
    - pynvml is not installed
    - No NVIDIA driver is present
    - Running in a container without GPU passthrough
    """
    if not _ensure_nvml():
        return []
    try:
        import pynvml
        count = pynvml.nvmlDeviceGetCount()
        result: list[GpuInfo] = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            result.append(GpuInfo(
                gpu_index=str(i),
                memory_total_mb=mem.total // (1024 * 1024),
                memory_free_mb=mem.free // (1024 * 1024),
                memory_used_mb=mem.used // (1024 * 1024),
            ))
        return result
    except Exception as exc:
        logger.warning("GPU info collection failed: %s", exc)
        return []
