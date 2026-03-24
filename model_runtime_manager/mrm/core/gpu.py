"""
mrm/core/gpu.py

GPU VRAM availability checker.

Uses nvidia-smi for accurate, real-time free-memory readings.
Falls back gracefully when NVML/nvidia-smi is unavailable (e.g. CI).
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger("MRM.gpu")


@dataclass
class GpuMemory:
    gpu_id: str
    total_mib: int
    used_mib: int
    free_mib: int

    @property
    def free_gb(self) -> float:
        return self.free_mib / 1024.0

    @property
    def total_gb(self) -> float:
        return self.total_mib / 1024.0


def query_gpu_memory_sync(gpu_id: str = "0") -> GpuMemory:
    """
    Synchronous: query GPU memory via nvidia-smi.

    Returns a GpuMemory with zeros on failure so callers can handle
    unavailability without crashing.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        total, used, free = [int(x.strip()) for x in out.split(",")]
        return GpuMemory(gpu_id=gpu_id, total_mib=total, used_mib=used, free_mib=free)
    except FileNotFoundError:
        logger.debug("[GPU] nvidia-smi not found — returning zero memory")
    except subprocess.TimeoutExpired:
        logger.warning("[GPU] nvidia-smi timed out for gpu_id=%s", gpu_id)
    except Exception as exc:
        logger.warning("[GPU] nvidia-smi failed for gpu_id=%s: %s", gpu_id, exc)
    return GpuMemory(gpu_id=gpu_id, total_mib=0, used_mib=0, free_mib=0)


async def query_gpu_memory(gpu_id: str = "0") -> GpuMemory:
    """Async wrapper around query_gpu_memory_sync."""
    return await asyncio.to_thread(query_gpu_memory_sync, gpu_id)


async def query_all_gpus(gpu_ids: List[str]) -> Dict[str, GpuMemory]:
    """Query all GPU IDs concurrently and return a mapping gpu_id → GpuMemory."""
    results = await asyncio.gather(*[query_gpu_memory(g) for g in gpu_ids])
    return {mem.gpu_id: mem for mem in results}


def estimate_model_vram_mib(gpu_memory_utilization: float, total_mib: int) -> int:
    """
    Estimate VRAM a model will consume given its gpu_memory_utilization fraction
    and the total GPU memory.

    This is an upper-bound estimate used for pre-flight checks.
    """
    return int(total_mib * gpu_memory_utilization)


def check_fits(
    required_mib: int,
    gpu_mem: GpuMemory,
    headroom_mib: int = 512,
) -> bool:
    """
    Return True if the GPU has enough free VRAM for the request.

    Args:
        required_mib:  Estimated model VRAM requirement in MiB.
        gpu_mem:       Current GPU memory snapshot.
        headroom_mib:  Safety margin to keep free (default 512 MiB).
    """
    effective_free = gpu_mem.free_mib - headroom_mib
    fits = required_mib <= effective_free
    logger.info(
        "[GPU] gpu=%s  free=%dMiB  required=%dMiB  headroom=%dMiB  fits=%s",
        gpu_mem.gpu_id,
        gpu_mem.free_mib,
        required_mib,
        headroom_mib,
        fits,
    )
    return fits


def select_free_gpu(
    allowed_gpus: List[str],
    required_mib: int,
) -> Optional[str]:
    """
    Synchronously scan *allowed_gpus* and return the first one with enough
    free VRAM, or None if none qualify.
    """
    for gid in allowed_gpus:
        mem = query_gpu_memory_sync(gid)
        if check_fits(required_mib, mem):
            return gid
    return None
