"""
node_agent/heartbeat.py

Background task that pushes GPU state and node identity to the Scheduler.

Push vs pull: the agent pushes so the Scheduler never needs to know a node's
address in advance.  Nodes self-register on first heartbeat.

A failed heartbeat is logged as a warning but does not crash the agent —
the scheduler will simply not refresh the node's TTL, and it will be
considered dead after node_ttl_sec seconds.
"""

from __future__ import annotations

import asyncio
import socket
import time

import httpx
import structlog

from node_agent.config import settings
from node_agent.gpu_reporter import get_gpu_info
from node_agent.observability import (
    GPU_FREE_MB,
    GPU_TOTAL_MB,
    GPU_USED_MB,
    HEARTBEAT_TIMESTAMP,
)

log = structlog.get_logger(__name__)


async def send_heartbeat(node_id: str) -> None:
    """
    Send one heartbeat to the Scheduler.

    Also updates Prometheus GPU gauges so /metrics reflects current GPU state
    even between Prometheus scrapes of the scheduler.
    """
    hostname = socket.gethostname()
    gpus = get_gpu_info()

    # Update Prometheus GPU gauges before sending so they are always current
    for gpu in gpus:
        labels = {"node_id": node_id, "gpu_index": gpu.gpu_index}
        GPU_FREE_MB.labels(**labels).set(gpu.memory_free_mb)
        GPU_TOTAL_MB.labels(**labels).set(gpu.memory_total_mb)
        GPU_USED_MB.labels(**labels).set(gpu.memory_used_mb)

    payload = {
        "node_id": node_id,
        "agent_url": settings.agent_url,
        "hostname": hostname,
        "gpus": [g.model_dump() for g in gpus],
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(f"{settings.scheduler_url}/heartbeat", json=payload)
        resp.raise_for_status()

    HEARTBEAT_TIMESTAMP.labels(node_id=node_id).set(time.time())

    log.debug(
        "heartbeat_sent",
        node_id=node_id,
        gpus=len(gpus),
        free_mb=sum(g.memory_free_mb for g in gpus),
    )


async def heartbeat_loop(node_id: str) -> None:
    """
    Send heartbeats on a fixed interval until cancelled.

    Designed to run as an asyncio background task started in the app lifespan.
    """
    log.info(
        "heartbeat_loop_started",
        node_id=node_id,
        interval=settings.heartbeat_interval_sec,
        scheduler=settings.scheduler_url,
    )
    while True:
        try:
            await send_heartbeat(node_id)
        except Exception as exc:
            log.warning("heartbeat_failed", node_id=node_id, error=str(exc))
        await asyncio.sleep(settings.heartbeat_interval_sec)
