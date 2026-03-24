"""
mrm/core/queue.py

FIFO queue for model load requests.

Each request carries:
  - base_model:   HF repo ID to load
  - preset:       vLLM preset name ("small_chat" | "7b_awq")
  - overrides:    Dict of allowed vLLM parameter overrides
  - gpu:          Target GPU ID (default "0")
  - request_id:   UUID for tracking
  - enqueued_at:  Unix timestamp
  - _future:      asyncio.Future resolved by the scheduler with the result

Callers await the future to get the result or an exception::

    req = LoadRequest(base_model="Qwen/...", preset="small_chat")
    await queue.enqueue(req)
    result = await req.future   # blocks until scheduler processes it
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("MRM.queue")


# ──────────────────────────────────────────────────────────────────────────────
# Request dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LoadRequest:
    base_model: str
    preset: str = "small_chat"
    overrides: Dict[str, Any] = field(default_factory=dict)
    gpu: str = "0"
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    enqueued_at: float = field(default_factory=time.time)
    # Internal: resolved by the scheduler
    _future: Optional[asyncio.Future] = field(default=None, repr=False)

    @property
    def future(self) -> asyncio.Future:
        if self._future is None:
            raise RuntimeError("Request has not been enqueued yet.")
        return self._future

    def set_result(self, result: Any) -> None:
        if self._future and not self._future.done():
            self._future.set_result(result)

    def set_exception(self, exc: Exception) -> None:
        if self._future and not self._future.done():
            self._future.set_exception(exc)


# ──────────────────────────────────────────────────────────────────────────────
# Queue
# ──────────────────────────────────────────────────────────────────────────────

class LoadQueue:
    """
    Async FIFO queue wrapping asyncio.Queue.

    Thread-safe for use within a single event loop.
    """

    def __init__(self, maxsize: int = 64) -> None:
        self._q: asyncio.Queue[LoadRequest] = asyncio.Queue(maxsize=maxsize)

    # ------------------------------------------------------------------
    # Producer side
    # ------------------------------------------------------------------

    async def enqueue(self, req: LoadRequest) -> asyncio.Future:
        """
        Add *req* to the queue and return its Future.

        Callers can await the future to get the load result or exception.

        Raises QueueFullError if maxsize is reached.
        """
        loop = asyncio.get_event_loop()
        req._future = loop.create_future()

        try:
            self._q.put_nowait(req)
        except asyncio.QueueFull as exc:
            req.set_exception(
                QueueFullError(
                    f"Load queue is full ({self._q.maxsize} slots). "
                    "Try again later."
                )
            )
            raise QueueFullError from exc

        logger.info(
            "[QUEUE] enqueued  model=%s  preset=%s  gpu=%s  request_id=%s  queue_size=%d",
            req.base_model,
            req.preset,
            req.gpu,
            req.request_id,
            self._q.qsize(),
        )
        return req._future

    # ------------------------------------------------------------------
    # Consumer side (used by Scheduler)
    # ------------------------------------------------------------------

    async def dequeue(self) -> LoadRequest:
        """Block until a request is available and return it."""
        req = await self._q.get()
        logger.info(
            "[QUEUE] dequeued  model=%s  request_id=%s  waited=%.1fs  queue_size=%d",
            req.base_model,
            req.request_id,
            time.time() - req.enqueued_at,
            self._q.qsize(),
        )
        return req

    def task_done(self) -> None:
        """Signal that the last dequeued item was processed."""
        self._q.task_done()

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._q.qsize()

    @property
    def empty(self) -> bool:
        return self._q.empty()


class QueueFullError(Exception):
    """Raised when the load queue has reached its capacity."""
