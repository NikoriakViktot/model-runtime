"""
mrm/core/scheduler.py

Sequential model load scheduler.

Guarantees that only ONE model loads at a time — GPU VRAM cannot be
double-allocated and vLLM startup is single-threaded by nature.

Architecture
------------
  LoadQueue  →  Scheduler (single consumer) →  ModelLoader
                                              ↘  FallbackService (on failure)

The Scheduler runs as a background asyncio task.  Each LoadRequest carries
an asyncio.Future; the caller awaits it to get the result or exception.

Execution flow per request
--------------------------
  1. Dequeue next LoadRequest
  2. Acquire _load_lock (prevents re-entrancy)
  3. Delegate to ModelLoader.load()
  4. On LoadError: try FallbackService.try_fallback()
  5. Resolve the request's Future
  6. Release _load_lock, mark queue task done, continue
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from .queue import LoadQueue, LoadRequest
from ..services.loader import ModelLoader, LoadError

logger = logging.getLogger("MRM.scheduler")


class LoadScheduler:
    """
    Background scheduler that serialises model load operations.

    Usage::

        scheduler = LoadScheduler(queue, loader, fallback)
        scheduler.start()        # launch background task
        await scheduler.stop()   # graceful shutdown
    """

    def __init__(
        self,
        queue: LoadQueue,
        loader: ModelLoader,
        fallback=None,   # FallbackService | None
    ) -> None:
        self._queue    = queue
        self._loader   = loader
        self._fallback = fallback
        self._lock     = asyncio.Lock()   # one load at a time
        self._running  = False
        self._task: Optional[asyncio.Task] = None
        self._active_model: Optional[str]  = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._task    = asyncio.create_task(self._consume(), name="load_scheduler")
        logger.info("[SCHEDULER] started")

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[SCHEDULER] stopped")

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def active_model(self) -> Optional[str]:
        """The model currently being loaded, or None."""
        return self._active_model

    @property
    def queue_size(self) -> int:
        return self._queue.size

    # ------------------------------------------------------------------
    # Consumer loop
    # ------------------------------------------------------------------

    async def _consume(self) -> None:
        logger.info("[SCHEDULER] consumer loop running")
        while self._running:
            try:
                req = await asyncio.wait_for(
                    self._queue.dequeue(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue   # empty queue — loop again
            except asyncio.CancelledError:
                break

            await self._process(req)

    async def _process(self, req: LoadRequest) -> None:
        """
        Process a single LoadRequest while holding _load_lock.
        """
        async with self._lock:
            self._active_model = req.base_model
            logger.info(
                "[SCHEDULER] processing  model=%s  request_id=%s",
                req.base_model, req.request_id,
            )

            try:
                result = await self._loader.load(req)
                req.set_result(result)

            except LoadError as exc:
                logger.error(
                    "[SCHEDULER] load failed  model=%s  error=%s",
                    req.base_model, exc,
                )

                # ── Attempt fallback ──────────────────────────────────
                fallback_result = None
                if self._fallback is not None:
                    try:
                        fallback_result = await self._fallback.try_fallback(
                            req.base_model, req.gpu
                        )
                    except Exception as fb_exc:
                        logger.error(
                            "[SCHEDULER] fallback raised  model=%s  error=%s",
                            req.base_model, fb_exc,
                        )

                if fallback_result:
                    # Resolve the original future with the fallback's result
                    req.set_result({
                        **fallback_result,
                        "_fallback": True,
                        "_original_model": req.base_model,
                    })
                else:
                    req.set_exception(exc)

            except Exception as exc:
                logger.error(
                    "[SCHEDULER] unexpected error  model=%s  error=%s",
                    req.base_model, exc, exc_info=True,
                )
                req.set_exception(exc)

            finally:
                self._active_model = None
                self._queue.task_done()
