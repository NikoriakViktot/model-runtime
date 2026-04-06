"""
mrm/core/health.py

Background health watcher.

Runs an async loop every POLL_INTERVAL seconds.  For each model in the
registry that has a container tracked by MRM it:

  1. Checks whether the Docker container is running.
  2. If running, calls the vLLM /v1/models endpoint (5 s timeout).
  3. Records consecutive failures with exponential back-off before marking FAILED.
  4. If the container is gone → STOPPED.
  5. Emits structured [HEALTH] log lines.

Failure thresholds
------------------
  CONSECUTIVE_FAIL_THRESHOLD = 3   probes must fail before state → FAILED
  TIMEOUT_SEC                 = 5   per HTTP probe

Back-off (seconds between probes on a failing model)
  attempts: 0  1  2  3+
  delay:    2  4  8  16  (capped at MAX_BACKOFF)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict

import httpx

from .state_machine import ModelState, StateMachine

logger = logging.getLogger("MRM.health")

POLL_INTERVAL            = 2.0   # seconds between watcher ticks
PROBE_TIMEOUT            = 5.0   # seconds per HTTP health check
CONSECUTIVE_FAIL_THRESHOLD = 3   # consecutive failures before FAILED
MAX_BACKOFF              = 16.0  # max seconds to wait after repeated failures


class HealthWatcher:
    """
    Async background worker that monitors running vLLM containers.

    Requires a reference to the ModelRuntimeManager instance so it can:
      - Iterate the registry
      - Check container state via Docker
      - Trigger state machine transitions
    """

    def __init__(self, mrm, state_machine: StateMachine) -> None:
        self._mrm            = mrm          # ModelRuntimeManager instance
        self._sm             = state_machine
        self._fail_counts:   Dict[str, int]   = {}   # consecutive probe failures
        self._last_probe_at: Dict[str, float] = {}   # last probe timestamp
        self._running        = False
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the background watcher task."""
        self._running = True
        self._task    = asyncio.create_task(self._loop(), name="health_watcher")
        logger.info("[HEALTH] watcher started (interval=%.0fs)", POLL_INTERVAL)

    async def stop(self) -> None:
        """Cancel the background task and wait for it to finish."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[HEALTH] watcher stopped")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._tick()
            except Exception as exc:
                logger.error("[HEALTH] unhandled error in watcher tick: %s", exc, exc_info=True)
            await asyncio.sleep(POLL_INTERVAL)

    async def _tick(self) -> None:
        """One watcher iteration: probe all tracked models."""
        registry = self._mrm.registry
        tasks = [
            asyncio.create_task(self._probe_model(base_model), name=f"probe_{base_model}")
            for base_model in list(registry.keys())
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Per-model probe
    # ------------------------------------------------------------------

    async def _probe_model(self, base_model: str) -> None:
        spec  = self._mrm.registry.get(base_model)
        if not spec:
            return

        current = await self._sm.get(base_model)

        # Only health-check models that are RUNNING.
        # LOADING models are owned by the Loader/Scheduler — don't interfere.
        if current != ModelState.RUNNING:
            self._fail_counts.pop(base_model, None)
            return

        # Apply back-off: if this model has recent failures, skip this tick
        if not self._should_probe(base_model):
            return

        # ── Docker container check ────────────────────────────────────
        container = await asyncio.to_thread(self._mrm._container_get, spec.container_name)
        if container is None:
            await self._handle_missing_container(base_model, current)
            return

        is_running = await asyncio.to_thread(self._mrm._container_is_running, container)
        if not is_running:
            await self._handle_container_stopped(base_model, current, spec.container_name)
            return

        # ── HTTP health probe ─────────────────────────────────────────
        url = f"http://{spec.container_name}:{spec.port}/v1/models"
        ok, latency = await self._http_probe(url)

        self._last_probe_at[base_model] = time.time()

        if ok:
            self._fail_counts[base_model] = 0
            if current != ModelState.RUNNING:
                # Container recovered or finished loading
                try:
                    await self._sm.transition(base_model, ModelState.RUNNING)
                except Exception:
                    pass
            logger.debug(
                "[HEALTH] OK  model=%s  latency=%.0fms",
                base_model, latency * 1000,
            )
        else:
            await self._handle_probe_failure(base_model, current, url)

    # ------------------------------------------------------------------
    # Failure handlers
    # ------------------------------------------------------------------

    async def _handle_missing_container(self, base_model: str, current: ModelState) -> None:
        logger.warning("[HEALTH] container missing  model=%s  current=%s", base_model, current)
        self._fail_counts[base_model] = 0
        if current in (ModelState.RUNNING, ModelState.LOADING):
            try:
                await self._sm.transition(base_model, ModelState.STOPPED)
            except Exception as exc:
                logger.error("[HEALTH] failed to mark STOPPED: %s", exc)

    async def _handle_container_stopped(
        self, base_model: str, current: ModelState, container_name: str
    ) -> None:
        logger.warning(
            "[HEALTH] container stopped  model=%s  container=%s  current=%s",
            base_model, container_name, current,
        )
        self._fail_counts[base_model] = 0
        if current in (ModelState.RUNNING, ModelState.LOADING):
            try:
                await self._sm.transition(base_model, ModelState.STOPPED)
            except Exception as exc:
                logger.error("[HEALTH] failed to mark STOPPED: %s", exc)

    async def _handle_probe_failure(
        self, base_model: str, current: ModelState, url: str
    ) -> None:
        count = self._fail_counts.get(base_model, 0) + 1
        self._fail_counts[base_model] = count

        backoff = min(2.0 ** (count - 1) * POLL_INTERVAL, MAX_BACKOFF)
        logger.warning(
            "[HEALTH] FAIL  model=%s  probe=%s  consecutive=%d  next_in=%.0fs",
            base_model, url, count, backoff,
        )

        if count >= CONSECUTIVE_FAIL_THRESHOLD and current == ModelState.RUNNING:
            logger.error(
                "[HEALTH] MARKING FAILED  model=%s  consecutive_failures=%d",
                base_model, count,
            )
            try:
                await self._sm.transition(base_model, ModelState.FAILED)
            except Exception as exc:
                logger.error("[HEALTH] transition to FAILED error: %s", exc)
            self._fail_counts[base_model] = 0
            # Stop the container so it releases GPU memory. Run in a thread
            # so we don't block the async health-watcher loop.
            asyncio.create_task(
                self._stop_failed_container(base_model),
                name=f"stop_failed_{base_model}",
            )

    async def _stop_failed_container(self, base_model: str) -> None:
        """
        Best-effort container stop after a model is marked FAILED.

        Frees GPU memory so other models can start. Errors are logged
        but never propagated — this is a background cleanup task.
        """
        spec = self._mrm.registry.get(base_model)
        if not spec:
            return
        try:
            container = await asyncio.to_thread(self._mrm._container_get, spec.container_name)
            if container:
                is_running = await asyncio.to_thread(self._mrm._container_is_running, container)
                if is_running:
                    logger.info(
                        "[HEALTH] stopping FAILED container %s to free GPU memory",
                        spec.container_name,
                    )
                    await asyncio.to_thread(self._mrm._container_stop, container, 15)
            # Release GPU reservation — but only if no ensure() is currently
            # holding the lock (OOM ladder may be retrying with this GPU).
            lock_held = await asyncio.to_thread(
                self._mrm.redis.exists, f"mrm:lock:{base_model}"
            )
            if lock_held:
                logger.info(
                    "[HEALTH] skipping GPU release for %s — ensure lock active (OOM retry in progress)",
                    base_model,
                )
            else:
                st = await asyncio.to_thread(self._mrm._get_state, base_model)
                gpu = st.get("gpu", "")
                if gpu:
                    await asyncio.to_thread(self._mrm._gpu_release, gpu, base_model)
                    await asyncio.to_thread(
                        self._mrm._set_state, base_model, {"gpu": ""}
                    )
                    logger.info(
                        "[HEALTH] released GPU %s reservation for FAILED model %s",
                        gpu, base_model,
                    )
        except Exception as exc:
            logger.error(
                "[HEALTH] error stopping FAILED container %s: %s",
                spec.container_name, exc,
            )

    # ------------------------------------------------------------------
    # HTTP probe
    # ------------------------------------------------------------------

    async def _http_probe(self, url: str) -> tuple[bool, float]:
        """
        GET *url* with PROBE_TIMEOUT.
        Returns (ok, elapsed_seconds).
        """
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=PROBE_TIMEOUT) as client:
                r = await client.get(url)
                elapsed = time.monotonic() - t0
                ok = r.status_code < 500
                return ok, elapsed
        except (httpx.TimeoutException, httpx.ConnectError):
            elapsed = time.monotonic() - t0
            return False, elapsed
        except Exception as exc:
            logger.debug("[HEALTH] probe exception url=%s: %s", url, exc)
            elapsed = time.monotonic() - t0
            return False, elapsed

    # ------------------------------------------------------------------
    # Back-off helper
    # ------------------------------------------------------------------

    def _should_probe(self, base_model: str) -> bool:
        """
        Return True if enough time has passed since the last probe given
        the current failure count (exponential back-off).
        """
        count   = self._fail_counts.get(base_model, 0)
        last    = self._last_probe_at.get(base_model, 0.0)
        backoff = min(2.0 ** count * POLL_INTERVAL, MAX_BACKOFF)
        return (time.time() - last) >= backoff
