"""
mrm/services/loader.py

Model loading execution service.

Wraps the existing ModelRuntimeManager.ensure_running() with:
  - Pre-flight GPU VRAM check
  - State machine transitions (STOPPED→LOADING→RUNNING | FAILED)
  - Secure HF_TOKEN injection
  - Structured [LOADER] logging
  - Timed execution tracking
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from ..core.gpu import query_gpu_memory_sync, check_fits, estimate_model_vram_mib
from ..core.state_machine import ModelState, StateMachine, InvalidTransitionError
from ..core.queue import LoadRequest

logger = logging.getLogger("MRM.loader")


class ModelLoader:
    """
    Executes model load operations sequentially.

    One Loader instance is shared across the application; the Scheduler
    ensures only one load runs at a time via its asyncio.Lock.
    """

    def __init__(
        self,
        mrm,
        state_machine: StateMachine,
        telemetry=None,   # Optional[TelemetryStore] — injected to avoid circular imports
    ) -> None:
        self._mrm   = mrm
        self._sm    = state_machine
        self._telem = telemetry

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def load(self, req: LoadRequest) -> Dict[str, Any]:
        """
        Execute a full model load cycle for *req*.

        Steps
        -----
        1. Validate model is registered
        2. GPU VRAM pre-flight check
        3. Transition → LOADING
        4. Delegate to mrm.ensure_running() (Docker + health wait)
        5. Transition → RUNNING on success, FAILED on error
        6. Return result dict

        Raises
        ------
        LoadError if any step fails.
        """
        base_model = req.base_model
        spec = self._mrm.registry.get(base_model)

        if spec is None:
            raise LoadError(f"Model not registered: {base_model}")

        logger.info(
            "[LOADER] start  model=%s  preset=%s  gpu=%s  request_id=%s",
            base_model, req.preset, req.gpu, req.request_id,
        )

        # ── 1. GPU VRAM pre-flight ────────────────────────────────────
        await self._check_vram(base_model, spec, req.gpu)

        # ── 2. Transition to LOADING ──────────────────────────────────
        try:
            await self._sm.transition(base_model, ModelState.LOADING)
        except InvalidTransitionError as exc:
            raise LoadError(str(exc)) from exc

        # ── 3. Execute Docker load (blocking, runs in thread) ─────────
        t0 = time.monotonic()
        try:
            result = await asyncio.to_thread(self._mrm.ensure_running, base_model)
            elapsed = time.monotonic() - t0

            # ── 4. Transition to RUNNING ──────────────────────────────
            await self._sm.transition(base_model, ModelState.RUNNING)

            if self._telem is not None:
                self._telem.record_load_success(base_model)

            logger.info(
                "[LOADER] READY  model=%s  elapsed=%.1fs  container=%s",
                base_model, elapsed, result.get("container"),
            )
            return result

        except Exception as exc:
            elapsed = time.monotonic() - t0
            exc_str = str(exc).lower()
            is_oom  = "out of memory" in exc_str or "cuda oom" in exc_str

            logger.error(
                "[LOADER] FAILED  model=%s  elapsed=%.1fs  is_oom=%s  error=%s",
                base_model, elapsed, is_oom, exc,
            )

            if self._telem is not None:
                self._telem.record_load_failure(base_model, is_oom=is_oom)

            # ── 5. Transition to FAILED ───────────────────────────────
            try:
                await self._sm.transition(base_model, ModelState.FAILED)
            except InvalidTransitionError:
                await self._sm.force_set(base_model, ModelState.FAILED)

            raise LoadError(f"Load failed for {base_model}: {exc}") from exc

    # ------------------------------------------------------------------
    # Explicit stop
    # ------------------------------------------------------------------

    async def stop(self, base_model: str) -> Dict[str, Any]:
        """Stop a running model and transition to STOPPED."""
        current = await self._sm.get(base_model)
        if current not in (ModelState.RUNNING, ModelState.FAILED):
            logger.info("[LOADER] stop skipped — model=%s state=%s", base_model, current)
            return {"base_model": base_model, "state": current.value}

        logger.info("[LOADER] stopping  model=%s", base_model)
        try:
            result = await asyncio.to_thread(self._mrm.stop, base_model)
            await self._sm.transition(base_model, ModelState.STOPPED)
            logger.info("[LOADER] stopped  model=%s", base_model)
            return result
        except Exception as exc:
            logger.error("[LOADER] stop failed  model=%s  error=%s", base_model, exc)
            raise

    # ------------------------------------------------------------------
    # VRAM pre-flight
    # ------------------------------------------------------------------

    async def _check_vram(
        self, base_model: str, spec, gpu_id: str
    ) -> None:
        """
        Query GPU free memory and reject the request early if the model
        will not fit, rather than letting the vLLM container OOM-crash.
        """
        gpu_mem = await asyncio.to_thread(query_gpu_memory_sync, gpu_id)

        if gpu_mem.total_mib == 0:
            # nvidia-smi not available — skip the check
            logger.debug("[GPU] VRAM check skipped (nvidia-smi unavailable)")
            return

        required_mib = estimate_model_vram_mib(
            spec.gpu_memory_utilization, gpu_mem.total_mib
        )

        logger.info(
            "[GPU] free=%dMiB  required=%dMiB (util=%.0f%%)  model=%s",
            gpu_mem.free_mib, required_mib,
            spec.gpu_memory_utilization * 100, base_model,
        )

        if not check_fits(required_mib, gpu_mem):
            raise LoadError(
                f"[GPU] Insufficient VRAM for {base_model}: "
                f"free={gpu_mem.free_mib}MiB required={required_mib}MiB "
                f"(gpu_util={spec.gpu_memory_utilization:.0%})"
            )


class LoadError(Exception):
    """Raised by ModelLoader when a load operation cannot proceed."""
