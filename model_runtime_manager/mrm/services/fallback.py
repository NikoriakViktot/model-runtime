"""
mrm/services/fallback.py

Smart fallback: when a model fails, automatically select and load the
next best candidate from the registry.

Selection uses the existing scorer pipeline:
  1. Build ModelMeta stubs from ModelSpec (no external HF call needed)
  2. Filter models that fit in free GPU VRAM
  3. Score by: downloads-proxy (spec order as tie-break) + arch bonus
  4. Skip the failed model
  5. Load the winner via ModelLoader

Limit: MAX_RETRIES fallback attempts per original failure.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from ..core.gpu import query_gpu_memory_sync
from ..core.state_machine import ModelState, StateMachine
from ..core.queue import LoadRequest

logger = logging.getLogger("MRM.fallback")

MAX_RETRIES = 2


class FallbackService:
    """
    Tries to load the next-best registered model when *failed_model* crashes.
    """

    def __init__(self, mrm, state_machine: StateMachine, loader) -> None:
        self._mrm    = mrm
        self._sm     = state_machine
        self._loader = loader   # ModelLoader instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def try_fallback(
        self,
        failed_model: str,
        gpu_id: str = "0",
        attempt: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to load an alternative model after *failed_model* crashed.

        Args:
            failed_model: The base_model that just failed.
            gpu_id:       GPU to load the fallback on.
            attempt:      Current retry depth (stops at MAX_RETRIES).

        Returns:
            Result dict from ModelLoader.load() on success, or None if
            no suitable fallback exists or all retries exhausted.
        """
        if attempt >= MAX_RETRIES:
            logger.error(
                "[FALLBACK] max retries exhausted  failed=%s  max=%d",
                failed_model, MAX_RETRIES,
            )
            return None

        candidates = await self._select_candidates(failed_model, gpu_id)
        if not candidates:
            logger.warning(
                "[FALLBACK] no suitable candidates  failed=%s  gpu=%s",
                failed_model, gpu_id,
            )
            return None

        for candidate in candidates:
            base_model = candidate["base_model"]
            logger.info(
                "[FALLBACK] switching to model=%s  (attempt=%d/%d)",
                base_model, attempt + 1, MAX_RETRIES,
            )

            req = LoadRequest(
                base_model=base_model,
                preset=candidate.get("preset", "small_chat"),
                gpu=gpu_id,
            )

            try:
                result = await self._loader.load(req)
                logger.info(
                    "[FALLBACK] SUCCESS  model=%s  attempt=%d",
                    base_model, attempt + 1,
                )
                return result
            except Exception as exc:
                logger.warning(
                    "[FALLBACK] candidate failed  model=%s  error=%s  trying next",
                    base_model, exc,
                )
                # Recursive retry with next candidate if retries remain
                return await self.try_fallback(base_model, gpu_id, attempt + 1)

        return None

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    async def _select_candidates(
        self, failed_model: str, gpu_id: str
    ) -> List[Dict[str, Any]]:
        """
        Return an ordered list of candidate model dicts to try.

        Selection criteria:
        1. Must be registered (in registry)
        2. Must not be the failed model
        3. Must not currently be LOADING or RUNNING (no sense loading twice)
        4. Must fit in free GPU VRAM (estimated from spec)
        5. Ordered by VRAM headroom (smallest fit first → conservative)
        """
        gpu_mem = await asyncio.to_thread(query_gpu_memory_sync, gpu_id)
        free_mib = gpu_mem.free_mib
        headroom_mib = 512

        registry = self._mrm.registry
        candidates = []

        for base_model, spec in registry.items():
            if base_model == failed_model:
                continue

            state = await self._sm.get(base_model)
            if state in (ModelState.LOADING, ModelState.RUNNING):
                continue

            # Estimate VRAM requirement from spec
            if gpu_mem.total_mib > 0:
                required_mib = int(gpu_mem.total_mib * spec.gpu_memory_utilization)
                if required_mib > (free_mib - headroom_mib):
                    logger.debug(
                        "[FALLBACK] skip model=%s  required=%dMiB  free=%dMiB",
                        base_model, required_mib, free_mib,
                    )
                    continue
            else:
                required_mib = 0  # nvidia-smi unavailable, allow all

            # Infer preset from quantization
            preset = "7b_awq" if spec.quantization in ("awq", "gptq") else "small_chat"

            candidates.append({
                "base_model":    base_model,
                "preset":        preset,
                "required_mib":  required_mib,
                "gpu_util":      spec.gpu_memory_utilization,
            })

        # Sort by required_mib ascending (smallest / safest first)
        candidates.sort(key=lambda x: x["required_mib"])
        return candidates
