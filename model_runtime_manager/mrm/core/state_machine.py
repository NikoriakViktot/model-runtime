"""
mrm/core/state_machine.py

Strict model lifecycle state machine.

States
------
  STOPPED  — container not running; GPU free
  LOADING  — container starting; GPU reserved; vLLM not yet ready
  WARMING  — OOM fallback ladder in progress (retrying with smaller config)
  RUNNING  — container healthy; serving requests
  FAILED   — container crashed or health-check timed out

Valid transitions
-----------------
  STOPPED  → LOADING             (new load request)
  LOADING  → RUNNING             (health-check passed)
  LOADING  → WARMING             (first OOM retry)
  LOADING  → FAILED              (startup timed out / crash)
  WARMING  → WARMING             (further OOM retries)
  WARMING  → RUNNING             (retry succeeded)
  WARMING  → FAILED              (all retries exhausted)
  WARMING  → STOPPED             (force-stop while warming)
  RUNNING  → STOPPED             (explicit stop)
  RUNNING  → LOADING             (reload / hot-swap)
  RUNNING  → FAILED              (runtime crash detected by health watcher)
  FAILED   → LOADING             (retry / fallback load)
  FAILED   → STOPPED             (give up after max retries)

All transitions are atomic: a Redis HSET is used as the persistent layer;
an in-process asyncio.Lock prevents concurrent transitions for the same model.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger("MRM.state_machine")


# ──────────────────────────────────────────────────────────────────────────────
# States
# ──────────────────────────────────────────────────────────────────────────────

class ModelState(str, Enum):
    STOPPED = "STOPPED"
    LOADING = "LOADING"
    WARMING = "WARMING"   # OOM fallback retry in progress
    RUNNING = "RUNNING"
    FAILED  = "FAILED"


# Map legacy runtime.py Redis state strings → ModelState
_LEGACY_MAP: Dict[str, ModelState] = {
    "ABSENT":   ModelState.STOPPED,
    "STOPPED":  ModelState.STOPPED,
    "STARTING": ModelState.LOADING,
    "READY":    ModelState.RUNNING,
    "STOPPING": ModelState.STOPPED,
    "FAILED":   ModelState.FAILED,
    "LOADING":  ModelState.LOADING,
    "RUNNING":  ModelState.RUNNING,
    "WARMING":  ModelState.WARMING,
}

# Map ModelState → legacy Redis value (for backward compat)
_TO_LEGACY: Dict[ModelState, str] = {
    ModelState.STOPPED: "STOPPED",
    ModelState.LOADING: "STARTING",
    ModelState.WARMING: "WARMING",
    ModelState.RUNNING: "READY",
    ModelState.FAILED:  "FAILED",
}

# ──────────────────────────────────────────────────────────────────────────────
# Valid transition whitelist
# ──────────────────────────────────────────────────────────────────────────────

_VALID: frozenset[tuple[ModelState, ModelState]] = frozenset({
    (ModelState.STOPPED,  ModelState.LOADING),
    (ModelState.LOADING,  ModelState.RUNNING),
    (ModelState.LOADING,  ModelState.WARMING),   # first OOM retry
    (ModelState.LOADING,  ModelState.FAILED),
    (ModelState.LOADING,  ModelState.STOPPED),   # catastrophic: container gone mid-load
    (ModelState.WARMING,  ModelState.WARMING),   # further OOM retries
    (ModelState.WARMING,  ModelState.RUNNING),   # retry succeeded
    (ModelState.WARMING,  ModelState.FAILED),    # all retries exhausted
    (ModelState.WARMING,  ModelState.STOPPED),   # force-stop while warming
    (ModelState.RUNNING,  ModelState.STOPPED),
    (ModelState.RUNNING,  ModelState.LOADING),
    (ModelState.RUNNING,  ModelState.FAILED),
    (ModelState.FAILED,   ModelState.LOADING),
    (ModelState.FAILED,   ModelState.STOPPED),
    # Idempotent self-transitions allowed for health updates
    (ModelState.RUNNING,  ModelState.RUNNING),
    (ModelState.STOPPED,  ModelState.STOPPED),
})


class InvalidTransitionError(Exception):
    """Raised when a requested state transition is not allowed."""


# ──────────────────────────────────────────────────────────────────────────────
# StateMachine
# ──────────────────────────────────────────────────────────────────────────────

class StateMachine:
    """
    Per-model state machine backed by Redis.

    All public methods are async and acquire a per-model asyncio.Lock to
    ensure only one coroutine mutates the state at a time.

    Usage::

        sm = StateMachine(redis_client)
        await sm.transition(base_model, ModelState.LOADING)
        state = await sm.get(base_model)
    """

    def __init__(self, redis_client) -> None:
        self._redis = redis_client
        # Per-model asyncio locks — created lazily
        self._locks: Dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_transition(
        self, current: ModelState, next_state: ModelState
    ) -> None:
        """
        Raise InvalidTransitionError if current→next_state is not allowed.
        """
        if (current, next_state) not in _VALID:
            raise InvalidTransitionError(
                f"[STATE] Invalid transition: {current} → {next_state}"
            )

    async def get(self, base_model: str) -> ModelState:
        """Return the current ModelState for *base_model*."""
        raw = await asyncio.to_thread(
            self._redis.hget, _key(base_model), "state"
        )
        if not raw:
            return ModelState.STOPPED
        return _LEGACY_MAP.get(raw, ModelState.STOPPED)

    async def transition(
        self, base_model: str, next_state: ModelState, *, extra: Optional[Dict] = None
    ) -> ModelState:
        """
        Atomically move *base_model* to *next_state*.

        Validates the transition, writes to Redis, logs the change.
        Returns the previous state.

        Args:
            base_model: HuggingFace repo ID used as registry key.
            next_state: Desired target state.
            extra:      Additional Redis hash fields to write in the same call.
        """
        lock = self._get_lock(base_model)
        async with lock:
            current = await self.get(base_model)
            self.validate_transition(current, next_state)

            updates: Dict[str, str] = {"state": _TO_LEGACY[next_state]}
            if extra:
                updates.update({k: str(v) for k, v in extra.items()})

            await asyncio.to_thread(
                self._redis.hset, _key(base_model), mapping=updates
            )

            if current != next_state:
                logger.info(
                    "[STATE] %s  %s → %s",
                    base_model,
                    current.value,
                    next_state.value,
                )

            return current

    async def force_set(self, base_model: str, state: ModelState) -> None:
        """
        Bypass transition validation and directly write *state*.

        Use only for initialisation or disaster recovery.
        """
        await asyncio.to_thread(
            self._redis.hset,
            _key(base_model),
            mapping={"state": _TO_LEGACY[state]},
        )
        logger.warning("[STATE] force_set %s → %s", base_model, state.value)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_lock(self, base_model: str) -> asyncio.Lock:
        if base_model not in self._locks:
            self._locks[base_model] = asyncio.Lock()
        return self._locks[base_model]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _key(base_model: str) -> str:
    return f"mrm:model:{base_model}"


def normalize_legacy_state(raw: Optional[str]) -> ModelState:
    """Convert a raw Redis state string (legacy or new) to ModelState."""
    if not raw:
        return ModelState.STOPPED
    return _LEGACY_MAP.get(raw.upper(), ModelState.STOPPED)
