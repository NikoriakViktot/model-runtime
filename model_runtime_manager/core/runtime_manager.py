"""
core/runtime_manager.py

Model Runtime Manager (MRM).

This is the central orchestrator for AI model container lifecycle.

Responsibilities
----------------
- Start, stop, and remove model containers
- Allocate and release GPU resources
- Track runtime state
- Delegate all engine-specific logic to :class:`EngineAdapter`

Explicitly NOT responsible for
-------------------------------
- vLLM flags, commands, or paths (that lives in ``engines/vllm.py``)
- Inference routing (that belongs in the Runtime Gateway)
- LoRA path conventions (owned by each EngineAdapter)
- Writing LiteLLM config files (Runtime Gateway concern)

Concurrency
-----------
Each model gets its own ``asyncio.Lock``.  Concurrent ``ensure_runtime``
calls for the same model are serialized behind that lock; different models
proceed in parallel.

State storage
-------------
Runtime handles are kept in an in-memory dict.  For multi-node deployments,
replace ``_runtimes`` with a Redis-backed store.  The interface does not
change — only the store implementation.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import replace
from typing import Optional

from core.specs import (
    AdapterMount,
    FallbackSpec,
    RuntimeHandle,
    RuntimeSpec,
    RuntimeState,
)
from engines.registry import EngineNotFoundError, EngineRegistry
from infra.docker_runner import DockerRunError, DockerRunner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RuntimeStartError(Exception):
    """Raised when a runtime fails to start after all attempts."""


class RuntimeNotFoundError(KeyError):
    """Raised when a requested runtime does not exist in the state store."""


class GPUNotAvailableError(Exception):
    """Raised when no GPU is available to back a new runtime."""


class AdapterNotSupportedError(Exception):
    """Raised when an engine does not support adapter loading."""


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_HEALTH_TIMEOUT = 300.0     # seconds to wait for engine readiness
_DEFAULT_POLL_INTERVAL = 2.0        # seconds between health check polls


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class ModelRuntimeManager:
    """
    Engine-agnostic runtime lifecycle manager.

    All engine knowledge is obtained through the :class:`EngineRegistry`;
    this class never imports or references engine modules directly.

    Args:
        engine_registry: Registry of available :class:`EngineAdapter` instances.
        docker_runner:   Infrastructure wrapper for Docker operations.
        available_gpus:  List of GPU index strings available to this manager
                         (e.g. ``["0", "1"]``).
        docker_network:  Docker network to attach containers to.
        health_timeout:  Seconds to wait for a container to become healthy.
        poll_interval:   Seconds between health check polls.
        keep_failed:     If ``True``, do not remove containers that fail startup.
                         Useful for debugging.
    """

    def __init__(
        self,
        engine_registry: EngineRegistry,
        docker_runner: DockerRunner,
        available_gpus: list[str],
        docker_network: Optional[str] = None,
        health_timeout: float = _DEFAULT_HEALTH_TIMEOUT,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        keep_failed: bool = False,
    ) -> None:
        self._registry = engine_registry
        self._docker = docker_runner
        self._network = docker_network
        self._health_timeout = health_timeout
        self._poll_interval = poll_interval
        self._keep_failed = keep_failed

        # Runtime state: model_id → RuntimeHandle
        # Replace with a Redis-backed store for multi-node deployments.
        self._runtimes: dict[str, RuntimeHandle] = {}

        # GPU pool: set of free GPU index strings
        self._gpu_pool: set[str] = set(available_gpus)

        # Per-model asyncio locks — lazily created, never deleted
        self._model_locks: dict[str, asyncio.Lock] = {}
        self._locks_mutex = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ensure_runtime(self, spec: RuntimeSpec) -> RuntimeHandle:
        """
        Ensure a model runtime is running and return a handle to it.

        If the model is already ``READY``, returns immediately without
        acquiring a lock.  Otherwise, starts the container and waits for
        it to become healthy.

        This is the primary entry point for the Runtime Gateway.

        Args:
            spec: Full runtime specification.

        Returns:
            :class:`RuntimeHandle` with ``api_base`` for inference routing.

        Raises:
            RuntimeStartError:        Container failed to become healthy.
            GPUNotAvailableError:     No GPU is free to back the runtime.
            EngineNotFoundError:      No adapter registered for ``spec.engine``.
            AdapterNotSupportedError: Engine does not support requested adapters.
        """
        # Fast path: already running — no lock needed for a dict read
        existing = self._runtimes.get(spec.model_id)
        if existing and existing.state == RuntimeState.READY:
            return existing

        lock = await self._get_model_lock(spec.model_id)
        async with lock:
            # Double-check after acquiring the lock: another coroutine may
            # have started the runtime while we were waiting.
            existing = self._runtimes.get(spec.model_id)
            if existing and existing.state == RuntimeState.READY:
                return existing

            return await self._start_with_fallbacks(spec)

    async def stop_runtime(self, model_id: str) -> None:
        """
        Gracefully stop a running model container and release its GPU.

        Args:
            model_id: Canonical model identifier.

        Raises:
            RuntimeNotFoundError: No runtime is registered for ``model_id``.
        """
        handle = self._runtimes.get(model_id)
        if handle is None:
            raise RuntimeNotFoundError(f"No runtime found for model '{model_id}'")

        lock = await self._get_model_lock(model_id)
        async with lock:
            handle = self._runtimes.get(model_id)
            if handle is None:
                return  # Removed by a concurrent call while we waited

            handle.state = RuntimeState.STOPPING
            logger.info(
                "Stopping runtime model=%s container=%s gpu=%s",
                model_id, handle.container_id[:12], handle.gpu_index,
            )

            await asyncio.to_thread(self._docker.stop_container, handle.container_id)
            self._release_gpu(handle.gpu_index)
            handle.state = RuntimeState.STOPPED

            logger.info("Runtime stopped: model=%s", model_id)

    async def remove_runtime(self, model_id: str) -> None:
        """
        Stop and remove a model container, clearing all tracked state.

        Args:
            model_id: Canonical model identifier.

        Raises:
            RuntimeNotFoundError: No runtime is registered for ``model_id``.
        """
        await self.stop_runtime(model_id)

        handle = self._runtimes.pop(model_id, None)
        if handle:
            await asyncio.to_thread(
                self._docker.remove_container, handle.container_id, force=False
            )
            logger.info("Runtime removed: model=%s", model_id)

    def get_runtime(self, model_id: str) -> Optional[RuntimeHandle]:
        """
        Return the current :class:`RuntimeHandle` for a model, or ``None``.

        Non-blocking.  Does not acquire a lock.
        """
        return self._runtimes.get(model_id)

    def list_runtimes(self) -> list[RuntimeHandle]:
        """Return all tracked runtime handles regardless of state."""
        return list(self._runtimes.values())

    def available_gpu_count(self) -> int:
        """Return the number of currently unallocated GPUs."""
        return len(self._gpu_pool)

    # ------------------------------------------------------------------
    # Internal — start lifecycle
    # ------------------------------------------------------------------

    async def _start_with_fallbacks(self, spec: RuntimeSpec) -> RuntimeHandle:
        """
        Attempt to start the runtime, retrying with OOM fallback specs on failure.

        GPU is allocated before the first attempt and released only on total
        failure.  On success, the handle holds the allocated GPU index.
        """
        adapter = self._registry.get(spec.engine.value)

        if spec.adapters and not adapter.supports_adapters():
            raise AdapterNotSupportedError(
                f"Engine '{spec.engine.value}' does not support adapters, "
                f"but {len(spec.adapters)} adapter(s) were requested."
            )

        gpu_index = self._allocate_gpu()

        logger.info(
            "Starting runtime model=%s engine=%s gpu=%s",
            spec.model_id, spec.engine.value, gpu_index,
        )

        try:
            return await self._attempt_start(spec, gpu_index)
        except RuntimeStartError:
            fallbacks = adapter.get_fallback_specs(spec)
            if not fallbacks:
                self._release_gpu(gpu_index)
                raise RuntimeStartError(
                    f"Model '{spec.model_id}' failed to start and no fallback "
                    f"specs are available for engine '{spec.engine.value}'."
                )

            for fallback in fallbacks:
                reduced = _apply_overrides(spec, fallback.overrides)
                logger.warning(
                    "Retrying model=%s with fallback '%s'",
                    spec.model_id, fallback.label,
                )
                try:
                    return await self._attempt_start(reduced, gpu_index)
                except RuntimeStartError:
                    continue

            self._release_gpu(gpu_index)
            tried = [f.label for f in fallbacks]
            raise RuntimeStartError(
                f"Model '{spec.model_id}' failed after exhausting all fallbacks: {tried}"
            )

    async def _attempt_start(self, spec: RuntimeSpec, gpu_index: str) -> RuntimeHandle:
        """
        Start one container attempt and wait for it to become healthy.

        On failure, removes the container (unless ``keep_failed`` is set)
        and raises :class:`RuntimeStartError`.

        Args:
            spec:      The (possibly reduced) runtime spec to attempt.
            gpu_index: Pre-allocated GPU index string.

        Returns:
            A :class:`RuntimeHandle` in ``READY`` state.

        Raises:
            RuntimeStartError: Container exited or health check timed out.
        """
        adapter = self._registry.get(spec.engine.value)

        # Build engine-specific mounts from generic adapter configs
        adapter_mounts: list[AdapterMount] = [
            adapter.build_adapter_mount(cfg) for cfg in spec.adapters
        ]

        command = adapter.build_command(spec, adapter_mounts)
        env = adapter.build_env(spec, gpu_index)
        volumes = adapter.build_volumes(spec, adapter_mounts)
        image = spec.image or adapter.default_image()
        container_name = _container_name(spec.model_id)

        # Remove stale container with the same name before starting fresh
        await self._remove_stale_container(container_name)

        try:
            container_info = await asyncio.to_thread(
                self._docker.run_container,
                name=container_name,
                image=image,
                command=command,
                environment=env,
                volumes=volumes,
                ports={f"{spec.container_port}/tcp": spec.host_port},
                gpu_index=gpu_index,
                network=self._network,
                shm_size=spec.shm_size,
                ipc_mode=spec.ipc_mode,
            )
        except DockerRunError as exc:
            raise RuntimeStartError(
                f"Docker failed to start container for '{spec.model_id}': {exc}"
            ) from exc

        # Register handle in STARTING state
        handle = RuntimeHandle.new(
            model_id=spec.model_id,
            container_id=container_info.id,
            container_name=container_name,
            engine=spec.engine,
            api_base=adapter.inference_base_url("localhost", spec.host_port),
            host_port=spec.host_port,
            gpu_index=gpu_index,
        )
        self._runtimes[spec.model_id] = handle

        # Wait for the engine to report ready
        ready = await self._wait_for_ready(adapter, "localhost", spec.host_port)

        if not ready:
            logs = await asyncio.to_thread(
                self._docker.get_logs, container_info.id, 150
            )
            status_info = await asyncio.to_thread(
                self._docker.get_container, container_info.id
            )
            container_status = status_info.status if status_info else "unknown"

            logger.error(
                "Runtime health check failed: model=%s container_status=%s\n"
                "Container logs (tail):\n%s",
                spec.model_id, container_status, logs,
            )

            if not self._keep_failed:
                await asyncio.to_thread(
                    self._docker.remove_container, container_info.id, force=True
                )

            handle.state = RuntimeState.FAILED
            raise RuntimeStartError(
                f"Model '{spec.model_id}' did not become healthy within "
                f"{self._health_timeout:.0f}s (container status: {container_status})."
            )

        handle.state = RuntimeState.READY
        logger.info(
            "Runtime READY: model=%s api_base=%s gpu=%s",
            spec.model_id, handle.api_base, gpu_index,
        )
        return handle

    # ------------------------------------------------------------------
    # Internal — health polling
    # ------------------------------------------------------------------

    async def _wait_for_ready(
        self,
        adapter,
        host: str,
        port: int,
    ) -> bool:
        """
        Poll ``adapter.is_ready()`` until the engine is healthy or timeout.

        Returns ``True`` on success, ``False`` on timeout.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + self._health_timeout

        while loop.time() < deadline:
            if await adapter.is_ready(host, port):
                return True
            await asyncio.sleep(self._poll_interval)

        return False

    # ------------------------------------------------------------------
    # Internal — container housekeeping
    # ------------------------------------------------------------------

    async def _remove_stale_container(self, name: str) -> None:
        """Remove any existing container with this name before a fresh start."""
        existing = await asyncio.to_thread(self._docker.get_container_by_name, name)
        if existing:
            logger.debug("Removing stale container: %s", name)
            await asyncio.to_thread(
                self._docker.remove_container, existing.id, force=True
            )

    # ------------------------------------------------------------------
    # Internal — GPU pool
    # ------------------------------------------------------------------

    def _allocate_gpu(self) -> str:
        """
        Claim the first available GPU from the pool.

        Raises:
            GPUNotAvailableError: Pool is empty.
        """
        if not self._gpu_pool:
            busy = [h.model_id for h in self._runtimes.values()
                    if h.state == RuntimeState.READY]
            raise GPUNotAvailableError(
                f"No GPUs available. Currently allocated models: {busy}"
            )
        return self._gpu_pool.pop()

    def _release_gpu(self, gpu_index: str) -> None:
        """Return a GPU index to the pool."""
        self._gpu_pool.add(gpu_index)
        logger.debug("Released GPU %s back to pool", gpu_index)

    # ------------------------------------------------------------------
    # Internal — lock management
    # ------------------------------------------------------------------

    async def _get_model_lock(self, model_id: str) -> asyncio.Lock:
        """Return (or lazily create) the per-model asyncio.Lock."""
        async with self._locks_mutex:
            if model_id not in self._model_locks:
                self._model_locks[model_id] = asyncio.Lock()
            return self._model_locks[model_id]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _container_name(model_id: str) -> str:
    """
    Derive a valid Docker container name from a model_id.

    ``"Qwen/Qwen1.5-1.8B-Chat"``  →  ``"mrm-qwen-qwen1-5-1-8b-chat"``
    """
    sanitized = re.sub(r"[^a-z0-9]", "-", model_id.lower())
    sanitized = re.sub(r"-+", "-", sanitized).strip("-")
    return f"mrm-{sanitized}"


def _apply_overrides(spec: RuntimeSpec, overrides: dict[str, object]) -> RuntimeSpec:
    """
    Return a new :class:`RuntimeSpec` with the given fields replaced.

    Uses :func:`dataclasses.replace` — the original spec is unchanged.
    ``overrides`` keys must be valid ``RuntimeSpec`` field names.
    """
    return replace(spec, **overrides)
