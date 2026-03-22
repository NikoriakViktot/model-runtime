"""
engines/registry.py

Engine adapter registry.

Maps engine-type strings to EngineAdapter instances.  The RuntimeManager
resolves adapters exclusively through this registry — it imports no engine
modules directly.

Thread safety
-------------
``register()`` and ``get()`` are protected by a ``threading.Lock`` so the
registry can be safely populated at startup from multiple threads or during
testing.
"""

from __future__ import annotations

import threading

from engines.base import EngineAdapter


class EngineNotFoundError(KeyError):
    """Raised when no adapter is registered for a requested engine type."""


class EngineAlreadyRegisteredError(ValueError):
    """Raised when trying to register an engine type that is already present."""


class EngineRegistry:
    """
    Thread-safe registry of :class:`EngineAdapter` instances.

    Typical usage::

        registry = EngineRegistry()
        registry.register(VLLMAdapter())
        registry.register(LlamaCppAdapter())

        adapter = registry.get("vllm")
    """

    def __init__(self) -> None:
        self._adapters: dict[str, EngineAdapter] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, adapter: EngineAdapter) -> None:
        """
        Register an engine adapter.

        Args:
            adapter: An :class:`EngineAdapter` instance.  Its
                     ``engine_type`` attribute is used as the registry key.

        Raises:
            EngineAlreadyRegisteredError: If ``engine_type`` is already
                registered.  Use :meth:`register_or_replace` to overwrite.
        """
        with self._lock:
            if adapter.engine_type in self._adapters:
                raise EngineAlreadyRegisteredError(
                    f"Engine '{adapter.engine_type}' is already registered. "
                    f"Call register_or_replace() to overwrite."
                )
            self._adapters[adapter.engine_type] = adapter

    def register_or_replace(self, adapter: EngineAdapter) -> None:
        """
        Register or overwrite an engine adapter.

        Useful in tests and during hot-reload scenarios.

        Args:
            adapter: An :class:`EngineAdapter` instance.
        """
        with self._lock:
            self._adapters[adapter.engine_type] = adapter

    def unregister(self, engine_type: str) -> None:
        """
        Remove an engine adapter from the registry.

        Args:
            engine_type: The engine type string to remove.

        Raises:
            EngineNotFoundError: If ``engine_type`` is not registered.
        """
        with self._lock:
            if engine_type not in self._adapters:
                raise EngineNotFoundError(engine_type)
            del self._adapters[engine_type]

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, engine_type: str) -> EngineAdapter:
        """
        Return the adapter registered for ``engine_type``.

        Args:
            engine_type: Engine type string, e.g. ``"vllm"``.

        Returns:
            The registered :class:`EngineAdapter` instance.

        Raises:
            EngineNotFoundError: If no adapter is registered for this type.
        """
        with self._lock:
            adapter = self._adapters.get(engine_type)

        if adapter is None:
            available = self.list_engines()
            raise EngineNotFoundError(
                f"No adapter registered for engine '{engine_type}'. "
                f"Available engines: {available}"
            )

        return adapter

    def is_registered(self, engine_type: str) -> bool:
        """Return ``True`` if an adapter exists for ``engine_type``."""
        with self._lock:
            return engine_type in self._adapters

    def list_engines(self) -> list[str]:
        """Return a sorted list of all registered engine type strings."""
        with self._lock:
            return sorted(self._adapters.keys())


# ---------------------------------------------------------------------------
# Default global registry
# ---------------------------------------------------------------------------
# Populated at application startup, e.g.:
#
#   from engines.registry import default_registry
#   from engines.vllm import VLLMAdapter
#   default_registry.register(VLLMAdapter())

default_registry = EngineRegistry()
