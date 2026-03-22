"""
engines/base.py

EngineAdapter abstract base class.

Every inference backend (vLLM, llama.cpp, Ollama, …) implements this
interface. The RuntimeManager uses only this interface — it imports no
engine-specific modules.

Responsibilities of an EngineAdapter:
  - Know the engine's startup command and required flags
  - Know which environment variables the engine needs
  - Know the volume layout the engine expects inside the container
  - Know how to detect engine readiness
  - Know the adapter (LoRA) mount convention for this engine
  - Know how to reduce resource usage on OOM (fallback ladder)

Must NOT:
  - Call Docker
  - Manage state
  - Allocate GPUs
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.specs import (
        AdapterConfig,
        AdapterMount,
        FallbackSpec,
        RuntimeSpec,
    )


class EngineAdapter(ABC):
    """
    Abstract base for engine-specific container configuration.

    Subclasses own all knowledge about one inference engine.  The
    RuntimeManager calls these methods to assemble Docker run parameters
    without ever knowing which engine is in use.
    """

    #: Must match the ``EngineType`` enum value string, e.g. ``"vllm"``.
    engine_type: str

    #: Default port the engine binds *inside* the container.
    default_port: int

    # ------------------------------------------------------------------
    # Container specification
    # ------------------------------------------------------------------

    @abstractmethod
    def build_command(
        self,
        spec: RuntimeSpec,
        adapter_mounts: list[AdapterMount],
    ) -> list[str]:
        """
        Return the command (argv) to pass to ``docker run``.

        For images that use an ENTRYPOINT the command becomes the argv
        appended to that entrypoint.  For plain Python images the command
        should start with ``["python", "-m", ...]``.

        Args:
            spec:           The full runtime specification.
            adapter_mounts: Resolved adapter mounts for this run.

        Returns:
            A list of strings forming the container command.
        """

    @abstractmethod
    def build_env(
        self,
        spec: RuntimeSpec,
        gpu_index: str,
    ) -> dict[str, str]:
        """
        Return environment variables to set inside the container.

        Args:
            spec:       The full runtime specification.
            gpu_index:  The GPU index string assigned by the RuntimeManager
                        (e.g. ``"0"`` or ``"1"``).

        Returns:
            A ``{name: value}`` dict of environment variables.
        """

    @abstractmethod
    def build_volumes(
        self,
        spec: RuntimeSpec,
        adapter_mounts: list[AdapterMount],
    ) -> dict[str, dict]:
        """
        Return volume mounts in Docker SDK format.

        Returns:
            ``{host_path: {"bind": container_path, "mode": "ro"|"rw"}}``
        """

    # ------------------------------------------------------------------
    # Readiness detection
    # ------------------------------------------------------------------

    @abstractmethod
    def health_check_url(self, host: str, port: int) -> str:
        """
        Return the URL the RuntimeManager should poll to detect readiness.

        The URL must return HTTP 200 once the engine is serving requests.
        """

    @abstractmethod
    async def is_ready(self, host: str, port: int) -> bool:
        """
        Return ``True`` when the engine is ready to serve inference requests.

        This method must *not* raise on connection errors — return ``False``
        instead.  The RuntimeManager owns the retry loop and timeout logic.
        """

    # ------------------------------------------------------------------
    # Inference routing
    # ------------------------------------------------------------------

    @abstractmethod
    def inference_base_url(self, host: str, port: int) -> str:
        """
        Return the base URL for inference requests.

        Stored in ``RuntimeHandle.api_base`` and used by the Runtime Gateway
        to forward client requests to the correct container.

        Examples:
            - vLLM:     ``"http://host:port/v1"``
            - Ollama:   ``"http://host:port"``
            - llama.cpp: ``"http://host:port/v1"``
        """

    # ------------------------------------------------------------------
    # OOM recovery
    # ------------------------------------------------------------------

    @abstractmethod
    def get_fallback_specs(self, spec: RuntimeSpec) -> list[FallbackSpec]:
        """
        Return an ordered list of reduced-resource specs for OOM recovery.

        Each entry carries field overrides that the RuntimeManager applies to
        ``RuntimeSpec`` via ``dataclasses.replace()`` before retrying.

        Return an empty list if this engine does not support graceful
        degradation (e.g. Ollama manages memory internally).
        """

    # ------------------------------------------------------------------
    # Adapter support
    # ------------------------------------------------------------------

    @abstractmethod
    def supports_adapters(self) -> bool:
        """
        Return ``True`` if this engine supports adapter (LoRA) hot-loading.

        The RuntimeManager checks this before accepting adapter requests and
        raises early with a clear error if ``False``.
        """

    @abstractmethod
    def build_adapter_mount(self, config: AdapterConfig) -> AdapterMount:
        """
        Translate a generic ``AdapterConfig`` to this engine's mount convention.

        Args:
            config: The adapter to mount (id + host path).

        Returns:
            An ``AdapterMount`` with the correct container path for this engine.

        Raises:
            NotImplementedError: If the engine does not support adapters.
        """

    # ------------------------------------------------------------------
    # Docker image
    # ------------------------------------------------------------------

    def default_image(self) -> str:
        """
        Return the default Docker image tag for this engine.

        Subclasses should override this.  The RuntimeManager uses
        ``spec.image`` first, falling back to ``adapter.default_image()``.

        Raises:
            NotImplementedError: If no default image is configured.
        """
        raise NotImplementedError(
            f"No default Docker image configured for engine '{self.engine_type}'. "
            f"Set RuntimeSpec.image explicitly or implement default_image()."
        )
