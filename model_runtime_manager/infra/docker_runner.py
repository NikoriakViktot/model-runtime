"""
infra/docker_runner.py

Thin wrapper over the Docker SDK.

Responsibilities
----------------
- Start, stop, and remove containers
- Query container status and logs
- Present a clean, typed interface to callers

Not responsible for
-------------------
- Engine-specific configuration (that belongs in EngineAdapter)
- Runtime state tracking (that belongs in ModelRuntimeManager)
- Business logic of any kind

Why Docker SDK instead of subprocess
-------------------------------------
The Docker SDK provides a typed Python API, structured error types
(``docker.errors.NotFound``, ``docker.errors.APIError``), and direct access
to container objects without shell-escaping risks.

All methods are **synchronous**.  Async callers (e.g. RuntimeManager) should
wrap calls with ``asyncio.to_thread()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import docker
import docker.errors
import docker.types
from docker import DockerClient
from docker.models.containers import Container  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)


class DockerRunError(Exception):
    """Raised when a container fails to start."""


class DockerNotFoundError(Exception):
    """Raised when a requested container does not exist."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ContainerInfo:
    """
    Minimal container metadata returned by :class:`DockerRunner`.

    Callers should not depend on Docker SDK types directly.
    """

    id: str
    name: str
    status: str     # "running" | "exited" | "created" | "paused" | "restarting"
    image: str


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class DockerRunner:
    """
    Manages Docker container lifecycle.

    Instantiate once and inject where needed::

        runner = DockerRunner()
        info = runner.run_container(name="my-container", image="ubuntu", ...)
    """

    def __init__(self, client: Optional[DockerClient] = None) -> None:
        """
        Args:
            client: An existing Docker client.  If ``None``, one is created
                    from the environment (``DOCKER_HOST``, socket, etc.).
        """
        self._client = client or docker.from_env()

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    def run_container(
        self,
        *,
        name: str,
        image: str,
        command: list[str],
        environment: dict[str, str],
        volumes: dict[str, dict],
        ports: dict[str, int],
        gpu_index: str,
        network: Optional[str] = None,
        shm_size: str = "1g",
        ipc_mode: Optional[str] = None,
        labels: Optional[dict[str, str]] = None,
        entrypoint: Optional[list[str]] = None,
    ) -> ContainerInfo:
        """
        Start a container in detached mode and return its metadata.

        Args:
            name:        Container name (must be unique on the host).
            image:       Docker image reference, e.g. ``"vllm/vllm-openai:latest"``.
            command:     Command / argv to pass to the container.
            environment: Environment variable dict.
            volumes:     ``{host_path: {"bind": container_path, "mode": "ro"}}``
            ports:       ``{"container_port/tcp": host_port}``
            gpu_index:   GPU index string to expose via NVIDIA Container Toolkit.
            network:     Docker network name (optional).
            shm_size:    Shared memory size, e.g. ``"1g"`` or ``"8gb"``.
            ipc_mode:    IPC namespace, e.g. ``"host"`` (recommended for vLLM).
            labels:      Arbitrary labels attached to the container.
            entrypoint:  Override the image's ENTRYPOINT (``None`` keeps default).

        Returns:
            :class:`ContainerInfo` for the started container.

        Raises:
            DockerRunError: If the container cannot be started.
        """
        device_requests = [
            docker.types.DeviceRequest(
                device_ids=[gpu_index],
                capabilities=[["gpu"]],
            )
        ]

        logger.info(
            "Starting container name=%s image=%s gpu=%s",
            name, image, gpu_index,
        )
        logger.debug("Container command: %s", " ".join(command))

        try:
            container: Container = self._client.containers.run(
                image=image,
                command=command,
                name=name,
                environment=environment,
                volumes=volumes,
                ports=ports,
                device_requests=device_requests,
                network=network,
                shm_size=shm_size,
                ipc_mode=ipc_mode,
                labels=labels or {},
                entrypoint=entrypoint,
                detach=True,
                remove=False,   # MRM manages removal explicitly
            )
        except docker.errors.ImageNotFound as exc:
            raise DockerRunError(
                f"Image not found: '{image}'. Pull it first or check the tag."
            ) from exc
        except docker.errors.APIError as exc:
            raise DockerRunError(
                f"Docker API error starting container '{name}': {exc}"
            ) from exc

        logger.info(
            "Container started: name=%s id=%s",
            name, container.id[:12],
        )
        return _to_info(container)

    # ------------------------------------------------------------------
    # Stop / Remove
    # ------------------------------------------------------------------

    def stop_container(self, container_id: str, timeout: int = 30) -> None:
        """
        Gracefully stop a running container (SIGTERM → wait → SIGKILL).

        Silently no-ops if the container is already stopped or not found.

        Args:
            container_id: Full or short container ID.
            timeout:      Seconds to wait for graceful shutdown before SIGKILL.
        """
        container = self._get_raw(container_id)
        if container is None:
            logger.debug("stop_container: %s not found, skipping", container_id[:12])
            return
        try:
            container.stop(timeout=timeout)
            logger.info("Container stopped: %s", container_id[:12])
        except docker.errors.APIError as exc:
            logger.warning("Error stopping container %s: %s", container_id[:12], exc)

    def remove_container(self, container_id: str, *, force: bool = False) -> None:
        """
        Remove a container.

        Silently no-ops if the container does not exist.

        Args:
            container_id: Full or short container ID.
            force:        If ``True``, removes even if the container is running.
        """
        container = self._get_raw(container_id)
        if container is None:
            logger.debug("remove_container: %s not found, skipping", container_id[:12])
            return
        try:
            container.remove(force=force)
            logger.info("Container removed: %s", container_id[:12])
        except docker.errors.APIError as exc:
            logger.warning("Error removing container %s: %s", container_id[:12], exc)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_container(self, container_id: str) -> Optional[ContainerInfo]:
        """
        Return fresh :class:`ContainerInfo` for a container ID.

        Reloads the container state from the Docker daemon before returning.

        Returns:
            :class:`ContainerInfo` or ``None`` if not found.
        """
        container = self._get_raw(container_id)
        if container is None:
            return None
        try:
            container.reload()
        except docker.errors.APIError:
            return None
        return _to_info(container)

    def get_container_by_name(self, name: str) -> Optional[ContainerInfo]:
        """
        Return :class:`ContainerInfo` for a container name.

        Returns:
            :class:`ContainerInfo` or ``None`` if not found.
        """
        try:
            # Anchor both sides to match exact name
            containers = self._client.containers.list(
                all=True,
                filters={"name": f"^/?{name}$"},
            )
            if not containers:
                return None
            containers[0].reload()
            return _to_info(containers[0])
        except docker.errors.APIError as exc:
            logger.warning("Error looking up container by name '%s': %s", name, exc)
            return None

    def is_running(self, container_id: str) -> bool:
        """Return ``True`` if the container exists and is in ``running`` state."""
        info = self.get_container(container_id)
        return info is not None and info.status == "running"

    def get_logs(self, container_id: str, tail: int = 100) -> str:
        """
        Return the last ``tail`` lines of combined stdout/stderr as a string.

        Returns an empty string if the container is not found or logs are
        unavailable.
        """
        container = self._get_raw(container_id)
        if container is None:
            return ""
        try:
            raw = container.logs(tail=tail, stdout=True, stderr=True)
            return raw.decode("utf-8", errors="replace")
        except docker.errors.APIError as exc:
            logger.debug("Could not retrieve logs for %s: %s", container_id[:12], exc)
            return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_raw(self, container_id: str) -> Optional[Container]:
        """Fetch the raw Docker SDK Container object, or None if not found."""
        try:
            return self._client.containers.get(container_id)
        except docker.errors.NotFound:
            return None
        except docker.errors.APIError as exc:
            logger.warning(
                "Docker API error fetching container %s: %s",
                container_id[:12], exc,
            )
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_info(container: Container) -> ContainerInfo:
    """Convert a Docker SDK Container to a ContainerInfo."""
    tags = container.image.tags if container.image else []
    image = tags[0] if tags else (container.image.id[:12] if container.image else "unknown")
    return ContainerInfo(
        id=container.id,
        name=container.name,
        status=container.status,
        image=image,
    )
