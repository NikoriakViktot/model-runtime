"""
gateway/services/mrm_client.py

Async HTTP client for the Model Runtime Manager.

Wraps MRM's REST API in a typed, async interface.  The gateway routes
import the module-level ``mrm`` singleton and never construct HTTP requests
themselves.

MRM endpoint contract (from runtime.py / app.py):
  POST /models/ensure   {"base_model": str}
    → {"base_model", "model_alias", "api_base", "container", "gpu", "state"}
  GET  /models/status   → list of the above
  GET  /models/status/{model}  → single status dict
  POST /factory/provision  {"repo_id", "preset", "gpu", "overrides"}
    → {"registered", "container", ...}
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from gateway.services.router import InstanceInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class EnsureResult:
    """Parsed response from POST /models/ensure."""

    base_model: str
    model_alias: str
    api_base: str       # includes /v1, e.g. "http://container:8000/v1"
    container: str
    gpu: str
    state: str
    # Multi-instance: one element for current MRM; multiple when MRM
    # returns an ``instances`` array.  Always non-empty.
    instances: list[InstanceInfo] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EnsureResult:
        api_base: str = d["api_base"]
        gpu: str = d.get("gpu", "")

        # New MRM format: explicit list of instances
        if "instances" in d:
            instances = [InstanceInfo.from_dict(i) for i in d["instances"]]
        else:
            # Current single-instance format: synthesise from api_base / gpu
            instances = [InstanceInfo(api_base=api_base, gpu=gpu)]

        return cls(
            base_model=d["base_model"],
            model_alias=d.get("model_alias", d["base_model"]),
            api_base=api_base,
            container=d.get("container", ""),
            gpu=gpu,
            state=d.get("state", "READY"),
            instances=instances,
        )


@dataclass
class ModelStatus:
    """Parsed response from GET /models/status[/{model}]."""

    base_model: str
    model_alias: str
    api_base: str
    state: str
    running: bool
    gpu: str
    active_loras: list[str]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelStatus:
        return cls(
            base_model=d.get("base_model", ""),
            model_alias=d.get("model_alias", ""),
            api_base=d.get("api_base", ""),
            state=d.get("state", "ABSENT"),
            running=bool(d.get("running", False)),
            gpu=d.get("gpu", ""),
            active_loras=d.get("active_loras", []),
        )


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MRMError(Exception):
    """Base exception for MRM client errors."""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class ModelNotFoundError(MRMError):
    """Model is not registered in MRM's registry."""


class ModelLockedError(MRMError):
    """Another operation is in progress for this model."""


class MRMUnavailableError(MRMError):
    """MRM service is unreachable."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MRMClient:
    """
    Async client for the Model Runtime Manager.

    Lifecycle is managed by the FastAPI lifespan:
        await mrm.setup()   # opens connection pool
        await mrm.teardown()  # closes it
    """

    def __init__(self) -> None:
        self._http: httpx.AsyncClient | None = None

    async def setup(self, base_url: str, ensure_timeout: float) -> None:
        """Open the underlying httpx connection pool."""
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            # ensure_timeout covers model cold-start (up to ~600s)
            timeout=httpx.Timeout(
                connect=10.0,
                read=ensure_timeout,
                write=30.0,
                pool=10.0,
            ),
        )
        logger.info("MRM client ready: base_url=%s", base_url)

    async def teardown(self) -> None:
        """Close the connection pool gracefully."""
        if self._http:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ensure(self, model: str) -> EnsureResult:
        """
        POST /models/ensure — blocks until the model container is READY.

        MRM's ensure endpoint handles cold-start, GPU allocation, and the
        OOM fallback ladder internally.  This call may take several minutes
        for large models.

        Raises:
            ModelNotFoundError:  MRM does not know this model.
            ModelLockedError:    A concurrent operation is in progress.
            MRMUnavailableError: MRM is unreachable.
            MRMError:            Any other MRM error.
        """
        self._assert_ready()
        logger.info("Ensuring model=%s", model)
        try:
            resp = await self._http.post(
                "/models/ensure",
                json={"base_model": model},
            )
        except httpx.ConnectError as exc:
            raise MRMUnavailableError(f"Cannot reach MRM: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise MRMUnavailableError(
                f"MRM timed out while ensuring '{model}'"
            ) from exc

        self._raise_for_status(resp, model)
        return EnsureResult.from_dict(resp.json())

    async def status(self, model: str) -> ModelStatus:
        """
        GET /models/status/{model} — return current runtime status.

        Raises:
            ModelNotFoundError: Model is not in MRM's registry.
            MRMUnavailableError: MRM is unreachable.
        """
        self._assert_ready()
        try:
            resp = await self._http.get(
                f"/models/status/{model}",
                timeout=30.0,
            )
        except httpx.ConnectError as exc:
            raise MRMUnavailableError(f"Cannot reach MRM: {exc}") from exc

        self._raise_for_status(resp, model)
        return ModelStatus.from_dict(resp.json())

    async def status_all(self) -> list[ModelStatus]:
        """
        GET /models/status — return status of all registered models.

        Returns an empty list if MRM has no registered models.
        """
        self._assert_ready()
        try:
            resp = await self._http.get("/models/status", timeout=30.0)
        except httpx.ConnectError as exc:
            raise MRMUnavailableError(f"Cannot reach MRM: {exc}") from exc

        resp.raise_for_status()
        data = resp.json()
        # MRM returns either a list or {"models": [...]}
        if isinstance(data, list):
            return [ModelStatus.from_dict(d) for d in data]
        return [ModelStatus.from_dict(d) for d in data.get("models", [])]

    async def provision(
        self,
        repo_id: str,
        preset: str,
        gpu: str,
        overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        POST /factory/provision — register a HuggingFace model on-the-fly.

        Used when ``auto_provision=True`` in settings and a model is not yet
        in MRM's registry.
        """
        self._assert_ready()
        payload = {
            "repo_id": repo_id,
            "preset": preset,
            "gpu": gpu,
            "overrides": overrides or {},
        }
        try:
            resp = await self._http.post(
                "/factory/provision",
                json=payload,
                timeout=60.0,
            )
        except httpx.ConnectError as exc:
            raise MRMUnavailableError(f"Cannot reach MRM: {exc}") from exc

        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assert_ready(self) -> None:
        if self._http is None:
            raise RuntimeError(
                "MRMClient has not been initialized. "
                "Call await mrm.setup() in the application lifespan."
            )

    @staticmethod
    def _raise_for_status(resp: httpx.Response, model: str) -> None:
        if resp.status_code == 404:
            raise ModelNotFoundError(
                f"Model '{model}' is not registered in MRM.",
                status_code=404,
            )
        if resp.status_code == 409:
            body = _safe_json(resp)
            raise ModelLockedError(
                f"Model '{model}' is locked: {body}",
                status_code=409,
            )
        if resp.status_code >= 500:
            body = _safe_json(resp)
            raise MRMError(
                f"MRM returned {resp.status_code} for '{model}': {body}",
                status_code=resp.status_code,
            )
        resp.raise_for_status()


# ---------------------------------------------------------------------------
# Module-level singleton — initialized in main.py lifespan
# ---------------------------------------------------------------------------

mrm = MRMClient()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return resp.text
