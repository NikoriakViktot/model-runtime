"""
node_agent/mrm_client.py

Thin async wrapper around the local Model Runtime Manager.

Design rule: this client forwards requests to MRM verbatim.  It adds no
business logic — all decisions stay in MRM (container lifecycle, GPU
allocation, OOM fallback ladder).

The module-level ``local_mrm`` singleton is initialised in app lifespan.
"""

from __future__ import annotations

import logging

import httpx

from node_agent.config import settings
from node_agent.models import LocalEnsureResponse, LocalStatusResponse

logger = logging.getLogger(__name__)


class LocalMRMClient:
    """
    Async HTTP client for the co-located MRM instance.

    Lifecycle managed by the FastAPI lifespan:
        await local_mrm.setup()
        await local_mrm.teardown()
    """

    def __init__(self) -> None:
        self._http: httpx.AsyncClient | None = None

    async def setup(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=settings.mrm_url.rstrip("/"),
            timeout=httpx.Timeout(
                connect=10.0,
                # Read timeout must exceed MRM_HEALTH_TIMEOUT_SEC (default 600s)
                read=700.0,
                write=30.0,
                pool=10.0,
            ),
        )
        logger.info("LocalMRMClient ready: mrm_url=%s", settings.mrm_url)

    async def teardown(self) -> None:
        if self._http:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ensure(self, model_id: str) -> LocalEnsureResponse:
        """
        POST /models/ensure — blocks until the model container is READY.

        MRM handles cold-start, GPU allocation, and OOM fallback internally.
        May take several minutes for large models.
        """
        self._assert_ready()
        resp = await self._http.post("/models/ensure", json={"base_model": model_id})
        resp.raise_for_status()
        d = resp.json()
        return LocalEnsureResponse(
            model_id=d.get("base_model", model_id),
            model_alias=d.get("model_alias", model_id),
            api_base=d["api_base"],
            gpu=d.get("gpu", ""),
            state=d.get("state", "READY"),
            container=d.get("container", ""),
        )

    async def stop(self, model_id: str) -> None:
        """POST /models/stop — stop and remove the model's container."""
        self._assert_ready()
        resp = await self._http.post(
            "/models/stop",
            json={"base_model": model_id},
            timeout=60.0,
        )
        resp.raise_for_status()

    async def status(self, model_id: str) -> LocalStatusResponse:
        """GET /models/status/{model} — return current runtime status."""
        self._assert_ready()
        resp = await self._http.get(f"/models/status/{model_id}", timeout=30.0)
        if resp.status_code == 404:
            return LocalStatusResponse(model_id=model_id, state="ABSENT", running=False)
        resp.raise_for_status()
        d = resp.json()
        return LocalStatusResponse(
            model_id=model_id,
            state=d.get("state", "UNKNOWN"),
            api_base=d.get("api_base", ""),
            gpu=d.get("gpu", ""),
            running=bool(d.get("running", False)),
            active_loras=d.get("active_loras", []),
        )

    async def status_all(self) -> list[dict]:
        """GET /models/status — return all registered models."""
        self._assert_ready()
        resp = await self._http.get("/models/status", timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return data.get("models", [])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assert_ready(self) -> None:
        if self._http is None:
            raise RuntimeError(
                "LocalMRMClient not initialised — call await local_mrm.setup() in lifespan."
            )


local_mrm = LocalMRMClient()
