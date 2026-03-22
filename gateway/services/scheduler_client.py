"""
gateway/services/scheduler_client.py

Async HTTP client for the Distributed Inference Scheduler.

This is a drop-in complement to mrm_client.py.  It returns the same
``EnsureResult`` type so the chat route needs no structural changes —
only the dispatch point changes when ``settings.use_scheduler=True``.

The Scheduler's /schedule/ensure response shape:
    {
        "model_id":    str,
        "model_alias": str,
        "api_base":    str,           # primary instance — backward compat
        "instances": [
            {
                "instance_id": str,
                "model_id":    str,
                "model_alias": str,
                "node_id":     str,
                "agent_url":   str,
                "api_base":    str,   # includes /v1
                "gpu":         str,
                "state":       str,
                "placed_at":   float
            },
            ...
        ]
    }
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

from gateway.services.mrm_client import EnsureResult
from gateway.services.router import InstanceInfo

logger = logging.getLogger(__name__)


class SchedulerUnavailableError(Exception):
    """Scheduler is unreachable."""


class SchedulerError(Exception):
    """Scheduler returned an unexpected error."""

    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class SchedulerClient:
    """
    Async HTTP client for the Scheduler service.

    Usage (in FastAPI lifespan)::

        await scheduler_client.setup(base_url=settings.scheduler_url)
        # ... serve traffic ...
        await scheduler_client.teardown()
    """

    def __init__(self) -> None:
        self._http: httpx.AsyncClient | None = None

    async def setup(self, base_url: str, timeout: float = 700.0) -> None:
        """Open the connection pool."""
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(
                connect=10.0,
                read=timeout,      # covers MRM cold-start on the remote node
                write=30.0,
                pool=10.0,
            ),
        )
        logger.info("SchedulerClient ready: base_url=%s", base_url)

    async def teardown(self) -> None:
        """Close the connection pool."""
        if self._http:
            await self._http.aclose()

    async def ensure(self, model: str) -> EnsureResult:
        """
        POST /schedule/ensure — returns an EnsureResult compatible with
        the existing chat route.

        The Scheduler selects the node, calls the Node Agent, and returns
        the routable api_base.  If the model is already placed, this
        returns immediately.

        Raises:
            SchedulerUnavailableError: Scheduler is unreachable.
            SchedulerError:            Scheduler returned 4xx/5xx.
        """
        self._assert_ready()
        logger.info("scheduler.ensure model=%s", model)
        try:
            resp = await self._http.post(
                "/schedule/ensure",
                json={"model": model},
            )
        except httpx.ConnectError as exc:
            raise SchedulerUnavailableError(f"Cannot reach Scheduler: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise SchedulerUnavailableError(
                f"Scheduler timed out while ensuring '{model}'"
            ) from exc

        if resp.status_code == 503:
            raise SchedulerError(
                f"No nodes available for '{model}': {resp.json()}",
                status_code=503,
            )
        resp.raise_for_status()

        d = resp.json()
        api_base: str = d.get("api_base", "")

        # Build InstanceInfo list for the router
        raw_instances = d.get("instances", [])
        if raw_instances:
            instances = [
                InstanceInfo(
                    api_base=inst["api_base"],
                    gpu=inst.get("gpu", ""),
                )
                for inst in raw_instances
            ]
        else:
            # Fallback: single instance from api_base
            instances = [InstanceInfo(api_base=api_base)]

        return EnsureResult(
            base_model=model,
            model_alias=d.get("model_alias", model),
            api_base=api_base,
            container="",   # managed by remote MRM — not visible here
            gpu=instances[0].gpu if instances else "",
            state="READY",
            instances=instances,
        )

    def _assert_ready(self) -> None:
        if self._http is None:
            raise RuntimeError(
                "SchedulerClient not initialised. "
                "Call await scheduler_client.setup() in the application lifespan."
            )


#: Module-level singleton.  Initialised in main.py lifespan when use_scheduler=True.
scheduler_client = SchedulerClient()
