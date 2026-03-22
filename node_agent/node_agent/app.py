"""
node_agent/app.py

FastAPI application entry point for the Node Agent.

One instance runs on every GPU server, alongside the local MRM.

Startup sequence
----------------
1. Resolve node_id (from config or hostname).
2. Open the local MRM HTTP client.
3. Send an immediate heartbeat so the scheduler knows we're alive.
4. Start the background heartbeat loop.

Shutdown sequence
-----------------
1. Cancel the heartbeat task.
2. Close the MRM HTTP client.
"""

from __future__ import annotations

import asyncio
import socket
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from node_agent.config import settings
from node_agent.heartbeat import heartbeat_loop, send_heartbeat
from node_agent.mrm_client import local_mrm
from node_agent.observability import (
    instrument_fastapi,
    metrics_app,
    setup_logging,
    setup_tracing,
)
from node_agent.routes import local as local_routes

setup_logging(settings.otel_service_name)

log = structlog.get_logger(__name__)


def _resolve_node_id() -> str:
    """Use NODE_AGENT_NODE_ID if set, otherwise fall back to hostname."""
    if settings.node_id:
        return settings.node_id
    return socket.gethostname()


@asynccontextmanager
async def lifespan(app: FastAPI):
    node_id = _resolve_node_id()
    app.state.node_id = node_id

    log.info(
        "node_agent_starting",
        node_id=node_id,
        agent_url=settings.agent_url,
        mrm=settings.mrm_url,
        scheduler=settings.scheduler_url,
    )

    setup_tracing(settings.otel_service_name, settings.otel_endpoint)
    instrument_fastapi(app)

    await local_mrm.setup()

    try:
        await send_heartbeat(node_id)
    except Exception as exc:
        log.warning("initial_heartbeat_failed", error=str(exc))

    hb_task = asyncio.create_task(heartbeat_loop(node_id))

    log.info("node_agent_ready", node_id=node_id, otel=settings.otel_endpoint or "disabled")

    yield

    log.info("node_agent_stopping", node_id=node_id)
    hb_task.cancel()
    try:
        await hb_task
    except asyncio.CancelledError:
        pass
    await local_mrm.teardown()
    log.info("node_agent_stopped", node_id=node_id)


app = FastAPI(
    title="Node Agent",
    description=(
        "Per-server sidecar for distributed inference. "
        "Wraps local MRM and reports GPU state to the Scheduler."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.mount("/metrics", metrics_app)

app.include_router(local_routes.router, tags=["Local"])


@app.get("/health", tags=["Operations"], summary="Node Agent liveness probe")
async def health():
    """Returns 200 immediately."""
    return {"status": "ok", "node_id": app.state.node_id}


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Node Agent",
        "node_id": app.state.node_id,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }
