"""
scheduler/app.py

FastAPI application entry point for the Distributed Inference Scheduler.

Responsibilities
----------------
- Open/close the Redis connection pool on startup/shutdown.
- Wire NodeRegistry, PlacementStore, and Scheduler together.
- Mount all routers.
- Provide /health, /ready, and /metrics probes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from redis.asyncio import Redis, from_url

from scheduler.config import settings
from scheduler.observability import (
    instrument_fastapi,
    metrics_app,
    setup_logging,
    setup_tracing,
)
from scheduler.placements import PlacementStore
from scheduler.registry import NodeRegistry
from scheduler.routes import nodes as nodes_routes
from scheduler.routes import schedule as schedule_routes
from scheduler.scheduler import Scheduler

setup_logging(settings.otel_service_name)

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("scheduler_starting", redis=settings.redis_url)

    setup_tracing(settings.otel_service_name, settings.otel_endpoint)
    instrument_fastapi(app)

    redis: Redis = from_url(settings.redis_url, decode_responses=True)
    registry = NodeRegistry(redis)
    placements = PlacementStore(redis)
    scheduler = Scheduler(registry, placements)

    app.state.scheduler = scheduler

    log.info(
        "scheduler_ready",
        strategy=settings.placement_strategy,
        node_ttl=settings.node_ttl_sec,
        otel=settings.otel_endpoint or "disabled",
    )

    yield

    log.info("scheduler_stopping")
    await redis.aclose()
    log.info("scheduler_stopped")


app = FastAPI(
    title="Distributed Inference Scheduler",
    description=(
        "Central placement service for distributed AI model inference. "
        "Maintains node registry and model placement map via Redis."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.mount("/metrics", metrics_app)

app.include_router(nodes_routes.router)
app.include_router(schedule_routes.router)


@app.get("/health", tags=["Operations"], summary="Scheduler liveness probe")
async def health():
    """Returns 200 immediately. Does not check Redis."""
    return {"status": "ok"}


@app.get("/ready", tags=["Operations"], summary="Scheduler readiness probe")
async def ready():
    """Returns 200 if Redis is reachable, 503 otherwise."""
    try:
        redis: Redis = app.state.scheduler._registry._redis
        await redis.ping()
        return {"status": "ready", "redis": settings.redis_url}
    except Exception as exc:
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": str(exc)},
        )


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "Distributed Inference Scheduler",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "metrics": "/metrics",
        "nodes": "/nodes",
        "placements": "/placements",
    }
