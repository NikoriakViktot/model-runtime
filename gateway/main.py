"""
gateway/main.py

FastAPI application entry point for the AI Runtime Gateway.

Responsibilities
----------------
- Expose OpenAI-compatible endpoints on top of MRM + vLLM
- Manage HTTP client lifecycles (connection pools)
- Initialise MLflow at startup
- Provide /health and /ready endpoints for orchestration
- Expose /metrics (Prometheus text format)
- Inject X-Request-ID and bind per-request structlog context

Starting the server
-------------------
    uvicorn gateway.main:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

import structlog
import structlog.contextvars as _ctxvars
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gateway.config import settings
from gateway.observability import (
    IN_FLIGHT,
    instrument_fastapi,
    metrics_app,
    new_request_id,
    setup_logging,
    setup_tracing,
)
from gateway.routes import admin, chat, embeddings, models
from gateway.services import mlflow_logger
from gateway.services.mrm_client import mrm
from gateway.services.proxy import proxy
from gateway.services.scheduler_client import scheduler_client

# Initialise structlog before any logger is used
setup_logging(settings.otel_service_name)

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    mode = "distributed (Scheduler)" if settings.use_scheduler else "single-node (MRM)"
    log.info("gateway_starting", mode=mode)

    # OTel — no-op when otel_endpoint is empty
    setup_tracing(settings.otel_service_name, settings.otel_endpoint)
    instrument_fastapi(app)

    if settings.use_scheduler:
        await scheduler_client.setup(
            base_url=settings.scheduler_url,
            timeout=settings.mrm_ensure_timeout,
        )
    else:
        await mrm.setup(
            base_url=settings.mrm_url,
            ensure_timeout=settings.mrm_ensure_timeout,
        )
    await proxy.setup(timeout=settings.proxy_timeout)

    if settings.mlflow_enabled:
        mlflow_logger.setup(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
        )

    log.info(
        "gateway_ready",
        mode=mode,
        backend=settings.scheduler_url if settings.use_scheduler else settings.mrm_url,
        otel=settings.otel_endpoint or "disabled",
    )

    yield

    log.info("gateway_stopping")
    if settings.use_scheduler:
        await scheduler_client.teardown()
    else:
        await mrm.teardown()
    await proxy.teardown()
    log.info("gateway_stopped")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


app = FastAPI(
    title="AI Runtime Gateway",
    description=(
        "OpenAI-compatible API gateway for the AI model runtime platform. "
        "Automatically manages model lifecycle via MRM."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Prometheus metrics — mounted as a sub-app to bypass FastAPI middleware overhead
app.mount("/metrics", metrics_app)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """
    Per-request observability:
      - Generate / propagate X-Request-ID
      - Bind request_id to structlog context vars (async-safe per-coroutine)
      - Track in-flight Prometheus gauge
      - Emit a structured log line for every request
    """
    request_id = request.headers.get("x-request-id") or new_request_id()
    request.state.request_id = request_id

    _ctxvars.clear_contextvars()
    _ctxvars.bind_contextvars(request_id=request_id, path=request.url.path)

    IN_FLIGHT.inc()
    t0 = time.perf_counter()
    request.state.t0 = t0

    response = await call_next(request)

    latency_ms = (time.perf_counter() - t0) * 1000
    IN_FLIGHT.dec()

    log.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        latency_ms=round(latency_ms, 1),
    )

    response.headers["x-request-id"] = request_id
    return response


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(chat.router, tags=["Chat"])
app.include_router(models.router, tags=["Models"])
app.include_router(embeddings.router, tags=["Embeddings"])
app.include_router(admin.router)


# ---------------------------------------------------------------------------
# Health / ready
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Operations"], summary="Gateway liveness probe")
async def health():
    """Returns 200 immediately. Does not check upstream services."""
    return {"status": "ok"}


@app.get("/ready", tags=["Operations"], summary="Gateway readiness probe")
async def ready():
    """Returns 200 if MRM is reachable, 503 otherwise."""
    try:
        await mrm.status_all()
        return {"status": "ready", "mrm": settings.mrm_url}
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": str(exc)},
        )


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "AI Runtime Gateway",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "metrics": "/metrics",
    }
