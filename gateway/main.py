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

import asyncio
import signal
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
# Shutdown / in-flight tracking
# ---------------------------------------------------------------------------
_gw_shutting_down: bool = False
_gw_in_flight: int = 0


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gw_shutting_down, _gw_in_flight
    _gw_shutting_down = False
    _gw_in_flight = 0

    loop = asyncio.get_running_loop()

    def _on_shutdown():
        global _gw_shutting_down
        _gw_shutting_down = True
        log.info("shutdown_signal_received", service="gateway")

    try:
        loop.add_signal_handler(signal.SIGTERM, _on_shutdown)
        loop.add_signal_handler(signal.SIGINT, _on_shutdown)
    except (NotImplementedError, OSError):
        pass

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
    await proxy.setup(
        timeout=settings.proxy_timeout,
        connect_timeout=settings.connect_timeout,
        read_timeout=settings.read_timeout,
    )

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

    _gw_shutting_down = True
    log.info("shutdown_started", service="gateway")

    timeout = settings.shutdown_timeout_sec
    t_start = loop.time()
    while _gw_in_flight > 0:
        elapsed = loop.time() - t_start
        if elapsed >= timeout:
            log.warning("shutdown_drain_timeout", in_flight=_gw_in_flight, timeout_sec=timeout)
            break
        log.info("waiting_for_requests", in_flight=_gw_in_flight)
        await asyncio.sleep(1.0)

    log.info("shutdown_complete", in_flight=_gw_in_flight, service="gateway")

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

_cors_origins = (
    [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    if settings.cors_origins != "*"
    else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    global _gw_in_flight

    # Fast-path rejects — before consuming any request body
    if _gw_shutting_down:
        from fastapi.responses import JSONResponse as _JR
        return _JR(status_code=503, content={"detail": "Service is shutting down."})

    if settings.max_in_flight > 0 and _gw_in_flight >= settings.max_in_flight:
        from fastapi.responses import JSONResponse as _JR
        return _JR(
            status_code=503,
            content={
                "detail": f"Gateway overloaded: {_gw_in_flight}/{settings.max_in_flight} requests in flight."
            },
        )

    request_id = request.headers.get("x-request-id") or new_request_id()
    request.state.request_id = request_id

    _ctxvars.clear_contextvars()
    _ctxvars.bind_contextvars(request_id=request_id, path=request.url.path)

    _gw_in_flight += 1
    IN_FLIGHT.inc()
    t0 = time.perf_counter()
    request.state.t0 = t0

    try:
        response = await call_next(request)
    finally:
        _gw_in_flight -= 1
        IN_FLIGHT.dec()

    latency_ms = (time.perf_counter() - t0) * 1000

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
    """Returns 200 if the active backend is reachable, 503 otherwise."""
    if settings.use_scheduler:
        result = await scheduler_client.health()
        if result.get("ok"):
            return {"status": "ready", "ok": True, **result}
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "ok": False, **result},
        )
    else:
        try:
            await mrm.status_all()
            return {"status": "ready", "ok": True, "mrm": settings.mrm_url}
        except Exception as exc:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "ok": False, "reason": str(exc)},
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
