"""
cpu_runtime/app.py

FastAPI application for the CPU inference service.

Single responsibility: expose POST /v1/chat/completions backed by
llama.cpp (GGUF models) in an OpenAI-compatible format.

The gateway treats this instance exactly like a vLLM instance:
same endpoint path, same response schema, same SSE streaming format.

Starting the server
-------------------
    uvicorn cpu_runtime.app:app --host 0.0.0.0 --port 8090
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import uuid
from contextlib import asynccontextmanager

import structlog
import structlog.contextvars as _ctxvars
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from cpu_runtime import inference as _inference_module
from cpu_runtime import state as _state
from cpu_runtime.config import settings
from cpu_runtime.inference import LlamaCppEngine
from cpu_runtime.load_shedder import shedder as _shedder
from cpu_runtime.observability import metrics_app
from cpu_runtime.routes import chat

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------


def _check_free_ram_mb() -> int:
    """Return free RAM in MiB at startup (not cached). -1 if unavailable."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return -1


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Reset shutdown flag (important for test isolation)
    _state.shutting_down = False

    # --- Signal handlers for graceful shutdown ---
    loop = asyncio.get_running_loop()

    def _on_shutdown_signal():
        _state.shutting_down = True
        log.info("shutdown_signal_received")

    try:
        loop.add_signal_handler(signal.SIGTERM, _on_shutdown_signal)
        loop.add_signal_handler(signal.SIGINT, _on_shutdown_signal)
    except (NotImplementedError, OSError):
        pass  # Windows / restricted environments

    # --- RAM guard (startup) ---
    free_mb = _check_free_ram_mb()
    if free_mb >= 0 and free_mb < settings.min_free_ram_mb:
        log.warning(
            "cpu_runtime_low_ram",
            free_mb=free_mb,
            min_free_mb=settings.min_free_ram_mb,
        )
    elif free_mb >= 0:
        log.info("cpu_runtime_ram_ok", free_mb=free_mb)

    log.info(
        "cpu_runtime_starting",
        model_path=settings.model_path,
        model_alias=settings.model_alias,
        n_ctx=settings.n_ctx,
        n_threads=settings.n_threads,
    )

    # --- Model load (non-fatal: service stays up to return proper 503/ready) ---
    _inference_module.load_state = "loading"
    engine = LlamaCppEngine(settings)
    try:
        if not os.path.exists(settings.model_path):
            raise FileNotFoundError(
                f"GGUF model not found at {settings.model_path!r}"
            )
        await engine.load()
        _inference_module.engine = engine
        _inference_module.load_state = "loaded"

        # Startup summary log — all tuneable parameters in one line
        log.info(
            "cpu_runtime_ready",
            model_alias=settings.model_alias,
            n_ctx=settings.n_ctx,
            n_threads=settings.n_threads,
            max_queue_depth=settings.max_queue_depth,
            generation_timeout_sec=settings.generation_timeout_sec,
            max_prompt_chars=settings.max_prompt_chars,
            max_total_tokens=settings.max_total_tokens,
            min_free_ram_mb=settings.min_free_ram_mb,
            latency_threshold_ms=settings.latency_threshold_ms,
            dynamic_concurrency_enabled=settings.dynamic_concurrency_enabled,
            free_ram_mb=free_mb,
        )
    except FileNotFoundError as exc:
        _inference_module.load_state = "not_found"
        _inference_module.load_error = str(exc)
        log.error("cpu_runtime_model_not_found", error=str(exc))
    except Exception as exc:
        _inference_module.load_state = "failed"
        _inference_module.load_error = str(exc)
        log.error("cpu_runtime_load_failed", error=str(exc))

    yield

    # --- Graceful shutdown ---
    _state.shutting_down = True
    log.info("shutdown_started")

    timeout = settings.shutdown_timeout_sec
    t_start = loop.time()
    while chat._active_requests > 0:
        elapsed = loop.time() - t_start
        if elapsed >= timeout:
            log.warning(
                "shutdown_drain_timeout",
                active_requests=chat._active_requests,
                timeout_sec=timeout,
            )
            break
        log.info("waiting_for_active_requests", count=chat._active_requests)
        await asyncio.sleep(1.0)

    log.info("shutdown_complete", active_requests=chat._active_requests)

    if _inference_module.engine is not None:
        await engine.unload()
    log.info("cpu_runtime_stopped")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CPU Inference Runtime",
    description=(
        "OpenAI-compatible inference endpoint backed by llama.cpp (GGUF). "
        "Drop-in replacement for vLLM in CPU-only deployments."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/metrics", metrics_app)

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


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """
    Propagate X-Request-ID across the request lifecycle.

    Reads from the incoming header (gateway already set it) or generates
    a fresh ID for requests arriving without one (e.g. direct clients).
    Binds to structlog context so every log line from this coroutine
    automatically includes the request_id.
    """
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    _ctxvars.clear_contextvars()
    _ctxvars.bind_contextvars(request_id=request_id)
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


app.include_router(chat.router, tags=["Chat"])


# ---------------------------------------------------------------------------
# Health / ready / models
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Operations"])
async def health():
    """Liveness probe. Always returns 200 — confirms the process is alive."""
    return {"status": "ok"}


@app.get("/ready", tags=["Operations"])
async def ready():
    """Readiness probe. Returns 200 only when the model is fully loaded."""
    state = _inference_module.load_state
    if state == "loaded":
        # RAM guard: critically low memory → not ready for traffic
        if settings.low_ram_mode_enabled:
            free_mb = _shedder.check_ram()
            if free_mb >= 0 and free_mb < settings.min_free_ram_mb:
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "low_memory",
                        "free_mb": free_mb,
                        "min_mb": settings.min_free_ram_mb,
                    },
                )
        return {"status": "ready", "model": settings.model_alias}
    if state in ("not_found", "failed"):
        return JSONResponse(
            status_code=503,
            content={
                "status": state,
                "error": _inference_module.load_error,
            },
        )
    # "loading" or "not_started"
    return JSONResponse(
        status_code=503,
        content={"status": state},
    )


@app.get("/v1/models", tags=["Models"])
async def list_models():
    """OpenAI-compatible model list. Returns the single loaded GGUF model."""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_alias,
                "object": "model",
                "created": 0,
                "owned_by": "cpu-runtime",
                "runtime_type": "cpu",
            }
        ],
    }


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "CPU Inference Runtime",
        "model": settings.model_alias,
        "runtime": "llama.cpp",
        "health": "/health",
        "ready": "/ready",
        "metrics": "/metrics",
        "docs": "/docs",
    }
