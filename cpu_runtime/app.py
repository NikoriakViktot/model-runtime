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

import logging
import os
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from cpu_runtime import inference as _inference_module
from cpu_runtime.config import settings
from cpu_runtime.inference import LlamaCppEngine
from cpu_runtime.observability import metrics_app
from cpu_runtime.routes import chat

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------


def _check_free_ram_mb() -> int:
    """Return free RAM in MiB, or -1 if unavailable."""
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
    # --- RAM guard ---
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
        log.info("cpu_runtime_ready", model_alias=settings.model_alias)
    except FileNotFoundError as exc:
        _inference_module.load_state = "not_found"
        _inference_module.load_error = str(exc)
        log.error("cpu_runtime_model_not_found", error=str(exc))
    except Exception as exc:
        _inference_module.load_state = "failed"
        _inference_module.load_error = str(exc)
        log.error("cpu_runtime_load_failed", error=str(exc))

    yield

    log.info("cpu_runtime_stopping")
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

app.include_router(chat.router, tags=["Chat"])


# ---------------------------------------------------------------------------
# Health / models
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
