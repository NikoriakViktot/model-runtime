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
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(
        "cpu_runtime_starting",
        model_path=settings.model_path,
        model_alias=settings.model_alias,
        n_ctx=settings.n_ctx,
        n_threads=settings.n_threads,
    )

    engine = LlamaCppEngine(settings)
    await engine.load()

    # Expose via module-level so the route can call get_engine()
    _inference_module.engine = engine

    log.info("cpu_runtime_ready", model_alias=settings.model_alias)

    yield

    log.info("cpu_runtime_stopping")
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, tags=["Chat"])


# ---------------------------------------------------------------------------
# Health / models
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Operations"])
async def health():
    """Liveness probe. Returns 200 once the model is loaded."""
    if _inference_module.engine is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "ok", "model": settings.model_alias}


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
        "metrics": "/metrics",
        "docs": "/docs",
    }
