"""
cpu_runtime/routes/chat.py

POST /v1/chat/completions — OpenAI-compatible endpoint backed by llama.cpp.

This is intentionally a thin route: all inference logic lives in
cpu_runtime.inference.LlamaCppEngine.

Supports:
  - Unary responses (stream=false, default)
  - Server-Sent Events streaming (stream=true)
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from cpu_runtime.inference import GenerationRequest, get_engine
from cpu_runtime.observability import (
    CPU_ERRORS_TOTAL,
    CPU_INFERENCE_LATENCY,
    CPU_INFERENCE_TOTAL,
    CPU_QUEUE_DEPTH,
)

log = structlog.get_logger(__name__)
router = APIRouter()

_MAX_TOKENS_HARD_LIMIT = 4096


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    """OpenAI-compatible chat completions via llama.cpp / GGUF."""
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Request body must be valid JSON.")

    messages = body.get("messages")
    if not messages:
        raise HTTPException(status_code=422, detail="'messages' field is required.")

    model_name = body.get("model", "cpu-model")
    stream: bool = bool(body.get("stream", False))
    max_tokens: int = min(
        int(body.get("max_tokens") or 512),
        _MAX_TOKENS_HARD_LIMIT,
    )
    temperature: float = float(body.get("temperature", 0.7))
    top_p: float = float(body.get("top_p", 0.95))
    stop = body.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]

    req = GenerationRequest(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
        stop=stop,
    )

    CPU_QUEUE_DEPTH.inc()
    t0 = time.perf_counter()

    try:
        eng = get_engine()
    except RuntimeError as exc:
        CPU_QUEUE_DEPTH.dec()
        raise HTTPException(status_code=503, detail=str(exc))

    if stream:
        return _streaming_response(eng, req, model_name, t0)

    return await _unary_response(eng, req, model_name, t0)


async def _unary_response(eng, req: GenerationRequest, model_name: str, t0: float) -> JSONResponse:
    try:
        result = await eng.generate(req)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        CPU_QUEUE_DEPTH.dec()
        CPU_ERRORS_TOTAL.labels(error_type="inference").inc()
        log.error("cpu_inference_error", error=str(exc), latency_ms=round(elapsed * 1000, 1))
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    elapsed = time.perf_counter() - t0
    CPU_QUEUE_DEPTH.dec()
    CPU_INFERENCE_TOTAL.labels(model=model_name, status="200").inc()
    CPU_INFERENCE_LATENCY.labels(model=model_name).observe(elapsed)

    log.info(
        "cpu_inference_done",
        model=model_name,
        latency_ms=round(elapsed * 1000, 1),
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )

    return JSONResponse(content={
        "id": f"chatcmpl-cpu-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(t0),
        "model": result.model or model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    })


def _streaming_response(eng, req: GenerationRequest, model_name: str, t0: float) -> StreamingResponse:
    async def _safe_stream():
        status_code = 200
        try:
            async for chunk in eng.stream(req):
                yield chunk
        except Exception as exc:
            status_code = 500
            CPU_ERRORS_TOTAL.labels(error_type="stream").inc()
            log.error("cpu_stream_error", model=model_name, error=str(exc))
            yield b"event: error\ndata: inference failure\n\n"
        finally:
            elapsed = time.perf_counter() - t0
            CPU_QUEUE_DEPTH.dec()
            CPU_INFERENCE_TOTAL.labels(model=model_name, status=str(status_code)).inc()
            CPU_INFERENCE_LATENCY.labels(model=model_name).observe(elapsed)
            log.info(
                "cpu_stream_done",
                model=model_name,
                latency_ms=round(elapsed * 1000, 1),
                status_code=status_code,
            )

    return StreamingResponse(
        _safe_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
