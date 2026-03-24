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

import asyncio
import time
import uuid
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from cpu_runtime import state as _state
from cpu_runtime.config import settings as _settings
from cpu_runtime.inference import GenerationRequest, get_engine
from cpu_runtime.load_shedder import shedder as _shedder
from cpu_runtime.observability import (
    CPU_ERRORS_TOTAL,
    CPU_INFERENCE_LATENCY,
    CPU_INFERENCE_TOTAL,
    CPU_QUEUE_DEPTH,
)

log = structlog.get_logger(__name__)
router = APIRouter()

# Active request counter — asyncio is single-threaded so plain int is safe
# (no await between check and increment).
_active_requests: int = 0


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    """OpenAI-compatible chat completions via llama.cpp / GGUF."""
    global _active_requests

    # Reject immediately if service is draining for shutdown
    if _state.shutting_down:
        raise HTTPException(
            status_code=503,
            detail="Service is shutting down. Please retry another instance.",
        )

    # Determine effective concurrency ceiling.
    # During latency spikes the ceiling is reduced proportionally so the
    # system sheds load before the queue fills completely.
    _ceiling = (
        _shedder.effective_max_requests(
            _settings.max_queue_depth,
            _settings.latency_threshold_ms,
        )
        if _settings.dynamic_concurrency_enabled
        else _settings.max_queue_depth
    )
    if _active_requests >= _ceiling:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Server busy: {_active_requests}/{_ceiling} "
                "requests in progress. Try again shortly."
            ),
            headers={"Retry-After": "5"},
        )

    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Request body must be valid JSON.")

    messages = body.get("messages")
    if not messages:
        raise HTTPException(status_code=422, detail="'messages' field is required.")

    # Prompt size guard — reject oversized inputs before touching the engine.
    total_chars = sum(len(m.get("content", "")) for m in messages if isinstance(m, dict))
    if total_chars > _settings.max_prompt_chars:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Prompt too large: {total_chars} chars "
                f"(limit {_settings.max_prompt_chars})."
            ),
        )

    # RAM guard: reject if memory is critically low
    if _settings.low_ram_mode_enabled:
        free_mb = _shedder.check_ram()
        if free_mb >= 0 and free_mb < _settings.min_free_ram_mb:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Insufficient memory: {free_mb} MiB free "
                    f"(minimum {_settings.min_free_ram_mb} MiB)."
                ),
            )

    model_name = body.get("model", "cpu-model")
    stream: bool = bool(body.get("stream", False))
    # Clamp max_tokens to the configured ceiling — never reject, just cap.
    max_tokens: int = min(
        int(body.get("max_tokens") or _settings.max_tokens_default),
        _settings.max_total_tokens,
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

    _active_requests += 1
    CPU_QUEUE_DEPTH.inc()
    t0 = time.perf_counter()

    try:
        eng = get_engine()
    except RuntimeError as exc:
        _active_requests -= 1
        CPU_QUEUE_DEPTH.dec()
        raise HTTPException(status_code=503, detail=str(exc))

    timeout = _settings.generation_timeout_sec or None  # 0.0 → disable

    if stream:
        return _streaming_response(eng, req, model_name, t0, timeout_sec=timeout)

    return await _unary_response(eng, req, model_name, t0, timeout_sec=timeout)


async def _unary_response(
    eng,
    req: GenerationRequest,
    model_name: str,
    t0: float,
    timeout_sec: float | None = None,
) -> JSONResponse:
    global _active_requests
    try:
        if timeout_sec:
            result = await asyncio.wait_for(eng.generate(req), timeout=timeout_sec)
        else:
            result = await eng.generate(req)
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - t0
        _active_requests -= 1
        CPU_QUEUE_DEPTH.dec()
        CPU_ERRORS_TOTAL.labels(error_type="timeout").inc()
        log.warning(
            "cpu_inference_timeout",
            model=model_name,
            timeout_sec=timeout_sec,
            latency_ms=round(elapsed * 1000, 1),
        )
        raise HTTPException(
            status_code=504,
            detail=f"Generation timed out after {timeout_sec:.0f}s.",
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        _active_requests -= 1
        CPU_QUEUE_DEPTH.dec()
        CPU_ERRORS_TOTAL.labels(error_type="inference").inc()
        log.error("cpu_inference_error", error=str(exc), latency_ms=round(elapsed * 1000, 1))
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    elapsed = time.perf_counter() - t0
    _active_requests -= 1
    CPU_QUEUE_DEPTH.dec()
    CPU_INFERENCE_TOTAL.labels(model=model_name, status="200").inc()
    CPU_INFERENCE_LATENCY.labels(model=model_name).observe(elapsed)
    _shedder.record_latency(elapsed * 1000)

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


def _streaming_response(
    eng,
    req: GenerationRequest,
    model_name: str,
    t0: float,
    timeout_sec: float | None = None,
) -> StreamingResponse:
    async def _safe_stream():
        global _active_requests
        status_code = 200
        try:
            async for chunk in eng.stream(req, timeout_sec=timeout_sec):
                yield chunk
        except Exception as exc:
            status_code = 500
            CPU_ERRORS_TOTAL.labels(error_type="stream").inc()
            log.error("cpu_stream_error", model=model_name, error=str(exc))
            yield b"event: error\ndata: inference failure\n\n"
        finally:
            elapsed = time.perf_counter() - t0
            _active_requests -= 1
            CPU_QUEUE_DEPTH.dec()
            CPU_INFERENCE_TOTAL.labels(model=model_name, status=str(status_code)).inc()
            CPU_INFERENCE_LATENCY.labels(model=model_name).observe(elapsed)
            _shedder.record_latency(elapsed * 1000)
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
