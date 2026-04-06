"""
gateway/routes/chat.py

POST /v1/chat/completions — OpenAI-compatible chat endpoint.

Flow
----
1. Parse request body; extract ``model`` and ``stream`` fields.
2. Call MRM ``ensure(model)`` — blocks until the container is READY.
   MRM handles cold-start, GPU allocation, and OOM fallback internally.
3. Use the ``api_base`` and ``model_alias`` from the ensure response.
   - ``api_base`` already contains ``/v1``, e.g. ``http://container:8000/v1``
   - ``model_alias`` is what vLLM was started with (--served-model-name)
4. Replace the ``model`` field in the request body with ``model_alias``
   so vLLM recognises the model name.
5. Forward to ``{api_base}/chat/completions``.
6. Return the response (or stream it for SSE requests).
7. Fire-and-forget MLflow logging after the response is sent.

Auto-provision
--------------
If ``settings.auto_provision`` is True and MRM returns 404 (model not
registered), the gateway calls ``/factory/provision`` and retries ensure.
This lets clients pass a raw HuggingFace repo ID without pre-registering.
"""

from __future__ import annotations

import random
import time
from typing import Any

import structlog
import structlog.contextvars as _ctxvars
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gateway.config import settings
from gateway.observability import (
    ERRORS_TOTAL,
    REQUEST_LATENCY,
    REQUEST_TOTAL,
    ROUTING_DECISIONS_TOTAL,
    get_tracer,
)

from gateway.services import mlflow_logger
from gateway.services.mrm_client import (
    EnsureResult,
    MRMError,
    MRMUnavailableError,
    ModelLockedError,
    ModelNotFoundError,
    mrm,
)
from gateway.services.proxy import UpstreamError, proxy
from gateway.services.retry import call_with_retry
from gateway.services.router import model_router
from gateway.services.scheduler_client import (
    SchedulerError,
    SchedulerUnavailableError,
    scheduler_client,
)

log = structlog.get_logger(__name__)
tracer = get_tracer("gateway.chat")

router = APIRouter()


# ---------------------------------------------------------------------------
# CPU model map — parsed once from settings at module load time
# ---------------------------------------------------------------------------


def _parse_cpu_model_map(raw: str) -> dict[str, str]:
    """
    Parse ``GATEWAY_CPU_MODEL_MAP`` into a dict[gpu_model_id → cpu_model_id].

    Format: "model-a:model-a-gguf,model-b:model-b-gguf"
    Empty string → empty map (CPU routing disabled).
    """
    result: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            gpu_id, cpu_id = pair.split(":", 1)
            result[gpu_id.strip()] = cpu_id.strip()
    return result


_CPU_MODEL_MAP: dict[str, str] = _parse_cpu_model_map(settings.cpu_model_map)


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    """
    OpenAI-compatible chat completions endpoint.

    Accepts the standard OpenAI request body and returns an OpenAI-compatible
    response.  Clients do not need to know about Docker, vLLM, or GPUs.

    Supports both unary and streaming (``"stream": true``) responses.
    """
    # ------------------------------------------------------------------
    # 1. Parse request body
    # ------------------------------------------------------------------
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Request body must be valid JSON.")

    model: str | None = body.get("model")
    if not model:
        raise HTTPException(status_code=422, detail="'model' field is required.")

    streaming: bool = bool(body.get("stream", False))
    max_tokens: int = int(body.get("max_tokens") or 1024)
    runtime_preference: str = body.get("runtime_preference", "auto")

    # Compute runtime decision: which backend to target
    runtime_decision = _decide_runtime(
        model=model,
        streaming=streaming,
        max_tokens=max_tokens,
        runtime_preference=runtime_preference,
    )
    # If routing to CPU, swap model ID to its registered CPU variant
    effective_model = _cpu_model_id(model) if runtime_decision == "cpu" else model

    # Bind model to structlog context so every log from this request includes it
    _ctxvars.bind_contextvars(model=effective_model, runtime=runtime_decision)

    # ------------------------------------------------------------------
    # 2. Ensure model runtime
    # ------------------------------------------------------------------
    with tracer.start_as_current_span("gateway.ensure") as span:
        try:
            span.set_attribute("model", effective_model)
            span.set_attribute("runtime_decision", runtime_decision)
        except Exception:
            pass
        ensure_result = await _ensure(effective_model)

    # ------------------------------------------------------------------
    # 3. Route to an instance
    # ------------------------------------------------------------------
    with tracer.start_as_current_span("gateway.route") as span:
        instance = model_router.choose_instance(ensure_result.instances, model_id=model)
        api_base: str = instance.api_base
        model_alias: str = ensure_result.model_alias
        try:
            span.set_attribute("instance", api_base)
            span.set_attribute("load", instance.load)
            span.set_attribute("strategy", model_router.strategy_name)
        except Exception:
            pass

    ROUTING_DECISIONS_TOTAL.labels(
        instance=api_base,
        strategy=model_router.strategy_name,
    ).inc()

    _ctxvars.bind_contextvars(instance=api_base)

    target_url = f"{api_base}/chat/completions"
    # Strip gateway-only fields before forwarding; upstream doesn't understand them
    proxy_body = {k: v for k, v in body.items() if k != "runtime_preference"}
    proxy_body["model"] = model_alias
    client_headers = dict(request.headers)
    # Ensure the gateway-generated request_id is forwarded to upstream
    _rid = getattr(request.state, "request_id", "")
    if _rid:
        client_headers["x-request-id"] = _rid

    # ------------------------------------------------------------------
    # 4. Forward to vLLM
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    runtime_type: str = ensure_result.runtime_type

    if streaming:
        return await _handle_stream(
            target_url=target_url,
            payload=proxy_body,
            headers=client_headers,
            model=effective_model,
            api_base=api_base,
            t0=t0,
            runtime_type=runtime_type,
        )

    return await _handle_unary(
        target_url=target_url,
        payload=proxy_body,
        headers=client_headers,
        model=effective_model,
        api_base=api_base,
        t0=t0,
        runtime_type=runtime_type,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decide_runtime(
    *,
    model: str,
    streaming: bool,
    max_tokens: int,
    runtime_preference: str,
) -> str:
    """
    Return "cpu" or "gpu" for this request.

    Priority:
      1. Explicit override via runtime_preference="cpu"|"gpu"
      2. Auto-routing rules:
         - streaming → always GPU (CPU latency too high for real-time SSE)
         - max_tokens <= threshold AND CPU variant registered → CPU
         - otherwise → GPU
    """
    if runtime_preference == "gpu":
        return "gpu"

    if runtime_preference == "cpu":
        # Only honour if a CPU variant is registered; otherwise fall back
        return "cpu" if _cpu_model_id(model) else "gpu"

    # Auto-routing
    threshold = settings.cpu_routing_max_tokens_threshold
    if threshold <= 0:
        return "gpu"

    if streaming and settings.cpu_routing_block_streaming:
        return "gpu"

    if max_tokens <= threshold and _cpu_model_id(model):
        return "cpu"

    return "gpu"


def _cpu_model_id(gpu_model: str) -> str:
    """Return the CPU variant model ID, or empty string if none registered."""
    return _CPU_MODEL_MAP.get(gpu_model, "")


async def _ensure(model: str) -> EnsureResult:
    """
    Dispatch to Scheduler (distributed mode) or local MRM (single-node mode).
    Both paths return an ``EnsureResult``.
    """
    if settings.use_scheduler:
        return await _ensure_via_scheduler(model)
    return await _ensure_with_autoprovision(model)


async def _ensure_via_scheduler(model: str) -> EnsureResult:
    try:
        return await scheduler_client.ensure(model)
    except SchedulerUnavailableError as exc:
        ERRORS_TOTAL.labels(model=model, error_type="scheduler_unavailable", runtime_type="unknown").inc()
        raise HTTPException(status_code=503, detail=f"Scheduler is unavailable: {exc}")
    except SchedulerError as exc:
        ERRORS_TOTAL.labels(model=model, error_type="scheduler_error", runtime_type="unknown").inc()
        raise HTTPException(
            status_code=exc.status_code or 502,
            detail=f"Scheduler error for '{model}': {exc}",
        )


async def _ensure_with_autoprovision(model: str) -> EnsureResult:
    try:
        return await mrm.ensure(model)

    except ModelNotFoundError:
        if not settings.auto_provision:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Model '{model}' is not registered in MRM. "
                    "Register it via the Streamlit UI or enable GATEWAY_AUTO_PROVISION."
                ),
            )
        log.info("auto_provisioning", model=model, preset=settings.default_preset)
        try:
            await mrm.provision(
                repo_id=model,
                preset=settings.default_preset,
                gpu=settings.default_gpu,
            )
        except MRMError as exc:
            ERRORS_TOTAL.labels(model=model, error_type="provision_failed", runtime_type="unknown").inc()
            raise HTTPException(
                status_code=503, detail=f"Auto-provision failed for '{model}': {exc}"
            )
        try:
            return await mrm.ensure(model)
        except MRMError as exc:
            ERRORS_TOTAL.labels(model=model, error_type="ensure_after_provision", runtime_type="unknown").inc()
            raise HTTPException(
                status_code=503,
                detail=f"Runtime unavailable for '{model}' after provision: {exc}",
            )

    except ModelLockedError:
        ERRORS_TOTAL.labels(model=model, error_type="model_locked", runtime_type="unknown").inc()
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model}' is busy with another operation. Retry shortly.",
            headers={"Retry-After": "5"},
        )

    except MRMUnavailableError as exc:
        ERRORS_TOTAL.labels(model=model, error_type="mrm_unavailable", runtime_type="unknown").inc()
        raise HTTPException(
            status_code=503, detail=f"Model runtime service is unavailable: {exc}"
        )

    except MRMError as exc:
        ERRORS_TOTAL.labels(model=model, error_type="mrm_error", runtime_type="unknown").inc()
        raise HTTPException(
            status_code=502, detail=f"MRM error for '{model}': {exc}"
        )


async def _handle_unary(
    *,
    target_url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    model: str,
    api_base: str,
    t0: float,
    runtime_type: str = "gpu",
) -> JSONResponse:
    """Forward a non-streaming request and return a JSON response."""
    request_id = headers.get("x-request-id", "")

    with tracer.start_as_current_span("gateway.proxy") as span:
        try:
            span.set_attribute("target_url", target_url)
            span.set_attribute("streaming", False)
            span.set_attribute("runtime_type", runtime_type)
        except Exception:
            pass

        try:
            async with model_router.track_inflight(api_base):
                result = await call_with_retry(
                    lambda: proxy.post(target_url, payload, client_headers=headers),
                    max_retries=settings.retry_max,
                    jitter_min_ms=settings.retry_jitter_min_ms,
                    jitter_max_ms=settings.retry_jitter_max_ms,
                    request_id=request_id,
                    on_retry=lambda attempt, exc: model_router.record(
                        api_base,
                        latency_ms=(time.perf_counter() - t0) * 1000,
                        error=True,
                    ),
                )
        except UpstreamError as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            model_router.record(api_base, latency_ms=latency_ms, error=True)
            REQUEST_TOTAL.labels(model=model, status=str(exc.status_code), runtime_type=runtime_type).inc()
            REQUEST_LATENCY.labels(model=model, runtime_type=runtime_type).observe(latency_ms / 1000)
            ERRORS_TOTAL.labels(model=model, error_type="upstream", runtime_type=runtime_type).inc()
            log.warning(
                "upstream_error",
                model=model,
                instance=api_base,
                latency_ms=round(latency_ms, 1),
                status_code=exc.status_code,
                runtime_type=runtime_type,
                error=str(exc),
            )
            raise HTTPException(
                status_code=exc.status_code,
                detail={"error": f"Upstream error: {exc}", "upstream_body": exc.body},
            )

    latency_ms = (time.perf_counter() - t0) * 1000
    response_bytes = len(str(result).encode())

    model_router.record(api_base, latency_ms=latency_ms, error=False)
    REQUEST_TOTAL.labels(model=model, status="200", runtime_type=runtime_type).inc()
    REQUEST_LATENCY.labels(model=model, runtime_type=runtime_type).observe(latency_ms / 1000)

    await mlflow_logger.log_inference(
        model=model,
        latency_ms=latency_ms,
        response_bytes=response_bytes,
        status_code=200,
        streaming=False,
    )

    log.info(
        "request_completed",
        model=model,
        instance=api_base,
        latency_ms=round(latency_ms, 1),
        response_bytes=response_bytes,
        streaming=False,
        runtime_type=runtime_type,
    )

    return JSONResponse(content=result)


async def _handle_stream(
    *,
    target_url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    model: str,
    api_base: str,
    t0: float,
    runtime_type: str = "gpu",
) -> StreamingResponse:
    """
    Forward a streaming request.  Returns a StreamingResponse immediately;
    chunks are yielded as they arrive from vLLM.

    X-Accel-Buffering: no disables Nginx proxy buffering for real-time SSE.
    """
    try:
        stream_gen = proxy.stream(target_url, payload, client_headers=headers)

        async def _safe_stream():
            total_bytes = 0
            status_code = 200
            try:
                async with model_router.track_inflight(api_base):
                    async for chunk in stream_gen:
                        total_bytes += len(chunk)
                        yield chunk
            except UpstreamError as exc:
                status_code = exc.status_code
                ERRORS_TOTAL.labels(model=model, error_type="stream_error", runtime_type=runtime_type).inc()
                log.warning(
                    "stream_upstream_error",
                    model=model,
                    instance=api_base,
                    status_code=exc.status_code,
                    runtime_type=runtime_type,
                    error=str(exc),
                )
                # Terminate the SSE stream with an error event so the client
                # receives a clean signal instead of an abrupt connection close.
                yield b"event: error\ndata: upstream failure\n\n"
            except Exception as exc:
                status_code = 500
                ERRORS_TOTAL.labels(model=model, error_type="stream_error", runtime_type=runtime_type).inc()
                log.error("stream_internal_error", model=model, instance=api_base, error=str(exc))
                yield b"event: error\ndata: internal error\n\n"
            finally:
                latency_ms = (time.perf_counter() - t0) * 1000
                error = status_code >= 400
                model_router.record(api_base, latency_ms=latency_ms, error=error)
                REQUEST_TOTAL.labels(model=model, status=str(status_code), runtime_type=runtime_type).inc()
                REQUEST_LATENCY.labels(model=model, runtime_type=runtime_type).observe(latency_ms / 1000)
                if error:
                    ERRORS_TOTAL.labels(model=model, error_type="stream_error", runtime_type=runtime_type).inc()
                await mlflow_logger.log_inference(
                    model=model,
                    latency_ms=latency_ms,
                    response_bytes=total_bytes,
                    status_code=status_code,
                    streaming=True,
                )
                log.info(
                    "request_completed",
                    model=model,
                    instance=api_base,
                    latency_ms=round(latency_ms, 1),
                    response_bytes=total_bytes,
                    streaming=True,
                    status_code=status_code,
                    runtime_type=runtime_type,
                )

        return StreamingResponse(
            _safe_stream(),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            },
        )

    except UpstreamError as exc:
        ERRORS_TOTAL.labels(model=model, error_type="upstream_stream", runtime_type=runtime_type).inc()
        raise HTTPException(
            status_code=exc.status_code,
            detail={"error": f"Upstream stream error: {exc}", "upstream_body": exc.body},
        )
