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
from gateway.services.router import model_router
from gateway.services.scheduler_client import (
    SchedulerError,
    SchedulerUnavailableError,
    scheduler_client,
)

log = structlog.get_logger(__name__)
tracer = get_tracer("gateway.chat")

router = APIRouter()


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

    # Bind model to structlog context so every log from this request includes it
    _ctxvars.bind_contextvars(model=model)

    # ------------------------------------------------------------------
    # 2. Ensure model runtime
    # ------------------------------------------------------------------
    with tracer.start_as_current_span("gateway.ensure") as span:
        try:
            span.set_attribute("model", model)
        except Exception:
            pass
        ensure_result = await _ensure(model)

    # ------------------------------------------------------------------
    # 3. Route to an instance
    # ------------------------------------------------------------------
    with tracer.start_as_current_span("gateway.route") as span:
        instance = model_router.choose_instance(ensure_result.instances)
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
    proxy_body = {**body, "model": model_alias}
    client_headers = dict(request.headers)

    # ------------------------------------------------------------------
    # 4. Forward to vLLM
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    if streaming:
        return await _handle_stream(
            target_url=target_url,
            payload=proxy_body,
            headers=client_headers,
            model=model,
            api_base=api_base,
            t0=t0,
        )

    return await _handle_unary(
        target_url=target_url,
        payload=proxy_body,
        headers=client_headers,
        model=model,
        api_base=api_base,
        t0=t0,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
        ERRORS_TOTAL.labels(model=model, error_type="scheduler_unavailable").inc()
        raise HTTPException(status_code=503, detail=f"Scheduler is unavailable: {exc}")
    except SchedulerError as exc:
        ERRORS_TOTAL.labels(model=model, error_type="scheduler_error").inc()
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
            ERRORS_TOTAL.labels(model=model, error_type="provision_failed").inc()
            raise HTTPException(
                status_code=503, detail=f"Auto-provision failed for '{model}': {exc}"
            )
        try:
            return await mrm.ensure(model)
        except MRMError as exc:
            ERRORS_TOTAL.labels(model=model, error_type="ensure_after_provision").inc()
            raise HTTPException(
                status_code=503,
                detail=f"Runtime unavailable for '{model}' after provision: {exc}",
            )

    except ModelLockedError:
        ERRORS_TOTAL.labels(model=model, error_type="model_locked").inc()
        raise HTTPException(
            status_code=503,
            detail=f"Model '{model}' is busy with another operation. Retry shortly.",
            headers={"Retry-After": "5"},
        )

    except MRMUnavailableError as exc:
        ERRORS_TOTAL.labels(model=model, error_type="mrm_unavailable").inc()
        raise HTTPException(
            status_code=503, detail=f"Model runtime service is unavailable: {exc}"
        )

    except MRMError as exc:
        ERRORS_TOTAL.labels(model=model, error_type="mrm_error").inc()
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
) -> JSONResponse:
    """Forward a non-streaming request and return a JSON response."""
    with tracer.start_as_current_span("gateway.proxy") as span:
        try:
            span.set_attribute("target_url", target_url)
            span.set_attribute("streaming", False)
        except Exception:
            pass

        try:
            result = await proxy.post(target_url, payload, client_headers=headers)
        except UpstreamError as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            model_router.record(api_base, latency_ms=latency_ms, error=True)
            REQUEST_TOTAL.labels(model=model, status="502").inc()
            REQUEST_LATENCY.labels(model=model).observe(latency_ms / 1000)
            ERRORS_TOTAL.labels(model=model, error_type="upstream").inc()
            log.warning(
                "upstream_error",
                model=model,
                instance=api_base,
                latency_ms=round(latency_ms, 1),
                error=str(exc),
            )
            raise HTTPException(
                status_code=502,
                detail={"error": f"Upstream error: {exc}", "upstream_body": exc.body},
            )

    latency_ms = (time.perf_counter() - t0) * 1000
    response_bytes = len(str(result).encode())

    model_router.record(api_base, latency_ms=latency_ms, error=False)
    REQUEST_TOTAL.labels(model=model, status="200").inc()
    REQUEST_LATENCY.labels(model=model).observe(latency_ms / 1000)

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
) -> StreamingResponse:
    """
    Forward a streaming request.  Returns a StreamingResponse immediately;
    chunks are yielded as they arrive from vLLM.

    X-Accel-Buffering: no disables Nginx proxy buffering for real-time SSE.
    """
    try:
        stream_gen = proxy.stream(target_url, payload, client_headers=headers)

        async def _counting_stream():
            total_bytes = 0
            error = False
            try:
                async for chunk in stream_gen:
                    total_bytes += len(chunk)
                    yield chunk
            except Exception:
                error = True
                raise
            finally:
                latency_ms = (time.perf_counter() - t0) * 1000
                status = "error" if error else "200"
                model_router.record(api_base, latency_ms=latency_ms, error=error)
                REQUEST_TOTAL.labels(model=model, status=status).inc()
                REQUEST_LATENCY.labels(model=model).observe(latency_ms / 1000)
                if error:
                    ERRORS_TOTAL.labels(model=model, error_type="stream_error").inc()
                await mlflow_logger.log_inference(
                    model=model,
                    latency_ms=latency_ms,
                    response_bytes=total_bytes,
                    status_code=200,
                    streaming=True,
                )
                log.info(
                    "request_completed",
                    model=model,
                    instance=api_base,
                    latency_ms=round(latency_ms, 1),
                    response_bytes=total_bytes,
                    streaming=True,
                    error=error,
                )

        return StreamingResponse(
            _counting_stream(),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            },
        )

    except UpstreamError as exc:
        ERRORS_TOTAL.labels(model=model, error_type="upstream_stream").inc()
        raise HTTPException(
            status_code=502,
            detail={"error": f"Upstream stream error: {exc}", "upstream_body": exc.body},
        )
