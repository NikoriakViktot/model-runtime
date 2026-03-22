"""
node_agent/routes/local.py

Endpoints called by the Scheduler (and optionally operators) to manage
the local MRM.

All routes are thin pass-throughs — no logic, just translation.
"""

from __future__ import annotations

import time

import httpx
import structlog
from fastapi import APIRouter, HTTPException, Request

from node_agent.heartbeat import send_heartbeat
from node_agent.models import (
    LocalEnsureRequest,
    LocalEnsureResponse,
    LocalStatusResponse,
    LocalStopRequest,
)
from node_agent.mrm_client import local_mrm
from node_agent.observability import (
    LOCAL_ENSURE_LATENCY,
    LOCAL_ENSURES_TOTAL,
    get_tracer,
)

log = structlog.get_logger(__name__)
tracer = get_tracer("node_agent.local")

router = APIRouter()


@router.post(
    "/local/ensure",
    response_model=LocalEnsureResponse,
    summary="Ensure a model is running on this node",
)
async def local_ensure(body: LocalEnsureRequest) -> LocalEnsureResponse:
    """
    Call local MRM /models/ensure.

    Blocks until the model is READY (may take several minutes for cold-start).
    Returns the same response shape as MRM, forwarded verbatim.
    """
    log.info("local_ensure_start", model=body.model)
    t0 = time.perf_counter()

    with tracer.start_as_current_span("node_agent.local_ensure") as span:
        try:
            span.set_attribute("model", body.model)
        except Exception:
            pass

        try:
            result = await local_mrm.ensure(body.model)
        except httpx.HTTPStatusError as exc:
            elapsed = time.perf_counter() - t0
            LOCAL_ENSURES_TOTAL.labels(model=body.model, status="error").inc()
            LOCAL_ENSURE_LATENCY.labels(model=body.model).observe(elapsed)
            log.warning(
                "local_ensure_mrm_error",
                model=body.model,
                status=exc.response.status_code,
                latency_ms=round(elapsed * 1000, 1),
            )
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=f"MRM error: {exc.response.text}",
            )
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            LOCAL_ENSURES_TOTAL.labels(model=body.model, status="error").inc()
            LOCAL_ENSURE_LATENCY.labels(model=body.model).observe(elapsed)
            log.error(
                "local_ensure_failed",
                model=body.model,
                error=str(exc),
                latency_ms=round(elapsed * 1000, 1),
            )
            raise HTTPException(status_code=503, detail=f"MRM unavailable: {exc}")

    elapsed = time.perf_counter() - t0
    LOCAL_ENSURES_TOTAL.labels(model=body.model, status="success").inc()
    LOCAL_ENSURE_LATENCY.labels(model=body.model).observe(elapsed)
    log.info(
        "local_ensure_done",
        model=body.model,
        latency_ms=round(elapsed * 1000, 1),
        api_base=result.api_base,
    )
    return result


@router.post("/local/stop", status_code=204, summary="Stop a model on this node")
async def local_stop(body: LocalStopRequest) -> None:
    """
    Call local MRM /models/stop.

    Errors are logged but do not propagate as 5xx — the scheduler considers
    a best-effort stop sufficient.
    """
    log.info("local_stop", model=body.model)
    try:
        await local_mrm.stop(body.model)
    except Exception as exc:
        log.warning("local_stop_failed", model=body.model, error=str(exc))


@router.get(
    "/local/status",
    response_model=list[dict],
    summary="Status of all models on this node",
)
async def local_status_all() -> list[dict]:
    """Return MRM /models/status — all registered models on this node."""
    return await local_mrm.status_all()


@router.get(
    "/local/status/{model_id}",
    response_model=LocalStatusResponse,
    summary="Status of one model on this node",
)
async def local_status_model(model_id: str) -> LocalStatusResponse:
    """Return MRM /models/status/{model_id}."""
    return await local_mrm.status(model_id)


@router.post(
    "/heartbeat",
    status_code=204,
    summary="Trigger an immediate heartbeat to the Scheduler",
)
async def trigger_heartbeat(request: Request) -> None:
    """
    Force the agent to send a heartbeat to the Scheduler right now.

    Useful for debugging or forcing a node to re-register after the
    Scheduler restarted.  The background loop continues on its normal interval.
    """
    node_id: str = request.app.state.node_id
    try:
        await send_heartbeat(node_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Heartbeat failed: {exc}")
