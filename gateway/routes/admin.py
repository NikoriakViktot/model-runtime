"""
gateway/routes/admin.py

Operational endpoints for observing and managing the gateway's internal state.

Endpoints
---------
GET  /v1/router/metrics          Per-instance request counters and latency.
DELETE /v1/router/metrics        Reset all per-instance counters.
DELETE /v1/router/metrics/{url}  Reset counters for one instance.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response

from gateway.services.router import model_router

router = APIRouter(tags=["Admin"])


@router.get("/v1/router/metrics", summary="Per-instance routing metrics")
async def get_router_metrics():
    """
    Return per-instance request counters, error counts, latency stats, and
    current in-flight count.

    Keys are ``api_base`` URLs (e.g. ``http://vllm_qwen:8000/v1``).
    Values contain ``requests``, ``errors``, ``avg_latency_ms``,
    ``total_latency_ms``, and ``inflight``.
    """
    return {
        "strategy": model_router.strategy_name,
        "instances": model_router.get_metrics(),
        "slo": model_router.slo_status(),
    }


@router.get("/v1/slo", summary="Fleet-wide SLO snapshot")
async def get_global_slo():
    """
    Return the global SLO snapshot: system-wide p50/p95/p99 latency and
    error rate aggregated across all instances over the last 1 000 requests.
    """
    return model_router.global_slo_snapshot()


@router.delete(
    "/v1/router/metrics",
    summary="Reset all routing metrics",
    status_code=204,
    response_model=None,
    response_class=Response,
)
async def reset_all_metrics():
    """Reset counters for every instance tracked by the router."""
    model_router.reset_metrics()


@router.delete(
    "/v1/router/metrics/{api_base:path}",
    summary="Reset routing metrics for one instance",
    status_code=204,
    response_model=None,
    response_class=Response,
)
async def reset_instance_metrics(api_base: str):
    """
    Reset counters for a single instance identified by its ``api_base`` URL.

    The ``api_base`` path parameter must be URL-encoded, e.g.
    ``http%3A%2F%2Fvllm_qwen%3A8000%2Fv1``.
    """
    model_router.reset_metrics(api_base)
