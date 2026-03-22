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

from gateway.services.router import model_router

router = APIRouter(tags=["Admin"])


@router.get("/v1/router/metrics", summary="Per-instance routing metrics")
async def get_router_metrics():
    """
    Return per-instance request counters, error counts, and latency stats.

    Keys are ``api_base`` URLs (e.g. ``http://vllm_qwen:8000/v1``).
    Values contain ``requests``, ``errors``, ``avg_latency_ms``, and
    ``total_latency_ms``.
    """
    return {
        "strategy": model_router.strategy_name,
        "instances": model_router.get_metrics(),
    }


@router.delete(
    "/v1/router/metrics",
    summary="Reset all routing metrics",
    status_code=204,
)
async def reset_all_metrics():
    """Reset counters for every instance tracked by the router."""
    model_router.reset_metrics()


@router.delete(
    "/v1/router/metrics/{api_base:path}",
    summary="Reset routing metrics for one instance",
    status_code=204,
)
async def reset_instance_metrics(api_base: str):
    """
    Reset counters for a single instance identified by its ``api_base`` URL.

    The ``api_base`` path parameter must be URL-encoded, e.g.
    ``http%3A%2F%2Fvllm_qwen%3A8000%2Fv1``.
    """
    model_router.reset_metrics(api_base)
