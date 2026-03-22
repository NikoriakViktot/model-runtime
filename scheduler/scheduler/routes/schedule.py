"""
scheduler/routes/schedule.py

Model placement endpoints.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from scheduler.models import EnsureRequest, EnsureResponse, Placement, StopRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/schedule/ensure", response_model=EnsureResponse, tags=["Schedule"],
             summary="Ensure a model is running somewhere in the cluster")
async def schedule_ensure(body: EnsureRequest, request: Request) -> EnsureResponse:
    """
    Ensure the model is running and return a routable ``api_base``.

    - If the model is already placed on a live node: return immediately.
    - If not placed (or placed on a dead node): select a node, call its
      Node Agent, persist the placement, return.

    This is the hot path — the gateway calls this on every request that
    needs a model not already in its local cache.
    """
    scheduler = request.app.state.scheduler
    try:
        return await scheduler.ensure(body.model)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error ensuring model=%s", body.model)
        raise HTTPException(status_code=502, detail=f"Scheduling error: {exc}")


@router.post("/schedule/stop", status_code=204, tags=["Schedule"],
             summary="Stop a model on all nodes and remove its placement")
async def schedule_stop(body: StopRequest, request: Request) -> None:
    """
    Stop the model on every node where it is currently placed and delete
    the placement record.

    Node Agent errors are logged but do not cause a 5xx response — the
    placement is always cleaned up so the scheduler stays consistent.
    """
    scheduler = request.app.state.scheduler
    await scheduler.stop(body.model)


@router.get("/placements", response_model=list[Placement], tags=["Schedule"],
            summary="List all active placements")
async def list_placements(request: Request) -> list[Placement]:
    """Return all model placements currently persisted in Redis."""
    scheduler = request.app.state.scheduler
    return await scheduler._placements.list_all()
