"""
scheduler/routes/nodes.py

Node management and heartbeat endpoints.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import Response

from scheduler.models import HeartbeatPayload, NodeSummary

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/heartbeat", status_code=204, tags=["Nodes"],
             summary="Receive a heartbeat from a Node Agent",
             response_model=None, response_class=Response)
async def heartbeat(payload: HeartbeatPayload, request: Request) -> None:
    """
    Called by every Node Agent on a fixed interval (default 15s).

    Updates the node's entry in Redis and resets its TTL.  A node that
    stops sending heartbeats will be evicted automatically when its TTL
    expires.
    """
    scheduler = request.app.state.scheduler
    await scheduler.handle_heartbeat(payload)


@router.get("/nodes", response_model=list[NodeSummary], tags=["Nodes"],
            summary="List all healthy nodes")
async def list_nodes(request: Request) -> list[NodeSummary]:
    """
    Return all nodes currently sending heartbeats.

    Includes per-node GPU memory and the list of models currently placed
    on each node.
    """
    scheduler = request.app.state.scheduler
    nodes = await scheduler._registry.list_alive()

    summaries: list[NodeSummary] = []
    for node in nodes:
        models = await scheduler._placements.models_on_node(node.node_id)
        summaries.append(NodeSummary(
            node_id=node.node_id,
            agent_url=node.agent_url,
            hostname=node.hostname,
            gpu_count=node.gpu_count,
            total_free_mb=node.total_free_mb,
            state=node.state,
            last_heartbeat=node.last_heartbeat,
            models=models,
        ))

    return summaries
