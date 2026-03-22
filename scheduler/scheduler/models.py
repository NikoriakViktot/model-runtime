"""
scheduler/models.py

Core domain types for the distributed inference scheduler.

Design rules:
- All types are Pydantic models so they serialise cleanly to/from Redis JSON.
- No business logic here — this module has zero imports from the rest of the package.
- ``RuntimeInstance`` carries a routable ``api_base`` including ``/v1``,
  matching the existing gateway contract.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NodeState(str, Enum):
    HEALTHY = "healthy"
    STALE = "stale"   # missed one heartbeat — still usable
    DEAD = "dead"     # TTL expired in Redis — removed from pool


# ---------------------------------------------------------------------------
# GPU info (reported by Node Agent on every heartbeat)
# ---------------------------------------------------------------------------


class GpuInfo(BaseModel):
    gpu_index: str
    memory_total_mb: int = 0
    memory_free_mb: int = 0
    memory_used_mb: int = 0


# ---------------------------------------------------------------------------
# Node — one physical/virtual server
# ---------------------------------------------------------------------------


class Node(BaseModel):
    """
    Represents one server in the cluster.

    Populated entirely from Node Agent heartbeats — the scheduler never
    probes nodes itself.  ``last_heartbeat`` is a Unix timestamp set by
    the scheduler when it receives the heartbeat, not by the agent.
    """

    node_id: str
    agent_url: str          # e.g. "http://10.0.0.5:8020"
    hostname: str = ""
    gpus: list[GpuInfo] = Field(default_factory=list)
    last_heartbeat: float = Field(default_factory=time.time)
    state: NodeState = NodeState.HEALTHY

    @property
    def total_free_mb(self) -> int:
        """Sum of free GPU memory across all GPUs on this node."""
        return sum(g.memory_free_mb for g in self.gpus)

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)


# ---------------------------------------------------------------------------
# RuntimeInstance — one running model on one node
# ---------------------------------------------------------------------------


class RuntimeInstance(BaseModel):
    """
    A model that has been placed and confirmed READY on a specific node.

    ``api_base`` is the full routable URL including ``/v1``, e.g.
    ``http://10.0.0.5:8000/v1``.  The gateway proxies directly to this URL.

    ``model_alias`` is what vLLM was started with (``--served-model-name``).
    The gateway must swap the ``model`` field in the request body to this value.
    """

    instance_id: str
    model_id: str       # HuggingFace repo ID or MRM model_id
    model_alias: str    # served model name in vLLM
    node_id: str
    agent_url: str      # Node Agent URL (for management ops)
    api_base: str       # inference endpoint including /v1
    gpu: str            # GPU index on the node
    state: str = "READY"
    placed_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Placement — scheduler's record of where a model is running
# ---------------------------------------------------------------------------


class Placement(BaseModel):
    """
    Maps a model_id to one or more RuntimeInstances.

    Currently one placement = one instance.  The schema supports multiple
    instances per model for future multi-replica support without a breaking
    change.
    """

    model_id: str
    instances: list[RuntimeInstance] = Field(default_factory=list)
    strategy_used: str = ""
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Request / Response bodies
# ---------------------------------------------------------------------------


class HeartbeatPayload(BaseModel):
    """Sent by Node Agent → Scheduler on every heartbeat interval."""

    node_id: str
    agent_url: str
    hostname: str = ""
    gpus: list[GpuInfo] = Field(default_factory=list)


class EnsureRequest(BaseModel):
    model: str


class EnsureResponse(BaseModel):
    """
    Returned by POST /schedule/ensure.

    Compatible with the gateway's existing EnsureResult contract:
    ``api_base`` is always populated (primary instance) for backward compat.
    ``instances`` carries the full list for multi-instance routing.
    """

    model_id: str
    model_alias: str
    api_base: str                           # primary instance — backward compat
    instances: list[RuntimeInstance] = Field(default_factory=list)


class StopRequest(BaseModel):
    model: str


class NodeSummary(BaseModel):
    """Lightweight node view for GET /nodes."""

    node_id: str
    agent_url: str
    hostname: str
    gpu_count: int
    total_free_mb: int
    state: NodeState
    last_heartbeat: float
    models: list[str] = Field(default_factory=list)
