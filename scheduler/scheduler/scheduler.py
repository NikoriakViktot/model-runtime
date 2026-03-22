"""
scheduler/scheduler.py

Core scheduling logic.

Responsibilities
----------------
- Receive heartbeats and maintain node state via NodeRegistry.
- On ensure: check if already placed, select a node, call Node Agent,
  persist placement, return EnsureResponse.
- On stop: call Node Agent to stop the model, remove placement.
- Per-model asyncio locks prevent thundering-herd on cold-start.

Node Agent contract (what we POST to /local/ensure)
----------------------------------------------------
    Request:  {"model": "<model_id>"}
    Response: {"model_id": str, "model_alias": str, "api_base": str,
               "gpu": str, "state": str, "container": str}

The response mirrors what MRM returns from /models/ensure, forwarded verbatim
by the Node Agent.  ``api_base`` includes ``/v1``.
"""

from __future__ import annotations

import asyncio
import time
import uuid

import httpx
import structlog

from scheduler.config import settings
from scheduler.models import (
    EnsureResponse,
    HeartbeatPayload,
    Node,
    Placement,
    RuntimeInstance,
)
from scheduler.observability import (
    ENSURE_LATENCY,
    FAILOVERS_TOTAL,
    HEARTBEATS_TOTAL,
    NODES_ALIVE,
    PLACEMENTS_TOTAL,
    get_tracer,
)
from scheduler.placements import PlacementStore
from scheduler.registry import NodeRegistry
from scheduler.strategy import PlacementStrategy, get_strategy

log = structlog.get_logger(__name__)
tracer = get_tracer("scheduler.scheduler")


class Scheduler:
    """
    Stateless coordinator — all durable state lives in Redis via the
    registry and placement store.

    The only in-process state is per-model asyncio Locks, which prevent
    duplicate ensure calls for the same model from racing to place it on
    two different nodes simultaneously.
    """

    def __init__(self, registry: NodeRegistry, placements: PlacementStore) -> None:
        self._registry = registry
        self._placements = placements
        self._strategy: PlacementStrategy = get_strategy(settings.placement_strategy)
        self._model_locks: dict[str, asyncio.Lock] = {}

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def handle_heartbeat(self, payload: HeartbeatPayload) -> None:
        """
        Upsert the node in the registry with fresh GPU state and a new TTL.
        Updates NODES_ALIVE gauge after the upsert.
        """
        node = Node(
            node_id=payload.node_id,
            agent_url=payload.agent_url,
            hostname=payload.hostname,
            gpus=payload.gpus,
            last_heartbeat=time.time(),
        )
        await self._registry.upsert(node)
        HEARTBEATS_TOTAL.labels(node_id=payload.node_id).inc()

        # Refresh alive-node gauge (inexpensive — set membership query)
        try:
            alive = await self._registry.list_alive()
            NODES_ALIVE.set(len(alive))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Ensure
    # ------------------------------------------------------------------

    async def ensure(self, model_id: str) -> EnsureResponse:
        """
        Ensure the model is running somewhere in the cluster.

        Fast path (already placed): return existing placement without
        acquiring a lock.

        Slow path (new placement): acquire a per-model lock, double-check
        (another coroutine may have placed it while we waited), select a
        node, call the Node Agent, persist, return.

        Raises:
            RuntimeError: No healthy nodes are available.
            httpx.HTTPError: Node Agent returned an error.
        """
        t_start = time.perf_counter()

        with tracer.start_as_current_span("scheduler.ensure") as span:
            try:
                span.set_attribute("model_id", model_id)
            except Exception:
                pass

            # --- Fast path ---
            placement = await self._placements.get(model_id)
            if placement and placement.instances:
                primary_node_id = placement.instances[0].node_id
                if await self._registry.get(primary_node_id) is not None:
                    elapsed = time.perf_counter() - t_start
                    ENSURE_LATENCY.labels(model=model_id, path="cache_hit").observe(elapsed)
                    try:
                        span.set_attribute("path", "cache_hit")
                        span.set_attribute("node_id", primary_node_id)
                    except Exception:
                        pass
                    return _to_response(placement)
                else:
                    log.warning(
                        "placement_dead_node",
                        model=model_id,
                        node=primary_node_id,
                    )
                    FAILOVERS_TOTAL.labels(model=model_id).inc()
                    try:
                        span.set_attribute("failover", True)
                    except Exception:
                        pass

            # --- Slow path ---
            lock = self._get_model_lock(model_id)
            async with lock:
                # Double-check after acquiring the lock
                placement = await self._placements.get(model_id)
                if placement and placement.instances:
                    primary_node_id = placement.instances[0].node_id
                    if await self._registry.get(primary_node_id) is not None:
                        elapsed = time.perf_counter() - t_start
                        ENSURE_LATENCY.labels(model=model_id, path="cache_hit").observe(elapsed)
                        return _to_response(placement)

                nodes = await self._registry.list_alive()
                NODES_ALIVE.set(len(nodes))

                if not nodes:
                    raise RuntimeError("No healthy nodes available in the cluster")

                node = self._strategy.select_node(nodes, model_id)
                log.info(
                    "placing_model",
                    model=model_id,
                    node=node.node_id,
                    agent=node.agent_url,
                    strategy=type(self._strategy).__name__,
                )

                with tracer.start_as_current_span("scheduler.place") as place_span:
                    try:
                        place_span.set_attribute("model_id", model_id)
                        place_span.set_attribute("node_id", node.node_id)
                    except Exception:
                        pass
                    instance = await self._call_node_ensure(node, model_id)

                placement = Placement(
                    model_id=model_id,
                    instances=[instance],
                    strategy_used=type(self._strategy).__name__,
                )
                await self._placements.save(placement)

                elapsed = time.perf_counter() - t_start
                ENSURE_LATENCY.labels(model=model_id, path="new_placement").observe(elapsed)
                PLACEMENTS_TOTAL.labels(
                    model=model_id,
                    node=node.node_id,
                    strategy=type(self._strategy).__name__,
                ).inc()

                try:
                    span.set_attribute("path", "new_placement")
                    span.set_attribute("node_id", node.node_id)
                    span.set_attribute("api_base", instance.api_base)
                except Exception:
                    pass

                log.info(
                    "placement_saved",
                    model=model_id,
                    node=node.node_id,
                    api_base=instance.api_base,
                )
                return _to_response(placement)

    # ------------------------------------------------------------------
    # Stop
    # ------------------------------------------------------------------

    async def stop(self, model_id: str) -> None:
        """
        Stop the model on all nodes where it is placed and remove the placement.

        Node Agent errors are logged but do not propagate — the placement is
        always removed so the scheduler stays consistent even if a node is
        unreachable.
        """
        placement = await self._placements.get(model_id)
        if not placement:
            log.debug("stop_noop", model=model_id)
            return

        for inst in placement.instances:
            node = await self._registry.get(inst.node_id)
            if node:
                await self._call_node_stop(node, model_id)
            else:
                log.warning("stop_dead_node", model=model_id, node=inst.node_id)

        await self._placements.delete(model_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_model_lock(self, model_id: str) -> asyncio.Lock:
        if model_id not in self._model_locks:
            self._model_locks[model_id] = asyncio.Lock()
        return self._model_locks[model_id]

    async def _call_node_ensure(self, node: Node, model_id: str) -> RuntimeInstance:
        """POST /local/ensure to the Node Agent and return a RuntimeInstance."""
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=settings.node_ensure_timeout_sec,
                write=30.0,
                pool=10.0,
            )
        ) as client:
            resp = await client.post(
                f"{node.agent_url}/local/ensure",
                json={"model": model_id},
            )
            resp.raise_for_status()
            d = resp.json()

        return RuntimeInstance(
            instance_id=str(uuid.uuid4()),
            model_id=d.get("model_id", model_id),
            model_alias=d.get("model_alias", model_id),
            node_id=node.node_id,
            agent_url=node.agent_url,
            api_base=d["api_base"],
            gpu=d.get("gpu", ""),
            state=d.get("state", "READY"),
        )

    async def _call_node_stop(self, node: Node, model_id: str) -> None:
        """POST /local/stop to the Node Agent. Errors are logged, not raised."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(
                    f"{node.agent_url}/local/stop",
                    json={"model": model_id},
                )
        except Exception as exc:
            log.warning("node_stop_failed", model=model_id, node=node.node_id, error=str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_response(placement: Placement) -> EnsureResponse:
    primary = placement.instances[0] if placement.instances else None
    return EnsureResponse(
        model_id=placement.model_id,
        model_alias=primary.model_alias if primary else placement.model_id,
        api_base=primary.api_base if primary else "",
        instances=placement.instances,
    )
