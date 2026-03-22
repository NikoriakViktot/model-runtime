"""
scheduler/placements.py

Redis-backed placement store.

Redis schema
------------
``scheduler:placement:{model_id}``   STRING  JSON-serialised Placement (no TTL —
                                              placements persist until explicitly
                                              deleted via /schedule/stop)
``scheduler:node_models:{node_id}``  SET     model_ids currently placed on a node
                                              (used to populate NodeSummary.models
                                               and for node-level cleanup)

No expiry is set on placement keys because a READY model is expected to stay
running until explicitly stopped.  If a node dies, the scheduler detects the
missing node on the next ensure call and re-places the model.
"""

from __future__ import annotations

import logging
import time

from redis.asyncio import Redis

from scheduler.models import Placement

logger = logging.getLogger(__name__)

_PLACEMENT_PREFIX = "scheduler:placement:"
_NODE_MODELS_PREFIX = "scheduler:node_models:"


def _placement_key(model_id: str) -> str:
    return f"{_PLACEMENT_PREFIX}{model_id}"


def _node_models_key(node_id: str) -> str:
    return f"{_NODE_MODELS_PREFIX}{node_id}"


class PlacementStore:
    """
    Wraps all Redis reads/writes for Placement objects.

    All methods are async.  The Redis client is injected at construction.
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def get(self, model_id: str) -> Placement | None:
        data = await self._redis.get(_placement_key(model_id))
        if data is None:
            return None
        return Placement.model_validate_json(data)

    async def list_all(self) -> list[Placement]:
        keys = await self._redis.keys(f"{_PLACEMENT_PREFIX}*")
        result: list[Placement] = []
        for key in keys:
            data = await self._redis.get(key)
            if data:
                result.append(Placement.model_validate_json(data))
        return result

    async def models_on_node(self, node_id: str) -> list[str]:
        """Return all model_ids currently placed on the given node."""
        members = await self._redis.smembers(_node_models_key(node_id))
        return list(members)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def save(self, placement: Placement) -> None:
        """Persist a placement and update the node → models index."""
        placement.updated_at = time.time()
        await self._redis.set(_placement_key(placement.model_id), placement.model_dump_json())
        for inst in placement.instances:
            await self._redis.sadd(_node_models_key(inst.node_id), placement.model_id)
        logger.debug("Saved placement model_id=%s nodes=%s", placement.model_id,
                     [i.node_id for i in placement.instances])

    async def delete(self, model_id: str) -> None:
        """Remove a placement and clean up node → models index."""
        placement = await self.get(model_id)
        if placement:
            for inst in placement.instances:
                await self._redis.srem(_node_models_key(inst.node_id), model_id)
        await self._redis.delete(_placement_key(model_id))
        logger.debug("Deleted placement model_id=%s", model_id)
