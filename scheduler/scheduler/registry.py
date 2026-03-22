"""
scheduler/registry.py

Redis-backed node registry.

Redis schema
------------
``scheduler:node:{node_id}``   STRING  JSON-serialised Node, TTL = node_ttl_sec
``scheduler:nodes``            SET     All known node_ids (never expires; stale
                                        entries cleaned lazily on list_alive)

The TTL on the per-node key is the dead-man's switch: if a node stops sending
heartbeats its key expires and it disappears from list_alive automatically.
No explicit eviction job is needed.
"""

from __future__ import annotations

import logging

from redis.asyncio import Redis

from scheduler.config import settings
from scheduler.models import Node, NodeState

logger = logging.getLogger(__name__)

_NODES_INDEX = "scheduler:nodes"


def _node_key(node_id: str) -> str:
    return f"scheduler:node:{node_id}"


class NodeRegistry:
    """
    Wraps all Redis reads/writes for Node objects.

    All methods are async and must be called from the asyncio event loop.
    The Redis client is injected at construction time; this class owns no
    connection state of its own.
    """

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def upsert(self, node: Node) -> None:
        """
        Persist a node and reset its TTL.

        Called on every heartbeat received from the node.
        """
        key = _node_key(node.node_id)
        await self._redis.set(key, node.model_dump_json(), ex=settings.node_ttl_sec)
        await self._redis.sadd(_NODES_INDEX, node.node_id)
        logger.debug("Registry upsert node_id=%s ttl=%ds", node.node_id, settings.node_ttl_sec)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def get(self, node_id: str) -> Node | None:
        """Return the node or None if the TTL has expired (node is dead)."""
        data = await self._redis.get(_node_key(node_id))
        if data is None:
            return None
        return Node.model_validate_json(data)

    async def list_alive(self) -> list[Node]:
        """
        Return all nodes whose TTL has not expired.

        Dead node_ids (TTL expired but still in the index SET) are lazily
        removed from the index on each call.
        """
        node_ids: set[str] = await self._redis.smembers(_NODES_INDEX)
        nodes: list[Node] = []
        dead_ids: list[str] = []

        for nid in node_ids:
            node = await self.get(nid)
            if node is None:
                dead_ids.append(nid)
            else:
                nodes.append(node)

        if dead_ids:
            await self._redis.srem(_NODES_INDEX, *dead_ids)
            logger.info("Evicted %d dead nodes from index: %s", len(dead_ids), dead_ids)

        return nodes
