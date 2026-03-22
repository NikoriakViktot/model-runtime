"""
scheduler/strategy.py

Pluggable placement strategies.

A strategy receives a non-empty list of healthy nodes and the model_id being
placed.  It returns the single node that should host the model.

Adding a new strategy
---------------------
1. Create a class with a ``select_node(nodes, model_id) -> Node`` method.
2. Register it in ``_STRATEGIES``.
3. Set ``SCHEDULER_PLACEMENT_STRATEGY=<name>`` in the environment.

No interface inheritance is required — the ``PlacementStrategy`` Protocol
is structural (duck-typed).
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from scheduler.models import Node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PlacementStrategy(Protocol):
    """
    Selects one node from a non-empty list of healthy nodes.

    Implementations must be safe to call concurrently from the asyncio loop.
    """

    def select_node(self, nodes: list[Node], model_id: str) -> Node:
        ...


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


class LeastLoadedStrategy:
    """
    Select the node with the most total free GPU memory.

    Rationale: free memory is the primary constraint for loading a large
    model.  Using free memory (not utilisation) avoids penalising nodes
    that are doing useful work on already-loaded models.

    Tiebreak: first node in the list (deterministic across calls).
    """

    def select_node(self, nodes: list[Node], model_id: str) -> Node:
        best = max(nodes, key=lambda n: n.total_free_mb)
        logger.debug(
            "LeastLoaded selected node_id=%s free_mb=%d for model=%s",
            best.node_id, best.total_free_mb, model_id,
        )
        return best


class FirstFitStrategy:
    """
    Select the first healthy node in the list.

    Useful when all nodes are identical or when you want deterministic
    placement for reproducibility in tests.
    """

    def select_node(self, nodes: list[Node], model_id: str) -> Node:
        logger.debug(
            "FirstFit selected node_id=%s for model=%s", nodes[0].node_id, model_id,
        )
        return nodes[0]


# ---------------------------------------------------------------------------
# Registry + factory
# ---------------------------------------------------------------------------


_STRATEGIES: dict[str, PlacementStrategy] = {
    "least_loaded": LeastLoadedStrategy(),
    "first_fit": FirstFitStrategy(),
}


def get_strategy(name: str) -> PlacementStrategy:
    strategy = _STRATEGIES.get(name)
    if strategy is None:
        logger.warning(
            "Unknown placement strategy '%s'. Falling back to 'least_loaded'. "
            "Valid options: %s",
            name, list(_STRATEGIES),
        )
        return _STRATEGIES["least_loaded"]
    logger.info("Placement strategy: %s", name)
    return strategy
