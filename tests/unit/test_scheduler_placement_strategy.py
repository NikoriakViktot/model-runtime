"""
tests/unit/test_scheduler_placement_strategy.py

Unit tests for the Scheduler's node placement strategies.

INVARIANTS
----------
- LeastLoaded always selects the node with the most free GPU memory.
- FirstFit always selects the first node in the list.
- An unknown strategy name falls back to least_loaded without crashing.
"""

import pytest

from scheduler.models import GpuInfo, Node
from scheduler.strategy import FirstFitStrategy, LeastLoadedStrategy, get_strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def node(node_id: str, free_mb: int, gpu_count: int = 1) -> Node:
    """Build a minimal Node with a given amount of free GPU memory."""
    gpus = [
        GpuInfo(
            gpu_index=str(i),
            memory_total_mb=24_000,
            memory_free_mb=free_mb // gpu_count,
            memory_used_mb=(24_000 - free_mb // gpu_count),
        )
        for i in range(gpu_count)
    ]
    return Node(node_id=node_id, agent_url=f"http://{node_id}:8020", gpus=gpus)


# ---------------------------------------------------------------------------
# LeastLoadedStrategy
# ---------------------------------------------------------------------------


class TestLeastLoadedStrategy:
    def test_selects_node_with_most_free_memory(self):
        """
        INVARIANT: the node with the most free GPU memory is selected.
        A high-memory node can load a model that a low-memory node cannot.
        """
        nodes = [node("a", free_mb=4_000), node("b", free_mb=20_000), node("c", free_mb=1_000)]
        chosen = LeastLoadedStrategy().select_node(nodes, "any-model")
        assert chosen.node_id == "b"

    def test_single_node_is_always_chosen(self):
        nodes = [node("only", free_mb=8_000)]
        chosen = LeastLoadedStrategy().select_node(nodes, "any-model")
        assert chosen.node_id == "only"

    def test_sums_across_multiple_gpus(self):
        """
        INVARIANT: total_free_mb sums across all GPUs on a node.
        A 2-GPU node with 6GB free each beats a 1-GPU node with 10GB free.
        """
        two_gpu = node("dual", free_mb=12_000, gpu_count=2)   # 6GB × 2 = 12GB
        one_gpu = node("single", free_mb=10_000, gpu_count=1)  # 10GB × 1 = 10GB
        chosen = LeastLoadedStrategy().select_node([two_gpu, one_gpu], "large-model")
        assert chosen.node_id == "dual"

    def test_tiebreak_deterministic(self):
        """Ties are broken by list position (first wins via max() stability)."""
        nodes = [node("a", free_mb=8_000), node("b", free_mb=8_000)]
        chosen = LeastLoadedStrategy().select_node(nodes, "any-model")
        # max() returns the first max element when values are equal
        assert chosen.node_id == "a"

    def test_model_id_does_not_affect_selection(self):
        """Strategy is model-agnostic — the model name has no influence."""
        nodes = [node("heavy", free_mb=20_000), node("light", free_mb=4_000)]
        for model_id in ("llama-2-7b", "qwen-72b", "unknown-model"):
            chosen = LeastLoadedStrategy().select_node(nodes, model_id)
            assert chosen.node_id == "heavy"


# ---------------------------------------------------------------------------
# FirstFitStrategy
# ---------------------------------------------------------------------------


class TestFirstFitStrategy:
    def test_always_returns_first_node(self):
        """
        INVARIANT: FirstFit picks the first node in the list regardless of
        memory state.  Useful for deterministic placement in tests.
        """
        nodes = [node("a", free_mb=1_000), node("b", free_mb=20_000)]
        chosen = FirstFitStrategy().select_node(nodes, "any-model")
        assert chosen.node_id == "a"

    def test_single_node_is_always_chosen(self):
        nodes = [node("only", free_mb=0)]
        chosen = FirstFitStrategy().select_node(nodes, "any-model")
        assert chosen.node_id == "only"


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


class TestGetStrategy:
    def test_returns_least_loaded_by_name(self):
        assert isinstance(get_strategy("least_loaded"), LeastLoadedStrategy)

    def test_returns_first_fit_by_name(self):
        assert isinstance(get_strategy("first_fit"), FirstFitStrategy)

    def test_unknown_name_falls_back_to_least_loaded(self):
        """
        INVARIANT: an unrecognised strategy name never crashes the scheduler;
        it falls back to least_loaded and logs a warning.
        """
        strategy = get_strategy("does_not_exist")
        assert isinstance(strategy, LeastLoadedStrategy)
