"""
tests/slo/conftest.py

Fixtures for SLO validation tests.

Re-exports the standard test fixtures so tests in this directory are
self-contained and don't rely on pytest_plugins magic.
"""

from __future__ import annotations

# Re-export core fixtures so pytest discovers them in this directory
from tests.resilience.conftest import (
    fake_redis,
    scheduler,
    gateway_client,
    reset_model_router_metrics,
    register_node,
    node_ensure_payload,
    MRM_URL,
    VLLM_API_BASE,
    MODEL,
    MODEL_ALIAS,
    VLLM_CHAT_RESPONSE,
    MRM_ENSURE_RESPONSE,
    CHAT_REQUEST,
    SSE_CHUNKS,
)

__all__ = [
    "fake_redis",
    "scheduler",
    "gateway_client",
    "reset_model_router_metrics",
    "register_node",
    "node_ensure_payload",
    "MRM_URL",
    "VLLM_API_BASE",
    "MODEL",
    "MODEL_ALIAS",
    "VLLM_CHAT_RESPONSE",
    "MRM_ENSURE_RESPONSE",
    "CHAT_REQUEST",
    "SSE_CHUNKS",
]
