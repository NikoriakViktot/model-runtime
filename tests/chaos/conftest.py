"""
tests/chaos/conftest.py

Fixtures for chaos tests.  Same fixture set as resilience/conftest.py,
kept local so tests in this directory are self-contained.
"""

from __future__ import annotations

import pytest
import fakeredis
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from scheduler.models import GpuInfo, HeartbeatPayload
from scheduler.placements import PlacementStore
from scheduler.registry import NodeRegistry
from scheduler.scheduler import Scheduler


MRM_URL = "http://mrm-test:8010"
VLLM_API_BASE = "http://vllm-test:8000/v1"
MODEL = "meta-llama/Llama-2-7b-hf"
MODEL_ALIAS = "llama-2-7b-hf"

MRM_ENSURE_RESPONSE = {
    "base_model": MODEL,
    "model_alias": MODEL_ALIAS,
    "api_base": VLLM_API_BASE,
    "container": "mrm-llama",
    "gpu": "0",
    "state": "READY",
}

VLLM_CHAT_RESPONSE = {
    "id": "chatcmpl-chaos",
    "object": "chat.completion",
    "model": MODEL_ALIAS,
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

CHAT_REQUEST = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
async def fake_redis():
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    yield redis
    await redis.aclose()


@pytest.fixture
async def scheduler(fake_redis):
    registry = NodeRegistry(fake_redis)
    placements = PlacementStore(fake_redis)
    return Scheduler(registry, placements)


@pytest.fixture
async def gateway_client(monkeypatch):
    from gateway.config import settings as gw_settings

    monkeypatch.setattr(gw_settings, "mrm_url", MRM_URL)
    monkeypatch.setattr(gw_settings, "use_scheduler", False)
    monkeypatch.setattr(gw_settings, "mlflow_enabled", False)

    from gateway.main import app

    async with LifespanManager(app, startup_timeout=10, shutdown_timeout=5) as manager:
        async with AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as client:
            yield client


async def register_node(
    sched: Scheduler,
    node_id: str,
    agent_url: str,
    free_mb: int = 20_000,
) -> None:
    await sched.handle_heartbeat(
        HeartbeatPayload(
            node_id=node_id,
            agent_url=agent_url,
            hostname=node_id,
            gpus=[GpuInfo(gpu_index="0", memory_total_mb=24_000, memory_free_mb=free_mb)],
        )
    )
