"""
tests/resilience/conftest.py

Fixtures for resilience, fault, and load tests.

Duplicates the core fixtures from tests/integration/conftest.py
(fake_redis, scheduler, gateway_client) so that pytest can find them
for tests in this directory without relying on pytest_plugins magic.

Additionally defines the resilience-specific fixture helpers:
    slow_instance_effect     side_effect with artificial latency
    failing_instance_effect  side_effect that always returns 5xx
    flaky_instance_effect    side_effect that fails every N calls
"""

from __future__ import annotations

import pytest
import fakeredis
import httpx
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from scheduler.models import GpuInfo, HeartbeatPayload
from scheduler.placements import PlacementStore
from scheduler.registry import NodeRegistry
from scheduler.scheduler import Scheduler
from tests.utils.fault_injection import simulate_failure, simulate_flaky, simulate_latency


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

MRM_URL = "http://mrm-test:8010"
SCHEDULER_URL = "http://scheduler-test:8030"
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
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": MODEL_ALIAS,
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}

CHAT_REQUEST = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}

SSE_CHUNKS = [
    b'data: {"id":"c1","choices":[{"delta":{"role":"assistant","content":""},"index":0}]}\n\n',
    b'data: {"id":"c1","choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n',
    b'data: {"id":"c1","choices":[{"delta":{"content":" World"},"index":0}]}\n\n',
    b"data: [DONE]\n\n",
]


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------


@pytest.fixture
async def fake_redis():
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    yield redis
    await redis.aclose()


# ---------------------------------------------------------------------------
# Scheduler (real logic, fake Redis)
# ---------------------------------------------------------------------------


@pytest.fixture
async def scheduler(fake_redis):
    registry = NodeRegistry(fake_redis)
    placements = PlacementStore(fake_redis)
    return Scheduler(registry, placements)


# ---------------------------------------------------------------------------
# Reset gateway model_router metrics between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_model_router_metrics():
    """Reset ModelRouter metrics before each test to prevent cross-test pollution."""
    from gateway.services.router import model_router
    model_router.reset_metrics()
    yield


# ---------------------------------------------------------------------------
# Gateway ASGI client
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def node_ensure_payload(
    model_id: str = MODEL,
    api_base: str = VLLM_API_BASE,
) -> dict:
    return {
        "model_id": model_id,
        "model_alias": model_id.split("/")[-1].lower(),
        "api_base": api_base,
        "gpu": "0",
        "state": "READY",
        "container": f"mrm-{model_id.replace('/', '-').lower()}",
    }


# ---------------------------------------------------------------------------
# Resilience fixture helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def slow_instance_effect():
    """
    Returns a factory for a respx side_effect with configurable latency.

    Usage::

        mock.post(VLLM_URL).mock(side_effect=slow_instance_effect(delay_sec=0.3))
    """
    def _factory(delay_sec: float = 0.3, response: dict | None = None):
        return simulate_latency(delay_sec, response or VLLM_CHAT_RESPONSE)
    return _factory


@pytest.fixture
def failing_instance_effect():
    """
    Returns a factory for a respx side_effect that always returns an error.

    Usage::

        mock.post(VLLM_URL).mock(side_effect=failing_instance_effect(status_code=503))
    """
    def _factory(status_code: int = 500, detail: str = "GPU OOM"):
        return simulate_failure(status_code=status_code, detail=detail)
    return _factory


@pytest.fixture
def flaky_instance_effect():
    """
    Returns a factory for a respx side_effect that fails every N-th call.

    Usage::

        mock.post(VLLM_URL).mock(side_effect=flaky_instance_effect(fail_every_n=3))
    """
    def _factory(fail_every_n: int = 2, fail_status: int = 500):
        return simulate_flaky(VLLM_CHAT_RESPONSE, fail_every_n=fail_every_n, fail_status=fail_status)
    return _factory
