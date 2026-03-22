"""
tests/integration/conftest.py

Shared fixtures for all integration tests.

Fixture overview
----------------
fake_redis          FakeAsyncRedis — no real Redis required
scheduler           Real Scheduler wired to fake_redis
register_node       Helper coroutine to add a node via heartbeat
gateway_client      Gateway ASGI test client (settings patched before lifespan)
node_agent_client   Node Agent ASGI test client (settings patched before lifespan)

HTTP mocking strategy
---------------------
All outbound HTTP calls (to MRM, vLLM, Node Agent, Scheduler) are intercepted
with respx.MockRouter inside each test.  The fixtures here never mock HTTP —
each test owns its own respx scope so mocks are fully isolated.
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


# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------


@pytest.fixture
async def fake_redis():
    """In-process fake Redis. No external service required."""
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    yield redis
    await redis.aclose()


# ---------------------------------------------------------------------------
# Scheduler (real logic, fake Redis)
# ---------------------------------------------------------------------------


@pytest.fixture
async def scheduler(fake_redis):
    """
    Real Scheduler instance backed by fake Redis.

    Nothing is mocked here — the scheduler's locking, placement, and
    failover logic runs exactly as in production.
    """
    registry = NodeRegistry(fake_redis)
    placements = PlacementStore(fake_redis)
    return Scheduler(registry, placements)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def register_node(
    sched: Scheduler,
    node_id: str,
    agent_url: str,
    free_mb: int = 20_000,
) -> None:
    """Register a node by sending a heartbeat to the scheduler."""
    await sched.handle_heartbeat(
        HeartbeatPayload(
            node_id=node_id,
            agent_url=agent_url,
            hostname=node_id,
            gpus=[GpuInfo(gpu_index="0", memory_total_mb=24_000, memory_free_mb=free_mb)],
        )
    )


def node_agent_ensure_payload(
    model_id: str,
    api_base: str = "http://vllm-test:8000/v1",
) -> dict:
    """Canonical Node Agent /local/ensure response body."""
    return {
        "model_id": model_id,
        "model_alias": model_id.split("/")[-1].lower().replace("-", "_"),
        "api_base": api_base,
        "gpu": "0",
        "state": "READY",
        "container": f"mrm-{model_id.replace('/', '-').lower()}",
    }


# ---------------------------------------------------------------------------
# Node Agent ASGI client
# ---------------------------------------------------------------------------


@pytest.fixture
async def node_agent_client(monkeypatch):
    """
    Node Agent served via ASGI transport with lifespan triggered.

    Settings are patched BEFORE the lifespan runs so that local_mrm.setup()
    and send_heartbeat() use the test URLs.  The heartbeat interval is set to
    1 hour to prevent background loops from firing during tests.

    The initial heartbeat in lifespan will fail (no scheduler running) but
    that error is caught and logged — it does not abort startup.
    """
    from node_agent.config import settings as agent_settings

    monkeypatch.setattr(agent_settings, "mrm_url", "http://mrm-test:8010")
    monkeypatch.setattr(agent_settings, "scheduler_url", "http://scheduler-test:8030")
    monkeypatch.setattr(agent_settings, "node_id", "test-node-01")
    monkeypatch.setattr(agent_settings, "agent_url", "http://node-agent-test:8020")
    monkeypatch.setattr(agent_settings, "heartbeat_interval_sec", 3600)

    from node_agent.app import app

    async with LifespanManager(app, startup_timeout=10, shutdown_timeout=5) as manager:
        async with AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as client:
            yield client


# ---------------------------------------------------------------------------
# Gateway ASGI client
# ---------------------------------------------------------------------------


@pytest.fixture
async def gateway_client(monkeypatch):
    """
    Gateway served via ASGI transport.

    Operates in single-node mode (use_scheduler=False) against a fake MRM URL.
    MLflow is disabled so no background threads are started.
    """
    from gateway.config import settings as gw_settings

    monkeypatch.setattr(gw_settings, "mrm_url", "http://mrm-test:8010")
    monkeypatch.setattr(gw_settings, "use_scheduler", False)
    monkeypatch.setattr(gw_settings, "mlflow_enabled", False)

    from gateway.main import app

    async with LifespanManager(app, startup_timeout=10, shutdown_timeout=5) as manager:
        async with AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as client:
            yield client


@pytest.fixture
async def gateway_client_with_scheduler(monkeypatch):
    """
    Gateway served via ASGI transport, in distributed mode (use_scheduler=True).
    """
    from gateway.config import settings as gw_settings

    monkeypatch.setattr(gw_settings, "scheduler_url", "http://scheduler-test:8030")
    monkeypatch.setattr(gw_settings, "use_scheduler", True)
    monkeypatch.setattr(gw_settings, "mlflow_enabled", False)

    from gateway.main import app

    async with LifespanManager(app, startup_timeout=10, shutdown_timeout=5) as manager:
        async with AsyncClient(
            transport=ASGITransport(app=manager.app), base_url="http://test"
        ) as client:
            yield client
