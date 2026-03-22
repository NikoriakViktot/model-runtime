"""
tests/system/conftest.py

Fixtures for system tests that run against a live docker-compose stack.

Prerequisites
-------------
The following services must be running before executing system tests:
    docker compose -f docker-compose.dev.yml up -d \\
        scheduler node_agent gateway model_runtime_manager

GATEWAY_USE_SCHEDULER should be set to "true" in the gateway environment for
the distributed-mode system tests, or "false" for single-node tests.

Skip behaviour
--------------
If the GATEWAY_URL env var is not set (or the gateway is not reachable),
all system tests are skipped automatically.  This prevents system tests from
accidentally running in unit/integration CI jobs.
"""

from __future__ import annotations

import os

import httpx
import pytest


GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8080")
SCHEDULER_URL = os.getenv("SCHEDULER_URL", "http://localhost:8030")
NODE_AGENT_URL = os.getenv("NODE_AGENT_URL", "http://localhost:8020")


def pytest_collection_modifyitems(config, items):
    """Skip all system tests if the gateway is not reachable."""
    skip_marker = pytest.mark.skip(reason="Gateway not reachable — start docker-compose first")
    try:
        with httpx.Client(timeout=3.0) as client:
            client.get(f"{GATEWAY_URL}/health")
    except Exception:
        for item in items:
            if "system" in str(item.fspath):
                item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def gateway_url() -> str:
    return GATEWAY_URL


@pytest.fixture(scope="session")
def scheduler_url() -> str:
    return SCHEDULER_URL


@pytest.fixture(scope="session")
def node_agent_url() -> str:
    return NODE_AGENT_URL


@pytest.fixture(scope="session")
async def live_gateway_client():
    """Long-lived httpx client for the live Gateway."""
    async with httpx.AsyncClient(
        base_url=GATEWAY_URL,
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
    ) as client:
        yield client


@pytest.fixture(scope="session")
async def live_scheduler_client():
    """Long-lived httpx client for the live Scheduler."""
    async with httpx.AsyncClient(base_url=SCHEDULER_URL, timeout=30.0) as client:
        yield client
