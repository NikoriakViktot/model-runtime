"""
tests/integration/test_node_agent.py

Integration tests for the Node Agent's API behavior.

The Node Agent is served via ASGI transport.  All outbound HTTP calls to
the local MRM and the Scheduler are intercepted by respx.

INVARIANTS UNDER TEST
---------------------
- /local/ensure proxies the request to MRM and returns MRM's response verbatim.
- /local/stop proxies the request to MRM.
- /local/status returns MRM's full status list.
- /heartbeat triggers a heartbeat POST to the Scheduler.
- /health always returns 200.
"""

from __future__ import annotations

import httpx
import pytest
import respx

MRM_URL = "http://mrm-test:8010"
SCHEDULER_URL = "http://scheduler-test:8030"
MODEL = "meta-llama/Llama-2-7b-hf"

MRM_ENSURE_RESPONSE = {
    "base_model": MODEL,
    "model_alias": "llama-2-7b-hf",
    "api_base": "http://vllm-test:8000/v1",
    "gpu": "0",
    "state": "READY",
    "container": "mrm-llama",
}

MRM_STATUS_LIST = [
    {
        "base_model": MODEL,
        "model_alias": "llama-2-7b-hf",
        "api_base": "http://vllm-test:8000/v1",
        "state": "READY",
        "running": True,
        "gpu": "0",
        "active_loras": [],
    }
]


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


async def test_health_returns_200(node_agent_client):
    """Node Agent health probe must always return 200."""
    resp = await node_agent_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# /local/ensure
# ---------------------------------------------------------------------------


async def test_local_ensure_proxies_to_mrm_and_returns_verbatim(node_agent_client):
    """
    INVARIANT: POST /local/ensure forwards the request to MRM and returns
    MRM's response without modification.
    """
    async with respx.MockRouter() as mock:
        mock.post(f"{MRM_URL}/models/ensure").mock(
            return_value=httpx.Response(200, json=MRM_ENSURE_RESPONSE)
        )
        # Ignore heartbeat calls (background loop + initial heartbeat)
        mock.post(f"{SCHEDULER_URL}/heartbeat").mock(return_value=httpx.Response(204))

        resp = await node_agent_client.post("/local/ensure", json={"model": MODEL})

    assert resp.status_code == 200
    body = resp.json()
    assert body["api_base"] == MRM_ENSURE_RESPONSE["api_base"]
    assert body["model_alias"] == MRM_ENSURE_RESPONSE["model_alias"]
    assert body["state"] == "READY"


async def test_local_ensure_sends_base_model_to_mrm(node_agent_client):
    """
    INVARIANT: the model name must be forwarded to MRM as ``base_model``,
    not under a different key.
    """
    captured_body: dict = {}

    async def capture(request: httpx.Request) -> httpx.Response:
        import json
        captured_body.update(json.loads(request.content))
        return httpx.Response(200, json=MRM_ENSURE_RESPONSE)

    async with respx.MockRouter(assert_all_called=False) as mock:
        mock.post(f"{MRM_URL}/models/ensure").mock(side_effect=capture)
        mock.post(f"{SCHEDULER_URL}/heartbeat").mock(return_value=httpx.Response(204))

        await node_agent_client.post("/local/ensure", json={"model": MODEL})

    assert captured_body.get("base_model") == MODEL


async def test_local_ensure_propagates_mrm_error_status(node_agent_client):
    """
    INVARIANT: if MRM returns a non-200 status, the Node Agent propagates
    an appropriate error response to the caller.
    """
    async with respx.MockRouter(assert_all_called=False) as mock:
        mock.post(f"{MRM_URL}/models/ensure").mock(
            return_value=httpx.Response(404, json={"detail": "Model not registered"})
        )
        mock.post(f"{SCHEDULER_URL}/heartbeat").mock(return_value=httpx.Response(204))

        resp = await node_agent_client.post("/local/ensure", json={"model": "unknown-model"})

    assert resp.status_code >= 400


# ---------------------------------------------------------------------------
# /local/stop
# ---------------------------------------------------------------------------


async def test_local_stop_proxies_to_mrm(node_agent_client):
    """
    INVARIANT: POST /local/stop forwards the stop request to MRM.
    """
    async with respx.MockRouter(assert_all_called=False) as mock:
        stop_route = mock.post(f"{MRM_URL}/models/stop").mock(
            return_value=httpx.Response(200, json={})
        )
        mock.post(f"{SCHEDULER_URL}/heartbeat").mock(return_value=httpx.Response(204))

        resp = await node_agent_client.post("/local/stop", json={"model": MODEL})

    assert resp.status_code == 204
    assert stop_route.called


# ---------------------------------------------------------------------------
# /local/status
# ---------------------------------------------------------------------------


async def test_local_status_returns_mrm_model_list(node_agent_client):
    """
    INVARIANT: GET /local/status returns MRM's full model list verbatim.
    """
    async with respx.MockRouter(assert_all_called=False) as mock:
        mock.get(f"{MRM_URL}/models/status").mock(
            return_value=httpx.Response(200, json=MRM_STATUS_LIST)
        )
        mock.post(f"{SCHEDULER_URL}/heartbeat").mock(return_value=httpx.Response(204))

        resp = await node_agent_client.get("/local/status")

    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == 1
    assert body[0]["base_model"] == MODEL


# ---------------------------------------------------------------------------
# /heartbeat (trigger)
# ---------------------------------------------------------------------------


async def test_heartbeat_endpoint_sends_to_scheduler(node_agent_client):
    """
    INVARIANT: POST /heartbeat causes the agent to immediately POST to the
    Scheduler's /heartbeat endpoint.
    """
    async with respx.MockRouter(assert_all_called=False) as mock:
        scheduler_hb = mock.post(f"{SCHEDULER_URL}/heartbeat").mock(
            return_value=httpx.Response(204)
        )

        resp = await node_agent_client.post("/heartbeat")

    assert resp.status_code == 204
    assert scheduler_hb.called, "Triggering /heartbeat must call the Scheduler"


async def test_heartbeat_payload_includes_gpu_info_field(node_agent_client):
    """
    INVARIANT: the heartbeat payload sent to the Scheduler always includes
    a ``gpus`` field (even if empty on CPU-only nodes).
    """
    import json as _json

    captured: dict = {}

    async def capture_hb(request: httpx.Request) -> httpx.Response:
        captured.update(_json.loads(request.content))
        return httpx.Response(204)

    async with respx.MockRouter(assert_all_called=False) as mock:
        mock.post(f"{SCHEDULER_URL}/heartbeat").mock(side_effect=capture_hb)

        await node_agent_client.post("/heartbeat")

    assert "gpus" in captured, "Heartbeat payload must include 'gpus' field"
    assert "node_id" in captured
    assert "agent_url" in captured
