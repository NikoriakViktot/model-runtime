"""
tests/integration/test_gateway.py

Integration tests for the Gateway's routing, proxying, and streaming behavior.

The Gateway is served via ASGI transport.  All outbound HTTP calls (to MRM,
Scheduler, vLLM) are intercepted by respx.

INVARIANTS UNDER TEST
---------------------
1. Routing correctness  — Gateway proxies to the api_base returned by MRM.
2. Alias swap           — ``model`` field in the forwarded body is the alias,
                          not the HuggingFace repo ID.
3. Streaming integrity  — SSE chunks arrive in order; last chunk is [DONE].
4. Backward compat      — Single api_base MRM response → instances list has
                          one element; routing still works.
5. Distributed mode     — When use_scheduler=True, Gateway calls Scheduler,
                          not MRM.
6. MRM unavailable      — Gateway returns 503, not an unhandled exception.
"""

from __future__ import annotations

import httpx
import pytest
import respx

pytestmark = pytest.mark.invariant

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
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}

CHAT_REQUEST = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Hi"}],
}

SSE_CHUNKS = (
    b'data: {"id":"c1","choices":[{"delta":{"role":"assistant","content":""},"index":0}]}\n\n'
    b'data: {"id":"c1","choices":[{"delta":{"content":"Hello"},"index":0}]}\n\n'
    b'data: {"id":"c1","choices":[{"delta":{"content":" World"},"index":0}]}\n\n'
    b"data: [DONE]\n\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def mock_mrm_ensure(mock: respx.MockRouter, response: dict = MRM_ENSURE_RESPONSE) -> None:
    mock.post(f"{MRM_URL}/models/ensure").mock(
        return_value=httpx.Response(200, json=response)
    )


def mock_vllm_chat(mock: respx.MockRouter, response: dict = VLLM_CHAT_RESPONSE) -> None:
    mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=response)
    )


# ---------------------------------------------------------------------------
# 1. Routing correctness
# ---------------------------------------------------------------------------


async def test_gateway_routes_to_api_base_from_mrm(gateway_client):
    """
    INVARIANT: the Gateway proxies the inference request to the api_base
    returned by MRM, not to a hardcoded URL.
    """
    async with respx.MockRouter() as mock:
        mock_mrm_ensure(mock)
        vllm_route = mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=VLLM_CHAT_RESPONSE)
        )

        resp = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert resp.status_code == 200
    assert vllm_route.called, "Gateway must forward the request to vLLM's api_base"


async def test_gateway_returns_vllm_response_body(gateway_client):
    """INVARIANT: the response body is the unmodified vLLM response."""
    async with respx.MockRouter() as mock:
        mock_mrm_ensure(mock)
        mock_vllm_chat(mock)

        resp = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert resp.json()["choices"][0]["message"]["content"] == "Hello!"


# ---------------------------------------------------------------------------
# 2. Model alias swap
# ---------------------------------------------------------------------------


async def test_gateway_swaps_model_field_to_alias(gateway_client):
    """
    INVARIANT: the ``model`` field in the body forwarded to vLLM must be
    the alias (what vLLM was started with), not the HuggingFace repo ID.

    vLLM rejects requests where ``model`` does not match ``--served-model-name``.
    """
    captured_body: dict = {}

    async def capture_vllm(request: httpx.Request) -> httpx.Response:
        import json
        captured_body.update(json.loads(request.content))
        return httpx.Response(200, json=VLLM_CHAT_RESPONSE)

    async with respx.MockRouter() as mock:
        mock_mrm_ensure(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(side_effect=capture_vllm)

        await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert captured_body.get("model") == MODEL_ALIAS, (
        f"Expected alias '{MODEL_ALIAS}', got '{captured_body.get('model')}'"
    )
    assert captured_body.get("model") != MODEL, (
        "Gateway must not forward the HuggingFace repo ID to vLLM"
    )


# ---------------------------------------------------------------------------
# 3. Streaming integrity
# ---------------------------------------------------------------------------


@pytest.mark.streaming
async def test_streaming_response_delivers_all_chunks_in_order(gateway_client):
    """
    INVARIANT: when stream=True, the Gateway delivers all SSE chunks to the
    client in the original order without corruption or truncation.
    The last meaningful line must be ``data: [DONE]``.
    """
    async with respx.MockRouter() as mock:
        mock_mrm_ensure(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                stream=httpx.ByteStream(SSE_CHUNKS),
                headers={"content-type": "text/event-stream"},
            )
        )

        collected: list[bytes] = []
        async with gateway_client.stream(
            "POST",
            "/v1/chat/completions",
            json={**CHAT_REQUEST, "stream": True},
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")
            async for chunk in response.aiter_bytes():
                if chunk:
                    collected.append(chunk)

    full_body = b"".join(collected)

    # All content tokens must be present
    assert b"Hello" in full_body
    assert b"World" in full_body

    # Last non-empty line must be the SSE terminator
    non_empty_lines = [ln for ln in full_body.split(b"\n") if ln.strip()]
    assert non_empty_lines[-1] == b"data: [DONE]", (
        "Streaming response must end with 'data: [DONE]'"
    )


@pytest.mark.streaming
async def test_streaming_response_has_correct_content_type(gateway_client):
    """
    INVARIANT: streaming responses must have Content-Type: text/event-stream
    so browsers and SSE clients can parse them correctly.
    """
    async with respx.MockRouter() as mock:
        mock_mrm_ensure(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                stream=httpx.ByteStream(SSE_CHUNKS),
                headers={"content-type": "text/event-stream"},
            )
        )

        async with gateway_client.stream(
            "POST",
            "/v1/chat/completions",
            json={**CHAT_REQUEST, "stream": True},
        ) as response:
            ct = response.headers.get("content-type", "")

    assert "text/event-stream" in ct


# ---------------------------------------------------------------------------
# 4. Backward compatibility — single api_base MRM response
# ---------------------------------------------------------------------------


async def test_backward_compat_single_api_base_mrm_response(gateway_client):
    """
    INVARIANT: when MRM returns the current single-instance format (no
    ``instances`` list), the Gateway must still route correctly.

    This ensures the migration to multi-instance MRM responses does not
    break existing single-node deployments.
    """
    # MRM response WITHOUT an instances list (current format)
    legacy_mrm_response = {
        "base_model": MODEL,
        "model_alias": MODEL_ALIAS,
        "api_base": VLLM_API_BASE,
        "gpu": "0",
        "state": "READY",
        # NOTE: no "instances" key
    }

    async with respx.MockRouter() as mock:
        mock.post(f"{MRM_URL}/models/ensure").mock(
            return_value=httpx.Response(200, json=legacy_mrm_response)
        )
        vllm_route = mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=VLLM_CHAT_RESPONSE)
        )

        resp = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert resp.status_code == 200
    assert vllm_route.called, "Legacy single-instance MRM response must still route"


async def test_backward_compat_single_api_base_synthesises_instances_list(monkeypatch):
    """
    INVARIANT: EnsureResult.from_dict() synthesises a one-element instances
    list from a single api_base so the router always has something to work with.
    """
    from gateway.services.mrm_client import EnsureResult

    legacy = {
        "base_model": MODEL,
        "model_alias": MODEL_ALIAS,
        "api_base": VLLM_API_BASE,
        "gpu": "0",
        "state": "READY",
    }
    result = EnsureResult.from_dict(legacy)

    assert len(result.instances) == 1
    assert result.instances[0].api_base == VLLM_API_BASE


# ---------------------------------------------------------------------------
# 5. Distributed mode — Gateway calls Scheduler
# ---------------------------------------------------------------------------


async def test_gateway_distributed_mode_calls_scheduler_not_mrm(
    gateway_client_with_scheduler,
):
    """
    INVARIANT: when GATEWAY_USE_SCHEDULER=True, the Gateway calls the
    Scheduler's /schedule/ensure endpoint instead of MRM directly.
    """
    scheduler_response = {
        "model_id": MODEL,
        "model_alias": MODEL_ALIAS,
        "api_base": VLLM_API_BASE,
        "instances": [
            {
                "instance_id": "inst-1",
                "model_id": MODEL,
                "model_alias": MODEL_ALIAS,
                "node_id": "node-1",
                "agent_url": "http://node-1:8020",
                "api_base": VLLM_API_BASE,
                "gpu": "0",
                "state": "READY",
                "placed_at": 1_700_000_000.0,
            }
        ],
    }

    async with respx.MockRouter(assert_all_called=False) as mock:
        scheduler_route = mock.post(f"{SCHEDULER_URL}/schedule/ensure").mock(
            return_value=httpx.Response(200, json=scheduler_response)
        )
        mrm_route = mock.post(f"{MRM_URL}/models/ensure").mock(
            return_value=httpx.Response(200, json=MRM_ENSURE_RESPONSE)
        )
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json=VLLM_CHAT_RESPONSE)
        )

        resp = await gateway_client_with_scheduler.post(
            "/v1/chat/completions", json=CHAT_REQUEST
        )

    assert resp.status_code == 200
    assert scheduler_route.called, "Distributed mode must call the Scheduler"
    assert not mrm_route.called, "Distributed mode must NOT call MRM directly"


# ---------------------------------------------------------------------------
# 6. Upstream unavailable → clean 503
# ---------------------------------------------------------------------------


async def test_gateway_returns_503_when_mrm_is_unreachable(gateway_client):
    """
    INVARIANT: if MRM is unreachable, the Gateway returns 503 with a
    meaningful error — it must not raise an unhandled exception.
    """
    async with respx.MockRouter() as mock:
        mock.post(f"{MRM_URL}/models/ensure").mock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        resp = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert resp.status_code == 503
    assert resp.json().get("detail") is not None


async def test_gateway_returns_502_when_vllm_errors(gateway_client):
    """
    INVARIANT: if vLLM returns a 5xx, the Gateway returns 502 to the client.
    """
    async with respx.MockRouter() as mock:
        mock_mrm_ensure(mock)
        mock.post(f"{VLLM_API_BASE}/chat/completions").mock(
            return_value=httpx.Response(500, json={"error": "CUDA OOM"})
        )

        resp = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# 7. Health endpoints are always available
# ---------------------------------------------------------------------------


async def test_gateway_health_does_not_depend_on_mrm(gateway_client):
    """/health must return 200 even when MRM is down."""
    async with respx.MockRouter(assert_all_called=False):
        resp = await gateway_client.get("/health")
    assert resp.status_code == 200
