"""
gateway/services/proxy.py

Async HTTP proxy for forwarding requests to upstream services (vLLM, embeddings).

Two modes:
  - Unary:    await proxy.post(url, payload) → dict
  - Streaming: async for chunk in proxy.stream(url, payload): yield chunk

The proxy preserves:
  - Request body (with model field overridden to model_alias)
  - Authorization headers from the original client request
  - Content-Type

The proxy does NOT forward hop-by-hop headers (Connection, Transfer-Encoding).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)

# Headers that must not be forwarded to upstream
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",  # httpx sets this from the re-serialized body
    }
)


class UpstreamError(Exception):
    """Raised when the upstream service returns an error."""

    def __init__(self, message: str, status_code: int, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class ProxyService:
    """
    Stateful proxy backed by a shared httpx.AsyncClient connection pool.

    Lifecycle is managed by the FastAPI lifespan:
        await proxy.setup()
        await proxy.teardown()
    """

    def __init__(self) -> None:
        self._http: httpx.AsyncClient | None = None

    async def setup(self, timeout: float) -> None:
        """Open the underlying httpx connection pool."""
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=timeout,
                write=30.0,
                pool=10.0,
            ),
            # Follow redirects from upstream (e.g. litellm → vllm)
            follow_redirects=True,
        )
        logger.info("Proxy service ready (timeout=%.0fs)", timeout)

    async def teardown(self) -> None:
        """Close the connection pool."""
        if self._http:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # Unary request
    # ------------------------------------------------------------------

    async def post(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        client_headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Forward a POST request and return the parsed JSON response.

        Args:
            url:            Full target URL, e.g. ``http://container:8000/v1/chat/completions``.
            payload:        JSON body to forward.
            client_headers: Headers from the original client request to pass through.

        Returns:
            Parsed JSON response from the upstream service.

        Raises:
            UpstreamError:          Upstream returned a non-2xx status.
            httpx.ConnectError:     Upstream is unreachable.
            httpx.TimeoutException: Upstream did not respond in time.
        """
        self._assert_ready()
        headers = _build_forward_headers(client_headers)

        logger.debug("Proxy POST → %s", url)
        try:
            resp = await self._http.post(url, json=payload, headers=headers)
        except httpx.ConnectError as exc:
            raise UpstreamError(
                f"Cannot reach upstream at {url}: {exc}",
                status_code=503,
            ) from exc
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                f"Upstream timed out at {url}",
                status_code=504,
            ) from exc

        if resp.status_code >= 400:
            body = _safe_json(resp)
            raise UpstreamError(
                f"Upstream {url} returned {resp.status_code}",
                status_code=resp.status_code,
                body=body,
            )

        return resp.json()

    # ------------------------------------------------------------------
    # Streaming request
    # ------------------------------------------------------------------

    async def stream(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        client_headers: dict[str, str] | None = None,
    ) -> AsyncIterator[bytes]:
        """
        Forward a POST request and yield raw byte chunks as they arrive.

        vLLM uses Server-Sent Events (``data: {...}\\n\\n``).  This method
        passes bytes through verbatim without parsing, preserving the SSE
        framing for the client.

        Usage::

            return StreamingResponse(
                proxy.stream(url, payload),
                media_type="text/event-stream",
            )

        Raises:
            UpstreamError: If the upstream returns a non-2xx status on the
                           initial connection.
        """
        self._assert_ready()
        headers = _build_forward_headers(client_headers)

        logger.debug("Proxy STREAM → %s", url)
        try:
            async with self._http.stream(
                "POST", url, json=payload, headers=headers
            ) as resp:
                if resp.status_code >= 400:
                    await resp.aread()
                    body = _safe_json(resp)
                    raise UpstreamError(
                        f"Upstream {url} returned {resp.status_code}",
                        status_code=resp.status_code,
                        body=body,
                    )
                async for chunk in resp.aiter_bytes():
                    yield chunk
        except httpx.ConnectError as exc:
            raise UpstreamError(
                f"Cannot reach upstream at {url}: {exc}",
                status_code=503,
            ) from exc
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                f"Upstream stream timed out at {url}",
                status_code=504,
            ) from exc

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assert_ready(self) -> None:
        if self._http is None:
            raise RuntimeError(
                "ProxyService has not been initialized. "
                "Call await proxy.setup() in the application lifespan."
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

proxy = ProxyService()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_forward_headers(
    client_headers: dict[str, str] | None,
) -> dict[str, str]:
    """
    Build the headers dict to send upstream.

    Filters out hop-by-hop headers.  Passes Authorization through so that
    clients behind an API key can authenticate with the upstream engine.
    """
    if not client_headers:
        return {}
    return {
        k: v
        for k, v in client_headers.items()
        if k.lower() not in _HOP_BY_HOP
    }


def _safe_json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except Exception:
        return resp.text
