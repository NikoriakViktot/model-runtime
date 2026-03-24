"""
gateway/routes/embeddings.py

POST /v1/embeddings — OpenAI-compatible embeddings endpoint.

Routes to the Infinity embeddings service (``embeddings_api:7997``).
Infinity is OpenAI-compatible, so this is a thin proxy with no
transformation required.

OpenAI embeddings request:
  {
    "model": "text-embedding-ada-002",   ← ignored; Infinity uses its own model
    "input": "text to embed"             ← string or list[string]
  }

OpenAI embeddings response:
  {
    "object": "list",
    "data": [{"object": "embedding", "embedding": [...], "index": 0}],
    "model": "...",
    "usage": {"prompt_tokens": N, "total_tokens": N}
  }
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from gateway.config import settings
from gateway.services.proxy import UpstreamError, proxy

logger = logging.getLogger(__name__)

router = APIRouter()

_EMBEDDINGS_PATH = "/v1/embeddings"


@router.post("/v1/embeddings")
async def create_embeddings(request: Request) -> JSONResponse:
    """
    Create embeddings via the Infinity embeddings service.

    The ``model`` field is forwarded as-is; Infinity will use its loaded model
    regardless.  Clients can pass ``"model": "sentence-transformers/all-MiniLM-L6-v2"``
    to be explicit.
    """
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Request body must be valid JSON.")

    if "input" not in body:
        raise HTTPException(status_code=422, detail="'input' field is required.")

    target_url = f"{settings.embeddings_url.rstrip('/')}{_EMBEDDINGS_PATH}"

    try:
        result = await proxy.post(
            target_url,
            body,
            client_headers=dict(request.headers),
        )
    except UpstreamError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": str(exc)},
        )

    logger.debug("POST /v1/embeddings → %d embeddings", len(result.get("data", [])))
    return JSONResponse(content=result)
