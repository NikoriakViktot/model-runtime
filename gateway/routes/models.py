"""
gateway/routes/models.py

GET /v1/models — OpenAI-compatible model list.

Fetches live status from MRM and maps it to the OpenAI ``/v1/models`` schema.
Only models that are READY or have a known registration are returned.

OpenAI model list format:
  {
    "object": "list",
    "data": [
      {
        "id": "<model_alias>",
        "object": "model",
        "created": <unix_timestamp>,
        "owned_by": "mrm"
      },
      ...
    ]
  }

The ``id`` field is set to ``model_alias`` (what vLLM serves under), which
matches what clients should pass in ``/v1/chat/completions`` requests.
The ``base_model`` (HuggingFace repo ID) is included as extra metadata for
clients that need it.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from gateway.services.mrm_client import MRMUnavailableError, ModelStatus, mrm

logger = logging.getLogger(__name__)

router = APIRouter()

_CREATED_EPOCH = int(time.time())   # approximate; MRM doesn't expose created_at


@router.get("/v1/models")
async def list_models() -> JSONResponse:
    """
    Return all models known to MRM in OpenAI-compatible format.

    Models in any state (READY, STOPPED, ABSENT) are listed so that clients
    can discover what is available, not just what is currently running.
    Clients can then POST /v1/chat/completions with any listed model ID and
    the gateway will start it on demand.
    """
    try:
        statuses = await mrm.status_all()
    except MRMUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model runtime service is unavailable: {exc}",
        )

    data = [_to_openai_model(s) for s in statuses]

    logger.debug("GET /v1/models → %d models", len(data))

    return JSONResponse(
        content={
            "object": "list",
            "data": data,
        }
    )


@router.get("/v1/models/{model_id}")
async def get_model(model_id: str) -> JSONResponse:
    """
    Return status for a single model.

    ``model_id`` can be either the HuggingFace repo ID or the model alias.
    """
    try:
        status = await mrm.status(model_id)
    except MRMUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model runtime service is unavailable: {exc}",
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found: {exc}")

    return JSONResponse(content=_to_openai_model(status))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_openai_model(status: ModelStatus) -> dict:
    """Convert an MRM ModelStatus to the OpenAI model object schema."""
    return {
        "id": status.model_alias or status.base_model,
        "object": "model",
        "created": _CREATED_EPOCH,
        "owned_by": "mrm",
        # Non-standard extension — gives clients full context without a
        # separate API call.
        "metadata": {
            "base_model": status.base_model,
            "state": status.state,
            "running": status.running,
            "gpu": status.gpu,
            "active_loras": status.active_loras,
        },
    }
