import os
import logging
import httpx
from uuid import UUID

logger = logging.getLogger(__name__)

DISPATCHER_URL = os.getenv("API_DISPATCHER_URL", "http://api_dispatcher:8005")

_ENDPOINTS = {
    "DATASET_BUILD": "/etl/build",
    "TRAIN": "/dispatch/train/start",
    "EVAL": "/eval/run",
}


async def dispatch(
    action: str,
    run_id: UUID,
    *,
    contract_payload: dict | None = None,
    artifacts: dict | None = None,
):
    endpoint = _ENDPOINTS.get(action)
    if endpoint is None:
        logger.warning("Unknown dispatch action: %s", action)
        return

    if not contract_payload:
        raise ValueError(f"{action} requires contract_payload")

    payload = {
        "run_id": str(run_id),
        **contract_payload,
        **(artifacts or {}),
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{DISPATCHER_URL}{endpoint}",
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
    except Exception as e:
        logger.error("Dispatch %s failed for run %s: %s", action, run_id, e)
        raise
