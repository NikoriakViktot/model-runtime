#infrastructure/dispatcher.py

import os
import requests
from uuid import UUID

# This points to the Dispatcher Microservice (ex-ETL service API)
DISPATCHER_URL = os.getenv("API_DISPATCHER_URL", "http://api_dispatcher:8005")
# infrastructure/dispatcher.py

def dispatch(
    action: str,
    run_id: UUID,
    *,
    contract_payload: dict | None = None,
    artifacts: dict | None = None,
):
    timeout = 10

    if action in ("TRAIN", "DATASET_BUILD", "EVAL"):
        if not contract_payload:
            raise ValueError(f"{action} requires contract_payload")

        payload = {
            "run_id": str(run_id),
            **contract_payload,
            **(artifacts or {}),
        }

    try:
        if action == "DATASET_BUILD":
            requests.post(
                f"{DISPATCHER_URL}/etl/build",
                json=payload,
                timeout=timeout,
            ).raise_for_status()
            return

        if action == "TRAIN":
            requests.post(
                f"{DISPATCHER_URL}/dispatch/train/start",
                json=payload,
                timeout=timeout,
            ).raise_for_status()
            return

        if action == "EVAL":
            requests.post(
                f"{DISPATCHER_URL}/eval/run",
                json=payload,
                timeout=timeout,
            ).raise_for_status()
            return

        if action == "NOTIFY":
            requests.post(
                f"{DISPATCHER_URL}/notify",
                json=artifacts or {},
                timeout=5,
            ).raise_for_status()
            return

        print(f"⚠️ [Dispatcher] Unknown action: {action}")

    except Exception as e:
        print(f"❌ [Dispatcher] Failed to dispatch {action} for run {run_id}: {e}")
