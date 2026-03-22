# etl_service/api/adapter_materializer.py

import os
import shutil
import logging
import requests
from pathlib import Path

import boto3
from fastapi import HTTPException

logger = logging.getLogger("dispatcher.adapter_materializer")

CONTROL_PLANE_URL = os.getenv("CONTROL_PLANE_URL", "http://control_plane:8004")
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "/artifacts"))

S3_BUCKET = os.getenv("AWS_S3_BUCKET")
S3_KEY = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

s3 = boto3.client(
    "s3",
    aws_access_key_id=S3_KEY,
    aws_secret_access_key=S3_SECRET,
    region_name=REGION,
)


def _download_s3_prefix(bucket: str, prefix: str, local_dest: Path) -> None:
    paginator = s3.get_paginator("list_objects_v2")
    found = False

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue

            rel = key[len(prefix.rstrip("/") + "/"):]
            local_file = local_dest / rel
            local_file.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_file))
            found = True

    if not found:
        raise HTTPException(
            status_code=404,
            detail=f"No adapter files found at s3://{bucket}/{prefix}"
        )


def _materialize_adapter_by_run_id(run_id: str) -> str:
    """
    ЄДИНЕ канонічне місце materialize LoRA адаптерів:
    локально: /artifacts/adapters/{run_id}
    """
    # 1) CP metadata (optional)
    try:
        resp = requests.get(f"{CONTROL_PLANE_URL}/runs/{run_id}", timeout=5)
        resp.raise_for_status()
        run_data = resp.json()
    except Exception as e:
        logger.warning(f"CP fetch failed for {run_id}: {e}")
        run_data = {}

    contract = run_data.get("contract", {}).get("payload", {}) or {}
    target_slug = contract.get("target_slug") or contract.get("target_name") or "Unknown_Target"

    # 2) Local dest
    local_root = ARTIFACTS_DIR / "adapters" / run_id

    # ✅ single readiness check (choose ONE)
    # vLLM LoRA usually needs adapter_model.safetensors (+ config.json)
    if local_root.exists() and (local_root / "adapter_model.safetensors").exists():
        logger.info(f"⚡ Adapter {run_id} already materialized")
        return str(local_root)

    # 3) Clean + create
    if local_root.exists():
        shutil.rmtree(local_root)
    local_root.mkdir(parents=True, exist_ok=True)

    # 4) Download
    ok = _download_s3_smart(S3_BUCKET, run_id, target_slug, local_root)
    if not ok:
        shutil.rmtree(local_root, ignore_errors=True)
        raise HTTPException(status_code=404, detail=f"Adapter artifacts not found in S3 bucket {S3_BUCKET}")

    return str(local_root)
