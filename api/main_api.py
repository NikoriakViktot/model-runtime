#etl_service/api/main_api.py
from __future__ import annotations

import os
import shutil
import uuid
import requests
import json
from pathlib import Path
from typing import Optional, List, Dict,  Any
import sqlite3
from typing import Literal
from fastapi import Query
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import httpx
from urllib.parse import urlparse
import boto3
import logging



# Внутрішні модулі
from etl_service.celery_app import celery_app
from .chat_rag import extract_triggers
from .neo4j_rag import Neo4jRAG
from .context_builder import build_context
from .chat_api import router as chat_router
from .adapter_materializer import _materialize_adapter_by_run_id
from ..database import list_remote_epitaphs
from .notify_registry import NOTIFY_TARGETS

logger = logging.getLogger("dispatcher.train")

# --- CONFIG ---
app = FastAPI(
    title="AI Dispatcher Gateway",
    description="Unified Entry Point: Orchestrates Traffic between Frontend, Control Plane, and Workers.",
    root_path="/dispatch"
)

CONTROL_PLANE_URL = os.getenv("CONTROL_PLANE_URL", "http://control_plane:8004")
LLM_API_BASE = os.getenv("LLM_API_BASE", "http://litellm:4000")
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://vllm:8000/v1")
MRM_URL = os.getenv("MRM_URL", "http://model_runtime_manager:8010")

base_artifacts = Path(os.getenv("ARTIFACTS_DIR", "/artifacts"))


SHARED_UPLOAD_DIR = base_artifacts / "uploads"
ARTIFACTS_DIR = base_artifacts

SHARED_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "adapters").mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "datasets").mkdir(parents=True, exist_ok=True)

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
# ==========================================
# 1. DATA MODELS (Contract & ViewModels)
# ==========================================

class ProdBuildRequest(BaseModel):
    """Прод: працює з існуючим джерелом в Postgres"""
    source_id: str
    epitaph_id: str
    dataset_type: Literal["graph", "linear", "sft"] = "graph"


class ExpBuildRequest(BaseModel):
    """Експеримент: працює з завантаженим файлом і кастомними промптами"""
    db_file_id: str  # Шлях до файлу (internal path)
    target_name: str
    dataset_type: Literal["linear", "graph"] = "linear"
    prompt_id: str = "etl.profile_generator"
    prompt_version: str = "latest"


class TrainReq(BaseModel):
    parent_run_id: str               # <-- ID датасет-рану (dataset.build.v1)
    target_slug: str
    run_name: str | None = None
    epochs: int = 3
    learning_rate: float = 2e-5
    method: Literal["SFT", "DPO"] = "SFT"
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"

class TrainFromDatasetReq(BaseModel):
    dataset_run_id: str
    base_model: str
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 1
    method: Literal["SFT", "DPO"] = "SFT"
    run_name: str | None = None

class DatasetRef(BaseModel):
    uri: str


class TrainingConfig(BaseModel):
    epochs: int
    learning_rate: float

class TrainStartCommand(BaseModel):
    run_id: str                 # ID train run
    parent_run_id: str          # ID dataset run
    target_slug: str            # epitaph / persona
    base_model: str
    dataset: DatasetRef
    training: TrainingConfig

class TrainedModelView(BaseModel):
    """View Model для UI (зібрана з сирих даних CP)"""
    id: str
    name: str
    lora_path: str
    metrics: Dict[str, Any]
    created_at: str


class LoadAdapterReq(BaseModel):
    lora_name: str
    lora_path: str


class UnifiedBuildRequest(BaseModel):
    run_id: str
    db_id: Optional[str] = None
    source_id: Optional[str] = None
    target_name: str
    dataset_type: str
    options: Dict[str, Any] = {}
    output: Dict[str, Any] = {}

class ChatReq(BaseModel):
    model: str = Field(..., description="Назва адаптера або базової моделі")
    messages: List[Dict[str, Any]] = Field(..., description="Історія чату")

    # Робимо поля опціональними. Ніякого хардкоду.
    # Якщо клієнт не передав - буде None.
    temperature: Optional[float] = Field(default=None, description="Креативність (0.0 - 1.0)")
    max_tokens: Optional[int] = Field(default=None, description="Ліміт генерації")
    top_p: Optional[float] = Field(default=None, description="Nucleus sampling")
    frequency_penalty: Optional[float] = Field(default=None)
    presence_penalty: Optional[float] = Field(default=None)

class RegisterLoraReq(BaseModel):
    run_id: str

class RestartReq(BaseModel):
    base_model: str


# ==========================================
# 2. HELPER: CONTROL PLANE COMMUNICATION
# ==========================================

def _register_contract(payload: dict) -> dict:
    """Реєструє намір в ядрі. Повертає Run ID та Next Action."""
    contract = {
        "type": "dataset.build.v1",
        "spec_version": "v1",
        "payload": payload
    }

    # Якщо це тренування - змінюємо тип
    if "training" in payload:
        method = payload.get("method", "SFT")
        contract["type"] = "train.dpo.v1" if method == "DPO" else "train.qlora.v1"

    try:
        res = requests.post(f"{CONTROL_PLANE_URL}/contracts", json=contract, timeout=10)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Control Plane Error: {e}")


def _download_s3_folder(bucket_name: str, s3_prefix: str, local_dir: Path):
    """
    Скачує вміст 'папки' з S3 у локальну директорію.
    Використовує s3_prefix як корінь.
    """
    print(f"🔄 Syncing S3 prefix: {s3_prefix} -> {local_dir}")

    paginator = s3.get_paginator('list_objects_v2')
    # Додаємо '/' в кінець префіксу, щоб точно шукати як папку
    prefix_filter = s3_prefix if s3_prefix.endswith('/') else f"{s3_prefix}/"

    found_files = False

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix_filter):
        if "Contents" not in page:
            continue

        for obj in page["Contents"]:
            key = obj["Key"]
            # Ігноруємо саму папку (якщо S3 повернув її як об'єкт)
            if key.endswith('/'):
                continue

            found_files = True

            # Логіка імені файлу: беремо тільки ім'я файлу (adapter_model.safetensors)
            # Це працює для плоскої структури LoRA (стандарт HuggingFace)
            filename = Path(key).name
            local_file_path = local_dir / filename

            # Скачуємо тільки якщо файлу немає або він має 0 байт
            if not local_file_path.exists() or local_file_path.stat().st_size == 0:
                print(f"   ⬇️ Downloading {filename}...")
                s3.download_file(bucket_name, key, str(local_file_path))
            else:
                print(f"   ✅ Skipped {filename} (exists)")

    if not found_files:
        print(f"⚠️ Warning: No files found in S3 under {s3_prefix}")


# ==========================================
# 🛠️ ROBUST S3 DOWNLOADER
# ==========================================

def _list_s3_debug(bucket: str, prefix: str):
    """Виводить в лог реальний вміст бакета (для дебагу)"""
    logger.info(f"🕵️ DEBUG: Listing S3 content under '{prefix}'...")
    try:
        # Беремо трохи ширше, щоб побачити сусідів
        search_prefix = "/".join(prefix.rstrip("/").split("/")[:-1])
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=search_prefix, MaxKeys=50)
        if "Contents" in resp:
            for obj in resp["Contents"]:
                logger.info(f"   📄 Found: {obj['Key']} (Size: {obj['Size']})")
        else:
            logger.warning(f"   ❌ No objects found starting with {search_prefix}")
    except Exception as e:
        logger.error(f"   💥 Error listing S3: {e}")


def _download_s3_smart(bucket: str, run_id: str, target_slug: str, local_dest: Path) -> bool:
    """
    Пробує знайти адаптер, перебираючи різні варіанти шляхів.
    """
    candidates = [
        f"loras/{target_slug}/{run_id}",  # Стандарт (як в базі)
        f"loras/{target_slug.replace(' ', '_')}/{run_id}",  # З підкресленням
        f"loras/{run_id}",  # Плоска структура
        f"artifacts/loras/{target_slug}/{run_id}"  # Альтернативний шлях
    ]

    for prefix in candidates:
        logger.info(f"🔎 Checking S3 candidate: s3://{bucket}/{prefix}")

        # Нормалізація слешів
        s3_prefix = prefix if prefix.endswith('/') else f"{prefix}/"

        paginator = s3.get_paginator('list_objects_v2')
        found_files = 0

        for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
            if "Contents" not in page: continue

            for obj in page["Contents"]:
                key = obj["Key"]
                if key.endswith('/'): continue  # Це папка

                # key: loras/Nancy Garcia/run123/config.json
                # s3_prefix: loras/Nancy Garcia/run123/
                # rel_path: config.json
                rel_path = key[len(s3_prefix):]

                local_file = local_dest / rel_path
                local_file.parent.mkdir(parents=True, exist_ok=True)

                # Скачуємо
                s3.download_file(bucket, key, str(local_file))
                found_files += 1

        if found_files > 0:
            logger.info(f"✅ Downloaded {found_files} files from {s3_prefix}")
            return True

    # Якщо нічого не знайшли
    logger.error(f"❌ Adapter {run_id} not found in any candidate path.")
    _list_s3_debug(bucket, f"loras/{target_slug}")
    return False


# ==========================================
# 🚀 DEPLOYMENT ENDPOINTS
# ==========================================
@app.get("/deploy/status")
def get_model_status(base_model: str):
    """Питає MRM про статус і список адаптерів"""
    try:
        # Проксіюємо запит до MRM
        # MRM URL: http://model_runtime_manager:8010/models/status/{base_model}
        # Але base_model містить слеші, тому краще через requests params або url encoding,
        # але MRM runtime.py приймає шлях.
        # Використаємо безпечніший варіант:
        resp = requests.get(f"{MRM_URL}/models/status_one", params={"base_model": base_model}, timeout=5)

        # Якщо status_one ще не реалізований як GET param, використаємо шлях:
        if resp.status_code == 404 or resp.status_code == 405:
            resp = requests.get(f"{MRM_URL}/models/status/{base_model}", timeout=5)

        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"Status Check Failed: {e}")
        # Повертаємо об'єкт з помилкою, щоб UI міг це показати
        return {"state": "UNREACHABLE", "active_loras": [], "error": str(e)}


@app.post("/deploy/restart")
def force_restart_model(req: RestartReq):
    """Примусовий рестарт (Stop -> Ensure)"""
    logger.info(f"🔄 FORCE RESTART REQUEST: {req.base_model}")
    try:
        # 1. Stop (зупиняє контейнер і звільняє GPU)
        requests.post(f"{MRM_URL}/models/stop", json={"base_model": req.base_model}, timeout=30)

        # 2. Ensure (це змусить MRM перечитати Redis, сформувати нову команду з адаптерами і запустити)
        res = requests.post(f"{MRM_URL}/models/ensure", json={"base_model": req.base_model}, timeout=600)
        res.raise_for_status()

        return res.json()
    except Exception as e:
        logger.error(f"Restart Error: {e}")
        raise HTTPException(status_code=502, detail=f"Restart failed: {e}")

@app.post("/deploy/register_lora")
def register_lora_and_reload(req: RegisterLoraReq):
    run_id = req.run_id
    logger.info(f"⚙️ REGISTER LORA REQUEST: {run_id}")

    # 1. Матеріалізація (Скачування)
    try:
        local_path = _materialize_adapter_by_run_id(run_id)
    except HTTPException as he:
        raise he  # Прокидаємо 404
    except Exception as e:
        logger.exception("Materialization crashed")
        raise HTTPException(status_code=500, detail=f"Materialization failed: {str(e)}")

    # 2. Реєстрація в MRM
    # MRM має бачити файл за тим самим шляхом (/artifacts/adapters/...)
    mrm_payload = {
        "base_model": "Qwen/Qwen1.5-1.8B-Chat",
        "lora_id": run_id,
        "host_path": local_path
    }

    # Спробуємо отримати base_model з бази
    try:
        r_info = requests.get(f"{CONTROL_PLANE_URL}/runs/{run_id}", timeout=2).json()
        base = r_info.get("contract", {}).get("payload", {}).get("base_model")
        if base:
            mrm_payload["base_model"] = base
    except:
        pass  # Fallback to default

    logger.info(f"Sending to MRM: {mrm_payload}")

    try:
        reg_resp = requests.post(f"{MRM_URL}/models/lora/register", json=mrm_payload, timeout=10)
        # Якщо MRM повертає помилку, ми хочемо бачити її текст
        if reg_resp.status_code >= 400:
            raise HTTPException(status_code=reg_resp.status_code, detail=f"MRM Error: {reg_resp.text}")

        reg_data = reg_resp.json()
    except Exception as e:
        logger.error(f"MRM Connection Error: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to connect to MRM: {e}")

    # 3. Рестарт (якщо треба)
    restarted = False
    if reg_data.get("needs_restart"):
        base_model = mrm_payload["base_model"]
        logger.info(f"🔄 Triggering restart for {base_model}...")
        try:
            requests.post(f"{MRM_URL}/models/stop", json={"base_model": base_model}, timeout=30)
            # Ensure запустить з новими параметрами
            requests.post(f"{MRM_URL}/models/ensure", json={"base_model": base_model}, timeout=600)
            restarted = True
        except Exception as e:
            logger.error(f"Restart failed: {e}")
            # Не фейлимо весь запит, бо реєстрація пройшла
            return {"status": "registered_but_restart_failed", "error": str(e)}

    return {
        "status": "success",
        "run_id": run_id,
        "local_path": local_path,
        "vllm_restarted": restarted
    }


@app.get("/deploy/available_adapters")
def list_available_adapters(
        base_model: str = Query(..., description="Base model to filter by"),
        limit: int = 50
):
    """Proxy to CP to get successful runs"""
    try:
        params = {
            "run_type": "train.qlora.v1",
            "base_model": base_model,
            "exclude_status": "FAILED",
            "limit": limit
        }
        resp = requests.get(f"{CONTROL_PLANE_URL}/runs-filter", params=params, timeout=10)
        return resp.json()
    except Exception as e:
        logger.error(f"CP Error: {e}")
        return []  # Return empty list on error


app.include_router(chat_router, tags=["chat"])

def ensure_base_model_running(base_model: str) -> dict:
    r = requests.post(f"{MRM_URL}/models/ensure", json={"base_model": base_model}, timeout=30)
    r.raise_for_status()
    return r.json()

app.include_router(chat_router, tags=["chat"])

# ==========================================
# 3. DATASET FLOWS (ETL)
# ==========================================

@app.get("/etl/targets")
def get_extraction_targets(db_id: str):
    """
    Повертає список унікальних персонажів (targets) з таблиці messages.
    """
    # 1. Пошук файлу (залишаємо як було виправлено раніше)
    db_path = Path(db_id)
    if not db_path.exists():
        potential_path = SHARED_UPLOAD_DIR / Path(db_id).name
        if potential_path.exists():
            db_path = potential_path
        else:
            print(f"❌ DB not found: {db_id}")
            raise HTTPException(404, f"Database file not found: {db_id}")

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # ✅ ВИПРАВЛЕНО: Запит не до sqlite_master, а до messages
            # Перевіряємо, чи існує таблиця messages
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages';")
            if not cursor.fetchone():
                return []  # Таблиці messages немає

            # Групуємо повідомлення за epitaph (ім'я персонажа), де роль = target
            query = """
                    SELECT epitaph, COUNT(*) as cnt
                    FROM messages
                    WHERE sender_role = 'target'
                    GROUP BY epitaph
                    ORDER BY cnt DESC \
                    """
            cursor.execute(query)
            rows = cursor.fetchall()

            # Формуємо список для фронтенду
            targets = [{"epitaph": row[0], "cnt": row[1]} for row in rows]

        return targets

    except Exception as e:
        print(f"Error reading DB: {e}")
        raise HTTPException(500, f"Failed to read DB: {str(e)}")
@app.get("/etl/datasets")
def list_available_datasets():
    runs_dir = ARTIFACTS_DIR / "runs"
    if not runs_dir.exists():
        return []

    out = []
    for run_dir in runs_dir.iterdir():
        ds_dir = run_dir / "datasets"
        if ds_dir.exists() and any(ds_dir.glob("*.jsonl")):
            out.append(run_dir.name)  # run_id
    return out

@app.post("/prod/build")
def start_production_build(req: ProdBuildRequest):
    # 1. Контракт
    payload = {
        "db_id": "postgres",  # Маркер джерела
        "source_id": req.source_id,
        "target_name": req.epitaph_id,
        "dataset_type": req.dataset_type,
        "options": {
            "prompt_id": "etl.profile_generator",
            "prompt_version": "prod-stable"
        },
        "output": {"base_uri": "s3://artifacts/prod/datasets"}
    }

    # 2. Реєстрація
    cp_resp = _register_contract(payload)

    # 3. Виконання
    if cp_resp.get("next_action") == "DATASET_BUILD":
        task = celery_app.send_task(
            "etl.build_dataset",
            kwargs={
                "run_id": cp_resp["run_id"],
                "epitaph_id": req.epitaph_id,
                "source_config": {
                    "type": "postgres",
                    "source_id": req.source_id,
                    "options": payload["options"]
                }
            },
            queue="etl_queue"
        )
        return {"status": "started", "run_id": cp_resp["run_id"], "task_id": task.id}

    return {"status": "registered", "run_id": cp_resp["run_id"]}


@app.post("/exp/upload")
async def upload_experimental_db(file: UploadFile = File(...)):
    """Приймає файл від ML-інженера"""
    file_id = uuid.uuid4().hex
    path = SHARED_UPLOAD_DIR / f"{file_id}_{file.filename}"
    with path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"file_id": str(path), "filename": file.filename}


@app.post("/exp/build")
def start_experimental_build(req: ExpBuildRequest):
    # 1. Контракт
    payload = {
        "db_id": req.db_file_id,
        "target_name": req.target_name,
        "dataset_type": req.dataset_type,
        "options": {
            "prompt_id": req.prompt_id,
            "prompt_version": req.prompt_version
        },
        "output": {"base_uri": "s3://artifacts/exp/datasets"}
    }

    # 2. Реєстрація
    cp_resp = _register_contract(payload)

    # 3. Виконання
    if cp_resp.get("next_action") == "DATASET_BUILD":
        task = celery_app.send_task(
            "etl.build_dataset",
            kwargs={
                "run_id": cp_resp["run_id"],
                "epitaph_id": req.target_name,
                "source_config": {
                    "type": "sqlite",
                    "db_path": req.db_file_id,
                    "target_name": req.target_name,
                    "options": payload["options"]
                }
            },
            queue="etl_queue"
        )
        return {"status": "started", "run_id": cp_resp["run_id"], "task_id": task.id}

    return {"status": "registered", "run_id": cp_resp["run_id"]}


# ==========================================
# 4. TRAINING FLOW
# ==========================================

# @app.post("/train/from-dataset")
# def train_from_dataset(req: TrainFromDatasetReq):
#     dataset_run_id = req.dataset_run_id
#
#     # 1️⃣ Забираємо dataset run
#     ds_run = requests.get(
#         f"{CONTROL_PLANE_URL}/runs/{dataset_run_id}",
#         timeout=10
#     ).json()
#
#     if not ds_run:
#         raise HTTPException(404, "Dataset run not found")
#
#     contract = ds_run.get("contract", {})
#     ds_payload = contract.get("payload", {})
#     artifacts = ds_run.get("artifacts", {})
#
#     target_slug = ds_payload.get("target_name")
#     if not target_slug:
#         raise HTTPException(409, "Dataset run has no target_name")
#
#     dataset_uri = (
#         artifacts.get("dataset_uri")
#         or artifacts.get("s3_uri")
#     )
#     if not dataset_uri:
#         raise HTTPException(409, "Dataset URI not found in dataset run")
#
#     # 2️⃣ Формуємо TRAIN контракт
#     train_payload = {
#         "parent_run_id": dataset_run_id,
#         "target_slug": target_slug,
#         "base_model": req.base_model,
#         "dataset": {"uri": dataset_uri},
#         "training": {
#             "epochs": req.epochs,
#             "learning_rate": req.learning_rate,
#             "batch_size": req.batch_size,
#         },
#         "method": req.method,
#         "run_name": req.run_name,
#         "output": {
#             "lora_base_uri": f"s3://{S3_BUCKET}/loras"
#         },
#     }
#
#     contract_type = "train.dpo.v1" if req.method == "DPO" else "train.qlora.v1"
#
#     # 3️⃣ Реєструємо контракт у Control Plane
#     cp_resp = requests.post(
#         f"{CONTROL_PLANE_URL}/contracts",
#         json={
#             "type": contract_type,
#             "spec_version": "v1",
#             "payload": train_payload,
#         },
#         timeout=10
#     )
#     cp_resp.raise_for_status()
#
#     train_run = cp_resp.json()
#     train_run_id = train_run["run_id"]
#
#     # 4️⃣ Одразу стартуємо тренінг
#     start_resp = requests.post(
#         f"{DISPATCHER_URL}/dispatch/train/start",
#         json={"run_id": train_run_id},
#         timeout=10
#     )
#     start_resp.raise_for_status()
#
#     return {
#         "status": "training_started",
#         "dataset_run_id": dataset_run_id,
#         "train_run_id": train_run_id,
#     }

# @app.post("/train/start")
# def start_training(req: dict):
#     run_id = req["run_id"]
#
#     # 1️⃣ Забираємо RUN з Control Plane
#     run = requests.get(
#         f"{CONTROL_PLANE_URL}/runs/{run_id}",
#         timeout=10
#     ).json()
#
#     if not run or "contract" not in run:
#         raise HTTPException(404, "Run or contract not found")
#
#     payload = run["contract"]["payload"]
#
#     # 2️⃣ ВИТЯГУЄМО ВСЮ ПРАВДУ З КОНТРАКТУ
#     parent_run_id = payload["parent_run_id"]
#     target_slug = payload["target_slug"]
#     base_model = payload["base_model"]
#
#     training = payload["training"]
#     epochs = training["epochs"]
#     learning_rate = training["learning_rate"]
#
#     method = payload.get("method", "SFT")
#
#     dataset_uri = (
#         payload.get("dataset", {}) or payload.get("prefs_dataset", {})
#     ).get("uri")
#
#     if not dataset_uri:
#         raise HTTPException(409, "Dataset URI missing in contract")
#
#     # 3️⃣ Матеріалізуємо датасет ЛОКАЛЬНО (dispatcher responsibility)
#     mat = materialize_dataset(parent_run_id)
#     dataset_local_path = mat["local_path"]
#
#     # 4️⃣ Запускаємо Celery — ОДИН РАЗ
#     task = celery_app.send_task(
#         "tasks_sft.run_training",
#         kwargs={
#             "run_id": run_id,
#             "target_slug": target_slug,
#             "dataset_path": dataset_local_path,
#             "run_name": payload.get("run_name", ""),
#             "epochs": epochs,
#             "learning_rate": learning_rate,
#             "method": method,
#             "base_model": base_model,
#             "dataset_version": parent_run_id,
#         },
#         queue="sft_queue",
#     )
#
#     # 5️⃣ Фіксуємо ФАКТ dispatch
#     requests.post(
#         f"{CONTROL_PLANE_URL}/events",
#         json={
#             "run_id": run_id,
#             "event": "TRAIN_DISPATCHED",
#             "artifacts": {
#                 "celery_task_id": task.id,
#                 "worker_queue": "sft_queue"
#             },
#         },
#         timeout=5,
#     )
#
#     return {
#         "status": "started",
#         "run_id": run_id,
#         "task_id": task.id,
#     }



@app.post("/train/start")
def start_training(req: dict):
    # ------------------------------------------------------------------
    # 1. Transport layer (HTTP)
    # ------------------------------------------------------------------
    if "run_id" not in req:
        raise HTTPException(status_code=422, detail="run_id is required")

    run_id: str = req["run_id"]

    # ------------------------------------------------------------------
    # 2. Load run from Control Plane
    # ------------------------------------------------------------------
    run = requests.get(
        f"{CONTROL_PLANE_URL}/runs/{run_id}",
        timeout=5
    ).json()

    if "contract" not in run or "payload" not in run["contract"]:
        raise HTTPException(
            status_code=422,
            detail="run.contract.payload is missing"
        )

    # ------------------------------------------------------------------
    # 3. Domain layer (TRAIN contract)
    # ------------------------------------------------------------------
    contract_payload: dict = run["contract"]["payload"]
    artifacts: dict = run.get("artifacts") or {}

    # --- required fields ---
    try:
        epitaph_id: str = contract_payload["epitaph_id"]
        target_slug: str = contract_payload["target_slug"]
        base_model: str = contract_payload["base_model"]
        training_cfg: dict = contract_payload["training"]
    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=f"TRAIN contract missing required field: {e}"
        )

    # ------------------------------------------------------------------
    # 4. Dataset resolution (artifact-first, payload-fallback)
    # ------------------------------------------------------------------
    dataset_uri: str | None = (
        artifacts.get("dataset_uri")
        or contract_payload.get("dataset", {}).get("uri")
    )

    if not dataset_uri:
        raise HTTPException(
            status_code=422,
            detail="dataset_uri missing for TRAIN"
        )

    dataset_version: str | None = contract_payload.get("parent_run_id")

    # ------------------------------------------------------------------
    # 5. Dispatch Celery task (execution layer)
    # ------------------------------------------------------------------
    task = celery_app.send_task(
        "tasks_sft.run_training",
        kwargs={
            "run_id": run_id,
            "epitaph_id": epitaph_id,
            "target_slug": target_slug,
            "run_name": target_slug,
            "dataset_uri": dataset_uri,
            "base_model": base_model,
            "epochs": training_cfg["epochs"],
            "learning_rate": training_cfg["learning_rate"],
            "dataset_version": dataset_version,
        },
        queue="sft_queue",
    )

    # ------------------------------------------------------------------
    # 6. Response (no domain logic here)
    # ------------------------------------------------------------------
    return {
        "status": "started",
        "task_id": task.id,
        "run_id": run_id,
    }


@app.post("/notify")
async def notify(payload: dict):
    notify = payload["notify"]

    target = notify["target"]
    epitaph_id = notify["epitaph_id"]
    status = notify["status"]

    cfg = NOTIFY_TARGETS[target]

    body = {
        "epitaph_id": epitaph_id,
        "status": status,
    }

    async with httpx.AsyncClient() as client:
        await client.post(
            cfg["url"],
            json=body,
            headers=cfg["headers"],
            timeout=10
        )

    return {"status": "sent"}


@app.get("/runs/{run_id}")
def get_run_status(run_id: str):
    """Proxy for Pollers"""
    return requests.get(f"{CONTROL_PLANE_URL}/runs/{run_id}").json()


@app.get("/deploy/available_adapters")
def list_available_adapters(
        base_model: str = Query(..., description="Base model to filter by"),
        limit: int = 50
):
    """
    Повертає список успішно натренованих адаптерів для конкретної моделі.
    Проксіює запит до Control Plane з правильними фільтрами.
    """
    try:
        # Формуємо параметри для Control Plane
        params = {
            "run_type": "train.qlora.v1",  # Тільки тренування
            "base_model": base_model,  # Тільки для цієї моделі
            "exclude_status": "FAILED",  # Без помилок (можна змінити на status="TRAIN_READY" або "DONE")
            "limit": limit
        }

        # Викликаємо Control Plane (зверни увагу на правильний шлях /runs-filter, який ти створив)
        resp = requests.get(f"{CONTROL_PLANE_URL}/runs-filter", params=params, timeout=10)
        resp.raise_for_status()

        runs = resp.json() or []

        result = []
        for r in runs:
            contract = (r.get("contract") or {}).get("payload") or {}
            artifacts = r.get("artifacts") or {}

            result.append({
                "run_id": r.get("id"),
                "name": contract.get("target_slug") or contract.get("target_name") or "Unknown",
                "created_at": r.get("created_at"),
                "status": r.get("state"),
                "metrics": artifacts.get("metrics_uri"),
            })
        return result

    except Exception as e:
        logger.error(f"Failed to fetch adapters: {e}")
        raise HTTPException(status_code=502, detail=f"Control Plane Error: {e}")
# ==========================================
# 5. MODELS & DEPLOYMENT API
# ==========================================

@app.get("/models/trained", response_model=List[TrainedModelView])
def list_trained_models():
    """
    Сканує S3 на наявність натренованих LoRA адаптерів.
    Повертає список моделей з метриками.
    """
    print(f"🔍 Scanning S3 bucket {S3_BUCKET} for models in 'loras/'...")

    try:
        # 1. Отримуємо список всіх об'єктів у папці loras/
        # Використовуємо paginator, щоб не втратити файли, якщо їх > 1000
        paginator = s3.get_paginator('list_objects_v2')
        page_iterator = paginator.paginate(Bucket=S3_BUCKET, Prefix="loras/")

        adapters = {}
        metrics_map = {}

        for page in page_iterator:
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]

                # Шукаємо головний файл адаптера
                if key.endswith("adapter_model.safetensors"):
                    # Структура шляху зазвичай: loras/<Target_Name>/<Run_ID>/adapter_model.safetensors
                    parts = key.split("/")

                    # Захист від нестандартних шляхів
                    if len(parts) >= 3:
                        run_id = parts[-2]  # Передостанній елемент - це ID рану
                        target_name = parts[-3]  # Перед ним - ім'я цілі
                    else:
                        run_id = "unknown"
                        target_name = "unknown"

                    # Зберігаємо інформацію про модель
                    parent_dir = "/".join(parts[:-1])  # Шлях до папки без файлу
                    adapters[run_id] = {
                        "id": run_id,
                        "name": target_name,
                        "lora_path": f"s3://{S3_BUCKET}/{parent_dir}",
                        "created_at": obj["LastModified"].isoformat()
                    }

                # Запам'ятовуємо, де лежать метрики (metrics.json)
                if key.endswith("metrics.json"):
                    parts = key.split("/")
                    if len(parts) >= 2:
                        run_id = parts[-2]  # ID рану - це назва папки
                        metrics_map[run_id] = key

        # 2. Формуємо фінальний список
        results = []
        for run_id, data in adapters.items():
            metrics = {}

            # Якщо для цього рану є файл метрик - читаємо його
            if run_id in metrics_map:
                try:
                    print(f"   ⬇️ Downloading metrics for {run_id}...")
                    metric_obj = s3.get_object(Bucket=S3_BUCKET, Key=metrics_map[run_id])
                    metrics_content = metric_obj["Body"].read().decode("utf-8")
                    metrics = json.loads(metrics_content)
                except Exception as e:
                    print(f"   ⚠️ Could not read metrics for {run_id}: {e}")

            results.append(TrainedModelView(
                id=data["id"],
                name=data["name"],
                lora_path=data["lora_path"],
                metrics=metrics,
                created_at=data["created_at"]
            ))

        # Сортуємо: найновіші зверху
        results.sort(key=lambda x: x.created_at, reverse=True)

        print(f"✅ Found {len(results)} trained models.")
        return results

    except Exception as e:
        print(f"❌ Error listing models from S3: {e}")
        # Повертаємо пустий список, щоб не ламати фронтенд помилкою 500
        return []

# ==========================================
# 6. PROMPT PROXY
# ==========================================

@app.get("/prompts")
def proxy_list_prompts():
    return requests.get(f"{CONTROL_PLANE_URL}/prompts/").json()


@app.get("/prompts/{pid}/versions")
def proxy_list_versions(pid: str):
    return requests.get(f"{CONTROL_PLANE_URL}/prompts/{pid}/versions").json()


@app.post("/prompts/version")
def proxy_create_version(payload: dict):
    r = requests.post(f"{CONTROL_PLANE_URL}/prompts/version", json=payload)
    return r.json()


# ==========================================
# 7. CHAT & INFERENCE
# ==========================================


@app.post("/etl/build")
def unified_dataset_build(req: UnifiedBuildRequest):
    print(f"📥 Received Build Request for Run: {req.run_id}")

    # 1. Отримуємо Epitaph UUID (він приходить з фронта)
    # Фронт кидає його в db_id або source_id, але це ЗАВЖДИ Epitaph UUID
    epitaph_uuid = req.db_id or req.source_id

    if not epitaph_uuid:
        raise HTTPException(status_code=400, detail="Missing Epitaph UUID")

    # 2. Формуємо конфіг для воркера
    source_config = {
        "type": "postgres",
        "db_id": str(epitaph_uuid),
        "options": req.options
    }

    # 3. Відправляємо задачу в чергу
    # В tasks.py epitaph_id буде саме UUID
    task = celery_app.send_task(
        "etl.build_dataset",
        kwargs={
            "run_id": req.run_id,
            "epitaph_id": str(epitaph_uuid),
            "dataset_type": req.dataset_type,
            "source_config": source_config
        },
        queue="etl_queue"
    )

    return {"status": "started", "task_id": task.id}


@app.get("/preview")
def preview_dataset(run_id: str, limit: int = 3):
    """
    1. Викликає materialize_dataset (скачує файл з S3).
    2. Читає локальний файл.
    3. Віддає JSON для UI.
    """
    # Це гарантує, що файл буде на диску
    materialize_dataset(run_id)

    # Формуємо шлях до локальної папки
    run_dir = ARTIFACTS_DIR / "datasets" / run_id

    # Шукаємо jsonl файли
    files = list(run_dir.glob("*.jsonl"))

    if not files:
        raise HTTPException(404, "No dataset files found locally after materialization")

    samples = []
    try:
        with open(files[0], "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                samples.append(json.loads(line))
    except Exception as e:
        print(f"❌ Error reading file {files[0]}: {e}")
        raise HTTPException(500, f"Error reading dataset file: {e}")

    return {
        "run_id": run_id,
        "file": files[0].name,
        "samples": samples,
    }


@app.post("/artifacts/materialize/{run_id}")
def materialize_dataset(run_id: str):
    """
    Завантажує файл з S3 на диск API Dispatcher-а.
    """
    print(f"📥 Materializing dataset for run: {run_id}")

    # 1. Отримуємо інфо про Run
    try:
        run_data = requests.get(f"{CONTROL_PLANE_URL}/runs/{run_id}", timeout=10).json()
    except Exception as e:
        raise HTTPException(500, f"Failed to connect to Control Plane: {e}")

    artifacts = run_data.get("artifacts") or {}

    # 2. Витягуємо URI (з вашої бази ми знаємо, що це dataset_uri)
    dataset_uri = artifacts.get("dataset_uri") or artifacts.get("s3_uri")

    if not dataset_uri:
        print(f"⚠️ No dataset URI found in artifacts: {artifacts}")
        raise HTTPException(409, "Dataset not ready (missing dataset_uri)")

    # 3. Парсимо S3 шлях
    # dataset_uri = s3://epitaphs-work-dir/datasets/Mark Smith/...
    parsed = urlparse(dataset_uri)
    bucket = parsed.netloc  # epitaphs-work-dir
    key = parsed.path.lstrip("/")  # datasets/Mark Smith/...

    # 4. Готуємо локальну папку
    local_dir = ARTIFACTS_DIR / "datasets" / run_id
    local_dir.mkdir(parents=True, exist_ok=True)

    local_path = local_dir / Path(key).name

    # 5. Скачуємо (якщо ще немає)
    if not local_path.exists():
        print(f"⬇️ Downloading s3://{bucket}/{key} -> {local_path}")
        try:
            # Використовуємо глобальний об'єкт s3, ініціалізований зверху
            s3.download_file(bucket, key, str(local_path))
            print("✅ Download successful")
        except Exception as e:
            print(f"❌ S3 Download Error: {e}")
            raise HTTPException(500, f"Failed to download from S3: {e}")

    # 6. Повідомляємо систему (опціонально)
    requests.post(
        f"{CONTROL_PLANE_URL}/events",
        json={
            "run_id": run_id,
            "event": "DATASET_MATERIALIZED",
            "artifacts": {
                "dataset_local_path": str(local_path)
            }
        },
        timeout=5,
    )

    return {
        "run_id": run_id,
        "dataset_uri": dataset_uri,
        "local_path": str(local_path),
    }
@app.delete("/artifacts/{run_id}")
def cleanup_run_artifacts(run_id: str):
    local_dir = ARTIFACTS_DIR / "runs" / run_id
    if local_dir.exists():
        shutil.rmtree(local_dir)

    return {"status": "cleaned", "run_id": run_id}


# Додай імпорт


@app.post("/deploy/register_lora")
def register_lora_and_reload(req: RegisterLoraReq):
    """
    1. Матеріалізує адаптер з S3.
    2. Реєструє його в MRM (Redis).
    3. ПЕРЕЗАПУСКАЄ vLLM, щоб він підхопив нові аргументи --lora-modules.
    """
    run_id = req.run_id
    logger.info(f"⚙️ Processing LoRA Registration for {run_id}")

    # 1. Отримуємо інфо про модель (Base Model)
    try:
        run_data = requests.get(f"{CONTROL_PLANE_URL}/runs/{run_id}", timeout=5).json()
        contract = run_data.get("contract", {}).get("payload", {})
        base_model = contract.get("base_model")
        if not base_model:
            raise HTTPException(400, "Base model not defined in run contract")
    except Exception as e:
        raise HTTPException(404, f"Run metadata not found: {e}")

    # 2. Materialize (Safely)
    try:
        local_path = _materialize_adapter_by_run_id(run_id)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Materialization crashed")
        raise HTTPException(status_code=500, detail=f"Materialization failed: {str(e)}")

    # 3. Реєструємо в MRM
    mrm_payload = {
        "base_model": base_model,
        "lora_id": run_id,
        "host_path": local_path
    }

    try:
        reg_resp = requests.post(f"{MRM_URL}/models/lora/register", json=mrm_payload, timeout=10)
        reg_resp.raise_for_status()
        reg_data = reg_resp.json()
    except Exception as e:
        raise HTTPException(502, f"MRM Registration Failed: {e}")

    # 4. Перезапуск (Restart Policy)
    # Якщо MRM каже, що контейнер запущений, його треба рестартнути,
    # щоб він підхопив нові аргументи запуску (--lora-modules)
    restarted = False
    if reg_data.get("needs_restart"):
        logger.info(f"🔄 Restarting vLLM for {base_model} to load new adapters...")
        try:
            # Stop
            requests.post(f"{MRM_URL}/models/stop", json={"base_model": base_model}, timeout=30)
            # Ensure (Start fresh with new args)
            requests.post(f"{MRM_URL}/models/ensure", json={"base_model": base_model}, timeout=600)
            restarted = True
        except Exception as e:
            raise HTTPException(502, f"Failed to restart model: {e}")

    return {
        "status": "registered",
        "run_id": run_id,
        "base_model": base_model,
        "vllm_restarted": restarted,
        "mrm_response": reg_data
    }


@app.post("/chat_via_litellm")
def chat_via_litellm(req: ChatReq):
    payload = req.model_dump(exclude_unset=True)
    model_input = payload.pop("model")  # Це run_id (LoRA) або base:...
    messages = payload.get("messages", [])

    lora_path = None
    epitaph_id = None
    dataset_run_id = None  # Важливо: ID рану, який породив граф

    # --- Змінна, щоб тримати інфо про активну модель ---
    active_model_alias = None

    # =============================
    # CASE 1: LoRA (model_input = run_id)
    # =============================
    if model_input and not model_input.startswith("base:"):
        logger.info(f"🧬 Chat with LoRA run_id={model_input}")

        # 1. Отримуємо інфо про тренування з Control Plane
        try:
            run = requests.get(f"{CONTROL_PLANE_URL}/runs/{model_input}", timeout=5).json()
        except Exception:
            run = None

        if not run:
            raise HTTPException(404, "Run not found")

        contract_payload = run["contract"]["payload"]

        # 2. Витягуємо параметри
        base_model = contract_payload["base_model"]

        # target_slug або target_name - це наш epitaph_id
        epitaph_id = contract_payload.get("target_slug") or contract_payload.get("target_name")

        # CRITICAL: Parent Run ID = Dataset Run ID
        # Саме він записаний у вузлах Neo4j
        dataset_run_id = contract_payload.get("parent_run_id")

        # 3. Гарантуємо, що базова модель запущена
        info = ensure_base_model_running(base_model)
        requests.post(f"{MRM_URL}/models/touch/{base_model}", timeout=1)

        # Запам'ятовуємо аліас (наприклад 'qwen-7b-instruct')
        active_model_alias = info["model_alias"]

        # 4. Матеріалізуємо адаптер (скачуємо з S3)
        host_path = _materialize_adapter_by_run_id(model_input)

        # 5. Шлях для контейнера (vLLM)
        # Оскільки ми мапимо /app/artifacts -> /app/artifacts один в один:
        lora_path = host_path

        # 6. Оновлюємо payload для LiteLLM
        payload["model"] = active_model_alias  # qwen-7b-instruct

        # Передаємо lora_request для vLLM
        payload["extra_body"] = {
            "lora_request": {
                "lora_name": model_input,
                "lora_path": lora_path
            }
        }


    # =============================
    # CASE 2: Base Model
    # =============================
    else:
        base_model = model_input.replace("base:", "")
        logger.info(f"🤖 Chat with base_model={base_model}")
        epitaph_id = "base"  # Немає персонажа

        info = ensure_base_model_running(base_model)
        requests.post(f"{MRM_URL}/models/touch", json={"base_model": base_model}, timeout=1)

        active_model_alias = info["model_alias"]
        payload["model"] = active_model_alias

    # =============================
    # 🧠 RAG CONTEXT INJECTION (WITH ACTIVE MODEL)
    # =============================
    last_user_msg = next((m for m in reversed(messages) if m["role"] == "user"), None)

    if last_user_msg and epitaph_id and epitaph_id != "base":
        try:
            user_text = last_user_msg["content"]

            # ТУТ ЗМІНА: Передаємо активну модель для екстракції ключових слів
            # Це працює швидко, бо модель вже прогріта викликом ensure вище
            keywords = extract_triggers(user_text, model_name=active_model_alias)

            rag = Neo4jRAG()
            # ТУТ КЛЮЧОВИЙ МОМЕНТ: Передаємо dataset_run_id для фільтрації пам'яті
            snippets = rag.retrieve(
                epitaph_id=epitaph_id,
                keywords=keywords,
                dataset_run_id=dataset_run_id,  # <--- Фільтр по пам'яті (тільки для LoRA)
                limit=5
            )
            rag.close()

            if snippets:
                logger.info(f"📚 RAG Found {len(snippets)} facts for {epitaph_id} (Dataset: {dataset_run_id})")
                context_str = build_context(snippets)

                # Додаємо System Prompt з контекстом
                system_prompt = f"""You are {epitaph_id}. 
                STRICTLY adhere to the following memory context. 
                If the context contains facts, use them. 
                Do not hallucinate info not present in your memory or 
                common knowledge consistent with the persona.
                
                Memory Context:
                {context_str}
                """
                # Вставляємо системне повідомлення на початок (або замінюємо існуюче)
                if messages[0]["role"] == "system":
                    messages[0]["content"] = system_prompt
                else:
                    messages.insert(0, {"role": "system", "content": system_prompt})
            else:
                logger.info("📚 RAG: No relevant memories found.")

        except Exception as e:
            logger.warning(f"⚠️ RAG logic failed: {e}")

    # Оновлюємо повідомлення в payload
    payload["messages"] = messages

    # =============================
    # Виклик LiteLLM
    # =============================
    try:
        # Важливо: LiteLLM повинен прокидати extra_body
        r = requests.post(
            f"{LLM_API_BASE}/chat/completions",
            json=payload,
            timeout=120,
        )
        if r.status_code != 200:
            logger.error(f"Inference Error: {r.text}")

        r.raise_for_status()
        return r.json()

    except Exception as e:
        logger.exception("❌ Inference critical failure")
        raise HTTPException(500, str(e))


@app.post("/deploy/prefetch/{run_id}")
def prefetch_lora(run_id: str):
    host_path = _materialize_adapter_by_run_id(run_id)
    return {"run_id": run_id, "local_path": host_path}


@app.get("/remote/epitaphs")
def get_remote_epitaphs_list():
    """
    Повертає список епітафій з бази даних Project 1 (Postgres).
    Використовується фронтендом для вибору джерела даних.
    """
    try:
        epitaphs = list_remote_epitaphs()
        # Серіалізуємо UUID у строки для JSON відповіді
        for e in epitaphs:
            if 'id' in e:
                e['id'] = str(e['id'])
            if 'created_at' in e and e['created_at']:
                e['created_at'] = str(e['created_at'])
        return epitaphs
    except Exception as e:
        logger.error(f"Failed to fetch remote epitaphs: {e}")
        # Повертаємо 500, щоб UI знав про помилку підключення
        raise HTTPException(status_code=500, detail=f"Database Connection Error: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok", "role": "Dispatcher/Gateway"}