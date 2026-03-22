#frontend/ui/dispatcher_api.py

from __future__ import annotations
import requests
from typing import Optional

from .config import DISPATCHER_URL
from .config import LITELLM_URL
from .http import Http

class DispatcherAPI:
    def __init__(self, http: Http):
        self.http = http

    def list_models(self) -> list[dict]:
        r = requests.get(f"{LITELLM_URL}/v1/models", timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])

    def list_remote_epitaphs(self) -> list[dict]:
        """Отримує список епітафій з Project 1 через ETL сервіс"""
        try:
            # Endpoint defined in ETL service (main_api.py)
            data = self.http.get(f"{DISPATCHER_URL}/remote/epitaphs", timeout=10)
            return data if isinstance(data, list) else []
        except Exception:
            return []


    def upload_db(self, file_bytes: bytes, filename: str) -> dict | None:
        files = {"file": (filename, file_bytes, "application/octet-stream")}
        data = self.http.post(f"{DISPATCHER_URL}/dispatch/exp/upload", files=files, timeout=90)
        return data if isinstance(data, dict) else None

    def etl_targets(self, db_id: str) -> list[dict]:
        data = self.http.get(f"{DISPATCHER_URL}/dispatch/etl/targets", params={"db_id": db_id}, timeout=20)
        return data if isinstance(data, list) else []

    def materialize(self, run_id: str) -> dict | None:
        data = self.http.post(f"{DISPATCHER_URL}/dispatch/artifacts/materialize/{run_id}", timeout=60)
        return data if isinstance(data, dict) else None

    def preview(self, run_id: str, limit: int = 5) -> dict | None:
        data = self.http.get(f"{DISPATCHER_URL}/dispatch/preview", params={"run_id": run_id, "limit": limit}, timeout=60)
        return data if isinstance(data, dict) else None

    def delete_artifacts(self, run_id: str) -> bool:
        return self.http.delete(f"{DISPATCHER_URL}/dispatch/artifacts/{run_id}", timeout=30)

    def trained_models(self) -> list[dict]:
        data = self.http.get(f"{DISPATCHER_URL}/dispatch/models/trained", timeout=20)
        return data if isinstance(data, list) else []

    def load_adapter(self, lora_name: str, lora_path: str) -> dict | None:
        payload = {"lora_name": lora_name, "lora_path": lora_path}
        data = self.http.post(f"{DISPATCHER_URL}/dispatch/deploy/load", json_body=payload, timeout=180)
        return data if isinstance(data, dict) else None

    # =========================
    # dispatcher_api.py (client)
    # =========================
    def chat(
            self,
            model,
            messages,
            temperature=0.7,
            max_tokens=512,
            use_rag=False,
            rag_k=3,
            system_prompt=None,
            route="litellm",
    ):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "use_rag": use_rag,
            "rag_k": rag_k,
            "system_prompt": system_prompt,
            "route": route,
        }
        return self.http.post(
            f"{DISPATCHER_URL}/dispatch/chat",
            json_body=payload,
            timeout=120,
        )

    def get_vllm_health(self, base_model: str) -> dict:
        try:
            return self.http.get(
                f"{DISPATCHER_URL}/dispatch/metrics",
                params={"base_model": base_model},
                timeout=5
            )
        except:
            return {"error": "unreachable"}

    def get_trained_models(self) -> list[dict]:
        return self.trained_models()

    def prefetch_lora(self, run_id: str) -> dict | None:
        data = self.http.post(f"{DISPATCHER_URL}/dispatch/deploy/prefetch/{run_id}", timeout=180)
        return data if isinstance(data, dict) else None

    def register_lora(self, run_id: str) -> dict | None:
        """Реєструє LoRA в MRM і перезапускає модель при необхідності"""
        payload = {"run_id": run_id}

        data = self.http.post(f"{DISPATCHER_URL}/dispatch/deploy/register_lora", json_body=payload, timeout=600)
        return data if isinstance(data, dict) else None

    def get_available_adapters(self, base_model: str) -> list[dict]:
        """
        Отримує список успішно натренованих адаптерів для конкретної моделі.
        """
        params = {"base_model": base_model, "limit": 50}
        try:
            # Використовуємо self.http.get, щоб не хардкодити requests
            data = self.http.get(f"{DISPATCHER_URL}/dispatch/deploy/available_adapters", params=params, timeout=10)
            return data if isinstance(data, list) else []
        except Exception:
            return []



    def list_available_adapters(self, base_model: str) -> list[dict]:
        """Отримує історію з Control Plane"""
        try:
            return self.http.get(
                f"{DISPATCHER_URL}/dispatch/deploy/available_adapters",
                params={"base_model": base_model, "limit": 100}
            )
        except Exception:
            return []


    def sync_adapter(self, run_id: str) -> dict:
        """Тільки скачує файли (матеріалізація)"""
        return self.http.post(f"{DISPATCHER_URL}/dispatch/artifacts/materialize/{run_id}")

    def register_adapter(self, run_id: str) -> dict:
        """Додає в Redis (без рестарту)"""
        payload = {"run_id": run_id}
        # Цей ендпоінт треба трохи змінити, щоб він мав опцію "restart=False"
        # Але поки використаємо існуючий register_lora, він все робить разом
        return self.http.post(f"{DISPATCHER_URL}/dispatch/deploy/register_lora", json_body=payload, timeout=600)


    def get_runtime_status(self, base_model: str) -> dict:
        """Отримує реальний статус з MRM (через Dispatcher)"""
        try:
            return self.http.get(
                f"{DISPATCHER_URL}/dispatch/deploy/status",
                params={"base_model": base_model},
                timeout=5
            )
        except Exception:
            return {"state": "UNREACHABLE", "active_loras": []}

    def force_restart(self, base_model: str) -> dict | None:
        """Примусово перезапускає контейнер"""
        payload = {"base_model": base_model}
        return self.http.post(
            f"{DISPATCHER_URL}/dispatch/deploy/restart",
            json_body=payload,
            timeout=600 # Довгий таймаут, бо модель вантажиться
        )