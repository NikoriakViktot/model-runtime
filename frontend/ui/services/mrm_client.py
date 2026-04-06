"""
ui/services/mrm_client.py
Typed client for the Model Runtime Manager (port 8010).
"""
from __future__ import annotations
import os
from .base import APIResult, BaseClient

_BASE = os.getenv("MRM_URL", "http://model_runtime_manager:8010")


class MRMClient(BaseClient):
    def __init__(self, base_url: str = _BASE):
        super().__init__(base_url, default_timeout=10)

    # ── Model lifecycle ───────────────────────────────────────────────

    def ensure(self, base_model: str) -> APIResult:
        return self._post("/models/ensure", body={"base_model": base_model}, timeout=660)

    def stop(self, base_model: str) -> APIResult:
        return self._post("/models/stop", body={"base_model": base_model}, timeout=60)

    def remove(self, base_model: str) -> APIResult:
        return self._post("/models/remove", body={"base_model": base_model}, timeout=60)

    def touch(self, base_model: str) -> APIResult:
        return self._post("/models/touch", body={"base_model": base_model}, timeout=10)

    # ── Status ────────────────────────────────────────────────────────

    def status(self, base_model: str) -> APIResult:
        return self._get(f"/models/status/{base_model}", timeout=5)

    def status_all(self) -> APIResult:
        return self._get("/models/status", timeout=5)

    # ── GPU ───────────────────────────────────────────────────────────

    def gpu_metrics(self) -> APIResult:
        return self._get("/gpu/metrics", timeout=5)

    # ── HuggingFace ───────────────────────────────────────────────────

    def hf_search(self, q: str, limit: int = 20) -> APIResult:
        return self._get("/hf/search", params={"q": q, "limit": limit}, timeout=20)

    def hf_recommend(self, q: str, gpu_id: str = "0", limit: int = 20) -> APIResult:
        return self._get("/hf/recommend", params={"q": q, "gpu_id": gpu_id, "limit": limit}, timeout=20)

    def register_from_hf(self, repo_id: str, gpu: str = "0",
                         preset: str | None = None,
                         overrides: dict | None = None) -> APIResult:
        return self._post("/models/register_from_hf", body={
            "repo_id": repo_id, "preset": preset,
            "gpu": gpu, "overrides": overrides or {},
        }, timeout=30)

    # ── LiteLLM ──────────────────────────────────────────────────────

    def litellm_config(self) -> APIResult:
        return self._get("/litellm/config", timeout=5)

    def litellm_reload(self) -> APIResult:
        return self._post("/litellm/reload", timeout=10)

    # ── Health ────────────────────────────────────────────────────────

    def health(self) -> APIResult:
        return self._get("/health", timeout=3)


mrm = MRMClient()
