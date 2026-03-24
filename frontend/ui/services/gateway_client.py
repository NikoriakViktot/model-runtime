"""
ui/services/gateway_client.py
Typed client for the AI Runtime Gateway (port 8080).
"""
from __future__ import annotations
import os
from .base import APIResult, BaseClient

_BASE = os.getenv("GATEWAY_URL", "http://gateway:8080")


class GatewayClient(BaseClient):
    def __init__(self, base_url: str = _BASE):
        super().__init__(base_url, default_timeout=15)

    # ── Inference ──────────────────────────────────────────────────────

    def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
        stream: bool = False,
        runtime_preference: str = "auto",
    ) -> APIResult:
        return self._post(
            "/v1/chat/completions",
            body={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
                "runtime_preference": runtime_preference,
            },
            timeout=180,
        )

    # ── Models ────────────────────────────────────────────────────────

    def list_models(self) -> APIResult:
        return self._get("/v1/models", timeout=5)

    # ── Health ────────────────────────────────────────────────────────

    def health(self) -> APIResult:
        return self._get("/health", timeout=3)

    def ready(self) -> APIResult:
        return self._get("/ready", timeout=5)

    # ── Metrics / SLO ─────────────────────────────────────────────────

    def slo(self) -> APIResult:
        return self._get("/v1/slo", timeout=5)

    def router_metrics(self) -> APIResult:
        return self._get("/v1/router/metrics", timeout=5)

    # ── Admin ─────────────────────────────────────────────────────────

    def admin_status(self) -> APIResult:
        return self._get("/admin/status", timeout=5)


# Module-level singleton
gateway = GatewayClient()
