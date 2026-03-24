"""
ui/services/scheduler_client.py
Typed client for the Distributed Inference Scheduler (port 8030).
"""
from __future__ import annotations
import os
from .base import APIResult, BaseClient

_BASE = os.getenv("SCHEDULER_URL", "http://scheduler:8030")


class SchedulerClient(BaseClient):
    def __init__(self, base_url: str = _BASE):
        super().__init__(base_url, default_timeout=10)

    def ensure(self, model: str, runtime_preference: str = "auto") -> APIResult:
        return self._post("/schedule/ensure", body={
            "model": model,
            "runtime_preference": runtime_preference,
        }, timeout=120)

    def stop(self, model: str) -> APIResult:
        return self._post("/schedule/stop", body={"model": model}, timeout=30)

    def list_nodes(self) -> APIResult:
        return self._get("/nodes", timeout=5)

    def list_placements(self) -> APIResult:
        return self._get("/placements", timeout=5)

    def health(self) -> APIResult:
        return self._get("/health", timeout=3)


scheduler = SchedulerClient()
