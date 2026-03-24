"""
ui/services/control_plane_client.py
Typed client for the Control Plane (port 8004).
"""
from __future__ import annotations
import os
from .base import APIResult, BaseClient

_BASE = os.getenv("CONTROL_PLANE_URL", "http://control_plane:8004")


class ControlPlaneClient(BaseClient):
    def __init__(self, base_url: str = _BASE):
        super().__init__(base_url, default_timeout=10)

    # ── Contracts ─────────────────────────────────────────────────────

    def submit_contract(self, contract_type: str, payload: dict) -> APIResult:
        return self._post("/contracts", body={
            "type": contract_type,
            "spec_version": "v1",
            "payload": payload,
        }, timeout=15)

    # ── Runs ──────────────────────────────────────────────────────────

    def list_runs(self) -> APIResult:
        return self._get("/runs", timeout=5)

    def get_run(self, run_id: str) -> APIResult:
        return self._get(f"/runs/{run_id}", timeout=5)

    def list_events(self, run_id: str | None = None) -> APIResult:
        params = {"run_id": run_id} if run_id else None
        return self._get("/events", params=params, timeout=5)

    # ── Health ────────────────────────────────────────────────────────

    def health(self) -> APIResult:
        return self._get("/health", timeout=3)


cp = ControlPlaneClient()
