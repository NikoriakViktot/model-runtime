"""
tests/unit/test_readiness.py

Tests for the /health (liveness) and /ready (readiness) endpoints.

Health/readiness contract
-------------------------
/health — always 200.  Kubernetes uses this to decide whether to restart
          the container.  The process being alive is sufficient.
/ready  — 200 only when the model is fully loaded (load_state == "loaded").
          Kubernetes uses this to decide whether to send traffic.
          MRM's depends_on health-check also points here.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock


def _get_client():
    from cpu_runtime.app import app
    return TestClient(app, raise_server_exceptions=False)


class TestLivenessProbe:
    """/health must always return 200 regardless of model load state."""

    def _set_state(self, state: str):
        import cpu_runtime.inference as inf_mod
        inf_mod.load_state = state
        inf_mod.engine = None

    def test_health_200_when_not_started(self):
        self._set_state("not_started")
        resp = _get_client().get("/health")
        assert resp.status_code == 200

    def test_health_200_when_loading(self):
        self._set_state("loading")
        resp = _get_client().get("/health")
        assert resp.status_code == 200

    def test_health_200_when_loaded(self):
        import cpu_runtime.inference as inf_mod
        inf_mod.load_state = "loaded"
        inf_mod.engine = MagicMock()
        resp = _get_client().get("/health")
        assert resp.status_code == 200

    def test_health_200_when_failed(self):
        import cpu_runtime.inference as inf_mod
        inf_mod.load_state = "failed"
        inf_mod.load_error = "OOM"
        inf_mod.engine = None
        resp = _get_client().get("/health")
        assert resp.status_code == 200

    def test_health_200_when_not_found(self):
        import cpu_runtime.inference as inf_mod
        inf_mod.load_state = "not_found"
        inf_mod.load_error = "no such file"
        inf_mod.engine = None
        resp = _get_client().get("/health")
        assert resp.status_code == 200

    def test_health_body_has_status_ok(self):
        import cpu_runtime.inference as inf_mod
        inf_mod.load_state = "loading"
        inf_mod.engine = None
        resp = _get_client().get("/health")
        assert resp.json()["status"] == "ok"


class TestReadinessProbe:
    """/ready must return 200 only when load_state == "loaded"."""

    def test_ready_503_while_loading(self):
        import cpu_runtime.inference as inf_mod
        inf_mod.load_state = "loading"
        inf_mod.engine = None
        resp = _get_client().get("/ready")
        assert resp.status_code == 503
        assert resp.json()["status"] == "loading"

    def test_ready_503_when_not_started(self):
        import cpu_runtime.inference as inf_mod
        inf_mod.load_state = "not_started"
        inf_mod.engine = None
        resp = _get_client().get("/ready")
        assert resp.status_code == 503

    def test_ready_200_when_loaded(self):
        import cpu_runtime.inference as inf_mod
        inf_mod.load_state = "loaded"
        inf_mod.engine = MagicMock()
        resp = _get_client().get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert "model" in data

    def test_ready_503_when_failed(self):
        import cpu_runtime.inference as inf_mod
        inf_mod.load_state = "failed"
        inf_mod.load_error = "segfault in llama.cpp"
        inf_mod.engine = None
        resp = _get_client().get("/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "failed"
        assert "segfault" in data["error"]

    def test_ready_503_when_not_found(self):
        import cpu_runtime.inference as inf_mod
        inf_mod.load_state = "not_found"
        inf_mod.load_error = "/models/model.gguf not found"
        inf_mod.engine = None
        resp = _get_client().get("/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_found"
        assert "/models/model.gguf" in data["error"]

    def test_health_and_ready_differ_while_loading(self):
        """Core contract: health=200, ready=503 while model is loading."""
        import cpu_runtime.inference as inf_mod
        inf_mod.load_state = "loading"
        inf_mod.engine = None
        client = _get_client()
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
