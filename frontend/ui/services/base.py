"""
ui/services/base.py
Base HTTP client with structured result type.
Every service client inherits from BaseClient.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class APIResult:
    """Typed container for every API call made from the UI."""
    ok: bool
    status_code: int | None
    data: Any
    error: str | None
    latency_ms: float
    url: str
    method: str
    request_body: Any = None

    @property
    def is_error(self) -> bool:
        return not self.ok


class BaseClient:
    def __init__(self, base_url: str, default_timeout: int = 10):
        self._base = base_url.rstrip("/")
        self._timeout = default_timeout

    def _get(self, path: str, *, params: dict | None = None, timeout: int | None = None) -> APIResult:
        url = f"{self._base}{path}"
        t0 = time.perf_counter()
        try:
            r = requests.get(url, params=params, timeout=timeout or self._timeout)
            ms = (time.perf_counter() - t0) * 1000
            try:
                data = r.json()
            except Exception:
                data = r.text
            return APIResult(
                ok=r.ok, status_code=r.status_code,
                data=data, error=None if r.ok else f"HTTP {r.status_code}: {r.text[:300]}",
                latency_ms=round(ms, 1), url=url, method="GET",
            )
        except requests.exceptions.ConnectionError as e:
            ms = (time.perf_counter() - t0) * 1000
            return APIResult(ok=False, status_code=None, data=None,
                             error=f"Connection error: {e}", latency_ms=round(ms, 1),
                             url=url, method="GET")
        except requests.exceptions.Timeout:
            ms = (time.perf_counter() - t0) * 1000
            return APIResult(ok=False, status_code=None, data=None,
                             error="Request timed out", latency_ms=round(ms, 1),
                             url=url, method="GET")
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            return APIResult(ok=False, status_code=None, data=None,
                             error=str(e), latency_ms=round(ms, 1),
                             url=url, method="GET")

    def _post(self, path: str, *, body: Any = None, timeout: int | None = None) -> APIResult:
        url = f"{self._base}{path}"
        t0 = time.perf_counter()
        try:
            r = requests.post(url, json=body, timeout=timeout or self._timeout)
            ms = (time.perf_counter() - t0) * 1000
            try:
                data = r.json()
            except Exception:
                data = r.text
            return APIResult(
                ok=r.ok, status_code=r.status_code,
                data=data, error=None if r.ok else f"HTTP {r.status_code}: {r.text[:300]}",
                latency_ms=round(ms, 1), url=url, method="POST", request_body=body,
            )
        except requests.exceptions.ConnectionError as e:
            ms = (time.perf_counter() - t0) * 1000
            return APIResult(ok=False, status_code=None, data=None,
                             error=f"Connection error: {e}", latency_ms=round(ms, 1),
                             url=url, method="POST", request_body=body)
        except requests.exceptions.Timeout:
            ms = (time.perf_counter() - t0) * 1000
            return APIResult(ok=False, status_code=None, data=None,
                             error="Request timed out", latency_ms=round(ms, 1),
                             url=url, method="POST", request_body=body)
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            return APIResult(ok=False, status_code=None, data=None,
                             error=str(e), latency_ms=round(ms, 1),
                             url=url, method="POST", request_body=body)

    def _delete(self, path: str, timeout: int | None = None) -> APIResult:
        url = f"{self._base}{path}"
        t0 = time.perf_counter()
        try:
            r = requests.delete(url, timeout=timeout or self._timeout)
            ms = (time.perf_counter() - t0) * 1000
            try:
                data = r.json()
            except Exception:
                data = r.text
            return APIResult(
                ok=r.ok, status_code=r.status_code,
                data=data, error=None if r.ok else f"HTTP {r.status_code}",
                latency_ms=round(ms, 1), url=url, method="DELETE",
            )
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            return APIResult(ok=False, status_code=None, data=None,
                             error=str(e), latency_ms=round(ms, 1),
                             url=url, method="DELETE")
