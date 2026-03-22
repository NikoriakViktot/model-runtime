from __future__ import annotations

from typing import Any
import requests

class Http:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def get(self, url: str, *, params: dict | None = None, timeout: int | None = None) -> Any:
        r = requests.get(url, params=params, timeout=timeout or self.timeout)
        r.raise_for_status()
        ct = (r.headers.get("content-type") or "").lower()
        if "application/json" in ct:
            return r.json()
        return r.text

    def post(self, url: str, *, json_body: dict | None = None, files=None, timeout: int | None = None) -> Any:
        r = requests.post(url, json=json_body, files=files, timeout=timeout or self.timeout)
        r.raise_for_status()
        ct = (r.headers.get("content-type") or "").lower()
        if "application/json" in ct:
            return r.json()
        return r.text

    def delete(self, url: str, *, timeout: int | None = None) -> bool:
        r = requests.delete(url, timeout=timeout or self.timeout)
        r.raise_for_status()
        return True
