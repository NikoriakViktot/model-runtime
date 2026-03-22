from __future__ import annotations

from typing import Any
from .config import CP_URL
from .http import Http

class ControlPlaneAPI:
    def __init__(self, http: Http):
        self.http = http

    def runs(self) -> list[dict]:
        data = self.http.get(f"{CP_URL}/runs", timeout=10)
        return data if isinstance(data, list) else []

    def run(self, run_id: str) -> dict | None:
        data = self.http.get(f"{CP_URL}/runs/{run_id}", timeout=15)
        return data if isinstance(data, dict) else None

    def post_contract(self, contract_type: str, payload: dict) -> dict | None:
        envelope = {"type": contract_type, "spec_version": "v1", "payload": payload}
        data = self.http.post(f"{CP_URL}/contracts", json_body=envelope, timeout=20)
        return data if isinstance(data, dict) else None

    def prompts(self) -> list[dict]:
        data = self.http.get(f"{CP_URL}/prompts/", timeout=15)
        return data if isinstance(data, list) else []

    def prompt_versions(self, prompt_id: str) -> list[dict]:
        data = self.http.get(f"{CP_URL}/prompts/{prompt_id}/versions", timeout=15)
        return data if isinstance(data, list) else []

    def create_prompt_family(self, prompt_id: str, description: str) -> dict | None:
        data = self.http.post(f"{CP_URL}/prompts/", json_body=None, timeout=10)
        _ = data
        data2 = self.http.post(f"{CP_URL}/prompts/", json_body=None, timeout=10)
        _ = data2
        import requests
        r = requests.post(f"{CP_URL}/prompts/", params={"id": prompt_id, "description": description}, timeout=10)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {"status": "ok"}

    def save_prompt_version(self, prompt_id: str, version_tag: str, template: str, commit_message: str) -> dict | None:
        payload = {
            "prompt_id": prompt_id,
            "version_tag": version_tag,
            "template": template,
            "commit_message": commit_message,
        }
        data = self.http.post(f"{CP_URL}/prompts/version", json_body=payload, timeout=30)
        return data if isinstance(data, dict) else None
