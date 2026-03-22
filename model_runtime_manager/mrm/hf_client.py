# hf_client.py
import requests
from typing import Any, Dict, List, Optional

HF_API = "https://huggingface.co/api"

class HFClient:
    def __init__(self, token: str = ""):
        self.token = token

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def search_models(self, q: str, limit: int = 10) -> List[Dict[str, Any]]:
        # /api/models?search=... returns list
        r = requests.get(
            f"{HF_API}/models",
            params={"search": q, "limit": limit, "full": "true"},
            headers=self._headers(),
            timeout=15,
        )
        r.raise_for_status()
        return r.json()

    def model_info(self, repo_id: str) -> Dict[str, Any]:
        r = requests.get(
            f"{HF_API}/models/{repo_id}",
            params={"full": "true"},
            headers=self._headers(),
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
