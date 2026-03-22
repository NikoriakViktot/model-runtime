import requests


class MRMApi:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    @staticmethod
    def _raise(r: requests.Response) -> None:
        try:
            r.raise_for_status()
        except Exception:
            # намагаємось дістати json
            body = None
            try:
                body = r.json()
            except Exception:
                body = r.text

            raise RuntimeError(
                {
                    "status_code": r.status_code,
                    "url": r.url,
                    "body": body,
                }
            )

    def status_all(self):
        r = requests.get(f"{self.base_url}/models/status", timeout=20)
        self._raise(r)
        return r.json()

    def status(self, base_model: str):
        r = requests.get(f"{self.base_url}/models/status/{base_model}", timeout=30)
        self._raise(r)
        return r.json()

    def ensure(self, base_model: str):
        r = requests.post(
            f"{self.base_url}/models/ensure",
            json={"base_model": base_model},
            timeout=300,
        )
        self._raise(r)
        return r.json()

    def remove(self, base_model: str):
        r = requests.post(
            f"{self.base_url}/models/remove",
            json={"base_model": base_model},
            timeout=120,
        )
        self._raise(r)
        return r.json()

    def stop(self, base_model: str):
        r = requests.post(
            f"{self.base_url}/models/stop",
            json={"base_model": base_model},
            timeout=120,
        )
        self._raise(r)
        return r.json()

    def hf_search(self, q: str, limit: int = 10):
        r = requests.get(
            f"{self.base_url}/hf/search",
            params={"q": q, "limit": limit},
            timeout=30,
        )
        self._raise(r)
        return r.json()

    def hf_model_info(self, repo_id: str):
        r = requests.get(
            f"{self.base_url}/hf/model_info",
            params={"repo_id": repo_id},
            timeout=30,
        )
        self._raise(r)
        return r.json()

    def register_from_hf(self, repo_id: str, preset: str, gpu: str, overrides: dict):
        payload = {"repo_id": repo_id, "preset": preset, "gpu": gpu, "overrides": overrides}
        r = requests.post(
            f"{self.base_url}/models/register_from_hf",
            json=payload,
            timeout=30,
        )
        self._raise(r)
        return r.json()

    def gpu_metrics(self):
        r = requests.get(f"{self.base_url}/gpu/metrics", timeout=10)
        self._raise(r)
        return r.json()

    def litellm_materialize(self):
        r = requests.post(f"{self.base_url}/litellm/materialize", timeout=30)
        self._raise(r)
        return r.json()

    def provision(self, repo_id: str, preset: str, gpu: str, overrides: dict):
        payload = {"repo_id": repo_id, "preset": preset, "gpu": gpu, "overrides": overrides}
        r = requests.post(f"{self.base_url}/factory/provision", json=payload, timeout=600)
        self._raise(r)
        return r.json()

