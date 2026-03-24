"""
ui/services/prometheus_client.py
Prometheus query helpers — returns DataFrames for charts.
"""
from __future__ import annotations
import os
import time as _time

import pandas as pd
import requests

_BASE = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
_SCRAPE_LABELS = frozenset({"instance", "job", "service", "__name__"})


class PrometheusClient:
    def __init__(self, base_url: str = _BASE):
        self._base = base_url.rstrip("/")

    def query(self, q: str) -> float | None:
        try:
            r = requests.get(f"{self._base}/api/v1/query",
                             params={"query": q}, timeout=5)
            r.raise_for_status()
            result = r.json()["data"]["result"]
            if result:
                v = float(result[0]["value"][1])
                return None if v != v else v
        except Exception:
            pass
        return None

    def query_range(self, q: str, minutes: int = 30, step: str = "30s") -> pd.DataFrame:
        end = int(_time.time())
        start = end - minutes * 60
        try:
            r = requests.get(
                f"{self._base}/api/v1/query_range",
                params={"query": q, "start": start, "end": end, "step": step},
                timeout=10,
            )
            r.raise_for_status()
            results = r.json()["data"]["result"]
            rows = []
            for i, series in enumerate(results):
                labels = {k: v for k, v in series["metric"].items()
                          if k not in _SCRAPE_LABELS}
                lbl = (",".join(f"{k}={v.replace(':', '_')}" for k, v in labels.items())
                       if labels else f"series_{i}" if len(results) > 1 else "value")
                for ts, val in series["values"]:
                    rows.append({"time": pd.Timestamp(ts, unit="s"),
                                 "value": float(val), "series": lbl})
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)
            return df.pivot_table(index="time", columns="series",
                                  values="value", aggfunc="mean")
        except Exception:
            return pd.DataFrame()

    def is_up(self) -> bool:
        try:
            requests.get(f"{self._base}/-/healthy", timeout=2).raise_for_status()
            return True
        except Exception:
            return False


prom = PrometheusClient()
