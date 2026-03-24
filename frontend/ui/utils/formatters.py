"""
ui/utils/formatters.py
Display-formatting helpers — pure functions, no Streamlit calls.
"""
from __future__ import annotations
import json
from datetime import datetime


def fmt_latency(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms/1000:.2f} s"


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n/1024:.1f} KB"
    return f"{n/1024**2:.2f} MB"


def fmt_ts(ts: str | float | None) -> str:
    if not ts:
        return "—"
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        return str(ts)[:19].replace("T", " ")
    except Exception:
        return str(ts)


def state_badge(state: str) -> str:
    icons = {
        "READY": "🟢", "STARTING": "🟡", "STOPPING": "🟡",
        "STOPPED": "⚫", "ABSENT": "⚫",
        "DONE": "✅", "FAILED": "❌",
        "healthy": "🟢", "stale": "🟡", "dead": "🔴",
    }
    return icons.get(state, "⚪") + f" {state}"


def runtime_badge(rt: str) -> str:
    return {"gpu": "⚡ GPU", "cpu": "🖥 CPU"}.get(rt, rt)


def pretty_json(obj) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def truncate(s: str, n: int = 80) -> str:
    return s if len(s) <= n else s[:n] + "…"
