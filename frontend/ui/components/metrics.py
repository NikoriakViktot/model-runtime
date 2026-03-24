"""
ui/components/metrics.py
Reusable metric display components.
"""
from __future__ import annotations
import streamlit as st
from ui.utils.formatters import fmt_latency, runtime_badge


def latency_badge(ms: float, *, label: str = "Latency") -> None:
    color = "🟢" if ms < 500 else ("🟡" if ms < 2000 else "🔴")
    st.metric(label, f"{color} {fmt_latency(ms)}")


def cost_card(result: dict) -> None:
    """Display cost + perf metrics from utils.cost.estimate_*()."""
    cols = st.columns(4)
    cols[0].metric("Runtime", runtime_badge(result.get("runtime", "?")))
    cols[1].metric("Latency", fmt_latency(result.get("latency_ms", 0)))
    cols[2].metric("Tokens/s", f"{result.get('tokens_per_sec', 0):.1f}")
    cols[3].metric("Est. cost", f"${result.get('cost_usd', 0):.5f}")


def slo_badge(label: str, value: float | None,
              warn: float, crit: float, unit: str = "") -> None:
    if value is None:
        st.metric(label, "N/A")
        return
    icon = "🟢" if value < warn else ("🟡" if value < crit else "🔴")
    st.metric(label, f"{icon} {value:.1f}{unit}")


def api_result_header(result, *, service: str = "") -> None:
    """Display status code + latency after any API call."""
    if result is None:
        return
    code = result.status_code or "—"
    icon = "✅" if result.ok else "❌"
    prefix = f"**{service}**  " if service else ""
    st.caption(
        f"{prefix}{icon} HTTP {code}  ·  "
        f"⏱ {fmt_latency(result.latency_ms)}  ·  "
        f"`{result.method} {result.url}`"
    )
