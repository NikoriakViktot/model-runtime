"""
ui/components/tables.py
Standardised DataFrame and dict-list renderers.
"""
from __future__ import annotations
import pandas as pd
import streamlit as st
from ui.utils.formatters import state_badge, runtime_badge, fmt_ts


def model_status_table(models: list[dict]) -> None:
    if not models:
        st.info("No models found.")
        return
    rows = []
    for m in models:
        state = m.get("state", "ABSENT")
        rows.append({
            "Model": m.get("base_model", "?"),
            "Alias": m.get("model_alias", ""),
            "State": state_badge(state),
            "Runtime": runtime_badge(m.get("runtime_type", "gpu")),
            "GPU": m.get("gpu", "—"),
            "LoRAs": len(m.get("active_loras", [])),
            "Last used": fmt_ts(m.get("last_used")),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def nodes_table(nodes: list[dict]) -> None:
    if not nodes:
        st.info("No nodes registered.")
        return
    rows = []
    for n in nodes:
        rows.append({
            "Node ID": n.get("node_id", "?"),
            "Hostname": n.get("hostname", ""),
            "State": state_badge(n.get("state", "?")),
            "GPUs": n.get("gpu_count", 0),
            "CPU cores": n.get("cpu_cores", "—"),
            "Free VRAM": f"{n.get('total_free_mb', 0)/1024:.1f} GiB",
            "Models": ", ".join(n.get("models", [])) or "—",
            "Last heartbeat": fmt_ts(n.get("last_heartbeat")),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def runs_table(runs: list[dict], *, on_select=None) -> None:
    if not runs:
        st.info("No runs found.")
        return
    rows = []
    for r in runs:
        rows.append({
            "ID": r.get("id", "")[:8],
            "Type": r.get("type", "?"),
            "State": state_badge(r.get("state", "?")),
            "Created": fmt_ts(r.get("created_at")),
            "Updated": fmt_ts(r.get("updated_at")),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def placements_table(placements: list[dict]) -> None:
    if not placements:
        st.info("No active placements.")
        return
    rows = []
    for p in placements:
        for inst in p.get("instances", [p]):
            rows.append({
                "Model": p.get("model_id", "?"),
                "Runtime": runtime_badge(inst.get("runtime_type", "gpu")),
                "Node": inst.get("node_id", "?"),
                "GPU": inst.get("gpu", "—"),
                "API base": inst.get("api_base", ""),
                "State": inst.get("state", ""),
                "Placed": fmt_ts(inst.get("placed_at")),
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
