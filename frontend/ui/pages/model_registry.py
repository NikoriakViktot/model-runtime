"""
ui/pages/model_registry.py
Model Registry — view, start, stop, and manage all registered models.
"""
from __future__ import annotations
import streamlit as st
from ui.services.mrm_client import mrm
from ui.components.tables import model_status_table
from ui.components.json_viewer import json_viewer
from ui.components.metrics import api_result_header
from ui.components.runtime_explain import runtime_summary_line, render_runtime_debug
from ui.utils.formatters import runtime_badge, fmt_ts


def render():
    st.title("📦 Model Registry")

    c_ref, c_gpu = st.columns([1, 5])
    with c_ref:
        if st.button("🔄 Refresh"):
            st.rerun()

    # GPU summary bar
    r_gpu = mrm.gpu_metrics()
    if r_gpu.ok:
        gd = r_gpu.data or {}
        gpus = gd.get("gpus") or ([gd] if gd else [])
        if gpus:
            g = gpus[0]
            total = g.get("memory_total_mib", 0)
            used  = g.get("memory_used_mib", 0)
            free  = total - used
            util  = g.get("utilization_percent", 0)
            with c_gpu:
                mc = st.columns(4)
                mc[0].metric("GPU util",    f"{util}%")
                mc[1].metric("VRAM total",  f"{total/1024:.1f} GiB")
                mc[2].metric("VRAM used",   f"{used/1024:.1f} GiB")
                mc[3].metric("VRAM free",   f"{free/1024:.1f} GiB")
                st.progress(used / total if total else 0,
                            text=f"{used/1024:.1f} / {total/1024:.1f} GiB")

    st.divider()

    r = mrm.status_all()
    api_result_header(r, service="MRM /models/status")

    if not r.ok:
        st.error(r.error)
        return

    models: list[dict] = r.data if isinstance(r.data, list) else []
    if not models:
        st.info("No models registered. Use **HF Hub** to register one.")
        return

    # Flat table overview
    model_status_table(models)
    st.divider()

    # Detailed cards
    for m in models:
        base    = m.get("base_model", "?")
        alias   = m.get("model_alias", "")
        state   = m.get("state", "ABSENT")
        running = m.get("running", False)
        gpu_id  = m.get("gpu", "")
        loras   = m.get("active_loras", [])
        rt      = m.get("runtime_type", "gpu")

        icon = {"READY": "🟢", "STARTING": "🟡", "STOPPING": "🟡",
                "STOPPED": "⚫", "ABSENT": "⚫"}.get(state, "⚪")

        with st.expander(
            f"{icon} **{base}**  `{alias}`  ·  {state}  ·  {runtime_badge(rt)}"
        ):
            c1, c2 = st.columns([3, 1])
            with c1:
                meta_cols = st.columns(3)
                meta_cols[0].caption(f"**GPU:** `{gpu_id or '—'}`")
                meta_cols[1].caption(f"**LoRAs:** {len(loras)}")
                meta_cols[2].caption(f"**Last used:** {fmt_ts(m.get('last_used'))}")
                st.caption(f"API base: `{m.get('api_base', '—')}`")
                if loras:
                    st.caption("LoRAs: " + ", ".join(f"`{l}`" for l in loras))

                # Runtime summary (shown when backend selection metadata is available)
                summary = runtime_summary_line(m)
                if summary:
                    st.caption(summary)
                    with st.expander("🔍 Runtime Selection Debug", expanded=False):
                        render_runtime_debug(m.get("debug"))

            with c2:
                if not running:
                    if st.button("▶ Start", key=f"reg_start_{base}", type="primary"):
                        with st.spinner(f"Starting {alias}…"):
                            res = mrm.ensure(base)
                        if res.ok and isinstance(res.data, dict) and res.data.get("state") == "READY":
                            st.success("Ready!")
                            st.rerun()
                        else:
                            st.error(res.error or f"Unexpected state: {res.data}")
                else:
                    if st.button("⏹ Stop", key=f"reg_stop_{base}"):
                        with st.spinner("Stopping…"):
                            res = mrm.stop(base)
                        if res.ok:
                            st.success("Stopped")
                            st.rerun()
                        else:
                            st.error(res.error)

                if st.button("🗑 Remove", key=f"reg_rem_{base}"):
                    with st.spinner("Removing…"):
                        res = mrm.remove(base)
                    if res.ok:
                        st.success("Removed")
                        st.rerun()
                    else:
                        st.error(res.error)
