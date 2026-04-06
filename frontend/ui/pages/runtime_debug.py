"""
ui/pages/runtime_debug.py

Runtime Debug console:
  - Live MRM state (Redis truth vs Docker)
  - Scheduler nodes + placements
  - CPU runtime instances
  - Routing decision log (last N requests)
  - Per-instance router metrics
"""
from __future__ import annotations

import streamlit as st

from ui.services.mrm_client import mrm
from ui.services.scheduler_client import scheduler
from ui.services.gateway_client import gateway
from ui.components.tables import model_status_table, nodes_table, placements_table
from ui.components.json_viewer import json_viewer
from ui.components.metrics import api_result_header
from ui.utils.formatters import state_badge, runtime_badge, fmt_ts


def render():
    st.title("🧠 Runtime Debug")
    st.caption("Live system state — MRM Redis truth, Scheduler placements, routing decisions.")

    col_refresh, col_auto = st.columns([1, 3])
    with col_refresh:
        if st.button("🔄 Refresh all", type="primary"):
            st.rerun()

    # ── Service health row ────────────────────────────────────────────
    st.subheader("Service health")
    hcols = st.columns(4)
    services = [
        ("Gateway",   gateway.health()),
        ("MRM",       mrm.health()),
        ("Scheduler", scheduler.health()),
    ]
    for i, (name, r) in enumerate(services):
        with hcols[i]:
            if r.ok:
                st.success(f"✅ {name}  ({r.latency_ms:.0f} ms)")
            else:
                st.error(f"❌ {name}  —  {r.error or 'unreachable'}")

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────
    tab_mrm, tab_sched, tab_routing, tab_gpu = st.tabs(
        ["📦 MRM state", "🗺 Scheduler", "🔀 Routing", "⚡ GPU metrics"]
    )

    # ── MRM state ─────────────────────────────────────────────────────
    with tab_mrm:
        st.subheader("Model states (Redis canonical view)")
        r = mrm.status_all()
        api_result_header(r, service="MRM /models/status")

        if not r.ok:
            st.error(r.error)
        else:
            models: list[dict] = r.data if isinstance(r.data, list) else []
            if not models:
                st.info("No models registered in MRM.")
            else:
                # Summary counters
                ccols = st.columns(4)
                ready  = sum(1 for m in models if m.get("state") == "READY")
                absent = sum(1 for m in models if m.get("state") == "ABSENT")
                other  = len(models) - ready - absent
                ccols[0].metric("Total", len(models))
                ccols[1].metric("🟢 Ready",  ready)
                ccols[2].metric("⚫ Absent", absent)
                ccols[3].metric("🟡 Other",  other)

                st.divider()
                model_status_table(models)

                st.subheader("Detail view")
                for m in models:
                    base  = m.get("base_model", "?")
                    state = m.get("state", "ABSENT")
                    rt    = m.get("runtime_type", "gpu")
                    icon  = {"READY": "🟢", "STARTING": "🟡", "STOPPING": "🟡",
                             "STOPPED": "⚫", "ABSENT": "⚫"}.get(state, "⚪")

                    with st.expander(
                        f"{icon} `{base}`  ·  {state}  ·  {runtime_badge(rt)}"
                    ):
                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.caption(f"Alias: `{m.get('model_alias','—')}`")
                            st.caption(f"Container: `{m.get('container','—')}`")
                            st.caption(f"API base: `{m.get('api_base','—')}`")
                            st.caption(f"GPU: `{m.get('gpu','—')}`")
                            st.caption(f"Last used: {fmt_ts(m.get('last_used'))}")
                            loras = m.get("active_loras", [])
                            if loras:
                                st.caption("LoRAs: " + ", ".join(f"`{l}`" for l in loras))
                        with c2:
                            json_viewer(m, label="Raw state", expanded=False)

                            if m.get("running"):
                                if st.button("⏹ Stop", key=f"dbg_stop_{base}"):
                                    with st.spinner("Stopping…"):
                                        res = mrm.stop(base)
                                    if res.ok:
                                        st.success("Stopped")
                                        st.rerun()
                                    else:
                                        st.error(res.error)
                            else:
                                if st.button("▶ Start", key=f"dbg_start_{base}"):
                                    with st.spinner("Starting… (may take minutes)"):
                                        res = mrm.ensure(base)
                                    if res.ok:
                                        st.success("Ready!")
                                        st.rerun()
                                    else:
                                        st.error(res.error)

    # ── Scheduler ─────────────────────────────────────────────────────
    with tab_sched:
        col_n, col_p = st.columns(2)

        with col_n:
            st.subheader("Registered nodes")
            r_nodes = scheduler.list_nodes()
            api_result_header(r_nodes, service="Scheduler /nodes")

            if not r_nodes.ok:
                st.error(r_nodes.error)
            else:
                nodes = r_nodes.data if isinstance(r_nodes.data, list) else []
                st.caption(f"{len(nodes)} node(s) alive")
                nodes_table(nodes)

                for node in nodes:
                    with st.expander(f"🖥 `{node.get('node_id')}`  —  {node.get('hostname','?')}"):
                        st.json(node)

        with col_p:
            st.subheader("Active placements")
            r_pl = scheduler.list_placements()
            api_result_header(r_pl, service="Scheduler /placements")

            if not r_pl.ok:
                st.error(r_pl.error)
            else:
                placements = r_pl.data if isinstance(r_pl.data, list) else []
                st.caption(f"{len(placements)} placement(s)")
                placements_table(placements)

                # CPU vs GPU breakdown
                cpu_count = sum(
                    1 for p in placements
                    for inst in p.get("instances", [])
                    if inst.get("runtime_type") == "cpu"
                )
                gpu_count = sum(
                    1 for p in placements
                    for inst in p.get("instances", [])
                    if inst.get("runtime_type") != "cpu"
                )
                if placements:
                    bc = st.columns(2)
                    bc[0].metric("⚡ GPU instances", gpu_count)
                    bc[1].metric("🖥 CPU instances", cpu_count)

    # ── Routing decisions ─────────────────────────────────────────────
    with tab_routing:
        st.subheader("Router metrics (gateway in-process)")
        r_router = gateway.router_metrics()
        api_result_header(r_router, service="Gateway /v1/router/metrics")

        if not r_router.ok:
            st.info("Router metrics endpoint not available or no traffic yet.")
        else:
            json_viewer(r_router.data, label="Router metrics", expanded=True)

        st.divider()
        st.subheader("SLO snapshot")
        r_slo = gateway.slo()
        api_result_header(r_slo, service="Gateway /v1/slo")

        if r_slo.ok and isinstance(r_slo.data, dict):
            slo = r_slo.data
            scols = st.columns(4)
            scols[0].metric("p50 latency",  f"{slo.get('p50_ms') or 0:.0f} ms")
            scols[1].metric("p95 latency",  f"{slo.get('p95_ms') or 0:.0f} ms")
            scols[2].metric("p99 latency",  f"{slo.get('p99_ms') or 0:.0f} ms")
            scols[3].metric("Error rate",   f"{(slo.get('error_rate') or 0) * 100:.2f}%")
        else:
            st.info("SLO data unavailable.")

    # ── GPU metrics ───────────────────────────────────────────────────
    with tab_gpu:
        st.subheader("GPU memory & utilisation")
        r_gpu = mrm.gpu_metrics()
        api_result_header(r_gpu, service="MRM /gpu/metrics")

        if not r_gpu.ok:
            st.error(r_gpu.error)
        else:
            gpu_data = r_gpu.data or {}
            gpus = gpu_data.get("gpus") or ([gpu_data] if gpu_data else [])

            if not gpus:
                st.info("No GPU data returned.")
            else:
                for g in gpus:
                    idx   = g.get("index", g.get("gpu_index", "?"))
                    total = g.get("memory_total_mib", 0)
                    used  = g.get("memory_used_mib", 0)
                    free  = total - used
                    util  = g.get("utilization_percent", 0)
                    temp  = g.get("temperature_c", "?")
                    name  = g.get("name", f"GPU {idx}")

                    st.subheader(f"⚡ {name}")
                    gc = st.columns(5)
                    gc[0].metric("Util",       f"{util}%")
                    gc[1].metric("VRAM total", f"{total/1024:.1f} GiB")
                    gc[2].metric("VRAM used",  f"{used/1024:.1f} GiB")
                    gc[3].metric("VRAM free",  f"{free/1024:.1f} GiB")
                    gc[4].metric("Temp",       f"{temp} °C")

                    pct = used / total if total else 0
                    st.progress(pct, text=f"{pct*100:.1f}% VRAM used")
