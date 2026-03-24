"""
ui/pages/monitoring.py

Observability dashboard — Prometheus charts, SLO status, per-service health,
request rates, latency percentiles, error rates, GPU utilisation over time.
"""
from __future__ import annotations

import streamlit as st

from ui.services.prometheus_client import prom
from ui.services.gateway_client import gateway
from ui.services.mrm_client import mrm
from ui.services.scheduler_client import scheduler
from ui.components.metrics import api_result_header

_RANGE_OPTIONS = {"5 min": 5, "15 min": 15, "30 min": 30, "1 hour": 60, "3 hours": 180}
_STEP_MAP = {5: "10s", 15: "30s", 30: "30s", 60: "60s", 180: "120s"}


def _prom_chart(title: str, query: str, minutes: int, color: str | None = None) -> None:
    df = prom.query_range(query, minutes=minutes, step=_STEP_MAP.get(minutes, "30s"))
    st.caption(title)
    if df.empty:
        st.info("No data (Prometheus unreachable or metric absent)")
    else:
        st.line_chart(df)


def _scalar(query: str, fmt: str = "{:.2f}") -> str:
    v = prom.query(query)
    if v is None:
        return "—"
    return fmt.format(v)


def render():
    st.title("📊 Monitoring")
    st.caption("Live Prometheus metrics — request rates, latencies, errors, GPU utilisation.")

    # ── Controls ──────────────────────────────────────────────────────
    col_range, col_refresh = st.columns([2, 1])
    with col_range:
        range_label = st.selectbox("Time range", list(_RANGE_OPTIONS.keys()),
                                   index=2, key="mon_range")
    minutes = _RANGE_OPTIONS[range_label]
    with col_refresh:
        st.write("")
        if st.button("🔄 Refresh", key="mon_refresh"):
            st.rerun()

    prom_up = prom.is_up()
    if not prom_up:
        st.warning("⚠️ Prometheus is not reachable — charts will be empty. Showing live API data only.")

    st.divider()

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
                st.error(f"❌ {name}")

    st.divider()

    # ── SLO snapshot ─────────────────────────────────────────────────
    st.subheader("SLO snapshot")

    r_slo = gateway.slo()
    if r_slo.ok and isinstance(r_slo.data, dict):
        slo = r_slo.data
        p50  = slo.get("p50_ms", 0)
        p95  = slo.get("p95_ms", 0)
        p99  = slo.get("p99_ms", 0)
        err  = slo.get("error_rate", 0) * 100

        slo_cols = st.columns(4)
        slo_cols[0].metric("p50 latency",  f"{p50:.0f} ms")
        slo_cols[1].metric("p95 latency",  f"{p95:.0f} ms",
                            delta=f"{'⚠' if p95 > 2000 else '✅'} {'above' if p95 > 2000 else 'within'} SLO")
        slo_cols[2].metric("p99 latency",  f"{p99:.0f} ms")
        slo_cols[3].metric("Error rate",   f"{err:.2f}%",
                            delta=f"{'⚠' if err > 1 else '✅'} {'above' if err > 1 else 'within'} SLO")
    else:
        st.info("SLO data unavailable (Gateway not reachable).")

    st.divider()

    # ── Prometheus charts ─────────────────────────────────────────────
    tab_req, tab_lat, tab_err, tab_gpu, tab_raw = st.tabs(
        ["📈 Requests", "⏱ Latency", "🚨 Errors", "⚡ GPU", "🔢 Raw metrics"]
    )

    # ── Request rate ─────────────────────────────────────────────────
    with tab_req:
        st.subheader("Request rate")

        rc = st.columns(3)
        rc[0].metric(
            "Total requests",
            _scalar("sum(gateway_requests_total)", "{:.0f}"),
        )
        rc[1].metric(
            "Req/sec (1m avg)",
            _scalar("sum(rate(gateway_requests_total[1m]))", "{:.2f}"),
        )
        rc[2].metric(
            "In-flight",
            _scalar("gateway_requests_in_flight", "{:.0f}"),
        )

        st.divider()
        _prom_chart(
            "Requests/sec by model",
            "sum by (model) (rate(gateway_requests_total[1m]))",
            minutes,
        )
        _prom_chart(
            "Requests/sec by runtime type",
            "sum by (runtime_type) (rate(gateway_requests_total[1m]))",
            minutes,
        )
        _prom_chart(
            "Requests/sec by status code",
            "sum by (status) (rate(gateway_requests_total[1m]))",
            minutes,
        )

    # ── Latency ──────────────────────────────────────────────────────
    with tab_lat:
        st.subheader("Request latency")

        lc = st.columns(3)
        lc[0].metric(
            "p50 (1m)",
            _scalar(
                "histogram_quantile(0.50, sum by (le) (rate(gateway_request_latency_seconds_bucket[1m])))",
                "{:.0f} ms" if False else "{:.3f} s",
            ),
        )
        lc[1].metric(
            "p95 (1m)",
            _scalar(
                "histogram_quantile(0.95, sum by (le) (rate(gateway_request_latency_seconds_bucket[1m])))",
                "{:.3f} s",
            ),
        )
        lc[2].metric(
            "p99 (1m)",
            _scalar(
                "histogram_quantile(0.99, sum by (le) (rate(gateway_request_latency_seconds_bucket[1m])))",
                "{:.3f} s",
            ),
        )

        st.divider()
        _prom_chart(
            "p95 latency over time (all models)",
            "histogram_quantile(0.95, sum by (le) (rate(gateway_request_latency_seconds_bucket[2m])))",
            minutes,
        )
        _prom_chart(
            "p95 latency by runtime type",
            "histogram_quantile(0.95, sum by (le, runtime_type) (rate(gateway_request_latency_seconds_bucket[2m])))",
            minutes,
        )

    # ── Errors ───────────────────────────────────────────────────────
    with tab_err:
        st.subheader("Errors")

        ec = st.columns(2)
        ec[0].metric(
            "Total errors",
            _scalar("sum(gateway_errors_total)", "{:.0f}"),
        )
        ec[1].metric(
            "Error rate (1m)",
            _scalar(
                "sum(rate(gateway_errors_total[1m])) / sum(rate(gateway_requests_total[1m]))",
                "{:.2%}",
            ),
        )

        st.divider()
        _prom_chart(
            "Errors/sec by type",
            "sum by (error_type) (rate(gateway_errors_total[1m]))",
            minutes,
        )
        _prom_chart(
            "Error rate % over time",
            "100 * sum(rate(gateway_errors_total[1m])) / sum(rate(gateway_requests_total[1m]))",
            minutes,
        )

    # ── GPU ──────────────────────────────────────────────────────────
    with tab_gpu:
        st.subheader("GPU utilisation")

        # Live from MRM API
        r_gpu = mrm.gpu_metrics()
        if r_gpu.ok:
            gpu_data = r_gpu.data or {}
            gpus = gpu_data.get("gpus") or ([gpu_data] if gpu_data else [])
            if gpus:
                gc = st.columns(len(gpus) if len(gpus) <= 4 else 4)
                for i, g in enumerate(gpus[:4]):
                    total = g.get("memory_total_mib", 0)
                    used  = g.get("memory_used_mib", 0)
                    util  = g.get("utilization_percent", 0)
                    name  = g.get("name", f"GPU {i}")
                    with gc[i]:
                        st.metric(f"⚡ {name} util", f"{util}%")
                        pct = used / total if total else 0
                        st.progress(pct, text=f"{used/1024:.1f}/{total/1024:.1f} GiB")

        st.divider()
        _prom_chart(
            "GPU utilisation % over time",
            "gpu_utilization_percent",
            minutes,
        )
        _prom_chart(
            "VRAM used (MiB) over time",
            "gpu_memory_used_mib",
            minutes,
        )

    # ── Raw metrics ───────────────────────────────────────────────────
    with tab_raw:
        st.subheader("Custom PromQL query")
        custom_q = st.text_input(
            "PromQL",
            placeholder="rate(gateway_requests_total[1m])",
            key="mon_custom_q",
        )
        custom_range = st.selectbox("Range", list(_RANGE_OPTIONS.keys()),
                                    index=2, key="mon_custom_range")
        if st.button("▶ Run query", key="mon_custom_run") and custom_q.strip():
            m = _RANGE_OPTIONS[custom_range]
            df = prom.query_range(custom_q, minutes=m, step=_STEP_MAP.get(m, "30s"))
            if df.empty:
                st.info("No data returned.")
            else:
                st.line_chart(df)
                with st.expander("Raw DataFrame"):
                    st.dataframe(df, use_container_width=True)

        st.divider()
        st.subheader("Instant query")
        instant_q = st.text_input(
            "PromQL (instant)",
            placeholder="sum(gateway_requests_total)",
            key="mon_instant_q",
        )
        if st.button("▶ Run instant", key="mon_instant_run") and instant_q.strip():
            val = prom.query(instant_q)
            if val is None:
                st.info("No result or Prometheus unreachable.")
            else:
                st.metric("Result", f"{val:.4f}")
