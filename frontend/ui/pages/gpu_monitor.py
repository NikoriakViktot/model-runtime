import time
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from ..config import MRM_BASE_URL
from ..mrm_api import MRMApi


def _fmt_mib(x) -> str:
    try:
        return f"{float(x):,.0f} MiB"
    except Exception:
        return "-"


def _build_proc_table(g: dict) -> pd.DataFrame:
    procs = (g.get("compute_procs") or []) + (g.get("graphics_procs") or [])
    rows = []
    for p in procs:
        used = p.get("used_mib") or 0.0
        docker_name = (p.get("docker_name") or "").strip()
        docker_id = (p.get("docker_id") or "")[:12]
        rows.append(
            {
                "Container": docker_name if docker_name else "unknown",
                "VRAM": float(used),
                "PID": p.get("pid"),
                "Docker ID": docker_id,
                "Cmd": (p.get("cmdline") or "")[:140],
            }
        )

    if not rows:
        return pd.DataFrame(columns=["Container", "VRAM", "PID", "Docker ID", "Cmd"])

    dfp = pd.DataFrame(rows)
    dfp["VRAM"] = pd.to_numeric(dfp["VRAM"], errors="coerce").fillna(0.0)
    dfp = dfp.sort_values("VRAM", ascending=False)
    return dfp


def _pie_by_container(dfp: pd.DataFrame, g: dict) -> None:
    total = float(g.get("mem_total_mib") or 0.0)
    used = float(g.get("mem_used_mib") or 0.0)

    agg = (
        dfp.groupby("Container", as_index=False)["VRAM"]
        .sum()
        .sort_values("VRAM", ascending=False)
    )

    sum_procs = float(agg["VRAM"].sum())
    other = max(0.0, used - sum_procs)
    if other > 1.0:
        agg = pd.concat(
            [agg, pd.DataFrame([{"Container": "other", "VRAM": other}])],
            ignore_index=True,
        )

    TOP = 6
    if len(agg) > TOP:
        top = agg.iloc[:TOP].copy()
        rest = float(agg.iloc[TOP:]["VRAM"].sum())
        top = pd.concat(
            [top, pd.DataFrame([{"Container": "rest", "VRAM": rest}])],
            ignore_index=True,
        )
        agg = top

    labels = [
        f"{row['Container']} ({row['VRAM'] / max(used, 1.0) * 100:.0f}%)"
        for _, row in agg.iterrows()
    ]

    fig, ax = plt.subplots()
    ax.pie(agg["VRAM"].values, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    ax.set_title(f"VRAM breakdown | used {int(used)}/{int(total)} MiB")
    st.pyplot(fig, clear_figure=True)


def render() -> None:
    st.title("🟩 GPU Monitor (NVML → MRM)")
    api = MRMApi(MRM_BASE_URL)

    left, right = st.columns([1, 1])
    with left:
        refresh_sec = st.slider("Refresh (sec)", 0.5, 5.0, 1.0, 0.5)
    with right:
        window = st.slider("Window (points)", 20, 300, 120, 10)

    if "gpu_hist" not in st.session_state:
        st.session_state["gpu_hist"] = []

    run = st.toggle("Live", value=True)
    ph_status = st.empty()

    # layout containers (stable UI)
    box = st.container()
    charts = st.container()
    proc_box = st.container()

    while run:
        try:
            m = api.gpu_metrics()
        except Exception as e:
            ph_status.error(str(e))
            time.sleep(refresh_sec)
            continue

        if not m.get("ok"):
            ph_status.error(m.get("error", "unknown error"))
            time.sleep(refresh_sec)
            continue

        gpus = m.get("gpus", [])
        if not gpus:
            ph_status.warning("No GPUs detected")
            time.sleep(refresh_sec)
            continue

        ids = [str(g["index"]) for g in gpus]
        gpu_id = st.selectbox("GPU index", ids, index=0, key="gpu_pick")
        g = next((x for x in gpus if str(x["index"]) == gpu_id), gpus[0])

        # ====== TOP STATUS (clear, readable) ======
        with box:
            c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1])
            c1.metric("GPU", f"{g['index']} — {g['name']}")
            c2.metric("Util", f"{int(g['util_gpu_pct'])}%")
            c3.metric("VRAM", f"{int(g['mem_used_mib'])}/{int(g['mem_total_mib'])} MiB")
            c4.metric("Temp", f"{g.get('temp_c') if g.get('temp_c') is not None else '-'} °C")
            c5.metric("Power", f"{int(g.get('power_w') or 0)} W" if g.get("power_w") is not None else "-")

        # ====== HIST (charts) ======
        st.session_state["gpu_hist"].append(
            {
                "ts": m["ts"],
                "util_gpu_pct": g["util_gpu_pct"],
                "util_mem_pct": g["util_mem_pct"],
                "mem_used_mib": g["mem_used_mib"],
                "temp_c": g.get("temp_c"),
                "power_w": g.get("power_w"),
                "sm_clock_mhz": g.get("sm_clock_mhz"),
            }
        )
        st.session_state["gpu_hist"] = st.session_state["gpu_hist"][-window:]

        df = pd.DataFrame(st.session_state["gpu_hist"])
        if not df.empty:
            df["t"] = pd.to_datetime(df["ts"], unit="s")
            with charts:
                st.subheader("History")
                a, b = st.columns(2)
                with a:
                    st.caption("Util (%)")
                    st.line_chart(df.set_index("t")[["util_gpu_pct", "util_mem_pct"]])
                with b:
                    st.caption("VRAM (MiB)")
                    st.line_chart(df.set_index("t")[["mem_used_mib"]])

                c, d = st.columns(2)
                with c:
                    st.caption("Temp (°C)")
                    if "temp_c" in df.columns:
                        st.line_chart(df.set_index("t")[["temp_c"]])
                with d:
                    st.caption("Power (W)")
                    if "power_w" in df.columns:
                        st.line_chart(df.set_index("t")[["power_w"]])

        # ====== WHO HOLDS VRAM ======
        st.subheader("Who holds VRAM")

        dfp = _build_proc_table(g)
        if dfp.empty:
            st.caption("No running GPU processes detected by NVML.")
        else:
            # summary per container
            agg = (
                dfp.groupby("Container", as_index=False)["VRAM"]
                .sum()
                .sort_values("VRAM", ascending=False)
            )

            total = float(g.get("mem_total_mib") or 0.0)
            used = float(g.get("mem_used_mib") or 0.0)

            sum_procs = float(agg["VRAM"].sum())
            other = max(0.0, used - sum_procs)
            if other > 1.0:
                agg = pd.concat(
                    [agg, pd.DataFrame([{"Container": "other", "VRAM": other}])],
                    ignore_index=True,
                )

            # show KPIs
            k1, k2, k3 = st.columns(3)
            k1.metric("GPU VRAM Used", f"{int(used)}/{int(total)} MiB")
            k2.metric("Attributed to procs", f"{int(sum_procs)} MiB")
            k3.metric("Other/driver/context", f"{int(other)} MiB")

            # tables
            st.caption("By container")
            agg_show = agg.copy()
            agg_show["VRAM"] = agg_show["VRAM"].map(_fmt_mib)
            st.dataframe(agg_show, width="stretch", hide_index=True)

            st.caption("Processes")
            dfp_show = dfp.copy()
            dfp_show["VRAM"] = dfp_show["VRAM"].map(_fmt_mib)
            st.dataframe(dfp_show, width="stretch", hide_index=True)

            # charts
            st.caption("Charts")

            # bar (most readable)
            fig1, ax1 = plt.subplots()
            top = agg.sort_values("VRAM", ascending=True).tail(8)
            ax1.barh(top["Container"], top["VRAM"])
            ax1.set_xlabel("MiB")
            ax1.set_title("Top containers by VRAM")
            st.pyplot(fig1, clear_figure=True)

            # pie
            fig2, ax2 = plt.subplots()
            pie = agg.sort_values("VRAM", ascending=False).head(8)
            labels = [f"{c}" for c in pie["Container"].tolist()]
            ax2.pie(pie["VRAM"].values, labels=labels, autopct="%1.1f%%", startangle=90)
            ax2.axis("equal")
            ax2.set_title("VRAM breakdown (top 8)")
            st.pyplot(fig2, clear_figure=True)

        time.sleep(refresh_sec)
        st.rerun()
