"""
ui/pages/experiments.py

Experiments page — CPU vs GPU head-to-head comparison.
Same prompt, both runtimes, side-by-side latency / cost / quality.
"""
from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import streamlit as st

from ui.services.gateway_client import gateway
from ui.components.metrics import api_result_header, cost_card
from ui.utils import state as S
from ui.utils.cost import estimate_gpu, estimate_cpu, savings_pct
from ui.utils.formatters import fmt_latency, fmt_ts

_BASE_MODELS = [
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]


def render():
    st.title("🧪 Experiments")
    st.caption("Head-to-head CPU vs GPU comparison — same prompt, both runtimes, measured side-by-side.")

    tab_run, tab_history, tab_analysis = st.tabs(
        ["▶ Run experiment", "📋 History", "📊 Analysis"]
    )

    # ── Run experiment ────────────────────────────────────────────────
    with tab_run:
        col_cfg, col_prompt = st.columns([1, 2])

        with col_cfg:
            st.subheader("Configuration")

            # Model selection
            models_result = gateway.list_models()
            live = []
            if models_result.ok and isinstance(models_result.data, dict):
                live = [m.get("id") for m in models_result.data.get("data", []) if m.get("id")]
            model_opts = live or _BASE_MODELS
            model = st.selectbox("Model", model_opts, key="exp_model")

            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05, key="exp_temp")
            max_tokens  = st.slider("Max tokens", 32, 1024, 200, 32, key="exp_maxtok")

            run_gpu = st.checkbox("Run GPU inference", value=True, key="exp_gpu")
            run_cpu = st.checkbox("Run CPU inference", value=True, key="exp_cpu")

            n_runs = st.number_input("Runs per runtime (avg)", 1, 5, 1, key="exp_nruns",
                                     help="More runs = more stable latency averages")

            exp_name = st.text_input("Experiment name (optional)",
                                     placeholder=f"exp-{datetime.utcnow().strftime('%H%M%S')}",
                                     key="exp_name")

        with col_prompt:
            st.subheader("Prompt")
            prompt = st.text_area(
                "User message",
                value="Explain quantum entanglement in 3 sentences.",
                height=120,
                key="exp_prompt",
            )
            sys_prompt = st.text_area(
                "System prompt (optional)",
                height=70,
                placeholder="You are a concise science communicator.",
                key="exp_sys",
            )

        if st.button("🚀 Run experiment", type="primary", key="exp_run_btn"):
            if not prompt.strip():
                st.warning("Prompt cannot be empty.")
                st.stop()

            if not run_gpu and not run_cpu:
                st.warning("Select at least one runtime.")
                st.stop()

            messages = []
            if sys_prompt.strip():
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": prompt})

            gpu_results = []
            cpu_results = []

            progress = st.progress(0, text="Running…")
            total_runs = (n_runs if run_gpu else 0) + (n_runs if run_cpu else 0)
            done = 0

            if run_gpu:
                for i in range(n_runs):
                    with st.spinner(f"GPU run {i+1}/{n_runs}…"):
                        r = gateway.chat(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            runtime_preference="gpu",
                        )
                    done += 1
                    progress.progress(done / total_runs, text=f"GPU run {i+1}/{n_runs} done")
                    gpu_results.append(r)

            if run_cpu:
                for i in range(n_runs):
                    with st.spinner(f"CPU run {i+1}/{n_runs}…"):
                        r = gateway.chat(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            runtime_preference="cpu",
                        )
                    done += 1
                    progress.progress(done / total_runs, text=f"CPU run {i+1}/{n_runs} done")
                    cpu_results.append(r)

            progress.empty()

            # ── Results side-by-side ──────────────────────────────────
            st.divider()
            st.subheader("Results")

            col_g, col_c = st.columns(2)

            def _avg(rs, key):
                vals = [r.latency_ms for r in rs if r.ok]
                return sum(vals) / len(vals) if vals else 0

            def _tokens(rs):
                for r in rs:
                    if r.ok and isinstance(r.data, dict):
                        return r.data.get("usage", {}).get("completion_tokens", max_tokens)
                return max_tokens

            def _response_text(rs):
                for r in rs:
                    if r.ok:
                        try:
                            return r.data["choices"][0]["message"]["content"]
                        except Exception:
                            pass
                return "(no response)"

            gpu_lat = _avg(gpu_results, "latency_ms")
            cpu_lat = _avg(cpu_results, "latency_ms")
            gpu_tok = _tokens(gpu_results)
            cpu_tok = _tokens(cpu_results)

            with col_g:
                st.subheader("⚡ GPU")
                if run_gpu:
                    ok_runs = [r for r in gpu_results if r.ok]
                    fail_runs = [r for r in gpu_results if not r.ok]
                    if fail_runs:
                        st.error(f"{len(fail_runs)} run(s) failed: {fail_runs[0].error}")
                    if ok_runs:
                        est_g = estimate_gpu(gpu_lat, gpu_tok)
                        cost_card(est_g)
                        st.markdown(f"**Response:**\n\n{_response_text(ok_runs)}")
                else:
                    st.info("GPU not selected")

            with col_c:
                st.subheader("🖥 CPU")
                if run_cpu:
                    ok_runs = [r for r in cpu_results if r.ok]
                    fail_runs = [r for r in cpu_results if not r.ok]
                    if fail_runs:
                        st.error(f"{len(fail_runs)} run(s) failed: {fail_runs[0].error}")
                    if ok_runs:
                        est_c = estimate_cpu(cpu_lat, cpu_tok)
                        cost_card(est_c)
                        st.markdown(f"**Response:**\n\n{_response_text(ok_runs)}")
                else:
                    st.info("CPU not selected")

            # ── Summary comparison ────────────────────────────────────
            if run_gpu and run_cpu and gpu_lat and cpu_lat:
                st.divider()
                st.subheader("📊 Summary")
                sc = st.columns(4)
                sc[0].metric("GPU latency",  fmt_latency(gpu_lat))
                sc[1].metric("CPU latency",  fmt_latency(cpu_lat),
                             delta=f"+{fmt_latency(cpu_lat - gpu_lat)}" if cpu_lat > gpu_lat else fmt_latency(cpu_lat - gpu_lat))
                sc[2].metric("Latency ratio", f"×{cpu_lat/gpu_lat:.1f}" if gpu_lat else "—")

                est_g = estimate_gpu(gpu_lat, gpu_tok)
                est_c = estimate_cpu(cpu_lat, cpu_tok)
                sc[3].metric("Cost savings (CPU)", f"{savings_pct(est_g['cost_usd'], est_c['cost_usd'])}%",
                             help="How much cheaper CPU was vs GPU for this request")

            # ── Persist to history ────────────────────────────────────
            record = {
                "ts": datetime.utcnow().isoformat(),
                "name": exp_name or f"exp-{datetime.utcnow().strftime('%H%M%S')}",
                "model": model,
                "prompt": prompt[:80],
                "max_tokens": max_tokens,
                "gpu_latency_ms": round(gpu_lat, 1) if run_gpu else None,
                "cpu_latency_ms": round(cpu_lat, 1) if run_cpu else None,
                "gpu_cost": round(estimate_gpu(gpu_lat, gpu_tok)["cost_usd"], 6) if run_gpu and gpu_lat else None,
                "cpu_cost": round(estimate_cpu(cpu_lat, cpu_tok)["cost_usd"], 6) if run_cpu and cpu_lat else None,
                "gpu_response": _response_text(gpu_results) if run_gpu else None,
                "cpu_response": _response_text(cpu_results) if run_cpu else None,
            }
            S.push("experiment_results", record)
            st.success("Experiment saved to history.")

    # ── History ───────────────────────────────────────────────────────
    with tab_history:
        st.subheader("Experiment history")
        history: list[dict] = S.get("experiment_results") or []

        if not history:
            st.info("No experiments yet. Run one in the first tab.")
        else:
            if st.button("🗑 Clear history", key="exp_clear"):
                S.set("experiment_results", [])
                st.rerun()

            for rec in reversed(history):
                label = (
                    f"**{rec.get('name','?')}**  ·  "
                    f"`{rec.get('model','?')}`  ·  "
                    f"{fmt_ts(rec.get('ts'))}"
                )
                with st.expander(label):
                    cc = st.columns(4)
                    cc[0].metric("GPU latency",
                                 fmt_latency(rec["gpu_latency_ms"]) if rec.get("gpu_latency_ms") else "N/A")
                    cc[1].metric("CPU latency",
                                 fmt_latency(rec["cpu_latency_ms"]) if rec.get("cpu_latency_ms") else "N/A")
                    cc[2].metric("GPU cost",
                                 f"${rec['gpu_cost']:.5f}" if rec.get("gpu_cost") else "N/A")
                    cc[3].metric("CPU cost",
                                 f"${rec['cpu_cost']:.5f}" if rec.get("cpu_cost") else "N/A")

                    st.caption(f"**Prompt:** {rec.get('prompt')}")

                    if rec.get("gpu_response") and rec.get("cpu_response"):
                        rc1, rc2 = st.columns(2)
                        with rc1:
                            st.markdown("**GPU response:**")
                            st.info(rec["gpu_response"][:400])
                        with rc2:
                            st.markdown("**CPU response:**")
                            st.info(rec["cpu_response"][:400])

    # ── Analysis ──────────────────────────────────────────────────────
    with tab_analysis:
        st.subheader("Aggregate analysis")
        history = S.get("experiment_results") or []

        if len(history) < 2:
            st.info("Run at least 2 experiments to see aggregate analysis.")
        else:
            rows = []
            for r in history:
                if r.get("gpu_latency_ms") is not None:
                    rows.append({"runtime": "GPU", "latency_ms": r["gpu_latency_ms"],
                                 "cost_usd": r.get("gpu_cost", 0),
                                 "model": r.get("model", ""),
                                 "ts": r.get("ts", "")})
                if r.get("cpu_latency_ms") is not None:
                    rows.append({"runtime": "CPU", "latency_ms": r["cpu_latency_ms"],
                                 "cost_usd": r.get("cpu_cost", 0),
                                 "model": r.get("model", ""),
                                 "ts": r.get("ts", "")})

            df = pd.DataFrame(rows)
            df["ts"] = pd.to_datetime(df["ts"])

            ac = st.columns(2)
            with ac[0]:
                st.caption("Latency distribution")
                chart_df = df.pivot_table(index="ts", columns="runtime",
                                          values="latency_ms", aggfunc="mean")
                st.line_chart(chart_df)

            with ac[1]:
                st.caption("Cost per request")
                cost_df = df.pivot_table(index="ts", columns="runtime",
                                         values="cost_usd", aggfunc="mean")
                st.bar_chart(cost_df)

            # Summary stats table
            st.caption("Aggregated stats")
            stats = df.groupby("runtime").agg(
                runs=("latency_ms", "count"),
                avg_latency=("latency_ms", "mean"),
                p95_latency=("latency_ms", lambda x: x.quantile(0.95)),
                avg_cost=("cost_usd", "mean"),
                total_cost=("cost_usd", "sum"),
            ).round(4).reset_index()
            st.dataframe(stats, use_container_width=True, hide_index=True)
