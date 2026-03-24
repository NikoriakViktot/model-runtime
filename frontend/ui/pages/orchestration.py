"""
ui/pages/orchestration.py

Orchestration — Control Plane contract submission and run management.
Full visibility into the Run state machine: CREATED → DONE/FAILED.
"""
from __future__ import annotations

import json

import streamlit as st

from ui.services.control_plane_client import cp
from ui.components.metrics import api_result_header
from ui.components.json_viewer import json_viewer
from ui.utils.formatters import fmt_ts

_STATE_COLORS = {
    "CREATED":         ("🆕", "blue"),
    "DATASET_RUNNING": ("🔄", "orange"),
    "DATASET_READY":   ("✅", "green"),
    "TRAIN_RUNNING":   ("🏋", "orange"),
    "TRAIN_READY":     ("✅", "green"),
    "DONE":            ("🟢", "green"),
    "FAILED":          ("🔴", "red"),
}

_CONTRACT_TYPES = [
    "dataset.build.v1",
    "train.qlora.v1",
    "train.dpo.v1",
    "eval.standard.v1",
]

_DEFAULT_PAYLOADS = {
    "dataset.build.v1": {
        "target_name": "my-dataset",
        "source_type": "hf",
        "hf_dataset": "tatsu-lab/alpaca",
        "max_rows": 1000,
    },
    "train.qlora.v1": {
        "base_model": "Qwen/Qwen1.5-1.8B-Chat",
        "dataset_name": "my-dataset",
        "epochs": 3,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
    },
    "train.dpo.v1": {
        "base_model": "Qwen/Qwen1.5-1.8B-Chat",
        "dataset_name": "my-preference-dataset",
        "epochs": 1,
        "beta": 0.1,
    },
    "eval.standard.v1": {
        "model": "Qwen/Qwen1.5-1.8B-Chat",
        "dataset_name": "my-eval-dataset",
        "metrics": ["accuracy", "bleu"],
    },
}


def _state_badge(state: str) -> str:
    icon, _ = _STATE_COLORS.get(state, ("⚪", "grey"))
    return f"{icon} {state}"


def _run_detail(run: dict) -> None:
    run_id   = run.get("run_id", "?")
    state    = run.get("state", "?")
    contract = run.get("contract_type", "?")
    created  = fmt_ts(run.get("created_at"))
    updated  = fmt_ts(run.get("updated_at"))
    payload  = run.get("payload") or run.get("contract", {})
    artifacts = run.get("artifacts") or {}

    icon, _ = _STATE_COLORS.get(state, ("⚪", "grey"))

    st.markdown(f"### {icon} Run `{run_id[:16]}…`")

    # State machine progress bar
    all_states = [
        "CREATED", "DATASET_RUNNING", "DATASET_READY",
        "TRAIN_RUNNING", "TRAIN_READY", "DONE",
    ]
    idx = next((i for i, s in enumerate(all_states) if s == state), -1)
    if state != "FAILED":
        progress = (idx + 1) / len(all_states) if idx >= 0 else 0
        st.progress(progress, text=f"{state}  ({idx+1}/{len(all_states)})")
    else:
        st.error("Run FAILED")

    cols = st.columns(4)
    cols[0].caption(f"**Contract:** `{contract}`")
    cols[1].caption(f"**State:** `{state}`")
    cols[2].caption(f"**Created:** {created}")
    cols[3].caption(f"**Updated:** {updated}")

    if artifacts:
        st.subheader("Artifacts")
        for k, v in artifacts.items():
            st.caption(f"📦 `{k}` → `{v}`")

    c1, c2 = st.columns(2)
    with c1:
        json_viewer(payload, label="Contract payload", expanded=False)
    with c2:
        json_viewer(run, label="Full run record", expanded=False)


def render():
    st.title("🎛 Orchestration")
    st.caption("Submit contracts, monitor runs, inspect the Control Plane state machine.")

    tab_submit, tab_runs, tab_run_detail = st.tabs(
        ["📝 Submit contract", "📋 All runs", "🔍 Run detail"]
    )

    # ── Submit contract ───────────────────────────────────────────────
    with tab_submit:
        st.subheader("Submit a new contract")

        col_type, col_spec = st.columns([1, 2])

        with col_type:
            contract_type = st.selectbox(
                "Contract type",
                _CONTRACT_TYPES,
                key="orch_contract_type",
            )
            st.caption(
                {
                    "dataset.build.v1": "Fetch & preprocess a dataset for training.",
                    "train.qlora.v1":   "QLoRA parameter-efficient fine-tuning.",
                    "train.dpo.v1":     "Direct Preference Optimization.",
                    "eval.standard.v1": "Evaluate model on a dataset.",
                }.get(contract_type, "")
            )

        with col_spec:
            default_payload = _DEFAULT_PAYLOADS.get(contract_type, {})
            payload_raw = st.text_area(
                "Payload (JSON)",
                value=json.dumps(default_payload, indent=2),
                height=250,
                key="orch_payload",
            )

        try:
            payload_parsed = json.loads(payload_raw)
            payload_valid = True
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            payload_valid = False
            payload_parsed = {}

        if st.button("🚀 Submit", type="primary", key="orch_submit", disabled=not payload_valid):
            with st.spinner(f"Submitting `{contract_type}`…"):
                res = cp.submit_contract(contract_type, payload_parsed)
            api_result_header(res, service="Control Plane /contracts")
            if res.ok:
                run_id = res.data.get("run_id", "?") if isinstance(res.data, dict) else "?"
                st.success(f"Contract submitted! Run ID: `{run_id}`")
                json_viewer(res.data, label="Response", expanded=True)
            else:
                st.error(res.error)

    # ── All runs ─────────────────────────────────────────────────────
    with tab_runs:
        col_ref, col_filt = st.columns([1, 3])
        with col_ref:
            if st.button("🔄 Refresh", key="orch_runs_ref"):
                st.rerun()
        with col_filt:
            state_filter = st.multiselect(
                "Filter by state",
                list(_STATE_COLORS.keys()),
                default=[],
                key="orch_state_filter",
            )

        r_runs = cp.list_runs()
        api_result_header(r_runs, service="Control Plane /runs")

        if not r_runs.ok:
            st.error(r_runs.error)
        else:
            runs = r_runs.data if isinstance(r_runs.data, list) else []

            if state_filter:
                runs = [r for r in runs if r.get("state") in state_filter]

            if not runs:
                st.info("No runs found. Submit a contract in the first tab.")
            else:
                # Summary KPIs
                from collections import Counter
                counts = Counter(r.get("state", "UNKNOWN") for r in runs)
                kpi_cols = st.columns(min(len(counts) + 1, 7))
                kpi_cols[0].metric("Total", len(runs))
                for i, (state, cnt) in enumerate(sorted(counts.items()), 1):
                    if i < 7:
                        icon, _ = _STATE_COLORS.get(state, ("⚪", "grey"))
                        kpi_cols[i].metric(f"{icon} {state}", cnt)

                st.divider()

                # Run table
                for run in sorted(runs, key=lambda r: r.get("created_at", ""), reverse=True):
                    run_id   = run.get("run_id", "?")
                    state    = run.get("state", "?")
                    contract = run.get("contract_type", "?")
                    created  = fmt_ts(run.get("created_at"))
                    icon, _  = _STATE_COLORS.get(state, ("⚪", "grey"))

                    with st.expander(
                        f"{icon} `{run_id[:16]}…`  ·  **{contract}**  ·  {state}  ·  {created}"
                    ):
                        _run_detail(run)

                        # Events for this run
                        st.subheader("Events")
                        r_ev = cp.list_events(run_id=run_id)
                        if r_ev.ok:
                            events = r_ev.data if isinstance(r_ev.data, list) else []
                            for ev in sorted(events, key=lambda e: e.get("ts", ""), reverse=True):
                                ev_type = ev.get("event_type", "?")
                                ts      = fmt_ts(ev.get("ts"))
                                detail  = ev.get("detail", "")
                                ev_icon = {"state_transition": "🔀", "error": "🔴",
                                           "artifact": "📦", "log": "📝"}.get(ev_type, "ℹ")
                                st.markdown(f"{ev_icon} `{ts}`  ·  **{ev_type}**  —  {detail}")
                        else:
                            st.caption("Events unavailable.")

    # ── Run detail ────────────────────────────────────────────────────
    with tab_run_detail:
        st.subheader("Inspect a specific run")

        run_id_input = st.text_input(
            "Run ID",
            placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            key="orch_run_id_input",
        )

        if st.button("🔍 Fetch run", key="orch_fetch_run") and run_id_input.strip():
            with st.spinner("Fetching run…"):
                r = cp.get_run(run_id_input.strip())
            api_result_header(r, service=f"Control Plane /runs/{run_id_input[:8]}…")

            if not r.ok:
                st.error(r.error)
            elif isinstance(r.data, dict):
                _run_detail(r.data)

                # Events
                st.subheader("Events")
                r_ev = cp.list_events(run_id=run_id_input.strip())
                if r_ev.ok:
                    events = r_ev.data if isinstance(r_ev.data, list) else []
                    if events:
                        for ev in sorted(events, key=lambda e: e.get("ts", ""), reverse=True):
                            ev_type = ev.get("event_type", "?")
                            ts      = fmt_ts(ev.get("ts"))
                            detail  = ev.get("detail", "")
                            ev_icon = {"state_transition": "🔀", "error": "🔴",
                                       "artifact": "📦", "log": "📝"}.get(ev_type, "ℹ")
                            st.markdown(f"{ev_icon} `{ts}`  ·  **{ev_type}**  —  {detail}")
                    else:
                        st.info("No events for this run.")
