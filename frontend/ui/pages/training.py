"""
ui/pages/training.py

Training pipeline UI:
  - Submit dataset build, QLoRA, DPO contracts to the Control Plane
  - Browse active runs and their state machine transitions
  - View run events timeline
  - Quick links to MLflow experiment tracking
"""
from __future__ import annotations

import json
from datetime import datetime

import streamlit as st

from ui.services.control_plane_client import cp
from ui.services.gateway_client import gateway
from ui.components.metrics import api_result_header
from ui.components.json_viewer import json_viewer
from ui.utils.formatters import fmt_ts

_STATE_ICON = {
    "CREATED":         "🆕",
    "DATASET_RUNNING": "🔄",
    "DATASET_READY":   "✅",
    "TRAIN_RUNNING":   "🏋",
    "TRAIN_READY":     "✅",
    "DONE":            "🟢",
    "FAILED":          "🔴",
}

_BASE_MODELS = [
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

_HF_DATASETS = [
    "tatsu-lab/alpaca",
    "databricks/databricks-dolly-15k",
    "Open-Orca/OpenOrca",
    "HuggingFaceH4/ultrafeedback_binarized",
]


def _state_progress(state: str) -> None:
    stages = [
        ("CREATED",         "Created"),
        ("DATASET_RUNNING", "Building dataset"),
        ("DATASET_READY",   "Dataset ready"),
        ("TRAIN_RUNNING",   "Training"),
        ("TRAIN_READY",     "Train ready"),
        ("DONE",            "Done"),
    ]
    idx = next((i for i, (s, _) in enumerate(stages) if s == state), -1)
    cols = st.columns(len(stages))
    for i, (s, label) in enumerate(stages):
        with cols[i]:
            if i < idx:
                st.markdown(f"✅ **{label}**")
            elif i == idx:
                st.markdown(f"▶ **{label}**")
            else:
                st.markdown(f"⬜ {label}")


def render():
    st.title("🏋 Training")
    st.caption("Submit training jobs, track runs, and monitor dataset + fine-tuning pipelines.")

    tab_dataset, tab_qlora, tab_dpo, tab_runs, tab_events = st.tabs(
        ["📦 Dataset Build", "🔬 QLoRA", "🔀 DPO", "📋 Runs", "📜 Events"]
    )

    # ── Dataset build ─────────────────────────────────────────────────
    with tab_dataset:
        st.subheader("Build dataset contract")
        st.caption("Fetch and preprocess a dataset from HuggingFace for fine-tuning.")

        c1, c2 = st.columns(2)
        with c1:
            ds_name = st.text_input(
                "Dataset name (output)",
                placeholder="my-alpaca-dataset",
                key="tr_ds_name",
            )
            source_type = st.selectbox("Source type", ["hf", "s3", "local"], key="tr_ds_source")
            if source_type == "hf":
                hf_dataset = st.selectbox(
                    "HuggingFace dataset",
                    _HF_DATASETS,
                    key="tr_ds_hf",
                )
            else:
                hf_dataset = st.text_input("Dataset path/URI", key="tr_ds_path")

        with c2:
            max_rows = st.number_input("Max rows", 100, 1_000_000, 1000, step=100, key="tr_ds_maxrows")
            split = st.text_input("Split", value="train", key="tr_ds_split")
            instruction_col = st.text_input("Instruction column", value="instruction", key="tr_ds_instcol")
            output_col = st.text_input("Output column", value="output", key="tr_ds_outcol")

        if st.button("📦 Submit dataset job", type="primary", key="tr_ds_submit"):
            if not ds_name.strip():
                st.warning("Dataset name is required.")
            else:
                payload = {
                    "target_name": ds_name.strip(),
                    "source_type": source_type,
                    "hf_dataset": hf_dataset,
                    "max_rows": int(max_rows),
                    "split": split,
                    "instruction_col": instruction_col,
                    "output_col": output_col,
                }
                with st.spinner("Submitting dataset build contract…"):
                    res = cp.submit_contract("dataset.build.v1", payload)
                api_result_header(res, service="Control Plane /contracts")
                if res.ok:
                    st.success(f"Dataset job submitted. Run ID: `{res.data.get('run_id', '?')}`")
                    json_viewer(res.data, label="Response", expanded=False)
                else:
                    st.error(res.error)

    # ── QLoRA ─────────────────────────────────────────────────────────
    with tab_qlora:
        st.subheader("QLoRA fine-tuning")
        st.caption("Launch parameter-efficient fine-tuning on a prepared dataset.")

        c1, c2 = st.columns(2)
        with c1:
            ql_model = st.selectbox("Base model", _BASE_MODELS, key="tr_ql_model")
            ql_dataset = st.text_input(
                "Dataset name (from dataset build)",
                placeholder="my-alpaca-dataset",
                key="tr_ql_ds",
            )
            ql_run_name = st.text_input(
                "Run name (optional)",
                placeholder=f"qlora-{datetime.utcnow().strftime('%Y%m%d-%H%M')}",
                key="tr_ql_name",
            )

        with c2:
            ql_epochs = st.number_input("Epochs", 1, 20, 3, key="tr_ql_epochs")
            ql_lr = st.select_slider(
                "Learning rate",
                options=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4],
                value=2e-4,
                key="tr_ql_lr",
            )
            ql_batch = st.number_input("Batch size", 1, 64, 4, key="tr_ql_batch")
            ql_r = st.slider("LoRA rank (r)", 4, 64, 16, key="tr_ql_r")
            ql_alpha = st.slider("LoRA alpha", 8, 128, 32, key="tr_ql_alpha")

        if st.button("🔬 Submit QLoRA job", type="primary", key="tr_ql_submit"):
            if not ql_dataset.strip():
                st.warning("Dataset name is required.")
            else:
                payload = {
                    "base_model": ql_model,
                    "dataset_name": ql_dataset.strip(),
                    "run_name": ql_run_name.strip() or None,
                    "epochs": int(ql_epochs),
                    "learning_rate": float(ql_lr),
                    "batch_size": int(ql_batch),
                    "lora_r": int(ql_r),
                    "lora_alpha": int(ql_alpha),
                }
                with st.spinner("Submitting QLoRA contract…"):
                    res = cp.submit_contract("train.qlora.v1", payload)
                api_result_header(res, service="Control Plane /contracts")
                if res.ok:
                    st.success(f"QLoRA job submitted. Run ID: `{res.data.get('run_id', '?')}`")
                    json_viewer(res.data, label="Response", expanded=False)
                else:
                    st.error(res.error)

    # ── DPO ───────────────────────────────────────────────────────────
    with tab_dpo:
        st.subheader("DPO fine-tuning")
        st.caption("Direct Preference Optimization — train from preference pairs.")

        c1, c2 = st.columns(2)
        with c1:
            dpo_model = st.selectbox("Base model", _BASE_MODELS, key="tr_dpo_model")
            dpo_dataset = st.text_input(
                "Preference dataset name",
                placeholder="my-preference-dataset",
                key="tr_dpo_ds",
            )
            dpo_run_name = st.text_input(
                "Run name (optional)",
                placeholder=f"dpo-{datetime.utcnow().strftime('%Y%m%d-%H%M')}",
                key="tr_dpo_name",
            )

        with c2:
            dpo_epochs = st.number_input("Epochs", 1, 10, 1, key="tr_dpo_epochs")
            dpo_beta = st.slider("β (DPO temperature)", 0.01, 1.0, 0.1, 0.01, key="tr_dpo_beta")
            dpo_lr = st.select_slider(
                "Learning rate",
                options=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
                value=1e-5,
                key="tr_dpo_lr",
            )
            dpo_batch = st.number_input("Batch size", 1, 32, 2, key="tr_dpo_batch")

        if st.button("🔀 Submit DPO job", type="primary", key="tr_dpo_submit"):
            if not dpo_dataset.strip():
                st.warning("Preference dataset name is required.")
            else:
                payload = {
                    "base_model": dpo_model,
                    "dataset_name": dpo_dataset.strip(),
                    "run_name": dpo_run_name.strip() or None,
                    "epochs": int(dpo_epochs),
                    "beta": float(dpo_beta),
                    "learning_rate": float(dpo_lr),
                    "batch_size": int(dpo_batch),
                }
                with st.spinner("Submitting DPO contract…"):
                    res = cp.submit_contract("train.dpo.v1", payload)
                api_result_header(res, service="Control Plane /contracts")
                if res.ok:
                    st.success(f"DPO job submitted. Run ID: `{res.data.get('run_id', '?')}`")
                    json_viewer(res.data, label="Response", expanded=False)
                else:
                    st.error(res.error)

    # ── Runs ─────────────────────────────────────────────────────────
    with tab_runs:
        st.subheader("All runs")
        col_ref, col_filt = st.columns([1, 3])
        with col_ref:
            if st.button("🔄 Refresh", key="tr_runs_ref"):
                st.rerun()
        with col_filt:
            state_filter = st.multiselect(
                "Filter by state",
                ["CREATED", "DATASET_RUNNING", "DATASET_READY",
                 "TRAIN_RUNNING", "TRAIN_READY", "DONE", "FAILED"],
                default=[],
                key="tr_runs_filter",
            )

        r_runs = cp.list_runs()
        api_result_header(r_runs, service="Control Plane /runs")

        if not r_runs.ok:
            st.error(r_runs.error)
        else:
            runs = r_runs.data if isinstance(r_runs.data, list) else []
            if state_filter:
                runs = [run for run in runs if run.get("state") in state_filter]

            if not runs:
                st.info("No runs found.")
            else:
                # Summary counters
                from collections import Counter
                counts = Counter(run.get("state", "UNKNOWN") for run in runs)
                sc = st.columns(min(len(counts), 6))
                for i, (state, cnt) in enumerate(counts.items()):
                    if i < 6:
                        sc[i].metric(f"{_STATE_ICON.get(state, '⚪')} {state}", cnt)
                st.divider()

                for run in sorted(runs, key=lambda r: r.get("created_at", ""), reverse=True):
                    run_id    = run.get("run_id", "?")
                    state     = run.get("state", "?")
                    contract  = run.get("contract_type", "?")
                    created   = fmt_ts(run.get("created_at"))
                    updated   = fmt_ts(run.get("updated_at"))
                    icon      = _STATE_ICON.get(state, "⚪")

                    with st.expander(f"{icon} `{run_id[:12]}…`  ·  **{contract}**  ·  {state}  ·  {created}"):
                        _state_progress(state)
                        st.divider()

                        ic = st.columns(3)
                        ic[0].caption(f"**Run ID:** `{run_id}`")
                        ic[1].caption(f"**Created:** {created}")
                        ic[2].caption(f"**Updated:** {updated}")

                        if run.get("artifacts"):
                            st.caption("**Artifacts:**")
                            for k, v in run["artifacts"].items():
                                st.caption(f"  `{k}` → `{v}`")

                        json_viewer(run, label="Raw run", expanded=False)

    # ── Events ───────────────────────────────────────────────────────
    with tab_events:
        st.subheader("Run events timeline")

        r_runs2 = cp.list_runs()
        run_ids = []
        if r_runs2.ok and isinstance(r_runs2.data, list):
            run_ids = [r.get("run_id") for r in r_runs2.data if r.get("run_id")]

        col_sel, col_all = st.columns([2, 1])
        with col_all:
            show_all = st.checkbox("All runs", value=True, key="tr_ev_all")
        with col_sel:
            if not show_all and run_ids:
                sel_run = st.selectbox("Run", run_ids, key="tr_ev_sel")
            else:
                sel_run = None

        if st.button("🔄 Refresh", key="tr_ev_ref"):
            st.rerun()

        r_events = cp.list_events(run_id=sel_run)
        api_result_header(r_events, service="Control Plane /events")

        if not r_events.ok:
            st.error(r_events.error)
        else:
            events = r_events.data if isinstance(r_events.data, list) else []
            if not events:
                st.info("No events found.")
            else:
                st.caption(f"{len(events)} event(s)")
                for ev in sorted(events, key=lambda e: e.get("ts", ""), reverse=True):
                    ev_type = ev.get("event_type", "?")
                    ts      = fmt_ts(ev.get("ts"))
                    rid     = ev.get("run_id", "")[:12]
                    detail  = ev.get("detail", "")

                    icon = {"state_transition": "🔀", "error": "🔴",
                            "artifact": "📦", "log": "📝"}.get(ev_type, "ℹ")
                    st.markdown(f"{icon} `{ts}`  ·  `{rid}…`  ·  **{ev_type}**  —  {detail}")
