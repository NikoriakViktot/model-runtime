"""
AI Runtime Control Plane — Streamlit UI
"""
import json
import os
import re
import time

import pandas as pd
import requests
import streamlit as st
from jinja2 import Template

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
CP_URL          = os.getenv("CONTROL_PLANE_URL",  "http://control_plane:8004")
GATEWAY_URL     = os.getenv("GATEWAY_URL",         "http://gateway:8080")
MRM_URL         = os.getenv("MRM_URL",             "http://model_runtime_manager:8010")
PROMETHEUS_URL  = os.getenv("PROMETHEUS_URL",      "http://prometheus:9090")
JAEGER_UI_URL   = os.getenv("JAEGER_UI_URL",       "http://localhost:16686")
S3_BUCKET       = os.getenv("AWS_BUCKET_NAME",     "model-runtime-artifacts")
MLFLOW_UI_URL   = "/mlflow/"

BASE_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen1.5-1.8B-Chat",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-2-2b-it",
    "microsoft/Phi-3-mini-4k-instruct",
    "HuggingFaceH4/zephyr-7b-beta",
]

st.set_page_config(page_title="AI Runtime", layout="wide", page_icon="🧠")


def init_session_state():
    defaults = {
        "messages":                 [],
        "selected_training_run_id": None,
        "last_run_id":              None,
        "hf_results":               [],
        "hf_query":                 "",
        "hf_gpu_id":                "0",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


init_session_state()


# ──────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────────────────────────────────────

def _http_get(url, *, params=None, timeout=10):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"HTTP GET failed: {e}")
        return None


def _http_post(url, *, json_body=None, timeout=30):
    try:
        r = requests.post(url, json=json_body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"HTTP POST failed: {e}")
        return None


def _http_get_silent(url, timeout=5):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _http_post_silent(url, *, json_body=None, timeout=30):
    try:
        r = requests.post(url, json=json_body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Domain helpers
# ──────────────────────────────────────────────────────────────────────────────

def cp_post_contract(contract_type, payload):
    envelope = {"type": contract_type, "spec_version": "v1", "payload": payload}
    return _http_post(f"{CP_URL}/contracts", json_body=envelope, timeout=10)


def cp_get_runs():
    data = _http_get(f"{CP_URL}/runs", timeout=5)
    return data if isinstance(data, list) else []


def cp_get_run(run_id):
    data = _http_get(f"{CP_URL}/runs/{run_id}", timeout=10)
    return data if isinstance(data, dict) else None


def gateway_chat(model, messages, temperature, max_tokens):
    payload = {"model": model, "messages": messages,
               "temperature": temperature, "max_tokens": max_tokens}
    return _http_post(f"{GATEWAY_URL}/v1/chat/completions", json_body=payload, timeout=120)


def gateway_list_models():
    data = _http_get_silent(f"{GATEWAY_URL}/v1/models")
    if isinstance(data, dict):
        return data.get("data", [])
    return []


def mrm_status_all():
    data = _http_get_silent(f"{MRM_URL}/models/status")
    return data if isinstance(data, list) else []


def mrm_gpu_metrics():
    return _http_get_silent(f"{MRM_URL}/gpu/metrics") or {}


def mrm_hf_search(q, limit=20):
    return _http_get_silent(f"{MRM_URL}/hf/search", timeout=20) or {}


def mrm_hf_recommend(q, gpu_id="0", limit=20):
    try:
        r = requests.get(
            f"{MRM_URL}/hf/recommend",
            params={"q": q, "gpu_id": gpu_id, "limit": limit},
            timeout=20,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def mrm_register_from_hf(repo_id, preset, gpu="0", overrides=None):
    payload = {"repo_id": repo_id, "preset": preset, "gpu": gpu,
               "overrides": overrides or {}}
    return _http_post_silent(f"{MRM_URL}/models/register_from_hf", json_body=payload)


def mrm_ensure(base_model):
    return _http_post_silent(f"{MRM_URL}/models/ensure",
                             json_body={"base_model": base_model}, timeout=600)


def mrm_stop(base_model):
    return _http_post_silent(f"{MRM_URL}/models/stop",
                             json_body={"base_model": base_model})


def mrm_remove(base_model):
    return _http_post_silent(f"{MRM_URL}/models/remove",
                             json_body={"base_model": base_model})


def get_datasets_from_cp():
    runs = _http_get_silent(f"{CP_URL}/runs") or []
    out = []
    for run in runs:
        if run.get("type") != "dataset.build.v1":
            continue
        if run.get("state") not in ("DONE", "DATASET_READY"):
            continue
        artifacts = run.get("artifacts") or {}
        contract_payload = (run.get("contract") or {}).get("payload") or {}
        uri = artifacts.get("dataset_uri") or artifacts.get("s3_uri")
        if not uri:
            continue
        out.append({
            "label":  f"{contract_payload.get('target_name','?')} | {run.get('created_at','')[:16]}",
            "uri":    uri,
            "slug":   contract_payload.get("target_name", "unknown"),
            "run_id": run.get("id", ""),
        })
    return sorted(out, key=lambda x: x["run_id"], reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# Prometheus helpers (Monitoring page)
# ──────────────────────────────────────────────────────────────────────────────

_PROM_SCRAPE_LABELS = frozenset({"instance", "job", "service", "__name__"})


def _prom_query(q):
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/query",
                         params={"query": q}, timeout=5)
        r.raise_for_status()
        result = r.json()["data"]["result"]
        if result:
            v = float(result[0]["value"][1])
            return None if v != v else v
    except Exception:
        pass
    return None


def _prom_range(q, minutes=30, step="30s"):
    import time as _t
    end = int(_t.time())
    start = end - minutes * 60
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range",
                         params={"query": q, "start": start, "end": end, "step": step},
                         timeout=10)
        r.raise_for_status()
        results = r.json()["data"]["result"]
        rows = []
        for i, series in enumerate(results):
            metric_labels = {k: v for k, v in series["metric"].items()
                             if k not in _PROM_SCRAPE_LABELS}
            if metric_labels:
                lbl = ",".join(f"{k}={v.replace(':', '_')}"
                               for k, v in metric_labels.items())
            else:
                lbl = f"series_{i}" if len(results) > 1 else "value"
            for ts, val in series["values"]:
                rows.append({"time": pd.Timestamp(ts, unit="s"),
                             "value": float(val), "series": lbl})
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.pivot_table(index="time", columns="series",
                              values="value", aggfunc="mean")
    except Exception:
        return pd.DataFrame()


def _slo_badge(label, value, warn_thresh, crit_thresh, unit=""):
    if value is None:
        st.metric(label, "N/A")
        return
    color = "🟢" if value < warn_thresh else ("🟡" if value < crit_thresh else "🔴")
    st.metric(label, f"{color} {value:.1f}{unit}")


# ──────────────────────────────────────────────────────────────────────────────
# Navigation
# ──────────────────────────────────────────────────────────────────────────────

st.sidebar.title("🧬 AI Runtime")
page = st.sidebar.radio(
    "Navigation",
    ["HF Hub", "Model Registry", "Chat", "Training", "Monitoring", "Orchestration", "Prompt Studio"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"📊 **[MLflow UI]({MLFLOW_UI_URL})**", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Page: HF Hub
# ──────────────────────────────────────────────────────────────────────────────

if page == "HF Hub":
    st.title("🤗 HuggingFace Hub")
    st.caption("Search models, get GPU-aware recommendations, and register them for inference.")

    tab_search, tab_recommend, tab_register = st.tabs(
        ["🔍 Search", "🎯 GPU Recommendations", "➕ Register"]
    )

    # ── Search ──────────────────────────────────────────────────────────
    with tab_search:
        col_q, col_lim, col_btn = st.columns([4, 1, 1])
        with col_q:
            query = st.text_input("Search HuggingFace", value=st.session_state.hf_query,
                                  placeholder="e.g. llama 7b instruct, qwen 1.8b chat")
        with col_lim:
            limit = st.number_input("Limit", 5, 50, 20, label_visibility="visible")
        with col_btn:
            st.write("")
            do_search = st.button("Search", type="primary")

        if do_search and query:
            st.session_state.hf_query = query
            with st.spinner("Searching HuggingFace…"):
                try:
                    r = requests.get(f"{MRM_URL}/hf/search",
                                     params={"q": query, "limit": limit}, timeout=20)
                    r.raise_for_status()
                    items = r.json().get("items", [])
                    st.session_state.hf_results = items
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    items = []

        items = st.session_state.get("hf_results", [])
        if items:
            st.caption(f"{len(items)} results")
            for m in items:
                repo_id = m.get("modelId") or m.get("id") or "?"
                downloads = m.get("downloads", 0)
                likes = m.get("likes", 0)
                tags = m.get("tags") or []
                pipeline = m.get("pipeline_tag", "")

                with st.expander(f"**{repo_id}**   ⬇ {downloads:,}  ❤ {likes:,}  {pipeline}"):
                    col_info, col_action = st.columns([3, 1])
                    with col_info:
                        st.caption(" · ".join(tags[:8]))
                        config = m.get("config") or {}
                        archs = config.get("architectures") or []
                        if archs:
                            st.caption(f"Architecture: `{archs[0]}`")
                    with col_action:
                        preset_opt = st.selectbox(
                            "Preset", ["small_chat", "7b_awq"],
                            key=f"preset_search_{repo_id}",
                        )
                        if st.button("Register", key=f"reg_search_{repo_id}"):
                            res = mrm_register_from_hf(repo_id, preset_opt)
                            if res:
                                st.success(f"Registered `{repo_id}` → preset `{preset_opt}`")
                            else:
                                st.error("Registration failed")

    # ── GPU Recommendations ─────────────────────────────────────────────
    with tab_recommend:
        st.subheader("GPU-Aware Model Recommendations")
        st.caption("Ranks models by popularity, architecture bonus, and VRAM fit on your GPU.")

        col_q2, col_gpu, col_btn2 = st.columns([3, 1, 1])
        with col_q2:
            rec_query = st.text_input("Query", placeholder="e.g. small instruct chat",
                                      key="rec_query")
        with col_gpu:
            gpu_id = st.text_input("GPU ID", value="0", key="rec_gpu")
        with col_btn2:
            st.write("")
            do_rec = st.button("Recommend", type="primary", key="do_rec")

        if do_rec and rec_query:
            with st.spinner("Fetching recommendations…"):
                rec_data = mrm_hf_recommend(rec_query, gpu_id=gpu_id)

            if rec_data is None:
                st.error("Failed to reach MRM /hf/recommend")
            else:
                free_gb = rec_data.get("gpu_free_gb", 0)
                total = rec_data.get("total_fetched", 0)
                fit = rec_data.get("total_fit", 0)
                st.info(
                    f"GPU {gpu_id} — **{free_gb:.1f} GiB free** · "
                    f"Fetched {total} models · {fit} fit in VRAM"
                )

                results = rec_data.get("results", [])
                if not results:
                    st.warning("No recommendations returned.")
                else:
                    for rank, item in enumerate(results, 1):
                        repo_id   = item.get("repo_id", "?")
                        score     = item.get("score", 0)
                        params_b  = item.get("params_b", 0)
                        vram_gb   = item.get("estimated_vram_gb", 0)
                        fits      = item.get("fits")
                        preset    = item.get("preset", "small_chat")
                        downloads = item.get("downloads", 0)
                        arch      = item.get("architecture") or ""

                        fit_icon = ("🟢" if fits else "🔴") if fits is not None else "⚪"
                        params_str = f"{params_b}B" if params_b else "?"
                        vram_str   = f"{vram_gb:.1f} GiB" if vram_gb else "?"

                        with st.expander(
                            f"{rank}. **{repo_id}**  {fit_icon}  "
                            f"{params_str}  |  {vram_str} VRAM  |  score {score:.2f}"
                        ):
                            c1, c2 = st.columns([2, 1])
                            with c1:
                                st.write(f"Architecture: `{arch}`")
                                st.write(f"Downloads: {downloads:,}  |  Preset: `{preset}`")
                                overrides = item.get("preset_overrides") or {}
                                if overrides:
                                    st.caption(f"Suggested overrides: `{json.dumps(overrides)}`")
                            with c2:
                                override_str = st.text_input(
                                    "Overrides JSON", value=json.dumps(overrides),
                                    key=f"ov_rec_{rank}_{repo_id}",
                                )
                                if st.button("Register", key=f"reg_rec_{rank}_{repo_id}"):
                                    try:
                                        ov = json.loads(override_str) if override_str else {}
                                    except json.JSONDecodeError:
                                        ov = {}
                                    res = mrm_register_from_hf(repo_id, preset, gpu=gpu_id, overrides=ov)
                                    if res:
                                        st.success("Registered!")
                                    else:
                                        st.error("Registration failed")

    # ── Manual Register ─────────────────────────────────────────────────
    with tab_register:
        st.subheader("Manual Registration")
        st.caption("Register any HuggingFace model directly by repo ID.")

        col_a, col_b = st.columns(2)
        with col_a:
            repo_id_manual = st.text_input("HuggingFace Repo ID",
                                           placeholder="e.g. Qwen/Qwen1.5-1.8B-Chat")
            preset_manual  = st.selectbox("Preset", ["small_chat", "7b_awq"])
            gpu_manual     = st.text_input("GPU", value="0")
        with col_b:
            st.markdown("**Preset Descriptions**")
            st.info(
                "**small_chat** — fp16, GPU util 0.40, max_len 2048, LoRA enabled\n\n"
                "**7b_awq** — AWQ 4-bit, GPU util 0.90, max_len 512, LoRA enabled"
            )
            overrides_str = st.text_area(
                "Overrides (JSON, optional)",
                value='{}',
                height=100,
                help="e.g. {\"max_model_len\": 1024, \"gpu_memory_utilization\": 0.7}",
            )

        if st.button("➕ Register Model", type="primary"):
            if not repo_id_manual:
                st.warning("Repo ID is required")
            else:
                try:
                    overrides = json.loads(overrides_str or "{}")
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON in overrides: {e}")
                    overrides = None

                if overrides is not None:
                    with st.spinner(f"Registering {repo_id_manual}…"):
                        res = mrm_register_from_hf(repo_id_manual, preset_manual,
                                                   gpu=gpu_manual, overrides=overrides)
                    if res:
                        st.success(f"Registered! Container: `{res.get('container')}`")
                        st.json(res)
                    else:
                        st.error("Registration failed — check MRM logs.")


# ──────────────────────────────────────────────────────────────────────────────
# Page: Model Registry
# ──────────────────────────────────────────────────────────────────────────────

elif page == "Model Registry":
    st.title("📦 Model Registry")

    col_ref, col_gpu = st.columns([1, 5])
    with col_ref:
        if st.button("🔄 Refresh"):
            st.rerun()

    # GPU summary
    gpu = mrm_gpu_metrics()
    if gpu:
        gpus = gpu.get("gpus") or [gpu]
        gm = gpus[0] if isinstance(gpus, list) and gpus else {}
        total_mib = gm.get("memory_total_mib", 0)
        used_mib  = gm.get("memory_used_mib", 0)
        free_mib  = total_mib - used_mib
        util      = gm.get("utilization_percent", 0)
        with col_gpu:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("GPU Util", f"{util}%")
            c2.metric("VRAM Total", f"{total_mib/1024:.1f} GiB")
            c3.metric("VRAM Used",  f"{used_mib/1024:.1f} GiB")
            c4.metric("VRAM Free",  f"{free_mib/1024:.1f} GiB")

    models = mrm_status_all()
    if not models:
        st.info("No models registered. Go to **HF Hub** to register one.")
        st.stop()

    for m in models:
        base   = m.get("base_model", "?")
        alias  = m.get("model_alias", "")
        state  = m.get("state", "ABSENT")
        running = m.get("running", False)
        gpu_id  = m.get("gpu", "")
        loras   = m.get("active_loras", [])

        state_icon = {"READY": "🟢", "STARTING": "🟡", "STOPPING": "🟡",
                      "STOPPED": "⚫", "ABSENT": "⚫"}.get(state, "⚪")

        with st.expander(f"{state_icon} **{base}**  `{alias}`  — {state}"):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.write(f"GPU: `{gpu_id or '—'}`  |  LoRAs loaded: {len(loras)}")
                if loras:
                    st.caption("LoRAs: " + ", ".join(f"`{l}`" for l in loras))
                st.caption(f"Last used: {m.get('last_used', '—')}")
                st.caption(f"API base: `{m.get('api_base', '—')}`")
            with c2:
                if not running:
                    if st.button("▶ Start", key=f"start_{base}"):
                        with st.spinner(f"Starting {alias}… (may take minutes)"):
                            res = mrm_ensure(base)
                        if res and res.get("state") == "READY":
                            st.success("Ready!")
                            st.rerun()
                        else:
                            st.error(f"Start failed: {res}")
                else:
                    if st.button("⏹ Stop", key=f"stop_{base}"):
                        res = mrm_stop(base)
                        if res:
                            st.success("Stopped")
                            st.rerun()

                if st.button("🗑 Remove", key=f"rem_{base}"):
                    res = mrm_remove(base)
                    if res:
                        st.success("Removed")
                        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Page: Chat
# ──────────────────────────────────────────────────────────────────────────────

elif page == "Chat":
    st.title("💬 Inference Chat")

    with st.sidebar:
        st.subheader("Model")
        live_models = gateway_list_models()
        model_options = [
            m.get("metadata", {}).get("base_model") or m.get("id")
            for m in live_models if m.get("id")
        ] or BASE_MODELS[:5]
        selected_model = st.selectbox("Model", model_options)

        st.subheader("Parameters")
        temp     = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.1)
        max_tok  = st.slider("Max Tokens",   64, 4096, 512, step=64)
        sys_msg  = st.text_area("System prompt (optional)", height=80)

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Send a message…")
    if prompt:
        messages_to_send = []
        if sys_msg:
            messages_to_send.append({"role": "system", "content": sys_msg})
        messages_to_send += st.session_state.messages
        messages_to_send.append({"role": "user", "content": prompt})

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                res = gateway_chat(selected_model, messages_to_send, temp, max_tok)
            if not res:
                st.error("Gateway returned no response.")
                st.stop()
            try:
                content = res["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                st.error(f"Unexpected response: {res}")
                st.stop()
            st.write(content)
            st.session_state.messages.append({"role": "assistant", "content": content})


# ──────────────────────────────────────────────────────────────────────────────
# Page: Training
# ──────────────────────────────────────────────────────────────────────────────

elif page == "Training":
    st.title("🔥 Model Training")

    if st.button("🔄 Refresh"):
        st.rerun()

    tab_train, tab_ds, tab_runs = st.tabs(["Train", "Datasets (ETL)", "Run History"])

    with tab_train:
        datasets_meta = get_datasets_from_cp()
        c1, c2 = st.columns(2)
        selected_ds = None
        with c1:
            if datasets_meta:
                selected_ds = st.selectbox("Dataset", datasets_meta,
                                           format_func=lambda x: x["label"])
                if selected_ds:
                    st.success(f"Selected: `{selected_ds['slug']}`")
            else:
                st.warning("No built datasets. Go to 'Datasets (ETL)' tab first.")
        with c2:
            base_model = st.selectbox("Base Model", BASE_MODELS, index=4)
            col_e, col_l = st.columns(2)
            epochs = col_e.slider("Epochs", 1, 10, 1)
            lr = col_l.select_slider("LR", options=[5e-6, 1e-5, 2e-5, 5e-5, 1e-4], value=2e-5)

        if st.button("🚀 Start Training", type="primary"):
            if not selected_ds:
                st.error("Select a dataset first")
            else:
                payload = {
                    "parent_run_id": selected_ds["run_id"],
                    "base_model":    base_model,
                    "training":      {"epochs": epochs, "learning_rate": lr, "batch_size": 1},
                    "output":        {"lora_base_uri": f"s3://{S3_BUCKET}/loras"},
                }
                res = cp_post_contract("train.qlora.v1", payload)
                if res:
                    st.session_state.selected_training_run_id = res.get("run_id")
                    st.success(f"Training started: {res.get('run_id')}")

    with tab_ds:
        st.subheader("Submit dataset.build.v1 contract")
        target_name  = st.text_input("Target Name", placeholder="e.g. my-dataset-v1")
        dataset_type = st.radio("Type", ["graph", "sft", "prefs", "linear"], horizontal=True)
        max_items    = st.number_input("Max Items", 10, 10000, 500)

        if st.button("🚀 Submit Build Contract"):
            if not target_name:
                st.warning("Target Name is required")
            else:
                payload = {
                    "target_name":  target_name,
                    "dataset_type": dataset_type,
                    "options":      {"max_items": max_items},
                    "output":       {"base_uri": f"s3://{S3_BUCKET}/datasets"},
                }
                res = cp_post_contract("dataset.build.v1", payload)
                if res and res.get("run_id"):
                    st.session_state.last_run_id = res["run_id"]
                    st.success(f"Run Created: {res['run_id']}")

        if st.session_state.last_run_id:
            st.divider()
            if st.button("🔄 Check Progress"):
                run_data = cp_get_run(st.session_state.last_run_id)
                if run_data:
                    st.info(f"State: **{run_data.get('state', 'UNKNOWN')}**")
                    if run_data.get("artifacts"):
                        st.json(run_data["artifacts"])

    with tab_runs:
        runs = cp_get_runs()
        if runs:
            df = pd.DataFrame(runs)
            cols = [c for c in ["id", "state", "type", "created_at"] if c in df.columns]
            st.dataframe(df[cols], use_container_width=True, hide_index=True)
        else:
            st.info("No runs found.")

        st.subheader("Inspect Run")
        run_id_input = st.text_input("Run ID")
        if st.button("Load") and run_id_input:
            data = cp_get_run(run_id_input)
            if data:
                st.json(data)
            else:
                st.error("Run not found")


# ──────────────────────────────────────────────────────────────────────────────
# Page: Monitoring
# ──────────────────────────────────────────────────────────────────────────────

elif page == "Monitoring":
    st.title("📡 System Monitoring")

    ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 4])
    with ctrl1:
        auto_refresh = st.toggle("Auto-refresh", value=False)
    with ctrl2:
        refresh_sec = st.selectbox("s", [5, 10, 30], index=1, label_visibility="collapsed")
    with ctrl3:
        if st.button("↺ Refresh"):
            st.rerun()

    tab_slo, tab_metrics, tab_instances, tab_tracing = st.tabs(
        ["SLO Overview", "Live Metrics", "Instances", "Tracing"]
    )

    with tab_slo:
        slo         = _http_get_silent(f"{GATEWAY_URL}/v1/slo")
        router_data = _http_get_silent(f"{GATEWAY_URL}/v1/router/metrics")

        if slo:
            p50      = slo.get("p50_ms")
            p95      = slo.get("p95_ms")
            p99      = slo.get("p99_ms")
            err_rate = slo.get("error_rate", 0.0)
            samples  = slo.get("samples", 0)

            if err_rate >= 0.05 or (p99 and p99 >= 5000):
                badge, color = "🔴 FAIL", "red"
            elif err_rate >= 0.01 or (p99 and p99 >= 2000):
                badge, color = "🟡 DEGRADED", "orange"
            else:
                badge, color = "🟢 OK", "green"

            st.markdown(f"### System SLO: **:{color}[{badge}]**")
            st.caption(f"Last {samples} requests (in-memory sliding window)")

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1: _slo_badge("P50", p50, 300, 1000, " ms")
            with m2: _slo_badge("P95", p95, 1000, 3000, " ms")
            with m3: _slo_badge("P99", p99, 2000, 5000, " ms")
            with m4:
                _slo_badge("Error Rate", err_rate * 100, 1.0, 5.0, "%")
            with m5:
                slo_ok = (router_data or {}).get("slo", {}).get("slo_ok", True)
                st.metric("SLO Check", "🟢 Passing" if slo_ok else "🔴 Failing")
        else:
            st.warning("Could not reach gateway /v1/slo")

        if router_data:
            for v in (router_data.get("slo") or {}).get("violations", []):
                st.error(f"SLO Violation: {v}")

    with tab_metrics:
        st.subheader("Latency Percentiles (30 min)")
        lat_frames = []
        for label, q in [
            ("P50", "histogram_quantile(0.50,sum(rate(gateway_request_latency_seconds_bucket[2m]))by(le))*1000"),
            ("P95", "histogram_quantile(0.95,sum(rate(gateway_request_latency_seconds_bucket[2m]))by(le))*1000"),
            ("P99", "histogram_quantile(0.99,sum(rate(gateway_request_latency_seconds_bucket[2m]))by(le))*1000"),
        ]:
            df = _prom_range(q)
            if not df.empty:
                df.columns = [label]
                lat_frames.append(df)
        if lat_frames:
            combined = lat_frames[0]
            for f in lat_frames[1:]:
                combined = combined.join(f, how="outer")
            st.line_chart(combined)
        else:
            st.info("No latency data yet — send some requests first.")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Request Rate (req/s)")
            rr = _prom_range("sum(rate(gateway_requests_total[2m]))")
            if not rr.empty:
                st.line_chart(rr)
            else:
                st.info("No data")
        with col_b:
            st.subheader("Error Rate (errors/s)")
            er = _prom_range("sum(rate(gateway_errors_total[5m]))")
            if not er.empty:
                st.line_chart(er)
            else:
                st.info("No data")

        st.subheader("In-Flight Requests")
        inf = _prom_range("gateway_in_flight_requests", step="15s")
        if not inf.empty:
            st.line_chart(inf)
        else:
            st.info("No data")

    with tab_instances:
        router_data = _http_get_silent(f"{GATEWAY_URL}/v1/router/metrics")
        if not router_data:
            st.warning("Cannot reach /v1/router/metrics")
        else:
            st.caption(f"Strategy: **{router_data.get('strategy', 'unknown')}**")
            instances = router_data.get("instances", {})
            if instances:
                rows = []
                for url, m in instances.items():
                    errors   = m.get("errors", 0)
                    reqs     = m.get("requests", 0)
                    err_rate = errors / reqs if reqs > 0 else 0.0
                    cb_raw   = m.get("circuit_state", "CLOSED")
                    anom     = m.get("anomalous", False)
                    health   = "🔴 OPEN" if cb_raw == "OPEN" else (
                               "🟡 DEGRADED" if cb_raw == "HALF_OPEN" or anom else "🟢 HEALTHY")
                    rows.append({
                        "Instance":       url,
                        "Health":         health,
                        "Requests":       reqs,
                        "Errors":         errors,
                        "Err %":          f"{err_rate*100:.1f}%",
                        "Avg Lat ms":     round(m.get("avg_latency_ms", 0), 1),
                        "EWMA ms":        round(m.get("ewma_latency_ms", 0), 1),
                        "Inflight":       m.get("inflight", 0),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No instances tracked yet.")

            st.markdown("---")
            st.subheader("Scheduler Nodes")
            nodes_alive = _prom_query("scheduler_nodes_alive")
            hb = _prom_range("sum(rate(scheduler_heartbeats_total[1m]))", minutes=15, step="15s")
            n1, n2 = st.columns(2)
            with n1:
                badge = "🟢" if (nodes_alive or 0) > 0 else "🔴"
                st.metric("Nodes Alive", f"{badge} {int(nodes_alive or 0)}")
            with n2:
                st.caption("Heartbeat rate")
                if not hb.empty:
                    st.line_chart(hb)
                else:
                    st.info("No data")

    with tab_tracing:
        st.subheader("Distributed Traces — Jaeger")
        st.markdown(f"**[→ Open Jaeger UI]({JAEGER_UI_URL})**")
        st.markdown(
            "Services instrumented: `gateway`, `scheduler`, `node-agent`\n\n"
            "Useful searches: Service=`gateway`, Op=`POST /v1/chat/completions`"
        )
        st.markdown("---")
        st.subheader("Recent Traces (last 5)")
        try:
            r = requests.get(
                f"{JAEGER_UI_URL}/api/traces",
                params={"service": "gateway", "limit": 5, "lookback": "1h"},
                timeout=5,
            )
            if r.status_code == 200:
                data = r.json().get("data", [])
                if data:
                    trace = data[0]
                    spans = trace.get("spans", [])
                    st.caption(f"Trace `{trace.get('traceID','')}` — {len(spans)} spans")
                    rows = [{"Operation": s.get("operationName",""),
                             "Duration ms": round(s.get("duration",0)/1000, 2),
                             "Service": (s.get("process") or {}).get("serviceName", "")}
                            for s in spans[:10]]
                    if rows:
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No recent traces. Send a request first.")
        except Exception as e:
            st.warning(f"Cannot reach Jaeger: {e}")

    if auto_refresh:
        time.sleep(refresh_sec)
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Page: Orchestration
# ──────────────────────────────────────────────────────────────────────────────

elif page == "Orchestration":
    st.title("🎛️ Orchestration Dashboard")

    if st.button("🔄 Refresh"):
        st.rerun()

    runs = cp_get_runs()
    if runs:
        df = pd.DataFrame(runs)
        cols = [c for c in ["id", "state", "type", "created_at"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
    else:
        st.info("No runs found.")

    st.subheader("Inspect Run")
    run_id_input = st.text_input("Run ID")
    if st.button("Load") and run_id_input:
        data = cp_get_run(run_id_input)
        if data:
            st.json(data)
        else:
            st.error("Run not found")


# ──────────────────────────────────────────────────────────────────────────────
# Page: Prompt Studio
# ──────────────────────────────────────────────────────────────────────────────

elif page == "Prompt Studio":
    st.title("🎨 Prompt Studio")

    prompts = _http_get(f"{CP_URL}/prompts/", timeout=10)
    prompts = prompts if isinstance(prompts, list) else []

    with st.sidebar:
        st.header("Library")
        new_pid = st.text_input("New Prompt ID")
        if st.button("➕ Create") and new_pid:
            requests.post(f"{CP_URL}/prompts/",
                          params={"id": new_pid, "description": "Created via UI"})
            st.rerun()
        st.divider()
        ids = [p.get("id") for p in prompts if isinstance(p, dict) and p.get("id")]
        selected_pid = st.radio("Select Family", ids, index=0 if ids else None)

    if not selected_pid:
        st.info("Create or select a prompt family.")
        st.stop()

    st.header(f"`{selected_pid}`")
    versions = _http_get(f"{CP_URL}/prompts/{selected_pid}/versions", timeout=10) or []

    tab_edit, tab_history, tab_test = st.tabs(["Editor", "History", "Playground"])

    with tab_edit:
        default_tmpl = versions[0].get("template") if versions else "You are an AI assistant."
        last_tag     = versions[0].get("version_tag") if versions else "v0.0"

        new_template = st.text_area("Template (Jinja2)", value=default_tmpl, height=400)
        vars_found   = sorted(set(re.findall(r"\{\{\s*(\w+)\s*\}\}", new_template)))
        if vars_found:
            st.info(f"Variables: {', '.join(vars_found)}")

        col1, col2 = st.columns([1, 2])
        with col1:
            try:
                v_num = int(last_tag.split("v")[-1].replace(".", ""))
                suggested_ver = f"v1.{v_num + 1}"
            except Exception:
                suggested_ver = "v1.0"
            new_tag = st.text_input("Version Tag", value=suggested_ver)
        with col2:
            commit_msg = st.text_input("Commit Message", placeholder="What changed?")

        if st.button("💾 Save Version", type="primary"):
            payload = {"prompt_id": selected_pid, "version_tag": new_tag,
                       "template": new_template, "commit_message": commit_msg or "UI Update"}
            res = _http_post(f"{CP_URL}/prompts/version", json_body=payload, timeout=30)
            if res is not None:
                st.success("Saved!")
                time.sleep(0.5)
                st.rerun()

    with tab_history:
        if not versions:
            st.info("No versions yet.")
        for v in versions:
            tag = v.get("version_tag", "?")
            created_at = (v.get("created_at") or "")[:16]
            with st.expander(f"{tag} | {created_at} | {v.get('commit_message','')}"):
                st.code(v.get("template", ""), language="jinja2")

    with tab_test:
        vars_found = sorted(set(re.findall(r"\{\{\s*(\w+)\s*\}\}", new_template)))
        inputs = {}
        for var in vars_found:
            inputs[var] = st.text_area(f"Input for `{var}`",
                                       height=200 if "conversation" in var else None)

        if st.button("👁️ Render"):
            try:
                rendered = Template(new_template).render(**inputs)
                st.session_state["preview_prompt"] = rendered
                st.text_area("Rendered Prompt", value=rendered, height=300)
            except Exception as e:
                st.error(f"Jinja2 error: {e}")

        if st.button("🚀 Run Inference"):
            preview = st.session_state.get("preview_prompt")
            if not preview:
                st.warning("Render first.")
            else:
                res = gateway_chat(BASE_MODELS[0],
                                   [{"role": "user", "content": preview}], 0.7, 500)
                if res:
                    st.success(res["choices"][0]["message"]["content"])
