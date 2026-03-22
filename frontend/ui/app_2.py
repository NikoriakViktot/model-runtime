import json
import os
import re
import time

import pandas as pd
import requests
import streamlit as st
from jinja2 import Template

CP_URL = os.getenv("CONTROL_PLANE_URL", "http://control_plane:8004")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway:8080")
S3_BUCKET = os.getenv("AWS_BUCKET_NAME", "model-runtime-artifacts")
MLFLOW_UI_URL = "/mlflow/"

BASE_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-Nemo-12B-Instruct-v1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-small-8k-instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "01-ai/Yi-1.5-6B-Chat",
    "01-ai/Yi-1.5-9B-Chat",
    "01-ai/Yi-1.5-34B-Chat",
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    "deepseek-ai/deepseek-llm-7b-chat",
    "deepseek-ai/deepseek-math-7b-instruct",
    "teknium/OpenHermes-2.5-Mistral-7B",
    "HuggingFaceH4/zephyr-7b-beta",
    "openchat/openchat-3.6-8b-20240522",
    "NousResearch/Hermes-3-Llama-3.1-8B",
]

st.set_page_config(page_title="AI Control Plane", layout="wide", page_icon="🧠")


def init_session_state():
    defaults = {
        "selected_dataset_run_id": None,
        "selected_training_run_id": None,
        "messages": [],
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


init_session_state()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _http_get(url: str, *, params: dict | None = None, timeout: int = 10) -> dict | list | None:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"HTTP GET failed: {e}")
        return None


def _http_post(url: str, *, json_body: dict | None = None, files=None, timeout: int = 30) -> dict | None:
    try:
        r = requests.post(url, json=json_body, files=files, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"HTTP POST failed: {e}")
        return None


def _http_delete(url: str, *, timeout: int = 30) -> bool:
    try:
        r = requests.delete(url, timeout=timeout)
        r.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"HTTP DELETE failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Control Plane helpers
# ---------------------------------------------------------------------------


def cp_post_contract(contract_type: str, payload: dict) -> dict | None:
    envelope = {"type": contract_type, "spec_version": "v1", "payload": payload}
    return _http_post(f"{CP_URL}/contracts", json_body=envelope, timeout=10)


def cp_get_runs() -> list:
    data = _http_get(f"{CP_URL}/runs", timeout=5)
    return data if isinstance(data, list) else []


def cp_get_run(run_id: str) -> dict | None:
    data = _http_get(f"{CP_URL}/runs/{run_id}", timeout=10)
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# Gateway helpers (inference)
# ---------------------------------------------------------------------------


def gateway_chat(model: str, messages: list[dict], temperature: float, max_tokens: int) -> dict | None:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return _http_post(f"{GATEWAY_URL}/v1/chat/completions", json_body=payload, timeout=120)


def gateway_list_models() -> list[dict]:
    data = _http_get(f"{GATEWAY_URL}/v1/models", timeout=5)
    if isinstance(data, dict):
        return data.get("data", [])
    return []


# ---------------------------------------------------------------------------
# Dataset helpers (from Control Plane run history)
# ---------------------------------------------------------------------------


def get_datasets_from_cp() -> list[dict]:
    runs = _http_get(f"{CP_URL}/runs", timeout=10)
    if not isinstance(runs, list):
        return []

    out: list[dict] = []
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

        target_name = contract_payload.get("target_name", "Unknown")
        run_id = run.get("id", "")
        created_at = run.get("created_at", "")

        out.append(
            {
                "label": f"{target_name} | {created_at[:16]}",
                "uri": uri,
                "slug": target_name,
                "run_id": run_id,
            }
        )

    return sorted(out, key=lambda x: x["run_id"], reverse=True)


# ---------------------------------------------------------------------------
# Prompt Studio
# ---------------------------------------------------------------------------


def render_prompt_studio() -> None:
    st.title("🎨 Prompt Engineering Studio")
    st.markdown("Model Behavior Management Center. Create, edit, and test prompts.")

    prompts = _http_get(f"{CP_URL}/prompts/", timeout=10)
    prompts = prompts if isinstance(prompts, list) else []

    with st.sidebar:
        st.header("Library")
        new_pid = st.text_input("New Prompt ID (e.g. system.default)")
        if st.button("➕ Create New Family") and new_pid:
            requests.post(f"{CP_URL}/prompts/", params={"id": new_pid, "description": "Created via UI"})
            st.rerun()

        st.divider()

        ids = [p.get("id") for p in prompts if isinstance(p, dict) and p.get("id")]
        selected_pid = st.radio("Select Prompt Family", ids, index=0 if ids else None)

    if not selected_pid:
        return

    st.header(f"Editing: `{selected_pid}`")

    versions = _http_get(f"{CP_URL}/prompts/{selected_pid}/versions", timeout=10)
    versions = versions if isinstance(versions, list) else []

    tab_edit, tab_history, tab_test = st.tabs(["✏️ Editor", "📜 History", "🧪 Playground"])

    with tab_edit:
        default_tmpl = versions[0].get("template") if versions else "You are an AI assistant. Context: {{ conversation }}"
        last_tag = versions[0].get("version_tag") if versions else "v0.0"

        st.caption("Use Jinja2 for variables: `{{ variable_name }}`")
        new_template = st.text_area("Template Text", value=default_tmpl, height=400)

        vars_found = sorted(set(re.findall(r"\{\{\s*(\w+)\s*\}\}", new_template)))
        if vars_found:
            st.info(f"Found variables: {', '.join(vars_found)}")

        col1, col2 = st.columns([1, 2])

        with col1:
            try:
                v_num = int(last_tag.split("v")[-1].replace(".", ""))
                suggested_ver = f"v1.{v_num + 1}"
            except Exception:
                suggested_ver = "v1.0"
            new_tag = st.text_input("Version Tag", value=suggested_ver)

        with col2:
            commit_msg = st.text_input("Commit Message", placeholder="What has changed in this version?")

        if st.button("💾 Save Version", type="primary"):
            payload = {
                "prompt_id": selected_pid,
                "version_tag": new_tag,
                "template": new_template,
                "commit_message": commit_msg or "UI Update",
            }
            res = _http_post(f"{CP_URL}/prompts/version", json_body=payload, timeout=30)
            if res is not None:
                st.success("Saved!")
                time.sleep(0.5)
                st.rerun()

    with tab_history:
        if not versions:
            st.info("History is empty.")
        for v in versions:
            tag = v.get("version_tag", "?")
            created_at = (v.get("created_at") or "")[:16]
            msg = v.get("commit_message", "")
            tmpl = v.get("template", "")
            with st.expander(f"{tag} | {created_at} | {msg}"):
                st.code(tmpl, language="jinja2")

    with tab_test:
        st.subheader("Test Rendering & Inference")

        vars_found = sorted(set(re.findall(r"\{\{\s*(\w+)\s*\}\}", new_template)))

        if not vars_found:
            st.info("No variables in template. You can run the test immediately.")
            return

        inputs: dict[str, str] = {}
        for var in vars_found:
            height = 200 if ("conversation" in var or "text" in var) else None
            inputs[var] = st.text_area(f"Input for: {var}", height=height)

        if st.button("👁️ Render Preview"):
            try:
                t = Template(new_template)
                rendered_prompt = t.render(**inputs)
                st.session_state["preview_prompt"] = rendered_prompt
                st.text_area("Final Rendered Prompt", value=rendered_prompt, height=300)
            except Exception as e:
                st.error(f"Jinja2 Error: {e}")

        if st.button("🚀 Run Inference"):
            if "preview_prompt" not in st.session_state:
                st.warning("First press -> 'Render Preview'")
                return

            payload = {
                "model": BASE_MODELS[0],
                "messages": [{"role": "user", "content": st.session_state["preview_prompt"]}],
                "temperature": 0.7,
                "max_tokens": 500,
            }

            res = gateway_chat(
                payload["model"],
                payload["messages"],
                payload["temperature"],
                payload["max_tokens"],
            )
            if res:
                st.success("Model Response:")
                st.write(res["choices"][0]["message"]["content"])


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

st.sidebar.title("🧬 AI Runtime")
page = st.sidebar.radio(
    "Navigation",
    ["Orchestration", "Prompt Studio", "Datasets (ETL)", "Training", "Chat"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"📊 **[Open MLflow UI]({MLFLOW_UI_URL})**", unsafe_allow_html=True)

if "db_id" not in st.session_state:
    st.session_state.db_id = None
if "target_name" not in st.session_state:
    st.session_state.target_name = None
if "last_run_id" not in st.session_state:
    st.session_state.last_run_id = None


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

if page == "Orchestration":
    st.title("🎛️ Orchestration Dashboard")
    st.markdown("Monitor the state of the AI Kernel.")

    c1, _ = st.columns([1, 5])
    with c1:
        if st.button("🔄 Refresh"):
            st.rerun()

    runs = cp_get_runs()
    if runs:
        df = pd.DataFrame(runs)
        cols = [c for c in ["id", "state", "type", "created_at"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)

    st.subheader("🔎 Inspect Run")
    run_id_input = st.text_input("Run ID")
    if st.button("Load Run"):
        if not run_id_input:
            st.warning("Provide Run ID")
        else:
            data = cp_get_run(run_id_input)
            if not data:
                st.error("Run not found")
            else:
                st.json(data)
    else:
        st.info("No runs found.")


elif page == "Prompt Studio":
    render_prompt_studio()


elif page == "Datasets (ETL)":
    st.title("📦 Dataset Construction")

    st.info(
        "Dataset construction requires the **API Dispatcher** service. "
        "Submit a `dataset.build.v1` contract directly via the Control Plane below, "
        "or implement the API Dispatcher to enable full ETL functionality."
    )

    st.subheader("Submit dataset.build.v1 contract")

    target_name = st.text_input("Target Name", placeholder="e.g. my-dataset-v1")
    dataset_type = st.radio("Dataset Type", ["graph", "sft", "prefs", "linear"], horizontal=True)
    max_items = st.number_input("Max Items", 10, 10000, 500)

    if st.button("🚀 Submit Build Contract"):
        if not target_name:
            st.warning("Target Name is required")
        else:
            payload = {
                "target_name": target_name,
                "dataset_type": dataset_type,
                "options": {"max_items": max_items},
                "output": {"base_uri": f"s3://{S3_BUCKET}/datasets"},
            }
            res = cp_post_contract("dataset.build.v1", payload)
            if res and res.get("run_id"):
                st.session_state.last_run_id = res["run_id"]
                st.success(f"Run Created: {res['run_id']}")

    if st.session_state.last_run_id:
        st.divider()
        st.subheader("Build Progress")
        if st.button("🔄 Check Status"):
            run_data = cp_get_run(st.session_state.last_run_id)
            if run_data:
                st.info(f"State: **{run_data.get('state', 'UNKNOWN')}**")
                artifacts = run_data.get("artifacts") or {}
                if artifacts:
                    st.json(artifacts)


elif page == "Training":
    st.title("🔥 Model Training")

    c_refresh, _ = st.columns([1, 4])
    with c_refresh:
        if st.button("🔄 Refresh"):
            st.rerun()

    datasets_meta = get_datasets_from_cp()

    c1, c2 = st.columns(2)

    selected_ds = None
    with c1:
        if datasets_meta:
            selected_ds = st.selectbox(
                "1. Select Dataset",
                options=datasets_meta,
                format_func=lambda x: x["label"],
            )
            if selected_ds:
                st.success(f"Selected: `{selected_ds['slug']}`")
                st.caption(f"Run ID: `{selected_ds['run_id']}`")
        else:
            st.warning("No built datasets found. Go to 'Datasets (ETL)' to build one.")

    with c2:
        base_model = st.selectbox("2. Base Model", BASE_MODELS, index=3)
        col_e, col_l = st.columns(2)
        epochs = col_e.slider("Epochs", 1, 10, 1)
        lr = col_l.select_slider("Learning Rate", options=[5e-6, 1e-5, 2e-5, 5e-5, 1e-4], value=2e-5)

    st.divider()

    if st.button("Start Training", type="primary"):
        if not selected_ds:
            st.error("Select a dataset first")
        else:
            payload = {
                "parent_run_id": selected_ds["run_id"],
                "base_model": base_model,
                "training": {
                    "epochs": epochs,
                    "learning_rate": lr,
                    "batch_size": 1,
                },
                "output": {
                    "lora_base_uri": f"s3://{S3_BUCKET}/loras",
                },
            }
            res = cp_post_contract("train.qlora.v1", payload)
            if res:
                st.session_state.selected_training_run_id = res.get("run_id")
                st.success(f"Training started: {res.get('run_id')}")
                st.info("Check 'Orchestration' tab for progress.")


elif page == "Chat":
    st.title("💬 Inference Chat")

    with st.sidebar:
        st.subheader("Model Selection")

        # List live models from Gateway
        live_models = gateway_list_models()
        model_options = [m.get("id") for m in live_models if m.get("id")] or BASE_MODELS[:5]

        selected_model = st.selectbox("Model", model_options)

        st.subheader("Inference Params")
        temp = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.1)
        max_tok = st.slider("Max Tokens", 64, 4096, 512, step=64)

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Send a message...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            res = gateway_chat(selected_model, st.session_state.messages, temp, max_tok)
            if not res:
                st.error("Gateway returned no response.")
                st.stop()

            try:
                content = res["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                st.error(f"Unexpected response shape: {res}")
                st.stop()

            st.write(content)
            st.session_state.messages.append({"role": "assistant", "content": content})
