import json
import os
import re
import time

import pandas as pd
import requests
import streamlit as st
from jinja2 import Template

CP_URL = os.getenv("CONTROL_PLANE_URL", "http://control_plane:8004")
DISPATCHER_URL = os.getenv("API_DISPATCHER_URL", "http://api_dispatcher:8005")
S3_BUCKET = os.getenv("AWS_BUCKET_NAME", "epitaphs-work-dir")
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


def cp_post_contract(contract_type: str, payload: dict) -> dict | None:
    envelope = {"type": contract_type, "spec_version": "v1", "payload": payload}
    return _http_post(f"{CP_URL}/contracts", json_body=envelope, timeout=10)


def cp_get_runs() -> list:
    data = _http_get(f"{CP_URL}/runs", timeout=5)
    return data if isinstance(data, list) else []


def cp_get_run(run_id: str) -> dict | None:
    data = _http_get(f"{CP_URL}/runs/{run_id}", timeout=10)
    return data if isinstance(data, dict) else None


def dispatcher_upload_db(file_bytes: bytes, filename: str) -> dict | None:
    files = {"file": (filename, file_bytes, "application/octet-stream")}
    return _http_post(f"{DISPATCHER_URL}/dispatch/exp/upload", files=files, timeout=60)


def dispatcher_get_targets(db_id: str) -> list:
    data = _http_get(f"{DISPATCHER_URL}/dispatch/etl/targets", params={"db_id": db_id}, timeout=10)
    return data if isinstance(data, list) else []


def dispatcher_materialize(run_id: str):
    return _http_post(
        f"{DISPATCHER_URL}/dispatch/artifacts/materialize/{run_id}",
        timeout=30,
    )


def dispatcher_preview(run_id: str, limit: int = 5):
    return _http_get(
        f"{DISPATCHER_URL}/dispatch/preview",
        params={"run_id": run_id, "limit": limit},
        timeout=30,
    )

def select_dataset():
    datasets = get_datasets_from_cp()
    if not datasets:
        st.info("No datasets available")
        return None

    selected = st.selectbox(
        "Select dataset",
        options=datasets,
        format_func=lambda d: d["label"],
        index=0,
    )

    if selected:
        st.session_state.selected_dataset_run_id = selected["run_id"]
        return selected

    return None

def render_dataset_preview(run_id: str):
    st.warning("Preview uses local execution cache (dev-only)")

    if st.button("Materialize + Preview"):
        with st.spinner("Materializing dataset..."):
            dispatcher_materialize(run_id)

        with st.spinner("Loading preview..."):
            data = dispatcher_preview(run_id, limit=5)

        if not data:
            st.error("Preview failed")
            return

        for i, sample in enumerate(data.get("samples", [])):
            with st.expander(f"Sample #{i+1}"):
                for msg in sample["messages"]:
                    st.chat_message(msg["role"]).write(msg["content"])


def dispatcher_delete_artifacts(run_id: str) -> bool:
    return _http_delete(f"{DISPATCHER_URL}/dispatch/artifacts/{run_id}", timeout=30)


def dispatcher_list_trained_models() -> list:
    data = _http_get(f"{DISPATCHER_URL}/dispatch/models/trained", timeout=10)
    return data if isinstance(data, list) else []


def dispatcher_load_adapter(lora_name: str, lora_path: str) -> dict | None:
    payload = {"lora_name": lora_name, "lora_path": lora_path}
    return _http_post(f"{DISPATCHER_URL}/dispatch/deploy/load", json_body=payload, timeout=120)


def dispatcher_chat(model: str, messages: list[dict], temperature: float, max_tokens: int) -> dict | None:
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    return _http_post(f"{DISPATCHER_URL}/dispatch/chat", json_body=payload, timeout=60)


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


def render_prompt_studio() -> None:
    st.title("🎨 Prompt Engineering Studio")
    st.markdown("Model Behavior Management Center. Create, edit, and test prompts.")

    prompts = _http_get(f"{CP_URL}/prompts/", timeout=10)
    prompts = prompts if isinstance(prompts, list) else []

    with st.sidebar:
        st.header("Library")
        new_pid = st.text_input("New Prompt ID (e.g. etl.gen_profile)")
        if st.button("➕ Create New Family") and new_pid:
            _http_post(f"{CP_URL}/prompts/", timeout=10)
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
            st.info("There are no variables in the template. You can run the test immediately.")
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

        if st.button("🚀 Run Inference (vLLM)"):
            if "preview_prompt" not in st.session_state:
                st.warning("First press -> 'Render Preview'")
                return

            payload = {
                "model": "qwen-base",
                "messages": [{"role": "user", "content": st.session_state["preview_prompt"]}],
                "temperature": 0.7,
                "max_tokens": 500,
            }

            try:
                r = requests.post("http://litellm:4000/chat/completions", json=payload, timeout=60)
                if r.status_code == 200:
                    st.success("Model Response:")
                    st.write(r.json()["choices"][0]["message"]["content"])
                else:
                    st.error(f"LLM Error: {r.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")


def start_training(dataset):
    st.divider()
    st.subheader("Training")

    base_model = st.selectbox("Base model", BASE_MODELS)
    epochs = st.slider("Epochs", 1, 5, 1)
    lr = st.select_slider(
        "Learning rate",
        options=[5e-6, 1e-5, 2e-5, 5e-5],
        value=2e-5,
    )

    if st.button("Start training", type="primary"):
        payload = {
            "parent_run_id": dataset["run_id"],
            "base_model": base_model,
            "dataset": {"uri": dataset["uri"]},
            "training": {
                "epochs": epochs,
                "learning_rate": lr,
                "batch_size": 1,
            },
            "output": {
                "lora_base_uri": f"s3://{S3_BUCKET}/loras"
            },
        }

        res = cp_post_contract("train.qlora.v1", payload)

        if not res:
            st.error("Failed to submit training")
            return

        st.session_state.selected_training_run_id = res["run_id"]
        st.success(f"Training run created: {res['run_id']}")

def render_orchestration():
    runs = cp_get_runs()
    if not runs:
        st.info("No runs")
        return

    df = pd.DataFrame(runs)
    st.dataframe(
        df[["id", "state", "type", "created_at"]],
        use_container_width=True,
        hide_index=True,
    )


st.sidebar.title("🧬 Control Plane")
page = st.sidebar.radio(
    "Navigation",
    ["Orchestration", "Prompt Studio", "Datasets (ETL)", "Training", "Evaluation", "Deploy", "Chat"],
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"📊 **[Open MLflow UI]({MLFLOW_UI_URL})**", unsafe_allow_html=True)

if "db_id" not in st.session_state:
    st.session_state.db_id = None
if "target_name" not in st.session_state:
    st.session_state.target_name = None
if "last_run_id" not in st.session_state:
    st.session_state.last_run_id = None


if page == "Orchestration":
    st.title("🎛️ Orchestration Dashboard")
    st.markdown("Monitor the state of the AI Kernel.")

    runs = cp_get_runs()
    if runs:
        df = pd.DataFrame(runs)
        cols = [c for c in ["id", "state", "type", "created_at"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
    else:
        st.info("No runs found.")


elif page == "Prompt Studio":
    render_prompt_studio()


elif page == "Datasets (ETL)":
    st.title("📦 Dataset Construction")

    st.subheader("1. Source Data")
    uploaded = st.file_uploader("Upload SQLite DB (.db)", type=["db"])
    if uploaded:
        with st.spinner("Uploading..."):
            meta = dispatcher_upload_db(uploaded.getvalue(), uploaded.name)
        if meta and meta.get("file_id"):
            st.session_state.db_id = meta["file_id"]
            st.success(f"Uploaded: {st.session_state.db_id}")

    if st.session_state.db_id:
        st.caption(f"DB ID: {st.session_state.db_id}")
    else:
        st.info("Please upload a database first.")
        st.stop()

    st.subheader("2. Configuration")
    targets = dispatcher_get_targets(st.session_state.db_id)
    if not targets:
        st.warning("No targets found in DB.")
        st.stop()

    t_map = {f"{t['epitaph']} ({t['cnt']} rows)": t["epitaph"] for t in targets if "epitaph" in t and "cnt" in t}
    sel = st.selectbox("Select Target Persona", list(t_map.keys()))
    st.session_state.target_name = t_map[sel]

    ds_type = st.radio("Dataset Type", ["graph", "sft", "prefs", "linear"], horizontal=True)
    prompt_id = "etl.profile_generator"
    prompt_ver = "latest"

    if ds_type == "linear":
        with st.expander("📝 Prompt Configuration", expanded=True):
            prompt_id = st.text_input("Prompt Family ID", value="etl.profile_generator")
            prompt_ver = st.text_input("Version Tag", value="latest")

    with st.expander("Advanced Options"):
        gen_rejected = st.checkbox("Generate Rejected Samples", value=(ds_type == "prefs"))
        max_items = st.number_input("Max Items", 10, 10000, 500)

    if st.button("🚀 Submit Build Contract"):
        payload = {
            "db_id": st.session_state.db_id,
            "target_name": st.session_state.target_name,
            "dataset_type": ds_type,
            "options": {
                "generate_rejected": gen_rejected,
                "prompt_id": prompt_id,
                "prompt_version": prompt_ver,
                "rejected_max_items": max_items,
            },
            "output": {"base_uri": f"s3://{S3_BUCKET}/datasets"},
        }

        res = cp_post_contract("dataset.build.v1", payload)
        if res and res.get("run_id"):
            st.session_state.last_run_id = res["run_id"]
            st.success(f"Run Created! ID: {res['run_id']}")
            if "state" in res:
                st.info(f"State: {res['state']}")

    if st.session_state.last_run_id:
        st.divider()
        st.subheader("👀 Build Progress & Preview")
        run_id = st.session_state.last_run_id

        if st.button("🔄 Check Status"):
            run_data = cp_get_run(run_id)
            if not run_data:
                st.stop()

            state = run_data.get("state", "UNKNOWN")
            st.info(f"Current State: **{state}**")

            if state in ("DATASET_READY", "DONE"):
                artifacts = run_data.get("artifacts") or {}

                if "stats" in artifacts:
                    st.json(artifacts["stats"])

                if "preview" in artifacts:
                    st.markdown("### 🎲 Dataset Samples (Random 5)")

                    for i, sample in enumerate(artifacts["preview"]):
                        with st.expander(f"Sample #{i + 1}", expanded=False):
                            for msg in sample.get("messages", []):
                                role = msg.get("role")
                                content = msg.get("content", "")
                                if role == "system":
                                    st.warning(f"**System**: {content[:300]}...")
                                elif role in ("user", "assistant"):
                                    st.chat_message(role).write(content)
                            st.caption("Raw JSON:")
                            st.code(json.dumps(sample, ensure_ascii=False), language="json")
                else:
                    st.warning("No preview data available in artifacts.")


elif page == "Training":
    st.title("🔥 Model Training")

    c_refresh, _ = st.columns([1, 4])
    with c_refresh:
        if st.button("🔄 Refresh List"):
            st.rerun()

    datasets_meta = get_datasets_from_cp()
    c1, c2 = st.columns(2)

    selected_ds = None
    with c1:
        if datasets_meta:
            selected_ds = st.selectbox(
                "1. Select Target Dataset (from History)",
                options=datasets_meta,
                format_func=lambda x: x["label"],
            )

            if selected_ds:
                st.success(f"Selected: `{selected_ds['slug']}`")
                st.caption(f"Run ID: `{selected_ds['run_id']}`")

                with st.expander("👀 Preview Dataset Content", expanded=False):
                    if st.button("Load Preview"):
                        with st.spinner("Preparing dataset..."):
                            dispatcher_materialize(selected_ds["run_id"])
                            data = dispatcher_preview(selected_ds["run_id"], limit=3)

                        st.caption(f"File: {data.get('file', '')}")
                        for i, sample in enumerate(data.get("samples", [])):
                            with st.expander(f"Sample #{i + 1}"):
                                for msg in sample.get("messages", []):
                                    role = msg.get("role", "user")
                                    content = msg.get("content", "")
                                    st.chat_message(role).write(content)
        else:
            st.warning("No built datasets found. Go to 'Datasets (ETL)' to build one.")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("⬇️ Materialize Dataset"):
            if not selected_ds:
                st.warning("Select a dataset first")
                st.stop()
            if dispatcher_materialize(selected_ds["run_id"]):
                st.success("Dataset cached locally")

    with col_b:
        if st.button("🗑️ Delete Local Artifacts"):
            if not selected_ds:
                st.warning("Select a dataset first")
                st.stop()
            if dispatcher_delete_artifacts(selected_ds["run_id"]):
                st.success("Artifacts deleted")

    with c2:
        default_idx = 3 if len(BASE_MODELS) > 3 else 0
        base_model = st.selectbox("2. Base Model (Hugging Face)", BASE_MODELS, index=default_idx)

        st.divider()
        method = st.radio("Method", ["SFT (QLoRA)", "DPO", "SimPO"], horizontal=True)

        col_e, col_l = st.columns(2)
        epochs = col_e.slider("Epochs", 1, 10, 1)
        lr = col_l.select_slider("Learning Rate", options=[5e-6, 1e-5, 2e-5, 5e-5, 1e-4], value=2e-5)

    st.divider()

    if st.button("Start training", type="primary"):
        run_id = st.session_state.selected_dataset_run_id
        if not run_id:
            st.error("Select dataset first")
            st.stop()

        payload = {
            "parent_run_id": run_id,
            "base_model": base_model,
            "training": {
                "epochs": epochs,
                "learning_rate": lr,
                "batch_size": 1,
            },
            "output": {
                "lora_base_uri": f"s3://{S3_BUCKET}/loras"
            },
        }

        res = cp_post_contract("train.qlora.v1", payload)

        if not res:
            st.error("Training submission failed")
            st.stop()

        st.session_state.selected_training_run_id = res["run_id"]
        st.success(f"Training started: {res['run_id']}")
        st.info("Check 'Orchestration' tab for progress.")


elif page == "Chat":
    st.title("💬 Inference Chat")

    with st.sidebar:
        st.subheader("Model Selection")

        models_data = dispatcher_list_trained_models()
        options = ["qwen-base"]
        model_map = {"qwen-base": "qwen-base"}
        paths_map: dict[str, str] = {}

        for m in models_data:
            metrics = m.get("metrics") or {}
            loss = metrics.get("train_loss", "N/A")
            if isinstance(loss, float):
                loss = f"{loss:.3f}"

            target_slug = m.get("target_slug", "unknown")
            lora_name = m.get("lora_name")
            lora_path = m.get("lora_path")

            if not lora_name or not lora_path:
                continue

            display_name = f"🛡️ {target_slug} (Loss: {loss})"
            options.append(display_name)
            model_map[display_name] = lora_name
            paths_map[lora_name] = lora_path

        selected_display = st.selectbox("Choose Adapter", options)
        selected_model_name = model_map[selected_display]

        if selected_model_name != "qwen-base":
            if st.button("🔌 Load Adapter to GPU"):
                with st.spinner("Loading LoRA..."):
                    res = dispatcher_load_adapter(selected_model_name, paths_map[selected_model_name])
                    if res is not None:
                        st.success(f"Loaded: {res.get('status', 'OK')}")

        st.subheader("Inference Params")
        temp = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.1)
        max_tok = st.slider("Max Tokens", 64, 4096, 512, step=64)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input()
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            res = dispatcher_chat(selected_model_name, st.session_state.messages, temp, max_tok)
            if not res:
                st.stop()

            try:
                content = res["choices"][0]["message"]["content"]
            except Exception:
                st.error("Bad response shape from Dispatcher.")
                st.stop()

            st.write(content)
            st.session_state.messages.append({"role": "assistant", "content": content})
