import streamlit as st
from ..config import MRM_BASE_URL
from ..mrm_api import MRMApi

HF_INFO_KEY = "hf_info_data"
BUSY_KEY = "busy"


def render() -> None:
    st.title("🔎 HF Search → Preset → Overrides → Register+Ensure")
    api = MRMApi(MRM_BASE_URL)

    if BUSY_KEY not in st.session_state:
        st.session_state[BUSY_KEY] = False

    q = st.text_input(
        "Search on Hugging Face",
        value="Qwen2.5",
        help="Searches Hugging Face models by keyword (e.g., Qwen, Mistral, Llama)."
    )
    limit = st.slider(
        "Limit",
        1, 50, 10,
        help="How many search results to fetch from Hugging Face."
    )

    if st.button(
        "🔎 Search",
        disabled=st.session_state[BUSY_KEY],
        help="Fetches model list from Hugging Face and populates the picker below."
    ):
        res = api.hf_search(q=q, limit=limit)
        st.session_state["hf_items"] = res.get("items", [])
        st.session_state.pop(HF_INFO_KEY, None)

    items = st.session_state.get("hf_items", [])
    if not items:
        st.info("Search for models to begin.")
        return

    labels, map_repo = [], {}
    for it in items:
        rid = it.get("modelId") or it.get("id")
        if not rid:
            continue
        downloads = it.get("downloads", 0)
        likes = it.get("likes", 0)
        label = f"{rid}  | ❤️ {likes} | ⬇️ {downloads}"
        labels.append(label)
        map_repo[label] = rid

    pick = st.selectbox(
        "Pick model",
        labels,
        help="Select the exact Hugging Face repo to register and run in vLLM."
    )
    repo_id = map_repo[pick]

    preset = st.selectbox(
        "Preset",
        ["small_chat", "7b_awq"],
        help="A predefined configuration template (image, defaults, quantization policy, etc.). Overrides below can adjust it."
    )

    with st.expander("Overrides", expanded=True):
        gpu = st.selectbox(
            "GPU",
            ["0"],
            help="Which GPU to reserve for this model (MRM will bind the container to this device)."
        )
        gpu_util = st.slider(
            "gpu_memory_utilization",
            0.30, 0.95, 0.60, 0.01,
            help="Target fraction of GPU VRAM vLLM is allowed to use (helps avoid OOM)."
        )
        max_len = st.slider(
            "max_model_len",
            256, 8192, 2048, 128,
            help="Max context length (tokens). Higher = more VRAM usage."
        )
        enable_lora = st.checkbox(
            "enable_lora",
            value=True,
            help="Enables LoRA adapter support in vLLM (for serving fine-tuned adapters)."
        )
        max_loras = st.slider(
            "max_loras",
            0, 30, 10, 1,
            help="Maximum number of LoRA adapters that can be loaded at once."
        )
        max_rank = st.slider(
            "max_lora_rank",
            8, 128, 32, 8,
            help="Maximum rank for LoRA adapters (must match your adapters; higher may increase memory/compute)."
        )

    overrides = {
        "gpu_memory_utilization": gpu_util,
        "max_model_len": max_len,
        "enable_lora": enable_lora,
        "max_loras": max_loras,
        "max_lora_rank": max_rank,
    }

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button(
            "ℹ️ Load model info",
            key="btn_hf_info",
            disabled=st.session_state[BUSY_KEY],
            help="Calls Hugging Face API to fetch metadata about the selected repo (tags, library, pipeline, etc.)."
        ):
            st.session_state[BUSY_KEY] = True
            try:
                st.session_state[HF_INFO_KEY] = api.hf_model_info(repo_id)
            finally:
                st.session_state[BUSY_KEY] = False

    with c2:
        if st.button(
            "📝 Register (preset+overrides)",
            key="register",
            disabled=st.session_state[BUSY_KEY],
            help="Registers the selected HF repo in MRM registry (creates/updates ModelSpec), but does NOT start vLLM."
        ):
            st.session_state[BUSY_KEY] = True
            try:
                out = api.register_from_hf(repo_id=repo_id, preset=preset, gpu=gpu, overrides=overrides)
                st.success(out)
            finally:
                st.session_state[BUSY_KEY] = False

    with c3:
        if st.button(
            "🏭 Provision (Register+Ensure+Write LiteLLM)",
            key="provision",
            disabled=st.session_state[BUSY_KEY],
            help=(
                "One-click flow: (1) Register model in MRM, (2) Ensure vLLM container is running, "
                "(3) Generate LiteLLM config so the UI/clients can use the model via LiteLLM."
            ),
        ):
            st.session_state[BUSY_KEY] = True
            try:
                out = api.provision(repo_id=repo_id, preset=preset, gpu=gpu, overrides=overrides)
                st.success(out.get("registered"))
                st.success(out.get("ensured"))
                st.success(out.get("litellm"))
            except Exception as e:
                msg = str(e)
                if "status_code" in msg and "409" in msg:
                    st.warning("MRM is busy with this model (409). Try again in a moment.")
                else:
                    raise
            finally:
                st.session_state[BUSY_KEY] = False

        if st.button(
            "🚀 Register + Ensure (debug)",
            key="reg_ensure",
            disabled=st.session_state[BUSY_KEY],
            help=(
                "Debug path: runs Register, then Ensure. Skips LiteLLM materialize/write. "
                "Useful to isolate vLLM startup issues."
            ),
        ):
            st.session_state[BUSY_KEY] = True
            try:
                out = api.register_from_hf(repo_id=repo_id, preset=preset, gpu=gpu, overrides=overrides)
                st.success(out)
                st.success(api.ensure(repo_id))
            except Exception as e:
                msg = str(e)
                if "status_code" in msg and "409" in msg:
                    st.warning("MRM is busy with this model (409). Try again in a moment.")
                else:
                    raise
            finally:
                st.session_state[BUSY_KEY] = False

    info = st.session_state.get(HF_INFO_KEY)
    if info:
        st.json({k: info.get(k) for k in ["id", "pipeline_tag", "library_name", "tags"] if k in info})
