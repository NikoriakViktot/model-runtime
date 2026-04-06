"""
ui/pages/hf_hub.py

HuggingFace Hub — search, GPU-aware recommendations, and one-click model registration.

Runtime configuration is explicit: users set max_model_len, GPU utilization,
max concurrent sequences, and LoRA directly. An auto-tune toggle lets the
backend choose the best config automatically when enabled.
"""
from __future__ import annotations

import json

import streamlit as st

from ui.services.mrm_client import mrm
from ui.components.metrics import api_result_header
from ui.components.json_viewer import json_viewer
from ui.components.runtime_explain import render_runtime_debug, render_vram_preview
from ui.utils.formatters import fmt_ts

_GPU_IDS = ["0", "1", "2", "3"]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _runtime_config_inputs(key_prefix: str) -> dict:
    """
    Render the runtime configuration panel and return the collected values.

    Returns a dict ready to pass as `overrides` to register_from_hf.
    """
    st.subheader("⚙️ Runtime Configuration")

    auto_tune = st.checkbox(
        "🤖 Auto-tune (recommended)",
        value=True,
        key=f"{key_prefix}_auto_tune",
    )

    st.info(
        "Auto-tune will automatically adjust context length and batching "
        "to fit your GPU memory. Disable it to manually control settings."
    )

    if not auto_tune:
        col1, col2 = st.columns(2)

        with col1:
            max_model_len = st.number_input(
                "Max context length",
                min_value=256,
                max_value=8192,
                value=1024,
                step=256,
                key=f"{key_prefix}_max_model_len",
            )
            gpu_util = st.slider(
                "GPU memory utilization",
                min_value=0.1,
                max_value=0.95,
                value=0.7,
                step=0.05,
                key=f"{key_prefix}_gpu_util",
            )

        with col2:
            max_num_seqs = st.number_input(
                "Max concurrent sequences",
                min_value=1,
                max_value=16,
                value=1,
                key=f"{key_prefix}_max_num_seqs",
            )
            enable_lora = st.checkbox(
                "Enable LoRA",
                value=False,
                key=f"{key_prefix}_enable_lora",
            )

        st.caption("⚠️ Large models may require reduced context length to fit GPU memory.")

        return {
            "max_model_len": int(max_model_len),
            "gpu_memory_utilization": float(gpu_util),
            "max_num_seqs": int(max_num_seqs),
            "enable_lora": bool(enable_lora),
        }

    # auto_tune=True → backend decides; pass no overrides
    return None


def _model_card(
    m: dict,
    gpu_id: str,
    overrides: dict | None,
    available_vram_gb: float | None = None,
) -> None:
    """Render a single model card with a one-click Register button."""
    repo = m.get("modelId") or m.get("repo_id", "?")
    downloads = m.get("downloads", 0)
    likes = m.get("likes", 0)
    tags = m.get("tags") or []
    size_note = m.get("size_note", "")
    fits = m.get("fits_in_gpu", None)

    fit_icon = {True: "✅", False: "❌", None: "❓"}.get(fits, "❓")

    with st.container():
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"**`{repo}`**")
            meta = []
            if downloads:
                meta.append(f"⬇ {downloads:,}")
            if likes:
                meta.append(f"❤ {likes:,}")
            if size_note:
                meta.append(f"📦 {size_note}")
            if tags:
                meta.append("  ".join(f"`{t}`" for t in tags[:5]))
            if meta:
                st.caption("  ·  ".join(meta))
            if fits is not None:
                st.caption(f"{fit_icon} {'Fits in GPU VRAM' if fits else 'Too large for GPU VRAM'}")
            # VRAM context when GPU info is available
            if available_vram_gb is not None:
                st.caption(f"GPU available: {available_vram_gb:.1f} GB free VRAM")
        with c2:
            btn_key = f"hf_reg_{repo}_{gpu_id}"
            if st.button("📥 Register", key=btn_key, type="primary"):
                with st.spinner(f"Registering `{repo}`…"):
                    res = mrm.register_from_hf(
                        repo_id=repo,
                        preset=None,
                        gpu=gpu_id,
                        overrides=overrides,
                    )
                api_result_header(res, service="MRM /models/register_from_hf")
                if res.ok:
                    st.success(f"Registered `{repo}` successfully.")
                    # ── Debug panel ──────────────────────────────────
                    debug = None
                    if isinstance(res.data, dict):
                        debug = res.data.get("debug")
                        if not debug and res.data:
                            # Entire data dict may itself be the debug block
                            # (future-proof: accept either shape)
                            json_viewer(res.data, label="Registration result", expanded=False)
                    with st.expander("🔍 Runtime Selection Debug", expanded=False):
                        render_runtime_debug(debug)
                else:
                    st.error(res.error)
        st.divider()


# ── Page ─────────────────────────────────────────────────────────────────────

def _gpu_free_vram_gb() -> float | None:
    """Fetch free VRAM (GB) for GPU 0 from MRM. Returns None on failure."""
    r = mrm.gpu_metrics()
    if not r.ok:
        return None
    gd = r.data or {}
    gpus = gd.get("gpus") or ([gd] if gd else [])
    if not gpus:
        return None
    g = gpus[0]
    total = g.get("memory_total_mib", 0)
    used  = g.get("memory_used_mib", 0)
    free  = total - used
    return round(free / 1024, 2) if total > 0 else None


def render():
    st.title("🤗 HuggingFace Hub")
    st.caption("Search models, get GPU-aware recommendations, and register them with one click.")

    # Fetch GPU metrics once per page render; used for VRAM previews in cards
    available_vram_gb: float | None = _gpu_free_vram_gb()

    tab_search, tab_recommend, tab_register = st.tabs(
        ["🔍 Search", "⚡ GPU Recommendations", "📋 Manual Register"]
    )

    # ── Search ────────────────────────────────────────────────────────
    with tab_search:
        col_q, col_lim = st.columns([3, 1])
        with col_q:
            query = st.text_input(
                "Search query",
                value="llama instruct",
                placeholder="e.g. mistral 7b instruct",
                key="hf_search_q",
            )
        with col_lim:
            limit = st.number_input("Max results", 5, 50, 20, key="hf_search_lim")

        gpu_id_s = st.selectbox("Target GPU", _GPU_IDS, key="hf_gpu_s")
        overrides_s = _runtime_config_inputs("search")

        if st.button("🔍 Search", type="primary", key="hf_search_btn"):
            with st.spinner(f"Searching HuggingFace for `{query}`…"):
                r = mrm.hf_search(q=query, limit=int(limit))

            api_result_header(r, service="MRM /hf/search")

            if not r.ok:
                st.error(r.error)
            else:
                results = r.data if isinstance(r.data, list) else (r.data or {}).get("models", [])
                if not results:
                    st.info("No results found.")
                else:
                    st.caption(f"{len(results)} model(s) found")
                    for m in results:
                        _model_card(
                            m,
                            gpu_id=gpu_id_s,
                            overrides=overrides_s,
                            available_vram_gb=available_vram_gb,
                        )

    # ── GPU Recommendations ───────────────────────────────────────────
    with tab_recommend:
        st.subheader("GPU-aware recommendations")
        st.caption("Results are filtered and ranked by VRAM fit for your selected GPU.")

        col_q2, col_gpu2 = st.columns(2)
        with col_q2:
            rec_query = st.text_input(
                "Use-case description",
                value="small instruct chat",
                key="hf_rec_q",
            )
        with col_gpu2:
            gpu_id_r = st.selectbox("Target GPU", _GPU_IDS, key="hf_gpu_r")

        rec_limit = st.number_input("Max results", 5, 50, 10, key="hf_rec_lim")
        overrides_r = _runtime_config_inputs("recommend")

        if st.button("⚡ Get recommendations", type="primary", key="hf_rec_btn"):
            with st.spinner("Fetching GPU-aware recommendations…"):
                r = mrm.hf_recommend(q=rec_query, gpu_id=gpu_id_r, limit=int(rec_limit))

            api_result_header(r, service="MRM /hf/recommend")

            if not r.ok:
                st.error(r.error)
            else:
                results = r.data if isinstance(r.data, list) else (r.data or {}).get("models", [])
                fits   = [m for m in results if m.get("fits_in_gpu")]
                doesnt = [m for m in results if not m.get("fits_in_gpu") and m.get("fits_in_gpu") is not None]
                unknown = [m for m in results if m.get("fits_in_gpu") is None]

                if fits:
                    st.success(f"✅ {len(fits)} model(s) fit in GPU VRAM")
                    for m in fits:
                        _model_card(
                            m,
                            gpu_id=gpu_id_r,
                            overrides=overrides_r,
                            available_vram_gb=available_vram_gb,
                        )

                if doesnt:
                    with st.expander(f"❌ {len(doesnt)} model(s) too large for GPU VRAM"):
                        for m in doesnt:
                            _model_card(
                                m,
                                gpu_id=gpu_id_r,
                                overrides=overrides_r,
                                available_vram_gb=available_vram_gb,
                            )

                if unknown and not fits and not doesnt:
                    st.info("No VRAM fit information available.")
                    for m in unknown:
                        _model_card(
                            m,
                            gpu_id=gpu_id_r,
                            overrides=overrides_r,
                            available_vram_gb=available_vram_gb,
                        )

    # ── Manual Register ───────────────────────────────────────────────
    with tab_register:
        st.subheader("Register model from HuggingFace")

        col_repo, col_gpu3 = st.columns(2)
        with col_repo:
            repo_id = st.text_input(
                "HuggingFace repo ID",
                placeholder="Qwen/Qwen2.5-7B-Instruct",
                key="hf_man_repo",
            )
        with col_gpu3:
            gpu_id_m = st.selectbox("GPU", _GPU_IDS, key="hf_man_gpu")

        alias_override = st.text_input(
            "Alias override (optional)",
            placeholder="my-model-alias",
            key="hf_man_alias",
        )

        overrides_m = _runtime_config_inputs("manual")

        # Merge alias override on top of the config overrides (if any)
        if alias_override.strip():
            if overrides_m is None:
                overrides_m = {}
            overrides_m["model_alias"] = alias_override.strip()

        with st.expander("Advanced overrides (JSON)"):
            raw_overrides = st.text_area(
                "Extra overrides",
                value="{}",
                height=100,
                key="hf_man_overrides",
            )
            try:
                extra = json.loads(raw_overrides)
                if extra:
                    if overrides_m is None:
                        overrides_m = {}
                    overrides_m.update(extra)
            except Exception:
                st.error("Invalid JSON in overrides field")

        # VRAM context — show available GPU memory before registering
        if available_vram_gb is not None:
            render_vram_preview(
                estimated_gb=None,
                available_gb=available_vram_gb,
                label="GPU free VRAM",
            )

        if st.button("📥 Register", type="primary", key="hf_man_reg_btn"):
            if not repo_id.strip():
                st.warning("Repo ID is required.")
            else:
                with st.spinner(f"Registering `{repo_id}`…"):
                    res = mrm.register_from_hf(
                        repo_id=repo_id.strip(),
                        preset=None,
                        gpu=gpu_id_m,
                        overrides=overrides_m,
                    )
                api_result_header(res, service="MRM /models/register_from_hf")
                if res.ok:
                    st.success(f"Registered `{repo_id}` successfully.")
                    debug = None
                    if isinstance(res.data, dict):
                        debug = res.data.get("debug")
                    with st.expander("🔍 Runtime Selection Debug", expanded=True):
                        render_runtime_debug(debug)
                else:
                    st.error(res.error)

        st.divider()
        st.subheader("Current LiteLLM config")
        if st.button("🔄 Reload LiteLLM config", key="hf_litellm_reload"):
            with st.spinner("Reloading…"):
                res = mrm.litellm_reload()
            if res.ok:
                st.success("LiteLLM config reloaded.")
            else:
                st.error(res.error)

        r_cfg = mrm.litellm_config()
        if r_cfg.ok:
            json_viewer(r_cfg.data, label="LiteLLM config", expanded=False)
        else:
            st.info("LiteLLM config unavailable.")
