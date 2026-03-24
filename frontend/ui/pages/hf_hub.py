"""
ui/pages/hf_hub.py

HuggingFace Hub — search, GPU-aware recommendations, and one-click model registration.
"""
from __future__ import annotations

import streamlit as st

from ui.services.mrm_client import mrm
from ui.components.metrics import api_result_header
from ui.components.json_viewer import json_viewer
from ui.utils.formatters import fmt_ts

_PRESETS = [
    "small_chat",
    "medium_chat",
    "large_chat",
    "code",
    "instruct",
]

_GPU_IDS = ["0", "1", "2", "3"]


def _model_card(m: dict, gpu_id: str, preset: str) -> None:
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
        with c2:
            btn_key = f"hf_reg_{repo}_{gpu_id}"
            if st.button("📥 Register", key=btn_key, type="primary"):
                with st.spinner(f"Registering `{repo}`…"):
                    res = mrm.register_from_hf(repo_id=repo, preset=preset, gpu=gpu_id)
                api_result_header(res, service="MRM /models/register_from_hf")
                if res.ok:
                    st.success(f"Registered `{repo}` with preset `{preset}`")
                else:
                    st.error(res.error)
        st.divider()


def render():
    st.title("🤗 HuggingFace Hub")
    st.caption("Search models, get GPU-aware recommendations, and register them with one click.")

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

        col_gpu_s, col_preset_s = st.columns(2)
        with col_gpu_s:
            gpu_id_s = st.selectbox("Target GPU", _GPU_IDS, key="hf_gpu_s")
        with col_preset_s:
            preset_s = st.selectbox("Register preset", _PRESETS, key="hf_preset_s")

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
                        _model_card(m, gpu_id=gpu_id_s, preset=preset_s)

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

        col_lim2, col_preset2 = st.columns(2)
        with col_lim2:
            rec_limit = st.number_input("Max results", 5, 50, 10, key="hf_rec_lim")
        with col_preset2:
            preset_r = st.selectbox("Register preset", _PRESETS, key="hf_preset_r")

        if st.button("⚡ Get recommendations", type="primary", key="hf_rec_btn"):
            with st.spinner("Fetching GPU-aware recommendations…"):
                r = mrm.hf_recommend(q=rec_query, gpu_id=gpu_id_r, limit=int(rec_limit))

            api_result_header(r, service="MRM /hf/recommend")

            if not r.ok:
                st.error(r.error)
            else:
                results = r.data if isinstance(r.data, list) else (r.data or {}).get("models", [])
                fits = [m for m in results if m.get("fits_in_gpu")]
                doesnt = [m for m in results if not m.get("fits_in_gpu") and m.get("fits_in_gpu") is not None]
                unknown = [m for m in results if m.get("fits_in_gpu") is None]

                if fits:
                    st.success(f"✅ {len(fits)} model(s) fit in GPU VRAM")
                    for m in fits:
                        _model_card(m, gpu_id=gpu_id_r, preset=preset_r)

                if doesnt:
                    with st.expander(f"❌ {len(doesnt)} model(s) too large for GPU VRAM"):
                        for m in doesnt:
                            _model_card(m, gpu_id=gpu_id_r, preset=preset_r)

                if unknown and not fits and not doesnt:
                    st.info("No VRAM fit information available.")
                    for m in unknown:
                        _model_card(m, gpu_id=gpu_id_r, preset=preset_r)

    # ── Manual Register ───────────────────────────────────────────────
    with tab_register:
        st.subheader("Register model from HuggingFace")

        col_repo, col_preset3 = st.columns(2)
        with col_repo:
            repo_id = st.text_input(
                "HuggingFace repo ID",
                placeholder="Qwen/Qwen2.5-7B-Instruct",
                key="hf_man_repo",
            )
        with col_preset3:
            preset_m = st.selectbox("Preset", _PRESETS, key="hf_man_preset")

        col_gpu3, col_alias = st.columns(2)
        with col_gpu3:
            gpu_id_m = st.selectbox("GPU", _GPU_IDS, key="hf_man_gpu")
        with col_alias:
            alias_override = st.text_input(
                "Alias override (optional)",
                placeholder="my-model-alias",
                key="hf_man_alias",
            )

        overrides: dict = {}
        if alias_override.strip():
            overrides["model_alias"] = alias_override.strip()

        with st.expander("Advanced overrides (JSON)"):
            import json
            raw_overrides = st.text_area(
                "Extra overrides",
                value="{}",
                height=100,
                key="hf_man_overrides",
            )
            try:
                extra = json.loads(raw_overrides)
                overrides.update(extra)
            except Exception:
                st.error("Invalid JSON in overrides field")

        if st.button("📥 Register", type="primary", key="hf_man_reg_btn"):
            if not repo_id.strip():
                st.warning("Repo ID is required.")
            else:
                with st.spinner(f"Registering `{repo_id}`…"):
                    res = mrm.register_from_hf(
                        repo_id=repo_id.strip(),
                        preset=preset_m,
                        gpu=gpu_id_m,
                        overrides=overrides,
                    )
                api_result_header(res, service="MRM /models/register_from_hf")
                if res.ok:
                    st.success(f"Registered `{repo_id}` successfully.")
                    if isinstance(res.data, dict):
                        json_viewer(res.data, label="Registration result", expanded=True)
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
