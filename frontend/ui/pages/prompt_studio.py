"""
ui/pages/prompt_studio.py

Prompt Engineering Studio with:
  - Named, versioned prompt templates (stored in session state)
  - Jinja2 variable substitution
  - Side-by-side prompt → response analysis
  - Version diff comparison
  - Export / import as JSON (future LoRA dataset format)
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import streamlit as st
from jinja2 import Template, TemplateError

from ui.services.gateway_client import gateway
from ui.components.metrics import api_result_header, cost_card
from ui.components.json_viewer import json_viewer
from ui.utils import state as S
from ui.utils.cost import estimate_gpu, estimate_cpu
from ui.utils.formatters import fmt_ts

_BASE_MODELS = [
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _all_prompts() -> dict:
    return S.get("prompt_versions") or {}


def _save_version(name: str, text: str, tags: list[str]) -> None:
    versions = _all_prompts()
    if name not in versions:
        versions[name] = []
    versions[name].append({
        "version": len(versions[name]) + 1,
        "text": text,
        "tags": tags,
        "created_at": datetime.utcnow().isoformat(),
        "responses": [],
    })
    S.set("prompt_versions", versions)


def _append_response(name: str, version: int, response: dict) -> None:
    versions = _all_prompts()
    for v in versions.get(name, []):
        if v["version"] == version:
            v["responses"].append(response)
            break
    S.set("prompt_versions", versions)


# ── Main render ──────────────────────────────────────────────────────────────

def render():
    st.title("🎨 Prompt Studio")
    st.caption("Version, test, and compare prompt templates · Designed for LoRA dataset collection")

    tab_editor, tab_library, tab_compare, tab_export = st.tabs(
        ["✏️ Editor", "📚 Library", "🔀 Compare versions", "📦 Export"]
    )

    # ── Editor ───────────────────────────────────────────────────────
    with tab_editor:
        col_left, col_right = st.columns([2, 3])

        with col_left:
            st.subheader("Prompt template")

            prompts = _all_prompts()
            prompt_names = list(prompts.keys())

            with st.container():
                mode = st.radio("Mode", ["New prompt", "Edit existing"],
                                horizontal=True, label_visibility="collapsed")

            if mode == "Edit existing" and prompt_names:
                selected_name = st.selectbox("Prompt", prompt_names, key="ps_sel_name")
                versions = prompts[selected_name]
                ver_nums = [v["version"] for v in versions]
                sel_ver = st.selectbox("Version", ver_nums,
                                       index=len(ver_nums) - 1,
                                       format_func=lambda v: f"v{v}",
                                       key="ps_sel_ver")
                existing = next((v for v in versions if v["version"] == sel_ver), None)
                default_text = existing["text"] if existing else ""
                default_tags = ", ".join(existing.get("tags", [])) if existing else ""
                name_input = selected_name
            else:
                name_input = st.text_input("Prompt name",
                                           placeholder="e.g. summarise-article",
                                           key="ps_new_name")
                default_text = "Summarise the following in {{max_sentences}} sentences:\n\n{{text}}"
                default_tags = ""

            prompt_text = st.text_area(
                "Template  (Jinja2: `{{ variable }}`)",
                value=default_text,
                height=200,
                key="ps_template",
            )
            tags_raw = st.text_input("Tags (comma-separated)", value=default_tags,
                                     placeholder="summarise, en, production")

            sys_prompt = st.text_area("System prompt (optional)", height=60,
                                      placeholder="You are a concise assistant.",
                                      key="ps_sys")

            if st.button("💾 Save version", type="primary"):
                if not name_input.strip():
                    st.warning("Enter a prompt name.")
                elif not prompt_text.strip():
                    st.warning("Template cannot be empty.")
                else:
                    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
                    _save_version(name_input.strip(), prompt_text, tags)
                    st.success(f"Saved v{len(_all_prompts().get(name_input.strip(), []))} of `{name_input}`")

        with col_right:
            st.subheader("Test run")

            # Variable extraction
            import re
            vars_found = re.findall(r"\{\{\s*(\w+)\s*\}\}", prompt_text)
            var_values: dict[str, str] = {}
            if vars_found:
                st.caption(f"Variables: `{'`, `'.join(vars_found)}`")
                for var in vars_found:
                    var_values[var] = st.text_input(
                        f"{{ {var} }}", key=f"ps_var_{var}",
                        placeholder=f"value for {var}",
                    )

            # Render preview
            try:
                rendered = Template(prompt_text).render(**var_values)
            except TemplateError as e:
                rendered = f"[Template error: {e}]"

            with st.expander("👁 Rendered prompt", expanded=True):
                st.markdown(f"```\n{rendered}\n```")

            st.divider()

            # Model + params
            c1, c2, c3 = st.columns(3)
            with c1:
                model_res = gateway.list_models()
                live = []
                if model_res.ok and isinstance(model_res.data, dict):
                    live = [m.get("id") for m in model_res.data.get("data", []) if m.get("id")]
                model_sel = st.selectbox("Model", live or _BASE_MODELS, key="ps_model")
            with c2:
                temp = st.slider("Temp", 0.0, 2.0, 0.7, 0.05, key="ps_temp")
            with c3:
                max_tok = st.slider("Max tokens", 32, 2048, 256, 32, key="ps_maxtok")

            runtime_pref = st.selectbox("Runtime", ["auto", "gpu", "cpu"], key="ps_runtime")

            if st.button("▶ Run", type="primary", key="ps_run"):
                messages = []
                if sys_prompt.strip():
                    messages.append({"role": "system", "content": sys_prompt})
                messages.append({"role": "user", "content": rendered})

                with st.spinner("Calling inference…"):
                    result = gateway.chat(
                        model=model_sel,
                        messages=messages,
                        temperature=temp,
                        max_tokens=max_tok,
                        runtime_preference=runtime_pref,
                    )

                api_result_header(result, service="Gateway")

                if not result.ok:
                    st.error(result.error)
                else:
                    content = ""
                    try:
                        content = result.data["choices"][0]["message"]["content"]
                        usage = result.data.get("usage", {})
                        rt = result.data.get("runtime_type", runtime_pref)
                    except Exception:
                        usage = {}
                        rt = runtime_pref

                    st.markdown(f"**Response:**\n\n{content}")

                    comp_tokens = usage.get("completion_tokens", max_tok)
                    est = (estimate_cpu if rt == "cpu" else estimate_gpu)(result.latency_ms, comp_tokens)
                    cost_card(est)

                    # Persist response
                    if name_input.strip() and name_input in _all_prompts():
                        versions = _all_prompts()[name_input]
                        if versions:
                            latest_ver = versions[-1]["version"]
                            _append_response(name_input, latest_ver, {
                                "ts": datetime.utcnow().isoformat(),
                                "model": model_sel,
                                "runtime": rt,
                                "variables": var_values,
                                "rendered_prompt": rendered,
                                "response": content,
                                "latency_ms": result.latency_ms,
                                "tokens": usage,
                                "cost_usd": est.get("cost_usd", 0),
                            })
                            st.success("Response logged to prompt version history.")

    # ── Library ──────────────────────────────────────────────────────
    with tab_library:
        st.subheader("All prompt versions")
        prompts = _all_prompts()

        if not prompts:
            st.info("No prompts saved yet. Create one in the Editor tab.")
        else:
            for pname, versions in prompts.items():
                with st.expander(f"**{pname}**  — {len(versions)} version(s)"):
                    for v in reversed(versions):
                        ver_num = v["version"]
                        tags = " · ".join(f"`{t}`" for t in v.get("tags", []))
                        st.markdown(f"**v{ver_num}**  {fmt_ts(v.get('created_at'))}  {tags}")
                        st.code(v["text"], language="jinja2")

                        responses = v.get("responses", [])
                        if responses:
                            st.caption(f"{len(responses)} test response(s)")
                            for resp in responses[-3:]:
                                with st.container():
                                    st.caption(
                                        f"🕐 {fmt_ts(resp.get('ts'))}  ·  "
                                        f"model: `{resp.get('model')}`  ·  "
                                        f"⏱ {resp.get('latency_ms', '?')} ms  ·  "
                                        f"cost: ${resp.get('cost_usd', 0):.5f}"
                                    )
                                    st.markdown(f"> {resp.get('response', '')[:300]}")
                        st.divider()

    # ── Compare versions ─────────────────────────────────────────────
    with tab_compare:
        st.subheader("Version comparison")
        prompts = _all_prompts()
        if not prompts:
            st.info("No prompts to compare.")
        else:
            pname = st.selectbox("Prompt", list(prompts.keys()), key="cmp_name")
            versions = prompts.get(pname, [])
            if len(versions) < 2:
                st.info("Need at least 2 versions to compare.")
            else:
                ver_nums = [v["version"] for v in versions]
                c1, c2 = st.columns(2)
                with c1:
                    va = st.selectbox("Version A", ver_nums, index=0, key="cmp_va",
                                      format_func=lambda x: f"v{x}")
                with c2:
                    vb = st.selectbox("Version B", ver_nums, index=len(ver_nums)-1,
                                      key="cmp_vb", format_func=lambda x: f"v{x}")

                def _get_ver(n):
                    return next((v for v in versions if v["version"] == n), {})

                va_data = _get_ver(va)
                vb_data = _get_ver(vb)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption(f"**v{va}** — {fmt_ts(va_data.get('created_at'))}")
                    st.code(va_data.get("text", ""), language="jinja2")
                    for r in va_data.get("responses", [])[-2:]:
                        st.info(f"⏱ {r.get('latency_ms')} ms · {r.get('response','')[:200]}")
                with col_b:
                    st.caption(f"**v{vb}** — {fmt_ts(vb_data.get('created_at'))}")
                    st.code(vb_data.get("text", ""), language="jinja2")
                    for r in vb_data.get("responses", [])[-2:]:
                        st.info(f"⏱ {r.get('latency_ms')} ms · {r.get('response','')[:200]}")

    # ── Export ────────────────────────────────────────────────────────
    with tab_export:
        st.subheader("Export prompt library")
        st.caption("JSON format compatible with future LoRA dataset pipeline.")

        prompts = _all_prompts()
        if not prompts:
            st.info("Nothing to export yet.")
        else:
            # Format: list of {instruction, input, output} pairs from responses
            lora_dataset = []
            for pname, versions in prompts.items():
                for v in versions:
                    for resp in v.get("responses", []):
                        lora_dataset.append({
                            "prompt_name": pname,
                            "version": v["version"],
                            "instruction": resp.get("rendered_prompt", ""),
                            "output": resp.get("response", ""),
                            "model": resp.get("model", ""),
                            "latency_ms": resp.get("latency_ms"),
                            "cost_usd": resp.get("cost_usd"),
                            "tags": v.get("tags", []),
                        })

            raw_library = json.dumps(prompts, indent=2, ensure_ascii=False)
            raw_dataset = json.dumps(lora_dataset, indent=2, ensure_ascii=False)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "⬇ Download prompt library (JSON)",
                    data=raw_library,
                    file_name="prompt_library.json",
                    mime="application/json",
                )
            with c2:
                st.download_button(
                    "⬇ Download LoRA dataset (JSONL-ready)",
                    data=raw_dataset,
                    file_name="lora_dataset.json",
                    mime="application/json",
                )

            st.caption(f"{len(lora_dataset)} response records ready for training.")
            with st.expander("Preview dataset (first 3 records)"):
                st.json(lora_dataset[:3])
