import re
import time
import streamlit as st
from jinja2 import Template
from ..cp_api import ControlPlaneAPI

def render(cp: ControlPlaneAPI) -> None:
    st.title("🎨 Prompt Engineering Studio")
    st.caption("Create, version, inspect, render, test prompts.")

    prompts = cp.prompts()

    with st.sidebar:
        st.subheader("Library")
        new_pid = st.text_input("New Prompt ID", placeholder="etl.gen_profile")
        new_desc = st.text_input("Description", value="Created via UI")

        if st.button("➕ Create Family"):
            if not new_pid:
                st.warning("Prompt ID required")
            else:
                try:
                    cp.create_prompt_family(new_pid, new_desc)
                    st.success("Created")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        st.divider()
        ids = [p.get("id") for p in prompts if isinstance(p, dict) and p.get("id")]
        selected_pid = st.radio("Prompt Families", ids, index=0 if ids else None)

    if not selected_pid:
        st.info("No prompts yet.")
        return

    st.subheader(f"Editing: `{selected_pid}`")

    versions = cp.prompt_versions(selected_pid)
    tab_edit, tab_history, tab_test = st.tabs(["✏️ Editor", "📜 History", "🧪 Playground"])

    with tab_edit:
        default_tmpl = versions[0].get("template") if versions else "You are an AI assistant. Context: {{ conversation }}"
        last_tag = versions[0].get("version_tag") if versions else "v0.0"

        new_template = st.text_area("Template", value=default_tmpl, height=420)

        vars_found = sorted(set(re.findall(r"\{\{\s*(\w+)\s*\}\}", new_template)))
        if vars_found:
            st.info("Vars: " + ", ".join(vars_found))

        c1, c2 = st.columns([1, 2])
        with c1:
            try:
                v_num = int(last_tag.split("v")[-1].replace(".", ""))
                suggested = f"v1.{v_num + 1}"
            except Exception:
                suggested = "v1.0"
            new_tag = st.text_input("Version Tag", value=suggested)

        with c2:
            commit_msg = st.text_input("Commit Message", placeholder="What changed?")

        if st.button("💾 Save Version", type="primary"):
            res = cp.save_prompt_version(selected_pid, new_tag, new_template, commit_msg or "UI Update")
            if res is not None:
                st.success("Saved")
                time.sleep(0.2)
                st.rerun()

    with tab_history:
        if not versions:
            st.info("No versions.")
        for v in versions:
            tag = v.get("version_tag", "?")
            created_at = (v.get("created_at") or "")[:16]
            msg = v.get("commit_message", "")
            tmpl = v.get("template", "")
            with st.expander(f"{tag} | {created_at} | {msg}"):
                st.code(tmpl, language="jinja2")

    with tab_test:
        vars_found = sorted(set(re.findall(r"\{\{\s*(\w+)\s*\}\}", new_template)))
        if not vars_found:
            st.info("No vars. Just copy and use.")
            return

        inputs = {}
        for var in vars_found:
            height = 220 if ("conversation" in var or "text" in var) else None
            inputs[var] = st.text_area(f"{var}", height=height)

        if st.button("👁️ Render Preview"):
            try:
                t = Template(new_template)
                rendered = t.render(**inputs)
                st.session_state.preview_prompt = rendered
                st.text_area("Rendered", value=rendered, height=320)
            except Exception as e:
                st.error(f"Jinja2 error: {e}")
