"""
ui/app.py

Main Streamlit entry point.
Sets up the sidebar navigation and routes to each page module.
"""
from __future__ import annotations

import streamlit as st

from ui.theme import apply as apply_theme

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Model Runtime",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# ── Navigation ────────────────────────────────────────────────────────────────

_PAGES = {
    "💬 Chat":             ("chat",          "chat"),
    "📦 Model Registry":   ("model_registry","model_registry"),
    "🤗 HuggingFace Hub":  ("hf_hub",        "hf_hub"),
    "🏋 Training":         ("training",       "training"),
    "🎛 Orchestration":    ("orchestration",  "orchestration"),
    "🎨 Prompt Studio":    ("prompt_studio",  "prompt_studio"),
    "🧪 Experiments":      ("experiments",    "experiments"),
    "📊 Monitoring":       ("monitoring",     "monitoring"),
    "🧠 Runtime Debug":    ("runtime_debug",  "runtime_debug"),
    "🔧 API Playground":   ("api_playground", "api_playground"),
}

with st.sidebar:
    st.markdown("## 🧠 Model Runtime")
    st.divider()

    page_label = st.radio(
        "Navigation",
        list(_PAGES.keys()),
        label_visibility="collapsed",
        key="nav_page",
    )

    st.divider()
    st.caption("Model Runtime Platform")

# ── Lazy import + render ──────────────────────────────────────────────────────

module_name, _ = _PAGES[page_label]

if module_name == "chat":
    from ui.pages.chat import render
elif module_name == "model_registry":
    from ui.pages.model_registry import render
elif module_name == "hf_hub":
    from ui.pages.hf_hub import render
elif module_name == "training":
    from ui.pages.training import render
elif module_name == "orchestration":
    from ui.pages.orchestration import render
elif module_name == "prompt_studio":
    from ui.pages.prompt_studio import render
elif module_name == "experiments":
    from ui.pages.experiments import render
elif module_name == "monitoring":
    from ui.pages.monitoring import render
elif module_name == "runtime_debug":
    from ui.pages.runtime_debug import render
elif module_name == "api_playground":
    from ui.pages.api_playground import render
else:
    def render():
        st.error(f"Unknown page: {module_name}")

render()
