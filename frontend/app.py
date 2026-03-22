import streamlit as st
from ui.state import init_state
from ui.http import Http
from ui.cp_api import ControlPlaneAPI
from ui.dispatcher_api import DispatcherAPI

from ui.pages import (
    home, orchestration, dataset_build, training, chat,
    prompt_studio, docs_tools, model_registry,
    hf_registry, gpu_monitor,
    deployment_manager
)

st.set_page_config(page_title="AI Control Plane", page_icon="🧠", layout="wide")

init_state()

http = Http()
cp = ControlPlaneAPI(http)
dsp = DispatcherAPI(http)

st.sidebar.title("🧬 Control Plane")
page = st.sidebar.radio(
    "Navigation",
    [
        "🧭 Home",
        "🎛 Orchestration",
        "🔎 HF Registry",
        "🧬 Model Registry",
        "🟩 GPU Monitor",
        "📦 Dataset Build",
        "🔥 Training",
        "🚀 Deployment",
        "💬 Chat",
        "🎨 Prompt Studio",
        "📚 Docs & Tools",
    ],
)

if st.sidebar.button("🧹 Clear chat"):
    st.session_state.messages = []
    st.session_state.pop("selected_training_run_id", None)
    st.rerun()


if page == "🧭 Home":
    home.render()
elif page == "🎛 Orchestration":
    orchestration.render(cp)
elif page == "🔎 HF Registry":
    hf_registry.render()
elif page == "🧬 Model Registry":
    model_registry.render()
elif page == "🟩 GPU Monitor":
    gpu_monitor.render()
elif page == "📦 Dataset Build":
    dataset_build.render(cp, dsp)
elif page == "🔥 Training":
    training.render(cp, dsp)
elif page == "🚀 Deployment":
    deployment_manager.render(dsp)
elif page == "💬 Chat":
    chat.render(dsp)
elif page == "🎨 Prompt Studio":
    prompt_studio.render(cp)
elif page == "📚 Docs & Tools":
    docs_tools.render()
