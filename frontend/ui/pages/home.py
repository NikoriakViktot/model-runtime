import streamlit as st
from ..config import (
    NEO4J_UI_URL,
    MLFLOW_UI_PATH,
    CP_SWAGGER_PATH,
    DISPATCH_SWAGGER_PATH,
    VLLM_DOCS_PATH,
    LITELLM_BASE_PATH,
)

def render() -> None:
    st.title("🧠 AI Control Plane")
    st.caption("Single entry point for experiments, models and infrastructure")

    st.subheader("🔗 System Interfaces")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🗄️ Databases")
        st.link_button("Neo4j Graph UI", NEO4J_UI_URL)

    with c2:
        st.markdown("### 📊 Experiment Tracking")
        st.link_button("MLflow UI", MLFLOW_UI_PATH)

    st.subheader("📜 API Documentation")
    c3, c4, c5 = st.columns(3)
    with c3:
        st.link_button("Control Plane Swagger", CP_SWAGGER_PATH)
    with c4:
        st.link_button("Dispatcher Swagger", DISPATCH_SWAGGER_PATH)

    st.subheader("🔌 Inference Gateways")
    st.link_button("LiteLLM (OpenAI /v1)", LITELLM_BASE_PATH)

    st.divider()
    st.markdown(
        """
**UI Contract**
- UI creates contracts, shows status, links tools.
- Execution happens in services (Control Plane / Dispatcher / Workers).
- No hidden side-effects in the browser.
"""
    )
