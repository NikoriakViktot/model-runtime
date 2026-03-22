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
    st.title("📚 Docs & Tools")

    st.subheader("Core")
    st.link_button("🔧 Control Plane Swagger", CP_SWAGGER_PATH)
    st.link_button("⚙️ Dispatcher Swagger", DISPATCH_SWAGGER_PATH)
    st.link_button("📊 MLflow UI", MLFLOW_UI_PATH)
    st.link_button("🧠 Neo4j Browser", NEO4J_UI_URL)

    st.subheader("Inference")
    st.link_button("vLLM OpenAPI", VLLM_DOCS_PATH)
    st.link_button("LiteLLM /v1", LITELLM_BASE_PATH)
