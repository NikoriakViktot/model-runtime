import streamlit as st

def init_state() -> None:
    st.session_state.setdefault("selected_dataset_run_id", None)
    st.session_state.setdefault("selected_training_run_id", None)
    st.session_state.setdefault("last_run_id", None)
    st.session_state.setdefault("db_id", None)
    st.session_state.setdefault("target_name", None)
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("preview_prompt", None)
