"""
ui/utils/state.py
Centralised session-state initialisation and typed accessors.
"""
from __future__ import annotations
import streamlit as st


_DEFAULTS: dict = {
    # Chat
    "messages": [],
    "chat_model": None,
    "chat_system_prompt": "",
    # Prompt Studio
    "prompt_versions": {},          # {name: [{version, text, ts, tags}]}
    "active_prompt_name": None,
    "active_prompt_text": "",
    # HF Hub
    "hf_results": [],
    "hf_query": "",
    # API Playground
    "playground_last_request": None,
    "playground_last_response": None,
    "playground_templates": {},
    # Experiments
    "experiment_results": [],
    # Debug
    "debug_routing_log": [],
}


def init():
    for k, v in _DEFAULTS.items():
        st.session_state.setdefault(k, v)


def get(key: str):
    return st.session_state.get(key)


def set(key: str, value) -> None:
    st.session_state[key] = value


def push(key: str, item) -> None:
    lst = st.session_state.get(key, [])
    lst.append(item)
    st.session_state[key] = lst
