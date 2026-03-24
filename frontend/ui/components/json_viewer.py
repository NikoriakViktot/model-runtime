"""
ui/components/json_viewer.py
Pretty JSON display with collapsible sections.
"""
from __future__ import annotations
import json
import streamlit as st
from ui.utils.formatters import pretty_json


def json_viewer(data, *, label: str = "Response", expanded: bool = True,
                height: int = 300) -> None:
    """Render JSON with syntax highlighting inside a code block."""
    if data is None:
        st.caption("(empty)")
        return
    txt = pretty_json(data)
    with st.expander(label, expanded=expanded):
        st.code(txt, language="json", line_numbers=False)


def json_editor(default: dict | str | None = None, *, key: str,
                label: str = "Request body (JSON)",
                height: int = 200) -> dict | None:
    """
    Editable JSON textarea.  Returns parsed dict or None if invalid.
    Shows a validation error inline.
    """
    if default is None:
        default = {}
    default_str = pretty_json(default) if not isinstance(default, str) else default

    raw = st.text_area(label, value=default_str, height=height, key=key)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        st.error(f"⚠️ Invalid JSON — {e}")
        return None
