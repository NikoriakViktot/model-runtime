"""
ui/pages/api_playground.py

Full API Playground — browse every service endpoint,
edit the request body, fire it, inspect response + latency.
"""
from __future__ import annotations

import json
import time

import streamlit as st

from ui.components.json_viewer import json_editor, json_viewer
from ui.components.metrics import api_result_header
from ui.services.base import BaseClient
from ui.utils import state as S
from ui.utils.formatters import pretty_json

# ── Endpoint catalogue ────────────────────────────────────────────────────────
#  Each entry: {service, method, path, desc, default_body, timeout}

_ENDPOINTS: list[dict] = [
    # Gateway
    {"service": "Gateway", "method": "GET",  "path": "/health",
     "desc": "Liveness probe", "default_body": None, "timeout": 5},
    {"service": "Gateway", "method": "GET",  "path": "/ready",
     "desc": "Readiness probe (checks MRM/Scheduler)", "default_body": None, "timeout": 5},
    {"service": "Gateway", "method": "GET",  "path": "/v1/models",
     "desc": "List available models", "default_body": None, "timeout": 5},
    {"service": "Gateway", "method": "POST", "path": "/v1/chat/completions",
     "desc": "OpenAI-compatible chat inference",
     "default_body": {"model": "Qwen/Qwen1.5-1.8B-Chat",
                      "messages": [{"role": "user", "content": "Hello!"}],
                      "temperature": 0.7, "max_tokens": 128,
                      "runtime_preference": "auto"}, "timeout": 120},
    {"service": "Gateway", "method": "GET",  "path": "/v1/slo",
     "desc": "Fleet SLO snapshot", "default_body": None, "timeout": 5},
    {"service": "Gateway", "method": "GET",  "path": "/v1/router/metrics",
     "desc": "Per-instance router metrics", "default_body": None, "timeout": 5},
    {"service": "Gateway", "method": "GET",  "path": "/admin/status",
     "desc": "Admin status overview", "default_body": None, "timeout": 5},

    # MRM
    {"service": "MRM", "method": "GET",  "path": "/health",
     "desc": "MRM liveness", "default_body": None, "timeout": 5},
    {"service": "MRM", "method": "GET",  "path": "/models/status",
     "desc": "All model states", "default_body": None, "timeout": 5},
    {"service": "MRM", "method": "GET",  "path": "/gpu/metrics",
     "desc": "GPU memory / utilisation", "default_body": None, "timeout": 5},
    {"service": "MRM", "method": "POST", "path": "/models/ensure",
     "desc": "Ensure model is running (cold-start safe)",
     "default_body": {"base_model": "Qwen/Qwen1.5-1.8B-Chat"}, "timeout": 660},
    {"service": "MRM", "method": "POST", "path": "/models/touch",
     "desc": "Reset idle timer to prevent eviction",
     "default_body": {"base_model": "Qwen/Qwen1.5-1.8B-Chat"}, "timeout": 10},
    {"service": "MRM", "method": "POST", "path": "/models/stop",
     "desc": "Stop a running model container",
     "default_body": {"base_model": "Qwen/Qwen1.5-1.8B-Chat"}, "timeout": 60},
    {"service": "MRM", "method": "POST", "path": "/models/remove",
     "desc": "Remove container + clear Redis state",
     "default_body": {"base_model": "Qwen/Qwen1.5-1.8B-Chat"}, "timeout": 60},
    {"service": "MRM", "method": "GET",  "path": "/litellm/config",
     "desc": "Current LiteLLM config", "default_body": None, "timeout": 5},
    {"service": "MRM", "method": "POST", "path": "/litellm/reload",
     "desc": "Reload LiteLLM config", "default_body": {}, "timeout": 15},
    {"service": "MRM", "method": "GET",  "path": "/hf/search",
     "desc": "Search HuggingFace Hub",
     "default_body": None, "params": {"q": "llama 7b", "limit": "10"}, "timeout": 20},
    {"service": "MRM", "method": "GET",  "path": "/hf/recommend",
     "desc": "GPU-aware model recommendations",
     "default_body": None, "params": {"q": "small instruct", "gpu_id": "0", "limit": "10"}, "timeout": 20},
    {"service": "MRM", "method": "POST", "path": "/models/register_from_hf",
     "desc": "Register HF model from HuggingFace",
     "default_body": {"repo_id": "Qwen/Qwen1.5-1.8B-Chat",
                      "gpu": "0", "overrides": {}}, "timeout": 30},

    # Scheduler
    {"service": "Scheduler", "method": "GET",  "path": "/health",
     "desc": "Scheduler liveness", "default_body": None, "timeout": 5},
    {"service": "Scheduler", "method": "GET",  "path": "/nodes",
     "desc": "List all registered nodes", "default_body": None, "timeout": 5},
    {"service": "Scheduler", "method": "GET",  "path": "/placements",
     "desc": "All active model placements", "default_body": None, "timeout": 5},
    {"service": "Scheduler", "method": "POST", "path": "/schedule/ensure",
     "desc": "Place model on a node",
     "default_body": {"model": "Qwen/Qwen1.5-1.8B-Chat", "runtime_preference": "auto"}, "timeout": 120},
    {"service": "Scheduler", "method": "POST", "path": "/schedule/stop",
     "desc": "Stop model on all nodes",
     "default_body": {"model": "Qwen/Qwen1.5-1.8B-Chat"}, "timeout": 30},
    {"service": "Scheduler", "method": "POST", "path": "/heartbeat",
     "desc": "Simulate a node heartbeat",
     "default_body": {"node_id": "test-node-01", "agent_url": "http://node_agent:8020",
                      "hostname": "test-host", "gpus": [], "has_gpu": False,
                      "cpu_cores": 4, "cpu_load": 0.2}, "timeout": 10},

    # Node Agent
    {"service": "Node Agent", "method": "GET",  "path": "/health",
     "desc": "Node Agent liveness", "default_body": None, "timeout": 5},
    {"service": "Node Agent", "method": "GET",  "path": "/local/status",
     "desc": "All models on this node", "default_body": None, "timeout": 5},
    {"service": "Node Agent", "method": "POST", "path": "/local/ensure",
     "desc": "Ensure model on this node via local MRM",
     "default_body": {"model": "Qwen/Qwen1.5-1.8B-Chat"}, "timeout": 660},
    {"service": "Node Agent", "method": "POST", "path": "/local/stop",
     "desc": "Stop model on this node",
     "default_body": {"model": "Qwen/Qwen1.5-1.8B-Chat"}, "timeout": 30},
    {"service": "Node Agent", "method": "POST", "path": "/heartbeat",
     "desc": "Force immediate heartbeat to Scheduler",
     "default_body": {}, "timeout": 10},

    # Control Plane
    {"service": "Control Plane", "method": "GET",  "path": "/health",
     "desc": "Control Plane liveness", "default_body": None, "timeout": 5},
    {"service": "Control Plane", "method": "GET",  "path": "/runs",
     "desc": "List all orchestration runs", "default_body": None, "timeout": 5},
    {"service": "Control Plane", "method": "GET",  "path": "/events",
     "desc": "Recent run events", "default_body": None, "timeout": 5},
    {"service": "Control Plane", "method": "POST", "path": "/contracts",
     "desc": "Submit a new contract",
     "default_body": {"type": "dataset.build.v1", "spec_version": "v1",
                      "payload": {"target_name": "my-dataset", "source_type": "hf",
                                  "hf_dataset": "tatsu-lab/alpaca", "max_rows": 1000}},
     "timeout": 15},
]

_SERVICE_URLS: dict[str, str] = {
    "Gateway":      __import__("os").getenv("GATEWAY_URL",       "http://gateway:8080"),
    "MRM":          __import__("os").getenv("MRM_URL",           "http://model_runtime_manager:8010"),
    "Scheduler":    __import__("os").getenv("SCHEDULER_URL",     "http://scheduler:8030"),
    "Node Agent":   __import__("os").getenv("NODE_AGENT_URL",    "http://node_agent:8020"),
    "Control Plane":__import__("os").getenv("CONTROL_PLANE_URL", "http://control_plane:8004"),
}

_SERVICE_COLORS: dict[str, str] = {
    "Gateway": "🔵", "MRM": "🟣", "Scheduler": "🟠",
    "Node Agent": "🟡", "Control Plane": "🟢",
}

_METHOD_COLORS: dict[str, str] = {
    "GET": "🔵", "POST": "🟢", "DELETE": "🔴", "PUT": "🟡",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _call(service: str, method: str, path: str,
          body: dict | None, params: dict | None, timeout: int):
    base_url = _SERVICE_URLS.get(service, "")
    client = BaseClient(base_url, default_timeout=timeout)
    if method == "GET":
        return client._get(path, params=params, timeout=timeout)
    elif method == "POST":
        return client._post(path, body=body, timeout=timeout)
    elif method == "DELETE":
        return client._delete(path, timeout=timeout)
    return None


# ── Render ────────────────────────────────────────────────────────────────────

def render():
    st.title("🔧 API Playground")
    st.caption("Browse, edit and fire every endpoint across all services.")

    # ── Template save/load in sidebar ─────────────────────────────────
    with st.sidebar:
        st.subheader("📋 Saved templates")
        templates: dict = S.get("playground_templates") or {}
        if templates:
            tpl_name = st.selectbox("Load template", ["— select —"] + list(templates.keys()),
                                    key="pg_tpl_sel")
            if tpl_name != "— select —" and st.button("Load", key="pg_tpl_load"):
                tpl = templates[tpl_name]
                S.set("playground_last_request", tpl)
                st.success(f"Loaded `{tpl_name}`")

        last = S.get("playground_last_request")
        if last:
            tpl_save_name = st.text_input("Save as template", key="pg_tpl_name")
            if st.button("💾 Save", key="pg_tpl_save_btn") and tpl_save_name:
                templates[tpl_save_name] = last
                S.set("playground_templates", templates)
                st.success("Saved!")

        st.divider()
        if st.button("🔁 Replay last", key="pg_replay"):
            S.set("playground_replay", True)

    # ── Service + endpoint selection ──────────────────────────────────
    services = sorted(set(e["service"] for e in _ENDPOINTS))
    col_svc, col_ep = st.columns([1, 2])

    with col_svc:
        service = st.selectbox(
            "Service",
            services,
            format_func=lambda s: f"{_SERVICE_COLORS.get(s,'⚪')} {s}",
            key="pg_service",
        )

    filtered = [e for e in _ENDPOINTS if e["service"] == service]

    with col_ep:
        endpoint_labels = [
            f"{_METHOD_COLORS.get(e['method'],'⚪')} {e['method']:6} {e['path']}  —  {e['desc']}"
            for e in filtered
        ]
        ep_idx = st.selectbox("Endpoint", range(len(filtered)),
                              format_func=lambda i: endpoint_labels[i],
                              key="pg_endpoint")

    ep = filtered[ep_idx]
    method = ep["method"]
    path   = ep["path"]

    st.caption(
        f"`{_SERVICE_URLS.get(service, '?')}{path}`  ·  "
        f"timeout {ep.get('timeout', 10)}s"
    )

    # ── Request editor ────────────────────────────────────────────────
    params_edited: dict | None = None
    body_edited:   dict | None = None

    if method == "GET" and ep.get("params"):
        st.subheader("Query parameters")
        params_raw: str = st.text_area(
            "Params (JSON object)",
            value=pretty_json(ep.get("params", {})),
            height=80,
            key="pg_params",
        )
        try:
            params_edited = json.loads(params_raw)
        except Exception:
            st.error("Invalid JSON for params")
            params_edited = None

    elif method in ("POST", "PUT") and ep.get("default_body") is not None:
        st.subheader("Request body")
        body_edited = json_editor(
            default=ep.get("default_body"),
            key="pg_body",
            height=220,
        )

    # ── Fire button ───────────────────────────────────────────────────
    col_fire, col_clear = st.columns([1, 5])
    with col_fire:
        fire = st.button("▶ Send", type="primary", key="pg_send")
    with col_clear:
        if st.button("🗑 Clear response", key="pg_clear"):
            S.set("playground_last_response", None)

    # Replay
    if S.get("playground_replay"):
        fire = True
        S.set("playground_replay", False)

    # ── Execute ───────────────────────────────────────────────────────
    if fire:
        req_record = {
            "service": service, "method": method, "path": path,
            "body": body_edited, "params": params_edited,
        }
        S.set("playground_last_request", req_record)

        with st.spinner(f"Calling {service} {method} {path}…"):
            result = _call(
                service=service,
                method=method,
                path=path,
                body=body_edited,
                params=params_edited,
                timeout=ep.get("timeout", 15),
            )
        S.set("playground_last_response", result)

    # ── Response display ──────────────────────────────────────────────
    result = S.get("playground_last_response")
    if result is None:
        st.info("Hit **▶ Send** to execute the request.")
        return

    st.divider()
    st.subheader("Response")

    # Status row
    api_result_header(result, service=service)

    if not result.ok:
        st.error(f"**Error:** {result.error}")

    # Tabs: response body / request echo / curl
    tab_resp, tab_req, tab_curl = st.tabs(["📦 Response", "📤 Request", "🐚 curl"])

    with tab_resp:
        json_viewer(result.data, label="Response body", expanded=True)

    with tab_req:
        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("URL")
            st.code(result.url)
            st.caption("Method")
            st.code(result.method)
        with col_b:
            if result.request_body is not None:
                st.caption("Body sent")
                st.code(pretty_json(result.request_body), language="json")

    with tab_curl:
        base = _SERVICE_URLS.get(service, "")
        if result.method == "GET":
            curl = f"curl -s '{result.url}'"
        else:
            body_str = pretty_json(result.request_body or {}).replace("'", "'\\''")
            curl = (
                f"curl -s -X POST '{result.url}' \\\n"
                f"  -H 'Content-Type: application/json' \\\n"
                f"  -d '{body_str}'"
            )
        st.code(curl, language="bash")
