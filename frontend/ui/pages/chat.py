"""
ui/pages/chat.py
Inference Chat — full-featured chat with parameter control and cost display.
"""
from __future__ import annotations
import time
import streamlit as st
from ui.services.gateway_client import gateway
from ui.components.metrics import api_result_header, cost_card
from ui.utils import state as S
from ui.utils.cost import estimate_gpu, estimate_cpu


_BASE_MODELS = [
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


def render():
    st.title("💬 Inference Chat")

    # ── Sidebar controls ──────────────────────────────────────────────
    with st.sidebar:
        st.subheader("⚙️ Model")
        models_result = gateway.list_models()
        live = []
        if models_result.ok and isinstance(models_result.data, dict):
            live = [m.get("id", "") for m in models_result.data.get("data", []) if m.get("id")]
        options = live or _BASE_MODELS
        selected_model = st.selectbox("Model", options, key="chat_model_sel")

        st.subheader("🎛 Parameters")
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
        max_tokens  = st.slider("Max tokens",  0, 4096, 512, 64,
                                help="0 = let model decide")
        runtime_pref = st.selectbox("Runtime preference",
                                    ["auto", "gpu", "cpu"],
                                    help="auto = gateway decides based on request size")

        st.subheader("📝 System prompt")
        sys_msg = st.text_area("System prompt", height=100, key="chat_sys_prompt",
                               placeholder="You are a helpful assistant…")

        st.divider()
        if st.button("🗑️ Clear conversation"):
            S.set("messages", [])
            st.rerun()

    # ── Chat history ──────────────────────────────────────────────────
    for msg in (S.get("messages") or []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "meta" in msg:
                m = msg["meta"]
                st.caption(
                    f"⏱ {m.get('latency_ms', '?')} ms  ·  "
                    f"runtime: {m.get('runtime_type', '?')}  ·  "
                    f"tokens: {m.get('completion_tokens', '?')}"
                )

    # ── Input ─────────────────────────────────────────────────────────
    prompt = st.chat_input("Send a message…")
    if not prompt:
        return

    # Build message list to send
    outgoing = []
    if sys_msg.strip():
        outgoing.append({"role": "system", "content": sys_msg})
    outgoing += [{"role": m["role"], "content": m["content"]}
                 for m in S.get("messages")]
    outgoing.append({"role": "user", "content": prompt})

    S.push("messages", {"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = gateway.chat(
                model=selected_model,
                messages=outgoing,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens > 0 else 512,
                runtime_preference=runtime_pref,
            )

        api_result_header(result, service="Gateway")

        if not result.ok:
            st.error(f"**{result.error}**")
            if result.data:
                st.json(result.data)
            return

        try:
            choice = result.data["choices"][0]
            content = choice["message"]["content"]
            usage = result.data.get("usage", {})
            rt = result.data.get("runtime_type", runtime_pref if runtime_pref != "auto" else "gpu")
        except (KeyError, IndexError, TypeError) as e:
            st.error(f"Unexpected response format: {e}")
            st.json(result.data)
            return

        st.markdown(content)

        # Cost estimate
        comp_tokens = usage.get("completion_tokens", max_tokens)
        est = (estimate_cpu if rt == "cpu" else estimate_gpu)(result.latency_ms, comp_tokens)
        with st.expander("📊 Request stats", expanded=False):
            cost_card(est)
            cols = st.columns(3)
            cols[0].metric("Prompt tokens",  usage.get("prompt_tokens", "?"))
            cols[1].metric("Output tokens",  usage.get("completion_tokens", "?"))
            cols[2].metric("Total tokens",   usage.get("total_tokens", "?"))

        meta = {
            "latency_ms": f"{result.latency_ms:.0f}",
            "runtime_type": rt,
            "completion_tokens": usage.get("completion_tokens", "?"),
        }
        S.push("messages", {"role": "assistant", "content": content, "meta": meta})
