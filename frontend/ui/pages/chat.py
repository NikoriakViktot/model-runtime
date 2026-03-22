# =========================
# streamlit/ui_chat.py
# =========================
import streamlit as st
import time
from ..dispatcher_api import DispatcherAPI


def render(dsp: DispatcherAPI) -> None:
    st.title("💬 Neuro-Semantic Chat")

    with st.sidebar:
        st.header("⚙️ Configuration")

        mode = st.radio("Inference Mode", ["🤖 Base Model", "🧬 LoRA Adapter"], index=1)
        base_model_name = "Qwen/Qwen1.5-1.8B-Chat"
        selected_model_id = None

        if mode == "🤖 Base Model":
            selected_model_id = f"base:{base_model_name}"
        else:
            try:
                # Отримуємо список доступних адаптерів
                adapters = dsp.get_available_adapters(base_model=base_model_name)
            except Exception as e:
                st.error(f"API Error: {e}")
                adapters = []

            opts = {}
            for a in adapters or []:
                rid = a.get("run_id") or a.get("id")
                if not rid:
                    continue
                name = a.get("name")
                if not name:
                    payload = a.get("contract", {}).get("payload", {})
                    name = payload.get("target_slug") or payload.get("target_name")
                if not name:
                    name = "Unknown Adapter"
                date = (a.get("created_at") or "")[:10]
                opts[f"{name} ({date}) [{rid[:6]}]"] = rid

            if opts:
                selected_model_id = opts[st.selectbox("Select Adapter", list(opts.keys()))]
                st.caption(f"Target Run ID: `{selected_model_id}`")
            else:
                st.warning("No adapters found.")

        st.divider()

        # --- Runtime Status Checker ---
        with st.expander("🔍 Runtime Health"):
            if st.button("Refresh Active LoRAs"):
                try:
                    st_info = dsp.get_runtime_status(base_model_name)
                    st.write(f"**State:** {st_info.get('state')}")
                    st.write(f"**GPU:** {st_info.get('gpu')}")

                    active = st_info.get("active_loras", [])
                    st.write(f"**Loaded LoRAs ({len(active)}):**")
                    if active:
                        for lora in active:
                            if lora == selected_model_id:
                                st.success(f"✅ {lora} (Selected)")
                            else:
                                st.code(lora)
                    else:
                        st.info("No LoRAs currently loaded in VRAM.")
                except Exception as e:
                    st.error(f"Status check failed: {e}")

        st.divider()
        st.subheader("🎛️ Parameters")
        use_rag = st.toggle("Enable Graph RAG", value=True)
        rag_k = st.slider("Context Facts", 1, 10, 3) if use_rag else 0
        temp = st.slider("Temperature", 0.0, 1.5, 0.5)
        max_tok = st.slider("Max Tokens", 2, 200, 128)

        # Optional System Prompt
        use_custom_system = st.toggle("Override system prompt", value=False)
        system_prompt = None
        if use_custom_system:
            system_prompt = st.text_area(
                "system_prompt",
                value="Reply briefly and naturally. Be direct. No fluff.",
                height=100,
            )

    # --- Chat History Rendering ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                # Якщо ми зберегли інфу про модель, покажемо її
                if "model_used" in msg:
                    st.caption(f"⚙️ {msg['model_used']}")

    # --- Input Handling ---
    if prompt := st.chat_input("Send a message..."):
        if not selected_model_id:
            st.error("Please select a model/adapter first.")
            return

        # Додаємо повідомлення користувача
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

        # --- Processing Block ---
        with st.status("🚀 Processing...", expanded=True) as status:
            t0 = time.time()
            try:
                # 1. UX Feedback: Routing
                status.write("📡 Routing request to Dispatcher...")

                # 2. UX Feedback: LoRA Specifics
                if mode == "🧬 LoRA Adapter":
                    status.write(f"🧬 Targeting Adapter: `{selected_model_id}`")
                    status.write("⏳ Ensuring adapter is loaded (first run takes ~10s, subsequent ~1s)...")

                # 3. API Call
                resp = dsp.chat(
                    model=selected_model_id,
                    messages=api_messages,
                    temperature=temp,
                    max_tokens=max_tok,
                    use_rag=use_rag,
                    rag_k=rag_k,
                    system_prompt=system_prompt,
                    route="direct",  # 👈 force vLLM
                )

                dt = time.time() - t0

                # 4. Extract Response
                content = resp["choices"][0]["message"]["content"]

                # 5. VERIFICATION: Яка модель реально відповіла?
                real_model = resp.get("model", "unknown")

                # Логування успіху/невідповідності
                if real_model == selected_model_id:
                    status.write(f"✅ **Confirmed:** Response generated by `{real_model}`")
                elif mode == "🧬 LoRA Adapter":
                    status.warning(f"⚠️ **Mismatch:** Requested `{selected_model_id}`, but answered by `{real_model}`")

                status.update(label=f"✅ Answered in {dt:.2f}s via {real_model}", state="complete", expanded=False)

                # 6. Save & Display Assistant Response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": content,
                    "model_used": real_model  # Зберігаємо для історії
                })

                with st.chat_message("assistant"):
                    st.write(content)
                    st.caption(f"⚙️ Model: {real_model} | ⏱️ {dt:.2f}s")

            except Exception as e:
                status.update(label="💥 Exception", state="error", expanded=True)
                st.error(f"Error during inference: {e}")