import streamlit as st
import pandas as pd
import time
from ..dispatcher_api import DispatcherAPI


def render(dsp: DispatcherAPI) -> None:
    st.title("🚀 Deployment Manager")

    st.markdown("""
    Use this panel to manage **LoRA Adapters**.  
    1. **Archive:** Models trained and stored in S3.  
    2. **Runtime:** Models currently loaded in vLLM.
    """)

    # --- 1. CONFIGURATION ---
    # Поки що хардкод, в майбутньому можна брати зі списку
    target_base_model = "Qwen/Qwen1.5-1.8B-Chat"

    st.info(f"Managing adapters for base model: **{target_base_model}**")

    col_archive, col_runtime = st.columns(2)

    # --- LEFT: ARCHIVE (Postgres/S3) ---
    with col_archive:
        st.header("📦 Archive (S3)")

        if st.button("🔄 Refresh Archive"):
            st.rerun()

        # Отримуємо список з бази
        try:
            # Викликаємо метод API (переконайся, що він є в dispatcher_api.py)
            archive_runs = dsp.get_available_adapters(base_model=target_base_model)
        except Exception as e:
            st.error(f"Failed to fetch archive: {e}")
            archive_runs = []

        if not archive_runs:
            st.warning("No adapters found in database.")
        else:
            # Формуємо список для відображення
            options = {}
            for r in archive_runs:
                # Витягуємо дані з JSON структури run
                payload = r.get("contract", {}).get("payload", {})
                target_name = payload.get("target_slug") or payload.get("target_name") or "Unknown"
                short_id = r["id"][:8]
                date = r["created_at"][:16].replace("T", " ")

                label = f"[{date}] {target_name} ({short_id})"
                options[label] = r["id"]

            selected_label = st.selectbox("Select Adapter to Deploy:", list(options.keys()))

            if selected_label:
                run_id = options[selected_label]
                st.caption(f"Full ID: `{run_id}`")

                deploy_btn = st.button("🚀 DEPLOY TO VLLM", type="primary", use_container_width=True)

                if deploy_btn:
                    with st.status("Deploying adapter...", expanded=True) as status:
                        st.write("1. Requesting Dispatcher...")
                        try:
                            # Викликаємо Register + Materialize + Restart
                            res = dsp.register_lora(run_id)

                            if res:
                                st.write("2. Artifacts downloaded.")
                                if res.get("vllm_restarted"):
                                    st.write("3. vLLM is restarting (wait ~30s)...")
                                else:
                                    st.write("3. Registered (Hot-load).")

                                status.update(label="✅ Success!", state="complete", expanded=False)
                                st.success(f"Adapter {run_id} deployed successfully!")
                                time.sleep(2)
                                st.rerun()
                            else:
                                status.update(label="❌ API Error", state="error")
                                st.error("No response from Dispatcher.")

                        except Exception as e:
                            status.update(label="❌ Failed", state="error")
                            st.error(f"Deployment failed: {e}")

        # --- RIGHT: RUNTIME (Redis/vLLM) ---
        with col_runtime:
            st.subheader("🟢 Runtime (vLLM)")

            if st.button("🔄 Refresh Runtime"):
                st.rerun()

            # 1. Отримуємо реальний статус
            # Це викличе ланцюжок UI -> Dispatcher -> MRM -> Redis/Docker
            status_data = dsp.get_runtime_status(target_base_model)

            # 2. Індикатор стану
            state = status_data.get("state", "UNKNOWN")
            container = status_data.get("container", "N/A")

            # Візуалізація стану
            if state == "READY":
                st.success(f"State: **{state}**")
            elif state == "STARTING":
                st.info(f"State: **{state}** (Loading model...)")
            elif state in ["UNKNOWN", "UNREACHABLE"]:
                st.error(f"State: **{state}** (Check Logs)")
            else:
                st.warning(f"State: **{state}**")

            st.caption(f"Container ID/Name: `{container}`")

            st.divider()

            # 3. Список активних адаптерів (З REDIS)
            # MRM повернув нам список, який він витягнув з 'smembers'
            active_loras = status_data.get("active_loras", [])

            st.write(f"**Loaded Adapters ({len(active_loras)}):**")

            if active_loras:
                for lora_id in active_loras:
                    # Можна додати кнопку "Unload" (видалення з Redis) в майбутньому
                    st.code(lora_id, language="text")
            else:
                if state == "READY":
                    st.info("Running base model only (No adapters).")
                else:
                    st.info("No runtime info.")

            st.divider()

            # 4. Кнопка примусового рестарту (ПРАЦЮЮЧА)
            st.write("🔧 **Operations**")
            if st.button("⚠️ Force Restart vLLM", type="primary", use_container_width=True):
                with st.spinner("Stopping container & Starting fresh... (This takes time)"):
                    try:
                        # Це справжній виклик API
                        res = dsp.force_restart(target_base_model)

                        if res:
                            st.success("Restart signal accepted!")
                            st.json(res)  # Покажемо відповідь MRM для певності
                            time.sleep(3)
                            st.rerun()
                        else:
                            st.error("Failed to restart (No response).")
                    except Exception as e:
                        st.error(f"Error: {e}")