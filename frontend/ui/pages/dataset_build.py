# frontend/ui/pages/dataset_build.py

import streamlit as st
from ..config import S3_BUCKET, MODEL_MAP
from ..cp_api import ControlPlaneAPI
from ..dispatcher_api import DispatcherAPI


def render(cp: ControlPlaneAPI, dsp: DispatcherAPI) -> None:
    st.title("📦 Dataset Build")
    st.caption("Submit dataset.build contracts. Upload DB -> choose target -> build dataset.")

    # --- КРОК 1: Вибір джерела ---
    st.subheader("1) Source Data")

    source_type = st.radio("Source Type", ["Remote DB (Postgres)", "Upload File (SQLite)"], horizontal=True)

    db_id = None
    target_name = None

    if source_type == "Upload File (SQLite)":
        uploaded = st.file_uploader("Upload SQLite DB (.db)", type=["db"])

        # Якщо файл вже завантажений раніше і збережений в session_state, використовуємо його
        if uploaded:
            # Завжди перезаливаємо, якщо користувач вибрав новий файл
            meta = dsp.upload_db(uploaded.getvalue(), uploaded.name)
            if meta and meta.get("file_id"):
                st.session_state.uploaded_db_id = meta["file_id"]
                st.success(f"Uploaded: {st.session_state.uploaded_db_id}")

        # Перевіряємо, чи є ID завантаженого файлу
        if "uploaded_db_id" in st.session_state:
            db_id = st.session_state.uploaded_db_id

            # Для SQLite треба вибрати ціль з файлу
            targets = dsp.etl_targets(db_id)
            if targets:
                t_map = {f"{t.get('epitaph', '?')} ({t.get('cnt', '?')} rows)": t.get("epitaph") for t in targets if
                         t.get("epitaph")}
                sel = st.selectbox("Target Persona inside DB", list(t_map.keys()))
                if sel:
                    target_name = t_map[sel]
            else:
                st.warning("No targets found in uploaded DB.")

    else:  # Remote DB (Postgres)
        # Отримуємо список з бекенду
        epitaphs = dsp.list_remote_epitaphs()

        if not epitaphs:
            st.warning("No remote epitaphs found or connection failed.")
        else:
            # Створюємо зручний список для вибору
            # Формат: "Name (Status) - ID"
            options = {f"{e.get('target_identifier', 'Unknown')} ({e.get('status', 'N/A')}) - {e.get('id')}": e for e in
                       epitaphs}

            selected_label = st.selectbox("Select Epitaph from Project 1", list(options.keys()))

            if selected_label:
                selected_epitaph = options[selected_label]
                db_id = selected_epitaph['id']
                target_name = selected_epitaph['target_identifier']
                st.info(f"Selected ID: {db_id}")
                st.caption(f"Target Name: {target_name}")

    if not db_id or not target_name:
        st.info("Please select a valid source and target to proceed.")
        # Не використовуємо st.stop(), щоб показати решту UI (але заблокувати кнопку)
        # Або можна використати st.stop(), якщо хочете приховати наступні кроки

    # Зберігаємо в session_state для використання при кліку
    st.session_state.db_id = db_id
    st.session_state.target_name = target_name

    # --- КРОК 2: Вибір Моделі ---
    st.subheader("2) Model Configuration")

    @st.cache_data(ttl=30)
    def _load_models():
        return dsp.list_models()

    models = _load_models()
    if not models:
        st.warning("No models returned from /v1/models")
        return

    aliases = [m["id"] for m in models if "id" in m]
    # Фільтруємо ті, що є в нашому мапінгу
    aliases = [a for a in aliases if a in MODEL_MAP]

    if not aliases:
        st.error("No selectable models: none of /v1/models are present in MODEL_MAP")
        st.write("LiteLLM returned:", [m.get("id") for m in models])
        st.write("MODEL_MAP keys:", list(MODEL_MAP.keys()))
        return

    sel_alias = st.selectbox("Generator model (alias)", aliases, index=0)

    st.session_state.llm_model = sel_alias
    st.session_state.base_model = MODEL_MAP[sel_alias]
    st.caption(f"MRM base_model: {st.session_state.base_model}")

    # --- КРОК 3: Тип датасету ---
    st.subheader("3) Dataset Settings")
    ds_type = st.radio("Dataset Type", ["graph", "sft", "prefs", "linear"], horizontal=True)

    prompt_id = "etl.profile_generator"
    prompt_ver = "latest"

    if ds_type == "linear":
        with st.expander("Prompt Configuration", expanded=True):
            prompt_id = st.text_input("Prompt Family ID", value=prompt_id)
            prompt_ver = st.text_input("Version Tag", value=prompt_ver)

    gen_rejected = False
    max_items = 500

    with st.expander("Options", expanded=False):
        gen_rejected = st.checkbox("Generate Rejected Samples", value=(ds_type == "prefs"))
        max_items = st.number_input("Max Items", 10, 100000, 500, step=50)

    # --- SUBMIT BUTTON ---
    # Блокуємо кнопку, якщо дані не вибрані
    btn_disabled = not (st.session_state.db_id and st.session_state.target_name)

    if st.button("🚀 Submit dataset.build", type="primary", disabled=btn_disabled):
        payload = {
            "base_model": st.session_state.base_model,
            "db_id": st.session_state.db_id,
            "target_name": st.session_state.target_name,
            "dataset_type": ds_type,
            "options": {
                "generate_rejected": gen_rejected,
                "prompt_id": prompt_id,
                "prompt_version": prompt_ver,
                "rejected_max_items": max_items,
            },
            "output": {"base_uri": f"s3://{S3_BUCKET}/datasets"},
        }

        # Для дебагу
        # st.write("Payload:", payload)

        res = cp.post_contract("dataset.build.v1", payload)
        if not res:
            st.error("Contract submit failed")
            return

        run_id = res.get("run_id") or res.get("id")
        st.session_state.last_run_id = run_id
        st.success(f"Run created: {run_id}")
        st.json(res)

    # --- КРОК 4: Статус ---
    if st.session_state.get("last_run_id"):
        st.divider()
        st.subheader("4) Build Status")

        rid = st.session_state.last_run_id
        st.code(rid)

        run = cp.run(rid)
        if not run:
            st.error("Failed to load run")
            return

        state = run.get("state", "UNKNOWN")
        st.info(f"State: {state}")

        artifacts = run.get("artifacts") or {}
        if artifacts:
            st.json(artifacts)

        if state == "DATASET_READY":
            if st.button("📥 Download dataset locally"):
                dsp.preview(rid, limit=0)
                st.success("Dataset downloaded and cached")

        # Відображення семплів
        if artifacts and "dataset_sample" in artifacts:
            st.subheader("🔍 Dataset sample (used for training)")
            for i, row in enumerate(artifacts.get("dataset_sample", [])):
                with st.expander(f"Sample #{i + 1}"):
                    st.json(row)

        if st.button("🔄 Refresh status"):
            st.rerun()