import streamlit as st
import pandas as pd
from ..config import MODEL_MAP, S3_BUCKET
from ..cp_api import ControlPlaneAPI
from ..dispatcher_api import DispatcherAPI


def render(cp: ControlPlaneAPI, dsp: DispatcherAPI) -> None:
    st.title("🔥 Training")

    st.subheader("Dataset Run ID")

    run_id = st.text_input("Dataset run_id", value=st.session_state.get("last_run_id", ""))

    if not run_id:
        st.info("Enter dataset run_id or build one in Dataset Build.")
        return

    run = cp.run(run_id)
    if not run:
        st.error("Run not found / failed to load")
        return
    contract = run.get("contract", {})
    ds_payload = contract.get("payload", {})

    target_slug = ds_payload.get("target_name")
    if not target_slug:
        st.error("Dataset run has no target_name")
        return


    artifacts = run.get("artifacts", {})
    ds_uri = (
            artifacts.get("dataset_uri")
            or artifacts.get("s3_uri")
    )
    if not ds_uri:
        st.error("Dataset URI not found in run artifacts")
        return

    # 🔥 ГАРАНТІЯ: датасет завжди буде локально
    if st.button("📥 Load dataset (auto)"):
        data = dsp.preview(run_id, limit=3)
        st.success("Dataset available locally")

        for i, sample in enumerate(data.get("samples", [])):
            with st.expander(f"Sample #{i+1}"):
                st.json(sample)

    st.divider()

    alias = st.selectbox("Base model (alias)", list(MODEL_MAP.keys()), index=0)
    base_model = MODEL_MAP[alias]
    st.caption(f"MRM base_model: {base_model}")
    epochs = st.slider("Epochs", 1, 10, 1)
    lr = st.select_slider(
        "Learning Rate",
        options=[5e-6, 1e-5, 2e-5, 5e-5, 1e-4],
        value=2e-5
    )
    batch = st.number_input("Batch size", 1, 16, 1)

    if st.button("🚀 Start training (QLoRA)", type="primary"):
        # 🔥 ще раз ensure (idempotent)
        dsp.preview(run_id, limit=0)

        res = cp.post_contract(
            "train.qlora.v1",
            {
                "parent_run_id": run_id,  # ← ОБОВʼЯЗКОВО
                "target_slug": target_slug,
                "base_model": base_model,
                "dataset": {"uri": ds_uri},
                "training": {
                    "epochs": int(epochs),
                    "learning_rate": float(lr),
                    "batch_size": int(batch),
                },
                "output": {
                    "lora_base_uri": f"s3://{S3_BUCKET}/loras"
                },
            }
        )

        with st.expander("🔎 Debug info", expanded=True):
            st.write("Run ID:", run_id)
            st.write("Target:", target_slug)
            st.write("Dataset URI:", ds_uri)
            st.json(run)

        st.success("Training started")
        st.json(res)
