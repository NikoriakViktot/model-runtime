import pandas as pd
import streamlit as st
from ..cp_api import ControlPlaneAPI

def render(cp: ControlPlaneAPI) -> None:
    st.title("🎛️ Orchestration")
    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("🔄 Refresh"):
            st.rerun()
    _ = c2

    runs = cp.runs()
    if not runs:
        st.info("No runs yet")
        return

    df = pd.DataFrame(runs)
    cols = [c for c in ["id", "type", "state", "created_at"] if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)

    st.subheader("🔎 Inspect Run")
    run_id = st.text_input("Run ID")
    if st.button("Load run"):
        if not run_id:
            st.warning("Provide run id")
            return
        data = cp.run(run_id)
        if not data:
            st.error("Run not found / error")
            return
        st.json(data)
