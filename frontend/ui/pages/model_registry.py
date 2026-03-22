import streamlit as st
from ..config import MRM_BASE_URL
from ..mrm_api import MRMApi


def render() -> None:
    st.title("🧬 Model Registry (MRM)")
    api = MRMApi(MRM_BASE_URL)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("🔄 Refresh"):
            st.rerun()
    with col_b:
        st.caption(MRM_BASE_URL)

    try:
        models = api.status_all()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        return

    for m in models:
        bm = m.get("base_model", "")
        state = m.get("state", "")
        gpu = m.get("gpu", "")
        container = m.get("container", "")
        api_base = m.get("api_base", "")

        with st.container(border=True):
            st.markdown(f"### {bm}")
            st.write(f"**State:** {state} | **GPU:** {gpu} | **Container:** `{container}`")
            st.code(api_base)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("🚀 Ensure", key=f"ensure:{bm}"):
                    try:
                        st.success(api.ensure(bm))
                    except Exception as e:
                        st.error(str(e))
            with c2:
                if st.button("🛑 Stop", key=f"stop:{bm}"):
                    try:
                        st.success(api.stop(bm))
                    except Exception as e:
                        st.error(str(e))
            with c3:
                if st.button("🧹 Remove", key=f"remove:{bm}"):
                    try:
                        st.success(api.remove(bm))
                    except Exception as e:
                        st.error(str(e))
            with c4:
                if st.button("ℹ️ Status", key=f"status:{bm}"):
                    try:
                        st.info(api.status(bm))
                    except Exception as e:
                        st.error(str(e))
