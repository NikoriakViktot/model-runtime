"""
ui/theme.py
Dark theme CSS injected once at app startup.
"""
import streamlit as st

_CSS = """
<style>
/* ── Base ──────────────────────────────────────────────── */
:root {
  --bg:        #0d1117;
  --surface:   #161b22;
  --surface2:  #21262d;
  --border:    #30363d;
  --text:      #e6edf3;
  --text-dim:  #8b949e;
  --accent:    #58a6ff;
  --accent2:   #3fb950;
  --warn:      #d29922;
  --danger:    #f85149;
  --cpu:       #a371f7;
  --gpu:       #58a6ff;
}

/* Page background */
.stApp, [data-testid="stAppViewContainer"] {
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
  background-color: var(--surface) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Metrics */
[data-testid="stMetric"] {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: var(--text-dim) !important; font-size: 12px; }
[data-testid="stMetricValue"] { color: var(--text) !important; font-size: 22px; font-weight: 700; }

/* Expanders */
[data-testid="stExpander"] {
  background: var(--surface);
  border: 1px solid var(--border) !important;
  border-radius: 8px;
}
[data-testid="stExpander"] summary { color: var(--text) !important; }

/* Data tables */
[data-testid="stDataFrame"] { background: var(--surface); border-radius: 8px; }
[data-testid="stDataFrame"] thead th {
  background: var(--surface2) !important;
  color: var(--text-dim) !important;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
[data-testid="stDataFrame"] tbody td { color: var(--text) !important; }

/* Tabs */
[data-testid="stTabs"] [role="tab"] {
  background: transparent !important;
  color: var(--text-dim) !important;
  border-bottom: 2px solid transparent;
  font-weight: 500;
  padding: 8px 16px;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
}

/* Buttons */
.stButton > button {
  background: var(--surface2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  font-weight: 500;
  transition: all .15s ease;
}
.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}
.stButton > button[kind="primary"] {
  background: var(--accent) !important;
  color: #0d1117 !important;
  border-color: var(--accent) !important;
  font-weight: 600;
}
.stButton > button[kind="primary"]:hover {
  background: #79c0ff !important;
}

/* Inputs / Selects / Sliders */
.stTextInput input, .stTextArea textarea, .stSelectbox select,
[data-testid="stNumberInput"] input {
  background: var(--surface2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(88,166,255,.2) !important;
}
[data-testid="stSlider"] [data-testid="stSliderThumb"] { background: var(--accent) !important; }

/* Code blocks */
.stCodeBlock pre, code {
  background: var(--surface2) !important;
  border: 1px solid var(--border);
  border-radius: 6px;
  font-family: 'JetBrains Mono', 'Fira Code', monospace;
  font-size: 13px;
}

/* Chat messages */
[data-testid="stChatMessage"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  margin-bottom: 8px;
}
[data-testid="stChatMessage"][data-testid*="user"] {
  border-left: 3px solid var(--accent) !important;
}
[data-testid="stChatMessage"][data-testid*="assistant"] {
  border-left: 3px solid var(--accent2) !important;
}

/* Alert boxes */
[data-testid="stAlert"][data-baseweb="notification"] {
  background: var(--surface2) !important;
  border-radius: 8px !important;
}

/* Section dividers */
hr { border-color: var(--border) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* Custom badges */
.badge-gpu  { color: var(--gpu);  font-weight: 700; }
.badge-cpu  { color: var(--cpu);  font-weight: 700; }
.badge-ok   { color: var(--accent2); font-weight: 700; }
.badge-warn { color: var(--warn); font-weight: 700; }
.badge-err  { color: var(--danger); font-weight: 700; }

/* Page header */
.page-title {
  font-size: 24px;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 4px;
}
.page-subtitle {
  font-size: 13px;
  color: var(--text-dim);
  margin-bottom: 20px;
}

/* Service status pill */
.status-pill {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 600;
}
.status-pill.up   { background: rgba(63,185,80,.15); color: var(--accent2); border: 1px solid var(--accent2); }
.status-pill.down { background: rgba(248,81,73,.15);  color: var(--danger);  border: 1px solid var(--danger); }
</style>
"""


def apply() -> None:
    """Call once at the top of app.py."""
    st.markdown(_CSS, unsafe_allow_html=True)
