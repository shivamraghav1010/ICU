import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Stroke Family Resilience",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME / CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  [data-testid="stSidebar"] { background: #0f172a; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

  [data-testid="metric-container"] {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 16px;
  }

  .section-header {
      background: linear-gradient(135deg, #1e3a5f 0%, #1e293b 100%);
      border-left: 4px solid #38bdf8;
      padding: 12px 20px;
      border-radius: 8px;
      margin: 20px 0 12px 0;
      color: #e2e8f0;
      font-size: 1.1rem;
      font-weight: 600;
  }

  .hero {
      background: linear-gradient(135deg, #0c1445 0%, #1e3a5f 50%, #0f172a 100%);
      border-radius: 16px;
      padding: 28px 32px;
      margin-bottom: 24px;
      border: 1px solid #334155;
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LABEL MAPS
# ─────────────────────────────────────────────
LABELS = {
    "Gender": {1: "Male", 2: "Female"},
    "Gender.1": {1: "Male", 2: "Female"},
    "Marital Status": {1: "Married", 2: "Other"},
    "Primary Caregiver": {1: "Spouse", 2: "Child", 3: "Other"},
    "Educational Level": {1: "Primary", 2: "Secondary", 3: "College+"},
    "Stroke Subtype": {1: "Ischemic", 2: "Hemorrhagic", 3: "Other"},
    "Medical Expense Payment": {1: "Self-pay", 2: "Insurance", 3: "Other"},
    "Residence": {1: "Urban", 2: "Rural"},
    "Outcome": {1: "Survived", 2: "Deceased"},
}

# ─────────────────────────────────────────────
# DATA LOADER (FIXED)
# ─────────────────────────────────────────────
@st.cache_data
def load_data(file_obj=None):
    try:
        if file_obj is not None:
            df = pd.read_csv(file_obj)
        else:
            df = pd.read_csv("data.csv")  # default dataset
    except Exception:
        st.error("❌ Failed to load data. Please upload a valid CSV.")
        st.stop()

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove unwanted columns
    df = df.drop(columns=[
        "Patient Characteristics",
        "Primary Caregiver Characteristics"
    ], errors="ignore")

    # Original logic
    df = df.drop(columns=[c for c in df.columns if df[c].isna().all()], errors="ignore")

    cols = list(df.columns)
    edu_idx = [i for i, c in enumerate(cols) if c.strip() == "Educational Level"]
    if len(edu_idx) >= 2:
        cols[edu_idx[1]] = "Educational Level (Caregiver)"
    df.columns = cols

    return df

def apply_labels(df):
    df2 = df.copy()
    for col, mapping in LABELS.items():
        if col in df2.columns:
            df2[col] = df2[col].map(mapping).fillna(df2[col])
    return df2

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 ICU Stroke Study")
    st.markdown("---")

    uploaded = st.file_uploader("📂 Upload your own CSV", type=["csv"])

    if uploaded:
        st.success("Using uploaded dataset ✅")
    else:
        st.info("Using default dataset (data.csv) 📊")

    st.markdown("---")

    st.markdown("### 🧭 Navigation")
    pages = [
        "🏠 Overview & Summary",
        "👤 Patient Demographics",
        "🔎 Data Explorer",
    ]

    # ✅ FIXED HERE
    page = st.radio("Navigation", pages, label_visibility="collapsed")

    st.markdown("---")

    raw = load_data(uploaded)
    df_all = apply_labels(raw)

# ─────────────────────────────────────────────
# PAGE 1
# ─────────────────────────────────────────────
if page == "🏠 Overview & Summary":
    st.markdown('<div class="hero"><h1>🧠 ICU Dashboard</h1></div>', unsafe_allow_html=True)

    st.metric("Total Rows", len(df_all))
    st.metric("Average Age", f"{df_all['Age'].mean():.1f}")

    fig = px.histogram(df_all, x="Age", nbins=20)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 2
# ─────────────────────────────────────────────
elif page == "👤 Patient Demographics":
    st.markdown('<div class="hero"><h1>👤 Patient Demographics</h1></div>', unsafe_allow_html=True)

    fig = px.pie(df_all, names="Gender")
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 3
# ─────────────────────────────────────────────
elif page == "🔎 Data Explorer":
    st.markdown('<div class="hero"><h1>🔎 Data Explorer</h1></div>', unsafe_allow_html=True)

    st.dataframe(df_all, use_container_width=True)
