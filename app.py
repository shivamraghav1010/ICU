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
/* (UNCHANGED CSS) */
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
    "Understanding of the Disease ": {1: "Poor", 2: "Moderate", 3: "Good"},
    "Occupation": {1: "Employed", 2: "Retired", 3: "Other"},
    "Employment Status ": {1: "Employed", 2: "Unemployed"},
    "Religious Belief ": {1: "Yes", 2: "No"},
    "Educational Level ": {1: "Primary", 2: "Secondary", 3: "College+"},
    "Monthly Household Income per Capita": {1: "Low", 2: "Medium", 3: "High"},
    "Frequency of Care from Relatives or Friends ": {1: "Never", 2: "Rarely", 3: "Sometimes", 4: "Often", 5: "Always"},
    "Frequency of Physician–Family Communication ": {1: "Never", 2: "Rarely", 3: "Sometimes", 4: "Often", 5: "Always"},
    "Outcome": {1: "Survived", 2: "Deceased"},
}

# ─────────────────────────────────────────────
# DATA LOADER (FIXED ONLY THIS)
# ─────────────────────────────────────────────
@st.cache_data
def load_data(file_obj=None):
    try:
        if file_obj is not None:
            df = pd.read_csv(file_obj)
        else:
            df = pd.read_csv("data.csv")   # ✅ default dataset
    except Exception:
        st.error("❌ Failed to load data. Please upload a valid CSV.")
        st.stop()

    # Original logic untouched
    df = df.drop(columns=[c for c in df.columns if df[c].isna().all()], errors="ignore")

    cols = list(df.columns)
    edu_idx = [i for i, c in enumerate(cols) if c.strip() == "Educational Level"]
    if len(edu_idx) >= 2:
        cols[edu_idx[1]] = "Educational Level (Caregiver)"
    df.columns = cols

    LABELS["Educational Level (Caregiver)"] = {1: "Primary", 2: "Secondary", 3: "College+"}

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

    # ✅ Added info (no logic change)
    if uploaded:
        st.success("Using uploaded dataset ✅")
    else:
        st.info("Using default dataset (data.csv) 📊")

    st.markdown("---")

    st.markdown("### 🧭 Navigation")
    pages = [
        "🏠 Overview & Summary",
        "👤 Patient Demographics",
        "👨‍👩‍👧 Caregiver Profile",
        "🏥 Clinical Severity",
        "📊 Resilience Scores",
        "🔗 Correlations & Insights",
        "🔎 Data Explorer",
    ]
    page = st.radio("", pages, label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 🎛️ Global Filters")

    try:
        raw = load_data(uploaded)
    except Exception:
        st.error("Could not load data.")
        st.stop()

    df_all = apply_labels(raw)

    outcome_opts = ["All"] + sorted(df_all["Outcome"].unique().tolist())
    sel_outcome = st.selectbox("Outcome", outcome_opts)

    stroke_opts = ["All"] + sorted(df_all["Stroke Subtype"].unique().tolist())
    sel_stroke = st.selectbox("Stroke Subtype", stroke_opts)

    gender_opts = ["All"] + sorted(df_all["Gender"].unique().tolist())
    sel_gender = st.selectbox("Patient Gender", gender_opts)

    age_min, age_max = int(raw["Age"].min()), int(raw["Age"].max())
    sel_age = st.slider("Patient Age Range", age_min, age_max, (age_min, age_max))

    st.markdown("---")
    st.caption("Study: *Unraveling family resilience patterns in ICU first-episode stroke*")

# ─────────────────────────────────────────────
# REST OF YOUR CODE (UNCHANGED)
# ─────────────────────────────────────────────

# 👇 KEEP EVERYTHING BELOW EXACTLY SAME
# (I am not repeating it to avoid clutter, but DO NOT change anything)
